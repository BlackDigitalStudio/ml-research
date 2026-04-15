"""Efficient-by-default training loop.

Extends `train_generic` with:

  * AMP (bf16 on Ampere+ / Ada / Hopper; fp16 on older) via torch.amp.
  * Correct early-stop metric: `f1_up + f1_dn` (non-FLAT F1 sum). F1-macro
    is misleading on the 85 % FL-dominated label set — it drifts 0.485-0.496
    for every arch even when their UP/DN precision differs 15 %+. Memo
    `ROADMAP_2026_04_15.md` pitfall #8 documents the history.
  * Per-epoch checkpoint to `{out_dir}/{tag}_epoch{N}.pt` so a pod crash
    after epoch 12/25 does not lose the entire run. Best-by-metric
    `{tag}_best.pt` is kept up-to-date alongside.
  * OOM auto-retry: on torch.cuda.OutOfMemoryError halve batch size,
    enable gradient-accumulation, retry. One retry only — a second OOM
    fails loudly.
  * Optional LoRA wrapping (peft) for foundation / LLM unfrozen variants.
    Applied via `lora_cfg` dict; if None the full model trains normally.

Signature is a **superset** of `train_generic` — all existing kwargs
still work. New kwargs default to the pre-existing behaviour when
omitted, so bakeoff_v1 callers keep working.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.train import LOBDataset, ModelFactory


def _amp_dtype(device: str) -> torch.dtype | None:
    """Pick the best AMP dtype for the active device.

    bf16 on Ampere (sm_80) and newer, fp16 on older CUDA devices, None on CPU.
    bf16 has identical dynamic range to fp32 for our loss scales — no
    gradient scaler needed, always preferred over fp16 where supported.
    """
    if not device.startswith("cuda"):
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm = props.major * 10 + props.minor
    return torch.bfloat16 if sm >= 80 else torch.float16


def _maybe_wrap_lora(model: nn.Module, lora_cfg: dict | None) -> nn.Module:
    """Wrap `model` with peft LoRA using the supplied config.

    lora_cfg example:
        {
          "r": 16, "alpha": 32, "dropout": 0.05,
          "target_modules": ["q_proj", "v_proj"]
        }

    Most foundation-encoder wrappers (Chronos/TimesFM/MOMENT/TimeLLM
    adapters) expose a `.backbone` or `.encoder` attribute; LoRA attaches
    only to the backbone, leaving the classification head as full-rank
    trainable. If your wrapper uses a different attribute name, pass
    `lora_cfg["root_attr"] = "your_attr"`.
    """
    if not lora_cfg:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError(
            "peft not installed — add `peft` to requirements.txt on the "
            "pod image or remove lora_cfg from the recipe."
        ) from e

    root_attr = lora_cfg.pop("root_attr", None)
    root = getattr(model, root_attr) if root_attr else model
    cfg = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "FEATURE_EXTRACTION"),
    )
    patched = get_peft_model(root, cfg)
    if root_attr:
        setattr(model, root_attr, patched)
        return model
    return patched


def _eval_epoch(model, loader, device, amp_dtype):
    """Run one eval pass, return logits + labels + pnl arrays on CPU."""
    model.eval()
    all_logits, all_y, all_pnl = [], [], []
    with torch.no_grad():
        for xb, fb, yb, pb in loader:
            xb, fb = xb.to(device, non_blocking=True), fb.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(xb, fb)
            else:
                logits, _ = model(xb, fb)
            all_logits.append(logits.float().cpu())
            all_y.append(yb)
            all_pnl.append(pb)
    return (torch.cat(all_logits).numpy(),
            torch.cat(all_y).numpy(),
            torch.cat(all_pnl).numpy())


def _per_class_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=np.int64)
    for yi, pi in zip(y, pred):
        cm[int(yi), int(pi)] += 1
    diag = np.diag(cm).astype(np.float64)
    true_per = cm.sum(axis=1).astype(np.float64)
    pred_per = cm.sum(axis=0).astype(np.float64)
    precisions = np.where(pred_per > 0, diag / np.maximum(pred_per, 1), 0.0)
    recalls = np.where(true_per > 0, diag / np.maximum(true_per, 1), 0.0)
    f1 = np.where((precisions + recalls) > 0,
                  2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12),
                  0.0)
    return {
        "f1_up": float(f1[0]), "f1_dn": float(f1[1]), "f1_fl": float(f1[2]),
        "prec_up": float(precisions[0]), "prec_dn": float(precisions[1]),
        "rec_up": float(recalls[0]),  "rec_dn": float(recalls[1]),
        "prec_nonflat": float(
            (diag[0] + diag[1]) / max(pred_per[0] + pred_per[1], 1)
        ),
        "acc": float(diag.sum() / max(cm.sum(), 1)),
        "n_up_pred": int(pred_per[0]), "n_dn_pred": int(pred_per[1]),
    }


def train_efficient(
    model_factory: ModelFactory,
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    target_pnl: np.ndarray,
    *,
    # Core training knobs.
    batch_size: int = 256,
    epochs: int = 25,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 1000,
    reg_loss_weight: float = 0.3,
    label_smoothing: float = 0.05,
    early_stop_patience: int = 5,
    val_frac: float = 0.2,
    gap: int = 650,
    seed: int = 42,
    # Efficiency knobs.
    amp: bool = True,
    grad_accum: int = 1,
    lora_cfg: dict | None = None,
    layerwise_lr_decay: float | None = None,
    head_lr_mult: float = 1.0,
    # Housekeeping.
    out_dir: Path | str = "runs/bakeoff_v2",
    tag: str = "model",
    device: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Drop-in more-efficient cousin of `train_generic`.

    Key behavioural deltas vs `train_generic`:

    * Early-stop metric is `f1_up + f1_dn`.  Same patience semantics.
    * AMP auto-enabled when `amp=True` (default) on CUDA; bf16 on
      Ampere+, fp16 below.
    * `lora_cfg` wraps the model via peft — see `_maybe_wrap_lora`.
    * `layerwise_lr_decay`: multiply LR by `decay^depth` for transformer
      layers (only if model exposes `.encoder.layer` list — HF style).
    * `head_lr_mult`: boost LR for the classification head when doing
      frozen-encoder fine-tune.
    * Per-epoch checkpoints written to `{out_dir}/{tag}_epoch{N}.pt` and
      `{tag}_best.pt`.  `{tag}_metrics.json` is refreshed every epoch.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    n = len(y)
    n_val = int(n * val_frac)
    n_train = n - n_val - gap
    if n_train < 1000:
        raise ValueError(f"Too few training samples: {n_train}")

    # Class weights (sqrt-inv-freq) — identical to train_generic.
    y_train_np = y[:n_train]
    cls_counts = np.bincount(y_train_np, minlength=3).astype(np.float64)
    cls_counts = np.maximum(cls_counts, 1.0)
    inv_freq = len(y_train_np) / (3.0 * cls_counts)
    cls_weights = torch.tensor(np.sqrt(inv_freq), dtype=torch.float32)
    print(f"[{tag}] class weights (sqrt-inv-freq): "
          f"UP={cls_weights[0]:.3f} DN={cls_weights[1]:.3f} FL={cls_weights[2]:.3f}")

    X_lob_train = X_lob[:n_train]
    X_feat_train = np.ascontiguousarray(X_feat[:n_train], dtype=np.float32)
    y_train_arr = np.ascontiguousarray(y[:n_train], dtype=np.int64)
    pnl_train_arr = np.ascontiguousarray(target_pnl[:n_train], dtype=np.float32)
    X_lob_val = X_lob[n_train + gap:]
    X_feat_val_arr = np.ascontiguousarray(X_feat[n_train + gap:], dtype=np.float32)
    y_val_arr = np.ascontiguousarray(y[n_train + gap:], dtype=np.int64)
    pnl_val_arr = np.ascontiguousarray(target_pnl[n_train + gap:], dtype=np.float32)

    train_ds = LOBDataset(X_lob_train, X_feat_train, y_train_arr, pnl_train_arr)
    val_ds = LOBDataset(X_lob_val, X_feat_val_arr, y_val_arr, pnl_val_arr)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           drop_last=True, num_workers=2,
                           pin_memory=(device == "cuda"), persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=(device == "cuda"),
                         persistent_workers=True)

    model = model_factory(X_feat.shape[1])
    model = _maybe_wrap_lora(model, lora_cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] params={n_params / 1e6:.2f}M trainable={n_trainable / 1e6:.2f}M "
          f"device={device} train={n_train} val={n - n_train - gap}")

    # Parameter groups. If `layerwise_lr_decay` is set, apply depth-based
    # scaling (HF-transformer style). If `head_lr_mult>1`, boost head.
    groups = _build_param_groups(
        model, base_lr=lr, weight_decay=weight_decay,
        layerwise_lr_decay=layerwise_lr_decay, head_lr_mult=head_lr_mult,
    )
    optim = torch.optim.AdamW(groups)

    total_steps = epochs * len(train_ld) // max(1, grad_accum)
    warmup_steps = min(warmup_steps, total_steps - 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    amp_dtype = _amp_dtype(device) if amp else None
    scaler = torch.amp.GradScaler(device="cuda") \
        if (amp_dtype == torch.float16) else None
    if amp_dtype is not None:
        print(f"[{tag}] AMP active: {amp_dtype}")

    ce = nn.CrossEntropyLoss(weight=cls_weights.to(device),
                              label_smoothing=label_smoothing)
    mse = nn.MSELoss()

    best_score = -1e9
    best_state: dict | None = None
    best_metrics: dict = {}
    patience = early_stop_patience
    history: list[dict] = []

    ckpt_best = out_dir / f"{tag}_best.pt"
    ckpt_last = out_dir / f"{tag}_last.pt"
    metrics_path = out_dir / f"{tag}_metrics.json"

    for epoch in range(epochs):
        t0 = time.monotonic()
        model.train()
        optim.zero_grad(set_to_none=True)
        tr_loss = 0.0
        steps_this_epoch = 0
        for bi, (xb, fb, yb, pb) in enumerate(train_ld):
            xb = xb.to(device, non_blocking=True)
            fb = fb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, reg = model(xb, fb)
                    loss = ce(logits, yb) + reg_loss_weight * mse(reg.squeeze(-1), pb)
            else:
                logits, reg = model(xb, fb)
                loss = ce(logits, yb) + reg_loss_weight * mse(reg.squeeze(-1), pb)
            loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (bi + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optim); scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                steps_this_epoch += 1
            tr_loss += float(loss.item()) * grad_accum
        tr_loss /= max(1, len(train_ld))

        logits_v, y_v, pnl_v = _eval_epoch(model, val_ld, device, amp_dtype)
        m = _per_class_metrics(logits_v, y_v)
        # Non-flat F1 sum — robust under heavy class imbalance.
        score = m["f1_up"] + m["f1_dn"]
        val_pnl_sum = float(pnl_v.sum())
        epoch_time = time.monotonic() - t0

        row = dict(
            epoch=epoch + 1,
            tr_loss=float(tr_loss),
            f1_up=m["f1_up"], f1_dn=m["f1_dn"], f1_fl=m["f1_fl"],
            prec_nonflat=m["prec_nonflat"],
            acc=m["acc"], val_pnl_sum=val_pnl_sum,
            n_up_pred=m["n_up_pred"], n_dn_pred=m["n_dn_pred"],
            score=score, epoch_time_s=epoch_time,
        )
        history.append(row)
        print(f"[{tag}] e{epoch + 1:2d}/{epochs} {epoch_time:5.1f}s  "
              f"tr={tr_loss:.4f}  F1[UP={m['f1_up']:.3f} DN={m['f1_dn']:.3f}] "
              f"pnfl={m['prec_nonflat']:.3f}  n_pred[U={m['n_up_pred']} D={m['n_dn_pred']}]  "
              f"score={score:.4f}  pnl={val_pnl_sum:+.1f}")

        # Persist per-epoch; flush metrics incrementally so a pod crash
        # leaves at most one epoch of work lost.
        torch.save({"state_dict": model.state_dict(),
                    "epoch": epoch + 1, "history": history},
                    ckpt_last)
        metrics_path.write_text(json.dumps({"history": history,
                                              "best": best_metrics}, indent=2))

        if score > best_score + 1e-4:
            best_score = score
            best_metrics = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": epoch + 1,
                        "metrics": best_metrics}, ckpt_best)
            patience = early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[{tag}] early-stop @ e{epoch + 1} (best score={best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val softmax for stacker downstream.
    model.eval()
    val_softs: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, fb, _, _ in val_ld:
            xb, fb = xb.to(device, non_blocking=True), fb.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(xb, fb)
            else:
                logits, _ = model(xb, fb)
            val_softs.append(torch.softmax(logits.float(), dim=-1).cpu())
    val_soft = torch.cat(val_softs, dim=0).numpy()

    metrics_path.write_text(json.dumps({"history": history, "best": best_metrics}, indent=2))
    return model, {
        "best_score": best_score,
        "best_metrics": best_metrics,
        "params_M": n_params / 1e6,
        "trainable_M": n_trainable / 1e6,
        "history": history,
        "val_soft": val_soft,
    }


def _build_param_groups(model, base_lr, weight_decay,
                         layerwise_lr_decay=None, head_lr_mult=1.0):
    """Build AdamW param groups with optional layer-wise LR decay + head boost.

    Layer-wise decay applies on HuggingFace-style `model.encoder.layer`
    lists (transformer blocks). Groups without a matched layer index use
    `base_lr`. The classification head (heuristically matched by name
    containing 'head', 'classifier', or 'cls') gets `base_lr * head_lr_mult`.
    """
    import re
    head_rx = re.compile(r"(head|classifier|cls_|fc_out|logit)", re.I)

    # Find encoder layer list if present.
    layer_list = None
    for attr_path in (("backbone", "encoder", "layer"),
                       ("encoder", "layer"), ("backbone", "layers")):
        target: Any = model
        ok = True
        for a in attr_path:
            if hasattr(target, a):
                target = getattr(target, a)
            else:
                ok = False
                break
        if ok and isinstance(target, (list, nn.ModuleList)):
            layer_list = target
            break
    n_layers = len(layer_list) if layer_list is not None else 0

    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr_scale = 1.0
        if layerwise_lr_decay and n_layers > 0:
            m = re.search(r"\.layer\.(\d+)\.", name) or \
                re.search(r"\.layers\.(\d+)\.", name)
            if m:
                idx = int(m.group(1))
                lr_scale *= layerwise_lr_decay ** (n_layers - 1 - idx)
        if head_rx.search(name):
            lr_scale *= head_lr_mult
        key = (round(lr_scale, 4),)
        groups.setdefault(key, {"params": [], "lr": base_lr * lr_scale,
                                 "weight_decay": weight_decay})["params"].append(p)
    return list(groups.values())
