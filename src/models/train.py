"""Generic training loop for architecture bake-offs.

Any model that exposes `forward(lob, feat) -> (logits, reg)` works here.
Mirrors `src.teacher.train_teacher` — same gap-split, class-weighted CE +
smooth-L1 reg, balanced-accuracy early stop, cosine LR with warmup — so
results are directly comparable.

Why not just reuse `train_teacher`? `train_teacher` hard-codes
`MultiStreamTransformer` for its instantiation. Splitting the loop lets the
bake-off layer stay read-only over the existing Transformer path.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ModelFactory = Callable[[int], nn.Module]  # num_feat -> model


def train_generic(
    model_factory: ModelFactory,
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    target_pnl: np.ndarray,
    *,
    # Training knobs — kept flat so callers can sweep independently of cfg dc.
    batch_size: int = 256,
    epochs: int = 40,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_frac: float = 0.1,
    reg_loss_weight: float = 0.3,
    label_smoothing: float = 0.05,
    early_stop_patience: int = 6,
    val_frac: float = 0.2,
    gap: int = 650,
    seed: int = 42,
    device: str | None = None,
    tag: str = "model",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train any (lob, feat) -> (logits, reg) model with the teacher protocol."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    n = len(y)
    n_val = int(n * val_frac)
    n_train = n - n_val - gap
    if n_train < 1000:
        raise ValueError(f"Too few training samples: {n_train}")

    y_train_np = y[:n_train]
    cls_counts = np.bincount(y_train_np, minlength=3).astype(np.float64)
    cls_counts = np.maximum(cls_counts, 1.0)
    # Softer class weights: sqrt of inverse frequency instead of raw inverse.
    # Raw inverse made the model over-predict UP/DN on FL samples, collapsing
    # precision to ~15% (bal_acc looked good by recall but trading was lossy).
    # sqrt keeps some upweighting for minorities without destroying argmax.
    inv_freq = len(y_train_np) / (3.0 * cls_counts)
    cls_weights_np = np.sqrt(inv_freq)
    cls_weights = torch.tensor(cls_weights_np, dtype=torch.float32)
    print(f"[{tag}] class weights (sqrt-inv-freq): "
          f"UP={cls_weights[0]:.3f} DOWN={cls_weights[1]:.3f} FLAT={cls_weights[2]:.3f}")

    X_lob_train = torch.from_numpy(np.asarray(X_lob[:n_train]).copy()).float()
    X_feat_train = torch.from_numpy(X_feat[:n_train]).float()
    y_train = torch.from_numpy(y[:n_train]).long()
    pnl_train = torch.from_numpy(target_pnl[:n_train]).float()
    X_lob_val = torch.from_numpy(np.asarray(X_lob[n_train + gap:]).copy()).float()
    X_feat_val = torch.from_numpy(X_feat[n_train + gap:]).float()
    y_val = torch.from_numpy(y[n_train + gap:]).long()
    pnl_val = torch.from_numpy(target_pnl[n_train + gap:]).float()

    train_ds = TensorDataset(X_lob_train, X_feat_train, y_train, pnl_train)
    val_ds = TensorDataset(X_lob_val, X_feat_val, y_val, pnl_val)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model_factory(X_feat.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params / 1e6:.2f}M device={device} "
          f"train={n_train} val={n - n_train - gap}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_ld)
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # Early stop on F1-macro instead of bal_acc — balanced accuracy is averaged
    # recall which is recall-only (ignores precision). For trading we need
    # high precision on non-FLAT predictions; F1-macro balances both.
    best_score = -1e9
    best_state: dict | None = None
    best_metrics: dict = {}
    patience = early_stop_patience
    ce = nn.CrossEntropyLoss(weight=cls_weights.to(device), label_smoothing=label_smoothing)

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        n_seen = 0
        for xb, fb, yb, pb in train_ld:
            xb, fb, yb, pb = xb.to(device), fb.to(device), yb.to(device), pb.to(device)
            optim.zero_grad()
            logits, reg = model(xb, fb)
            loss_cls = ce(logits, yb)
            loss_reg = F.smooth_l1_loss(reg, pb)
            loss = loss_cls + reg_loss_weight * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            tr_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        tr_loss /= n_seen

        model.eval()
        vl_loss = 0.0
        # Confusion matrix: cm[true, pred]
        cm = np.zeros((3, 3), dtype=np.int64)
        total = 0
        reg_mae = 0.0
        val_pnl_sum = 0.0
        with torch.no_grad():
            for xb, fb, yb, pb in val_ld:
                xb, fb, yb, pb = xb.to(device), fb.to(device), yb.to(device), pb.to(device)
                logits, reg = model(xb, fb)
                loss_cls = ce(logits, yb)
                loss_reg = F.smooth_l1_loss(reg, pb)
                vl_loss += (loss_cls + reg_loss_weight * loss_reg).item() * xb.size(0)
                pred = logits.argmax(dim=-1)
                total += xb.size(0)
                reg_mae += (reg - pb).abs().sum().item()
                yb_np = yb.cpu().numpy()
                pred_np = pred.cpu().numpy()
                for t in range(3):
                    for p_ in range(3):
                        cm[t, p_] += int(((yb_np == t) & (pred_np == p_)).sum())
                # Trading pnl proxy: sum of target_pnl where model predicted
                # non-FLAT (i.e., would take a trade). Sign handling: if model
                # predicts correct direction, pnl_val is positive; wrong
                # direction on true UP/DN → negative; predicting non-FLAT on
                # true FL → pnl_val is negative (FL samples have pnl_val<0).
                pb_np = pb.cpu().numpy()
                signal = pred_np != 2  # FLAT = class 2
                val_pnl_sum += float(pb_np[signal].sum())

        vl_loss /= total
        val_acc = float(np.trace(cm) / total) if total > 0 else 0.0
        reg_mae /= total
        # Per-class recalls + precisions + F1
        totals_per_class = cm.sum(axis=1)  # true counts
        preds_per_class = cm.sum(axis=0)   # predicted counts
        diag = np.diag(cm)
        recalls = np.where(totals_per_class > 0,
                           diag / np.maximum(totals_per_class, 1), np.nan)
        precisions = np.where(preds_per_class > 0,
                              diag / np.maximum(preds_per_class, 1), np.nan)
        f1 = np.where((precisions + recalls) > 0,
                      2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12),
                      0.0)
        bal_acc = float(np.nanmean(recalls))
        f1_macro = float(np.nanmean(f1))
        # Precision on non-FLAT predictions combined (key for trading)
        nonflat_pred = preds_per_class[0] + preds_per_class[1]
        nonflat_correct = diag[0] + diag[1]
        precision_nonflat = float(nonflat_correct / max(nonflat_pred, 1))

        history.append({
            "epoch": epoch + 1, "tr_loss": tr_loss, "vl_loss": vl_loss,
            "val_acc": val_acc, "bal_acc": bal_acc,
            "f1_macro": f1_macro,
            "precision_up": float(precisions[0]), "precision_dn": float(precisions[1]),
            "precision_nonflat": precision_nonflat,
            "val_pnl_sum": val_pnl_sum, "reg_mae": reg_mae,
        })
        print(f"[{tag}] epoch {epoch + 1:3d}/{epochs} "
              f"tr={tr_loss:.4f} vl={vl_loss:.4f} "
              f"acc={val_acc:.3f} bal={bal_acc:.3f} F1={f1_macro:.3f} "
              f"prec[UP={precisions[0]:.2f}/DN={precisions[1]:.2f}/nonFL={precision_nonflat:.2f}] "
              f"rec[UP={recalls[0]:.2f}/DN={recalls[1]:.2f}/FL={recalls[2]:.2f}] "
              f"pnl={val_pnl_sum:+.1f}")

        # Early stop: use F1-macro (balances precision + recall across classes).
        # This selects checkpoints where UP/DN are accurately predicted AND
        # NOT over-predicted into FL territory. bal_acc alone was fooled by
        # recall-heavy models with catastrophic precision.
        score = f1_macro
        if score > best_score + 1e-4:
            best_score = score
            best_metrics = history[-1].copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[{tag}] early stop at epoch {epoch + 1}  (best F1={best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val softmax — needed by the stacker downstream.
    model.eval()
    val_softs: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, fb, _, _ in val_ld:
            xb, fb = xb.to(device), fb.to(device)
            logits, _ = model(xb, fb)
            val_softs.append(torch.softmax(logits, dim=-1).cpu())
    val_soft_np = torch.cat(val_softs, dim=0).numpy()

    return model, {
        "best_f1_macro": best_score,
        "best_bal_acc": best_metrics.get("bal_acc", 0.0),
        "best_metrics": best_metrics,
        "params_M": n_params / 1e6,
        "history": history,
        "tag": tag,
        "val_softmax": val_soft_np,        # (n_val, 3) for stacking
        "y_val": y[n_train + gap:].copy(), # alignment ref for stacker
        "X_feat_val": X_feat[n_train + gap:].copy(),
    }
