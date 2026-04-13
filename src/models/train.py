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
    cls_weights_np = len(y_train_np) / (3.0 * cls_counts)
    cls_weights = torch.tensor(cls_weights_np, dtype=torch.float32)
    print(f"[{tag}] class weights: UP={cls_weights[0]:.3f} "
          f"DOWN={cls_weights[1]:.3f} FLAT={cls_weights[2]:.3f}")

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

    best_bal = -1e9
    best_state: dict | None = None
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
        correct_per_class = np.zeros(3, dtype=np.int64)
        total_per_class = np.zeros(3, dtype=np.int64)
        correct = 0
        total = 0
        reg_mae = 0.0
        with torch.no_grad():
            for xb, fb, yb, pb in val_ld:
                xb, fb, yb, pb = xb.to(device), fb.to(device), yb.to(device), pb.to(device)
                logits, reg = model(xb, fb)
                loss_cls = ce(logits, yb)
                loss_reg = F.smooth_l1_loss(reg, pb)
                vl_loss += (loss_cls + reg_loss_weight * loss_reg).item() * xb.size(0)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += xb.size(0)
                reg_mae += (reg - pb).abs().sum().item()
                yb_np = yb.cpu().numpy()
                pred_np = pred.cpu().numpy()
                for cls in range(3):
                    mask = (yb_np == cls)
                    total_per_class[cls] += mask.sum()
                    correct_per_class[cls] += ((pred_np == cls) & mask).sum()
        vl_loss /= total
        val_acc = correct / total
        reg_mae /= total
        recalls = np.where(total_per_class > 0,
                           correct_per_class / np.maximum(total_per_class, 1), np.nan)
        bal_acc = float(np.nanmean(recalls))

        history.append({
            "epoch": epoch + 1, "tr_loss": tr_loss, "vl_loss": vl_loss,
            "val_acc": val_acc, "bal_acc": bal_acc, "reg_mae": reg_mae,
        })
        print(f"[{tag}] epoch {epoch + 1:3d}/{epochs} "
              f"tr={tr_loss:.4f} vl={vl_loss:.4f} "
              f"val_acc={val_acc:.4f} bal_acc={bal_acc:.4f} "
              f"recalls=UP{recalls[0]:.2f}/DN{recalls[1]:.2f}/FL{recalls[2]:.2f} "
              f"reg_mae={reg_mae:.4f}")

        if bal_acc > best_bal + 1e-4:
            best_bal = bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[{tag}] early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_bal_acc": best_bal,
        "params_M": n_params / 1e6,
        "history": history,
        "tag": tag,
    }
