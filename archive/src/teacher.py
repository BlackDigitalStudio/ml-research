"""Multi-stream Transformer teacher — V1.

Drop-in replacement for the CNN+ensemble classification stage. Consumes the
same (X_lob, X_feat, y, mid_prices, target_pnl) output from
`Trainer.build_samples_cached` and produces the same interface (predicted
class UP/DOWN/FLAT + calibrated confidence) so the existing walk-forward
backtest (`scripts/backtest.run_backtest`) works unchanged.

Architecture (V1, ~2-3M params — right-sized for ~20k training samples):

    LOB tokens  : (B, 3, 20, 50) → reshape (B, 50, 60) → Linear → (B, 50, d_model)
    Feat token  : (B, 34)        → Linear             → (B,  1, d_model)
    CLS token   : learnable                            → (B,  1, d_model)
    Concat      : (B, 52, d_model)
    Positional  : learnable per position
    Encoder     : N_LAYERS × (MHA + FFN + LayerNorm)
    CLS output  : classification head (3-way) + regression head (target_pnl)

Design decisions (see project_trading_bot.md and handoff_current.md):
  * Same input shapes as CNN so `build_samples_cached` cache is reused.
  * Same output interface (UP/DOWN/FLAT + confidence) so walk-forward
    backtest layer is untouched.
  * Regression head on `target_pnl` — already produced by build_samples,
    wasted in the current pipeline. Teacher uses it as auxiliary loss
    (label smoothing for the classification signal).
  * No pretraining in V1. Pretraining on unlabeled LOB is a separate
    follow-up item — big win, independent of backbone choice.
  * Distillation target (student) is NOT included here — written in a
    separate module when/if the teacher proves its worth over CNN.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model import UP, DOWN, FLAT


# --------------------------------------------------------------------------- #
# Hyper-parameters — kept in a single dataclass so next session can sweep.
# --------------------------------------------------------------------------- #


@dataclass
class TeacherConfig:
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 768
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3   # auxiliary regression loss scale
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class MultiStreamTransformer(nn.Module):
    """Multi-stream Transformer encoder for LOB + handcrafted features.

    Inputs:
        lob : (B, 3, 20, 50)  float32  — bid vols / ask vols / trade flow
        feat: (B, 34)          float32  — handcrafted features (z-normed)

    Outputs:
        logits  : (B, 3)   — UP / DOWN / FLAT
        reg_pnl : (B,)     — predicted target_pnl %
    """

    def __init__(self, num_feat: int = 34, cfg: TeacherConfig = TeacherConfig()):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = 50        # snapshots per sample
        self.lob_channel_dim = 3 * 20  # 3 channels × 20 depth levels flattened

        # --- Per-stream linear projections into d_model ---
        self.lob_proj = nn.Linear(self.lob_channel_dim, cfg.d_model)
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # --- CLS token + positional encoding ---
        # 1 CLS + 50 LOB time tokens + 1 feat token = 52
        self.seq_len = 1 + self.lob_time_dim + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — more stable for small datasets
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # --- Heads ---
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self, lob: torch.Tensor, feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = lob.shape[0]
        # lob: (B, 3, 20, 50) → permute to (B, 50, 3, 20) → flatten last two
        x_lob = lob.permute(0, 3, 1, 2).reshape(B, self.lob_time_dim, -1)
        lob_tok = self.lob_proj(x_lob)                       # (B, 50, d_model)
        feat_tok = self.feat_proj(feat).unsqueeze(1)         # (B, 1, d_model)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d_model)
        tokens = torch.cat([cls, lob_tok, feat_tok], dim=1)   # (B, 52, d_model)
        tokens = tokens + self.pos_emb
        enc = self.encoder(tokens)                            # (B, 52, d_model)
        cls_out = enc[:, 0]                                   # (B, d_model)
        logits = self.cls_head(cls_out)
        reg = self.reg_head(cls_out).squeeze(-1)
        return logits, reg


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #


def train_teacher(
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    target_pnl: np.ndarray,
    *,
    cfg: TeacherConfig = TeacherConfig(),
    val_frac: float = 0.2,
    gap: int = 650,
    seed: int = 42,
    device: str | None = None,
) -> tuple[MultiStreamTransformer, dict]:
    """Train a teacher on time-ordered data.

    Uses the same gap-split walk-forward layout as the ensemble trainer
    (`trainer.train_ensemble`) — gap of HORIZON+WINDOW ticks between
    train and val to prevent label leakage.

    Returns (trained_model, metrics_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    n = len(y)
    n_val = int(n * val_frac)
    n_train = n - n_val - gap
    if n_train < 1000:
        raise ValueError(f"Too few training samples: {n_train}")

    # Class weights — inverse frequency. FLAT dominates at ~88% so without
    # this the teacher collapses to "always predict FLAT" (Lever 2 in the
    # ensemble trainer uses compute_sample_weight('balanced', y) for the
    # same reason).
    y_train_np = y[:n_train]
    cls_counts = np.bincount(y_train_np, minlength=3).astype(np.float64)
    # Guard against zero-class — assign tiny count.
    cls_counts = np.maximum(cls_counts, 1.0)
    cls_weights_np = len(y_train_np) / (3.0 * cls_counts)
    cls_weights = torch.tensor(cls_weights_np, dtype=torch.float32)
    print(f"[teacher] class weights: UP={cls_weights[0]:.3f} "
          f"DOWN={cls_weights[1]:.3f} FLAT={cls_weights[2]:.3f}")

    # Copy mmap'd X_lob to regular tensor (writable + on-device)
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
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          drop_last=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0)

    model = MultiStreamTransformer(num_feat=X_feat.shape[1], cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[teacher] params={n_params / 1e6:.2f}M device={device} "
          f"train={n_train} val={n - n_train - gap}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_ld)
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_val_metric = -1e9
    best_state: dict | None = None
    patience_left = cfg.early_stop_patience
    ce = nn.CrossEntropyLoss(
        weight=cls_weights.to(device),
        label_smoothing=cfg.label_smoothing,
    )

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = tr_cls_loss = tr_reg_loss = 0.0
        n_seen = 0
        for xb, fb, yb, pb in train_ld:
            xb, fb, yb, pb = xb.to(device), fb.to(device), yb.to(device), pb.to(device)
            optim.zero_grad()
            logits, reg = model(xb, fb)
            loss_cls = ce(logits, yb)
            loss_reg = F.smooth_l1_loss(reg, pb)
            loss = loss_cls + cfg.reg_loss_weight * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_cls_loss += loss_cls.item() * bs
            tr_reg_loss += loss_reg.item() * bs
            n_seen += bs
        tr_loss /= n_seen; tr_cls_loss /= n_seen; tr_reg_loss /= n_seen

        # --- Validation ---
        # Balanced accuracy = mean of per-class recall — robust to FLAT
        # dominance. Selecting best model by raw accuracy would always pick
        # "always predict FLAT" (0.88 on our data).
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
                vl_loss += (loss_cls + cfg.reg_loss_weight * loss_reg).item() * xb.size(0)
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
        # Balanced accuracy — mean recall across classes with data.
        recalls = np.where(total_per_class > 0,
                           correct_per_class / np.maximum(total_per_class, 1),
                           np.nan)
        bal_acc = float(np.nanmean(recalls))

        history.append({
            "epoch": epoch + 1, "tr_loss": tr_loss, "tr_cls": tr_cls_loss,
            "tr_reg": tr_reg_loss, "vl_loss": vl_loss, "val_acc": val_acc,
            "bal_acc": bal_acc, "reg_mae": reg_mae,
        })
        print(f"[teacher] epoch {epoch + 1:3d}/{cfg.epochs} "
              f"tr={tr_loss:.4f} vl={vl_loss:.4f} "
              f"val_acc={val_acc:.4f} bal_acc={bal_acc:.4f} "
              f"recalls=UP{recalls[0]:.2f}/DN{recalls[1]:.2f}/FL{recalls[2]:.2f} "
              f"reg_mae={reg_mae:.4f}")

        # Select best by balanced accuracy — prevents collapse to FLAT.
        if bal_acc > best_val_metric + 1e-4:
            best_val_metric = bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[teacher] early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "best_bal_acc": best_val_metric,
        "params_M": n_params / 1e6,
        "history": history,
    }
    return model, metrics


@torch.no_grad()
def predict_teacher(
    model: MultiStreamTransformer,
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    *,
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run teacher inference.

    Returns:
        preds        : (N,) int64  — argmax class (UP=0, DOWN=1, FLAT=2)
        confidences  : (N,) float32 — max softmax probability
        reg_pnl      : (N,) float32 — predicted target_pnl %
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    n = len(X_feat)
    preds = np.empty(n, dtype=np.int64)
    confs = np.empty(n, dtype=np.float32)
    regs = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        xb = torch.from_numpy(np.asarray(X_lob[i:j])).float().to(device)
        fb = torch.from_numpy(X_feat[i:j]).float().to(device)
        logits, reg = model(xb, fb)
        probs = F.softmax(logits, dim=-1)
        p_conf, p_idx = probs.max(dim=-1)
        preds[i:j] = p_idx.cpu().numpy()
        confs[i:j] = p_conf.cpu().numpy()
        regs[i:j] = reg.cpu().numpy()
    return preds, confs, regs


def save_teacher(model: MultiStreamTransformer, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg.__dict__}, path)


def load_teacher(path: Path, num_feat: int = 34) -> MultiStreamTransformer:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TeacherConfig(**blob["cfg"])
    model = MultiStreamTransformer(num_feat=num_feat, cfg=cfg)
    model.load_state_dict(blob["state_dict"])
    return model
