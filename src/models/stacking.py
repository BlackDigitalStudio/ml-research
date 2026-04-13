"""Level-2 stacking meta-learner for architecture ensemble.

Given N primary models' predictions on a hold-out set, trains an XGBoost
meta-learner that learns to weight their votes (optionally using the raw
handcrafted features as additional input).

Typical setup:
    1. Train N primaries (Transformer, PatchTST, Mamba, TCN, Chronos-FT, ...)
       on train split.
    2. Predict each primary on val split → N × (Nval, 3) softmaxes.
    3. `train_stacker(primary_softmaxes, y_val, X_feat_val)` → meta.
    4. At inference: concat primaries' softmax → meta → final primary prediction.
    5. Feed meta prediction into meta-labeling (src.models.meta_label) for
       the final "should we trade" decision.

Walk-forward correctness:
    The simple hold-out stacking here uses a split of the val window. For
    a stronger out-of-fold (OOF) setup (K-fold), see `train_stacker_oof`
    which retrains each primary K times on K-1 folds — K× cost but correct
    stacking on the FULL train set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StackerConfig:
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.01
    reg_lambda: float = 1.0
    use_feats: bool = True          # concat handcrafted feats with softmaxes
    objective: str = "multi:softprob"
    num_class: int = 3


def stack_inputs(
    primary_softmaxes: list[np.ndarray],  # list of N arrays shape (M, 3)
    X_feat: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate N softmaxes + optional market feats into (M, N*3 + F)."""
    # Per-model summary: softmax + max + entropy + margin → 6 feats per model.
    parts = []
    for soft in primary_softmaxes:
        soft = soft.astype(np.float32)
        p_max = soft.max(axis=-1, keepdims=True)
        p_margin = (np.sort(soft, axis=-1)[:, -1] - np.sort(soft, axis=-1)[:, -2])[:, None]
        p_entropy = -(soft * np.log(soft.clip(1e-9))).sum(axis=-1, keepdims=True)
        parts.append(np.concatenate([soft, p_max, p_margin, p_entropy], axis=-1))
    stacked = np.concatenate(parts, axis=-1)
    if X_feat is not None:
        stacked = np.concatenate([stacked, X_feat.astype(np.float32)], axis=-1)
    return stacked


def train_stacker(
    primary_softmaxes: list[np.ndarray],
    y_true: np.ndarray,
    X_feat: np.ndarray | None = None,
    *,
    cfg: StackerConfig = StackerConfig(),
    val_frac: float = 0.25,
    seed: int = 42,
):
    """Train XGBoost meta-learner with time-ordered val split."""
    import xgboost as xgb

    X_stack = stack_inputs(primary_softmaxes, X_feat=X_feat if cfg.use_feats else None)
    n = len(y_true)
    n_val = int(n * val_frac)
    n_tr = n - n_val

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        objective=cfg.objective,
        num_class=cfg.num_class,
        early_stopping_rounds=40,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_stack[:n_tr], y_true[:n_tr],
        eval_set=[(X_stack[n_tr:], y_true[n_tr:])],
        verbose=False,
    )

    # Metrics on held-out tail.
    val_soft = model.predict_proba(X_stack[n_tr:])
    val_pred = val_soft.argmax(axis=-1)
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                   f1_score, log_loss)
    y_val = y_true[n_tr:]
    metrics = {
        "n_train": int(n_tr),
        "n_val": int(n_val),
        "val_acc": float(accuracy_score(y_val, val_pred)),
        "val_bal_acc": float(balanced_accuracy_score(y_val, val_pred)),
        "val_f1_macro": float(f1_score(y_val, val_pred, average="macro")),
        "val_logloss": float(log_loss(y_val, val_soft, labels=[0, 1, 2])),
        "per_class_recall": _per_class_recall(y_val, val_pred),
    }
    return model, metrics


def predict_stacked(
    model,
    primary_softmaxes: list[np.ndarray],
    X_feat: np.ndarray | None = None,
    *,
    use_feats: bool = True,
) -> np.ndarray:
    """Return stacker softmax over the 3 primary classes."""
    X_stack = stack_inputs(primary_softmaxes, X_feat=X_feat if use_feats else None)
    return model.predict_proba(X_stack)


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    out = {}
    for c in range(3):
        mask = y_true == c
        if mask.sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float((y_pred[mask] == c).mean())
    return out
