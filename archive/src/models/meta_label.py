"""Meta-labeling (Lopez de Prado, "Advances in Financial Machine Learning").

Two-stage setup that typically adds +10-20% WR on filtered trades:
  1. **Primary model** predicts direction (UP/DOWN/FLAT). We use whatever
     primary is trained — Transformer, PatchTST, Chronos-finetuned, etc.
  2. **Meta model** learns "conditional on primary's prediction, is this
     a trade worth taking?" Trained as binary classifier where:
         meta_label = 1 iff (primary_argmax != FLAT) AND (trade was profitable)
         meta_label = 0 otherwise (including wrong direction AND FLAT signals)

At inference:
     take_trade  = primary.argmax != FLAT AND meta_prob > meta_threshold
     direction   = primary.argmax
     confidence  = primary.max_softmax * meta_prob

This is orthogonal to architecture choice — meta-labeling stacks on top of
ANY primary model. Our default meta-learner is XGBoost (fast, handles
tabular meta-features natively, calibrates well).

Usage:
    # 1. Train primary (existing teacher pipeline).
    primary = train_teacher(...)

    # 2. Get primary predictions on train+val splits.
    p_train = primary.predict(X_train)  # softmax (N, 3)
    p_val   = primary.predict(X_val)

    # 3. Build meta features + labels.
    m_X_train, m_y_train, m_w_train = build_meta_dataset(
        primary_softmax=p_train, y_true=y_train, target_pnl=pnl_train,
        X_feat=X_feat_train,
    )

    # 4. Train meta.
    meta = train_meta(m_X_train, m_y_train, m_w_train)

    # 5. Inference: combine.
    take, direction, conf = combine(primary_softmax, meta, X_feat, ...)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Class indices — must match src.model
UP, DOWN, FLAT = 0, 1, 2


@dataclass
class MetaConfig:
    # XGBoost hyperparameters
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.01
    reg_lambda: float = 1.0

    # What counts as "profitable enough to meta-label as 1"
    pnl_threshold_pct: float = 0.0   # require net PnL strictly > 0 after commissions

    # Inference thresholds (tuned on validation)
    meta_threshold: float = 0.5
    min_primary_conf: float = 0.45   # floor on primary confidence


def build_meta_dataset(
    primary_softmax: np.ndarray,  # (N, 3)
    y_true: np.ndarray,           # (N,) primary labels UP/DOWN/FLAT
    target_pnl: np.ndarray,       # (N,) net PnL (%) the sample would have earned
    X_feat: np.ndarray | None = None,  # (N, F) market features to concat
    cfg: MetaConfig = MetaConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (X_meta, y_meta, sample_weight) for meta training.

    Only rows where primary predicts non-FLAT get meta-labeled. FLAT rows
    are dropped — meta is "should we act on this non-FLAT signal?". If the
    primary says FLAT, we trivially don't trade, no meta decision needed.

    sample_weight is the magnitude of target_pnl, so the meta learner pays
    more attention to large-PnL decisions (both wins and losses).
    """
    primary_pred = primary_softmax.argmax(axis=-1)
    # keep only non-FLAT primary decisions
    mask = primary_pred != FLAT
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise ValueError("Primary predicted FLAT for 100% of samples — nothing to meta-label.")

    p_soft = primary_softmax[mask]          # (M, 3)
    p_pred = primary_pred[mask]             # (M,)
    y = y_true[mask]                        # (M,)
    pnl = target_pnl[mask]                  # (M,)

    # Meta-label: 1 iff primary was directionally correct AND the realized
    # trade (as simulated by live_sim) was profitable enough.
    correct_direction = (p_pred == y)
    profitable = pnl > cfg.pnl_threshold_pct
    y_meta = (correct_direction & profitable).astype(np.int64)

    # Meta features: primary softmax + distribution summaries + handcrafted
    # market feats if available.
    p_max = p_soft.max(axis=-1, keepdims=True)
    p_margin = (np.sort(p_soft, axis=-1)[:, -1] - np.sort(p_soft, axis=-1)[:, -2])[:, None]
    p_entropy = -(p_soft * np.log(p_soft.clip(1e-9))).sum(axis=-1, keepdims=True)
    primary_feats = np.concatenate([p_soft, p_max, p_margin, p_entropy,
                                     (p_pred == UP).astype(np.float32)[:, None]], axis=-1)
    if X_feat is not None:
        market_feats = X_feat[mask].astype(np.float32)
        X_meta = np.concatenate([primary_feats.astype(np.float32), market_feats], axis=-1)
    else:
        X_meta = primary_feats.astype(np.float32)

    # Weight by |pnl| so meta focuses on decisions that matter.
    sample_weight = np.abs(pnl).astype(np.float32)
    # Avoid zero weights (causes XGBoost instability).
    sample_weight = np.maximum(sample_weight, 1e-3)

    return X_meta, y_meta, sample_weight


def train_meta(
    X_meta: np.ndarray, y_meta: np.ndarray, sample_weight: np.ndarray,
    *, cfg: MetaConfig = MetaConfig(), val_frac: float = 0.2, seed: int = 42,
):
    """Train XGBoost meta-classifier with walk-forward split (time-ordered)."""
    import xgboost as xgb

    n = len(y_meta)
    n_val = int(n * val_frac)
    n_tr = n - n_val
    if n_tr < 200:
        raise ValueError(f"Too few meta samples: {n_tr}")

    Xtr, ytr, wtr = X_meta[:n_tr], y_meta[:n_tr], sample_weight[:n_tr]
    Xva, yva, wva = X_meta[n_tr:], y_meta[n_tr:], sample_weight[n_tr:]

    # Handle class imbalance — meta-positive rate often <50%.
    pos = ytr.sum()
    neg = len(ytr) - pos
    scale_pos_weight = max(neg / max(pos, 1), 1.0)

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(Xtr, ytr, sample_weight=wtr,
              eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)

    # Report key metrics on val
    p_val = model.predict_proba(Xva)[:, 1]
    pred_val = (p_val > cfg.meta_threshold).astype(np.int64)

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score)
    metrics = {
        "n_train": int(n_tr),
        "n_val": int(n_val),
        "pos_rate_train": float(pos / n_tr),
        "val_acc": float(accuracy_score(yva, pred_val)),
        "val_precision": float(precision_score(yva, pred_val, zero_division=0)),
        "val_recall": float(recall_score(yva, pred_val, zero_division=0)),
        "val_f1": float(f1_score(yva, pred_val, zero_division=0)),
        "val_auc": float(roc_auc_score(yva, p_val)) if len(np.unique(yva)) > 1 else float("nan"),
    }
    return model, metrics


def combine(
    primary_softmax: np.ndarray,
    meta_model,
    X_feat: np.ndarray | None = None,
    *, cfg: MetaConfig = MetaConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the trained meta model to primary predictions.

    Returns:
        take:       (N,) bool   — True iff both primary and meta say "go"
        direction:  (N,) int    — primary argmax (UP/DOWN/FLAT)
        confidence: (N,) float  — primary_max * meta_prob (0 when FLAT)
    """
    primary_pred = primary_softmax.argmax(axis=-1)
    primary_max = primary_softmax.max(axis=-1)
    non_flat_mask = primary_pred != FLAT

    # Build meta features same way build_meta_dataset does.
    p_soft = primary_softmax
    p_max = p_soft.max(axis=-1, keepdims=True)
    p_margin = (np.sort(p_soft, axis=-1)[:, -1] - np.sort(p_soft, axis=-1)[:, -2])[:, None]
    p_entropy = -(p_soft * np.log(p_soft.clip(1e-9))).sum(axis=-1, keepdims=True)
    primary_feats = np.concatenate([
        p_soft, p_max, p_margin, p_entropy,
        (primary_pred == UP).astype(np.float32)[:, None],
    ], axis=-1)
    if X_feat is not None:
        X_meta = np.concatenate([primary_feats.astype(np.float32),
                                  X_feat.astype(np.float32)], axis=-1)
    else:
        X_meta = primary_feats.astype(np.float32)

    meta_prob = np.zeros(len(primary_softmax), dtype=np.float32)
    if non_flat_mask.any():
        meta_prob[non_flat_mask] = meta_model.predict_proba(X_meta[non_flat_mask])[:, 1]

    take = (
        non_flat_mask
        & (primary_max >= cfg.min_primary_conf)
        & (meta_prob >= cfg.meta_threshold)
    )
    confidence = primary_max * meta_prob
    return take, primary_pred, confidence
