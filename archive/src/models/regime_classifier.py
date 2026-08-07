"""Regime classifier — predicts whether the next-K-sample window is a
"profitable regime" for our trading strategy.

Motivation (2026-04-14 B3 finding): the current 5-arch ensemble shows
regime-dependent edge — profitable on 5/10 time chunks, losing on 5/10.
A regime classifier trained on microstructure + time features lets the
grid gate out samples whose upcoming regime the model can't handle,
pulling the global OOS number above break-even without changing the
primary signal itself.

This module is INTENTIONALLY minimal:

    input:  X_feat (N, 34)   — handcrafted features at sample tick
            timestamps (N,)  — ms since epoch, used to derive time-of-day +
                                rolling pnl_history (strictly causal).
    target: rolling_forward_pnl_positive? — mean(pnl[i:i+K]) > threshold
    model:  XGBClassifier binary
    output: regime_prob in [0, 1]

The threshold and K are configurable. A practical default is
`K=500, threshold=0.0` — "on average, the next 500 samples have positive
net PnL per candidate trade."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RegimeConfig:
    lookahead_k: int = 500          # forward window size
    # Regime = "high-opportunity density": sum of POSITIVE target_pnl in
    # next K samples > threshold. target_pnl is >0 only for TB-winner
    # samples (UP/DOWN correctly predicted hits TP first); FLAT samples
    # have negative target_pnl (hypothetical losses). Summing only the
    # positive part approximates "how much profit-opportunity is in this
    # window if our primary is right on the non-FLAT samples".
    pnl_sum_threshold_pct: float = 0.20  # cumulative bp of positive pnl
    rolling_window: int = 200       # past window for rolling pnl stats
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    seed: int = 42

    # Which X_feat columns the regime classifier pays attention to on the
    # post-Stage-E 49-feature set. Stage E pruned raw indices 5/17/18/19/
    # 21/22/23 — removing the features the pre-Stage-E regime classifier
    # relied on (oi_delta, vol_ratio, hurst, trade_intensity_ratio).
    # Replacement uses richer Stage-A..D horizon-tier features that were
    # specifically designed to match the 60-180 s holding zone.
    #
    #   3  = spread                     (raw 3, unchanged)
    #   9  = volatility_1s              (raw 10 - 1 drop before it)
    #   12 = funding_rate               (raw 13)
    #   18 = cancel_rate_diff           (raw 25)
    #   20 = ofi_5s                     (raw 27)
    #   21 = ofi_30s                    (raw 28)
    #   30 = realized_vol_60s           (Stage A — replaces vol_ratio)
    #   31 = realized_vol_120s          (Stage A)
    #   32 = bipower_var_120s           (Stage A — jump-robust variation)
    #   33 = ofi_60s                    (Stage B — horizon OFI)
    #   34 = ofi_120s                   (Stage B)
    #   40 = kyle_lambda_60s            (Stage C — price-impact regime)
    #   41 = vpin_60s                   (Stage C — toxic flow proxy)
    #   48 = eth_btc_corr_30s           (Stage D — cross-asset regime)
    feat_columns: tuple[int, ...] = (3, 9, 12, 18, 20, 21,
                                       30, 31, 32, 33, 34, 40, 41, 48)


def _time_of_day_features(timestamps_ms: np.ndarray) -> np.ndarray:
    """(N, 4) — hour sin/cos + day_of_week one-hot collapsed to sin/cos."""
    sec = timestamps_ms // 1000
    # Hour of day [0, 24).
    hour = (sec % 86400) / 3600.0
    hs = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hc = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    # Day of week [0, 7), we use sin/cos to preserve cyclic nature.
    dow = ((sec // 86400) + 4) % 7   # UNIX epoch = Thu, so +4 → Mon=0
    ds = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    dc = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    return np.stack([hs, hc, ds, dc], axis=1)


def _rolling_pnl_features(pnl: np.ndarray, window: int) -> np.ndarray:
    """(N, 3) — causal rolling [win_rate, mean_pnl, abs_pnl] over past K."""
    n = len(pnl)
    win = (pnl > 0).astype(np.float64)
    cs_win = np.concatenate(([0.0], np.cumsum(win)))
    cs_pnl = np.concatenate(([0.0], np.cumsum(pnl.astype(np.float64))))
    cs_abs = np.concatenate(([0.0], np.cumsum(np.abs(pnl).astype(np.float64))))
    hi = np.arange(1, n + 1)
    lo = np.maximum(hi - window, 0)
    denom = np.maximum((hi - lo).astype(np.float64), 1.0)
    rw = ((cs_win[hi] - cs_win[lo]) / denom).astype(np.float32)
    rp = ((cs_pnl[hi] - cs_pnl[lo]) / denom).astype(np.float32)
    ra = ((cs_abs[hi] - cs_abs[lo]) / denom).astype(np.float32)
    return np.stack([rw, rp, ra], axis=1)


def build_regime_features(
    X_feat: np.ndarray,
    timestamps_ms: np.ndarray,
    pnl: np.ndarray,
    cfg: RegimeConfig = RegimeConfig(),
) -> np.ndarray:
    """Concatenate selected X_feat columns + time-of-day + rolling PnL."""
    sel = X_feat[:, list(cfg.feat_columns)].astype(np.float32)
    tod = _time_of_day_features(timestamps_ms)
    roll = _rolling_pnl_features(pnl, cfg.rolling_window)
    return np.concatenate([sel, tod, roll], axis=1)


def build_regime_labels(
    pnl: np.ndarray, cfg: RegimeConfig = RegimeConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-looking binary label (1 = next-K *positive* pnl sum > threshold).

    target_pnl is positive only for triple-barrier-winner samples (UP/DOWN
    that hit TP first); FLAT samples have negative target_pnl (hypothetical
    losses). We score "opportunity density" by summing only the positive
    part over the forward window, so the label captures "how much winning
    PnL is available if the primary is right on the non-FLAT samples".

    Returns (y_regime, valid_mask). valid_mask excludes the last K samples
    where the forward window isn't fully observable.
    """
    n = len(pnl)
    k = cfg.lookahead_k
    if n <= k:
        raise ValueError(f"Need more than {k} samples; got {n}")
    pos = np.where(pnl > 0, pnl, 0.0).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(pos)))      # length n+1
    fw_pos_sum = (cs[k:] - cs[:n - k + 1])             # length n-k+1
    y = np.zeros(n, dtype=np.int64)
    y[: n - k + 1] = (fw_pos_sum > cfg.pnl_sum_threshold_pct).astype(np.int64)
    valid = np.zeros(n, dtype=bool)
    valid[: n - k + 1] = True
    return y, valid


def train_regime_classifier(
    X_reg: np.ndarray, y_reg: np.ndarray, valid: np.ndarray,
    *, cfg: RegimeConfig = RegimeConfig(), val_frac: float = 0.2,
):
    """Walk-forward XGBoost training. Returns (model, metrics)."""
    import xgboost as xgb
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                   recall_score, roc_auc_score)

    X = X_reg[valid]
    y = y_reg[valid]
    n = len(y)
    n_val = int(n * val_frac)
    n_tr = n - n_val
    if n_tr < 500:
        raise ValueError(f"Too few regime samples: {n_tr}")

    pos = int(y[:n_tr].sum())
    neg = n_tr - pos
    scale_pos_weight = max(neg / max(pos, 1), 1.0)

    m = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=40,
        random_state=cfg.seed, n_jobs=-1, verbosity=0,
    )
    m.fit(X[:n_tr], y[:n_tr], eval_set=[(X[n_tr:], y[n_tr:])], verbose=False)

    p_val = m.predict_proba(X[n_tr:])[:, 1]
    pred_val = (p_val > 0.5).astype(np.int64)
    metrics = {
        "n_train": int(n_tr), "n_val": int(n_val),
        "pos_rate_train": float(pos / n_tr),
        "val_acc":       float(accuracy_score(y[n_tr:], pred_val)),
        "val_precision": float(precision_score(y[n_tr:], pred_val, zero_division=0)),
        "val_recall":    float(recall_score(y[n_tr:], pred_val, zero_division=0)),
        "val_f1":        float(f1_score(y[n_tr:], pred_val, zero_division=0)),
        "val_auc":       float(roc_auc_score(y[n_tr:], p_val)) if len(np.unique(y[n_tr:])) > 1 else float("nan"),
    }
    return m, metrics


def predict_regime_prob(
    model, X_feat: np.ndarray, timestamps_ms: np.ndarray, pnl: np.ndarray,
    cfg: RegimeConfig = RegimeConfig(),
) -> np.ndarray:
    X_reg = build_regime_features(X_feat, timestamps_ms, pnl, cfg)
    return model.predict_proba(X_reg)[:, 1].astype(np.float32)
