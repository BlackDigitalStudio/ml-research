"""Contract tests for meta-labeling (Lopez de Prado pattern)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.meta_label import (
    MetaConfig, build_meta_dataset, train_meta, combine,
    UP, DOWN, FLAT,
)


def _synth_predictions(n: int, seed: int = 42,
                        acc: float = 0.65,
                        pnl_edge: float = 0.08):
    """Synth primary predictions with known accuracy + PnL edge on wins."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=n).astype(np.int64)
    # Build primary softmax biased toward correct answer at `acc` rate
    primary_soft = rng.dirichlet([0.5, 0.5, 0.5], size=n).astype(np.float32)
    for i in range(n):
        if rng.random() < acc:
            primary_soft[i] = primary_soft[i] * 0.3
            primary_soft[i, y_true[i]] += 0.7
            primary_soft[i] /= primary_soft[i].sum()
    target_pnl = rng.normal(-0.02, 0.05, size=n).astype(np.float64)
    primary_pred = primary_soft.argmax(axis=-1)
    # Winning trades bias toward positive PnL
    target_pnl[primary_pred == y_true] += pnl_edge
    X_feat = rng.standard_normal((n, 34)).astype(np.float32)
    return primary_soft, y_true, target_pnl, X_feat


def test_build_meta_dataset_drops_flat_decisions():
    soft, y, pnl, feats = _synth_predictions(1000)
    X_m, y_m, w_m = build_meta_dataset(soft, y, pnl, X_feat=feats)
    # meta dataset should only contain non-FLAT primary decisions
    pred_from_soft = soft.argmax(axis=-1)
    assert len(X_m) == int((pred_from_soft != FLAT).sum())
    assert len(y_m) == len(X_m)
    assert len(w_m) == len(X_m)


def test_build_meta_dataset_all_flat_raises():
    n = 100
    # Force primary always FLAT
    soft = np.zeros((n, 3), dtype=np.float32)
    soft[:, FLAT] = 1.0
    y = np.zeros(n, dtype=np.int64)
    pnl = np.zeros(n)
    with pytest.raises(ValueError, match="FLAT for 100%"):
        build_meta_dataset(soft, y, pnl)


def test_meta_label_positive_rate_tracks_accuracy():
    """Higher primary accuracy → higher meta-positive rate."""
    _, _, _, feats = _synth_predictions(2000, acc=0.75)
    soft75, y75, pnl75, feats75 = _synth_predictions(2000, acc=0.75, seed=1)
    _, y_m75, _ = build_meta_dataset(soft75, y75, pnl75, X_feat=feats75)

    soft30, y30, pnl30, feats30 = _synth_predictions(2000, acc=0.30, seed=2)
    _, y_m30, _ = build_meta_dataset(soft30, y30, pnl30, X_feat=feats30)

    assert y_m75.mean() > y_m30.mean() + 0.15


def test_train_meta_learns_signal():
    """Trained meta XGBoost should reach AUC > 0.55 on synth with planted edge."""
    soft, y, pnl, feats = _synth_predictions(3000, acc=0.65, seed=7)
    X_m, y_m, w_m = build_meta_dataset(soft, y, pnl, X_feat=feats)
    model, metrics = train_meta(X_m, y_m, w_m, val_frac=0.25)
    assert metrics["val_auc"] > 0.55, f"AUC too low: {metrics['val_auc']}"


def test_combine_inference_respects_thresholds():
    soft, y, pnl, feats = _synth_predictions(1500, acc=0.65, seed=3)
    X_m, y_m, w_m = build_meta_dataset(soft, y, pnl, X_feat=feats)
    model, _ = train_meta(X_m, y_m, w_m)
    # Tight threshold → few takes
    take_strict, _, _ = combine(soft, model, feats,
                                cfg=MetaConfig(meta_threshold=0.8,
                                                min_primary_conf=0.6))
    # Loose threshold → more takes
    take_loose, _, _ = combine(soft, model, feats,
                                cfg=MetaConfig(meta_threshold=0.3,
                                                min_primary_conf=0.4))
    assert take_loose.sum() > take_strict.sum()


def test_combine_never_takes_flat():
    soft, y, pnl, feats = _synth_predictions(1000, seed=9)
    X_m, y_m, w_m = build_meta_dataset(soft, y, pnl, X_feat=feats)
    model, _ = train_meta(X_m, y_m, w_m)
    take, direction, _ = combine(soft, model, feats)
    # When take=True, direction must not be FLAT (by construction)
    assert not (take & (direction == FLAT)).any()
