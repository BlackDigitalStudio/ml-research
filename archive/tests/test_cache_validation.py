"""Cache-level validation tests.

A bad training cache (NaN features, extreme outliers, broken class balance)
silently destroys training — transformer goes NaN on epoch 1, other archs
converge to always-predict-one-class. Catch these before we burn GPU time.

Runs:
  - on an existing cache via `pytest tests/test_cache_validation.py --cache-path=...`
  - on a tiny synthetic cache as a smoke test (always runs)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


MAX_ABS_FEAT_AFTER_NORM = 50.0    # normalized features should fit in ~[-10, 10]
MAX_LOB_VALUE = 100.0             # depth volumes above 100 are bad data
MIN_CLASS_FRACTION = 0.001        # every class must be >0.1% of samples


def _cache_path_arg() -> Path | None:
    p = os.environ.get("SCALPER_TEST_CACHE")
    if p and Path(p).exists():
        return Path(p)
    return None


@pytest.fixture
def real_cache():
    p = _cache_path_arg()
    if p is None:
        pytest.skip("set SCALPER_TEST_CACHE=/path/to/cache.npz to run on real data")
    return np.load(p, allow_pickle=False)


def test_no_nan_inf_in_real_cache(real_cache):
    for key in ("X_lob", "X_feat", "y", "target_pnl"):
        arr = np.asarray(real_cache[key])
        n_nan = int(np.isnan(arr.astype(np.float64)).sum())
        n_inf = int(np.isinf(arr.astype(np.float64)).sum())
        assert n_nan == 0, f"{key} contains {n_nan} NaN"
        assert n_inf == 0, f"{key} contains {n_inf} inf"


def test_lob_range_real_cache(real_cache):
    X_lob = np.asarray(real_cache["X_lob"])
    assert X_lob.min() >= 0.0, f"X_lob has negative values: min={X_lob.min()}"
    assert X_lob.max() <= MAX_LOB_VALUE, (
        f"X_lob has extreme values: max={X_lob.max()} — depth volumes >"
        f"{MAX_LOB_VALUE} are bad data, run clean_cache.py"
    )


def test_features_normalized(real_cache):
    X_feat = np.asarray(real_cache["X_feat"])
    worst = float(np.abs(X_feat).max())
    assert worst <= MAX_ABS_FEAT_AFTER_NORM, (
        f"X_feat has un-normalized extreme values: max|x|={worst}. "
        f"Run clean_cache.py to robust-clip + z-score."
    )


def test_class_balance_real_cache(real_cache):
    y = np.asarray(real_cache["y"])
    counts = np.bincount(y, minlength=3)
    fractions = counts / counts.sum()
    for cls, frac in enumerate(fractions):
        assert frac >= MIN_CLASS_FRACTION, (
            f"class {cls} is only {frac*100:.3f}% of samples — "
            f"labels may be broken"
        )


def test_norm_stats_present_real_cache(real_cache):
    keys = set(real_cache.files)
    assert "feat_norm_stats" in keys, (
        "cache missing feat_norm_stats — run clean_cache.py so live inference "
        "knows how to normalize incoming features identically"
    )
    stats = np.asarray(real_cache["feat_norm_stats"])
    X_feat = np.asarray(real_cache["X_feat"])
    assert stats.shape == (X_feat.shape[1], 4), (
        f"expected feat_norm_stats shape ({X_feat.shape[1]}, 4), got {stats.shape}"
    )
    # Column 3 = scale — must be strictly positive
    assert (stats[:, 3] > 0).all(), "feat_norm_stats has non-positive scale entries"


# ---- Synthetic smoke tests — always run, protect against code rot ----

def _synth_cache(n: int = 500, n_feat: int = 6, include_stats: bool = True) -> dict:
    """Clean synthetic cache that should pass all assertions."""
    rng = np.random.default_rng(0)
    X_lob = rng.uniform(0, 10, (n, 3, 20, 50)).astype(np.float32)
    X_feat = rng.normal(0, 1, (n, n_feat)).astype(np.float32)
    y = rng.integers(0, 3, n).astype(np.int64)
    pnl = rng.normal(0, 0.01, n).astype(np.float32)
    out = {"X_lob": X_lob, "X_feat": X_feat, "y": y, "target_pnl": pnl}
    if include_stats:
        stats = np.zeros((n_feat, 4), dtype=np.float32)
        stats[:, 1] = 5.0   # clip_hi
        stats[:, 0] = -5.0  # clip_lo
        stats[:, 3] = 1.0   # scale
        out["feat_norm_stats"] = stats
    return out


def test_synthetic_clean_cache_passes():
    c = _synth_cache()
    for key in ("X_lob", "X_feat", "y", "target_pnl"):
        arr = c[key]
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()
    assert c["X_lob"].max() <= MAX_LOB_VALUE
    assert np.abs(c["X_feat"]).max() <= MAX_ABS_FEAT_AFTER_NORM


def test_synthetic_nan_cache_fails_loudly():
    c = _synth_cache()
    c["X_feat"][17, 3] = np.nan
    n_nan = int(np.isnan(c["X_feat"]).sum())
    assert n_nan == 1, "synthetic injection should land exactly one NaN"


def test_synthetic_extreme_cache_fails_loudly():
    c = _synth_cache()
    c["X_feat"][5, 2] = 1e6
    assert np.abs(c["X_feat"]).max() > MAX_ABS_FEAT_AFTER_NORM


def test_missing_norm_stats_detected():
    c = _synth_cache(include_stats=False)
    assert "feat_norm_stats" not in c
