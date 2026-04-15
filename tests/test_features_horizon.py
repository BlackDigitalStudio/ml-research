"""Parity test: streaming `FeatureExtEngine` must match the batch
`compute_ext_features_batch` on the same mid-price series within f32
tolerance. This is the canonical spec for Stage-A features; the Rust
port must then match the batch version.

Run:  pytest tests/test_features_ext.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features_ext import (
    EXT_FEATURE_KEYS,
    FeatureExtEngine,
    NUM_EXT_FEATURES,
    compute_ext_features_batch,
)


def _synthetic_mid(n: int, seed: int = 42) -> np.ndarray:
    """Random-walk mid with occasional jumps (stresses bipower)."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 1e-5, size=n - 1)
    # Inject a handful of 5-sigma jumps.
    jumps = rng.choice(n - 1, size=max(1, n // 500), replace=False)
    log_ret[jumps] += rng.choice([-1.0, 1.0], size=len(jumps)) * 5e-5
    log_mid = np.concatenate([[np.log(50_000.0)], np.log(50_000.0) + np.cumsum(log_ret)])
    return np.exp(log_mid)


def _stream_ref(mid: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Drive FeatureExtEngine tick-by-tick; sample at `indices`."""
    eng = FeatureExtEngine()
    out = np.zeros((len(indices), NUM_EXT_FEATURES), dtype=np.float32)
    wanted = {int(i): k for k, i in enumerate(indices)}
    for t, m in enumerate(mid):
        eng.on_mid(float(m))
        if t in wanted:
            out[wanted[t]] = eng.get().copy()
    return out


def test_ext_keys_are_six() -> None:
    assert NUM_EXT_FEATURES == 6
    assert len(EXT_FEATURE_KEYS) == 6


def test_streaming_matches_batch_dense() -> None:
    n = 2500
    mid = _synthetic_mid(n)
    indices = np.arange(0, n, dtype=np.int64)

    ref = _stream_ref(mid, indices)
    got = compute_ext_features_batch(mid.astype(np.float64), indices)

    for col, name in enumerate(EXT_FEATURE_KEYS):
        # Float accumulation path differs (stream uses running sums, batch
        # uses cumsum) so we compare in f32 with a loose absolute tolerance
        # that's still far tighter than any real signal scale.
        max_abs = float(np.max(np.abs(ref[:, col] - got[:, col])))
        assert max_abs < 1e-6, f"column {col} ({name}) diverged: max |Δ| = {max_abs:g}"


def test_streaming_matches_batch_sparse_indices() -> None:
    n = 3000
    mid = _synthetic_mid(n, seed=7)
    indices = np.array([0, 10, 299, 300, 599, 600, 1199, 1200, 2500, 2999], dtype=np.int64)

    ref = _stream_ref(mid, indices)
    got = compute_ext_features_batch(mid.astype(np.float64), indices)

    diff = np.max(np.abs(ref - got))
    assert diff < 1e-6, f"max |Δ| across all cols = {diff:g}"


def test_zero_and_short_series_are_safe() -> None:
    # Fewer ticks than any window → all zeros.
    mid = np.full(50, 50_000.0, dtype=np.float64)
    out = compute_ext_features_batch(mid, np.arange(50, dtype=np.int64))
    assert np.all(out == 0.0)

    # Constant mid → returns all zero → all features zero even with full windows.
    mid = np.full(2_000, 50_000.0, dtype=np.float64)
    out = compute_ext_features_batch(mid, np.arange(0, 2_000, 200, dtype=np.int64))
    assert np.all(np.abs(out) < 1e-10)


def test_momentum_sign_is_correct() -> None:
    # Monotone up mid → momentum features must be positive once window fills.
    n = 1500
    mid = np.linspace(50_000.0, 50_500.0, n, dtype=np.float64)
    indices = np.array([n - 1], dtype=np.int64)
    out = compute_ext_features_batch(mid, indices)
    assert out[0, 0] > 0  # momentum_30s
    assert out[0, 1] > 0  # momentum_60s
    assert out[0, 2] > 0  # momentum_120s


def test_bipower_robustness_to_single_jump() -> None:
    # Flat series with one isolated jump — bipower should stay small because
    # the jump only contributes through ONE adjacent-product pair.
    n = 2_000
    mid = np.full(n, 50_000.0, dtype=np.float64)
    mid[1_000:] += 50.0     # single 0.1 % jump, flat after
    indices = np.array([n - 1], dtype=np.int64)
    out = compute_ext_features_batch(mid, indices)
    rv = out[0, 4]          # realized_vol_120s
    bv = out[0, 5]          # bipower_var_120s
    # RV squares the jump once; BV multiplies it by an adjacent zero return
    # on both sides, so BV << RV^2. If this is ever violated, bipower is broken.
    assert bv < rv * rv * 0.5, f"bipower not robust to jumps: BV={bv:g}, RV²={rv*rv:g}"
