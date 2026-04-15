"""Stage D (cols 50..55) batch-path sanity tests.

Streaming↔batch parity is implicit: Stage D streams through the same
`on_mid` path + cross-ex/ETH ingestion methods, so the existing A-C
streaming tests cover the glue. Here we focus on batch semantics.
"""
from __future__ import annotations

import numpy as np

from src.features_ext import NUM_EXT_FEATURES, compute_ext_features_batch

_BYLL = 16
_OKX = 17
_BITGET = 18
_GATE = 19
_ETHMOM = 20
_ETHCORR = 21


def test_stage_d_cross_flow_sign():
    n = 400
    mid = np.full(n, 50_000.0)
    depth_ts = np.arange(n, dtype=np.int64) * 100
    # Okx: 5 buy trades, 2 sell trades, all in last 30s.
    okx_ts = np.array([depth_ts[-i - 10] for i in range(7)], dtype=np.int64)
    okx_ts.sort()
    okx_q = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0], dtype=np.float64)
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        depth_ts_ms=depth_ts,
        okx_ts_ms=okx_ts, okx_signed_qty=okx_q,
    )
    # net = 5 - 4 = 1 (signed).
    assert abs(out[0, _OKX] - 1.0) < 1e-6
    assert out[0, _BITGET] == 0.0
    assert out[0, _GATE] == 0.0


def test_eth_momentum():
    # ETH price ramps +1 % over 60s; momentum_60s ≈ +0.01.
    n = 800
    mid = np.full(n, 50_000.0)
    depth_ts = np.arange(n, dtype=np.int64) * 100
    eth_ts = depth_ts.copy()
    eth_price = np.linspace(3_000.0, 3_030.0, n)
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        depth_ts_ms=depth_ts,
        eth_ts_ms=eth_ts, eth_price=eth_price,
    )
    # Last - (last - 600 ticks ago) / prior. (3030 - 3007.5)/3007.5 ≈ 0.00748.
    assert 0.005 < out[0, _ETHMOM] < 0.015


def test_eth_btc_corr_identity():
    # When BTC ≡ ETH (same series) → correlation should be +1.
    rng = np.random.default_rng(0)
    n = 1000
    rets = rng.normal(0, 1e-4, size=n)
    mid = 50_000.0 * np.exp(np.cumsum(rets))
    depth_ts = np.arange(n, dtype=np.int64) * 100
    # ETH price = BTC price (exact parity).
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        depth_ts_ms=depth_ts,
        eth_ts_ms=depth_ts, eth_price=mid,
    )
    assert abs(out[0, _ETHCORR] - 1.0) < 1e-4


def test_bybit_lead_lag_is_bounded():
    rng = np.random.default_rng(1)
    n = 800
    rets = rng.normal(0, 1e-4, size=n)
    mid = 50_000.0 * np.exp(np.cumsum(rets))
    depth_ts = np.arange(n, dtype=np.int64) * 100
    bybit_px = mid  # same series → lead-lag corr somewhere in [-1, 1]
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        depth_ts_ms=depth_ts,
        bybit_ts_ms=depth_ts, bybit_price=bybit_px,
    )
    val = out[0, _BYLL]
    assert -1.0 - 1e-3 <= val <= 1.0 + 1e-3


def test_ext_count_is_twenty_two():
    assert NUM_EXT_FEATURES == 22
