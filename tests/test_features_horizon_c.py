"""Stage C (cols 45..49) streaming↔batch parity + sanity tests.

Covers microprice_deviation, ofi_top5_weighted, kyle_lambda_60s, vpin_60s,
cancel_to_trade_ratio_30s.
"""
from __future__ import annotations

import numpy as np

from src.features_ext import (
    EXT_FEATURE_KEYS,
    FeatureExtEngine,
    NUM_EXT_FEATURES,
    compute_ext_features_batch,
)

# Stage C slots in the ext vector.
_MICRO = 11
_OFI5 = 12
_KYLE = 13
_VPIN = 14
_CTR = 15


def _synth(n: int, n_trades: int = 500, seed: int = 13):
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 1e-5, size=n - 1)
    log_mid = np.concatenate([[np.log(50_000.0)], np.log(50_000.0) + np.cumsum(log_ret)])
    mid = np.exp(log_mid)
    # top-5 bid/ask: realistic decaying qty, price steps of $1.
    bp5 = (mid[:, None] - np.arange(5) - 1).astype(np.float64)
    ap5 = (mid[:, None] + np.arange(5) + 1).astype(np.float64)
    bq5 = np.maximum(0.1, 5.0 + rng.normal(0, 0.3, size=(n, 5))).astype(np.float64)
    aq5 = np.maximum(0.1, 5.0 + rng.normal(0, 0.3, size=(n, 5))).astype(np.float64)

    base_ts = 1_700_000_000_000
    depth_ts = base_ts + np.arange(n, dtype=np.int64) * 100

    trade_ts = np.sort(
        rng.integers(base_ts, base_ts + n * 100, size=n_trades).astype(np.int64)
    )
    trade_q = rng.uniform(0.01, 2.0, size=n_trades).astype(np.float64)
    trade_side = rng.integers(0, 2, size=n_trades).astype(bool)  # True → seller-aggressor
    signed = np.where(trade_side, -trade_q, trade_q)
    return mid, bp5, bq5, ap5, aq5, depth_ts, trade_ts, signed


def _stream_ref(mid, bp5, bq5, ap5, aq5, depth_ts, trade_ts, signed, indices):
    eng = FeatureExtEngine()
    out = np.zeros((len(indices), NUM_EXT_FEATURES), dtype=np.float32)
    wanted = {int(i): k for k, i in enumerate(indices)}
    n = len(mid)
    tr_cursor = 0
    n_tr = len(trade_ts)
    for t in range(n):
        # Flush trades strictly before this tick first.
        while tr_cursor < n_tr and trade_ts[tr_cursor] < depth_ts[t]:
            eng.on_trade(int(trade_ts[tr_cursor]), float(signed[tr_cursor]))
            tr_cursor += 1
        eng.on_depth_l5(int(depth_ts[t]), float(mid[t]),
                        bp5[t], bq5[t], ap5[t], aq5[t])
        # Flush trades whose ts equals current tick (matches searchsorted "right" semantics).
        while tr_cursor < n_tr and trade_ts[tr_cursor] <= depth_ts[t]:
            eng.on_trade(int(trade_ts[tr_cursor]), float(signed[tr_cursor]))
            tr_cursor += 1
        if t in wanted:
            out[wanted[t]] = eng.get().copy()
    return out


def test_stage_c_keys_present():
    assert EXT_FEATURE_KEYS[_MICRO] == "microprice_deviation"
    assert EXT_FEATURE_KEYS[_OFI5] == "ofi_top5_weighted"
    assert EXT_FEATURE_KEYS[_KYLE] == "kyle_lambda_60s"
    assert EXT_FEATURE_KEYS[_VPIN] == "vpin_60s"
    assert EXT_FEATURE_KEYS[_CTR] == "cancel_to_trade_ratio_30s"


def test_microprice_deviation_sign():
    # Fat bid → microprice above mid → positive deviation.
    n = 5
    bp5 = np.tile(np.array([99.0, 98.0, 97.0, 96.0, 95.0]), (n, 1))
    ap5 = np.tile(np.array([101.0, 102.0, 103.0, 104.0, 105.0]), (n, 1))
    bq5 = np.tile(np.array([10.0, 1.0, 1.0, 1.0, 1.0]), (n, 1))
    aq5 = np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), (n, 1))
    mid = 0.5 * (bp5[:, 0] + ap5[:, 0])
    depth_ts = np.arange(n, dtype=np.int64) * 100
    out = compute_ext_features_batch(
        mid, np.arange(n, dtype=np.int64),
        bid_prices_top5=bp5, bid_qtys_top5=bq5,
        ask_prices_top5=ap5, ask_qtys_top5=aq5,
        depth_ts_ms=depth_ts,
    )
    # microprice = (1·99 + 10·101) / 11 ≈ 100.818; mid = 100; spread = 2
    # dev = (100.818 - 100) / 2 ≈ 0.409
    assert out[0, _MICRO] > 0.3 and out[0, _MICRO] < 0.5


def test_cancel_to_trade_smoke():
    # No cancels (qtys constant) → ratio = 0.
    n = 600
    bp5 = np.tile(np.array([99.0, 98.0, 97.0, 96.0, 95.0]), (n, 1))
    ap5 = np.tile(np.array([101.0, 102.0, 103.0, 104.0, 105.0]), (n, 1))
    bq5 = np.ones((n, 5))
    aq5 = np.ones((n, 5))
    mid = np.full(n, 100.0)
    depth_ts = np.arange(n, dtype=np.int64) * 100
    trade_ts = depth_ts[::10]
    signed = np.ones_like(trade_ts, dtype=np.float64)
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        bid_prices_top5=bp5, bid_qtys_top5=bq5,
        ask_prices_top5=ap5, ask_qtys_top5=aq5,
        depth_ts_ms=depth_ts,
        trade_ts_ms=trade_ts, trade_signed_qty=signed,
    )
    assert out[0, _CTR] == 0.0


def test_stream_matches_batch_stage_c():
    n = 2000
    mid, bp5, bq5, ap5, aq5, depth_ts, trade_ts, signed = _synth(n, n_trades=800)
    indices = np.array([0, 299, 300, 599, 600, 1199, 1200, 1999], dtype=np.int64)
    ref = _stream_ref(mid, bp5, bq5, ap5, aq5, depth_ts, trade_ts, signed, indices)
    got = compute_ext_features_batch(
        mid.astype(np.float64), indices,
        bid_qty0=bq5[:, 0], ask_qty0=aq5[:, 0], depth_ts_ms=depth_ts,
        trade_ts_ms=trade_ts, trade_signed_qty=signed,
        bid_prices_top5=bp5, bid_qtys_top5=bq5,
        ask_prices_top5=ap5, ask_qtys_top5=aq5,
    )
    # Microprice is per-sample and identical (no window drift).
    assert float(np.max(np.abs(ref[:, _MICRO] - got[:, _MICRO]))) < 1e-5
    # Weighted OFI uses running sum vs cumsum; f32 tolerance.
    assert float(np.max(np.abs(ref[:, _OFI5] - got[:, _OFI5]))) < 2e-5
    # Kyle's lambda: accumulates f32 ratio; denominators are tiny (x² of trade qty),
    # so the final ratio is sensitive to f32 rounding — allow 1 % of max.
    # Absolute tol 1e-6 is adequate at this synthetic scale (~5e-7).
    assert float(np.max(np.abs(ref[:, _KYLE] - got[:, _KYLE]))) < 1e-6
    # VPIN: bounded in [0,1]; running ring buckets vs searchsorted — must match
    # almost exactly because trade ingestion uses exact ts.
    assert float(np.max(np.abs(ref[:, _VPIN] - got[:, _VPIN]))) < 1e-6
