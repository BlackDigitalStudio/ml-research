"""Stage B (cols 40..44) streaming↔batch parity + sanity tests.

Mirrors `tests/test_features_horizon.py` for Stage A. The engine is fed
mid + L1 quantities via `on_mid`, trades via `on_trade`, mark price via
`set_funding`. The batch function receives the same inputs in array form.
"""
from __future__ import annotations

import numpy as np

from src.features_ext import (
    EXT_FEATURE_KEYS,
    FeatureExtEngine,
    NUM_EXT_FEATURES,
    compute_ext_features_batch,
)

# Stage B columns in the ext vector (NOT in the full 45-col feature vector).
_OFI_60 = 6
_OFI_120 = 7
_TFI = 8
_FTN = 9
_BASIS = 10


def _synthetic_state(n: int, n_trades: int = 400, seed: int = 42):
    """Random-walk mid + slowly varying L1 qtys + sparse trade stream."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 1e-5, size=n - 1)
    log_mid = np.concatenate([[np.log(50_000.0)], np.log(50_000.0) + np.cumsum(log_ret)])
    mid = np.exp(log_mid)

    bq = 5.0 + np.abs(np.cumsum(rng.normal(0, 0.05, size=n)))
    aq = 5.0 + np.abs(np.cumsum(rng.normal(0, 0.05, size=n)))

    # Depth timestamps: start near a non-boundary UTC moment. Each tick = 100 ms.
    base_ts = 1_700_000_000_000  # ms, a real-ish epoch not aligned with funding.
    depth_ts = base_ts + np.arange(n, dtype=np.int64) * 100

    # Trade stream: random sample of timestamps within depth window.
    trade_ts = np.sort(rng.integers(base_ts, base_ts + n * 100, size=n_trades).astype(np.int64))
    trade_qty = rng.uniform(0.01, 2.0, size=n_trades)
    trade_side = rng.integers(0, 2, size=n_trades).astype(bool)  # True => sell aggressor
    signed = np.where(trade_side, -trade_qty, trade_qty)

    return mid, bq, aq, depth_ts, trade_ts, signed, trade_side, trade_qty


def _stream_ref(mid, bq, aq, depth_ts, trade_ts, signed, mark_price, indices):
    """Drive FeatureExtEngine end-to-end; emit ext vectors at `indices`."""
    eng = FeatureExtEngine()
    if mark_price is not None and mark_price > 0:
        eng.set_funding(float(mark_price))
    out = np.zeros((len(indices), NUM_EXT_FEATURES), dtype=np.float32)
    wanted = {int(i): k for k, i in enumerate(indices)}

    n = len(mid)
    tr_cursor = 0
    n_tr = len(trade_ts)
    for t in range(n):
        # Feed all trades with ts < depth_ts[t] before ingesting the tick.
        while tr_cursor < n_tr and trade_ts[tr_cursor] < depth_ts[t]:
            eng.on_trade(int(trade_ts[tr_cursor]), float(signed[tr_cursor]))
            tr_cursor += 1
        eng.on_mid(int(depth_ts[t]), float(mid[t]), float(bq[t]), float(aq[t]))
        # Drain trades whose ts equals this tick too (side="right" join
        # semantics in the batch path).
        while tr_cursor < n_tr and trade_ts[tr_cursor] <= depth_ts[t]:
            eng.on_trade(int(trade_ts[tr_cursor]), float(signed[tr_cursor]))
            tr_cursor += 1
        if t in wanted:
            out[wanted[t]] = eng.get().copy()
    return out


def test_ext_has_all_eleven():
    assert NUM_EXT_FEATURES == 11
    assert EXT_FEATURE_KEYS[_OFI_60] == "ofi_60s"
    assert EXT_FEATURE_KEYS[_OFI_120] == "ofi_120s"
    assert EXT_FEATURE_KEYS[_TFI] == "trade_flow_imbalance_60s"
    assert EXT_FEATURE_KEYS[_FTN] == "funding_time_to_next_min"
    assert EXT_FEATURE_KEYS[_BASIS] == "funding_basis_bps"


def test_stream_matches_batch_stage_b():
    n = 3000
    mid, bq, aq, depth_ts, trade_ts, signed, _, _ = _synthetic_state(n, n_trades=500)
    funding_ts = np.array([depth_ts[0] - 5_000, depth_ts[1000]], dtype=np.int64)
    funding_mark = np.array([0.0, mid[1000] * (1.0 + 3e-4)], dtype=np.float64)
    # Streaming engine only tracks the most recent mark, not history.
    latest_mark = funding_mark[-1]

    indices = np.array([0, 100, 599, 600, 1200, 1500, 2500, 2999], dtype=np.int64)

    ref = _stream_ref(mid, bq, aq, depth_ts, trade_ts, signed, latest_mark, indices)
    got = compute_ext_features_batch(
        mid.astype(np.float64), indices,
        bid_qty0=bq, ask_qty0=aq, depth_ts_ms=depth_ts,
        trade_ts_ms=trade_ts, trade_signed_qty=signed,
        funding_ts_ms=funding_ts, funding_mark=funding_mark,
    )

    # Stage B columns parity. Note: basis matches only for samples whose
    # sample_ts >= funding_ts[1] (before that the streaming path also had
    # the mark set — the engine carries the latest value forward). For
    # realism we gate the comparison to cols whose semantics are identical
    # in both paths for all sample indices.
    for col in (_OFI_60, _OFI_120, _TFI, _FTN):
        diff = float(np.max(np.abs(ref[:, col] - got[:, col])))
        name = EXT_FEATURE_KEYS[col]
        # OFI uses running incremental sums in the stream vs cumsum in the
        # batch path — f32 rounding diverges slightly on large windows.
        tol = 2e-5 if col in (_OFI_60, _OFI_120) else 1e-6
        assert diff < tol, f"{name} stream↔batch diverged: max |Δ| = {diff:g}"


def test_ofi_sign_and_zero_edges():
    n = 1500
    bq = np.full(n, 5.0)
    aq = np.full(n, 5.0)
    # Manufacture a clean bid-add pulse for 100 ticks in the middle.
    bq[700:800] = 6.0
    mid = np.full(n, 50_000.0)
    depth_ts = np.arange(n, dtype=np.int64) * 100
    idx = np.array([699, 750, 799, 800, 1400], dtype=np.int64)

    out = compute_ext_features_batch(
        mid, idx,
        bid_qty0=bq, ask_qty0=aq, depth_ts_ms=depth_ts,
    )
    # Window [t-600, t-1]. At idx=1400 → window [800..1399] sees only the
    # remove-leg of the pulse (bq 6→5 at tick 800) because the add-leg at
    # tick 700 is out of window. Sum = -1.
    assert abs(out[4, _OFI_60] - (-1.0)) < 1e-6
    # Before window saturates cols stay 0.
    assert out[0, _OFI_60] == 0.0
    # Inside the pulse-on phase we see a positive imbalance.
    # At idx=750 the window covers [150..749]; bid_qty went 5→6 at tick 700,
    # so raw OFI spiked +1 once. Sum across the window should be +1.
    assert abs(out[1, _OFI_60] - 1.0) < 1e-6


def test_tfi_buy_heavy():
    n = 200
    mid = np.full(n, 50_000.0)
    bq = aq = np.full(n, 5.0)
    depth_ts = np.arange(n, dtype=np.int64) * 100
    # 50 trades in the last 60s, 80 % buyer-initiated.
    trade_ts = np.linspace(depth_ts[-1] - 50_000, depth_ts[-1] - 100, 50).astype(np.int64)
    signed = np.where(np.arange(50) < 40, 1.0, -1.0)
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        bid_qty0=bq, ask_qty0=aq, depth_ts_ms=depth_ts,
        trade_ts_ms=trade_ts, trade_signed_qty=signed,
    )
    # Expected: (40 - 10) / 50 = 0.6
    assert abs(out[0, _TFI] - 0.6) < 1e-6


def test_funding_time_to_next_is_in_range():
    # Depth ts placed exactly at a funding boundary -> 0 min; then 1 hour past -> 7 hours.
    boundary = 1_700_006_400_000  # arbitrary 8-hour aligned ms (divisible by 8h).
    period = 8 * 3600 * 1000
    boundary = (boundary // period) * period  # align
    n = 10
    mid = np.full(n, 50_000.0)
    depth_ts = boundary + np.array([0, 3_600_000, 2 * 3_600_000, 7 * 3_600_000,
                                    7 * 3_600_000 + 1, period - 1, period, period + 1,
                                    period + 3_600_000, period + 7 * 3_600_000],
                                   dtype=np.int64)
    out = compute_ext_features_batch(
        mid, np.arange(n, dtype=np.int64),
        bid_qty0=np.zeros(n), ask_qty0=np.zeros(n), depth_ts_ms=depth_ts,
    )
    got = out[:, _FTN]
    # Assertions: all >= 0 and <= 480 (8h * 60 min).
    assert (got >= 0.0).all() and (got <= 480.0).all()
    # First sample exactly on boundary -> 0.
    assert got[0] == 0.0
    # Second sample 1h past boundary -> 7 h = 420 min.
    assert abs(got[1] - 420.0) < 1e-3


def test_basis_bps_sign():
    n = 50
    mid = np.full(n, 50_000.0)
    depth_ts = np.arange(n, dtype=np.int64) * 100
    # Mark 0.1 % above mid at ts=0; basis = (50050 - 50000)/50000*1e4 = 10 bps.
    funding_ts = np.array([depth_ts[0] - 1000], dtype=np.int64)
    funding_mark = np.array([50_050.0], dtype=np.float64)
    out = compute_ext_features_batch(
        mid, np.array([n - 1], dtype=np.int64),
        bid_qty0=np.zeros(n), ask_qty0=np.zeros(n), depth_ts_ms=depth_ts,
        funding_ts_ms=funding_ts, funding_mark=funding_mark,
    )
    assert abs(out[0, _BASIS] - 10.0) < 1e-4


def test_stage_b_cols_zero_when_inputs_missing():
    n = 500
    mid = np.full(n, 50_000.0)
    out = compute_ext_features_batch(mid, np.arange(0, n, 50, dtype=np.int64))
    # Only Stage A cols can be non-zero here (and they are zero on flat mid).
    assert np.all(out[:, 6:] == 0.0)
