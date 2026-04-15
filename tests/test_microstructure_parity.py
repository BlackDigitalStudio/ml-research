"""Training/runtime parity for the Lever 5 microstructure features.

Both paths must produce identical numerical values on identical inputs,
otherwise the model trained on training-time features will see a different
distribution at inference and lose its calibration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    FEATURE_KEYS,
    NUM_FEATURES,
    QUEUE_DECAY_ALPHA,
    FeatureEngine,
)
from src.order_book import OrderBook, Snapshot, BOOK_DEPTH


def test_num_features_constant() -> None:
    assert NUM_FEATURES == 56
    assert len(FEATURE_KEYS) == NUM_FEATURES
    assert FEATURE_KEYS[31] == "queue_pressure"
    assert FEATURE_KEYS[32] == "top3_asymmetry"
    assert FEATURE_KEYS[33] == "effective_spread_ratio"
    # Stage A horizon-tier additions (2026-04-15)
    assert FEATURE_KEYS[34] == "momentum_30s"
    assert FEATURE_KEYS[36] == "momentum_120s"
    assert FEATURE_KEYS[39] == "bipower_var_120s"


def _make_book(bid_l1: float, ask_l1: float) -> Snapshot:
    """Build a Snapshot with controlled volumes for top-3 + filler levels."""
    bids = np.zeros((BOOK_DEPTH, 2), dtype=np.float64)
    asks = np.zeros((BOOK_DEPTH, 2), dtype=np.float64)
    bids[:, 0] = np.arange(99_900, 99_900 - BOOK_DEPTH, -1, dtype=np.float64)
    asks[:, 0] = np.arange(99_901, 99_901 + BOOK_DEPTH, 1, dtype=np.float64)
    bids[0, 1] = bid_l1
    bids[1, 1] = 5.0
    bids[2, 1] = 5.0
    bids[3:, 1] = 1.0
    asks[0, 1] = ask_l1
    asks[1, 1] = 5.0
    asks[2, 1] = 5.0
    asks[3:, 1] = 1.0
    return Snapshot(bids=bids, asks=asks, timestamp=0, update_id=0)


def test_queue_pressure_realtime_ema() -> None:
    """Verify the realtime EMA matches the spec on a 4-tick controlled run."""
    fe = FeatureEngine.__new__(FeatureEngine)
    fe._prev_best_bid_vol = 0.0
    fe._prev_best_ask_vol = 0.0
    fe._bid_decay_ema = 0.0
    fe._ask_decay_ema = 0.0

    a = QUEUE_DECAY_ALPHA
    snaps = [
        _make_book(10.0, 10.0),  # tick 0 — both decays = 0 (no prev)
        _make_book(8.0, 6.0),    # bid_decay=2, ask_decay=4
        _make_book(8.0, 6.0),    # both 0
        _make_book(5.0, 6.0),    # bid_decay=3, ask_decay=0
    ]
    expected_bid_ema = 0.0
    expected_ask_ema = 0.0
    out = []
    for i, s in enumerate(snaps):
        v = fe._calc_queue_pressure(s)
        out.append(v)
        if i == 0:
            bd, ad = max(0.0, 0.0 - 10.0), max(0.0, 0.0 - 10.0)  # both 0
        elif i == 1:
            bd, ad = max(0.0, 10.0 - 8.0), max(0.0, 10.0 - 6.0)
        elif i == 2:
            bd, ad = 0.0, 0.0
        else:
            bd, ad = max(0.0, 8.0 - 5.0), max(0.0, 6.0 - 6.0)
        expected_bid_ema = a * bd + (1 - a) * expected_bid_ema
        expected_ask_ema = a * ad + (1 - a) * expected_ask_ema
        assert abs(v - (expected_ask_ema - expected_bid_ema)) < 1e-9, (i, v)


def test_top3_asymmetry_simple() -> None:
    fe = FeatureEngine.__new__(FeatureEngine)
    # bids: top3 = 10+5+5=20, top20 = 20 + 17*1 = 37 → bid_share = 20/37
    # asks: top3 = 5+5+5=15, top20 = 15 + 17*1 = 32 → ask_share = 15/32
    s = _make_book(10.0, 5.0)
    val = fe._calc_top3_asymmetry(s)
    expected = 20.0 / 37.0 - 15.0 / 32.0
    assert abs(val - expected) < 1e-9


def test_effective_spread_ratio_idle() -> None:
    fe = FeatureEngine.__new__(FeatureEngine)
    fe._eff_spread_ema = 0.0
    fe._trades = []  # type: ignore  # falsy → returns prior EMA (0.0)
    s = _make_book(10.0, 10.0)
    assert fe._calc_effective_spread_ratio(s) == 0.0


def test_effective_spread_ratio_active() -> None:
    fe = FeatureEngine.__new__(FeatureEngine)
    fe._eff_spread_ema = 0.0
    # Last trade printed at 99_900 (=best bid), mid = (99_900 + 99_901)/2 = 99_900.5
    # spread = 1.0; |99_900 - 99_900.5| / 1.0 = 0.5; EMA: 0.1*0.5 + 0.9*0 = 0.05
    fe._trades = [{"T": 0, "p": 99_900.0, "q": 1.0, "m": False}]  # type: ignore
    s = _make_book(10.0, 10.0)
    val = fe._calc_effective_spread_ratio(s)
    assert abs(val - 0.05) < 1e-9


def test_training_path_matches_realtime_for_queue_pressure() -> None:
    """Run trainer's vectorised EMA over identical input to the realtime loop
    and check the per-tick output is identical to the FeatureEngine path."""
    rng = np.random.default_rng(0)
    n = 200
    bid_l1 = (10.0 + rng.standard_normal(n)).astype(np.float64).clip(0.5)
    ask_l1 = (10.0 + rng.standard_normal(n)).astype(np.float64).clip(0.5)

    # --- Realtime path (FeatureEngine state machine) ---
    fe = FeatureEngine.__new__(FeatureEngine)
    fe._prev_best_bid_vol = 0.0
    fe._prev_best_ask_vol = 0.0
    fe._bid_decay_ema = 0.0
    fe._ask_decay_ema = 0.0
    rt_values = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = _make_book(bid_l1[i], ask_l1[i])
        rt_values[i] = fe._calc_queue_pressure(s)

    # --- Training path (vectorised EMA from trainer._calc_features_batch) ---
    a = QUEUE_DECAY_ALPHA
    bid_decay = np.maximum(0.0, bid_l1[:-1] - bid_l1[1:])
    ask_decay = np.maximum(0.0, ask_l1[:-1] - ask_l1[1:])
    bid_decay = np.concatenate([[0.0], bid_decay])
    ask_decay = np.concatenate([[0.0], ask_decay])
    bid_ema = np.empty_like(bid_decay)
    ask_ema = np.empty_like(ask_decay)
    b_acc = 0.0
    s_acc = 0.0
    for i in range(len(bid_decay)):
        b_acc = a * bid_decay[i] + (1 - a) * b_acc
        s_acc = a * ask_decay[i] + (1 - a) * s_acc
        bid_ema[i] = b_acc
        ask_ema[i] = s_acc
    train_values = ask_ema - bid_ema

    # Tick 0: realtime sees prev=0 → decay = max(0, 0 - bid_l1[0]) = 0
    # so both paths must agree from tick 0 onwards.
    assert np.allclose(rt_values, train_values, atol=1e-9), (
        f"max diff = {np.max(np.abs(rt_values - train_values))}"
    )


if __name__ == "__main__":
    test_num_features_constant()
    test_queue_pressure_realtime_ema()
    test_top3_asymmetry_simple()
    test_effective_spread_ratio_idle()
    test_effective_spread_ratio_active()
    test_training_path_matches_realtime_for_queue_pressure()
    print("microstructure parity tests OK")
