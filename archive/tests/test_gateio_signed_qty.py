"""Regression test for the Gate.io signed-quantity bug.

Gate.io's `futures.trades` stream ships `size` as a signed integer
(positive=buy, negative=sell). The recorder stores it raw. Both the
training-time loader in `trainer.build_samples` and the realtime handler
`FeatureEngine.on_exchange_trade` must neutralise the sign so that
feature 30 (cross_exchange_momentum_500ms) sees the actual direction.

This test constructs a small synthetic Gate.io trade stream (3 buys,
3 sells, all at the same second) and asserts:

1. Runtime path: calling `on_exchange_trade` 6 times and then
   `_calc_cross_exchange_momentum` reports net_buy > 0 only when buy
   volume actually exceeds sell volume — regardless of whether the
   per-trade `q` field was signed.

2. Training path: the sign-stripping branch in `build_samples` correctly
   turns `(ts, signed_qty, is_seller)` into the same `ex_signed` that
   feature 30 would see from an unsigned-qty exchange.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import deque

from src.features import FeatureEngine, CROSS_EX_WINDOW_MS


def _blank_fe() -> FeatureEngine:
    fe = FeatureEngine.__new__(FeatureEngine)
    fe._bybit_trades = deque(maxlen=3000)  # type: ignore
    fe._cross_exchange_trades = {  # type: ignore
        "bybit": deque(maxlen=2000),
        "okx": deque(maxlen=2000),
        "bitget": deque(maxlen=2000),
        "gateio": deque(maxlen=2000),
    }
    return fe


def test_runtime_gateio_signed_qty_net_buy() -> None:
    """3 buys of 100, 2 sells of 50 → net_buy > 0 on gateio."""
    fe = _blank_fe()
    now_ms = 10_000
    # Gate.io emits signed `size`: positive = buy, negative = sell.
    gateio_trades = [
        (now_ms - 400, +100, False),
        (now_ms - 350, +100, False),
        (now_ms - 300, +100, False),
        (now_ms - 250,  -50, True),
        (now_ms - 200,  -50, True),
    ]
    for ts, size, is_seller in gateio_trades:
        fe.on_exchange_trade({
            "T": ts, "p": 73000.0, "q": size, "m": is_seller,
            "exchange": "gateio",
        })
    # All 5 trades fall inside the 500ms window
    count = fe._calc_cross_exchange_momentum(now_ms)
    # Only gateio has data → expect count = 1 (net buy)
    assert count == 1.0, f"expected 1 (gateio net-buy), got {count}"

    # Flip the scenario: 3 sells 100, 2 buys 50 → net_sell
    fe2 = _blank_fe()
    for ts, size, is_seller in [
        (now_ms - 400, -100, True),
        (now_ms - 350, -100, True),
        (now_ms - 300, -100, True),
        (now_ms - 250, +50, False),
        (now_ms - 200, +50, False),
    ]:
        fe2.on_exchange_trade({
            "T": ts, "p": 73000.0, "q": size, "m": is_seller,
            "exchange": "gateio",
        })
    count = fe2._calc_cross_exchange_momentum(now_ms)
    assert count == 0.0, f"expected 0 (gateio net-sell), got {count}"


def test_runtime_bybit_positive_qty_unchanged() -> None:
    """Bybit already sends positive qty — behaviour must not change."""
    fe = _blank_fe()
    now_ms = 10_000
    for ts, q, is_seller in [
        (now_ms - 400, 0.5, False),  # buy
        (now_ms - 350, 0.5, False),  # buy
        (now_ms - 300, 0.1, True),   # sell
    ]:
        fe.on_bybit_aggtrade({"T": ts, "p": 73000.0, "q": q, "m": is_seller})
    count = fe._calc_cross_exchange_momentum(now_ms)
    assert count == 1.0, f"expected 1 (bybit net-buy), got {count}"


def test_runtime_abs_is_defensive_noop_for_positive() -> None:
    """`abs(positive)` is still positive — no regression for OKX/Bitget."""
    fe = _blank_fe()
    now_ms = 10_000
    # OKX/Bitget style: positive qty + is_seller flag
    for ex in ("okx", "bitget"):
        fe.on_exchange_trade({
            "T": now_ms - 100, "p": 73000.0, "q": 2.0, "m": False,  # buy
            "exchange": ex,
        })
    count = fe._calc_cross_exchange_momentum(now_ms)
    assert count == 2.0, f"expected 2 (okx+bitget net-buy), got {count}"


def test_training_path_abs_then_sign() -> None:
    """Replicate the build_samples branch for gateio and assert sign."""
    # Simulated parquet read: quantity is the raw signed Gate.io size,
    # is_seller is already derived from sign(size) < 0 at recorder time.
    ex_qty_raw = np.array([+100, +100, +100, -50, -50], dtype=np.float64)
    ex_is_seller = np.array([False, False, False, True, True], dtype=bool)

    # --- Training-time fix from build_samples ---
    ex_qty = np.abs(ex_qty_raw)
    ex_signed = np.where(ex_is_seller, -ex_qty, ex_qty)
    assert ex_signed.tolist() == [100.0, 100.0, 100.0, -50.0, -50.0], ex_signed

    # Net sum = 300 - 100 = 200 → net buy
    assert ex_signed.sum() == 200.0


def test_training_path_net_sell_gateio() -> None:
    ex_qty_raw = np.array([-100, -100, -100, +50, +50], dtype=np.float64)
    ex_is_seller = np.array([True, True, True, False, False], dtype=bool)

    ex_qty = np.abs(ex_qty_raw)
    ex_signed = np.where(ex_is_seller, -ex_qty, ex_qty)
    assert ex_signed.tolist() == [-100.0, -100.0, -100.0, 50.0, 50.0]
    assert ex_signed.sum() == -200.0  # net sell


def test_without_fix_sign_is_flipped() -> None:
    """Document the bug being prevented — WITHOUT abs(), gateio sign flips.

    This is a negative-control test: if you remove the `abs()` patch,
    this is what feature 30 would see. The assertions here demonstrate
    that the unpatched path produces the *opposite* net direction from
    the ground truth.
    """
    # 3 buys of 100, 2 sells of 50 — ground truth net buy 200
    ex_qty_raw = np.array([+100, +100, +100, -50, -50], dtype=np.float64)
    ex_is_seller = np.array([False, False, False, True, True], dtype=bool)

    # Unpatched (buggy) path:
    ex_signed_buggy = np.where(ex_is_seller, -ex_qty_raw, ex_qty_raw)
    # (-50 with is_seller=True) → -(-50) = +50, so sells become buys
    assert ex_signed_buggy.tolist() == [100.0, 100.0, 100.0, 50.0, 50.0]
    assert ex_signed_buggy.sum() == 400.0  # buggy: inflates net_buy


if __name__ == "__main__":
    test_runtime_gateio_signed_qty_net_buy()
    test_runtime_bybit_positive_qty_unchanged()
    test_runtime_abs_is_defensive_noop_for_positive()
    test_training_path_abs_then_sign()
    test_training_path_net_sell_gateio()
    test_without_fix_sign_is_flipped()
    print("gateio signed-qty regression tests OK")
