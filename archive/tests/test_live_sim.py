"""Unit tests for src/live_sim.py.

Each test crafts a deterministic mid-price path that drives `simulate_trade`
to hit a specific Tier-1 divergence from `handoff_current.md`. Naming mirrors
the `TradeOutcome.REASONS` tags so a failure points straight at the branch.

No asyncio, no model loading, no feature engine — pure numpy + dataclasses.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live_sim import (  # noqa: E402
    LiveSimConfig,
    SimDirection,
    TradeOutcome,
    label_from_outcomes,
    simulate_trade,
)
from src.model import UP, DOWN, FLAT  # noqa: E402


# ---- Test helpers -------------------------------------------------------


def _cfg(**overrides) -> LiveSimConfig:
    """LiveSimConfig with production-matching defaults (TP=0.20%, SL=0.10%,
    timeout=60s ≈ 600 ticks, commissions from src/config.Config)."""
    base = dict(
        tp_pct=0.20,
        sl_pct=0.10,
        timeout_ticks=600,
        commission_win_pct=0.04,
        commission_loss_pct=0.07,
    )
    base.update(overrides)
    return LiveSimConfig(**base)


def _flat_path(base: float, length: int = 1300) -> np.ndarray:
    return np.full(length, base, dtype=np.float64)


def _step_path(base: float, pct_moves: list[tuple[int, float]], length: int = 1300) -> np.ndarray:
    """Build a mid-price path from (tick_index, percent_move_from_base) pairs."""
    path = np.full(length, base, dtype=np.float64)
    for tick, pct in pct_moves:
        path[tick:] = base * (1 + pct / 100.0)
    return path


# ---- Exit-mode coverage ------------------------------------------------


def test_tp_hit_long() -> None:
    """LONG: mid jumps +0.25% > TP=0.20% at tick 5 — tp_hit (full TP, maker)."""
    base = 100_000.0
    path = _step_path(base, [(5, 0.25)])
    out = simulate_trade(SimDirection.LONG, base, path, _cfg())
    assert out.exit_reason == "tp_hit", out
    # Gross = +0.20% (TP price), net = 0.20 - 0.04 = 0.16%.
    assert abs(out.gross_pnl_pct - 0.20) < 1e-9
    assert abs(out.net_pnl_pct - 0.16) < 1e-9
    assert not out.partial_filled
    assert out.trailing_step_reached == 0


def test_tp_hit_short() -> None:
    """SHORT: mid drops -0.25% at tick 3 — tp_hit."""
    base = 100_000.0
    path = _step_path(base, [(3, -0.25)])
    out = simulate_trade(SimDirection.SHORT, base, path, _cfg())
    assert out.exit_reason == "tp_hit", out
    assert abs(out.gross_pnl_pct - 0.20) < 1e-9
    assert abs(out.net_pnl_pct - 0.16) < 1e-9


def test_sl_hit_long() -> None:
    """LONG: mid drops -0.15% > SL=0.10% at tick 2 — sl_hit (taker)."""
    base = 100_000.0
    path = _step_path(base, [(2, -0.15)])
    out = simulate_trade(SimDirection.LONG, base, path, _cfg())
    assert out.exit_reason == "sl_hit", out
    # Exit at the SL price (-0.10%), net = -0.10 - 0.07 = -0.17%.
    assert abs(out.gross_pnl_pct + 0.10) < 1e-9
    assert abs(out.net_pnl_pct + 0.17) < 1e-9


def test_sl_hit_short() -> None:
    base = 100_000.0
    path = _step_path(base, [(4, 0.15)])
    out = simulate_trade(SimDirection.SHORT, base, path, _cfg())
    assert out.exit_reason == "sl_hit", out
    assert abs(out.gross_pnl_pct + 0.10) < 1e-9


def test_partial_plus_tp_long() -> None:
    """LONG: mid rises to 50% of TP then onward to full TP — partial + full TP.

    Expected: partial fires at 0.10% (= 50% of 0.20% TP), remainder closes at
    the TP limit (0.20%). Gross = 0.5*0.10 + 0.5*0.20 = 0.15. Net = 0.15 -
    commission_win_pct (both legs maker) = 0.11%.
    """
    base = 100_000.0
    # Tick 10: price to +0.11% (safely above the 50% = 0.10% threshold —
    # avoids float-rounding flakiness where base*1.001 is actually slightly
    # below 100_100). Tick 50: price to +0.25% (guarantees TP limit fills).
    path = _step_path(base, [(10, 0.11), (50, 0.25)])
    out = simulate_trade(SimDirection.LONG, base, path, _cfg())
    assert out.exit_reason == "partial_plus_tp", out
    assert out.partial_filled
    # Gross = 0.5 * partial_leg + 0.5 * tp_leg
    #       = 0.5 * 0.11 + 0.5 * 0.20 = 0.155. Net = 0.155 - 0.04 = 0.115%.
    assert abs(out.gross_pnl_pct - 0.155) < 1e-9
    assert abs(out.net_pnl_pct - 0.115) < 1e-9


def test_partial_plus_trailing_sl_long() -> None:
    """LONG: hits 50% progress (partial + move SL to +0.08% min), then drops.

    With TP=0.20%, 30% × TP_dist = 0.06%, which is below the 0.08% floor, so
    the trailing SL lands at entry + 0.08%. Then the price collapses back
    below that level → trailing_sl_1 exit on the remaining half.
    """
    base = 100_000.0
    # Tick 10: +0.11% (trips 50% threshold, partial fills at 0.11%).
    # Tick 40: price crashes to -0.05% (crosses the new +0.08% SL).
    path = _step_path(base, [(10, 0.11), (40, -0.05)])
    out = simulate_trade(SimDirection.LONG, base, path, _cfg())
    assert out.exit_reason == "partial_plus_trailing_sl_1", out
    assert out.partial_filled
    assert out.trailing_step_reached == 1
    # Gross = 0.5*0.11 + 0.5*0.08 = 0.095. Net = 0.095 - 0.07
    # (taker because remainder exited via stop-market) = 0.025%.
    assert abs(out.gross_pnl_pct - 0.095) < 1e-9
    assert abs(out.net_pnl_pct - 0.025) < 1e-9


def test_trailing_sl_2_short() -> None:
    """SHORT: reaches 75% of TP then reverses past the 50%×TP trailing level.

    Path must skip the 50% partial window cleanly — a single-tick jump to
    75% progress hits step 2 directly (see the ordering guard in the
    simulator: step 2 check runs before step 1).
    """
    base = 100_000.0
    # Tick 5: SHORT progress = 80% → step 2 fires (partial skipped). Trailing
    # SL at entry - 50% × TP_dist = -0.10% (i.e. 0.10% move in price).
    # Tick 30: mid rebounds above the trailing level.
    path = _step_path(base, [(5, -0.16), (30, -0.08)])
    out = simulate_trade(SimDirection.SHORT, base, path, _cfg())
    assert out.exit_reason == "trailing_sl_2", out
    assert out.trailing_step_reached == 2
    assert not out.partial_filled


def test_timeout_limit_long() -> None:
    """LONG: flat inside the window, small recovery in the limit window → timeout_limit.

    Path shape: entry at base, drift to -0.03% by timeout, then recover to
    +0.01% within the next 20 ticks. Limit GTX at timeout mid fills when
    price comes back up.
    """
    base = 100_000.0
    timeout_ticks = 50  # tiny timeout so the test runs fast
    path = _flat_path(base, length=timeout_ticks + 20 + 5)
    path[5:timeout_ticks] = base * (1 - 0.0003)   # -0.03%, inside barriers
    # Limit window: first tick goes below, second tick returns to the limit
    # price (base*0.9997) → fills there.
    path[timeout_ticks] = base * (1 - 0.0004)
    path[timeout_ticks + 1] = base * (1 - 0.0003)  # fill target
    path[timeout_ticks + 2:] = base * (1 - 0.0003)
    cfg = _cfg(timeout_ticks=timeout_ticks)
    out = simulate_trade(SimDirection.LONG, base, path, cfg)
    assert out.exit_reason == "timeout_limit", out
    # Gross = -0.03%. Net = -0.03 - commission_win (maker+maker) = -0.07%.
    assert abs(out.gross_pnl_pct + 0.03) < 1e-9
    assert abs(out.net_pnl_pct + 0.07) < 1e-9


def test_timeout_market_long() -> None:
    """LONG: flat drift -0.05% at timeout, then drifts further DOWN in the
    limit-close window — limit never fills, fallback market close triggers.

    (A constant -0.05% path would match the limit price on equality and we
    would get `timeout_limit` instead; the downward drift forces the taker
    branch.)
    """
    base = 100_000.0
    timeout_ticks = 50
    length = timeout_ticks + 20
    path = np.full(length, base, dtype=np.float64)
    path[3:timeout_ticks] = base * (1 - 0.0005)         # -0.05% at timeout
    path[timeout_ticks:] = base * (1 - 0.0008)          # -0.08% in limit window
    cfg = _cfg(timeout_ticks=timeout_ticks)
    out = simulate_trade(SimDirection.LONG, base, path, cfg)
    assert out.exit_reason == "timeout_market", out
    # Fallback market close fires at the last observed tick of the limit
    # window → mid = base * (1 - 0.0008), gross = -0.08%. Taker commission.
    assert abs(out.gross_pnl_pct + 0.08) < 1e-9
    assert abs(out.net_pnl_pct + 0.15) < 1e-9


def test_fast_fill_adverse_immediate_close() -> None:
    """fill_latency_ms < 50 → instant market close at the very first tick."""
    base = 100_000.0
    # Give the path any profitable move — it should be ignored, we close at
    # mid_path[0] before the TP or SL logic runs.
    path = _step_path(base, [(1, 0.5)])
    out = simulate_trade(SimDirection.LONG, base, path, _cfg(), fill_latency_ms=10.0)
    assert out.exit_reason == "fast_fill_adverse", out
    # Exit at mid_path[0] = base → gross = 0, net = -commission_loss_pct.
    assert abs(out.gross_pnl_pct) < 1e-9
    assert abs(out.net_pnl_pct + 0.07) < 1e-9


def test_fast_fill_sl_tightening_triggers_at_half_distance() -> None:
    """50 ≤ fill_latency_ms < 100 → SL tightened to 0.5 × sl_pct = 0.05%.

    The normal 0.10% SL would let a -0.08% dip pass through — but with fast-
    fill tightening to 0.05%, the -0.08% dip breaches the tight SL and the
    trade exits as `sl_hit` (reason is still the generic tag — the commission
    is the same and the test only cares that the tighter barrier fired).
    """
    base = 100_000.0
    path = _step_path(base, [(3, -0.08)])
    out = simulate_trade(
        SimDirection.LONG, base, path, _cfg(), fill_latency_ms=75.0,
    )
    assert out.exit_reason == "sl_hit", out
    # Exit at the TIGHT sl price: entry - 0.05% = -0.05%. Net = -0.05 - 0.07
    # = -0.12%.
    assert abs(out.gross_pnl_pct + 0.05) < 1e-9
    assert abs(out.net_pnl_pct + 0.12) < 1e-9


# ---- Degenerate / graceful-clamp paths ---------------------------------


def test_no_forward_data_returns_zero_outcome() -> None:
    out = simulate_trade(SimDirection.LONG, 100_000.0, np.array([]), _cfg())
    assert out.exit_reason == "no_forward_data"
    assert out.net_pnl_pct == 0.0


def test_short_path_clamps_timeout_gracefully() -> None:
    """A path shorter than `timeout_ticks + limit_window` must not raise.

    Simulator clamps timeout to fit — the remaining behaviour is a best-effort
    timeout_market at the last available tick.
    """
    base = 100_000.0
    path = np.full(25, base, dtype=np.float64)
    out = simulate_trade(SimDirection.LONG, base, path, _cfg(timeout_ticks=100))
    # 25 - 20 (limit window) = 5 ticks for monitoring. No barriers hit, so
    # we fall into the timeout branch with a 5-tick window. Either
    # timeout_limit (mid equal) or timeout_market — on a flat path the
    # limit-window fill check sees mid ≥ timeout_mid and we get timeout_limit.
    assert out.exit_reason in ("timeout_limit", "timeout_market")


# ---- label_from_outcomes helper -----------------------------------------


def test_label_helper_picks_profitable_direction() -> None:
    cfg = _cfg()
    long_out = simulate_trade(
        SimDirection.LONG, 100_000.0,
        _step_path(100_000.0, [(5, 0.25)]), cfg,
    )
    short_out = simulate_trade(
        SimDirection.SHORT, 100_000.0,
        _step_path(100_000.0, [(5, 0.25)]), cfg,
    )
    label, pnl = label_from_outcomes(long_out, short_out)
    assert label == UP
    assert pnl > 0


def test_label_helper_returns_flat_when_both_lose() -> None:
    cfg = _cfg()
    base = 100_000.0
    # Tiny wiggle inside ±0.05% — neither direction profits, both eat fees.
    path = np.full(1300, base * 1.00005, dtype=np.float64)
    long_out = simulate_trade(SimDirection.LONG, base, path, cfg)
    short_out = simulate_trade(SimDirection.SHORT, base, path, cfg)
    label, pnl = label_from_outcomes(long_out, short_out)
    assert label == FLAT
    assert pnl < 0


# ---- Exit reason catalog is exhaustive ---------------------------------


def test_all_exit_reasons_are_catalogued() -> None:
    """Every `exit_reason` string produced above must appear in the catalog."""
    catalog = set(TradeOutcome.REASONS)
    cases = {
        "tp_hit", "sl_hit", "trailing_sl_2",
        "partial_plus_tp", "partial_plus_trailing_sl_1",
        "timeout_limit", "timeout_market",
        "fast_fill_adverse",
    }
    assert cases.issubset(catalog), cases - catalog


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok: {name}")
    print("live_sim tests OK")
