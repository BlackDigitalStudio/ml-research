"""Unified trade simulator — single entry point for grid_live and the
(future) live-inference module.

Rationale (2026-04-16 pivot): the project had two separate simulators —
`src/live_sim.py::simulate_trade` (Python, per-trade, used at label time
via `trainer.build_samples`) and `rust_ingest::sim_labels` wrapped by
`rust_bridge::simulate_labels` (Rust, batched, used by grid scripts).
They implement the same trade lifecycle but with subtly different
defaults (partial/trailing on/off, fill-latency) and accept different
shapes of inputs. This split caused train↔grid mismatch debugging.

`simulate_trade()` below is the single high-level API. It takes a
state vector + a policy callable `pi(state) -> TradeParams` and returns
a `TradeOutcome`. Internally it dispatches to the Rust simulator in
batch mode. Callers that want per-sample policies (IQL head) pass
arrays, callers that want fixed parameters (grid_live) pass constants.

This module is additive; existing `live_sim.simulate_trade` and
`rust_bridge.simulate_labels` stay in place for back-compat until the
last caller is migrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# Trade outcome taxonomy — mirrors live_sim.TradeOutcome.REASONS exactly.
EXIT_REASONS = (
    "tp_hit",
    "sl_hit",
    "trailing_sl_1",
    "trailing_sl_2",
    "partial_plus_tp",
    "partial_plus_trailing_sl_1",
    "partial_plus_trailing_sl_2",
    "timeout_limit",
    "timeout_market",
)


@dataclass
class TradeParams:
    """Parameters for one sample's trade. A policy returns these per-sample.

    All percent fields are in 'percent of entry price' (0.30 == 0.30 %).
    timeout_ticks is in depth-tick units (100 ms each at our cadence).
    direction: 0 = UP (LONG), 1 = DOWN (SHORT), 2 = FLAT (no trade).
    """
    tp_pct: float
    sl_pct: float
    timeout_ticks: int
    kelly_fraction: float = 0.25
    direction: int = 0
    spread_bps: float = 0.0
    partial_enabled: bool = True
    trailing_enabled: bool = True
    fill_latency_ms: float = 150.0


@dataclass
class TradeBatch:
    """Batched input for `simulate_trade`. All arrays must be same length N."""
    entry_long: np.ndarray          # (N,) float64 — entry price at tick for LONG
    entry_short: np.ndarray         # (N,) float64 — entry price at tick for SHORT
    mid_paths: np.ndarray           # (N, T) float64 — forward mid-price path
    direction: np.ndarray           # (N,) int8 — 0 UP / 1 DOWN / 2 FLAT
    tp_pct: np.ndarray              # (N,) float64
    sl_pct: np.ndarray              # (N,) float64
    timeout_ticks: np.ndarray       # (N,) int64
    kelly_fraction: np.ndarray | None = None   # (N,) float32, None = 1.0
    spread_bps: np.ndarray | None = None        # (N,) float32, None = 0.0
    commission_win_pct: float = 0.04
    commission_loss_pct: float = 0.07
    partial_enabled: bool = True
    trailing_enabled: bool = True
    fill_latency_ms: float = 150.0


@dataclass
class SimResult:
    """Per-trade outcome, vectorised across batch."""
    pnl_pct: np.ndarray             # (N,) realised net PnL (%); 0 on FLAT
    win: np.ndarray                 # (N,) bool — pnl_pct > 0
    reason: np.ndarray              # (N,) int8 — index into EXIT_REASONS; -1 on FLAT
    sized_pnl_pct: np.ndarray       # (N,) pnl × kelly_fraction — portfolio-level
    n_trades: int                   # non-FLAT count
    sum_pnl_pct: float              # sum of sized_pnl_pct


def simulate_trade(batch: TradeBatch) -> SimResult:
    """Simulate a batch of trades using the Rust backend.

    Directions handled per-sample:
      UP    → use entry_long, compare against pnl_long from simulate_labels
      DOWN  → use entry_short, compare against pnl_short from simulate_labels
      FLAT  → pnl = 0, reason = -1

    The underlying Rust `simulate_labels` returns pnl_long + pnl_short
    for ALL samples; we select the direction-appropriate value per
    sample.
    """
    from src import rust_bridge

    # Normalise optional arrays
    N = len(batch.direction)
    kelly = batch.kelly_fraction
    if kelly is None:
        kelly = np.ones(N, dtype=np.float32)
    spread = batch.spread_bps
    if spread is None:
        spread = np.zeros(N, dtype=np.float32)

    out = rust_bridge.simulate_labels(
        batch.entry_long, batch.entry_short, batch.mid_paths,
        batch.tp_pct, batch.sl_pct, batch.timeout_ticks,
        commission_win_pct=batch.commission_win_pct,
        commission_loss_pct=batch.commission_loss_pct,
        partial_enabled=batch.partial_enabled,
        trailing_enabled=batch.trailing_enabled,
        fill_latency_ms=batch.fill_latency_ms,
    )

    pnl_long = out["pnl_long"].astype(np.float64)
    pnl_short = out["pnl_short"].astype(np.float64)
    reason_long = out["reason_long"].astype(np.int8)
    reason_short = out["reason_short"].astype(np.int8)

    direction = batch.direction.astype(np.int8)
    raw_pnl = np.where(direction == 0, pnl_long,
                np.where(direction == 1, pnl_short, 0.0))
    reason = np.where(direction == 0, reason_long,
                np.where(direction == 1, reason_short, -1)).astype(np.int8)

    # Subtract spread cost per trade (in percent — spread_bps/100).
    pnl_after_spread = raw_pnl - (spread.astype(np.float64) / 100.0)
    # FLAT trades always have zero pnl.
    pnl_after_spread = np.where(direction == 2, 0.0, pnl_after_spread)

    sized = pnl_after_spread * kelly.astype(np.float64)
    non_flat = direction != 2
    return SimResult(
        pnl_pct=pnl_after_spread.astype(np.float32),
        win=(pnl_after_spread > 0) & non_flat,
        reason=reason,
        sized_pnl_pct=sized.astype(np.float32),
        n_trades=int(non_flat.sum()),
        sum_pnl_pct=float(sized.sum()),
    )


def simulate_with_policy(
    state_batch: np.ndarray,               # (N, state_dim) — policy input
    *,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    mid_paths: np.ndarray,
    policy: Callable[[np.ndarray], TradeParams | Sequence[TradeParams]],
    commission_win_pct: float = 0.04,
    commission_loss_pct: float = 0.07,
) -> SimResult:
    """Policy-driven simulation.

    `policy(state)` is called once with the full batch state array and
    returns either a single `TradeParams` (broadcast across the batch)
    or a list of N `TradeParams` (one per sample). Flexible enough for
    fixed-config grid scans, per-sample IQL-policy dispatch, and
    live-inference routing.
    """
    N = len(state_batch)
    out = policy(state_batch)
    if isinstance(out, TradeParams):
        tp = np.full(N, out.tp_pct, dtype=np.float64)
        sl = np.full(N, out.sl_pct, dtype=np.float64)
        to = np.full(N, out.timeout_ticks, dtype=np.int64)
        direction = np.full(N, out.direction, dtype=np.int8)
        kelly = np.full(N, out.kelly_fraction, dtype=np.float32)
        spread = np.full(N, out.spread_bps, dtype=np.float32)
        partial = out.partial_enabled
        trailing = out.trailing_enabled
        fill_latency = out.fill_latency_ms
    else:
        assert len(out) == N, f"policy returned {len(out)} params for batch of {N}"
        tp = np.array([p.tp_pct for p in out], dtype=np.float64)
        sl = np.array([p.sl_pct for p in out], dtype=np.float64)
        to = np.array([p.timeout_ticks for p in out], dtype=np.int64)
        direction = np.array([p.direction for p in out], dtype=np.int8)
        kelly = np.array([p.kelly_fraction for p in out], dtype=np.float32)
        spread = np.array([p.spread_bps for p in out], dtype=np.float32)
        partial = out[0].partial_enabled
        trailing = out[0].trailing_enabled
        fill_latency = out[0].fill_latency_ms

    batch = TradeBatch(
        entry_long=entry_long, entry_short=entry_short, mid_paths=mid_paths,
        direction=direction, tp_pct=tp, sl_pct=sl, timeout_ticks=to,
        kelly_fraction=kelly, spread_bps=spread,
        commission_win_pct=commission_win_pct,
        commission_loss_pct=commission_loss_pct,
        partial_enabled=partial, trailing_enabled=trailing,
        fill_latency_ms=fill_latency,
    )
    return simulate_trade(batch)
