"""Deterministic forward-simulation of the live executor's trade lifecycle.

This module is the **single source of truth** for how a live trade would
realize PnL. It mirrors every Tier-1 divergence between training labels and
live execution that was catalogued in handoff_current.md: adaptive TP/SL,
dynamic position timeout, partial TP at 50% progress, stepped trailing SL,
fast-fill adverse-selection tightening, asymmetric commissions, and
timeout limit→market close.

Three callers share this module:
    * `trainer.build_samples` — labels each sample by simulating LONG and SHORT
      against the forward mid path.
    * `scripts/backtest.py:run_backtest` — walk-forward evaluation replays
      predictions through the same simulator.
    * `executor.py` — runs the simulator in parallel on every live trade and
      logs divergence between simulated and realized PnL.

Because all three consume the same implementation, any divergence is caught
immediately by the executor's diagnostic logger.

Pure numpy/dataclass code — no asyncio, no I/O, no global state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np


class SimDirection(IntEnum):
    LONG = 1
    SHORT = -1


# ---- Config + outcome dataclasses ---------------------------------------


@dataclass(frozen=True)
class LiveSimConfig:
    """Per-trade configuration for `simulate_trade`.

    All percent fields are in "percent of entry price" units (0.20 == 0.20%).
    Commission fields are **round-trip** totals matching executor semantics
    — `commission_win_pct` covers entry-maker + exit-maker, and
    `commission_loss_pct` covers entry-maker + exit-taker. We don't split
    into per-leg rates because the executor itself doesn't (`_finalize_trade`
    lines 503-508 apply one aggregate rate based on the exit reason), so
    matching the live bookkeeping is the simplest way to keep the simulator
    honest.
    """

    tp_pct: float                     # adapted per-sample from vol_ratio
    sl_pct: float                     # adapted per-sample from vol_ratio
    timeout_ticks: int                # dynamic_timeout × 10 (sec → 100ms ticks)
    commission_win_pct: float         # round-trip maker+maker, e.g. 0.04
    commission_loss_pct: float        # round-trip maker+taker, e.g. 0.07

    # Tier 1 divergence knobs — defaults mirror src/executor.py exactly.
    partial_tp_progress: float = 0.50          # Item 15
    trailing_step1_progress: float = 0.50      # Item 14 step 1
    trailing_step1_sl_floor_pct: float = 0.08  # max(+0.08%, 30% × TP_dist)
    trailing_step1_sl_ratio: float = 0.30
    trailing_step2_progress: float = 0.75      # Item 14 step 2
    trailing_step2_sl_ratio: float = 0.50
    fast_fill_adverse_ms: float = 50.0         # < adverse → instant close
    fast_fill_threshold_ms: float = 100.0      # [50, 100) → tighten SL
    fast_fill_sl_multiplier: float = 0.5       # SL → 0.5 × sl_pct
    timeout_limit_ticks: int = 20              # 2 sec limit-close window

    # Grid-search toggles — defaults preserve original behavior so existing
    # tests and callers see no change. Setting either to False disables the
    # corresponding feature in the monitoring loop.
    partial_enabled: bool = True               # fire partial TP at partial_tp_progress
    trailing_enabled: bool = True              # move SL on trailing progress thresholds


@dataclass(frozen=True)
class TradeOutcome:
    """Result of forward-simulating a single trade.

    `net_pnl_pct` is the field the training loss ultimately optimizes against.
    Everything else is diagnostic / feature-engineering fodder.
    """

    net_pnl_pct: float        # after commissions, as % of entry price
    gross_pnl_pct: float      # before commissions
    exit_reason: str
    duration_ticks: int
    partial_filled: bool
    trailing_step_reached: int  # 0 | 1 | 2

    # Valid exit_reason tags — used by tests and executor diagnostics to
    # catch typos.
    REASONS = (
        "tp_hit",
        "sl_hit",
        "trailing_sl_1",
        "trailing_sl_2",
        "partial_plus_tp",
        "partial_plus_trailing_sl_1",
        "partial_plus_trailing_sl_2",
        "timeout_limit",
        "timeout_market",
        "fast_fill_adverse",
        "fast_fill_sl",
        "no_forward_data",
    )


# ---- Core simulator -----------------------------------------------------


def _make_outcome(
    direction: SimDirection,
    entry_px: float,
    exit_px_remainder: float,
    partial_px: float | None,
    exit_reason: str,
    duration_ticks: int,
    partial_filled: bool,
    trailing_step_reached: int,
    cfg: LiveSimConfig,
) -> TradeOutcome:
    """Fold a simulation into a TradeOutcome with commissions applied.

    Gross PnL is the sum of the two legs (partial half + remainder half) if
    partial fired, else the whole position against the single exit. We apply
    *one* aggregate commission rate based on the final exit side — maker for
    TP/timeout-limit, taker for any SL flavour (including trailing) or the
    market fallback. This matches `executor.py:_finalize_trade` where the
    switch is a simple `if reason == "stop_loss"` branch. The partial half's
    maker-maker fee saving is already baked into the `commission_win_pct`
    constant (= 0.04% round-trip, both sides maker), so the conservative
    treatment for mixed partial+taker exits is to use `commission_loss_pct`.
    """
    sign = 1.0 if direction == SimDirection.LONG else -1.0

    if partial_filled and partial_px is not None:
        # Leg A — partial half closed at partial_px (50% of notional).
        gross_a_pct = sign * (partial_px - entry_px) / entry_px * 100.0
        # Leg B — remaining half closed at exit_px_remainder.
        gross_b_pct = sign * (exit_px_remainder - entry_px) / entry_px * 100.0
        gross_pct = 0.5 * gross_a_pct + 0.5 * gross_b_pct
    else:
        gross_pct = sign * (exit_px_remainder - entry_px) / entry_px * 100.0

    # Final-exit commission branch — exactly matches executor _finalize_trade.
    # Any SL flavour or the MARKET fallback pays taker on exit; everything
    # else is fully maker.
    taker_exits = {
        "sl_hit",
        "trailing_sl_1",
        "trailing_sl_2",
        "partial_plus_trailing_sl_1",
        "partial_plus_trailing_sl_2",
        "fast_fill_sl",
        "fast_fill_adverse",
        "timeout_market",
    }
    if exit_reason in taker_exits:
        commission_pct = cfg.commission_loss_pct
    else:
        commission_pct = cfg.commission_win_pct

    net_pct = gross_pct - commission_pct

    return TradeOutcome(
        net_pnl_pct=float(net_pct),
        gross_pnl_pct=float(gross_pct),
        exit_reason=exit_reason,
        duration_ticks=int(duration_ticks),
        partial_filled=bool(partial_filled),
        trailing_step_reached=int(trailing_step_reached),
    )


def simulate_trade(
    direction: SimDirection,
    entry_px: float,
    mid_path: np.ndarray | Sequence[float],
    config: LiveSimConfig,
    fill_latency_ms: float = 150.0,
) -> TradeOutcome:
    """Forward-simulate a single live trade deterministically.

    Parameters
    ----------
    direction
        SimDirection.LONG (+1) or SimDirection.SHORT (-1). LONGs enter at best
        bid and profit on upward moves; SHORTs enter at best ask and profit
        on downward moves. The caller is responsible for passing the correct
        `entry_px` (best_bid for LONG, best_ask for SHORT) — this function
        treats `entry_px` as ground truth.

    entry_px
        The filled entry price. NOT the mid — the caller MUST pass the
        best-bid/best-ask side (see `signal.py:_calc_size` + executor for
        how the live bot picks this). Training-label construction stores
        `best_bid[i]` and `best_ask[i]` in the sample cache for this reason.

    mid_path
        1-D array of forward mid prices starting at the tick AFTER the fill.
        Length determines how far we can look ahead — we clamp timeout and
        the limit-close window to fit the array, so short paths degrade
        gracefully. Callers should pass at least `timeout_ticks + timeout_
        limit_ticks` ticks; `build_samples` uses SIM_HORIZON = 1300 to give
        the maximum dynamic timeout room to breathe.

    config
        See `LiveSimConfig`. Adaptive TP/SL and dynamic timeout must already
        be computed by the caller (via `filters.adaptive_tp_sl` etc.) — this
        function does not re-derive them from `volatility_ratio`, so it can
        stay a pure numpy kernel with no dependency on the feature engine.

    fill_latency_ms
        Simulated time from order submission to fill. At label-build time we
        pass 150.0 (= no fast-fill effects) because fill latency is an
        execution artefact and irrelevant to the forward price path. At live
        time the executor passes the actual measured latency so the
        simulator's diagnostic PnL matches what the bot did.
    """
    # Normalise inputs and guard against degenerate inputs. A zero or
    # negative entry price turns the percent math into NaN — defensive check
    # here avoids surprising downstream users.
    if entry_px <= 0:
        return TradeOutcome(0.0, 0.0, "no_forward_data", 0, False, 0)
    path = np.asarray(mid_path, dtype=np.float64)
    if path.size == 0:
        return TradeOutcome(0.0, 0.0, "no_forward_data", 0, False, 0)

    # ---- Fast-fill adverse-selection branches --------------------------
    # executor._on_filled lines 332-337: <50ms → market close immediately,
    # 50-100ms → keep position but tighten SL to 0.5×. We replicate both.
    # Note that label-building passes fill_latency_ms=150 → neither branch
    # fires; this path is real only when the executor drives the simulator
    # online.
    sl_pct_eff = config.sl_pct
    if fill_latency_ms < config.fast_fill_adverse_ms:
        # Extreme adverse selection: executor does a MARKET close on the
        # very next tick. Taker on exit, so use the loss commission.
        exit_px = float(path[0])
        return _make_outcome(
            direction, entry_px, exit_px, None,
            "fast_fill_adverse", 1, False, 0, config,
        )
    if fill_latency_ms < config.fast_fill_threshold_ms:
        sl_pct_eff = config.sl_pct * config.fast_fill_sl_multiplier

    # ---- Initial barriers ----------------------------------------------
    sign = 1.0 if direction == SimDirection.LONG else -1.0
    tp_dist = entry_px * config.tp_pct / 100.0
    sl_dist = entry_px * sl_pct_eff / 100.0
    tp_px = entry_px + sign * tp_dist
    sl_px = entry_px - sign * sl_dist  # current SL, mutated by trailing

    # ---- Graceful clamp on timeout ------------------------------------
    # `timeout_ticks` from `filters.dynamic_timeout` can be up to 1200
    # (120 sec / 100 ms), plus 20 for the limit-close window. If the caller
    # passed a shorter path we just truncate — the bar is that `simulate
    # _trade` never reads past the end of `mid_path`.
    max_mon_ticks = max(0, path.size - config.timeout_limit_ticks)
    timeout_ticks = min(int(config.timeout_ticks), max_mon_ticks)
    if timeout_ticks <= 0:
        # Not enough forward ticks to even run the timeout — treat as an
        # immediate market close at the first tick (conservative), taker
        # commission.
        exit_px = float(path[0])
        return _make_outcome(
            direction, entry_px, exit_px, None,
            "timeout_market", 1, False, 0, config,
        )

    # ---- Monitoring loop ----------------------------------------------
    # State machine: walk forward tick-by-tick. At each tick:
    #   1) if stop level breached → stop-market fill at the stop price
    #      (reason tagged by which trailing_step we're in);
    #   2) elif TP breached → limit fill at tp_px;
    #   3) elif progress ≥ step-2 threshold → move SL to 50% × TP dist;
    #   4) elif progress ≥ step-1 threshold → place partial (optimistic
    #      instant fill at current mid) AND move SL to max(+0.08%, 30%×TP);
    #   5) else advance.
    # SL wins ties on the same tick (conservative — the existing triple-
    # barrier test suite encodes this convention).
    trailing_step = 0
    partial_filled = False
    partial_px: float | None = None

    for t in range(timeout_ticks):
        px = float(path[t])

        if direction == SimDirection.LONG:
            hit_sl = px <= sl_px
            hit_tp = px >= tp_px
            progress = (px - entry_px) / tp_dist if tp_dist > 0 else 0.0
        else:
            hit_sl = px >= sl_px
            hit_tp = px <= tp_px
            progress = (entry_px - px) / tp_dist if tp_dist > 0 else 0.0

        if hit_sl:
            # Label the exit by which trailing step we were in when it hit.
            # `partial_plus_*` variants let the commission switch know a
            # partial already took place but the remainder exited via taker.
            if trailing_step == 0:
                reason = "sl_hit"
            elif trailing_step == 1:
                reason = "trailing_sl_1" if not partial_filled else "partial_plus_trailing_sl_1"
            else:
                reason = "trailing_sl_2" if not partial_filled else "partial_plus_trailing_sl_2"
            return _make_outcome(
                direction, entry_px, sl_px, partial_px,
                reason, t + 1, partial_filled, trailing_step, config,
            )

        if hit_tp:
            reason = "tp_hit" if not partial_filled else "partial_plus_tp"
            return _make_outcome(
                direction, entry_px, tp_px, partial_px,
                reason, t + 1, partial_filled, trailing_step, config,
            )

        # Order the progress checks largest-first so a single tick can
        # legally jump both steps (e.g. a 0.8×TP_dist spike past 50%).
        # Trailing SL moves are gated by config.trailing_enabled; partial TP
        # fill is gated by config.partial_enabled. Both default True, so the
        # original coupled strategy is preserved unless grid search flips them.
        if config.trailing_enabled and progress >= config.trailing_step2_progress and trailing_step < 2:
            sl_offset = tp_dist * config.trailing_step2_sl_ratio
            sl_px = entry_px + sign * sl_offset
            trailing_step = 2
        elif progress >= config.trailing_step1_progress and trailing_step < 1:
            if config.partial_enabled and not partial_filled:
                partial_px = px
                partial_filled = True
            if config.trailing_enabled:
                min_offset = entry_px * (config.trailing_step1_sl_floor_pct / 100.0)
                ratio_offset = tp_dist * config.trailing_step1_sl_ratio
                sl_offset = max(min_offset, ratio_offset)
                sl_px = entry_px + sign * sl_offset
                trailing_step = 1

    # ---- Timeout branch ----------------------------------------------
    # executor._position_timeout (lines 407-471): at the timeout mark we
    # cancel TP/SL, place a LIMIT GTX at the current mid, wait 2 s, and
    # fall back to MARKET if unfilled. In the simulator we approximate
    # "unfilled after 2 s" by asking whether the mid crossed the limit
    # price at least once in the next `timeout_limit_ticks` ticks —
    # because a GTX SELL at price P fills only when a market buy lifts
    # it, and on mids that's equivalent to the mid touching P from the
    # right side. If yes → timeout_limit (maker); else → timeout_market
    # (taker) at the last observed mid.
    timeout_mid = float(path[timeout_ticks - 1])
    # Guard on the window: if the path is exactly timeout_ticks long we
    # have no limit-close window to observe, so fall straight to market.
    limit_window_start = timeout_ticks
    limit_window_end = min(
        timeout_ticks + config.timeout_limit_ticks, path.size,
    )
    filled = False
    fill_tick = limit_window_end
    if limit_window_end > limit_window_start:
        window = path[limit_window_start:limit_window_end]
        if direction == SimDirection.LONG:
            # SELL limit at timeout_mid fills when mid ≥ timeout_mid later.
            hits = np.where(window >= timeout_mid)[0]
        else:
            # BUY limit at timeout_mid fills when mid ≤ timeout_mid later.
            hits = np.where(window <= timeout_mid)[0]
        if hits.size > 0:
            filled = True
            fill_tick = limit_window_start + int(hits[0])

    if filled:
        return _make_outcome(
            direction, entry_px, timeout_mid, partial_px,
            "timeout_limit", fill_tick + 1, partial_filled, trailing_step, config,
        )
    # Fallback: market close at the last observed mid in the window (or at
    # timeout_mid if the window is empty).
    fallback_tick = limit_window_end - 1 if limit_window_end > limit_window_start else timeout_ticks - 1
    fallback_px = float(path[fallback_tick])
    return _make_outcome(
        direction, entry_px, fallback_px, partial_px,
        "timeout_market", fallback_tick + 1, partial_filled, trailing_step, config,
    )


# ---- Convenience helpers for build_samples / executor ------------------


def label_from_outcomes(
    long_outcome: TradeOutcome, short_outcome: TradeOutcome,
) -> tuple[int, float]:
    """Pick the best-PnL direction for training labels.

    Returns `(label, target_pnl_pct)` where label is UP/DOWN/FLAT (matching
    `src.model` constants via its own ids — 0/1/2) and `target_pnl_pct` is
    the continuous target used by the regression head. FLAT is returned when
    neither direction is net-profitable; the stored `target_pnl` is then the
    better (less negative) of the two so the regressor still learns "don't
    trade" from the magnitude.
    """
    from src.model import UP, DOWN, FLAT  # local import avoids circular

    long_pnl = long_outcome.net_pnl_pct
    short_pnl = short_outcome.net_pnl_pct

    # Both negative → FLAT, target is the better-of-two as the closest-to-
    # zero net outcome the model could have produced.
    if long_pnl <= 0 and short_pnl <= 0:
        return FLAT, max(long_pnl, short_pnl)
    if long_pnl >= short_pnl:
        return UP, long_pnl
    return DOWN, short_pnl
