"""Stateless Tier-2 entry filters + adaptive TP/SL + dynamic timeout.

Shared between `src/signal.py` (live entries) and `src/trainer.py`
(build_samples label construction) and `scripts/backtest.py` (walk-forward
evaluation). Having the three callers consume the same pure functions is
the mechanism that keeps training labels, backtest metrics, and live
economics from drifting against each other — the catastrophic failure mode
documented in handoff_current.md.

**Excluded by design** (not in this module):
    * `recent_wr_pause` — stateful (needs rolling 10-trade PnL history),
      approximated as "no pause" at label time.
    * `fill_rate` — stateful (needs last 20 entry attempts), approximated
      as 1.0 at label time.
    * `dynamic_confidence_threshold` — stateful (executor.tune_threshold),
      approximated as the static config baseline at label time.
All three live inline inside `SignalGenerator.generate` and the label
construction uses `config.env` defaults in their place. This matches the
design-decision-3 contract in handoff_current.md.

**Numeric defaults** are kept in sync with signal.py's module-level Lever-3
constants. If you change one of those constants you MUST change the default
here too — the signal-parity test will otherwise drift silently.
"""
from __future__ import annotations

from datetime import datetime, timezone

# ---- Constants (mirrored from signal.py Lever 3) -----------------------

# 3.1 Time-of-day skip: Asia night (UTC).
ASIA_NIGHT_START_UTC = 4
ASIA_NIGHT_END_UTC = 7

# 3.5 Funding proximity guard: ±FUNDING_GUARD_MIN around each funding hour.
FUNDING_GUARD_MIN = 3
FUNDING_HOURS_UTC = (0, 8, 16)

# 3.6 Spread ceiling — 2 ticks on BTCUSDT (tick size = $0.10).
# The legacy value in signal.py was 0.02, which contradicted its own
# inline comment ("$0.02 (2 ticks)") — an off-by-10× typo. Real BTC
# futures data from data/depth shows median spread = $0.10 and 0% of
# samples ≤ $0.02, so the old filter rejected every live sample and
# training labels would collapse to zero. Fixed to 0.20 as part of the
# label/live-sync rewrite.
MAX_SPREAD_USD = 0.20

# 3.3 Volatility band.
VOL_BAND_LOW = 0.7
VOL_BAND_HIGH = 2.5

# 3.2 Liquidity depth gate.
#
# Real BTCUSDT futures depth on the top-5 levels has median ~4.8 BTC
# (bids) / ~4.0 BTC (asks) during normal conditions — the legacy 100.0
# constant was inherited from the ETH config and rejected 100% of
# samples on the 8h probe (`data/depth/*.parquet`). 2.0 BTC passes
# ~60% of samples which matches the "avoid the thinnest books"
# intent without zeroing out the training set. Bundled into the
# label/live-sync rewrite because build_samples would not produce
# any labels otherwise.
MIN_TOP5_BTC_LIQUIDITY = 2.0

# 3 (spoof + imbalance + hurst): threshold constants.
SPOOF_SCORE_MAX = 0.5
IMBALANCE_LONG_MIN = 0.15
IMBALANCE_SHORT_MAX = -0.15
HURST_MEAN_REVERTING_MAX = 0.45
HURST_STRICTER_CONFIDENCE_BONUS = 0.05

# Adaptive TP/SL bounds (percentage of price).
MIN_TP_PCT = 0.10
MAX_TP_PCT = 0.60
MIN_SL_PCT = 0.05
MAX_SL_PCT = 0.30

# Volatility-ratio clamp window before it multiplies TP/SL.
VOL_RATIO_MIN = 0.5
VOL_RATIO_MAX = 3.0

# Dynamic timeout clamp (seconds).
TIMEOUT_SEC_MIN = 15.0
TIMEOUT_SEC_MAX = 120.0


# ---- Pure filter predicates --------------------------------------------
# All functions return True when the sample PASSES (i.e. the trade may
# proceed). Callers aggregate the results with `all(...)` or short-circuit
# as soon as one fails.


def passes_time_of_day(ts_unix_ms: float) -> bool:
    """Reject samples in the Asia-night window [04:00, 07:00) UTC.

    The input is a Unix timestamp in MILLISECONDS to match the depth parquet
    schema (depth_ts is int64 ms). We convert once here so upstream call
    sites don't repeat the arithmetic.
    """
    hour = datetime.fromtimestamp(ts_unix_ms / 1000.0, tz=timezone.utc).hour
    return not (ASIA_NIGHT_START_UTC <= hour < ASIA_NIGHT_END_UTC)


def passes_funding_blackout(ts_unix_ms: float) -> bool:
    """Reject samples within ±FUNDING_GUARD_MIN of any funding settlement.

    Binance Futures funding settles at 00:00/08:00/16:00 UTC. The guard
    wraps around midnight (a 23:58 sample is inside the guard for 00:00).
    """
    now = datetime.fromtimestamp(ts_unix_ms / 1000.0, tz=timezone.utc)
    for h in FUNDING_HOURS_UTC:
        delta_min = abs((now.hour - h) * 60 + now.minute)
        delta_min = min(delta_min, 24 * 60 - delta_min)
        if delta_min <= FUNDING_GUARD_MIN:
            return False
    return True


def passes_spread(spread_usd: float) -> bool:
    return spread_usd <= MAX_SPREAD_USD


def passes_vol_spike(vol_1s: float, vol_3sigma: float) -> bool:
    """Reject when current 1-second vol breaches the 3σ rolling bound."""
    return vol_1s <= vol_3sigma


def passes_vol_band(vol_ratio: float) -> bool:
    return VOL_BAND_LOW < vol_ratio < VOL_BAND_HIGH


def passes_liquidity(top5_bid_btc: float, top5_ask_btc: float) -> bool:
    return (
        top5_bid_btc >= MIN_TOP5_BTC_LIQUIDITY
        and top5_ask_btc >= MIN_TOP5_BTC_LIQUIDITY
    )


def passes_spoof(spoof_score: float) -> bool:
    return spoof_score <= SPOOF_SCORE_MAX


def passes_imbalance_long(imbalance_ratio: float) -> bool:
    return imbalance_ratio > IMBALANCE_LONG_MIN


def passes_imbalance_short(imbalance_ratio: float) -> bool:
    return imbalance_ratio < IMBALANCE_SHORT_MAX


def passes_hurst_regime(
    hurst: float, confidence: float, base_threshold: float,
) -> bool:
    """Mean-reverting regime requires a stricter confidence threshold.

    If H >= 0.45 the gate is a no-op. Below 0.45 (mean-reversion) we bump
    the required confidence by HURST_STRICTER_CONFIDENCE_BONUS = 0.05.
    """
    if hurst >= HURST_MEAN_REVERTING_MAX:
        return True
    return confidence >= base_threshold + HURST_STRICTER_CONFIDENCE_BONUS


# ---- Adaptive parameters ------------------------------------------------


def adaptive_tp_sl(
    vol_ratio: float, base_tp_pct: float, base_sl_pct: float,
) -> tuple[float, float]:
    """Scale TP/SL by volatility ratio, clamped to the adaptive bounds.

    Mirrors `src/signal.py:_adaptive_tp_sl`. Separated out so trainer and
    backtest can call the identical function.
    """
    vr = max(VOL_RATIO_MIN, min(vol_ratio, VOL_RATIO_MAX))
    tp_pct = base_tp_pct * vr
    sl_pct = base_sl_pct * vr
    tp_pct = max(MIN_TP_PCT, min(tp_pct, MAX_TP_PCT))
    sl_pct = max(MIN_SL_PCT, min(sl_pct, MAX_SL_PCT))
    return tp_pct, sl_pct


def dynamic_timeout_sec(
    avg_volatility: float, current_volatility: float, base_timeout_sec: float,
) -> float:
    """Scale base timeout inversely with current volatility.

    Mirrors `src/executor.py:_position_timeout`. Clamped to [15, 120]
    seconds, returns `base_timeout_sec` unchanged when vol info is missing.
    """
    if avg_volatility <= 0 or current_volatility <= 0:
        return base_timeout_sec
    t = base_timeout_sec * (avg_volatility / max(current_volatility, 1e-10))
    return max(TIMEOUT_SEC_MIN, min(t, TIMEOUT_SEC_MAX))


def dynamic_timeout_ticks(
    avg_volatility: float,
    current_volatility: float,
    base_timeout_sec: float,
    tick_ms: float = 100.0,
) -> int:
    """Dynamic timeout converted to ticks (100 ms per tick by default)."""
    seconds = dynamic_timeout_sec(avg_volatility, current_volatility, base_timeout_sec)
    return int(round(seconds * 1000.0 / tick_ms))
