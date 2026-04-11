from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src import filters
from src.config import Config
from src.features import FeatureEngine
from src.model import HybridModel, UP, DOWN, FLAT

logger = logging.getLogger(__name__)

# Lever 3 — entry filter tightening. The numeric constants live in
# `src/filters.py` (the shared filter module consumed by signal.py,
# trainer.py and backtest.py). We re-export the ones legacy tests still
# import by name, but the canonical source of truth is `filters`.
#
# DO NOT change these constants without also changing `src/filters.py`
# — `tests/test_signal_filters.py` is a parity guard between the two.

# Adaptive TP/SL bounds (percentage of price)
MIN_TP_PCT = filters.MIN_TP_PCT
MAX_TP_PCT = filters.MAX_TP_PCT
MIN_SL_PCT = filters.MIN_SL_PCT
MAX_SL_PCT = filters.MAX_SL_PCT

# 3.1 Asia-night skip.
ASIA_NIGHT_START_UTC = filters.ASIA_NIGHT_START_UTC
ASIA_NIGHT_END_UTC = filters.ASIA_NIGHT_END_UTC

# 3.2 Liquidity depth gate.
MIN_TOP5_BTC_LIQUIDITY = filters.MIN_TOP5_BTC_LIQUIDITY

# 3.3 Volatility band.
VOL_BAND_LOW = filters.VOL_BAND_LOW
VOL_BAND_HIGH = filters.VOL_BAND_HIGH

# 3.4 Recent-WR pause — STATEFUL, stays in signal.py (trainer approximates
# "no pause" at label time per handoff_current.md design decision 3).
RECENT_WR_FLOOR = 0.40
RECENT_WR_PAUSE_SEC = 600  # 10 minutes

# 3.5 Funding proximity guard.
FUNDING_GUARD_MIN = filters.FUNDING_GUARD_MIN
FUNDING_HOURS_UTC = filters.FUNDING_HOURS_UTC

# 3.6 Spread ceiling.
MAX_SPREAD_USD = filters.MAX_SPREAD_USD


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    direction: Direction = Direction.NONE
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    size: float = 0.0
    confidence: float = 0.0


Signal.NONE = Signal()


class SignalGenerator:
    def __init__(
        self,
        config: Config,
        model: HybridModel,
        features: FeatureEngine,
    ) -> None:
        self._cfg = config
        self._model = model
        self._features = features
        self._executor = None  # set via set_executor()
        # State for the recent-WR pause gate (Lever 3.4): once triggered, the
        # pause lasts RECENT_WR_PAUSE_SEC and is not re-evaluated until then.
        self._recent_wr_pause_until: float = 0.0

    def set_executor(self, executor) -> None:
        """Set reference to executor for dynamic threshold and fill rate."""
        self._executor = executor

    def _is_funding_window(self, now_utc: datetime) -> bool:
        """True if `now_utc` is within ±FUNDING_GUARD_MIN of any funding hour.

        Thin wrapper around `filters.passes_funding_blackout` so existing
        unit tests (`tests/test_signal_filters.py`) that test this method
        by name keep passing after the refactor.
        """
        ts_ms = now_utc.timestamp() * 1000.0
        return not filters.passes_funding_blackout(ts_ms)

    def generate(self, balance: float) -> Signal:
        fe = self._features
        raw = fe.features_raw

        if not raw:
            return Signal.NONE

        lob_tensor = fe.build_lob_tensor()
        if lob_tensor is None:
            return Signal.NONE

        if not self._model.is_loaded:
            return Signal.NONE

        # === Lever 3 — pre-prediction filters (cheap, time-only) ===
        # Run these first so we don't pay the CNN+ensemble cost when we know
        # the entry will be filtered out anyway.
        now_ms = time.time() * 1000.0
        if not filters.passes_time_of_day(now_ms):
            return Signal.NONE
        if not filters.passes_funding_blackout(now_ms):
            return Signal.NONE

        # Recent-WR pause: STATEFUL, not in shared filters. If the rolling
        # 10-trade WR fell below the floor, we paused for RECENT_WR_PAUSE_SEC
        # and refuse signals until then.
        now_mono = time.monotonic()
        if now_mono < self._recent_wr_pause_until:
            return Signal.NONE
        if self._executor is not None:
            wr = self._executor.recent_wr
            if wr is not None and wr < RECENT_WR_FLOOR:
                self._recent_wr_pause_until = now_mono + RECENT_WR_PAUSE_SEC
                logger.warning(
                    "Recent 10-trade WR=%.0f%% < %.0f%% floor — pausing entries for %d sec",
                    wr * 100, RECENT_WR_FLOOR * 100, RECENT_WR_PAUSE_SEC,
                )
                return Signal.NONE

        prediction, confidence = self._model.predict(lob_tensor, fe.features)

        # --- Dynamic threshold (self-tuning or base) — STATEFUL ---
        threshold = self._cfg.confidence_threshold
        if self._executor is not None:
            threshold = self._executor.dynamic_threshold

        # --- Filters ---
        if confidence < threshold:
            return Signal.NONE

        spread = raw.get("spread", 999)
        if not filters.passes_spread(spread):
            return Signal.NONE

        vol = raw.get("volatility_1s", 0)
        vol_3s = raw.get("volatility_3sigma", vol * 3)
        if not filters.passes_vol_spike(vol, vol_3s):
            return Signal.NONE

        # Lever 3.3 — volatility band gate
        vol_ratio = raw.get("volatility_ratio", 1.0)
        if not filters.passes_vol_band(vol_ratio):
            return Signal.NONE

        # Lever 3.2 — liquidity depth gate. raw["depth_ratio_l5"] is a ratio,
        # so we go straight to the order book for absolute volumes.
        ob_snap = fe._ob.current
        if ob_snap is None:
            return Signal.NONE
        top5_bid_btc = float(ob_snap.bids[:5, 1].sum())
        top5_ask_btc = float(ob_snap.asks[:5, 1].sum())
        if not filters.passes_liquidity(top5_bid_btc, top5_ask_btc):
            return Signal.NONE

        # Spoof filter
        spoof = raw.get("spoof_score", 0)
        if not filters.passes_spoof(spoof):
            logger.debug("Spoof detected (%.2f), skipping signal", spoof)
            return Signal.NONE

        # Fill rate filter: STATEFUL — too few orders filling ⇒ market is
        # too fast. Not in shared filters (label time assumes fill_rate=1.0).
        if self._executor is not None and self._executor.fill_rate < 0.25:
            logger.debug("Low fill rate (%.0f%%), pausing entries", self._executor.fill_rate * 100)
            return Signal.NONE

        imbalance = raw.get("imbalance_ratio", 0)
        best_bid = fe._ob.current.best_bid
        best_ask = fe._ob.current.best_ask

        # Hurst-based regime adjustment
        hurst = raw.get("hurst_exponent", 0.5)
        n_votes = getattr(self._model, 'last_n_votes', 5)
        size = self._calc_size(balance, hurst, confidence=confidence, n_votes=n_votes)
        if size <= 0:
            return Signal.NONE

        # In mean-reversion regime (H < 0.45), skip momentum signals unless
        # confidence clears a higher bar.
        if prediction in (UP, DOWN) and not filters.passes_hurst_regime(
            hurst, confidence, threshold,
        ):
            return Signal.NONE

        # Adaptive TP/SL (percentage-based, scaled by volatility)
        tp_pct, sl_pct = self._adaptive_tp_sl(raw)

        # Liquidation cluster boost
        liq_prox = raw.get("liquidation_proximity", 0)

        # Sweep boost: recent sweep in signal direction = stronger conviction
        sweep = raw.get("sweep_intensity", 0)

        if prediction == UP and filters.passes_imbalance_long(imbalance):
            adj_confidence = confidence
            if liq_prox > 0.005:
                adj_confidence = min(confidence + 0.05, 1.0)
            if sweep >= 3:
                adj_confidence = min(adj_confidence + 0.03, 1.0)

            # TP/SL in USD from percentage
            sl_usd = best_bid * sl_pct / 100
            tp_usd = best_bid * tp_pct / 100
            return Signal(
                direction=Direction.LONG,
                entry_price=best_bid,
                stop_loss=best_bid - sl_usd,
                take_profit=best_bid + tp_usd,
                size=size,
                confidence=adj_confidence,
            )

        if prediction == DOWN and filters.passes_imbalance_short(imbalance):
            adj_confidence = confidence
            if liq_prox < -0.005:
                adj_confidence = min(confidence + 0.05, 1.0)
            if sweep >= 3:
                adj_confidence = min(adj_confidence + 0.03, 1.0)

            sl_usd = best_ask * sl_pct / 100
            tp_usd = best_ask * tp_pct / 100
            return Signal(
                direction=Direction.SHORT,
                entry_price=best_ask,
                stop_loss=best_ask + sl_usd,
                take_profit=best_ask - tp_usd,
                size=size,
                confidence=adj_confidence,
            )

        return Signal.NONE

    def _adaptive_tp_sl(self, raw: dict) -> tuple[float, float]:
        """Scale TP/SL percentage by current volatility.

        Delegates to `filters.adaptive_tp_sl` so trainer / backtest /
        live-signal all share the same math. The existing signature is
        kept so downstream tests that call it by name still work.
        """
        return filters.adaptive_tp_sl(
            vol_ratio=raw.get("volatility_ratio", 1.0),
            base_tp_pct=self._cfg.take_profit_pct,
            base_sl_pct=self._cfg.stop_loss_pct,
        )

    def _calc_size(
        self, balance: float, hurst: float = 0.5,
        confidence: float = 0.6, n_votes: int = 5,
    ) -> float:
        if balance <= 0:
            return 0.0

        notional = balance * self._cfg.leverage
        position_notional = notional * self._cfg.position_size_pct / 100

        # Hurst-based sizing: reduce in uncertain regime
        if 0.45 <= hurst <= 0.55:
            position_notional *= 0.5  # uncertain regime → half size
        # In trending or mean-reverting regime → full size

        # Ensemble confidence scaling (Item 4)
        if n_votes >= 5 and confidence > 0.65:
            pass  # full size
        elif n_votes >= 4:
            position_notional *= 0.75
        elif n_votes >= 3:
            position_notional *= 0.50

        mid = self._features._ob.current.mid_price
        if mid <= 0:
            return 0.0

        size = position_notional / mid
        size = round(size, 3)  # BTC step size = 0.001
        if size * mid < 100:   # BTC min notional = $100
            return 0.0
        return size
