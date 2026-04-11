from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.config import Config
from src.features import FeatureEngine
from src.model import HybridModel, UP, DOWN, FLAT

logger = logging.getLogger(__name__)

# Adaptive TP/SL bounds as percentage of price (prevent extremes)
MIN_TP_PCT = 0.10   # 0.10%
MAX_TP_PCT = 0.60   # 0.60%
MIN_SL_PCT = 0.05   # 0.05%
MAX_SL_PCT = 0.30   # 0.30%

# Lever 3 — entry filter tightening. Each constant maps directly to a filter
# in `SignalGenerator.generate`. Keeping them at module scope makes A/B
# experiments cheap and the rationale auditable in one place.

# 3.1 Time-of-day skip: Asia night (UTC) — thin liquidity, spoofing prevalent.
ASIA_NIGHT_START_UTC = 4
ASIA_NIGHT_END_UTC = 7    # half-open interval [start, end)

# 3.2 Liquidity depth gate: require thicker top-5 than the existing 50 BTC
# emergency circuit breaker before opening a position.
MIN_TOP5_BTC_LIQUIDITY = 100.0

# 3.3 Volatility band: below 0.7 the market is dead (model output is noise),
# above 2.5 it is news-driven (model can't predict). Sweet spot is moderate
# turbulence where microstructure signals work.
VOL_BAND_LOW = 0.7
VOL_BAND_HIGH = 2.5

# 3.4 Recent performance gate: if the rolling 10-trade winrate falls below
# RECENT_WR_FLOOR, pause entries for RECENT_WR_PAUSE_SEC. Catches regime
# shifts in real time, before the 4-hour retraining cycle notices.
RECENT_WR_FLOOR = 0.40
RECENT_WR_PAUSE_SEC = 600  # 10 minutes

# 3.5 Funding proximity: extend existing ±2 min to ±3 min around settlement
# (00:00, 08:00, 16:00 UTC). Funding settlement is the worst time to be in
# a position because of mark-price spikes and predatory liquidations.
FUNDING_GUARD_MIN = 3
FUNDING_HOURS_UTC = (0, 8, 16)

# 3.6 Spread tightening: drop from $0.03 (3 ticks) to $0.02 (2 ticks). The
# old value was a calibration constant from the data-collection phase; live
# economics need a tighter floor since the entry maker fee is fixed.
MAX_SPREAD_USD = 0.02


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

        Funding settles at 00:00, 08:00, 16:00 UTC on Binance Futures. The
        window is symmetric, e.g. for 08:00 the guard is [07:57, 08:03].
        """
        for h in FUNDING_HOURS_UTC:
            delta_min = abs((now_utc.hour - h) * 60 + now_utc.minute)
            # Wrap around midnight (e.g. 23:58 ↔ 00:00).
            delta_min = min(delta_min, 24 * 60 - delta_min)
            if delta_min <= FUNDING_GUARD_MIN:
                return True
        return False

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
        now_utc = datetime.now(timezone.utc)
        if ASIA_NIGHT_START_UTC <= now_utc.hour < ASIA_NIGHT_END_UTC:
            return Signal.NONE
        if self._is_funding_window(now_utc):
            return Signal.NONE

        # Recent-WR pause: if the rolling 10-trade WR fell below the floor,
        # we paused for RECENT_WR_PAUSE_SEC and refuse signals until then.
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

        # --- Dynamic threshold (self-tuning or base) ---
        threshold = self._cfg.confidence_threshold
        if self._executor is not None:
            threshold = self._executor.dynamic_threshold

        # --- Filters ---
        if confidence < threshold:
            return Signal.NONE

        spread = raw.get("spread", 999)
        if spread > MAX_SPREAD_USD:
            return Signal.NONE

        vol = raw.get("volatility_1s", 0)
        vol_3s = raw.get("volatility_3sigma", vol * 3)
        if vol > vol_3s:
            return Signal.NONE

        # Lever 3.3 — volatility band gate
        vol_ratio = raw.get("volatility_ratio", 1.0)
        if not (VOL_BAND_LOW < vol_ratio < VOL_BAND_HIGH):
            return Signal.NONE

        # Lever 3.2 — liquidity depth gate. raw["depth_ratio_l5"] is a ratio,
        # so we go straight to the order book for absolute volumes.
        ob_snap = fe._ob.current
        if ob_snap is None:
            return Signal.NONE
        top5_bid_btc = float(ob_snap.bids[:5, 1].sum())
        top5_ask_btc = float(ob_snap.asks[:5, 1].sum())
        if top5_bid_btc < MIN_TOP5_BTC_LIQUIDITY or top5_ask_btc < MIN_TOP5_BTC_LIQUIDITY:
            return Signal.NONE

        # Spoof filter
        spoof = raw.get("spoof_score", 0)
        if spoof > 0.5:
            logger.debug("Spoof detected (%.2f), skipping signal", spoof)
            return Signal.NONE

        # Fill rate filter: if too few orders filling, market is too fast
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

        # In mean-reversion regime (H < 0.45), skip momentum signals
        if hurst < 0.45 and prediction in (UP, DOWN):
            # Only take signals with very high confidence in mean-reversion
            if confidence < threshold + 0.05:
                return Signal.NONE

        # Adaptive TP/SL (percentage-based, scaled by volatility)
        tp_pct, sl_pct = self._adaptive_tp_sl(raw)

        # Liquidation cluster boost
        liq_prox = raw.get("liquidation_proximity", 0)

        # Sweep boost: recent sweep in signal direction = stronger conviction
        sweep = raw.get("sweep_intensity", 0)

        if prediction == UP and imbalance > 0.15:
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

        if prediction == DOWN and imbalance < -0.15:
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

        Returns (tp_pct, sl_pct) as percentages of price.
        """
        vol_ratio = raw.get("volatility_ratio", 1.0)
        vol_ratio = max(0.5, min(vol_ratio, 3.0))

        tp_pct = self._cfg.take_profit_pct * vol_ratio
        sl_pct = self._cfg.stop_loss_pct * vol_ratio

        tp_pct = max(MIN_TP_PCT, min(tp_pct, MAX_TP_PCT))
        sl_pct = max(MIN_SL_PCT, min(sl_pct, MAX_SL_PCT))

        return tp_pct, sl_pct

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
