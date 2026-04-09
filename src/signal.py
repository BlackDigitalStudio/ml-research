from __future__ import annotations

import logging
from dataclasses import dataclass
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

    def set_executor(self, executor) -> None:
        """Set reference to executor for dynamic threshold and fill rate."""
        self._executor = executor

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

        prediction, confidence = self._model.predict(lob_tensor, fe.features)

        # --- Dynamic threshold (self-tuning or base) ---
        threshold = self._cfg.confidence_threshold
        if self._executor is not None:
            threshold = self._executor.dynamic_threshold

        # --- Filters ---
        if confidence < threshold:
            return Signal.NONE

        spread = raw.get("spread", 999)
        if spread > 0.03:
            return Signal.NONE

        vol = raw.get("volatility_1s", 0)
        vol_3s = raw.get("volatility_3sigma", vol * 3)
        if vol > vol_3s:
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
