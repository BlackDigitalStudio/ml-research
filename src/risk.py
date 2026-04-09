from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.config import Config
from src.features import FeatureEngine
from src.order_book import OrderBook
from src.ws_client import BinanceWSClient

logger = logging.getLogger(__name__)

# Funding settlement times (UTC hours)
FUNDING_HOURS = (0, 8, 16)
FUNDING_WINDOW_SEC = 120  # ±2 minutes


class RiskManager:
    def __init__(
        self,
        config: Config,
        features: FeatureEngine,
        order_book: OrderBook,
        ws: BinanceWSClient,
    ) -> None:
        self._cfg = config
        self._features = features
        self._ob = order_book
        self._ws = ws

        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = 0
        self.total_win_amount: float = 0.0
        self.total_loss_amount: float = 0.0
        self._pause_until: float = 0.0
        self._last_reset_day: int = 0
        self._deposit: float = 50.0

    def set_deposit(self, balance: float) -> None:
        self._deposit = balance

    def record_trade(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self.daily_trades += 1

        if pnl > 0:
            self.daily_wins += 1
            self.consecutive_losses = 0
            self.total_win_amount += pnl
        else:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            self.total_loss_amount += abs(pnl)

            if self.consecutive_losses >= self._cfg.max_consecutive_losses:
                self._pause_until = time.monotonic() + 1800  # 30 min pause
                logger.warning(
                    "Consecutive losses = %d, pausing 30 min",
                    self.consecutive_losses,
                )

    def reset_daily(self) -> None:
        today = datetime.now(timezone.utc).toordinal()
        if today != self._last_reset_day:
            logger.info(
                "Daily reset: P&L=$%.2f, trades=%d, WR=%.1f%%",
                self.daily_pnl,
                self.daily_trades,
                (self.daily_wins / self.daily_trades * 100) if self.daily_trades > 0 else 0,
            )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.consecutive_losses = 0
            self.max_consecutive_losses = 0
            self.total_win_amount = 0.0
            self.total_loss_amount = 0.0
            self._last_reset_day = today

    def is_trading_allowed(self) -> tuple[bool, str]:
        self.reset_daily()

        # Daily loss limit
        max_loss = self._deposit * self._cfg.max_daily_loss_pct / 100
        if self.daily_pnl <= -max_loss:
            return False, "daily_loss_limit"

        # Consecutive losses pause
        if self._pause_until > 0 and time.monotonic() < self._pause_until:
            return False, "consecutive_losses_pause"

        raw = self._features.features_raw
        if not raw:
            return False, "no_features"

        # Volatility spike
        vol = raw.get("volatility_1s", 0)
        vol_3s = raw.get("volatility_3sigma", vol * 3)
        if vol > vol_3s and vol_3s > 0:
            return False, "volatility_spike"

        # Spread blowout
        if raw.get("spread", 0) > 1.00:
            return False, "spread_blowout"

        # Funding settlement window
        if self._is_near_funding():
            return False, "funding_window"

        # WebSocket health
        ws_age = time.monotonic() - self._ws.last_message_time
        if self._ws.last_message_time > 0 and ws_age > 3.0:
            return False, "ws_stale"

        # Low liquidity
        if self._ob.current is not None:
            bid_depth = float(self._ob.current.bids[:5, 1].sum())
            if bid_depth < 50:
                return False, "low_liquidity"

        return True, "ok"

    def should_emergency_close(self) -> tuple[bool, str]:
        # WebSocket stale > 3 seconds
        ws_age = time.monotonic() - self._ws.last_message_time
        if self._ws.last_message_time > 0 and ws_age > 3.0:
            return True, "ws_stale"

        return False, "ok"

    def _is_near_funding(self) -> bool:
        now = datetime.now(timezone.utc)
        for h in FUNDING_HOURS:
            funding_ts = now.replace(hour=h, minute=0, second=0, microsecond=0)
            diff = abs((now - funding_ts).total_seconds())
            if diff < FUNDING_WINDOW_SEC:
                return True
        return False

    @property
    def avg_win(self) -> float:
        if self.daily_wins == 0:
            return 0.0
        return self.total_win_amount / self.daily_wins

    @property
    def avg_loss(self) -> float:
        losses = self.daily_trades - self.daily_wins
        if losses == 0:
            return 0.0
        return self.total_loss_amount / losses
