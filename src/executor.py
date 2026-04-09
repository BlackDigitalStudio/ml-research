from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from src.config import Config
from src.notifier import Notifier
from src.recorder import Recorder
from src.risk import RiskManager
from src.signal import Signal, Direction
from src.ws_client import BinanceWSClient

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "IDLE"
    ENTERING = "ENTERING"
    IN_POSITION = "IN_POSITION"
    EXITING = "EXITING"


class Executor:
    def __init__(
        self,
        config: Config,
        ws: BinanceWSClient,
        risk: RiskManager,
        recorder: Recorder,
        notifier: Notifier,
    ) -> None:
        self._cfg = config
        self._ws = ws
        self._risk = risk
        self._recorder = recorder
        self._notifier = notifier

        self.state = State.IDLE
        self.balance: float = 0.0

        # Current position
        self._direction: Direction = Direction.NONE
        self._entry_order_id: str = ""
        self._sl_order_id: str = ""
        self._tp_order_id: str = ""
        self._entry_price: float = 0.0
        self._size: float = 0.0
        self._entry_time: float = 0.0
        self._position_time: float = 0.0

        # Timeout handles
        self._order_timeout_task: asyncio.Task | None = None
        self._position_timeout_task: asyncio.Task | None = None

        # API latency tracking
        self.last_api_latency_ms: float = 0.0

        # Fill rate tracking (last 20 entry attempts)
        self._fill_history: deque[bool] = deque(maxlen=20)  # True=filled, False=rejected/timeout

        # Self-tuning threshold
        self._recent_trades_pnl: deque[float] = deque(maxlen=100)
        self.dynamic_threshold: float = config.confidence_threshold
        self._last_threshold_tune: float = 0.0

    @property
    def fill_rate(self) -> float:
        """Fill rate of last 20 entry attempts (0.0 - 1.0)."""
        if not self._fill_history:
            return 1.0
        return sum(self._fill_history) / len(self._fill_history)

    def tune_threshold(self) -> None:
        """Recalculate confidence threshold to maximize profit factor on recent trades.

        Called every hour from main loop.
        """
        if len(self._recent_trades_pnl) < 20:
            return

        pnls = list(self._recent_trades_pnl)
        base = self._cfg.confidence_threshold

        # Test thresholds 0.50 - 0.70 in steps of 0.02
        # Since we can't replay trades, use heuristic:
        # If win rate is high → lower threshold (more trades)
        # If win rate is low → raise threshold (fewer, better trades)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls)
        total_win = sum(p for p in pnls if p > 0)
        total_loss = abs(sum(p for p in pnls if p < 0))
        pf = total_win / total_loss if total_loss > 0 else 2.0

        if pf > 1.5 and wr > 0.55:
            # Performing well → can be slightly less strict
            new_thresh = max(0.50, base - 0.02)
        elif pf < 1.0 or wr < 0.48:
            # Underperforming → be stricter
            new_thresh = min(0.70, base + 0.02)
        else:
            new_thresh = base

        if new_thresh != self.dynamic_threshold:
            logger.info(
                "Threshold tuned: %.2f → %.2f (PF=%.2f, WR=%.1f%%)",
                self.dynamic_threshold, new_thresh, pf, wr * 100,
            )
            self.dynamic_threshold = new_thresh

    async def start(self) -> None:
        self._ws.on_user_data(self._on_user_data)
        await self._sync_balance()
        logger.info("Executor started, balance=$%.2f, state=%s", self.balance, self.state.value)

    async def _sync_balance(self) -> None:
        data = await self._ws.rest_get("/fapi/v2/balance", signed=True)
        if isinstance(data, list):
            for b in data:
                if b.get("asset") == "USDT":
                    self.balance = float(b.get("balance", 0))
                    self._risk.set_deposit(self.balance)
                    return

    async def process_signal(self, signal: Signal) -> None:
        if self.state != State.IDLE:
            return

        if signal.direction == Direction.NONE:
            return

        # Check risk
        allowed, reason = self._risk.is_trading_allowed()
        if not allowed:
            logger.debug("Trade blocked by risk: %s", reason)
            return

        await self._enter(signal)

    async def _enter(self, signal: Signal) -> None:
        self.state = State.ENTERING
        self._direction = signal.direction
        self._size = signal.size
        self._entry_time = time.monotonic()

        side = "BUY" if signal.direction == Direction.LONG else "SELL"

        # Place limit entry order (Post-Only GTX)
        t0 = time.monotonic()
        result = await self._ws.rest_post("/fapi/v1/order", params={
            "symbol": self._cfg.symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTX",
            "quantity": f"{signal.size:.3f}",
            "price": f"{signal.entry_price:.2f}",
            "newOrderRespType": "RESULT",
        })
        self.last_api_latency_ms = (time.monotonic() - t0) * 1000

        if "orderId" not in result:
            code = result.get("code", "?")
            msg = result.get("msg", str(result))
            logger.warning("Entry order rejected: [%s] %s", code, msg)
            self.state = State.IDLE
            self._direction = Direction.NONE
            return

        self._entry_order_id = str(result["orderId"])
        self._entry_price = signal.entry_price
        logger.info(
            "Entry order placed: %s %s %.3f @ $%.2f (id=%s, %.1fms)",
            signal.direction.value, self._cfg.symbol, signal.size,
            signal.entry_price, self._entry_order_id, self.last_api_latency_ms,
        )

        # Place backup stop-loss on exchange
        sl_side = "SELL" if signal.direction == Direction.LONG else "BUY"
        sl_result = await self._ws.rest_post("/fapi/v1/order", params={
            "symbol": self._cfg.symbol,
            "side": sl_side,
            "type": "STOP_MARKET",
            "stopPrice": f"{signal.stop_loss:.2f}",
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        })
        if "orderId" in sl_result:
            self._sl_order_id = str(sl_result["orderId"])
            logger.info("Backup SL placed: $%.2f (id=%s)", signal.stop_loss, self._sl_order_id)

        # Start order timeout
        self._order_timeout_task = asyncio.create_task(
            self._order_timeout(self._entry_order_id)
        )

    async def _order_timeout(self, order_id: str) -> None:
        await asyncio.sleep(self._cfg.order_timeout_sec)
        if self.state == State.ENTERING and self._entry_order_id == order_id:
            logger.info("Entry order timeout, cancelling")
            self._fill_history.append(False)
            await self._cancel_order(order_id)
            await self._cancel_order(self._sl_order_id)
            self._reset()

    async def _on_filled(self, data: dict) -> None:
        """Called when entry order is filled."""
        self._fill_history.append(True)
        self.state = State.IN_POSITION
        self._position_time = time.monotonic()

        fill_price = float(data.get("ap", self._entry_price))  # avg price
        self._entry_price = fill_price

        logger.info(
            "Position opened: %s %.3f @ $%.2f",
            self._direction.value, self._size, fill_price,
        )
        self._notifier.trade(self._direction.value, fill_price, self._size)

        # Place take-profit
        tp_side = "SELL" if self._direction == Direction.LONG else "BUY"
        if self._direction == Direction.LONG:
            tp_price = fill_price + self._cfg.take_profit_usd
        else:
            tp_price = fill_price - self._cfg.take_profit_usd

        tp_result = await self._ws.rest_post("/fapi/v1/order", params={
            "symbol": self._cfg.symbol,
            "side": tp_side,
            "type": "LIMIT",
            "timeInForce": "GTX",
            "quantity": f"{self._size:.3f}",
            "price": f"{tp_price:.2f}",
            "newOrderRespType": "RESULT",
        })
        if "orderId" in tp_result:
            self._tp_order_id = str(tp_result["orderId"])
            logger.info("TP placed: $%.2f (id=%s)", tp_price, self._tp_order_id)

        # Start position timeout
        self._position_timeout_task = asyncio.create_task(
            self._position_timeout()
        )

    async def _position_timeout(self) -> None:
        await asyncio.sleep(self._cfg.position_timeout_sec)
        if self.state == State.IN_POSITION:
            logger.info("Position timeout, closing at market")
            await self._close_position("timeout")

    async def _close_position(self, reason: str) -> None:
        if self.state not in (State.IN_POSITION, State.EXITING):
            return

        self.state = State.EXITING

        # Cancel TP and SL
        await self._cancel_order(self._tp_order_id)
        await self._cancel_order(self._sl_order_id)

        # Market close
        close_side = "SELL" if self._direction == Direction.LONG else "BUY"
        result = await self._ws.rest_post("/fapi/v1/order", params={
            "symbol": self._cfg.symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": f"{self._size:.3f}",
            "newOrderRespType": "RESULT",
        })

        exit_price = float(result.get("avgPrice", 0))
        self._finalize_trade(exit_price, reason)

    def _finalize_trade(self, exit_price: float, reason: str) -> None:
        if self._direction == Direction.LONG:
            pnl = (exit_price - self._entry_price) * self._size
        else:
            pnl = (self._entry_price - exit_price) * self._size

        duration = time.monotonic() - self._position_time
        # Different commission for win (maker+maker) vs loss (maker+taker SL)
        notional = self._entry_price * self._size
        if reason == "stop_loss":
            fees = notional * self._cfg.commission_loss_pct / 100
        else:
            fees = notional * self._cfg.commission_win_pct / 100
        net_pnl = pnl - fees

        self._risk.record_trade(net_pnl)
        self._recent_trades_pnl.append(net_pnl)
        self.balance += net_pnl

        self._recorder.record_bot_trade(
            direction=self._direction.value,
            entry_price=self._entry_price,
            exit_price=exit_price,
            size=self._size,
            pnl=net_pnl,
            fees=fees,
            duration_sec=duration,
            reason=reason,
        )

        logger.info(
            "Trade closed: %s entry=$%.2f exit=$%.2f PnL=$%.4f (%s) duration=%.1fs",
            self._direction.value, self._entry_price, exit_price,
            net_pnl, reason, duration,
        )
        self._notifier.trade(
            f"CLOSE {self._direction.value}",
            exit_price, self._size, pnl=net_pnl,
        )

        self._reset()

    async def _cancel_order(self, order_id: str) -> None:
        if not order_id:
            return
        try:
            await self._ws.rest_delete("/fapi/v1/order", params={
                "symbol": self._cfg.symbol,
                "orderId": order_id,
            })
        except Exception as e:
            logger.debug("Cancel order %s: %s", order_id, e)

    def _reset(self) -> None:
        self.state = State.IDLE
        self._direction = Direction.NONE
        self._entry_order_id = ""
        self._sl_order_id = ""
        self._tp_order_id = ""
        self._entry_price = 0.0
        self._size = 0.0
        if self._order_timeout_task:
            self._order_timeout_task.cancel()
        if self._position_timeout_task:
            self._position_timeout_task.cancel()

    async def _on_user_data(self, data: dict) -> None:
        event_type = data.get("e", "")

        if event_type == "ORDER_TRADE_UPDATE":
            order = data.get("o", {})
            order_id = str(order.get("i", ""))
            status = order.get("X", "")
            side = order.get("S", "")

            # Entry filled
            if order_id == self._entry_order_id and status == "FILLED":
                if self._order_timeout_task:
                    self._order_timeout_task.cancel()
                await self._on_filled(order)

            # Entry expired/cancelled (GTX rejected)
            elif order_id == self._entry_order_id and status in ("EXPIRED", "CANCELED"):
                logger.info("Entry order %s: %s", status, order_id)
                self._fill_history.append(False)
                await self._cancel_order(self._sl_order_id)
                self._reset()

            # TP filled
            elif order_id == self._tp_order_id and status == "FILLED":
                exit_price = float(order.get("ap", 0))
                await self._cancel_order(self._sl_order_id)
                if self._position_timeout_task:
                    self._position_timeout_task.cancel()
                self._finalize_trade(exit_price, "take_profit")

            # SL filled
            elif order_id == self._sl_order_id and status == "FILLED":
                exit_price = float(order.get("ap", 0))
                await self._cancel_order(self._tp_order_id)
                if self._position_timeout_task:
                    self._position_timeout_task.cancel()
                self._finalize_trade(exit_price, "stop_loss")

    async def emergency_close(self, reason: str) -> None:
        logger.warning("EMERGENCY CLOSE: %s", reason)
        self._notifier.alert("Emergency Close", reason)

        # Cancel all open orders
        await self._ws.rest_delete("/fapi/v1/allOpenOrders", params={
            "symbol": self._cfg.symbol,
        })

        if self.state == State.IN_POSITION:
            close_side = "SELL" if self._direction == Direction.LONG else "BUY"
            result = await self._ws.rest_post("/fapi/v1/order", params={
                "symbol": self._cfg.symbol,
                "side": close_side,
                "type": "MARKET",
                "quantity": f"{self._size:.3f}",
                "newOrderRespType": "RESULT",
            })
            exit_price = float(result.get("avgPrice", 0))
            self._finalize_trade(exit_price, f"emergency:{reason}")
        else:
            self._reset()

    async def handle_reversal_signal(self, signal: Signal) -> None:
        """Close current position on reversal signal and optionally enter new one."""
        if self.state != State.IN_POSITION:
            return

        if signal.direction == Direction.NONE:
            return

        # Only reverse if opposite direction
        if (self._direction == Direction.LONG and signal.direction == Direction.SHORT) or \
           (self._direction == Direction.SHORT and signal.direction == Direction.LONG):
            logger.info("Reversal signal: %s → %s", self._direction.value, signal.direction.value)
            await self._close_position("reversal")
