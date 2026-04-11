from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
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

        # Order book reference (set via set_order_book)
        self._ob = None

        # Timeout tracking (Item 2)
        self._recent_timeouts: deque[float] = deque(maxlen=100)

        # Volatility tracking (Item 3)
        self._avg_volatility: float = 0.0
        self._current_volatility: float = 0.0

        # Stepped trailing stop-loss (Item 14)
        self._tp_target: float = 0.0
        self._tp_dist: float = 0.0
        self._trailing_sl: float = 0.0
        self._trailing_step: int = 0  # 0=none, 1=50%, 2=75%
        self._trailing_task: asyncio.Task | None = None

        # Partial take-profit (Item 15)
        self._partial_filled: bool = False
        self._partial_order_id: str = ""

        # Adverse selection (Item 16)
        self._entry_submit_time: float = 0.0

    @property
    def fill_rate(self) -> float:
        """Fill rate of last 20 entry attempts (0.0 - 1.0)."""
        if not self._fill_history:
            return 1.0
        return sum(self._fill_history) / len(self._fill_history)

    @property
    def recent_wr(self) -> float | None:
        """Win-rate of the last 10 closed trades.

        Returns None if fewer than 10 trades have been seen, signalling that
        callers should not gate on this metric yet (sample too small).
        Lever 3.4 uses this to pause entries during sudden regime shifts that
        the 4-hour retraining cycle would otherwise miss.
        """
        if len(self._recent_trades_pnl) < 10:
            return None
        last10 = list(self._recent_trades_pnl)[-10:]
        wins = sum(1 for p in last10 if p > 0)
        return wins / 10.0

    def set_order_book(self, ob) -> None:
        """Set order book reference for mid-price access."""
        self._ob = ob

    def update_volatility(self, avg_vol: float, current_vol: float) -> None:
        """Update volatility tracking (called from main loop)."""
        if self._avg_volatility == 0.0:
            self._avg_volatility = avg_vol
        else:
            self._avg_volatility = 0.95 * self._avg_volatility + 0.05 * avg_vol
        self._current_volatility = current_vol

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

        # Timeout rate check (Item 2): if >30% of recent trades are timeouts, raise threshold
        now = time.monotonic()
        recent_timeouts = sum(1 for t in self._recent_timeouts if now - t < 3600)
        if len(pnls) > 0 and recent_timeouts / len(pnls) > 0.30:
            new_thresh = min(0.70, new_thresh + 0.02)
            logger.info(
                "High timeout rate (%d/%d in last hour) → threshold bumped to %.2f",
                recent_timeouts, len(pnls), new_thresh,
            )

        if new_thresh != self.dynamic_threshold:
            logger.info(
                "Threshold tuned: %.2f → %.2f (PF=%.2f, WR=%.1f%%)",
                self.dynamic_threshold, new_thresh, pf, wr * 100,
            )
            self.dynamic_threshold = new_thresh

    async def start(self) -> None:
        self._ws.on_user_data(self._on_user_data)
        await self._sync_balance()
        await self._recover_position()
        logger.info("Executor started, balance=$%.2f, state=%s", self.balance, self.state.value)

    async def _sync_balance(self) -> None:
        data = await self._ws.rest_get("/fapi/v2/balance", signed=True)
        if isinstance(data, list):
            for b in data:
                if b.get("asset") == "USDT":
                    self.balance = float(b.get("balance", 0))
                    self._risk.set_deposit(self.balance)
                    return

    async def _recover_position(self) -> None:
        """Recover open position state on startup."""
        data = await self._ws.rest_get("/fapi/v2/positionRisk", signed=True)
        if not isinstance(data, list):
            return

        for entry in data:
            if entry.get("symbol") != self._cfg.symbol:
                continue
            pos_amt = float(entry.get("positionAmt", 0))
            if pos_amt == 0:
                return

            # Recover position state
            self.state = State.IN_POSITION
            self._direction = Direction.LONG if pos_amt > 0 else Direction.SHORT
            self._size = abs(pos_amt)
            self._entry_price = float(entry.get("entryPrice", 0))
            self._position_time = time.monotonic()

            # Place backup SL (same pattern as in _enter)
            sl_side = "SELL" if self._direction == Direction.LONG else "BUY"
            sl_dist = self._entry_price * self._cfg.stop_loss_pct / 100
            if self._direction == Direction.LONG:
                sl_price = self._entry_price - sl_dist
            else:
                sl_price = self._entry_price + sl_dist
            sl_result = await self._ws.rest_post("/fapi/v1/order", params={
                "symbol": self._cfg.symbol,
                "side": sl_side,
                "type": "STOP_MARKET",
                "stopPrice": f"{sl_price:.2f}",
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            })
            if "orderId" in sl_result:
                self._sl_order_id = str(sl_result["orderId"])

            # Start position timeout task
            self._position_timeout_task = asyncio.create_task(
                self._position_timeout()
            )

            logger.info(
                "Recovered position: %s %.3f @ $%.2f (SL=$%.2f)",
                self._direction.value, self._size, self._entry_price, sl_price,
            )
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
        self._tp_target = signal.take_profit
        self._entry_submit_time = time.monotonic()

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

        # --- Adverse selection check (Item 16) ---
        fill_ms = (time.monotonic() - self._entry_submit_time) * 1000
        if fill_ms < 100:
            logger.warning("Fast fill %.0fms — adverse selection risk", fill_ms)
            if fill_ms < 50:
                logger.warning("Extremely fast fill — closing immediately")
                await self._close_position("adverse_selection")
                return

        logger.info(
            "Position opened: %s %.3f @ $%.2f",
            self._direction.value, self._size, fill_price,
        )
        self._notifier.trade(self._direction.value, fill_price, self._size)

        # Place take-profit
        tp_side = "SELL" if self._direction == Direction.LONG else "BUY"
        tp_price = self._tp_target
        if tp_price == 0.0:
            # Fallback: compute from config percentage
            tp_usd = fill_price * self._cfg.take_profit_pct / 100
            if self._direction == Direction.LONG:
                tp_price = fill_price + tp_usd
            else:
                tp_price = fill_price - tp_usd
            self._tp_target = tp_price

        # Trailing stop setup (Item 14)
        self._tp_dist = abs(tp_price - fill_price)
        self._trailing_sl = 0.0
        self._trailing_step = 0
        self._partial_filled = False
        self._partial_order_id = ""

        # Adverse selection: tighten SL for fast fills (50-100ms)
        if fill_ms < 100:
            # Tighten SL to 50% of normal distance
            await self._cancel_order(self._sl_order_id)
            sl_side = "SELL" if self._direction == Direction.LONG else "BUY"
            sl_dist = fill_price * self._cfg.stop_loss_pct / 100 * 0.5
            if self._direction == Direction.LONG:
                tight_sl = fill_price - sl_dist
            else:
                tight_sl = fill_price + sl_dist
            sl_result = await self._ws.rest_post("/fapi/v1/order", params={
                "symbol": self._cfg.symbol,
                "side": sl_side,
                "type": "STOP_MARKET",
                "stopPrice": f"{tight_sl:.2f}",
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            })
            if "orderId" in sl_result:
                self._sl_order_id = str(sl_result["orderId"])
                logger.info("Tightened SL (fast fill): $%.2f (id=%s)", tight_sl, self._sl_order_id)

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

        # Start trailing monitor (Item 14)
        self._trailing_task = asyncio.create_task(self._monitor_trailing())

        # Start position timeout
        self._position_timeout_task = asyncio.create_task(
            self._position_timeout()
        )

    async def _position_timeout(self) -> None:
        # Dynamic timeout based on volatility (Item 3)
        base_timeout = self._cfg.position_timeout_sec
        if self._current_volatility > 0 and self._avg_volatility > 0:
            timeout = base_timeout * (self._avg_volatility / max(self._current_volatility, 1e-10))
            timeout = max(15.0, min(timeout, 120.0))
        else:
            timeout = base_timeout

        await asyncio.sleep(timeout)
        if self.state != State.IN_POSITION:
            return

        logger.info("Position timeout (%.0fs), attempting limit close", timeout)

        # Item 1: Try limit close at mid price first
        # Cancel TP and SL
        await self._cancel_order(self._tp_order_id)
        await self._cancel_order(self._sl_order_id)

        # Get mid price from order book
        mid_price = None
        if self._ob is not None and self._ob.current is not None:
            mid_price = self._ob.current.mid_price

        if mid_price and mid_price > 0:
            close_side = "SELL" if self._direction == Direction.LONG else "BUY"
            limit_result = await self._ws.rest_post("/fapi/v1/order", params={
                "symbol": self._cfg.symbol,
                "side": close_side,
                "type": "LIMIT",
                "timeInForce": "GTX",
                "quantity": f"{self._size:.3f}",
                "price": f"{mid_price:.2f}",
                "newOrderRespType": "RESULT",
            })

            if "orderId" in limit_result:
                limit_order_id = str(limit_result["orderId"])
                logger.info("Timeout limit close at $%.2f (id=%s)", mid_price, limit_order_id)

                # Wait 2 seconds for fill
                await asyncio.sleep(2.0)

                if self.state == State.IN_POSITION:
                    # Not filled yet — cancel and fall back to market
                    await self._cancel_order(limit_order_id)
                    logger.info("Timeout limit not filled, falling back to market")
                else:
                    # Already closed (filled via _on_user_data)
                    return

        # Fallback: market close
        if self.state == State.IN_POSITION:
            self.state = State.EXITING
            close_side = "SELL" if self._direction == Direction.LONG else "BUY"
            result = await self._ws.rest_post("/fapi/v1/order", params={
                "symbol": self._cfg.symbol,
                "side": close_side,
                "type": "MARKET",
                "quantity": f"{self._size:.3f}",
                "newOrderRespType": "RESULT",
            })
            exit_price = float(result.get("avgPrice", 0))
            self._finalize_trade(exit_price, "timeout")

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

        # Track timeouts (Item 2)
        if "timeout" in reason:
            self._recent_timeouts.append(time.monotonic())

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
        # Reset trailing state (Item 14)
        if self._trailing_task:
            self._trailing_task.cancel()
            self._trailing_task = None
        self._tp_target = 0.0
        self._tp_dist = 0.0
        self._trailing_sl = 0.0
        self._trailing_step = 0
        self._partial_filled = False
        self._partial_order_id = ""

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

            # Partial TP filled (Item 15)
            elif order_id == self._partial_order_id and status == "FILLED":
                fill_price = float(order.get("ap", 0))
                self._size = round(self._size / 2, 3)
                self._partial_filled = True
                logger.info(
                    "Partial TP filled: closed 50%% at $%.2f, remaining=%.3f",
                    fill_price, self._size,
                )
                # Update the TP order quantity to match remaining size
                await self._cancel_order(self._tp_order_id)
                if self._tp_target > 0 and self._size > 0:
                    tp_side = "SELL" if self._direction == Direction.LONG else "BUY"
                    tp_result = await self._ws.rest_post("/fapi/v1/order", params={
                        "symbol": self._cfg.symbol,
                        "side": tp_side,
                        "type": "LIMIT",
                        "timeInForce": "GTX",
                        "quantity": f"{self._size:.3f}",
                        "price": f"{self._tp_target:.2f}",
                        "newOrderRespType": "RESULT",
                    })
                    if "orderId" in tp_result:
                        self._tp_order_id = str(tp_result["orderId"])

    async def _monitor_trailing(self) -> None:
        """Monitor price progress toward TP and adjust SL / partial TP (Items 14, 15)."""
        try:
            while self.state == State.IN_POSITION:
                await asyncio.sleep(0.1)
                if self._ob is None or self._ob.current is None:
                    continue

                current_price = self._ob.current.mid_price
                if self._tp_dist <= 0:
                    continue

                # Calculate progress toward TP
                if self._direction == Direction.LONG:
                    progress = (current_price - self._entry_price) / self._tp_dist
                else:
                    progress = (self._entry_price - current_price) / self._tp_dist

                # Step 1 at 50% progress: partial TP + move SL to profit
                if progress >= 0.5 and self._trailing_step < 1:
                    # Partial take-profit (Item 15)
                    if not self._partial_filled and self._size > 0:
                        half_size = round(self._size / 2, 3)
                        if half_size > 0:
                            close_side = "SELL" if self._direction == Direction.LONG else "BUY"
                            partial_result = await self._ws.rest_post("/fapi/v1/order", params={
                                "symbol": self._cfg.symbol,
                                "side": close_side,
                                "type": "LIMIT",
                                "timeInForce": "GTX",
                                "quantity": f"{half_size:.3f}",
                                "price": f"{current_price:.2f}",
                                "newOrderRespType": "RESULT",
                            })
                            if "orderId" in partial_result:
                                self._partial_order_id = str(partial_result["orderId"])
                                logger.info(
                                    "Partial TP order placed: %.3f @ $%.2f (id=%s)",
                                    half_size, current_price, self._partial_order_id,
                                )

                    # Move SL to max(+0.08%, 30% of TP distance)
                    min_sl_offset = self._entry_price * 0.0008
                    tp_30_offset = self._tp_dist * 0.3
                    sl_offset = max(min_sl_offset, tp_30_offset)
                    if self._direction == Direction.LONG:
                        new_sl = self._entry_price + sl_offset
                    else:
                        new_sl = self._entry_price - sl_offset
                    await self._update_trailing_sl(new_sl)
                    self._trailing_step = 1
                    logger.info(
                        "Trailing SL step 1: $%.2f (progress=%.0f%%)",
                        new_sl, progress * 100,
                    )

                # Step 2 at 75% progress: move SL to 50% of TP distance
                elif progress >= 0.75 and self._trailing_step < 2:
                    sl_offset = self._tp_dist * 0.5
                    if self._direction == Direction.LONG:
                        new_sl = self._entry_price + sl_offset
                    else:
                        new_sl = self._entry_price - sl_offset
                    await self._update_trailing_sl(new_sl)
                    self._trailing_step = 2
                    logger.info(
                        "Trailing SL step 2: $%.2f (progress=%.0f%%)",
                        new_sl, progress * 100,
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Trailing monitor error: %s", e)

    async def _update_trailing_sl(self, new_price: float) -> None:
        """Cancel existing SL and place new STOP_MARKET at new_price."""
        await self._cancel_order(self._sl_order_id)
        sl_side = "SELL" if self._direction == Direction.LONG else "BUY"
        sl_result = await self._ws.rest_post("/fapi/v1/order", params={
            "symbol": self._cfg.symbol,
            "side": sl_side,
            "type": "STOP_MARKET",
            "stopPrice": f"{new_price:.2f}",
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        })
        if "orderId" in sl_result:
            self._sl_order_id = str(sl_result["orderId"])
            self._trailing_sl = new_price

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
