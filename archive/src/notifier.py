from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.config import Config

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.constants import ParseMode

    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False


class Notifier:
    def __init__(self, config: Config) -> None:
        self._token = config.telegram_bot_token
        self._chat_id = config.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id and _HAS_TELEGRAM)
        self._bot: Bot | None = None
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)

        if self._enabled:
            self._bot = Bot(token=self._token)
            logger.info("Telegram notifier enabled")
        else:
            logger.warning("Telegram notifier disabled (missing token/chat_id or library)")

    async def start(self) -> None:
        if self._enabled:
            asyncio.create_task(self._sender_loop())

    async def _sender_loop(self) -> None:
        while True:
            msg = await self._queue.get()
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=msg,
                    parse_mode=ParseMode.HTML,
                )
            except Exception as e:
                logger.error("Telegram send failed: %s", e)
            self._queue.task_done()

    def _enqueue(self, msg: str) -> None:
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning("Telegram queue full, dropping message")

    def alert(self, title: str, body: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        msg = f"🚨 <b>{title}</b>\n{body}\n<i>{ts} UTC</i>"
        logger.warning("ALERT: %s — %s", title, body)
        self._enqueue(msg)

    def info(self, title: str, body: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        msg = f"ℹ️ <b>{title}</b>\n{body}\n<i>{ts} UTC</i>"
        logger.info("INFO: %s — %s", title, body)
        self._enqueue(msg)

    def trade(self, direction: str, price: float, size: float, pnl: float | None = None) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        pnl_str = f"  P&L: ${pnl:+.2f}" if pnl is not None else ""
        msg = (
            f"📊 <b>{direction}</b> {size} ETH @ ${price:.2f}{pnl_str}\n"
            f"<i>{ts} UTC</i>"
        )
        self._enqueue(msg)

    def daily_report(
        self,
        trades: int,
        wins: int,
        pnl: float,
        avg_win: float,
        avg_loss: float,
        max_consec_losses: int,
    ) -> None:
        wr = (wins / trades * 100) if trades > 0 else 0
        losses = trades - wins
        pf = (wins * avg_win / (losses * abs(avg_loss))) if losses > 0 and avg_loss != 0 else float("inf")
        msg = (
            f"📋 <b>Daily Report</b>\n"
            f"Trades: {trades}  |  WR: {wr:.1f}%\n"
            f"P&L: ${pnl:+.2f}  |  PF: {pf:.2f}\n"
            f"Avg win: ${avg_win:.2f}  |  Avg loss: ${avg_loss:.2f}\n"
            f"Max consec losses: {max_consec_losses}\n"
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d')} UTC</i>"
        )
        self._enqueue(msg)
