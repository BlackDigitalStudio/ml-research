from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

from src.ws_client import BinanceWSClient

logger = logging.getLogger(__name__)

BOOK_DEPTH = 20
RING_SIZE = 100  # 10 seconds at 100ms


class Snapshot:
    __slots__ = ("bids", "asks", "timestamp", "update_id")

    def __init__(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        timestamp: int,
        update_id: int,
    ) -> None:
        self.bids = bids          # (20, 2) — [price, qty], sorted desc by price
        self.asks = asks          # (20, 2) — [price, qty], sorted asc by price
        self.timestamp = timestamp
        self.update_id = update_id

    @property
    def best_bid(self) -> float:
        return float(self.bids[0, 0]) if len(self.bids) > 0 else 0.0

    @property
    def best_ask(self) -> float:
        return float(self.asks[0, 0]) if len(self.asks) > 0 else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


class OrderBook:
    def __init__(self, ws: BinanceWSClient, symbol: str = "ETHUSDT") -> None:
        self._ws = ws
        self._symbol = symbol
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_update_id: int = 0
        self._synced = False
        self._buffer: list[dict] = []
        self.ring: deque[Snapshot] = deque(maxlen=RING_SIZE)
        self.current: Snapshot | None = None
        self._on_snapshot = None  # callback after each snapshot update

    def on_snapshot(self, cb) -> None:
        self._on_snapshot = cb

    async def start(self) -> None:
        self._ws.on_depth(self._on_depth_update)
        await self._fetch_snapshot()

    async def _fetch_snapshot(self) -> None:
        self._synced = False
        self._awaiting_first = True
        logger.info("Fetching order book snapshot...")
        data = await self._ws.rest_get(
            "/fapi/v1/depth",
            params={"symbol": self._symbol, "limit": BOOK_DEPTH},
        )
        self._bids.clear()
        self._asks.clear()

        for price_s, qty_s in data.get("bids", []):
            p, q = float(price_s), float(qty_s)
            if q > 0:
                self._bids[p] = q

        for price_s, qty_s in data.get("asks", []):
            p, q = float(price_s), float(qty_s)
            if q > 0:
                self._asks[p] = q

        self._last_update_id = data.get("lastUpdateId", 0)
        self._synced = True
        logger.info(
            "Snapshot loaded: %d bids, %d asks, lastUpdateId=%d",
            len(self._bids), len(self._asks), self._last_update_id,
        )

        # Apply buffered updates that are newer than snapshot
        applied = 0
        for msg in self._buffer:
            u_last = msg.get("u", 0)
            if u_last <= self._last_update_id:
                continue  # skip stale
            self._apply_update(msg)
            self._awaiting_first = False
            applied += 1
        self._buffer.clear()
        if applied:
            logger.info("Applied %d buffered updates", applied)

        self._build_snapshot(int(time.time() * 1000))

    async def _on_depth_update(self, data: dict) -> None:
        if not self._synced:
            self._buffer.append(data)
            return

        first_id = data.get("U", 0)
        final_id = data.get("u", 0)

        # Drop stale events
        if final_id <= self._last_update_id:
            return

        # First event after snapshot: accept any event newer than snapshot
        if self._awaiting_first:
            self._awaiting_first = False
            self._apply_update(data)
            ts = data.get("E", int(time.time() * 1000))
            self._build_snapshot(ts)
            return

        # Normal continuity check via pu
        prev_id = data.get("pu", 0)
        if prev_id != self._last_update_id:
            logger.warning(
                "Order book desync: expected pu=%d, got pu=%d. Re-fetching.",
                self._last_update_id, prev_id,
            )
            self._buffer.clear()
            await self._fetch_snapshot()
            return

        self._apply_update(data)
        ts = data.get("E", int(time.time() * 1000))
        self._build_snapshot(ts)

    def _apply_update(self, data: dict) -> None:
        for price_s, qty_s in data.get("b", []):
            p, q = float(price_s), float(qty_s)
            if q == 0:
                self._bids.pop(p, None)
            else:
                self._bids[p] = q

        for price_s, qty_s in data.get("a", []):
            p, q = float(price_s), float(qty_s)
            if q == 0:
                self._asks.pop(p, None)
            else:
                self._asks[p] = q

        self._last_update_id = data.get("u", self._last_update_id)

    def _build_snapshot(self, timestamp: int) -> None:
        # Top 20 bids (descending by price)
        sorted_bids = sorted(self._bids.items(), key=lambda x: -x[0])[:BOOK_DEPTH]
        # Top 20 asks (ascending by price)
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])[:BOOK_DEPTH]

        bids = np.array(sorted_bids, dtype=np.float64) if sorted_bids else np.zeros((0, 2))
        asks = np.array(sorted_asks, dtype=np.float64) if sorted_asks else np.zeros((0, 2))

        # Pad to BOOK_DEPTH if fewer levels
        if len(bids) < BOOK_DEPTH:
            pad = np.zeros((BOOK_DEPTH - len(bids), 2))
            bids = np.vstack([bids, pad]) if len(bids) > 0 else pad
        if len(asks) < BOOK_DEPTH:
            pad = np.zeros((BOOK_DEPTH - len(asks), 2))
            asks = np.vstack([asks, pad]) if len(asks) > 0 else pad

        snap = Snapshot(
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            update_id=self._last_update_id,
        )
        self.current = snap
        self.ring.append(snap)

        if self._on_snapshot is not None:
            self._on_snapshot(snap)

    @property
    def is_synced(self) -> bool:
        return self._synced and self.current is not None
