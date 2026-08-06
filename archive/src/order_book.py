from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

import numpy as np

from src.ws_client import BinanceWSClient

logger = logging.getLogger(__name__)

BOOK_DEPTH = 20
RING_SIZE = 100  # 10 seconds at 100ms
STALE_AFTER_SEC = 15.0  # if no update applied for this long, force resync
WATCHDOG_INTERVAL_SEC = 5.0
MAX_BUFFER_SIZE = 5000  # cap buffered diffs while unsynced (~1 min @ ~80 events/sec)


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
    def __init__(self, ws: BinanceWSClient, symbol: str = "ETHUSDT", *, secondary: bool = False) -> None:
        self._ws = ws
        self._symbol = symbol
        self._secondary = secondary
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_update_id: int = 0
        self._synced = False
        self._awaiting_first = True
        self._buffer: list[dict] = []
        self.ring: deque[Snapshot] = deque(maxlen=RING_SIZE)
        self.current: Snapshot | None = None
        self._on_snapshot = None  # callback after each snapshot update
        self._last_apply_ts: float = 0.0  # monotonic; updated on each applied diff
        self._resync_lock = asyncio.Lock()
        self._watchdog_task: asyncio.Task | None = None
        self._stopped = False

    def on_snapshot(self, cb) -> None:
        self._on_snapshot = cb

    async def start(self) -> None:
        if self._secondary:
            self._ws.on_secondary_depth(self._on_depth_update)
        else:
            self._ws.on_depth(self._on_depth_update)
        await self._fetch_snapshot()
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._watchdog_task:
            self._watchdog_task.cancel()

    async def _fetch_snapshot(self) -> None:
        """Fetch REST snapshot with retries. Never raises — guarantees the
        OrderBook either becomes synced or stays in a recoverable unsynced state.
        """
        async with self._resync_lock:
            self._synced = False
            self._awaiting_first = True
            label = "ETH" if self._secondary else "BTC"
            logger.info("Fetching %s order book snapshot...", label)

            backoff = 0.5
            for attempt in range(1, 11):
                try:
                    data = await self._ws.rest_get(
                        "/fapi/v1/depth",
                        params={"symbol": self._symbol, "limit": BOOK_DEPTH},
                    )
                    if not isinstance(data, dict) or "lastUpdateId" not in data:
                        raise RuntimeError(f"unexpected snapshot payload: {data!r}")

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

                    self._last_update_id = int(data.get("lastUpdateId", 0))
                    self._synced = True
                    self._last_apply_ts = time.monotonic()
                    logger.info(
                        "%s snapshot loaded: %d bids, %d asks, lastUpdateId=%d",
                        label, len(self._bids), len(self._asks), self._last_update_id,
                    )
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        "%s snapshot fetch attempt %d failed: %r", label, attempt, e
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
            else:
                logger.error(
                    "%s snapshot fetch exhausted retries; will retry from watchdog",
                    label,
                )
                return

            # Apply buffered updates newer than snapshot
            applied = 0
            for msg in self._buffer:
                u_last = msg.get("u", 0)
                if u_last <= self._last_update_id:
                    continue
                self._apply_update(msg)
                self._awaiting_first = False
                applied += 1
            self._buffer.clear()
            if applied:
                logger.info("%s applied %d buffered updates", label, applied)

            self._build_snapshot(int(time.time() * 1000))

    async def _on_depth_update(self, data: dict) -> None:
        if not self._synced:
            # Buffer with cap so we never grow unbounded if resync stalls
            if len(self._buffer) < MAX_BUFFER_SIZE:
                self._buffer.append(data)
            return

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
            # Throttle: if a resync is already in progress, skip — don't queue
            # a parade of redundant fetches when many out-of-sync events arrive
            # in the same window after a reconnect.
            if self._resync_lock.locked():
                return
            logger.warning(
                "%s order book desync: expected pu=%d, got pu=%d. Re-fetching.",
                "ETH" if self._secondary else "BTC",
                self._last_update_id, prev_id,
            )
            self._buffer.clear()
            # Fire and forget — never let this propagate up and break the WS loop
            asyncio.create_task(self._fetch_snapshot())
            return

        self._apply_update(data)
        ts = data.get("E", int(time.time() * 1000))
        self._build_snapshot(ts)

    async def _watchdog_loop(self) -> None:
        """Periodically detect stalled book and force resync.

        Triggers re-fetch if no diff has been applied for STALE_AFTER_SEC,
        or if we have buffered diffs but never managed to sync.
        """
        label = "ETH" if self._secondary else "BTC"
        while not self._stopped:
            try:
                await asyncio.sleep(WATCHDOG_INTERVAL_SEC)
                now = time.monotonic()
                stalled = (
                    self._last_apply_ts > 0
                    and (now - self._last_apply_ts) > STALE_AFTER_SEC
                )
                stuck_unsynced = (not self._synced) and len(self._buffer) > 0
                if stalled or stuck_unsynced:
                    # Skip if resync already running — the in-progress fetch
                    # will recover us; spamming new tasks just queues work.
                    if self._resync_lock.locked():
                        continue
                    logger.warning(
                        "%s book watchdog: stalled=%s stuck_unsynced=%s "
                        "(idle=%.1fs synced=%s buf=%d) — forcing resync",
                        label, stalled, stuck_unsynced,
                        (now - self._last_apply_ts) if self._last_apply_ts else -1,
                        self._synced, len(self._buffer),
                    )
                    asyncio.create_task(self._fetch_snapshot())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("%s watchdog error: %r", label, e)

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
        self._last_apply_ts = time.monotonic()

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
