from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import Config

logger = logging.getLogger(__name__)

FLUSH_INTERVAL = 60  # flush to disk every 60 seconds
RETENTION_HOURS = 72


class Recorder:
    def __init__(self, config: Config) -> None:
        self._data_dir = config.data_dir
        self._depth_dir = self._data_dir / "depth"
        self._trades_dir = self._data_dir / "trades"
        self._bot_trades_dir = self._data_dir / "bot_trades"
        self._bybit_dir = self._data_dir / "bybit_trades"

        for d in (self._depth_dir, self._trades_dir, self._bot_trades_dir, self._bybit_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._depth_buf: list[dict] = []
        self._trade_buf: list[dict] = []
        self._bot_trade_buf: list[dict] = []
        self._bybit_buf: list[dict] = []
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._flush_task = asyncio.create_task(self._flush_loop())
        asyncio.create_task(self._rotation_loop())
        logger.info("Recorder started (depth=%s, trades=%s)", self._depth_dir, self._trades_dir)

    async def stop(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
        self._flush_all()

    def record_depth(self, snapshot) -> None:
        """Record a full OrderBook snapshot (not raw WS diff).

        Args:
            snapshot: order_book.Snapshot with .bids (20,2), .asks (20,2), .timestamp
        """
        bids = [(float(snapshot.bids[i, 0]), float(snapshot.bids[i, 1]))
                for i in range(len(snapshot.bids)) if snapshot.bids[i, 1] > 0]
        asks = [(float(snapshot.asks[i, 0]), float(snapshot.asks[i, 1]))
                for i in range(len(snapshot.asks)) if snapshot.asks[i, 1] > 0]
        self._depth_buf.append({
            "timestamp": snapshot.timestamp,
            "bids": bids,
            "asks": asks,
        })

    def record_trade(self, data: dict) -> None:
        self._trade_buf.append({
            "timestamp": data.get("T", int(time.time() * 1000)),
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "is_buyer_maker": data.get("m", False),
        })

    def record_bot_trade(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        fees: float,
        duration_sec: float,
        reason: str,
    ) -> None:
        self._bot_trade_buf.append({
            "timestamp": int(time.time() * 1000),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "fees": fees,
            "duration_sec": duration_sec,
            "reason": reason,
        })

    def record_bybit_trade(self, data: dict) -> None:
        self._bybit_buf.append({
            "timestamp": int(data.get("T", 0)),
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "is_seller": data.get("m", False),
        })

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            self._flush_all()

    def _flush_all(self) -> None:
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y%m%d_%H")

        if self._depth_buf:
            self._write_depth(hour_key)
        if self._trade_buf:
            self._write_trades(hour_key)
        if self._bot_trade_buf:
            day_key = now.strftime("%Y%m%d")
            self._write_bot_trades(day_key)
        if self._bybit_buf:
            self._write_bybit_trades(hour_key)

    def _write_depth(self, hour_key: str) -> None:
        rows = self._depth_buf
        self._depth_buf = []

        timestamps = [r["timestamp"] for r in rows]
        bids = [r["bids"] for r in rows]
        asks = [r["asks"] for r in rows]

        table = pa.table({
            "timestamp": pa.array(timestamps, type=pa.int64()),
            "bids": pa.array(bids),
            "asks": pa.array(asks),
        })

        path = self._depth_dir / f"{hour_key}.parquet"
        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")
        logger.debug("Flushed %d depth records to %s", len(rows), path.name)

    def _write_trades(self, hour_key: str) -> None:
        rows = self._trade_buf
        self._trade_buf = []

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        path = self._trades_dir / f"{hour_key}.parquet"
        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")
        logger.debug("Flushed %d trade records to %s", len(rows), path.name)

    def _write_bot_trades(self, day_key: str) -> None:
        rows = self._bot_trade_buf
        self._bot_trade_buf = []

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        path = self._bot_trades_dir / f"{day_key}.parquet"
        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")
        logger.debug("Flushed %d bot trade records to %s", len(rows), path.name)

    def _write_bybit_trades(self, hour_key: str) -> None:
        rows = self._bybit_buf
        self._bybit_buf = []

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        path = self._bybit_dir / f"{hour_key}.parquet"
        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")
        logger.debug("Flushed %d bybit trade records to %s", len(rows), path.name)

    async def _rotation_loop(self) -> None:
        while True:
            await asyncio.sleep(3600)  # check every hour
            self._rotate_old_files()

    def _rotate_old_files(self) -> None:
        cutoff = time.time() - RETENTION_HOURS * 3600
        count = 0
        for d in (self._depth_dir, self._trades_dir, self._bybit_dir):
            for f in d.iterdir():
                if f.suffix == ".parquet" and f.stat().st_mtime < cutoff:
                    f.unlink()
                    count += 1
        if count:
            logger.info("Rotated %d old parquet files (>%dh)", count, RETENTION_HOURS)
