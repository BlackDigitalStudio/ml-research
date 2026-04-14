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
PARTS_SUBDIR = ".parts"
DEPTH_LEVELS = 20


def _snapshot_to_flat_row(snapshot) -> dict:
    """Convert an OrderBook Snapshot to flat columnar row.

    Snapshot.bids / .asks are numpy arrays (DEPTH_LEVELS, 2) already sorted
    and zero-padded (qty=0 for unused slots). We copy directly into four
    fixed-length f64 vectors so downstream Arrow can store them as
    FixedSizeList<f64, 20> with zero post-processing.
    """
    bids = np.asarray(snapshot.bids, dtype=np.float64)
    asks = np.asarray(snapshot.asks, dtype=np.float64)
    bp = np.zeros(DEPTH_LEVELS, dtype=np.float64)
    bq = np.zeros(DEPTH_LEVELS, dtype=np.float64)
    ap = np.zeros(DEPTH_LEVELS, dtype=np.float64)
    aq = np.zeros(DEPTH_LEVELS, dtype=np.float64)
    n_b = min(DEPTH_LEVELS, bids.shape[0])
    n_a = min(DEPTH_LEVELS, asks.shape[0])
    if n_b:
        bp[:n_b] = bids[:n_b, 0]
        bq[:n_b] = bids[:n_b, 1]
    if n_a:
        ap[:n_a] = asks[:n_a, 0]
        aq[:n_a] = asks[:n_a, 1]
    return {
        "timestamp": int(snapshot.timestamp),
        "bid_prices": bp,
        "bid_qtys": bq,
        "ask_prices": ap,
        "ask_qtys": aq,
    }


def _build_flat_depth_table(rows: list[dict]) -> pa.Table:
    """Assemble FixedSizeList<f64,20> depth table from flat rows."""
    n = len(rows)
    ts = np.empty(n, dtype=np.int64)
    bp = np.empty((n, DEPTH_LEVELS), dtype=np.float64)
    bq = np.empty((n, DEPTH_LEVELS), dtype=np.float64)
    ap = np.empty((n, DEPTH_LEVELS), dtype=np.float64)
    aq = np.empty((n, DEPTH_LEVELS), dtype=np.float64)
    for i, r in enumerate(rows):
        ts[i] = r["timestamp"]
        bp[i] = r["bid_prices"]
        bq[i] = r["bid_qtys"]
        ap[i] = r["ask_prices"]
        aq[i] = r["ask_qtys"]

    def _fsl(flat: np.ndarray) -> pa.Array:
        return pa.FixedSizeListArray.from_arrays(
            pa.array(flat.reshape(-1), type=pa.float64()), DEPTH_LEVELS
        )

    return pa.table({
        "timestamp":  pa.array(ts, type=pa.int64()),
        "bid_prices": _fsl(bp),
        "bid_qtys":   _fsl(bq),
        "ask_prices": _fsl(ap),
        "ask_qtys":   _fsl(aq),
    })


class Recorder:
    """Crash-safe append-mode recorder.

    Each flush writes a small immutable *part* file under `<stream>/.parts/`.
    On hour rollover (or process restart), parts for completed hours are
    compacted into a single canonical `<stream>/<hour_key>.parquet` and the
    parts deleted.

    Memory is bounded by 1 flush interval (≤ ~60s) of buffered events per
    stream — no read-back-and-rewrite of growing files. Existing canonical
    files are never re-read during normal operation.
    """

    def __init__(self, config: Config) -> None:
        self._data_dir = config.data_dir
        self._depth_dir = self._data_dir / "depth"
        self._trades_dir = self._data_dir / "trades"
        self._bot_trades_dir = self._data_dir / "bot_trades"
        self._bybit_dir = self._data_dir / "bybit_trades"
        self._eth_trades_dir = self._data_dir / "eth_trades"
        self._eth_depth_dir = self._data_dir / "eth_depth"
        self._funding_dir = self._data_dir / "funding"
        self._derivatives_dir = self._data_dir / "derivatives"
        # Cross-exchange trade directories (3 exchanges).
        # HTX and Deribit removed due to structural instability — see
        # ws_client.py BinanceWSClient.start() comment.
        self._exchange_dirs: dict[str, Path] = {}
        for ex in ("okx", "bitget", "gateio"):
            self._exchange_dirs[ex] = self._data_dir / f"{ex}_trades"

        # All hourly-rotated dirs (everything except bot_trades which is daily)
        self._hourly_dirs: list[Path] = [
            self._depth_dir, self._trades_dir,
            self._bybit_dir, self._eth_trades_dir, self._eth_depth_dir,
            self._funding_dir, self._derivatives_dir,
        ] + list(self._exchange_dirs.values())

        all_dirs = self._hourly_dirs + [self._bot_trades_dir]
        for d in all_dirs:
            d.mkdir(parents=True, exist_ok=True)
        for d in self._hourly_dirs:
            (d / PARTS_SUBDIR).mkdir(parents=True, exist_ok=True)

        self._depth_buf: list[dict] = []
        self._trade_buf: list[dict] = []
        self._bot_trade_buf: list[dict] = []
        self._bybit_buf: list[dict] = []
        self._eth_trade_buf: list[dict] = []
        self._eth_depth_buf: list[dict] = []
        self._funding_buf: list[dict] = []
        self._derivatives_buf: list[dict] = []
        self._exchange_bufs: dict[str, list[dict]] = {ex: [] for ex in self._exchange_dirs}

        # Dedup ring: (timestamp, price, qty, side) → skip if seen recently
        self._trade_dedup: dict[str, set] = {}

        # Per-stream sequence counters for part filenames (reset on hour change)
        self._part_seq: dict[str, int] = defaultdict(int)
        self._current_hour_key: str = ""

        self._flush_task: asyncio.Task | None = None
        self._rotation_task: asyncio.Task | None = None

    # ---- public dedup helper ----
    def _dedup_trade(self, source: str, ts: int, price: float, qty: float, side: bool) -> bool:
        key = (ts, price, qty, side)
        s = self._trade_dedup.setdefault(source, set())
        if key in s:
            return True
        s.add(key)
        if len(s) > 50_000:
            # Prune to bound memory; set ordering is not guaranteed but a
            # rough trim is sufficient for short-window dedup.
            self._trade_dedup[source] = set(list(s)[-25_000:])
        return False

    # ---- lifecycle ----
    async def start(self) -> None:
        # Recover any orphaned part files from a previous run
        self._recover_orphan_parts()
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        logger.info("Recorder started (depth=%s, trades=%s)", self._depth_dir, self._trades_dir)

    async def stop(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
        if self._rotation_task:
            self._rotation_task.cancel()
        try:
            self._flush_all()
        except Exception as e:
            logger.error("Final flush failed: %r", e)
        # Compact whatever the current hour has
        if self._current_hour_key:
            for d in self._hourly_dirs:
                try:
                    self._compact_hour(d, self._current_hour_key)
                except Exception as e:
                    logger.error("Final compact %s/%s failed: %r", d.name, self._current_hour_key, e)

    # ---- record_* (called from WS callbacks) ----
    def record_depth(self, snapshot) -> None:
        self._depth_buf.append(_snapshot_to_flat_row(snapshot))

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
        ts = int(data.get("T", 0))
        price = float(data.get("p", 0))
        qty = float(data.get("q", 0))
        side = data.get("m", False)
        if self._dedup_trade("bybit", ts, price, qty, side):
            return
        self._bybit_buf.append({"timestamp": ts, "price": price, "quantity": qty, "is_seller": side})

    def record_eth_trade(self, data: dict) -> None:
        self._eth_trade_buf.append({
            "timestamp": data.get("T", int(time.time() * 1000)),
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "is_buyer_maker": data.get("m", False),
        })

    def record_eth_depth(self, snapshot) -> None:
        self._eth_depth_buf.append(_snapshot_to_flat_row(snapshot))

    def record_funding(self, data: dict) -> None:
        self._funding_buf.append({
            "timestamp": int(time.time() * 1000),
            "funding_rate": float(data.get("r", 0)),
            "mark_price": float(data.get("p", 0)),
        })

    def record_derivatives(self, oi: float, ls_ratio: float) -> None:
        self._derivatives_buf.append({
            "timestamp": int(time.time() * 1000),
            "open_interest": oi,
            "long_short_ratio": ls_ratio,
        })

    def record_exchange_trade(self, data: dict) -> None:
        ex = data.get("exchange", "")
        if ex in self._exchange_bufs:
            ts = int(data.get("T", 0))
            price = float(data.get("p", 0))
            qty = float(data.get("q", 0))
            side = data.get("m", False)
            if self._dedup_trade(ex, ts, price, qty, side):
                return
            self._exchange_bufs[ex].append({"timestamp": ts, "price": price, "quantity": qty, "is_seller": side})

    # ---- flush / part writers ----
    async def _flush_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL)
                self._flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("flush_loop error: %r", e)

    def _flush_all(self) -> None:
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y%m%d_%H")

        # Hour rollover: compact prior hour's parts into one canonical file
        if self._current_hour_key and hour_key != self._current_hour_key:
            prior = self._current_hour_key
            for d in self._hourly_dirs:
                try:
                    self._compact_hour(d, prior)
                except Exception as e:
                    logger.error("compact %s/%s failed: %r", d.name, prior, e)
            self._part_seq.clear()
        self._current_hour_key = hour_key

        # Drain each buffer to a flat-or-depth-shaped part file
        if self._depth_buf:
            self._write_depth_part(self._depth_dir, hour_key, self._depth_buf)
            self._depth_buf = []
        if self._trade_buf:
            self._write_flat_part(self._trades_dir, hour_key, self._trade_buf)
            self._trade_buf = []
        if self._bybit_buf:
            self._write_flat_part(self._bybit_dir, hour_key, self._bybit_buf)
            self._bybit_buf = []
        if self._eth_trade_buf:
            self._write_flat_part(self._eth_trades_dir, hour_key, self._eth_trade_buf)
            self._eth_trade_buf = []
        if self._eth_depth_buf:
            self._write_depth_part(self._eth_depth_dir, hour_key, self._eth_depth_buf)
            self._eth_depth_buf = []
        if self._funding_buf:
            self._write_flat_part(self._funding_dir, hour_key, self._funding_buf)
            self._funding_buf = []
        if self._derivatives_buf:
            self._write_flat_part(self._derivatives_dir, hour_key, self._derivatives_buf)
            self._derivatives_buf = []
        for ex, buf in list(self._exchange_bufs.items()):
            if buf:
                self._write_flat_part(self._exchange_dirs[ex], hour_key, buf)
                self._exchange_bufs[ex] = []

        # bot_trades is daily, low volume — keep simple read-modify-write
        if self._bot_trade_buf:
            day_key = now.strftime("%Y%m%d")
            self._write_bot_trades(day_key)

    def _next_part_path(self, directory: Path, hour_key: str) -> Path:
        key = f"{directory.name}/{hour_key}"
        seq = self._part_seq[key]
        self._part_seq[key] = seq + 1
        return directory / PARTS_SUBDIR / f"{hour_key}_{seq:05d}.parquet"

    def _atomic_write(self, table: pa.Table, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp, compression="snappy")
        tmp.replace(path)

    def _write_flat_part(self, directory: Path, hour_key: str, rows: list[dict]) -> None:
        try:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df, preserve_index=False)
            path = self._next_part_path(directory, hour_key)
            self._atomic_write(table, path)
        except Exception as e:
            logger.error("write part failed for %s: %r", directory.name, e)

    def _write_depth_part(self, directory: Path, hour_key: str, rows: list[dict]) -> None:
        try:
            table = _build_flat_depth_table(rows)
            path = self._next_part_path(directory, hour_key)
            self._atomic_write(table, path)
        except Exception as e:
            logger.error("write depth part failed for %s: %r", directory.name, e)

    def _write_bot_trades(self, day_key: str) -> None:
        # Bot trades are daily and low-volume; legacy read-modify-write is fine.
        rows = self._bot_trade_buf
        self._bot_trade_buf = []
        try:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df, preserve_index=False)
            path = self._bot_trades_dir / f"{day_key}.parquet"
            if path.exists():
                existing = pq.read_table(path)
                table = pa.concat_tables([existing, table])
            self._atomic_write(table, path)
        except Exception as e:
            logger.error("write bot_trades failed: %r", e)

    # ---- compaction & recovery ----
    def _compact_hour(self, directory: Path, hour_key: str) -> None:
        parts_dir = directory / PARTS_SUBDIR
        if not parts_dir.exists():
            return
        parts = sorted(parts_dir.glob(f"{hour_key}_*.parquet"))
        if not parts:
            return
        out = directory / f"{hour_key}.parquet"
        try:
            tables = []
            for p in parts:
                try:
                    tables.append(pq.read_table(p))
                except Exception as e:
                    logger.error("compact: failed to read %s: %r", p, e)
            if not tables:
                return
            # If a canonical file already exists (e.g. recovery after crash within
            # the same hour), include it so we keep prior data.
            if out.exists():
                try:
                    tables.insert(0, pq.read_table(out))
                except Exception as e:
                    logger.error("compact: failed to read existing %s: %r", out, e)
            merged = pa.concat_tables(tables, promote_options="default")
            self._atomic_write(merged, out)
        except Exception as e:
            logger.error("compact %s/%s failed: %r", directory.name, hour_key, e)
            return
        # Delete parts only after successful write
        for p in parts:
            try:
                p.unlink()
            except Exception as e:
                logger.error("compact: failed to delete %s: %r", p, e)
        logger.info("Compacted %d parts → %s/%s.parquet", len(parts), directory.name, hour_key)

    def _recover_orphan_parts(self) -> None:
        """On startup, compact any leftover parts from a prior run."""
        for d in self._hourly_dirs:
            parts_dir = d / PARTS_SUBDIR
            if not parts_dir.exists():
                continue
            # Clean stale .tmp from interrupted writes
            for tmp in parts_dir.glob("*.tmp"):
                try:
                    tmp.unlink()
                except Exception:
                    pass
            groups: dict[str, list[Path]] = defaultdict(list)
            for p in parts_dir.glob("*.parquet"):
                # filename: <hour_key>_<seq>.parquet → hour_key has format YYYYMMDD_HH
                stem = p.stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    hour_key = parts[0]
                    groups[hour_key].append(p)
            for hour_key in sorted(groups):
                try:
                    self._compact_hour(d, hour_key)
                except Exception as e:
                    logger.error("recovery compact %s/%s failed: %r", d.name, hour_key, e)

    # ---- retention ----
    async def _rotation_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(3600)
                self._rotate_old_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("rotation_loop error: %r", e)

    def _rotate_old_files(self) -> None:
        cutoff = time.time() - RETENTION_HOURS * 3600
        count = 0
        for d in self._hourly_dirs:
            for f in d.iterdir():
                if f.is_file() and f.suffix == ".parquet" and f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        count += 1
                    except Exception as e:
                        logger.error("rotate: failed to unlink %s: %r", f, e)
            # Also drop very old orphan parts (shouldn't happen normally)
            parts_dir = d / PARTS_SUBDIR
            if parts_dir.exists():
                for f in parts_dir.iterdir():
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        try:
                            f.unlink()
                            count += 1
                        except Exception as e:
                            logger.error("rotate: failed to unlink part %s: %r", f, e)
        if count:
            logger.info("Rotated %d old parquet files (>%dh)", count, RETENTION_HOURS)
