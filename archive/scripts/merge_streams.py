#!/usr/bin/env python3
"""Streaming merge of per-hour/day parquets into `data/_merged/*.parquet`.

Prerequisite for the direct-Rust sample-build path (see
src/trainer.py::_build_samples_rust_direct). Runs **once** per data
addition, then repeated training iterations consume the merged files.

Memory budget: one source file loaded at a time (~100 MB peak). The
merged file is written via `pyarrow.ParquetWriter` append, so no
materialisation of the full merged table happens in RAM.

Usage:
    # merge every stream from the full recorder corpus
    python scripts/merge_streams.py --data-dir /home/scalper/scalper-bot/data

    # merge only the last 100 hourly files per stream (skip old data)
    python scripts/merge_streams.py --data-dir ./data --hours 100

    # merge a specific stream
    python scripts/merge_streams.py --data-dir ./data --stream depth
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("merge_streams")

# Per-stream source directory names and whether the target parquet has a
# flat depth schema (vs scalar trades/funding/derivs).
STREAMS = {
    "depth":  {"dir": "depth",         "schema": "depth"},
    "trades": {"dir": "trades",        "schema": "trades"},
    "eth":    {"dir": "eth_trades",    "schema": "trades"},
    "bybit":  {"dir": "bybit_trades",  "schema": "cross_ex"},
    "okx":    {"dir": "okx_trades",    "schema": "cross_ex"},
    "bitget": {"dir": "bitget_trades", "schema": "cross_ex"},
    "gateio": {"dir": "gateio_trades", "schema": "cross_ex"},
    "funding":  {"dir": "funding",      "schema": "funding"},
    "derivs":   {"dir": "derivatives",  "schema": "derivs"},
}


def _make_schema(kind: str) -> pa.Schema:
    if kind == "depth":
        fsl = pa.list_(pa.float64(), 20)
        return pa.schema([
            ("timestamp",   pa.int64()),
            ("bid_prices",  fsl),
            ("bid_qtys",    fsl),
            ("ask_prices",  fsl),
            ("ask_qtys",    fsl),
        ])
    if kind == "trades":
        return pa.schema([
            ("timestamp",      pa.int64()),
            ("price",          pa.float64()),
            ("quantity",       pa.float64()),
            ("is_buyer_maker", pa.bool_()),
        ])
    if kind == "cross_ex":
        return pa.schema([
            ("timestamp", pa.int64()),
            ("price",     pa.float64()),
            ("quantity",  pa.float64()),
            ("is_seller", pa.bool_()),
        ])
    if kind == "funding":
        return pa.schema([
            ("timestamp",    pa.int64()),
            ("funding_rate", pa.float64()),
            ("mark_price",   pa.float64()),
        ])
    if kind == "derivs":
        return pa.schema([
            ("timestamp",         pa.int64()),
            ("open_interest",     pa.float64()),
            ("long_short_ratio",  pa.float64()),
        ])
    raise ValueError(f"unknown kind: {kind}")


def _depth_table_to_flat_chunk(t: pa.Table) -> pa.Table:
    """Convert a depth parquet to the flat FixedSizeList schema. Handles
    both legacy (list-of-[price, qty]) and already-flat input."""
    names = set(t.schema.names)
    if {"bid_prices", "bid_qtys", "ask_prices", "ask_qtys"} <= names:
        # Already flat — just cast timestamp if needed.
        return t.select(["timestamp", "bid_prices", "bid_qtys",
                          "ask_prices", "ask_qtys"])
    raise ValueError(
        f"Legacy depth schema ({sorted(names)}) — run migrate_legacy_depth.py first"
    )


def _trades_table(t: pa.Table) -> pa.Table:
    # Recorder writes some files with `is_buyer_maker` and some files
    # without a price column (old). Fill defensively.
    cols = {"timestamp": t["timestamp"].cast(pa.int64()),
            "price":     t["price"].cast(pa.float64()) if "price" in t.schema.names
                         else pa.array(np.zeros(t.num_rows), type=pa.float64()),
            "quantity":  t["quantity"].cast(pa.float64()),
            "is_buyer_maker": t["is_buyer_maker"].cast(pa.bool_()),
            }
    return pa.table(cols)


def _cross_ex_table(t: pa.Table) -> pa.Table:
    names = set(t.schema.names)
    side_col = "is_seller" if "is_seller" in names else "is_buyer_maker"
    cols = {"timestamp": t["timestamp"].cast(pa.int64()),
            "price":     t["price"].cast(pa.float64()) if "price" in names
                         else pa.array(np.zeros(t.num_rows), type=pa.float64()),
            "quantity":  t["quantity"].cast(pa.float64()),
            "is_seller": t[side_col].cast(pa.bool_()),
            }
    return pa.table(cols)


def _funding_table(t: pa.Table) -> pa.Table:
    names = set(t.schema.names)
    mark = t["mark_price"].cast(pa.float64()) if "mark_price" in names \
           else pa.array(np.zeros(t.num_rows), type=pa.float64())
    return pa.table({
        "timestamp":    t["timestamp"].cast(pa.int64()),
        "funding_rate": t["funding_rate"].cast(pa.float64()),
        "mark_price":   mark,
    })


def _derivs_table(t: pa.Table) -> pa.Table:
    return pa.table({
        "timestamp":        t["timestamp"].cast(pa.int64()),
        "open_interest":    t["open_interest"].cast(pa.float64()),
        "long_short_ratio": t["long_short_ratio"].cast(pa.float64()),
    })


def merge_stream(
    source_files: Iterable[Path],
    out_path: Path,
    kind: str,
) -> tuple[int, float]:
    """Stream-merge `source_files` into `out_path`. Returns (rows, seconds)."""
    schema = _make_schema(kind)
    t_start = time.monotonic()
    total_rows = 0

    # Accumulate files, sort by timestamp once at the end? Simpler: sort
    # files by name (recorder names are YYYYMMDD_HH so filename order ==
    # time order) and write in order. Within each file sort by timestamp
    # to repair any intra-file disorder.
    files = sorted(source_files)
    if not files:
        log.warning("no source files for %s", out_path)
        return 0, 0.0

    with pq.ParquetWriter(str(out_path), schema, compression="snappy") as w:
        for i, f in enumerate(files):
            t = pq.read_table(f)
            if kind == "depth":
                t = _depth_table_to_flat_chunk(t)
            elif kind == "trades":
                t = _trades_table(t)
            elif kind == "cross_ex":
                t = _cross_ex_table(t)
            elif kind == "funding":
                t = _funding_table(t)
            elif kind == "derivs":
                t = _derivs_table(t)
            # Re-order by timestamp to protect against out-of-order rows.
            ts_np = t["timestamp"].to_numpy(zero_copy_only=False)
            if len(ts_np) > 1 and not np.all(ts_np[1:] >= ts_np[:-1]):
                order = np.argsort(ts_np, kind="stable")
                t = t.take(pa.array(order))
            w.write_table(t)
            total_rows += t.num_rows
            if (i + 1) % 50 == 0 or i == len(files) - 1:
                log.info("[%s] %d/%d files, %d rows", out_path.name,
                         i + 1, len(files), total_rows)
            del t
    dt = time.monotonic() - t_start
    return total_rows, dt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--stream", choices=list(STREAMS), default=None,
                   help="merge just this stream (default: all)")
    p.add_argument("--hours", type=int, default=0,
                   help="if > 0, take only the last N files per stream")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (args.data_dir / "_merged")
    out_dir.mkdir(exist_ok=True)

    which = [args.stream] if args.stream else list(STREAMS)
    for name in which:
        spec = STREAMS[name]
        src_dir = args.data_dir / spec["dir"]
        if not src_dir.exists():
            log.info("skip %s (no dir %s)", name, src_dir)
            continue
        files = sorted(src_dir.glob("*.parquet"))
        if args.hours > 0:
            files = files[-args.hours:]
        out_path = out_dir / f"{name}.parquet"
        log.info("merging %s → %s (%d files)", name, out_path, len(files))
        rows, dt = merge_stream(files, out_path, spec["schema"])
        log.info("%s: %d rows in %.1fs (%.0f MB)", name, rows, dt,
                 out_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
