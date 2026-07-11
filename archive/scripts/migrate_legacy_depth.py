#!/usr/bin/env python3
"""Migrate legacy nested-depth parquet files to flat FixedSizeList<f64,20> schema.

Legacy recorder wrote:
    timestamp: int64
    bids:      list<list<f64>>   # per-row variable-length list of [price, qty]
    asks:      list<list<f64>>

Flat schema (Tardis + new recorder + Rust readers):
    timestamp:  int64
    bid_prices: FixedSizeList<f64, 20>  # zero-padded
    bid_qtys:   FixedSizeList<f64, 20>
    ask_prices: FixedSizeList<f64, 20>
    ask_qtys:   FixedSizeList<f64, 20>

This script rewrites legacy files in place (atomic via .tmp + rename). Files
already flat are left untouched. Dry-run mode available.

Usage:
    python3 scripts/migrate_legacy_depth.py
    python3 scripts/migrate_legacy_depth.py --dry-run
    python3 scripts/migrate_legacy_depth.py --dir /path/to/data/depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEPTH_LEVELS = 20
FLAT_COLS = {"bid_prices", "bid_qtys", "ask_prices", "ask_qtys"}


def _is_legacy(schema_names: set[str]) -> bool:
    return "bids" in schema_names and "asks" in schema_names and not FLAT_COLS <= schema_names


def _convert_table(t: pa.Table) -> pa.Table:
    """Nested list-of-tuples → flat FixedSizeList, zero-padded to 20 levels."""
    t = t.combine_chunks()
    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    bids_raw = t["bids"].to_pylist()
    asks_raw = t["asks"].to_pylist()
    n = len(ts)

    bp = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    bq = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    ap = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    aq = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)

    for i in range(n):
        for j, pq_pair in enumerate(bids_raw[i][:DEPTH_LEVELS]):
            bp[i, j] = pq_pair[0]
            bq[i, j] = pq_pair[1]
        for j, pq_pair in enumerate(asks_raw[i][:DEPTH_LEVELS]):
            ap[i, j] = pq_pair[0]
            aq[i, j] = pq_pair[1]

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


def migrate_file(path: Path, dry_run: bool) -> str:
    """Returns status string: flat|legacy|migrated|error."""
    try:
        t = pq.read_table(path)
    except Exception as e:
        return f"error:read:{e}"
    names = set(t.schema.names)
    if FLAT_COLS <= names:
        return "flat"
    if not _is_legacy(names):
        return f"error:unknown_schema:{sorted(names)}"
    if dry_run:
        return "legacy"
    try:
        new_t = _convert_table(t)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(new_t, tmp, compression="snappy")
        tmp.replace(path)
        return "migrated"
    except Exception as e:
        return f"error:write:{e}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", action="append", default=None,
                   help="Directory to scan (default: data/depth + data/eth_depth). Repeatable.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--root", default="/home/scalper/scalper-bot/data",
                   help="Data root (used when --dir not given)")
    args = p.parse_args()

    if args.dir:
        dirs = [Path(d) for d in args.dir]
    else:
        root = Path(args.root)
        dirs = [root / "depth", root / "eth_depth"]

    total = {"flat": 0, "legacy": 0, "migrated": 0, "error": 0}
    for d in dirs:
        if not d.exists():
            print(f"[skip] {d} does not exist")
            continue
        files = sorted(d.glob("*.parquet"))
        # Also include parts subdir if present (orphaned parts).
        parts_dir = d / ".parts"
        if parts_dir.exists():
            files += sorted(parts_dir.glob("*.parquet"))
        print(f"\n== {d} ({len(files)} files) ==")
        for f in files:
            status = migrate_file(f, args.dry_run)
            bucket = "error" if status.startswith("error") else status
            total[bucket] = total.get(bucket, 0) + 1
            if status != "flat":
                print(f"  {status:10s} {f.name}")

    print(f"\nSummary: flat={total['flat']} legacy={total['legacy']} "
          f"migrated={total['migrated']} error={total['error']}")
    return 0 if total.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
