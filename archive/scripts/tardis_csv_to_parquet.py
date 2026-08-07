#!/usr/bin/env python3
"""Convert Tardis free-tier CSVs (book_snapshot_25 / trades / derivative_ticker)
into the project's flat parquet schema (matches recorder + Tardis ingester
already on disk).

Reads from `--src` (default: /home/scalper/scalper-bot/data_tardis_free) and
writes into `--dst` (default: /home/scalper/scalper-bot/data), skipping files
that already exist (idempotent).

Output layout (mirrors recorder):
    data/depth/{YYYYMMDD}_tardis.parquet              (BTC)
    data/eth_depth/{YYYYMMDD}_tardis.parquet
    data/sol_depth/{YYYYMMDD}_tardis.parquet
    data/bnb_depth/{YYYYMMDD}_tardis.parquet
    data/trades/{YYYYMMDD}_tardis.parquet             (BTC)
    data/eth_trades/{YYYYMMDD}_tardis.parquet
    data/sol_trades/{YYYYMMDD}_tardis.parquet
    data/bnb_trades/{YYYYMMDD}_tardis.parquet
    data/{bybit,okx,bitget,gateio}_trades/{YYYYMMDD}_tardis_btc.parquet
    data/{...}_trades/{YYYYMMDD}_tardis_eth.parquet
    data/funding/{YYYYMMDD}_tardis.parquet            (from BTC ticker)
    data/derivatives/{YYYYMMDD}_tardis.parquet        (from BTC ticker)

Tardis schemas in (verified 2026-04-14):
    book_snapshot_25 cols: exchange,symbol,timestamp,local_timestamp,
                            asks[0..24].price, asks[0..24].amount,
                            bids[0..24].price, bids[0..24].amount
    trades cols:          exchange,symbol,timestamp,local_timestamp,
                            id, side, price, amount
    derivative_ticker cols: exchange,symbol,timestamp,local_timestamp,
                            funding_timestamp, funding_rate,
                            predicted_funding_rate, open_interest,
                            last_price, index_price, mark_price

Tardis timestamps are MICROSECONDS. We divide by 1000 → milliseconds
(matches recorder + downstream pipeline).
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEPTH_LEVELS = 20  # we keep top-20 of Tardis 25-level book


# --------------------------------------------------------------------------- #
# Output schema helpers (match recorder.py)
# --------------------------------------------------------------------------- #

def _fsl(flat: np.ndarray) -> pa.Array:
    """Build FixedSizeList<f64, 20> from a flat 1-D array."""
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat.reshape(-1).astype(np.float64, copy=False), type=pa.float64()),
        DEPTH_LEVELS,
    )


def _atomic_write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Per-stream parsers — each returns a pyarrow Table or None (empty file)
# --------------------------------------------------------------------------- #

def _parse_book_snapshot(csv_path: Path) -> pa.Table | None:
    """book_snapshot_25 → flat depth (top-20)."""
    cols = ["timestamp"]
    for i in range(DEPTH_LEVELS):
        cols += [f"asks[{i}].price", f"asks[{i}].amount",
                 f"bids[{i}].price", f"bids[{i}].amount"]
    df = pd.read_csv(csv_path, usecols=cols, dtype={"timestamp": np.int64})
    n = len(df)
    if n == 0:
        return None

    ts = (df["timestamp"].to_numpy(dtype=np.int64) // 1000)  # μs → ms
    bp = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    bq = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    ap = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    aq = np.zeros((n, DEPTH_LEVELS), dtype=np.float64)
    for i in range(DEPTH_LEVELS):
        bp[:, i] = df[f"bids[{i}].price"].to_numpy(dtype=np.float64, na_value=0.0)
        bq[:, i] = df[f"bids[{i}].amount"].to_numpy(dtype=np.float64, na_value=0.0)
        ap[:, i] = df[f"asks[{i}].price"].to_numpy(dtype=np.float64, na_value=0.0)
        aq[:, i] = df[f"asks[{i}].amount"].to_numpy(dtype=np.float64, na_value=0.0)

    return pa.table({
        "timestamp":  pa.array(ts, type=pa.int64()),
        "bid_prices": _fsl(bp),
        "bid_qtys":   _fsl(bq),
        "ask_prices": _fsl(ap),
        "ask_qtys":   _fsl(aq),
    })


def _parse_trades_binance(csv_path: Path) -> pa.Table | None:
    """Tardis trades → recorder schema (timestamp, price, quantity, is_buyer_maker).

    Tardis side: "buy" = buyer is the aggressor (taker); the *maker* was a seller.
                 So is_buyer_maker = False.
                 "sell" = seller aggressor → is_buyer_maker = True.
    """
    df = pd.read_csv(csv_path, usecols=["timestamp", "side", "price", "amount"],
                      dtype={"timestamp": np.int64, "side": "string",
                             "price": np.float64, "amount": np.float64})
    if len(df) == 0:
        return None
    ts = df["timestamp"].to_numpy(dtype=np.int64) // 1000
    is_buyer_maker = (df["side"].to_numpy() == "sell")
    return pa.table({
        "timestamp":      pa.array(ts, type=pa.int64()),
        "price":          pa.array(df["price"].to_numpy(), type=pa.float64()),
        "quantity":       pa.array(df["amount"].to_numpy(), type=pa.float64()),
        "is_buyer_maker": pa.array(is_buyer_maker, type=pa.bool_()),
    })


def _parse_trades_xex(csv_path: Path) -> pa.Table | None:
    """Cross-exchange trades → recorder schema (timestamp, price, quantity, is_seller).

    is_seller mirrors recorder convention: True iff seller was the aggressor.
    """
    df = pd.read_csv(csv_path, usecols=["timestamp", "side", "price", "amount"],
                      dtype={"timestamp": np.int64, "side": "string",
                             "price": np.float64, "amount": np.float64})
    if len(df) == 0:
        return None
    ts = df["timestamp"].to_numpy(dtype=np.int64) // 1000
    is_seller = (df["side"].to_numpy() == "sell")
    return pa.table({
        "timestamp":  pa.array(ts, type=pa.int64()),
        "price":      pa.array(df["price"].to_numpy(), type=pa.float64()),
        "quantity":   pa.array(df["amount"].to_numpy(), type=pa.float64()),
        "is_seller":  pa.array(is_seller, type=pa.bool_()),
    })


def _parse_funding_from_ticker(csv_path: Path) -> pa.Table | None:
    """derivative_ticker → funding stream (timestamp, funding_rate, mark_price).

    Tardis emits a row whenever any field changes; we forward-fill to keep rows
    where funding_rate is meaningful and drop rows with NaN funding_rate.
    """
    df = pd.read_csv(csv_path, usecols=["timestamp", "funding_rate", "mark_price"],
                      dtype={"timestamp": np.int64, "funding_rate": np.float64,
                             "mark_price": np.float64})
    if len(df) == 0:
        return None
    df = df.dropna(subset=["funding_rate"])
    if len(df) == 0:
        return None
    ts = df["timestamp"].to_numpy(dtype=np.int64) // 1000
    return pa.table({
        "timestamp":    pa.array(ts, type=pa.int64()),
        "funding_rate": pa.array(df["funding_rate"].to_numpy(), type=pa.float64()),
        "mark_price":   pa.array(df["mark_price"].fillna(0.0).to_numpy(), type=pa.float64()),
    })


def _parse_derivatives_from_ticker(csv_path: Path) -> pa.Table | None:
    """derivative_ticker → derivatives stream (timestamp, open_interest,
    long_short_ratio).

    Tardis free tier doesn't include long_short_ratio (Binance API-only) — we
    fill with NaN; trainer's cross-asset features tolerate NaN via masking.
    """
    df = pd.read_csv(csv_path, usecols=["timestamp", "open_interest"],
                      dtype={"timestamp": np.int64, "open_interest": np.float64})
    if len(df) == 0:
        return None
    df = df.dropna(subset=["open_interest"])
    if len(df) == 0:
        return None
    ts = df["timestamp"].to_numpy(dtype=np.int64) // 1000
    return pa.table({
        "timestamp":         pa.array(ts, type=pa.int64()),
        "open_interest":     pa.array(df["open_interest"].to_numpy(), type=pa.float64()),
        "long_short_ratio":  pa.array(np.full(len(df), np.nan), type=pa.float64()),
    })


# --------------------------------------------------------------------------- #
# Job table — (src_glob, parser_fn, dst_dir, dst_suffix)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Job:
    src_subdir: str            # path under data_tardis_free/
    symbol_in_filename: str    # which Tardis symbol in CSV name to pick (e.g. "BTCUSDT")
    parser: Callable[[Path], pa.Table | None]
    dst_subdir: str            # path under data/
    dst_suffix: str            # suffix before .parquet (after YYYYMMDD_)


JOBS: tuple[Job, ...] = (
    # Binance Futures depth (BTC + ETH + SOL + BNB)
    Job("binance-futures/book_snapshot_25", "BTCUSDT", _parse_book_snapshot,
        "depth", "tardis"),
    Job("binance-futures/book_snapshot_25", "ETHUSDT", _parse_book_snapshot,
        "eth_depth", "tardis"),
    Job("binance-futures/book_snapshot_25", "SOLUSDT", _parse_book_snapshot,
        "sol_depth", "tardis"),
    Job("binance-futures/book_snapshot_25", "BNBUSDT", _parse_book_snapshot,
        "bnb_depth", "tardis"),

    # Binance Futures trades (BTC was already in data/trades/ from prior runs;
    # we re-emit harmlessly under same name — adapter skips if file exists).
    Job("binance-futures/trades", "ETHUSDT", _parse_trades_binance,
        "eth_trades", "tardis"),
    Job("binance-futures/trades", "SOLUSDT", _parse_trades_binance,
        "sol_trades", "tardis"),
    Job("binance-futures/trades", "BNBUSDT", _parse_trades_binance,
        "bnb_trades", "tardis"),

    # Cross-exchange trades — Tardis free tier covers BTC + ETH on each.
    Job("bybit/trades",            "BTCUSDT",       _parse_trades_xex,
        "bybit_trades",   "tardis_btc"),
    Job("bybit/trades",            "ETHUSDT",       _parse_trades_xex,
        "bybit_trades",   "tardis_eth"),
    Job("okex-swap/trades",        "BTC-USDT-SWAP", _parse_trades_xex,
        "okx_trades",     "tardis_btc"),
    Job("okex-swap/trades",        "ETH-USDT-SWAP", _parse_trades_xex,
        "okx_trades",     "tardis_eth"),
    Job("bitget-futures/trades",   "BTCUSDT",       _parse_trades_xex,
        "bitget_trades",  "tardis_btc"),
    Job("bitget-futures/trades",   "ETHUSDT",       _parse_trades_xex,
        "bitget_trades",  "tardis_eth"),
    Job("gate-io-futures/trades",  "BTC_USDT",      _parse_trades_xex,
        "gateio_trades",  "tardis_btc"),
    Job("gate-io-futures/trades",  "ETH_USDT",      _parse_trades_xex,
        "gateio_trades",  "tardis_eth"),

    # Funding + open interest from ETH ticker (BTC ticker isn't in the free
    # download script — we already have continuous BTC funding from the
    # recorder, so historical BTC funding back to 2022 isn't critical).
    Job("binance-futures/derivative_ticker", "ETHUSDT", _parse_funding_from_ticker,
        "eth_funding",     "tardis"),
    Job("binance-futures/derivative_ticker", "ETHUSDT", _parse_derivatives_from_ticker,
        "eth_derivatives", "tardis"),
)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _process_one(args) -> dict:
    job, csv_path, dst_root = args
    # Filename is YYYY-MM-DD_SYMBOL.csv.gz
    name = csv_path.name
    if not name.endswith(".csv.gz"):
        return {"status": "skip:not_gz", "src": str(csv_path)}
    base = name[:-len(".csv.gz")]
    try:
        date_str, sym = base.split("_", 1)
    except ValueError:
        return {"status": "skip:badname", "src": str(csv_path)}
    if sym != job.symbol_in_filename:
        return {"status": "skip:wrong_symbol"}
    yyyymmdd = date_str.replace("-", "")
    out_path = dst_root / job.dst_subdir / f"{yyyymmdd}_{job.dst_suffix}.parquet"
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "exists", "dst": str(out_path)}
    try:
        table = job.parser(csv_path)
    except Exception as e:
        return {"status": "error:parse", "src": str(csv_path), "err": repr(e)}
    if table is None or table.num_rows == 0:
        return {"status": "empty", "src": str(csv_path)}
    try:
        _atomic_write(table, out_path)
    except Exception as e:
        return {"status": "error:write", "dst": str(out_path), "err": repr(e)}
    return {"status": "ok", "dst": str(out_path), "rows": table.num_rows,
            "size": out_path.stat().st_size}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/home/scalper/scalper-bot/data_tardis_free",
                   type=Path)
    p.add_argument("--dst", default="/home/scalper/scalper-bot/data", type=Path)
    p.add_argument("--workers", type=int, default=8,
                   help="parallel CSV parsing workers (each uses ~500MB peak)")
    p.add_argument("--filter-job", default=None,
                   help="only run jobs whose dst_subdir contains this substring "
                        "(e.g. 'depth', 'trades', 'funding')")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.src.exists():
        print(f"ERROR: src missing {args.src}", file=sys.stderr); return 2

    # Build job list — for each Job, every matching CSV in src.
    work: list[tuple[Job, Path, Path]] = []
    for job in JOBS:
        if args.filter_job and args.filter_job not in job.dst_subdir:
            continue
        src_dir = args.src / job.src_subdir
        if not src_dir.exists():
            continue
        for csv_path in sorted(src_dir.glob(f"*_{job.symbol_in_filename}.csv.gz")):
            work.append((job, csv_path, args.dst))

    print(f"[adapter] {len(work)} files queued (workers={args.workers})")
    if args.dry_run:
        for job, csv_path, _ in work[:5]:
            print(f"  {csv_path}  →  {job.dst_subdir}/{job.dst_suffix}.parquet")
        print(f"  ... and {len(work) - 5} more")
        return 0

    t0 = time.time()
    stats = {"ok": 0, "exists": 0, "empty": 0, "error": 0, "skip": 0}
    bytes_out = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, w): w for w in work}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            s = res["status"].split(":")[0]
            stats[s] = stats.get(s, 0) + 1
            if s == "ok":
                bytes_out += res.get("size", 0)
            if s in ("ok", "error", "empty") or done % 100 == 0:
                pct = 100.0 * done / len(work)
                print(f"[{done:5d}/{len(work)} {pct:5.1f}%] {res['status']:15s}  "
                      f"{res.get('dst', res.get('src', ''))[-72:]}  "
                      f"(out {bytes_out/1e9:.2f} GB; ok={stats['ok']} "
                      f"exists={stats['exists']} empty={stats.get('empty',0)} "
                      f"err={stats['error']})")
                if "err" in res:
                    print(f"        !! {res['err']}")

    dt = time.time() - t0
    print(f"\n[adapter] DONE in {dt/60:.1f} min — {bytes_out/1e9:.2f} GB written")
    for k, v in stats.items():
        print(f"  {k:8s} = {v}")
    return 0 if stats.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
