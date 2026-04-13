"""Ingest Tardis free-tier historical CSV samples into our recorder parquet layout.

Tardis offers the first day of each month free (no API key) at
`https://datasets.tardis.dev/v1/{exchange}/{data_type}/YYYY/MM/DD/{SYMBOL}.csv.gz`.
This script downloads, parses, and writes data in the exact schema the
recorder produces — so `Trainer.build_samples_cached` consumes it without
any code change.

**Streaming ingest**: download one day → parse in-memory → write parquet →
delete gz. Peak disk ≈ 2 GB gzipped + one in-flight decompressed CSV.

Supported sources (see `SOURCES` below):
  * binance-futures BTCUSDT / ETHUSDT — incremental_book_L2 → our depth
    schema (100ms snapshots, top-20 levels). Trades + derivative_ticker
    normalized.
  * bybit / okex-swap / bitget-futures / gate-io-futures BTC trades — for
    feature 30 cross-exchange momentum.

Each day's raw CSV goes to `--tmp-dir` (default `/tmp/tardis`), gets parsed,
written as `{data_dir}/{subdir}/{YYYYMMDD}_00.parquet` (one file per day,
mirroring recorder's hourly files in aggregate form), and deleted.

Usage:
    python scripts/ingest_tardis.py --months 2024-06,2024-07,2024-08
    python scripts/ingest_tardis.py --all-free  # all 1st-of-month from 2020-01
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_tardis")


# --------------------------------------------------------------------------- #
# Source table — (exchange, data_type, symbol, output_subdir, parser_name)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Source:
    slug: str               # key used in CLI and logs
    exchange: str           # Tardis exchange id
    data_type: str          # Tardis feed name (incremental_book_L2 / trades / derivative_ticker)
    symbol: str             # Tardis symbol id
    output_subdir: str      # where our recorder would have put it
    parser: str             # 'depth', 'trades_binance', 'trades_xex', 'ticker_binance'
    available_from: date    # earliest free-tier month
    extra_subdirs: tuple[str, ...] = ()  # ticker splits into funding/ + derivatives/


SOURCES: tuple[Source, ...] = (
    Source("binance_btc_depth", "binance-futures", "incremental_book_L2", "BTCUSDT",
           "depth", "depth", date(2020, 1, 7)),
    Source("binance_btc_trades", "binance-futures", "trades", "BTCUSDT",
           "trades", "trades_binance", date(2019, 11, 17)),
    Source("binance_btc_ticker", "binance-futures", "derivative_ticker", "BTCUSDT",
           "ticker", "ticker_binance", date(2019, 11, 17),
           extra_subdirs=("funding", "derivatives")),
    Source("binance_eth_depth", "binance-futures", "incremental_book_L2", "ETHUSDT",
           "eth_depth", "depth", date(2020, 1, 7)),
    Source("binance_eth_trades", "binance-futures", "trades", "ETHUSDT",
           "eth_trades", "trades_binance", date(2019, 11, 17)),
    Source("bybit_btc_trades", "bybit", "trades", "BTCUSDT",
           "bybit_trades", "trades_xex", date(2020, 5, 28)),
    Source("okex_btc_trades", "okex-swap", "trades", "BTC-USDT-SWAP",
           "okx_trades", "trades_xex", date(2019, 12, 4)),
    Source("bitget_btc_trades", "bitget-futures", "trades", "BTCUSDT",
           "bitget_trades", "trades_xex", date(2024, 11, 8)),
    Source("gateio_btc_trades", "gate-io-futures", "trades", "BTC_USDT",
           "gateio_trades", "trades_xex", date(2020, 7, 1)),
)


# --------------------------------------------------------------------------- #
# Tardis CSV → numpy helpers
# --------------------------------------------------------------------------- #


def tardis_url(src: Source, day: date) -> str:
    return (f"https://datasets.tardis.dev/v1/{src.exchange}/{src.data_type}/"
            f"{day:%Y/%m/%d}/{src.symbol}.csv.gz")


def download_to_file(url: str, dst: Path, timeout: float = 120.0) -> int:
    """Stream-download. Returns total bytes written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            raise FileNotFoundError(f"404 — not in free tier: {url}")
        r.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                n += len(chunk)
    return n


def _write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Parser: incremental_book_L2 → 100ms top-20 snapshots (depth schema)
# --------------------------------------------------------------------------- #

SNAPSHOT_INTERVAL_MS = 100
DEPTH_LEVELS = 20
RUST_BINARY_DEFAULT = (
    Path(__file__).resolve().parent.parent
    / "rust_ingest" / "target" / "release" / "depth_parser"
)


def parse_depth(csv_gz_path: Path, out_path: Path,
                 warmup_minutes: int = 30,
                 max_rows: int | None = None,
                 rust_binary: Path | None = None) -> dict:
    """Reconstruct order book from Tardis incremental_book_L2 → emit 100ms
    top-20 snapshots matching recorder.py output schema.

    Schema out:
        timestamp: int64 (ms)
        bids: list<list<f64>> length 20, each [price, qty]
        asks: list<list<f64>> length 20, each [price, qty]

    Implementation:
        Rust binary (`rust_ingest/target/release/depth_parser`) does the
        hot loop (~500k rows/s single core, 100× faster than Python). It
        emits fixed-size 648-byte records. We read them via numpy.fromfile
        and build the arrow table.

    `warmup_minutes` handled by the Rust binary (hardcoded to 30 min — the
    book reconstruction from incremental updates alone needs time to fill
    before snapshots are meaningful. `max_rows` is not supported in the
    Rust fast path; use trades/ticker parsers for quick smoke-tests.
    """
    t0 = time.time()
    rust_binary = rust_binary or RUST_BINARY_DEFAULT
    if not rust_binary.exists():
        raise FileNotFoundError(
            f"Rust depth_parser not built. Run:\n"
            f"  cd rust_ingest && cargo build --release\n"
            f"Missing: {rust_binary}"
        )

    tmp_bin = out_path.with_suffix(out_path.suffix + ".bin")
    tmp_bin.parent.mkdir(parents=True, exist_ok=True)

    # Invoke Rust parser — it writes the binary and prints N snapshots.
    cmd = [str(rust_binary), str(csv_gz_path), str(tmp_bin)]
    logger.info("  depth parser: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Rust depth_parser failed ({e.returncode}): {e.stderr}"
        ) from e
    n_snap = int(res.stdout.strip())
    rust_log = res.stderr.strip()
    logger.info("  %s", rust_log)

    # Load the binary: 648 bytes per record.
    # i64 timestamp, f64[40] bids_flat, f64[40] asks_flat.
    record_dtype = np.dtype([
        ("timestamp", "<i8"),
        ("bids_flat", "<f8", 40),
        ("asks_flat", "<f8", 40),
    ])
    data = np.fromfile(str(tmp_bin), dtype=record_dtype)
    if len(data) != n_snap:
        raise RuntimeError(
            f"snapshot count mismatch: rust said {n_snap}, file has {len(data)}"
        )

    # Build arrow table in the FLAT v3 schema (schema_version=3 attribute):
    #   timestamp: int64
    #   bid_prices: FixedSizeList<f64, 20>
    #   bid_qtys:   FixedSizeList<f64, 20>
    #   ask_prices: FixedSizeList<f64, 20>
    #   ask_qtys:   FixedSizeList<f64, 20>
    # Flat columns make trainer.load_depth_data zero-copy numpy.
    ts = pa.array(data["timestamp"], type=pa.int64())
    n = len(data)

    # bids_flat / asks_flat are (n, 40) in [p0, q0, p1, q1, ...] layout.
    # Reshape to (n, 20, 2) then split prices vs qtys along last axis.
    bids = data["bids_flat"].reshape(n, 20, 2)
    asks = data["asks_flat"].reshape(n, 20, 2)

    def _fixed_list(arr_2d: np.ndarray) -> pa.FixedSizeListArray:
        """(n, 20) float64 → FixedSizeList<f64, 20>."""
        values = pa.array(arr_2d.reshape(-1), type=pa.float64())
        return pa.FixedSizeListArray.from_arrays(values, list_size=20)

    bid_prices_col = _fixed_list(bids[:, :, 0])
    bid_qtys_col = _fixed_list(bids[:, :, 1])
    ask_prices_col = _fixed_list(asks[:, :, 0])
    ask_qtys_col = _fixed_list(asks[:, :, 1])

    table = pa.table({
        "timestamp": ts,
        "bid_prices": bid_prices_col,
        "bid_qtys": bid_qtys_col,
        "ask_prices": ask_prices_col,
        "ask_qtys": ask_qtys_col,
    })
    _write_parquet(table, out_path)
    tmp_bin.unlink(missing_ok=True)

    t_tot = time.time() - t0
    logger.info("  depth wrote %d snapshots to %s (%.1fs total)",
                 len(data), out_path, t_tot)
    return {"snapshots_out": int(len(data)), "seconds": t_tot}


# --------------------------------------------------------------------------- #
# Parser: Binance trades → recorder schema
# --------------------------------------------------------------------------- #


def parse_trades_binance(csv_gz_path: Path, out_path: Path) -> dict:
    """Tardis Binance trades CSV columns:
        exchange,symbol,timestamp,local_timestamp,id,side,price,amount
    Our recorder trades schema:
        timestamp:int64, price:f64, quantity:f64, is_buyer_maker:bool
    side 'sell' means the taker sold → maker bought → is_buyer_maker=True
    side 'buy'  means the taker bought → maker sold → is_buyer_maker=False
    """
    t0 = time.time()
    df = pd.read_csv(csv_gz_path, compression="gzip",
                     usecols=["timestamp", "side", "price", "amount"],
                     dtype={"side": "category"})
    ts_ms = (df["timestamp"].values // 1000).astype(np.int64)
    price = df["price"].values.astype(np.float64)
    qty = df["amount"].values.astype(np.float64)
    side = df["side"].values.astype(str)
    # Tardis "sell" = aggressive seller → buyer was maker → is_buyer_maker True
    is_buyer_maker = (side == "sell")
    table = pa.table({
        "timestamp": pa.array(ts_ms, type=pa.int64()),
        "price": pa.array(price, type=pa.float64()),
        "quantity": pa.array(qty, type=pa.float64()),
        "is_buyer_maker": pa.array(is_buyer_maker, type=pa.bool_()),
    })
    _write_parquet(table, out_path)
    return {"rows": len(df), "seconds": time.time() - t0}


def parse_trades_xex(csv_gz_path: Path, out_path: Path, exchange_slug: str) -> dict:
    """Cross-exchange trades for feature 30. Recorder schema:
        timestamp:int64, price:f64, quantity:f64, is_seller:bool
    """
    t0 = time.time()
    df = pd.read_csv(csv_gz_path, compression="gzip",
                     usecols=["timestamp", "side", "price", "amount"],
                     dtype={"side": "category"})
    ts_ms = (df["timestamp"].values // 1000).astype(np.int64)
    price = df["price"].values.astype(np.float64)
    qty = df["amount"].values.astype(np.float64)
    side = df["side"].values.astype(str)
    is_seller = (side == "sell")
    table = pa.table({
        "timestamp": pa.array(ts_ms, type=pa.int64()),
        "price": pa.array(price, type=pa.float64()),
        "quantity": pa.array(qty, type=pa.float64()),
        "is_seller": pa.array(is_seller, type=pa.bool_()),
    })
    _write_parquet(table, out_path)
    return {"rows": len(df), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# Parser: Binance derivative_ticker → funding + derivatives split
# --------------------------------------------------------------------------- #


def parse_ticker_binance(csv_gz_path: Path,
                          funding_path: Path,
                          derivatives_path: Path) -> dict:
    """Binance futures derivative_ticker columns:
        exchange,symbol,timestamp,local_timestamp,funding_timestamp,
        funding_rate,predicted_funding_rate,open_interest,last_price,
        index_price,mark_price
    Split into:
      - funding: timestamp, funding_rate, mark_price
      - derivatives: timestamp, open_interest, long_short_ratio (N/A → 0)
    """
    t0 = time.time()
    df = pd.read_csv(csv_gz_path, compression="gzip",
                     usecols=["timestamp", "funding_rate", "mark_price",
                              "open_interest"])
    ts_ms = (df["timestamp"].values // 1000).astype(np.int64)

    funding_table = pa.table({
        "timestamp": pa.array(ts_ms, type=pa.int64()),
        "funding_rate": pa.array(df["funding_rate"].values.astype(np.float64),
                                  type=pa.float64()),
        "mark_price": pa.array(df["mark_price"].values.astype(np.float64),
                                type=pa.float64()),
    })
    _write_parquet(funding_table, funding_path)

    # derivatives: open_interest + long_short_ratio placeholder 0
    oi = df["open_interest"].values.astype(np.float64)
    ls = np.zeros_like(oi)
    deriv_table = pa.table({
        "timestamp": pa.array(ts_ms, type=pa.int64()),
        "open_interest": pa.array(oi, type=pa.float64()),
        "long_short_ratio": pa.array(ls, type=pa.float64()),
    })
    _write_parquet(deriv_table, derivatives_path)
    return {"rows": len(df), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def month_iter(months: list[str]) -> Iterator[date]:
    """Yield the 1st-of-month date for each 'YYYY-MM' string."""
    for m in months:
        y, mo = map(int, m.split("-"))
        yield date(y, mo, 1)


def all_free_months(start: date = date(2020, 1, 1),
                     end: date | None = None) -> list[date]:
    end = end or date.today()
    out = []
    d = date(start.year, start.month, 1)
    while d <= end:
        out.append(d)
        # next month
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)
    return out


def ingest_one(src: Source, day: date, tmp_dir: Path, data_dir: Path,
                skip_existing: bool = True) -> dict | None:
    if day < src.available_from:
        logger.info("  %s @ %s — before available_from %s, skipping",
                    src.slug, day, src.available_from)
        return None
    out_path = data_dir / src.output_subdir / f"{day:%Y%m%d}_tardis.parquet"
    if skip_existing and out_path.exists():
        logger.info("  %s @ %s — parquet exists, skipping", src.slug, day)
        return {"skipped": True}

    url = tardis_url(src, day)
    gz_path = tmp_dir / f"{src.slug}_{day:%Y%m%d}.csv.gz"
    logger.info("  %s @ %s → downloading", src.slug, day)
    try:
        n_bytes = download_to_file(url, gz_path)
    except FileNotFoundError:
        logger.warning("  %s @ %s — 404 (not free?)", src.slug, day)
        return {"error": "404"}
    logger.info("  %s @ %s — downloaded %.1f MB", src.slug, day, n_bytes / 1e6)

    try:
        if src.parser == "depth":
            stats = parse_depth(gz_path, out_path)
        elif src.parser == "trades_binance":
            stats = parse_trades_binance(gz_path, out_path)
        elif src.parser == "trades_xex":
            stats = parse_trades_xex(gz_path, out_path, src.slug)
        elif src.parser == "ticker_binance":
            funding_path = data_dir / "funding" / f"{day:%Y%m%d}_tardis.parquet"
            deriv_path = data_dir / "derivatives" / f"{day:%Y%m%d}_tardis.parquet"
            stats = parse_ticker_binance(gz_path, funding_path, deriv_path)
        else:
            raise ValueError(f"unknown parser {src.parser}")
    finally:
        gz_path.unlink(missing_ok=True)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=str, default="",
                   help="Comma-separated YYYY-MM (1st-of-month dates are used)")
    p.add_argument("--all-free", action="store_true",
                   help="Ingest every 1st-of-month from 2020-01 to today")
    p.add_argument("--sources", type=str, default="",
                   help="Comma-separated source slugs (default: all)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tmp-dir", type=str, default="/tmp/tardis")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--force", action="store_true",
                   help="Rewrite existing parquet files")
    args = p.parse_args()

    if args.all_free:
        days = all_free_months()
    elif args.months:
        days = list(month_iter(args.months.split(",")))
    else:
        p.error("one of --months / --all-free is required")

    sources = SOURCES
    if args.sources:
        slugs = set(args.sources.split(","))
        sources = tuple(s for s in SOURCES if s.slug in slugs)

    data_dir = Path(args.data_dir)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Tardis ingest")
    logger.info("  Days: %d | Sources: %d | data_dir: %s",
                len(days), len(sources), data_dir)
    logger.info("=" * 60)

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for day in days:
        for src in sources:
            try:
                r = ingest_one(src, day, tmp_dir, data_dir,
                                skip_existing=not args.force)
                if r is None:
                    n_skip += 1
                elif r.get("skipped"):
                    n_skip += 1
                elif r.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
            except Exception as e:
                logger.exception("  %s @ %s — FAILED: %r", src.slug, day, e)
                n_err += 1

    logger.info("=" * 60)
    logger.info("  done: %d ok, %d skipped, %d errors (%.1fm)",
                n_ok, n_skip, n_err, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
