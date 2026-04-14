#!/usr/bin/env python3
"""Download Tardis free 1st-of-each-month data.

Tardis publishes first day of every month openly at
datasets.tardis.dev — no API key needed. Covers L2 book snapshots,
trades, and derivative ticker at the same quality as paid feeds.

This script grabs additional symbols + cross-exchange data beyond what
our recorder already captures live:
  - binance-futures: ETHUSDT, SOLUSDT, BNBUSDT (book_snapshot_25 + trades)
  - cross-exchange (bybit/okx/bitget/gate-io): BTCUSDT + ETHUSDT (trades
    only — same depth profile as our recorder's cross-exchange streams)

Free window: 2020-01 through current month. Saves as .csv.gz into
--out/{exchange}/{data_type}/{YYYY-MM-DD}_{SYMBOL}.csv.gz.

Usage:
    python scripts/download_tardis_free.py \\
        --out /workspace/scalper-bot/data_tardis_free \\
        --start 2022-01 --end 2026-04 \\
        --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path


# Exchange / symbol / data-type matrix.
# Symbol IDs are Tardis normalized exchange-native codes — check
# https://docs.tardis.dev/downloadable-csv-files for the up-to-date list.
TARGETS = [
    # Main Binance Futures — ETH/SOL/BNB (BTC we already have).
    ("binance-futures", "BTCUSDT", "book_snapshot_25"),   # supplement for BTC too
    ("binance-futures", "ETHUSDT", "book_snapshot_25"),
    ("binance-futures", "ETHUSDT", "trades"),
    ("binance-futures", "ETHUSDT", "derivative_ticker"),
    ("binance-futures", "SOLUSDT", "book_snapshot_25"),
    ("binance-futures", "SOLUSDT", "trades"),
    ("binance-futures", "SOLUSDT", "derivative_ticker"),
    ("binance-futures", "BNBUSDT", "book_snapshot_25"),
    ("binance-futures", "BNBUSDT", "trades"),
    ("binance-futures", "BNBUSDT", "derivative_ticker"),

    # Cross-exchange — trades only (mirrors what our recorder captures live).
    ("bybit",             "BTCUSDT",         "trades"),
    ("bybit",             "ETHUSDT",         "trades"),
    ("okex-swap",         "BTC-USDT-SWAP",   "trades"),
    ("okex-swap",         "ETH-USDT-SWAP",   "trades"),
    ("bitget",            "BTCUSDT_UMCBL",   "trades"),   # USDT-M futures
    ("bitget",            "ETHUSDT_UMCBL",   "trades"),
    ("gate-io-futures",   "BTC_USDT",        "trades"),
    ("gate-io-futures",   "ETH_USDT",        "trades"),
]

BASE_URL = "https://datasets.tardis.dev/v1/{exchange}/{dtype}/{year}/{month:02d}/{day:02d}/{symbol}.csv.gz"


def _parse_ym(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def _monthly_firsts(start_ym: tuple[int, int], end_ym: tuple[int, int]) -> list[date]:
    """All 1st-of-month dates in [start, end] inclusive."""
    dates = []
    y, m = start_ym
    ey, em = end_ym
    while (y, m) <= (ey, em):
        dates.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def _download_one(exchange: str, symbol: str, dtype: str, d: date,
                  out_root: Path, retries: int = 3) -> tuple[str, str, int]:
    """Returns (status, path, bytes). status in {ok, exists, missing, error}."""
    out_dir = out_root / exchange / dtype
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{d.isoformat()}_{symbol}.csv.gz"

    if out_path.exists() and out_path.stat().st_size > 0:
        return ("exists", str(out_path), out_path.stat().st_size)

    url = BASE_URL.format(
        exchange=exchange, dtype=dtype,
        year=d.year, month=d.month, day=d.day, symbol=symbol,
    )

    tmp = out_path.with_suffix(".csv.gz.tmp")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scalper-tardis-free/1"})
            with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
                total = 0
                while True:
                    chunk = r.read(1 << 20)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            tmp.rename(out_path)
            return ("ok", str(out_path), total)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                tmp.unlink(missing_ok=True)
                return ("missing", url, 0)
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                return ("error", f"HTTP {e.code} {url}", 0)
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                return ("error", f"{type(e).__name__}: {e} {url}", 0)
            time.sleep(2 ** attempt)

    return ("error", url, 0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output root dir")
    p.add_argument("--start", default="2022-01", help="earliest YYYY-MM")
    p.add_argument("--end", default="2026-04", help="latest YYYY-MM (inclusive)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    start_ym = _parse_ym(args.start)
    end_ym = _parse_ym(args.end)
    dates = _monthly_firsts(start_ym, end_ym)
    print(f"[tardis] {len(dates)} monthly dates from {dates[0]} to {dates[-1]}")
    print(f"[tardis] {len(TARGETS)} (exchange, symbol, data_type) targets")

    jobs = [(ex, sym, dt, d) for (ex, sym, dt) in TARGETS for d in dates]
    print(f"[tardis] {len(jobs)} total downloads, workers={args.workers}")

    if args.dry_run:
        for ex, sym, dt, d in jobs[:5]:
            url = BASE_URL.format(exchange=ex, dtype=dt, year=d.year,
                                   month=d.month, day=d.day, symbol=sym)
            print(f"  {url}")
        print(f"  ... and {len(jobs) - 5} more")
        return 0

    t0 = time.time()
    stats = {"ok": 0, "exists": 0, "missing": 0, "error": 0}
    total_bytes = 0

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, ex, sym, dt, d, out_root): (ex, sym, dt, d)
            for (ex, sym, dt, d) in jobs
        }
        done = 0
        for fut in cf.as_completed(futures):
            ex, sym, dt, d = futures[fut]
            status, info, nb = fut.result()
            stats[status] = stats.get(status, 0) + 1
            total_bytes += nb
            done += 1
            if status in ("ok", "error") or done % 50 == 0:
                pct = 100.0 * done / len(jobs)
                gb = total_bytes / 1e9
                print(f"[{done:4d}/{len(jobs)} {pct:5.1f}%] "
                      f"{status:7s}  {ex}/{dt}/{d.isoformat()}_{sym}  "
                      f"(total {gb:.2f} GB; ok={stats['ok']} "
                      f"exists={stats['exists']} missing={stats['missing']} err={stats['error']})")
                if status == "error":
                    print(f"         !! {info}")

    dt_s = time.time() - t0
    print(f"\n[tardis] DONE in {dt_s/60:.1f} min — "
          f"{total_bytes/1e9:.1f} GB downloaded")
    print(f"  ok={stats['ok']} exists={stats['exists']} "
          f"missing={stats['missing']} error={stats['error']}")
    return 0 if stats.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
