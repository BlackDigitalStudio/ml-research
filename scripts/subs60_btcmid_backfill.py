#!/usr/bin/env python3
"""Minimal feats_sub60/BTC {td,mid} backfill from raw BTC book, for the btc_ret lead.

The maker-label build (subs60_makerlabel_build.load_btc_mid) reads ONLY td+mid from
feats_sub60/BTC. The full orchestrator needs BTC trades (limiting coverage to
book&trades-days); BTC mid only needs book. This builds {td,mid} on the 1s grid
(identical to subs60_orch.py) for every BTC book-day in the window -> btc_ret covers
all book-days, not just book&trades-days. Idempotent (skips existing).
"""
import datetime as dt, io, sys
import numpy as np, pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
NS = 1_000_000_000; GRID_S = 1.0; SYM = "BTC-USDT-PERP"
bk = storage.Client(project=PROJ).bucket(BUCKET)


def main():
    start, end = sys.argv[1], sys.argv[2]
    d0 = dt.date.fromisoformat(start); d1 = dt.date.fromisoformat(end)
    days = [(d0 + dt.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    built = 0
    for day in days:
        out = f"feats_sub60/{SYM}/{day}.npz"
        if bk.blob(out).exists():
            print(day, "skip-exists"); continue
        pref = f"raw/book/exchange=BINANCE_FUTURES/symbol={SYM}/dt={day}/"
        n = next((b.name for b in bk.client.list_blobs(bk, prefix=pref) if b.name.endswith(".parquet")), None)
        if not n:
            print(day, "no-book"); continue
        t = pq.read_table(io.BytesIO(bk.blob(n).download_as_bytes()),
                          columns=["timestamp", "bid_0_price", "ask_0_price"])
        ts = t["timestamp"].to_numpy().astype(np.int64)
        mid = 0.5 * (t["bid_0_price"].to_numpy().astype(np.float64) + t["ask_0_price"].to_numpy().astype(np.float64))
        o = np.argsort(ts, kind="stable"); ts = ts[o]; mid = mid[o]
        if len(ts) < 2000:
            print(day, "short"); continue
        grid = np.arange(ts[0] + 120 * NS, ts[-1] - 70 * NS, int(GRID_S * NS), dtype=np.int64)
        idx = np.unique(np.clip(np.searchsorted(ts, grid, "right") - 1, 0, len(ts) - 1)).astype(np.int64)
        buf = io.BytesIO()
        np.savez_compressed(buf, td=ts[idx].astype(np.int64), mid=mid[idx].astype(np.float64))
        bk.blob(out).upload_from_string(buf.getvalue())
        built += 1; print(day, f"built n={len(idx)}")
    print("DONE built", built)


if __name__ == "__main__":
    main()
