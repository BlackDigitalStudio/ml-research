#!/usr/bin/env python3
"""CROSS-SOURCE parity: does our recorder (chronos) see the same market as cryptolake (training
source)? Compares TOP-OF-BOOK on an overlap day: cryptolake raw/book (timestamp ns, bid_0/ask_0)
vs chronos depth_snapshot (exchange_event_ts_us, bid_prices[0]/ask_prices[0]), matched within 100ms.
This is the existential gate: train-on-cryptolake / trade-on-recorder works ONLY if the data agrees.
Usage: python3 subs60_xsource_parity.py [DAY]   (chronos copied to tmp_chronos_parity/DOGEUSDT/...)
"""
import sys
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-06-05"
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; TICK = 0.00001  # DOGE tick
bk = storage.Client(project=PROJ).bucket(BUCKET)

# --- cryptolake book (training source) ---
clb = next(b.name for b in bk.client.list_blobs(bk, prefix=f"raw/book/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/") if b.name.endswith(".parquet"))
bk.blob(clb).download_to_filename("/tmp/clb.parquet")
ct = pq.read_table("/tmp/clb.parquet", columns=["timestamp", "bid_0_price", "ask_0_price"])
cl_ts = ct["timestamp"].to_numpy().astype(np.int64) // 1000  # ns -> us
cl_bid = ct["bid_0_price"].to_numpy().astype(np.float64); cl_ask = ct["ask_0_price"].to_numpy().astype(np.float64)
o = np.argsort(cl_ts); cl_ts, cl_bid, cl_ask = cl_ts[o], cl_bid[o], cl_ask[o]
print(f"cryptolake book {DAY}: {len(cl_ts)} rows | span {(cl_ts[-1]-cl_ts[0])/3.6e9:.1f}h", flush=True)

# --- chronos depth_snapshot (our recorder); null-robust via pandas ---
ch_ts, ch_bid, ch_ask = [], [], []
for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/depth_snapshot/"):
    if not b.name.endswith(".parquet"):
        continue
    bk.blob(b.name).download_to_filename("/tmp/ch.parquet")
    df = pq.read_table("/tmp/ch.parquet", columns=["exchange_event_ts_us", "bid_prices", "ask_prices"]).to_pandas()
    df = df[df["exchange_event_ts_us"].notna()]
    df = df[df["bid_prices"].apply(lambda x: x is not None and len(x) > 0 and x[0] is not None)]
    df = df[df["ask_prices"].apply(lambda x: x is not None and len(x) > 0 and x[0] is not None)]
    ch_ts += df["exchange_event_ts_us"].astype("int64").tolist()
    ch_bid += df["bid_prices"].apply(lambda x: float(x[0])).tolist()
    ch_ask += df["ask_prices"].apply(lambda x: float(x[0])).tolist()
ch_ts = np.array(ch_ts, np.int64); ch_bid = np.array(ch_bid, np.float64); ch_ask = np.array(ch_ask, np.float64)
o = np.argsort(ch_ts); ch_ts, ch_bid, ch_ask = ch_ts[o], ch_bid[o], ch_ask[o]
print(f"chronos snapshots {DAY}: {len(ch_ts)} | span {(ch_ts[-1]-ch_ts[0])/3.6e9:.1f}h", flush=True)

# --- align each cryptolake row to nearest chronos snapshot within 100ms ---
j = np.clip(np.searchsorted(ch_ts, cl_ts), 1, len(ch_ts) - 1)
dl = np.abs(ch_ts[j - 1] - cl_ts); dr = np.abs(ch_ts[j] - cl_ts)
jn = np.where(dl < dr, j - 1, j); dt = np.minimum(dl, dr)
ok = dt < 100_000  # 100ms
cb, ca = cl_bid[ok], cl_ask[ok]; hb, ha = ch_bid[jn[ok]], ch_ask[jn[ok]]
print(f"\n=== CROSS-SOURCE TOP-OF-BOOK PARITY ({DAY}) ===", flush=True)
print(f"  matched {int(ok.sum())}/{len(cl_ts)} cryptolake rows to chronos within 100ms (med dt={np.median(dt[ok])/1000:.1f}ms)", flush=True)
exact = (np.abs(cb - hb) < TICK / 2) & (np.abs(ca - ha) < TICK / 2)
dbid = np.abs(cb - hb) / TICK; dask = np.abs(ca - ha) / TICK; dmid = np.abs((cb + ca) / 2 - (hb + ha) / 2) / TICK
print(f"  top-of-book EXACT (bid&ask within 0.5 tick): {100*exact.mean():.1f}%", flush=True)
print(f"  |Δbid| ticks: med={np.median(dbid):.2f} p95={np.percentile(dbid,95):.2f} max={dbid.max():.1f}", flush=True)
print(f"  |Δask| ticks: med={np.median(dask):.2f} p95={np.percentile(dask,95):.2f} max={dask.max():.1f}", flush=True)
print(f"  |Δmid| ticks: med={np.median(dmid):.2f} p95={np.percentile(dmid,95):.2f}", flush=True)
within1 = ((dbid <= 1) & (dask <= 1)).mean()
print(f"  within 1 tick (bid&ask): {100*within1:.1f}%", flush=True)
verdict = "PARITY OK (same market; residual = snapshot-timing within 100ms)" if exact.mean() > 0.8 or within1 > 0.95 else "DIVERGENCE -- investigate (timestamp offset / different feed / depth)"
print(f"  -> {verdict}", flush=True)
