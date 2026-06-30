#!/usr/bin/env python3
"""Validate the sampling-robust feature fix BEFORE the Rust rewrite + retrain. Compares realized_vol
computed the OLD way (tick-based: Sigma of per-snapshot squared returns over N ticks) vs the NEW way
(fixed 1s-grid: returns over equal 1s intervals), from cl-sparse (1.76/s) vs chronos-dense (9/s) on
2026-06-05. If grid-RV matches cl<->chronos (corr~1, ratio~1) while tick-RV drifts, the fix concept
works -> proceed to features.rs rewrite + retrain. Also checks an OFI-sum (telescoping) feature.
"""
import numpy as np, pyarrow.parquet as pq
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; NS = 1_000_000_000
bk = storage.Client(project=PROJ).bucket(BUCKET); DAY = "2026-06-05"


def cl_book():
    n = next(b.name for b in bk.client.list_blobs(bk, prefix=f"raw/book/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/") if b.name.endswith(".parquet"))
    bk.blob(n).download_to_filename("/tmp/cl.parquet")
    t = pq.read_table("/tmp/cl.parquet", columns=["timestamp", "bid_0_price", "ask_0_price", "bid_0_size", "ask_0_size"]).to_pandas().sort_values("timestamp")
    ts = t["timestamp"].to_numpy().astype(np.int64)
    mid = (t["bid_0_price"].to_numpy() + t["ask_0_price"].to_numpy()) / 2
    ofi_raw = np.diff((t["bid_0_size"].to_numpy() - t["ask_0_size"].to_numpy()), prepend=0)
    return ts, mid, ofi_raw


def ch_book():
    ts, mid, bs, as_ = [], [], [], []
    for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/depth_snapshot/"):
        if not b.name.endswith(".parquet"):
            continue
        bk.blob(b.name).download_to_filename("/tmp/ch.parquet")
        df = pq.read_table("/tmp/ch.parquet", columns=["exchange_event_ts_us", "bid_prices", "ask_prices", "bid_qtys", "ask_qtys"]).to_pandas()
        df = df[df["exchange_event_ts_us"].notna()]
        ts.append((df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
        mid.append(((df["bid_prices"].apply(lambda x: x[0]) + df["ask_prices"].apply(lambda x: x[0])) / 2).to_numpy())
        bs.append(df["bid_qtys"].apply(lambda x: x[0]).to_numpy()); as_.append(df["ask_qtys"].apply(lambda x: x[0]).to_numpy())
    ts = np.concatenate(ts); mid = np.concatenate(mid); l0 = np.concatenate(bs) - np.concatenate(as_)
    o = np.argsort(ts); ts, mid, l0 = ts[o], mid[o], l0[o]
    return ts, mid, np.diff(l0, prepend=0)


cl_ts, cl_mid, cl_ofi = cl_book()
ch_ts, ch_mid, ch_ofi = ch_book()
lo = max(cl_ts[0], ch_ts[0]); hi = min(cl_ts[-1], ch_ts[-1])
dec = np.arange(lo + 200 * NS, hi, NS, dtype=np.int64)  # 1/s decision pts, 200s warmup
print(f"cl {len(cl_ts)} ({len(cl_ts)/((cl_ts[-1]-cl_ts[0])/NS):.1f}/s) | chronos {len(ch_ts)} ({len(ch_ts)/((ch_ts[-1]-ch_ts[0])/NS):.1f}/s) | {len(dec)} decision pts", flush=True)


def tick_rv(ts, mid, dec, ticks):  # OLD: sqrt(sum of last `ticks` per-snapshot sq returns)
    r2 = np.diff(np.log(mid)) ** 2; csq = np.concatenate([[0], np.cumsum(r2)])  # csq[i]=sum r2[:i]
    idx = np.clip(np.searchsorted(ts, dec, "right") - 1, 0, len(ts) - 1)
    lo_i = np.clip(idx - ticks, 0, None)
    return np.sqrt(np.maximum(csq[idx] - csq[lo_i], 0))


def grid_rv(ts, mid, dec, gridS, win):  # NEW: sqrt(sum of last `win` fixed-gridS-interval sq returns)
    g = np.arange(ts[0], ts[-1] + 1, int(gridS * NS), dtype=np.int64)
    gi = np.clip(np.searchsorted(ts, g, "right") - 1, 0, len(ts) - 1)
    gr2 = np.diff(np.log(mid[gi])) ** 2; csq = np.concatenate([[0], np.cumsum(gr2)])
    k = np.clip(np.searchsorted(g, dec, "right") - 1, 0, len(g) - 2)
    lo_k = np.clip(k - win, 0, None)
    return np.sqrt(np.maximum(csq[k] - csq[lo_k], 0))


def tick_ofisum(ts, ofi, dec, ticks):  # telescoping window sum
    cs = np.concatenate([[0.0], np.cumsum(ofi)]); idx = np.clip(np.searchsorted(ts, dec, "right") - 1, 0, len(ts) - 1)
    return cs[idx] - cs[np.clip(idx - ticks, 0, None)]


def time_ofisum(ts, ofi, dec, secs):  # NEW: sum over fixed time window
    cs = np.concatenate([[0.0], np.cumsum(ofi)]); hi_i = np.clip(np.searchsorted(ts, dec, "right") - 1, 0, len(ts) - 1)
    lo_i = np.clip(np.searchsorted(ts, dec - secs * NS, "right") - 1, 0, len(ts) - 1)
    return cs[hi_i] - cs[lo_i]


def cmp(name, a, b):
    m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    c = np.corrcoef(a[m], b[m])[0, 1]; ratio = np.median(b[m] / a[m])
    print(f"  {name:28} corr {c:+.3f} | median chronos/cl ratio {ratio:.2f}", flush=True)
    return c


print("\n=== realized_vol_60s: OLD tick (600 ticks) vs NEW grid (60x1s) ===", flush=True)
cmp("OLD tick-RV (600 ticks)", tick_rv(cl_ts, cl_mid, dec, 600), tick_rv(ch_ts, ch_mid, dec, 600))
cmp("NEW grid-RV (60x1s)", grid_rv(cl_ts, cl_mid, dec, 1.0, 60), grid_rv(ch_ts, ch_mid, dec, 1.0, 60))
print("=== ofi_60s: OLD tick (600 ticks) vs NEW time (60s) ===", flush=True)
cmp("OLD tick-OFIsum (600 ticks)", tick_ofisum(cl_ts, cl_ofi, dec, 600), tick_ofisum(ch_ts, ch_ofi, dec, 600))
cmp("NEW time-OFIsum (60s)", time_ofisum(cl_ts, cl_ofi, dec, 60), time_ofisum(ch_ts, ch_ofi, dec, 60))
print("\n-> fix validated if NEW corr ~>0.9 & ratio ~1 while OLD drifts", flush=True)
