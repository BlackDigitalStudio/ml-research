#!/usr/bin/env python3
"""Decisive test of the cross-source feature drift cause, using ONLY cryptolake data: run
feature_builder on the cl book at FULL density vs the SAME book DOWNSAMPLED (every-Kth snapshot),
evaluated at identical decision timestamps (trades held identical). If the same ~features drift,
they are BOOK-SAMPLING-SENSITIVE -> the chronos(dense) vs cryptolake(sparse) drift is explained as a
sampling-cadence artifact (data is identical), and the fix is to align the live book sampling.
"""
import os, subprocess, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; FB = "/tmp/feature_builder"; NS = 1_000_000_000
bk = storage.Client(project=PROJ).bucket(BUCKET); TD = tempfile.mkdtemp(); DAY = "2026-06-05"
K = int(os.environ.get("K", "4"))


def first(prefix, dst):
    n = next(b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet"))
    bk.blob(n).download_to_filename(dst); return dst


book = first(f"raw/book/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/", f"{TD}/b.parquet")
trd = first(f"raw/trades/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/", f"{TD}/t.parquet")
t = pq.read_table(book); cols = t.column_names
arr = {c: t[c].to_numpy() for c in cols}
o = np.argsort(arr["timestamp"].astype(np.int64), kind="stable"); arr = {c: arr[c][o] for c in cols}
bts = arr["timestamp"].astype(np.int64)
grid = np.arange(bts[0] + 120 * NS, bts[-1] - 70 * NS, NS, dtype=np.int64)  # common decision timestamps


def runfb(bts_b, bookpath, tag):
    idx = np.unique(np.clip(np.searchsorted(bts_b, grid, "right") - 1, 0, len(bts_b) - 1)).astype(np.int64)
    np.save(f"{TD}/idx_{tag}.npy", idx)
    op = f"{TD}/f_{tag}.npy"
    r = subprocess.run([FB, "--depth", bookpath, "--indices", f"{TD}/idx_{tag}.npy", "--out", op, "--trades", trd], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-200:]
    return bts_b[idx], np.load(op).astype(np.float64)

# full
td0, X0 = runfb(bts, book, "full")
# downsampled every-K
keep = np.arange(0, len(bts), K)
pq.write_table(pa.table({c: arr[c][keep] for c in cols}), f"{TD}/b_ds.parquet")
td1, X1 = runfb(bts[keep], f"{TD}/b_ds.parquet", "ds")

# match by decision timestamp
j = np.clip(np.searchsorted(td1, td0), 1, len(td1) - 1); dl = np.abs(td1[j - 1] - td0); dr = np.abs(td1[j] - td0)
jn = np.where(dl < dr, j - 1, j); ok = np.minimum(dl, dr) < 100_000_000
A = X0[ok]; B = X1[jn[ok]]; sd = X0.std(0) + 1e-9; reld = np.abs(A - B) / sd
drift = np.where(np.median(reld, 0) > 0.05)[0]
print(f"[cl self-downsample K={K}, {DAY}] full n={len(td0)} -> ds n={len(td1)} | matched {int(ok.sum())}", flush=True)
print(f"  features that DRIFT when cl is downsampled {K}x: {len(drift)}/{X0.shape[1]}", flush=True)
print(f"  drift cols: {drift.tolist()}", flush=True)
print(f"  (cross-source drift cols incl: 38,37,23,22,21,41,28,46) -> overlap = sampling-sensitive", flush=True)
print(f"  median |ΔX|/std overall: {np.median(reld):.4f} | worst-8: {[(int(c),round(float(np.median(reld[:,c])),2)) for c in np.argsort(-np.median(reld,0))[:8]]}", flush=True)
