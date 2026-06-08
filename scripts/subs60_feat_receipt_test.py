#!/usr/bin/env python3
"""Isolate whether the cross-source feature drift is driven by receipt_timestamp / sequence_number
(columns set differently in the chronos conversion). Uses ONLY cryptolake data: run feature_builder
on the cl book as-is vs the SAME book with receipt_timestamp=timestamp and sequence_number=arange.
Any feature that changes depends on receipt/sequence -> that's the conversion-driven drift, fixable.
"""
import os, subprocess, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; FB = "/tmp/feature_builder"; NS = 1_000_000_000
bk = storage.Client(project=PROJ).bucket(BUCKET); TD = tempfile.mkdtemp()
DAY = "2026-06-05"


def first(prefix, dst):
    n = next(b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet"))
    bk.blob(n).download_to_filename(dst); return dst


book = first(f"raw/book/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/", f"{TD}/b.parquet")
trd = first(f"raw/trades/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/", f"{TD}/t.parquet")
t = pq.read_table(book)
ts = t["timestamp"].to_numpy().astype(np.int64); o = np.argsort(ts, kind="stable")
ts = ts[o]
grid = np.arange(ts[0] + 120 * NS, ts[-1] - 70 * NS, NS, dtype=np.int64)
idx = np.unique(np.clip(np.searchsorted(ts, grid, "right") - 1, 0, len(ts) - 1)).astype(np.int64)
np.save(f"{TD}/idx.npy", idx)


def runfb(bookpath, tag):
    op = f"{TD}/f_{tag}.npy"
    r = subprocess.run([FB, "--depth", bookpath, "--indices", f"{TD}/idx.npy", "--out", op, "--trades", trd], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-200:]
    return np.load(op).astype(np.float64)


X0 = runfb(book, "orig")
# modified book: receipt_timestamp = timestamp, sequence_number = arange (chronos-conversion convention)
d = {c: t[c].to_numpy() for c in t.column_names}
if "receipt_timestamp" in d:
    d["receipt_timestamp"] = d["timestamp"].astype(np.int64)
if "sequence_number" in d:
    d["sequence_number"] = np.arange(len(d["timestamp"]), dtype=np.int64)
pq.write_table(pa.table({c: d[c] for c in t.column_names}), f"{TD}/b_mod.parquet")
X1 = runfb(f"{TD}/b_mod.parquet", "mod")

sd = X0.std(0) + 1e-9
reld = np.abs(X0 - X1) / sd
changed = np.where(np.median(reld, 0) > 0.02)[0]
print(f"[receipt/sequence isolation, {DAY}] X{X0.shape}", flush=True)
print(f"  features CHANGED by receipt+sequence override: {len(changed)}/{X0.shape[1]}", flush=True)
print(f"  changed cols: {changed.tolist()}", flush=True)
print(f"  (cross-source worst-drift cols were: 38,37,23,22,21,41,28,46)", flush=True)
print(f"  max median |ΔX|/std among changed: {float(np.median(reld[:,changed],0).max()) if len(changed) else 0:.3f}", flush=True)
