#!/usr/bin/env python3
"""FULL-FEATURE rebuild of tb3s F: recompute X64 at the SAME tb3s decision ticks with ALL
FB inputs (book+trades+funding+liquidations+open-interest+eth-trades) — the robust rebuild
had silently zeroed cols 13, 14-16/55, 56-60 by omitting these inputs. Labels/btc/tod are
reused from the existing tb3s daily npzs (labels don't depend on features).
Output: research_runs/maker_labels_tb3s_full/daily/DOGE_{day}.npz (then COMBINE via
subs60_build_tb3s_labels.py with OUTSUB=research_runs/maker_labels_tb3s_full)."""
import io, os, subprocess
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RAW = "raw/{s}/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
DOGE = "DOGE-USDT-PERP"; ETH = "ETH-USDT-PERP"
FB = "/tmp/fb_target/release/feature_builder"
SRC = "research_runs/maker_labels_tb3s/daily"; DST = "research_runs/maker_labels_tb3s_full/daily"
TD = "/home/delmi/tb3s_ff"; os.makedirs(TD, exist_ok=True)
bk = storage.Client(project=PROJ).bucket(BUCKET)
CHK = {13: "funding", 14: "eth5", 56: "liq_s5", 59: "oi_d30"}


def dl(stream, sym, day, dst):
    p = RAW.format(s=stream, sym=sym, day=day)
    n = next((b.name for b in bk.client.list_blobs(bk, prefix=p) if b.name.endswith(".parquet")), None)
    if not n:
        return None
    bk.blob(n).download_to_filename(dst); return dst


days = sorted(b.name.split("_")[-1][:-4] for b in bk.client.list_blobs(bk, prefix=f"{SRC}/DOGE_") if b.name.endswith(".npz"))
done = {b.name.split("_")[-1][:-4] for b in bk.client.list_blobs(bk, prefix=f"{DST}/DOGE_") if b.name.endswith(".npz")}
todo = [d for d in days if d not in done]
print(f"[fullfeat] {len(days)} days, {len(todo)} to do", flush=True)

for i, day in enumerate(todo):
    z = dict(np.load(io.BytesIO(bk.blob(f"{SRC}/DOGE_{day}.npz").download_as_bytes())))
    ts = z["ts"].astype(np.int64)
    bp = dl("book", DOGE, day, f"{TD}/b.parquet")
    tp = dl("trades", DOGE, day, f"{TD}/t.parquet")
    if not bp or not tp:
        print(f"  {day}: no book/trades", flush=True); continue
    bt = pq.read_table(bp, columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    idx = np.searchsorted(bt, ts, "right") - 1
    if not ((idx >= 0).all() and (np.abs(bt[np.clip(idx, 0, len(bt) - 1)] - ts) <= 1_000_000).all()):
        print(f"  {day}: tick mismatch", flush=True); continue
    np.save(f"{TD}/idx.npy", idx.astype(np.int64))
    cmd = [FB, "--depth", bp, "--indices", f"{TD}/idx.npy", "--out", f"{TD}/f.npy", "--trades", tp]
    for stream, sym, flag in (("funding", DOGE, "--funding"), ("liquidations", DOGE, "--liquidations"),
                              ("open_interest", DOGE, "--open-interest"), ("trades", ETH, "--eth")):
        f = dl(stream, sym, day, f"{TD}/{flag.strip('-')}.parquet")
        if f:
            cmd += [flag, f]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {day}: FB fail {r.stderr[-160:]}", flush=True); continue
    X = np.load(f"{TD}/f.npy").astype(np.float32)
    if len(X) != len(ts):
        print(f"  {day}: len {len(X)} != {len(ts)}", flush=True); continue
    Fold = z["F"]; Fnew = Fold.copy(); Fnew[:, :64] = X   # keep btc(64-66)+tod(67-70)
    z["F"] = Fnew
    buf = io.BytesIO(); np.savez_compressed(buf, **z)
    bk.blob(f"{DST}/DOGE_{day}.npz").upload_from_string(buf.getvalue())
    if i % 40 == 0 or i == 0:
        nz = {n: f"{100*float((Fnew[:, c] != 0).mean()):.0f}%" for c, n in CHK.items()}
        print(f"  {day} ({i+1}/{len(todo)}): nonzero {nz}", flush=True)
print("[fullfeat DONE]", flush=True)
