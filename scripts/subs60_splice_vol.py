#!/usr/bin/env python3
"""Splice the sampling-robust vol/momentum columns (34-39) into the existing DOGE maker-label
feature matrix, recomputed with the NEW feature_builder at the SAME decision points. Labels (pnl,
fills, rH) are unchanged (they don't depend on features). Avoids the full feats_sub60 + grid_sim
rebuild. Day-0 sanity gate: the new builder's unchanged cols (0-4) must match the original F
(confirms index alignment) or the run aborts. Output: maker_labels_pegexit_qm1_robustvol/DOGE.npz.
"""
import io, subprocess, tempfile, os, datetime, sys
import numpy as np, pyarrow.parquet as pq
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FB = "/tmp/fb_target/release/feature_builder"; SYMF = "DOGE-USDT-PERP"
SRC = "research_runs/maker_labels_pegexit_qm1/DOGE.npz"
DST = "research_runs/maker_labels_pegexit_qm1_robust2/DOGE.npz"
bk = storage.Client(project=PROJ).bucket(BUCKET)
TD = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")

z = np.load(io.BytesIO(bk.blob(SRC).download_as_bytes()), allow_pickle=True)
F = z["F"].astype(np.float32).copy(); day = z["day"]; ts = z["ts"].astype(np.int64)
ndays = int(day.max()) + 1
print(f"loaded F{F.shape}, {ndays} days", flush=True)


def dl(prefix, dst):
    n = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    return (bk.blob(n).download_to_filename(dst) or dst) if n else None


def run_day(date, dts):
    book = dl(f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={date}/", f"{TD}/b.parquet")
    trd = dl(f"raw/trades/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={date}/", f"{TD}/t.parquet")
    if not book or not trd:
        return None
    bts = pq.read_table(book, columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    idx = np.clip(np.searchsorted(bts, dts), 0, len(bts) - 1).astype(np.int64)
    np.save(f"{TD}/idx.npy", idx)
    r = subprocess.run([FB, "--depth", book, "--indices", f"{TD}/idx.npy", "--out", f"{TD}/f.npy", "--trades", trd], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FB FAIL {date}: {r.stderr[-150:]}", flush=True); return None
    return np.load(f"{TD}/f.npy").astype(np.float32)


def date_of(dts):
    return datetime.datetime.utcfromtimestamp(int(dts[0]) / 1e9).date().isoformat()


# ---- day-0 sanity gate ----
d0 = ts[day == 0]; nf0 = run_day(date_of(d0), d0)
assert nf0 is not None, "day0 build failed"
drift04 = float(np.abs(nf0[:, :5] - F[day == 0][:, :5]).max())
print(f"day0 sanity: unchanged cols[0-4] max|new-orig| = {drift04:.2e} (must be ~0)", flush=True)
assert drift04 < 1e-3, f"ALIGNMENT FAIL (cols 0-4 differ by {drift04}) -- aborting splice"
F[day == 0, :64] = nf0[:, :64]

# ---- splice all days ----
nfail = 0
for d in range(1, ndays):
    mask = day == d; dts = ts[mask]
    if len(dts) == 0:
        continue
    nf = run_day(date_of(dts), dts)
    if nf is None:
        nfail += 1; continue
    F[mask, :64] = nf[:, :64]
    if d % 50 == 0:
        print(f"  day {d}/{ndays} {date_of(dts)} spliced", flush=True)
print(f"splice done; {nfail} days failed", flush=True)

buf = io.BytesIO()
np.savez(buf, F=F, rH60=z["rH60"], rH15=z["rH15"], rH30=z["rH30"], day=day, ts=z["ts"],
         pnl_long=z["pnl_long"], pnl_short=z["pnl_short"], fill_long=z["fill_long"], fill_short=z["fill_short"],
         feat_names=z["feat_names"], meta=z["meta"])
bk.blob(DST).upload_from_string(buf.getvalue())
print(f"saved {DST} ({buf.tell()/1e6:.0f} MB)", flush=True)
