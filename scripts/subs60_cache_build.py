#!/usr/bin/env python3
"""Build the sub-60s COMBINED 2-stream cache for HD2 Mamba (#3).

Uses the EXTENDED Rust feature_builder (--lob-out) for the fast 80-ch LOB stream;
Python only does cheap glue (t0 map, BTC-lead, time-of-day, labels). One npz per
(symbol, day), decision grid = feats_sub60 dense 1s, H=60s:
  lob    (n_ticks,80) f16   stream-1 (Rust): [bid_p|bid_s|ask_p|ask_s], (p-mid)/mid, sign*log1p|s|
  t0     (n_dp,)      i64    decision book-index = searchsorted(book_ts, dtd)
  feat   (n_dp,F=71)  f32    stream-2: feats_sub60 X(64) + signed BTC-lead{5,30,60}s + ToD{4}
  dtd    (n_dp,)      i64    decision ts (ns, 1s grid)
  rH60   (n_dp,)      f32    forward 60s logret (bp)
  y60    (n_dp,)      i8     NON-FLAT = |rH60|>=13bp   (Model A target)
  updn   (n_dp,)      i8     UP = rH60>0               (Model B target)
  v60    (n_dp,)      bool   valid; meta json

Run on VM:  python3 subs60_cache_build.py --symbols DOGE-USDT-PERP ETH-USDT-PERP LINK-USDT-PERP
"""
import argparse, io, json, os, subprocess, tempfile
import numpy as np
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FEATS = "feats_sub60"; RAWBOOK = "raw/book/exchange=BINANCE_FUTURES"
OUTPREF = "hd2_sub60_cache"; H = 60; THR = 13.0; NS = 1_000_000_000
FB = "/tmp/rust_ingest/target/release/feature_builder"
bk = storage.Client(project=PROJ).bucket(BUCKET)


def load_btc_mid(workers, max_days=None):
    blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{FEATS}/BTC-USDT-PERP/")
                   if b.name.endswith(".npz"))
    if max_days and len(blobs) > max_days:
        blobs = blobs[::max(1, len(blobs)//max_days)][:max_days]
    def fetch(name):
        return np.load(io.BytesIO(bk.blob(name).download_as_bytes()))
    tds, mids = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(fetch, blobs):
            tds.append(d["td"].astype(np.int64)); mids.append(d["mid"].astype(np.float64))
    td = np.concatenate(tds); mid = np.concatenate(mids); o = np.argsort(td, kind="stable")
    return td[o], mid[o]


def btc_lead(dtd, bt, bm):
    nb = len(bt); i = np.clip(np.searchsorted(bt, dtd, side="right")-1, 0, nb-1)
    out = []
    for W in (5, 30, 60):
        j = np.clip(np.searchsorted(bt, dtd-int(W*NS), side="right")-1, 0, nb-1)
        a = bm[j]; b = bm[i]
        out.append(np.where((a > 0) & (b > 0),
                   np.log(np.where(a > 0, b/np.where(a > 0, a, 1.0), 1.0)), 0.0)*1e4)
    return np.stack(out, axis=1).astype(np.float32)


def time_of_day(dtd):
    h = ((dtd/NS) % 86400.0)/3600.0; hf = h % 8.0
    return np.stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                     np.sin(2*np.pi*hf/8), np.cos(2*np.pi*hf/8)], axis=1).astype(np.float32)


def build_day(sym, day, bt, bm, tmp):
    fb = bk.blob(f"{FEATS}/{sym}/{day}.npz")
    pbk = bk.blob(f"{RAWBOOK}/symbol={sym}/dt={day}/1.snappy.parquet")
    if not fb.exists() or not pbk.exists():
        return None
    fz = np.load(io.BytesIO(fb.download_as_bytes()))
    dtd = fz["td"].astype(np.int64); X = fz["X"].astype(np.float32)
    rH60 = fz["rH_60"].astype(np.float32); v60 = fz["valid_60"].astype(bool)
    if len(dtd) < 200:
        return None
    bp = os.path.join(tmp, f"book_{sym}_{day}.parquet")
    lp = os.path.join(tmp, f"lob_{sym}_{day}.npy")
    pbk.download_to_filename(bp)
    r = subprocess.run([FB, "--depth", bp, "--indices", "/tmp/_noidx", "--lob-out", lp],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(lp):
        os.path.exists(bp) and os.remove(bp)
        return {"err": f"rust:{r.stderr[-160:]}", "day": day}
    lob = np.load(lp).astype(np.float16)                       # (n_ticks,80)
    book_ts = pq.read_table(bp, columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    n_ticks = len(book_ts)
    t0 = np.clip(np.searchsorted(book_ts, dtd, side="right")-1, 0, n_ticks-1).astype(np.int64)
    feat = np.concatenate([X, btc_lead(dtd, bt, bm), time_of_day(dtd)], axis=1).astype(np.float32)
    y60 = ((np.abs(rH60) >= THR) & v60).astype(np.int8); updn = (rH60 > 0).astype(np.int8)
    meta = {"symbol": sym, "day": day, "n_ticks": int(n_ticks), "n_dp": int(len(dtd)),
            "F": int(feat.shape[1]), "H": H,
            "nonflat_frac": float(y60[v60].mean()) if v60.any() else 0.0}
    buf = io.BytesIO()
    np.savez(buf, lob=lob, t0=t0, feat=feat, dtd=dtd, rH60=rH60, y60=y60, updn=updn,
             v60=v60, meta=json.dumps(meta)); buf.seek(0)
    bk.blob(f"{OUTPREF}/{sym}/{day}.npz").upload_from_file(buf)
    for f in (bp, lp):
        os.path.exists(f) and os.remove(f)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    open("/tmp/_noidx", "wb").close()    # placeholder path; not read in LOB-only mode
    print("[loading global BTC mid...]", flush=True)
    bt, bm = load_btc_mid(8, a.max_days or None)
    print(f"[BTC: {len(bt)} ticks]", flush=True)
    tmp = tempfile.mkdtemp(prefix="cache_", dir="/tmp")
    for sym in a.symbols:
        days = sorted(b.name.split("/")[-1][:-4] for b in
                      bk.client.list_blobs(bk, prefix=f"{FEATS}/{sym}/") if b.name.endswith(".npz"))
        if a.max_days and len(days) > a.max_days:
            days = days[::max(1, len(days)//a.max_days)][:a.max_days]
        print(f"\n=== {sym}: {len(days)} days ===", flush=True)
        done = 0; nf = []
        def work(day):
            try:
                return build_day(sym, day, bt, bm, tmp)
            except Exception as e:
                return {"err": str(e), "day": day}
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for m in ex.map(work, days):
                if m and "err" not in m:
                    done += 1; nf.append(m["nonflat_frac"])
                    if done % 50 == 0:
                        print(f"  {sym}: {done}/{len(days)} (nonflat~{np.mean(nf):.3f})", flush=True)
                elif m and "err" in m:
                    print(f"  ERR {m['day']}: {m['err']}", flush=True)
        print(f"  {sym} DONE: {done}/{len(days)} built, mean nonflat={np.mean(nf) if nf else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
