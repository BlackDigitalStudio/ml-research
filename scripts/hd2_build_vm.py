#!/usr/bin/env python3
"""HD2 ALL-SYMBOLS stream builder for a GCP VM (96 vCPU, ADC auth) -> GCS.

Reads raw/book parquet + features_v1 indices from GCS, builds per-(symbol,day)
80-ch LOB streams (FROZEN hd2_stream_build.build_day) + midts (ts+mid for
train-time relabel), writes one .npz per (sym,day) to a GCS prefix.
Multiprocessing over (symbol,day); idempotent (skips outputs already in GCS).
Auth = Application Default Credentials (the VM service account) -- no Modal
secret. Streams then hydrate GCS -> Modal Volume for H100 training.

  # on the VM (after: pip install numpy pyarrow scipy scikit-learn google-cloud-storage):
  python hd2_build_vm.py --what both --workers 90              # all 8 symbols, full days
  python hd2_build_vm.py --symbols SOL-USDT-PERP --what streams # subset / resume
"""
import argparse
import os
import sys
import tempfile
import time
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
OUT_PREFIX = "hd2_cache_v1"        # gs://{BUCKET}/{OUT_PREFIX}/{streams,midts}/{sym}/{day}.npz
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
HS = (180, 600, 1800)


def _bucket():
    from google.cloud import storage
    return storage.Client(project=GCP_PROJECT).bucket(BUCKET)


def _list_days(bk, sym):
    """Buildable days = dt= present in BOTH raw/book AND features_v1."""
    def _dts(pref):
        it = bk.client.list_blobs(bk, prefix=pref, delimiter="/")
        for _ in it:
            pass
        return {p.split("dt=")[1].rstrip("/") for p in it.prefixes}
    book = _dts(f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/")
    feat = _dts(f"features_v1/symbol={sym}/")
    return sorted(book & feat)


def _build_one(task, what):
    sym, day = task
    try:
        bk = _bucket()
        s_blob = f"{OUT_PREFIX}/streams/{sym}/{day}.npz"
        m_blob = f"{OUT_PREFIX}/midts/{sym}/{day}.npz"
        need_s = what in ("streams", "both") and not bk.blob(s_blob).exists()
        need_m = what in ("midts", "both") and not bk.blob(m_blob).exists()
        if not need_s and not need_m:
            return (sym, day, "skip")
        pq_blob = (f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
                   f"1.snappy.parquet")
        with tempfile.TemporaryDirectory() as td:
            pql = os.path.join(td, "book.parquet")
            bk.blob(pq_blob).download_to_filename(pql)
            if need_s:
                import hd2_stream_build as B
                import numpy as np
                idxl = os.path.join(td, "idx.npy")
                with open(idxl, "wb") as f:
                    f.write(bk.blob(f"features_v1/symbol={sym}/dt={day}/"
                                    f"indices.npy").download_as_bytes())
                raw = os.path.join(td, "s_raw.npz")
                B.build_day(pql, idxl, raw, sym, day, Hs=HS)   # frozen (np.savez)
                d = dict(np.load(raw, allow_pickle=True))       # -> recompress
                outl = os.path.join(td, "s.npz")
                np.savez_compressed(outl, **d)
                bk.blob(s_blob).upload_from_filename(outl)
            if need_m:
                import numpy as np
                import pyarrow.parquet as pqf
                t = pqf.read_table(pql, columns=["timestamp", "bid_0_price",
                                                 "ask_0_price"])
                ts = t["timestamp"].to_numpy().astype(np.int64)
                mid = (0.5 * (t["bid_0_price"].to_numpy()
                              + t["ask_0_price"].to_numpy())).astype(np.float32)
                outm = os.path.join(td, "m.npz")
                np.savez_compressed(outm, ts=ts, mid=mid)
                bk.blob(m_blob).upload_from_filename(outm)
        return (sym, day, "built")
    except Exception as e:
        return (sym, day, f"ERR {type(e).__name__}: {str(e)[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMS)
    ap.add_argument("--what", choices=["streams", "midts", "both"], default="both")
    ap.add_argument("--workers", type=int, default=90)
    a = ap.parse_args()
    bk = _bucket()
    tasks = []
    for s in a.symbols:
        ds = _list_days(bk, s)
        print(f"{s}: {len(ds)} buildable days", flush=True)
        tasks += [(s, d) for d in ds]
    print(f"TOTAL {len(tasks)} symbol-days | workers={a.workers} | what={a.what} "
          f"| out=gs://{BUCKET}/{OUT_PREFIX}/", flush=True)
    t0 = time.time(); built = skip = err = 0; errs = []
    with mp.Pool(a.workers) as pool:
        for i, (sym, day, st) in enumerate(pool.imap_unordered(
                partial(_build_one, what=a.what), tasks, chunksize=4)):
            if st == "built":
                built += 1
            elif st == "skip":
                skip += 1
            else:
                err += 1
                if len(errs) < 20:
                    errs.append(f"{sym}/{day}: {st}")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)} built={built} skip={skip} err={err} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"BUILD_DONE built={built} skip={skip} err={err} total={len(tasks)} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    for e in errs:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
