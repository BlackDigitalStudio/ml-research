#!/usr/bin/env python3
"""HD2 FULL FEATURE rebuild for a GCP VM (96 vCPU, ADC auth) -> GCS.

Recomputes the COMPLETE engineered-feature set from raw (book + trades + funding
+ ETH) via the fast Rust `feature_builder` binary, per (symbol, day), aligned to
the SAME decision points the HD2 stream cache uses (features_v1 indices). Unlike
`features_v1` (which was BOOK-ONLY -> trade/funding/ETH cols are placeholder
zero), this restores the real directional features: trade-flow / cvd / vpin /
kyle, funding, and ETH leading signals (eth_momentum/eth_ofi/eth_leading_signal).
Cross-exchange (bybit/okx/bitget/gateio) is skipped -- raw has only BINANCE.

Output: one .npy per (sym,day) -> gs://{BUCKET}/{OUT_PREFIX}/{sym}/{day}.npy
(shape (n_dp, 59), aligned 1:1 with features_v1 indices for that day).

Same-region VM<->GCS (europe-west1 <-> EUROPE-WEST1) => egress is free.
Auth = ADC (VM service account). Idempotent (skips outputs already in GCS).

  # on the VM (after: cargo build --release --bin feature_builder; pip install
  #   numpy pyarrow google-cloud-storage):
  python hd2_feats_vm.py --workers 90                         # all 8 symbols, full days
  python hd2_feats_vm.py --symbols LTC-USDT-PERP              # subset / resume
"""
import argparse
import os
import sys
import subprocess
import tempfile
import time
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GCP_PROJECT = "project-0998ac51-36ba-445c-bc7"   # migrated to virgin.ship03 (2026-05-26)
BUCKET = "market-data-0998ac51"                  # was blackdigital-scalper-data
OUT_PREFIX = "feats_v2"            # gs://{BUCKET}/{OUT_PREFIX}/{sym}/{day}.npy
ETH_SYM = "ETH-USDT-PERP"         # secondary instrument for the eth_* leading features
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
# feature_builder binary (built in the repo on the VM via cargo build --release)
FB = os.environ.get("FEATURE_BUILDER",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "rust_ingest", "target", "release",
                                 "feature_builder"))


def _bucket():
    from google.cloud import storage
    return storage.Client(project=GCP_PROJECT).bucket(BUCKET)


def _dts(bk, pref):
    it = bk.client.list_blobs(bk, prefix=pref, delimiter="/")
    for _ in it:
        pass
    return {p.split("dt=")[1].rstrip("/") for p in it.prefixes}


def _list_days(bk, sym):
    """Buildable days = dt= present in book AND trades AND features_v1 (the
    three required inputs). funding + ETH are best-effort per day."""
    book = _dts(bk, f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/")
    trades = _dts(bk, f"raw/trades/exchange=BINANCE_FUTURES/symbol={sym}/")
    feat = _dts(bk, f"features_v1/symbol={sym}/")
    return sorted(book & trades & feat)


def _first_parquet(bk, pref):
    """Return blob name of the first *.parquet under pref, or None."""
    for b in bk.client.list_blobs(bk, prefix=pref):
        if b.name.endswith(".parquet"):
            return b.name
    return None


def _dl(bk, blob_name, dest):
    bk.blob(blob_name).download_to_filename(dest)
    return dest


def _build_one(task):
    sym, day = task
    try:
        bk = _bucket()
        out_blob = f"{OUT_PREFIX}/{sym}/{day}.npz"     # compressed before Modal (user)
        if bk.blob(out_blob).exists():
            return (sym, day, "skip")
        with tempfile.TemporaryDirectory() as td:
            # required inputs: book (depth), trades, features_v1 indices
            depth = _dl(bk, f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/"
                            f"dt={day}/1.snappy.parquet", os.path.join(td, "book.parquet"))
            tr_blob = _first_parquet(bk, f"raw/trades/exchange=BINANCE_FUTURES/"
                                         f"symbol={sym}/dt={day}/")
            if tr_blob is None:
                return (sym, day, "no_trades")
            trades = _dl(bk, tr_blob, os.path.join(td, "trades.parquet"))
            idx = os.path.join(td, "indices.npy")
            with open(idx, "wb") as f:
                f.write(bk.blob(f"features_v1/symbol={sym}/dt={day}/"
                                f"indices.npy").download_as_bytes())
            out = os.path.join(td, "feat.npy")
            cmd = [FB, "--depth", depth, "--indices", idx, "--out", out,
                   "--trades", trades]
            # best-effort optional inputs (funding + ETH secondary trades)
            fund_blob = _first_parquet(bk, f"raw/funding/exchange=BINANCE_FUTURES/"
                                           f"symbol={sym}/dt={day}/")
            if fund_blob:
                cmd += ["--funding", _dl(bk, fund_blob, os.path.join(td, "fund.parquet"))]
            eth_blob = _first_parquet(bk, f"raw/trades/exchange=BINANCE_FUTURES/"
                                         f"symbol={ETH_SYM}/dt={day}/")
            if eth_blob:
                cmd += ["--eth", _dl(bk, eth_blob, os.path.join(td, "eth.parquet"))]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return (sym, day, f"FB_ERR {r.stderr.strip()[:90]}")
            # compress before upload (feature_builder writes raw .npy)
            import numpy as np
            arr = np.load(out)
            outz = os.path.join(td, "feat.npz")
            np.savez_compressed(outz, feat=arr.astype(np.float32))
            bk.blob(out_blob).upload_from_filename(outz)
        return (sym, day, "built")
    except Exception as e:
        return (sym, day, f"ERR {type(e).__name__}: {str(e)[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMS)
    ap.add_argument("--workers", type=int, default=90)
    a = ap.parse_args()
    assert os.path.exists(FB), f"feature_builder not found at {FB} (cargo build --release --bin feature_builder)"
    bk = _bucket()
    tasks = []
    for s in a.symbols:
        ds = _list_days(bk, s)
        print(f"{s}: {len(ds)} buildable days", flush=True)
        tasks += [(s, d) for d in ds]
    print(f"TOTAL {len(tasks)} symbol-days | workers={a.workers} "
          f"| out=gs://{BUCKET}/{OUT_PREFIX}/", flush=True)
    t0 = time.time(); built = skip = err = 0; errs = []
    with mp.Pool(a.workers) as pool:
        for i, (sym, day, st) in enumerate(pool.imap_unordered(_build_one, tasks, chunksize=4)):
            if st == "built":
                built += 1
            elif st == "skip":
                skip += 1
            else:
                err += 1
                if len(errs) < 30:
                    errs.append(f"{sym}/{day}: {st}")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)}  built={built} skip={skip} err={err} "
                      f"| {time.time()-t0:.0f}s", flush=True)
    print(f"DONE built={built} skip={skip} err={err} in {time.time()-t0:.0f}s", flush=True)
    for e in errs:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
