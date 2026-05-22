#!/usr/bin/env python3
"""HD2 500-day stream builder on Modal CPU (one-time; HD2 rev1 pre-reg).

Reads raw/book parquet + features_v1 indices from GCS, builds the
per-(symbol,day) streaming cache (fp16 80-ch LOB + 6 frozen globals + per-H
first-passage labels/rH/R1-reach via the FROZEN hd1_seq_core), writes one .npz
per symbol-day to the Modal Volume. Idempotent: a day with a DONE marker is
skipped, so preemption/re-run resumes cheaply. Parallel via .map over
(symbol, day). The training hot path is GPU-streamed elsewhere; this build is
one-time, so Python-on-many-core-CPU (frozen-core parity) is the right tool.

  modal run scripts/hd2_build_modal.py                 # plan + validate(1 day)
  modal run scripts/hd2_build_modal.py --full          # full 500-day x 2-sym build
"""
import io
import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
SYMS = ["SOL-USDT-PERP", "LTC-USDT-PERP"]
N_DAYS = 500
HS = (180, 600, 1800)

IMG = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.4", "pyarrow==17.0.0", "scipy",
                 "scikit-learn", "google-cloud-storage")
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
)
VOL = modal.Volume.from_name("hd2-cache", create_if_missing=True)
MNT = "/cache"
app = modal.App("hd2-build")


def _gcs():
    """Bucket from the bare ya29 OAuth token in the hd1-gcp secret
    (mirrors hd1_seq_modal._gcs; bare token => no refresh)."""
    from google.cloud import storage
    tok = "".join(os.environ["GCP_ACCESS_TOKEN"].split())
    import google.oauth2.credentials

    class _Static(google.oauth2.credentials.Credentials):
        def refresh(self, request):
            return
    cl = storage.Client(project=GCP_PROJECT, credentials=_Static(token=tok))
    return cl.bucket(BUCKET)


def _list_days(bk, sym):
    pref = f"features_v1/symbol={sym}/"
    it = bk.client.list_blobs(bk, prefix=pref, delimiter="/")
    for _ in it:
        pass
    return sorted(p.split("dt=")[1].rstrip("/") for p in it.prefixes)


@app.function(image=IMG, timeout=900,
              secrets=[modal.Secret.from_name("hd1-gcp")])
def plan_window():
    """Most-recent N_DAYS COMMON gap-free days (both symbols). Intersection
    drops any day absent from either symbol -> handles source gaps."""
    bk = _gcs()
    per = {s: set(_list_days(bk, s)) for s in SYMS}
    common = sorted(set.intersection(*per.values()))
    window = common[-N_DAYS:]
    out = {"n_common": len(common), "n_window": len(window),
           "start": window[0], "end": window[-1],
           "per_symbol_total": {s: len(per[s]) for s in SYMS}}
    print("PLAN " + json.dumps(out))
    return window, out


@app.function(image=IMG, cpu=2.0, timeout=1200, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")], retries=3)
def build_one(task):
    """Build one (symbol, day) -> /cache/hd2/{sym}/{day}.npz. Idempotent."""
    import numpy as np
    import sys
    sys.path.insert(0, "/root/scripts")
    import hd2_stream_build as B

    sym, day = task["sym"], task["day"]
    outdir = f"{MNT}/hd2/{sym}"
    outpath = f"{outdir}/{day}.npz"
    if os.path.exists(outpath):
        return {"sym": sym, "day": day, "skip": True}
    os.makedirs(outdir, exist_ok=True)

    bk = _gcs()
    pq_blob = (f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
               f"1.snappy.parquet")
    idx_blob = f"features_v1/symbol={sym}/dt={day}/indices.npy"
    pq_local = f"/tmp/{sym}_{day}.parquet"
    bk.blob(pq_blob).download_to_filename(pq_local)
    idx_bytes = bk.blob(idx_blob).download_as_bytes()
    idx_local = f"/tmp/{sym}_{day}_idx.npy"
    with open(idx_local, "wb") as f:
        f.write(idx_bytes)

    meta = B.build_day(pq_local, idx_local, outpath, sym, day, Hs=HS)
    os.remove(pq_local); os.remove(idx_local)
    VOL.commit()
    return {"sym": sym, "day": day, "n_dp": meta["n_dp"],
            "n_ticks": meta["n_ticks"]}


@app.local_entrypoint()
def main(full: bool = False):
    window, info = plan_window.remote()
    print("window:", info)
    tasks = [{"sym": s, "day": d} for s in SYMS for d in window]
    if not full:
        # validate: build the single most-recent SOL day, confirm GCS+build
        t = {"sym": "SOL-USDT-PERP", "day": window[-1]}
        print("VALIDATE one day:", t)
        print(json.dumps(build_one.remote(t), default=str))
        print(f"(validate ok; {len(tasks)} tasks queued for --full)")
        return
    print(f"FULL build: {len(tasks)} (symbol,day) tasks")
    done = 0
    res = list(build_one.map(tasks, order_outputs=False))
    built = [r for r in res if not r.get("skip")]
    skipped = [r for r in res if r.get("skip")]
    print(f"BUILD_DONE built={len(built)} skipped={len(skipped)} "
          f"total={len(res)}")
