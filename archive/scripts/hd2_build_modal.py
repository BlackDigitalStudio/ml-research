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
GCP_PROJECT = "project-0998ac51-36ba-445c-bc7"   # migrated to virgin.ship03 (2026-05-26)
BUCKET = "market-data-0998ac51"                  # was blackdigital-scalper-data
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


@app.function(image=IMG, cpu=2.0, timeout=1200, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")], retries=3)
def build_mid_one(task):
    """Per-(symbol,day) -> /cache/midts/{sym}/{day}.npz = {ts(i64), mid(f32)}
    per tick (HD2 rev5 unlock: enables train-time relabeling at any barrier/
    target-form). Reads only timestamp/bid_0/ask_0 from raw/book. Idempotent."""
    import numpy as np
    import pyarrow.parquet as pq
    sym, day = task["sym"], task["day"]
    outdir = f"{MNT}/midts/{sym}"
    outpath = f"{outdir}/{day}.npz"
    if os.path.exists(outpath):
        return {"sym": sym, "day": day, "skip": True}
    os.makedirs(outdir, exist_ok=True)
    bk = _gcs()
    pq_blob = (f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
               f"1.snappy.parquet")
    pq_local = f"/tmp/{sym}_{day}_m.parquet"
    bk.blob(pq_blob).download_to_filename(pq_local)
    t = pq.read_table(pq_local, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts = t["timestamp"].to_numpy().astype(np.int64)
    mid = (0.5 * (t["bid_0_price"].to_numpy() + t["ask_0_price"].to_numpy())
           ).astype(np.float32)
    np.savez(outpath, ts=ts, mid=mid)
    os.remove(pq_local)
    VOL.commit()
    return {"sym": sym, "day": day, "n": int(len(ts))}


@app.function(image=IMG, cpu=4.0, timeout=3600, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")], retries=2)
def hydrate_one(task):
    """Download one (symbol, kind) from gs://{BUCKET}/hd2_cache_v1/{kind}/{sym}/
    to the Volume: streams -> /cache/hd2/{sym}/, midts -> /cache/midts/{sym}/.
    Idempotent (skips files already on the Volume). Compressed npz pass through."""
    import os
    sym, kind = task["sym"], task["kind"]      # kind in {streams, midts}
    dest = "hd2" if kind == "streams" else "midts"
    outdir = f"{MNT}/{dest}/{sym}"
    os.makedirs(outdir, exist_ok=True)
    bk = _gcs()
    n = skip = 0
    for blob in bk.client.list_blobs(bk, prefix=f"hd2_cache_v1/{kind}/{sym}/"):
        if not blob.name.endswith(".npz"):
            continue
        outp = f"{outdir}/{blob.name.split('/')[-1]}"
        if os.path.exists(outp):
            skip += 1
            continue
        blob.download_to_filename(outp)
        n += 1
    VOL.commit()
    print(f"HYDRATE {sym}/{kind}: dl={n} skip={skip}")
    return {"sym": sym, "kind": kind, "dl": n, "skip": skip}


@app.function(image=IMG, cpu=4.0, timeout=3600, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")], retries=2)
def hydrate_feats(task):
    """LEGACY 49-feature restore: download features_v1/symbol={sym}/dt=*/
    {features,indices}.npy from GCS -> /cache/feats/{sym}/{day}.{features,indices}.npy.
    Idempotent. features_v1 decision-points == the HD2 cache decision-points
    (same indices the stream build read), so the trainer aligns by select_decision_idx.
    59 cols: 46 real microstructure + 13 placeholder-zero (live-only: ETH/cross-exch/
    funding/cancel — absent in Cryptolake). Feeds the two-head direction model stream 2."""
    import os
    sym = task["sym"]
    outdir = f"{MNT}/feats/{sym}"
    os.makedirs(outdir, exist_ok=True)
    bk = _gcs()
    n = skip = 0
    for blob in bk.client.list_blobs(bk, prefix=f"feats_v2/{sym}/"):
        if not blob.name.endswith(".npz"):
            continue
        outp = f"{outdir}/{blob.name.split('/')[-1]}"   # {day}.npz
        if os.path.exists(outp):
            skip += 1
            continue
        blob.download_to_filename(outp)
        n += 1
    VOL.commit()
    print(f"HYDRATE_FEATS {sym}: dl={n} skip={skip}")
    return {"sym": sym, "dl": n, "skip": skip}


ALL_SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
            "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]


@app.local_entrypoint()
def main(full: bool = False, mid: bool = False, hydrate: bool = False,
         midts_ltc: bool = False, feats: bool = False):
    import json
    if feats:
        # LEGACY 49-feature restore: hydrate features_v1 (8 syms) -> /cache/feats.
        tasks = [{"sym": s} for s in ALL_SYMS]
        handles = [hydrate_feats.spawn(t) for t in tasks]
        print(f"HYDRATE_FEATS SPAWNED {len(handles)} syms gs://{BUCKET}/features_v1 "
              f"-> /cache/feats:")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  {t['sym']}")
        return
    if midts_ltc:
        # RL side-experiment: need LTC price path (ts+mid per tick). Streams were
        # hydrated without midts (terminal fast-path), so pull LTC midts only.
        t = {"sym": "LTC-USDT-PERP", "kind": "midts"}
        h = hydrate_one.spawn(t)
        print(f"MIDTS_LTC SPAWNED {h.object_id}  LTC midts -> /cache/midts/LTC-USDT-PERP")
        return
    if hydrate:
        # streams ONLY: capacity sweeps use target="terminal" (cached rH in the
        # stream npz via the fast-path), so midts is not needed -> ~half data/time.
        tasks = [{"sym": s, "kind": k} for s in ALL_SYMS
                 for k in ("streams",)]
        handles = [hydrate_one.spawn(t) for t in tasks]
        print(f"HYDRATE_GCS SPAWNED {len(handles)} (sym,kind) gs://{BUCKET}/"
              f"hd2_cache_v1 -> Volume (poll /cache/hd2 + /cache/midts):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  {t['sym']}/{t['kind']}")
        return
    window, info = plan_window.remote()
    print("window:", info)
    tasks = [{"sym": s, "day": d} for s in SYMS for d in window]
    fn = build_mid_one if mid else build_one
    label = "MID" if mid else "STREAM"
    if not full:
        t = {"sym": "SOL-USDT-PERP", "day": window[-1]}
        print(f"VALIDATE {label} one day:", t)
        print(json.dumps(fn.remote(t), default=str))
        print(f"(validate ok; {len(tasks)} tasks queued for --full)")
        return
    print(f"FULL {label} build: {len(tasks)} (symbol,day) tasks")
    res = list(fn.map(tasks, order_outputs=False))
    built = [r for r in res if not r.get("skip")]
    skipped = [r for r in res if r.get("skip")]
    print(f"BUILD_DONE built={len(built)} skipped={len(skipped)} total={len(res)}")
