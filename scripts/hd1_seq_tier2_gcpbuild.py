#!/usr/bin/env python3
"""HD1 Tier-2 (rev45) standalone GCP-VM cache builder — MAX_L=1536.

Runs the FROZEN HD1-seq build/reduce numeric path on a plain GCE VM
(NOT Modal): the GCS->VM read is intra-region (bucket is europe-west1),
which eliminates the dominant historical cross-cloud egress. This file
changes ONLY the execution location + sink; the numeric content is
byte-identical to the frozen pipeline (hd1_seq_modal.build_symbol_day /
reduce_symbol): same Rust binary `hd1_seq_build --max-l L`, same npz
schema, same window rule (_list_days/_window, FLOOR/N_DAYS), same
hd1_seq_core.honest_split. rev25 reimplementation clause: divergence is
a bug, not a DOF. The frozen hd1_seq_modal.py is untouched.

rev45 frozen scope honored here:
  * MAX_L = 1536 (the single superset pack; L in {512,1024} are
    bit-exact right-causal slices at train time, validated by the
    Tier-2 runner's parity guard new-1536[:,-512:] == frozen 512 pack).
  * Window plan computed over ALL 4 SYMS (SOL included) so winlo/winhi
    stay byte-identical to the frozen comparability invariant; only the
    BUILD is restricted to the 3 symbols the rev45 cells use
    (BTC/ETH/LTC) -- each symbol is built independently so this changes
    no built symbol's content.
  * Output written to packed_l1536/{sym}.npz (a NEW path) and staged to
    gs://<bucket>/hd1seq_tier2_pack/l1536/ -- the frozen 512 pack is
    NOT clobbered.

Usage (on the VM, repo root):
  python3 scripts/hd1_seq_tier2_gcpbuild.py --max-l 1536 \
      --build-syms BTC-USDT-PERP,ETH-USDT-PERP,LTC-USDT-PERP \
      --stage gs://blackdigital-scalper-data/hd1seq_tier2_pack/l1536

Resumable: per-day shards are skip-if-exists; reduce is skip-if-packed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# --- frozen constants (verbatim: hd1_seq_modal.py) -----------------------
BUCKET = "blackdigital-scalper-data"
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"
SYMS = ["SOL-USDT-PERP", "BTC-USDT-PERP", "ETH-USDT-PERP", "LTC-USDT-PERP"]
HS = (180, 300, 600)
FLOOR = "2025-05-09"
N_DAYS = 360

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts import hd1_seq_core as C  # noqa: E402  (frozen honest_split)


def _gcs():
    """GCS bucket via ADC / VM metadata SA (storage.Client default).

    On the GCE VM this resolves the attached service account from the
    metadata server (refreshable, no TTL problem). Falls back to the
    repo's env-token pattern if ADC is absent."""
    from google.cloud import storage
    try:
        return storage.Client(project=GCP_PROJECT).bucket(BUCKET)
    except Exception:
        pass
    # fallback: mirror hd1_seq_modal._gcs env-token handling
    for name in ("GCP_ACCESS_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY"):
        v = os.environ.get(name)
        if not v:
            continue
        if name == "GCP_SA_KEY_B64":
            import base64
            v = base64.b64decode(v).decode()
        s = "".join(v.split())
        if s.startswith("ya29.") or s.startswith("ya29_"):
            import google.oauth2.credentials as goc

            class _Static(goc.Credentials):
                def refresh(self, request):
                    return
            return storage.Client(
                project=GCP_PROJECT,
                credentials=_Static(token=s)).bucket(BUCKET)
        if s[:1] == "{":
            info = json.loads(v)
            if info.get("type") == "authorized_user":
                import google.oauth2.credentials as goc
                return storage.Client(
                    project=GCP_PROJECT,
                    credentials=goc.Credentials.from_authorized_user_info(
                        info)).bucket(BUCKET)
            return storage.Client.from_service_account_info(
                info, project=GCP_PROJECT).bucket(BUCKET)
    raise RuntimeError("no GCS credential (ADC/metadata or env token)")


def _list_days(bk, sym):
    """Verbatim hd1_seq_modal._list_days (FROZEN)."""
    pref = f"features_v1/symbol={sym}/"
    it = bk.client.list_blobs(bk, prefix=pref, delimiter="/")
    for _ in it:
        pass
    return sorted(p.split("dt=")[1].rstrip("/") for p in it.prefixes)


def _window(per):
    """Verbatim hd1_seq_modal._window (FROZEN comparability invariant).

    Computed over ALL 4 SYMS so winlo/winhi are byte-identical to the
    frozen 512-pack window regardless of which symbols we then build."""
    if not all(per.values()):
        return None, None, {}
    start = max([FLOOR] + [d[0] for d in per.values()])
    end = min(d[-1] for d in per.values())
    lo_cal = (dt.date.fromisoformat(end)
              - dt.timedelta(days=N_DAYS - 1)).isoformat()
    winlo = max(start, lo_cal)
    psd = {s: [d for d in per[s] if winlo <= d <= end] for s in SYMS}
    return winlo, end, psd


def _build_symbol_day(bk, rust_bin, sym, day, day_ord, max_l, shard_dir):
    """Verbatim numeric path of hd1_seq_modal.build_symbol_day, off-Modal.

    Only the sink (local shard_dir instead of a Modal Volume) and the
    --max-l value differ; the Rust invocation, dtypes and npz schema are
    byte-identical to the frozen function."""
    os.makedirs(shard_dir, exist_ok=True)
    shard = f"{shard_dir}/{day_ord:04d}_{day}.npz"
    if os.path.exists(shard):
        return {"sym": sym, "day": day, "cached": True}

    pref = f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
    blobs = sorted((b for b in bk.client.list_blobs(bk, prefix=pref)
                    if b.name.endswith(".parquet")), key=lambda b: b.name)
    if not blobs:
        return {"sym": sym, "day": day, "error": "no book parquet"}

    with tempfile.TemporaryDirectory() as td:
        bookf = []
        for n, b in enumerate(blobs):
            p = f"{td}/book_{n:04d}.parquet"
            b.download_to_filename(p)
            bookf.append(p)
        idxf = f"{td}/indices.npy"
        bk.blob(f"features_v1/symbol={sym}/dt={day}/indices.npy"
                ).download_to_filename(idxf)
        odir = f"{td}/out"
        r = subprocess.run(
            [rust_bin, "--book", *bookf, "--indices", idxf,
             "--out-dir", odir, "--max-l", str(max_l)],
            capture_output=True, text=True)
        if r.returncode != 0:
            return {"sym": sym, "day": day,
                    "error": f"rust rc={r.returncode}: {r.stderr[-400:]}"}

        i = np.load(f"{odir}/i.npy").astype(np.int64)
        if i.size == 0:
            np.savez_compressed(shard, empty=True)
            return {"sym": sym, "day": day, "n_dp": 0}
        X = np.load(f"{odir}/X.npy").astype(np.float32)
        t0 = np.load(f"{odir}/t0.npy").astype(np.int64)
        lab = {}
        for H in HS:
            lab[f"y0_{H}"] = np.load(f"{odir}/y0_{H}.npy").astype(np.int8)
            lab[f"rH_{H}"] = np.load(
                f"{odir}/rH_{H}.npy").astype(np.float32)

    np.savez_compressed(shard, X=X, i=i, t0=t0,
                        day_ord=np.int32(day_ord),
                        n_dp=np.int64(i.size), **lab)
    return {"sym": sym, "day": day, "n_dp": int(i.size),
            "shard_bytes": os.path.getsize(shard)}


def _reduce_symbol(sym, shard_dir, packed_path):
    """Verbatim numeric path of hd1_seq_modal.reduce_symbol, off-Modal.

    Concatenation order, honest_split, dtypes and npz schema are
    byte-identical to the frozen function. Shards are dropped after a
    successful reduce (same as frozen); the packed file is the durable
    artifact."""
    if os.path.exists(packed_path):
        return {"sym": sym, "cached_packed": True}
    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz"))
    Xs, t0s, lab = [], [], {f"y0_{H}": [] for H in HS}
    for H in HS:
        lab[f"rH_{H}"] = []
    for f in files:
        d = np.load(f"{shard_dir}/{f}")
        if "empty" in d.files:
            continue
        Xs.append(d["X"])
        t0s.append(d["t0"])
        for H in HS:
            lab[f"y0_{H}"].append(d[f"y0_{H}"])
            lab[f"rH_{H}"].append(d[f"rH_{H}"])
    X = np.concatenate(Xs)
    t0 = np.concatenate(t0s)
    n = X.shape[0]
    tr, te, n_tr = C.honest_split(n)
    packed = {"X": X, "t0": t0, "n": np.int64(n), "n_tr": np.int64(n_tr)}
    for H in HS:
        packed[f"y0_{H}"] = np.concatenate(
            lab[f"y0_{H}"]).astype(np.int8)
        packed[f"rH_{H}"] = np.concatenate(
            lab[f"rH_{H}"]).astype(np.float32)
    os.makedirs(os.path.dirname(packed_path), exist_ok=True)
    tmp = packed_path + ".tmp"
    np.savez(tmp, **packed)
    os.replace(tmp, packed_path)
    for f in files:
        os.remove(f"{shard_dir}/{f}")
    return {"sym": sym, "n": int(n), "n_tr": int(n_tr),
            "packed_gib": round(X.nbytes / 2**30, 3)}


def _stage_upload(bk, local_path, gs_uri):
    """Upload the packed file to a GCS staging prefix (intra-region,
    free). The Modal side later pulls GCS->Volume (the only unavoidable
    cross-cloud egress, on the compact pack)."""
    assert gs_uri.startswith("gs://")
    rest = gs_uri[len("gs://"):]
    bname, _, key = rest.partition("/")
    blob = (bk if bname == BUCKET else bk.client.bucket(bname)).blob(key)
    blob.upload_from_filename(local_path, timeout=3600)
    return f"gs://{bname}/{key} ({os.path.getsize(local_path)} bytes)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-l", type=int, default=1536)
    ap.add_argument("--build-syms",
                    default="BTC-USDT-PERP,ETH-USDT-PERP,LTC-USDT-PERP")
    ap.add_argument("--rust-bin",
                    default=os.path.join(
                        REPO, "rust_ingest/target/release/hd1_seq_build"))
    ap.add_argument("--work", default="/mnt/work")
    ap.add_argument("--stage",
                    default=("gs://blackdigital-scalper-data/"
                             "hd1seq_tier2_pack/l1536"))
    a = ap.parse_args()
    build_syms = [s for s in a.build_syms.split(",") if s]
    assert all(s in SYMS for s in build_syms), build_syms
    if not os.path.exists(a.rust_bin):
        sys.exit(f"rust bin missing: {a.rust_bin} (cargo build --release "
                 f"-p depth_parser --bin hd1_seq_build first)")

    t_all = time.time()
    bk = _gcs()
    per = {s: _list_days(bk, s) for s in SYMS}      # ALL 4 (frozen win)
    winlo, winhi, psd = _window(per)
    print(f"[plan] window {winlo}..{winhi}  days/sym="
          f"{ {s: len(psd[s]) for s in SYMS} }  "
          f"build_syms={build_syms} max_l={a.max_l}", flush=True)

    summary = {"max_l": a.max_l, "winlo": winlo, "winhi": winhi,
               "build_syms": build_syms, "per_sym": {}}
    for sym in build_syms:
        sdir = f"{a.work}/shards_l{a.max_l}/{sym}"
        packed = f"{a.work}/packed_l{a.max_l}/{sym}.npz"
        t_s = time.time()
        days = psd[sym]
        ndp = 0
        for di, day in enumerate(days):
            r = _build_symbol_day(bk, a.rust_bin, sym, day, di,
                                  a.max_l, sdir)
            if r.get("error"):
                print(f"[build][{sym}] day={day} ERROR {r['error']}",
                      flush=True)
                sys.exit(f"build error {sym} {day}: {r['error']}")
            ndp += r.get("n_dp", 0)
            if di % 50 == 0 or di == len(days) - 1:
                print(f"[build][{sym}] {di+1}/{len(days)} "
                      f"day={day} cum_dp={ndp} "
                      f"{time.time()-t_s:.0f}s", flush=True)
        red = _reduce_symbol(sym, sdir, packed)
        print(f"[reduce][{sym}] {red}", flush=True)
        st = _stage_upload(bk, packed,
                           f"{a.stage}/{sym}.npz")
        print(f"[stage][{sym}] -> {st}  "
              f"(sym wall {time.time()-t_s:.0f}s)", flush=True)
        summary["per_sym"][sym] = {"n": red.get("n"),
                                   "packed_gib": red.get("packed_gib"),
                                   "n_dp": ndp}

    summary["wall_s"] = round(time.time() - t_all, 1)
    done = f"{a.work}/TIER2_BUILD_DONE.json"
    with open(done, "w") as fh:
        json.dump(summary, fh, indent=2)
    _stage_upload(bk, done, f"{a.stage}/TIER2_BUILD_DONE.json")
    print(f"[done] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
