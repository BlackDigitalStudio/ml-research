#!/usr/bin/env python3
"""HD1 Tier-2 (rev45) GCS->Modal transfer stage.

The GCP VM (hd1_seq_tier2_gcpbuild.py) builds the MAX_L=1536 packed
cache and stages one npz per symbol to
  gs://blackdigital-scalper-data/hd1seq_tier2_pack/l1536/{sym}.npz
(np.savez, UNCOMPRESSED: members X.npy,t0.npy,n.npy,n_tr.npy,
 y0_{H}.npy,rH_{H}.npy) plus TIER2_BUILD_DONE.json.

This stage moves that compact pack GCS->Modal (the single unavoidable
cross-cloud egress, ~$18 for ~150 GiB f32) and lays it out in the
exact contract hd1_seq_tier2.py expects:
  /cache/packed_l1536/{sym}_X.npy     (n,1536,46) f32, MMAP-ABLE
  /cache/packed_l1536/{sym}_meta.npz  t0,n,n_tr,y0_{H},rH_{H}

Memory-safe: an npz is a zip of .npy members; we STREAM-copy the
`X.npy` member straight to {sym}_X.npy (no np.load of the ~50 GiB X,
no RAM blow-up) and load only the small members into the meta npz.

The frozen 512 pack at /cache/packed/{sym}.npz is NOT touched (the
runner's rev45 parity guard needs it intact). Idempotent / resumable:
a symbol whose {sym}_X.npy + {sym}_meta.npz already exist is skipped.

GCS auth is the FROZEN hd1_seq_modal._gcs (Modal secret hd1-gcp) --
the same path that historically egressed the raw ~392 GiB; reused
verbatim, not reimplemented.

Run (a SEPARATE user-gated stage; spends the ~$18 egress):
  modal run scripts/hd1_seq_tier2_transfer.py            (after DONE)
  modal run scripts/hd1_seq_tier2_transfer.py --verify-only
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

SYMBOLS = ["BTC-USDT-PERP", "ETH-USDT-PERP", "LTC-USDT-PERP"]
HS = (180, 300, 600)
GCS_PREFIX = "hd1seq_tier2_pack/l1536"
PACK_L = 1536
N_FEAT = 46

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-tier2-transfer")
IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("numpy==2.2.4", "google-cloud-storage")
       .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"
PACK_DIR = f"{MNT}/packed_l1536"


def _mark(name, txt):
    import os
    os.makedirs(f"{MNT}/tier2_transfer", exist_ok=True)
    open(f"{MNT}/tier2_transfer/{name}", "w").write(str(txt))
    VOL.commit()


@app.function(image=IMG, timeout=600,
              secrets=[modal.Secret.from_name("hd1-gcp")],
              volumes={MNT: VOL})
def build_done():
    """Read TIER2_BUILD_DONE.json from GCS (the GCP VM wrote it).
    Returns the per-symbol n / packed_gib so the runner `memory=` can
    be reconfirmed before the paid sweep, or {} if not built yet."""
    sys.path.insert(0, "/root/proj")
    from scripts.hd1_seq_modal import _gcs
    bk = _gcs()
    blob = bk.blob(f"{GCS_PREFIX}/TIER2_BUILD_DONE.json")
    if not blob.exists():
        fail = bk.blob(f"{GCS_PREFIX}/TIER2_BUILD_FAILED.txt")
        return {"done": False,
                "failed": fail.download_as_text() if fail.exists()
                else None}
    return {"done": True, "summary": json.loads(blob.download_as_text())}


@app.function(image=IMG, timeout=21600, memory=4096,
              secrets=[modal.Secret.from_name("hd1-gcp")],
              volumes={MNT: VOL}, retries=2)
def transfer_symbol(sym: str):
    """Stream gs://.../l1536/{sym}.npz directly into a mmap-able
    {sym}_X.npy + a small {sym}_meta.npz on the Volume. NO staging
    file: the previous design wrote 84 GiB staging to the Volume,
    then read it back and wrote another 84 GiB X.npy -- two large
    Volume writes per symbol, and Modal Volume sustained write rate
    (~10 MB/s observed) made that the wall-clock bottleneck. The new
    pipe uses google-cloud-storage Blob.open('rb') (a seekable
    BlobReader doing HTTP range reads under the hood); zipfile only
    needs an end-seek for the central directory + the member offset,
    so the X.npy bytes flow GCS -> zipfile -> Volume in 16 MiB chunks
    with ONE Volume write per symbol. Resumable."""
    import os
    import shutil
    import zipfile
    import numpy as np
    sys.path.insert(0, "/root/proj")
    from scripts.hd1_seq_modal import _gcs

    os.makedirs(PACK_DIR, exist_ok=True)
    xpath = f"{PACK_DIR}/{sym}_X.npy"
    mpath = f"{PACK_DIR}/{sym}_meta.npz"
    if os.path.exists(xpath) and os.path.exists(mpath):
        mm = np.load(xpath, mmap_mode="r")
        return {"sym": sym, "skipped": True,
                "x_shape": list(mm.shape)}

    bk = _gcs()
    blob = bk.blob(f"{GCS_PREFIX}/{sym}.npz")
    if not blob.exists():
        return {"sym": sym, "error": f"missing gs://{GCS_PREFIX}/"
                f"{sym}.npz"}
    blob.reload()
    src_gib = round((blob.size or 0) / 2**30, 2)
    t0 = time.time()

    # Stream X.npy member directly from GCS -> Volume (one write).
    tmp_x = xpath + ".part"
    with blob.open("rb") as g:
        with zipfile.ZipFile(g) as zf:
            names = set(zf.namelist())
            if "X.npy" not in names:
                return {"sym": sym, "error": f"no X.npy in npz "
                        f"(members={sorted(names)})"}
            with zf.open("X.npy") as src, open(tmp_x, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
    os.replace(tmp_x, xpath)
    x_written_s = round(time.time() - t0, 1)

    # Second short pass for the SMALL meta members (a few KB-MB total).
    meta = {}
    with blob.open("rb") as g:
        with np.load(g) as P:
            for k in ("n", "n_tr"):
                if k in P.files:
                    meta[k] = P[k]
            meta["t0"] = P["t0"]
            for H in HS:
                meta[f"y0_{H}"] = P[f"y0_{H}"]
                meta[f"rH_{H}"] = P[f"rH_{H}"]
    tmp_m = f"{PACK_DIR}/_meta_{sym}.tmp.npz"
    np.savez(tmp_m, **meta)
    os.replace(tmp_m, mpath)
    VOL.commit()

    mm = np.load(xpath, mmap_mode="r")
    ok = (mm.dtype == np.float32 and mm.ndim == 3
          and mm.shape[1] == PACK_L and mm.shape[2] == N_FEAT
          and int(meta["n"]) == mm.shape[0])
    return {"sym": sym, "ok": bool(ok), "x_shape": list(mm.shape),
            "x_dtype": str(mm.dtype), "n": int(meta["n"]),
            "src_gib": src_gib, "x_written_s": x_written_s,
            "total_s": round(time.time() - t0, 1),
            "mb_per_s": round(src_gib * 1024 / x_written_s, 1)
            if x_written_s else None}


@app.function(image=IMG, timeout=900, volumes={MNT: VOL})
def verify():
    """Confirm the runner's data contract is satisfied and the frozen
    512 pack is still present (parity-guard precondition)."""
    import os
    import numpy as np
    VOL.reload()
    rep = {}
    for sym in SYMBOLS:
        xp, mp = f"{PACK_DIR}/{sym}_X.npy", f"{PACK_DIR}/{sym}_meta.npz"
        fp = f"{MNT}/packed/{sym}.npz"
        r = {"x": os.path.exists(xp), "meta": os.path.exists(mp),
             "frozen512": os.path.exists(fp)}
        if r["x"]:
            mm = np.load(xp, mmap_mode="r")
            r["x_shape"] = list(mm.shape)
            r["x_dtype"] = str(mm.dtype)
        if r["meta"]:
            md = np.load(mp)
            r["meta_keys"] = sorted(md.files)
            r["n"] = int(md["n"]) if "n" in md.files else None
        rep[sym] = r
    ok = all(v.get("x") and v.get("meta") and v.get("frozen512")
             and v.get("x_dtype") == "float32"
             and v.get("x_shape", [0, 0, 0])[1:] == [PACK_L, N_FEAT]
             for v in rep.values())
    _mark("VERIFY_OK" if ok else "VERIFY_FAIL",
          json.dumps(rep, indent=2))
    return {"ok": ok, "report": rep}


@app.local_entrypoint()
def main(verify_only: int = 0, force: int = 0):
    if verify_only:
        print(json.dumps(verify.remote(), indent=2, default=str))
        return
    bd = build_done.remote()
    if not bd.get("done"):
        print("[transfer] build NOT done yet (no "
              "TIER2_BUILD_DONE.json in GCS). "
              f"failed_marker={bd.get('failed')}")
        print("[transfer] re-run after the GCP build self-completes.")
        raise SystemExit(2)
    summ = bd.get("summary", {})
    print(f"[transfer] build DONE: winlo..winhi="
          f"{summ.get('winlo')}..{summ.get('winhi')} "
          f"wall_s={summ.get('wall_s')}")
    for s, d in (summ.get("per_sym") or {}).items():
        pg = d.get("packed_gib")
        print(f"[transfer]   {s}: n={d.get('n')} "
              f"packed_gib={pg} n_dp={d.get('n_dp')}")
    # MEM reconfirm note for the paid runner (rev45 discipline):
    mx = max((d.get("packed_gib") or 0)
             for d in (summ.get("per_sym") or {"_": {}}).values())
    print(f"[transfer] MEM NOTE: largest per-symbol 1536 pack "
          f"~{mx} GiB; hd1_seq_tier2.tier2_all mmaps it -> resident "
          f"= fit/val/te subsets only. Reconfirm tier2_all memory= "
          f"(currently 131072MB) against this before the paid sweep.")

    t0 = time.time()
    res = list(transfer_symbol.map(SYMBOLS))
    for r in res:
        print(f"[transfer] {r}")
    bad = [r for r in res if r.get("error") or r.get("ok") is False]
    v = verify.remote()
    print("[verify] " + json.dumps(v, indent=2, default=str))
    print(f"[transfer] done in {round(time.time()-t0)}s; "
          f"failures={bad if bad else 'none'}")
    if bad or not v.get("ok"):
        raise SystemExit(1)
    print("[transfer] OK -> /cache/packed_l1536/ ready for "
          "hd1_seq_tier2.py (rev45 parity guard will bit-check it "
          "against the frozen 512 pack).")
