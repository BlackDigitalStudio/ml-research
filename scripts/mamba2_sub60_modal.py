#!/usr/bin/env python3
"""HD2 #3 — sub-60s 2×2 cascade on Modal (cheap L4 GPU + cpu=8; A100 overkill for
tiny models, CPU-side data handling is the bottleneck). Reuses the PROVEN image
(torch2.4.1 + prebuilt causal_conv1d/mamba_ssm cu122 wheels) and _gcs() pattern
from hd2_objsweep_modal / hd2_build_modal; runs the architecturally-correct 2×2
cascade (scripts/mamba2_cascade.py) sized to n_eff (small-d Mamba2 via conv-fallback).

  modal run scripts/mamba2_sub60_modal.py --hydrate     # GCS(compressed) -> volume
  modal run scripts/mamba2_sub60_modal.py --smoke       # L4 1-sym mamba2 sanity
  modal run scripts/mamba2_sub60_modal.py --train       # A (per-sym) + B (pooled)
"""
import os, json, glob
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent
GCP_PROJECT = "project-0998ac51-36ba-445c-bc7"
BUCKET = "market-data-0998ac51"
CACHE_Z = "hd2_sub60_cache_z"           # compressed combined cache on GCS
TOP3 = ["DOGE-USDT-PERP", "ETH-USDT-PERP", "LINK-USDT-PERP"]

_CCV = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/"
        "causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
_MAMBA = ("https://github.com/state-spaces/mamba/releases/download/v2.2.2/"
          "mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
IMG = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy==2.2.4", "scipy", "einops", "packaging",
                 "transformers==4.43.3", "google-cloud-storage")
    .pip_install(_CCV, _MAMBA)
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
)
VOL = modal.Volume.from_name("hd2-cache", create_if_missing=True)
MNT = "/cache"; SUB = f"{MNT}/sub60"
app = modal.App("hd2-mamba-sub60")


def _gcs():
    from google.cloud import storage
    import google.oauth2.credentials
    tok = "".join(os.environ["GCP_ACCESS_TOKEN"].split())

    class _Static(google.oauth2.credentials.Credentials):
        def refresh(self, request):
            return
    return storage.Client(project=GCP_PROJECT, credentials=_Static(token=tok)).bucket(BUCKET)


@app.function(image=IMG, cpu=4.0, timeout=3600, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")], retries=2)
def hydrate_one(sym):
    """gs://{BUCKET}/hd2_sub60_cache_z/{sym}/ -> /cache/sub60/{sym}/ (idempotent)."""
    outdir = f"{SUB}/{sym}"; os.makedirs(outdir, exist_ok=True)
    bk = _gcs(); n = skip = 0
    for blob in bk.client.list_blobs(bk, prefix=f"{CACHE_Z}/{sym}/"):
        if not blob.name.endswith(".npz"):
            continue
        outp = f"{outdir}/{blob.name.split('/')[-1]}"
        if os.path.exists(outp):
            skip += 1; continue
        blob.download_to_filename(outp); n += 1
    VOL.commit()
    print(f"HYDRATE {sym}: dl={n} skip={skip}")
    return {"sym": sym, "dl": n, "skip": skip}


def _streams(syms, max_days, pool_ids=None):
    """pool_ids: explicit sym_id per sym (TOP3 index) so a single-symbol B2 fine-tune
    uses the SAME symbol-embedding row the pooled base learned (else wrong row)."""
    import sys; sys.path.insert(0, "/root/scripts")
    import mamba2_cascade as mc
    paths, ids = [], []
    for si, sym in enumerate(syms):
        sid = pool_ids[si] if pool_ids else si
        ds = sorted(glob.glob(f"{SUB}/{sym}/*.npz"))
        if max_days:
            ds = ds[:max_days]
        for di, p in enumerate(ds):
            paths.append(p); ids.append((sid, di))
    return mc, mc.load_cache(paths, ids)


@app.function(image=IMG, gpu="L4", cpu=8.0, timeout=10800, volumes={MNT: VOL},
              memory=32768, retries=4)
def train_gpu(task):
    # Tiny models + short context -> GPU compute trivial, CPU-side data handling is
    # the bottleneck. Cheap L4 + cpu=8 (not A100). PREEMPTION-ROBUST: idempotent
    # skip-if-done + per-epoch checkpoint on the Volume (resume on retry).
    import sys; sys.path.insert(0, "/root/scripts")
    VOL.reload()
    head = task["head"]
    tag = task.get("tag", f"{head}_{'pool' if head=='B' else task['symbols'][0]}")
    rdir = f"{MNT}/results/sub60"; os.makedirs(rdir, exist_ok=True)
    rpath = f"{rdir}/{tag}.json"
    if os.path.exists(rpath) and not task.get("force"):
        with open(rpath) as f:
            print(f"[{tag}] SKIP (done)"); return json.load(f)
    init_from = task.get("init_from", "")
    if init_from and not os.path.exists(init_from):     # warm-start base from GCS if absent on vol
        bk = _gcs(); bk.blob(f"gru_models/{os.path.basename(init_from)}").download_to_filename(init_from)
        print(f"[{tag}] fetched base {init_from} from GCS")
    mc, streams = _streams(task["symbols"], task.get("max_days", 0), task.get("pool_ids"))
    cfg = mc.Cfg(head=head, cell=task.get("cell", "mamba2"), device="cuda",
                 epochs=task.get("epochs", 4), L=task.get("L", 3000),
                 dec_stride_s=task.get("dec_stride_s", 25),
                 d1=task.get("d1", 128 if head == "A" else 64),
                 n1=task.get("n1", 2), d2=task.get("d2", 64 if head == "A" else 32),
                 n2=task.get("n2", 1), dropout=task.get("dropout", 0.1),
                 lr=task.get("lr", 1e-3), seed=task.get("seed", 0),
                 patience=task.get("patience", 0),
                 pooled=(head == "B"), payoff=task.get("payoff", "hold"),
                 init_from=init_from, ckpt_path=f"{rdir}/{tag}.ckpt")
    print(f"[{tag}] streams={len(streams)} cfg={cfg}")
    res = mc.run(streams, cfg, log=lambda s: print(tag, s), on_ckpt=lambda: VOL.commit())
    with open(rpath, "w") as f:
        json.dump(res, f, default=float)
    VOL.commit()
    if task.get("gcs_out"):                              # mirror to GCS — NON-FATAL (weights safe on volume)
        try:
            bk = _gcs(); bp = f"{rdir}/{tag}.best.pt"
            if os.path.exists(bp):
                bk.blob(f"gru_models/{tag}.best.pt").upload_from_filename(bp)
            bk.blob(f"research_runs/gru_finetune/{tag}.json").upload_from_string(json.dumps(res, default=float))
            print(f"[{tag}] mirrored to GCS gru_models/{tag}.best.pt")
        except Exception as e:
            print(f"[{tag}] GCS mirror SKIPPED ({type(e).__name__}: {e}); model+json safe on volume")
    print(f"[{tag}] RESULT " + json.dumps(res, default=float))
    return res


@app.local_entrypoint()
def main(hydrate: bool = False, smoke: bool = False, train: bool = False, finetune: bool = False):
    if finetune:
        # Stage-2: per-symbol fine-tune of B on the EXECUTED payoff at the symbol's
        # grid-optimal config. ETH grid-optimal = HOLD (hold>bracket) -> payoff from rH,
        # no path prep. Arch MUST match base B_pool_gru (GB) so warm-start loads. The
        # base's sym-embedding row is kept via pool_ids (ETH=1).
        GB = dict(cell="stub", d1=64, d2=32, n1=2, n2=1)
        BASE = f"{SUB.rsplit('/',1)[0]}/results/sub60/B_pool_gru.best.pt"   # /cache/results/sub60/...
        tasks = [{"head": "B2", "symbols": ["ETH-USDT-PERP"], "pool_ids": [TOP3.index("ETH-USDT-PERP")],
                  "tag": "B2_ETH_hold", "payoff": "hold", "init_from": BASE,
                  "epochs": 30, "patience": 4, "max_days": 0, "lr": 3e-4,   # train to early-stop (resumes ckpt ep8)
                  "gcs_out": True, "force": True, **GB}]
        handles = [train_gpu.spawn(t) for t in tasks]
        print(f"FINETUNE SPAWNED {len(handles)} (detached -> gru_models/ on GCS):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  {t['tag']}")
        return
    if hydrate:
        handles = [hydrate_one.spawn(s) for s in TOP3]
        print(f"HYDRATE SPAWNED {len(handles)} syms (compressed GCS -> /cache/sub60):")
        for s, h in zip(TOP3, handles):
            print(f"  {h.object_id}  {s}")
        return
    if smoke:
        # GPU sanity: small-d Mamba2, 1 symbol, 2 ep, few days. Validates the
        # orchestration the CPU stub couldn't finish + confirms small-d mamba2 runs.
        t = {"head": "A", "symbols": ["DOGE-USDT-PERP"], "cell": "mamba2",
             "epochs": 2, "max_days": 40, "d1": 256, "d2": 256, "n1": 1, "n2": 1,
             "tag": "SMOKE_A_DOGE"}
        print("SMOKE:", t)
        print(json.dumps(train_gpu.remote(t), indent=2, default=float))
        return
    if train:
        # HEAD-TO-HEAD: compact Mamba2 (d=256/n=1, ~1M, proven fused kernel) vs
        # GRU-small (sized-to-n_eff). Both as A per-symbol (top-3) + B pooled top-3.
        # Day-caps keep RAM in budget (f16 lob). Compare OOS + overfit gap.
        M = dict(cell="mamba2", d1=256, d2=256, n1=1, n2=1)
        GA = dict(cell="stub", d1=128, d2=64, n1=2, n2=1)     # GRU Model-A (~165k)
        GB = dict(cell="stub", d1=64, d2=32, n1=2, n2=1)      # GRU Model-B (~50k)
        tasks = []
        for s in TOP3:
            sn = s.split("-")[0]
            tasks.append({"head": "A", "symbols": [s], "tag": f"A_{sn}_m2",
                          "epochs": 6, "max_days": 300, **M})
            tasks.append({"head": "A", "symbols": [s], "tag": f"A_{sn}_gru",
                          "epochs": 6, "max_days": 300, **GA})
        tasks.append({"head": "B", "symbols": TOP3, "tag": "B_pool_m2",
                      "epochs": 8, "max_days": 150, **M})
        tasks.append({"head": "B", "symbols": TOP3, "tag": "B_pool_gru",
                      "epochs": 8, "max_days": 150, **GB})
        handles = [train_gpu.spawn(t) for t in tasks]
        print(f"TRAIN SPAWNED {len(handles)} L4 units (detached -> /cache/results/sub60):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  {t['tag']}")
        return
    print("specify --hydrate | --smoke | --train")
