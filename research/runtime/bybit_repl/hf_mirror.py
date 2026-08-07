#!/usr/bin/env python3
"""HBV1: mirror the artifact CORE from the bybit-cl Modal volume to a PRIVATE
Hugging Face dataset repo (account-independence: the volume is the only copy of
the computed artifacts, and we just lost the CL archive with a billing account).

Mirrored (~7-8 GB): combined datasets + anch/v1/nooi prefixes (PERFOLD, MODELS,
SEED/ENS/HBV1_* jsons) + bybit_aux + VENUE_NOTE.md.
Skipped (re-derivable / re-downloadable): raw/ converted parquets, daily/ npz,
feats_sub60 BTC mids, coinalyze_fwd.

Needs Modal secret `hf-write-token` with HF_TOKEN=<write-scope token>:
  modal secret create hf-write-token HF_TOKEN=hf_...
Run:  modal run hf_mirror.py [--repo delmiron27/hbv1-bybit-artifacts]
"""
import os

import modal

app = modal.App("hbv1-hf-mirror")
vol = modal.Volume.from_name("bybit-cl")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]")

ROOT = "/vol/gcs/market-data-0998ac51"
# INCLUDE env ("path,path,..."): overrides the default DOGE core list — lets the
# same mirror run per account/symbol (rf/fb prefixes on 08, 1000PEPE/LTC cells
# on 05/06). Secret hf-write-token replicated to all virginship0{5,6,7,8}
# (2026-08-07); default repo stays delmiron27/hbv1-bybit-artifacts.
INCLUDE = [x for x in os.environ.get("INCLUDE", "").split(",") if x] or [
    "research_runs/maker_labels_tb3s_h150/DOGE.npz",
    "research_runs/maker_labels_tb3s_h150anch",
    "research_runs/maker_labels_tb3s_h150anch_v1",
    "research_runs/maker_labels_tb3s_h150anch_v2_nooi",
    "bybit_aux",
    "VENUE_NOTE.md",
]


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=4, memory=8192,
              secrets=[modal.Secret.from_name("hf-write-token")])
def mirror(repo: str):
    import shutil
    from huggingface_hub import HfApi

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    stage = "/tmp/stage"
    shutil.rmtree(stage, ignore_errors=True)
    total = 0
    for rel in INCLUDE:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(stage, rel)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        else:
            print(f"  missing: {rel}", flush=True)
            continue
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(dst) for f in fs) \
            if os.path.isdir(dst) else os.path.getsize(dst)
        total += sz
        print(f"  staged {rel} ({sz/1e9:.2f} GB)", flush=True)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write("# HBV1 Bybit replication artifacts (mirror)\n\nPrivate mirror of the Modal volume "
                "`bybit-cl` artifact core (ledger: BlackDigitalStudio/ml-research "
                "research/experiments.jsonl, hypothesis HBV1). Venue = BYBIT LINEAR; see VENUE_NOTE.md. "
                "raw/ + daily/ intentionally omitted (re-derivable from public Bybit archives via "
                "research/runtime/bybit_repl/).\n")
    print(f"[staged total {total/1e9:.2f} GB] uploading...", flush=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=stage)
    return f"mirrored {total/1e9:.2f} GB -> hf://datasets/{repo}"


@app.local_entrypoint()
def main(repo: str = "delmiron27/hbv1-bybit-artifacts"):
    print(mirror.remote(repo))
