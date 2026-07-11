#!/usr/bin/env python3
"""Transfer the Modal `hd2-cache` Volume -> Hugging Face datasets (in-cloud,
Modal->HF, so the ~35 GB never crosses the local link). Account portability:
HF is free + not tied to the Modal account.

Two private datasets (user 2026-05-23):
  delmiron27/scalper-bot-hd2-cache    <- /cache/hd2/        (500d streams, ~35 GB)
  delmiron27/scalper-bot-hd2-results  <- /cache/results/hd2 (JSON+ckpt+preds)

Needs Modal secret `hf-token` with HF_TOKEN (write scope).
  modal run scripts/hd2_to_hf.py --what results   # small first (verify)
  modal run scripts/hd2_to_hf.py --what cache      # the 35 GB
  modal run scripts/hd2_to_hf.py --what both
"""
import modal

IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("huggingface_hub[hf_transfer]==0.26.2"))
VOL = modal.Volume.from_name("hd2-cache")
app = modal.App("hd2-to-hf")
USER = "delmiron27"
CACHE_DS = f"{USER}/scalper-bot-hd2-cache"
RESULTS_DS = f"{USER}/scalper-bot-hd2-results"


@app.function(image=IMG, volumes={"/cache": VOL}, timeout=14400,
              secrets=[modal.Secret.from_name("hf-token")])
def push(what: str = "both"):
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    out = {}

    def _du(path):
        t = 0
        for r, _d, fs in os.walk(path):
            for f in fs:
                t += os.path.getsize(os.path.join(r, f))
        return t / 1e9

    if what in ("results", "both"):
        src = "/cache/results/hd2"
        api.create_repo(RESULTS_DS, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=RESULTS_DS, repo_type="dataset",
                          folder_path=src, path_in_repo="results")
        out["results"] = {"ds": RESULTS_DS, "gb": round(_du(src), 3)}
        print("RESULTS_DONE", out["results"])

    if what in ("cache", "both"):
        src = "/cache/hd2"
        api.create_repo(CACHE_DS, repo_type="dataset", private=True, exist_ok=True)
        # upload_large_folder: resumable, multi-commit, LFS-aware (good for 35 GB)
        api.upload_large_folder(repo_id=CACHE_DS, repo_type="dataset",
                                folder_path=src)
        out["cache"] = {"ds": CACHE_DS, "gb": round(_du(src), 3)}
        print("CACHE_DONE", out["cache"])

    return out


@app.function(image=IMG, volumes={"/cache": VOL}, timeout=14400,
              secrets=[modal.Secret.from_name("hf-token")])
def hydrate(what: str = "cache"):
    """Reverse of push: HF dataset -> Modal Volume. Used to re-hydrate a FRESH
    account's Volume after a `modal token set` switch (Volumes are account-scoped).
      cache  -> /cache/hd2/         (500d streams)
      results-> /cache/results/hd2_hf
    NOTE: midts (/cache/midts) is NOT on HF; push it separately via
    `modal volume put hd2-cache <local_midts> midts`."""
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    out = {}

    def _count(path):
        n = 0
        for _r, _d, fs in os.walk(path):
            n += sum(1 for f in fs if f.endswith(".npz"))
        return n

    if what in ("cache", "both"):
        snapshot_download(repo_id=CACHE_DS, repo_type="dataset",
                          local_dir="/cache/hd2", token=os.environ["HF_TOKEN"],
                          max_workers=16)
        VOL.commit()
        out["cache"] = {"dir": "/cache/hd2", "npz": _count("/cache/hd2")}
        print("HYDRATE_CACHE_DONE", out["cache"])
    if what in ("results", "both"):
        snapshot_download(repo_id=RESULTS_DS, repo_type="dataset",
                          local_dir="/cache/results/hd2_hf",
                          token=os.environ["HF_TOKEN"], max_workers=16)
        VOL.commit()
        out["results"] = {"dir": "/cache/results/hd2_hf",
                          "npz": _count("/cache/results/hd2_hf")}
        print("HYDRATE_RESULTS_DONE", out["results"])
    return out


@app.local_entrypoint()
def main(what: str = "results", pull: bool = False):
    """pull=False -> push Volume->HF (upload); pull=True -> hydrate HF->Volume."""
    import json
    fn = hydrate if pull else push
    print(json.dumps(fn.remote(what), indent=2, default=str))
