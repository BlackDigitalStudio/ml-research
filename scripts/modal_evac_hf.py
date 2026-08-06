#!/usr/bin/env python3
"""Evacuate the surviving Modal `hd2-cache` material to PUBLIC Hugging Face
datasets (in-cloud Modal->HF; the local link never carries data). After the
GCP outage of 2026-08-05 the Modal volumes are the only complete copy of the
surviving training substrate; HF is free, account-independent, and public per
user sign-off 2026-08-06. Modal replaces GCP as the compute home, so the
volumes stay in place — HF is the durable mirror.

Target datasets (all public, user delmiron27):
  ml-research-hd2-streams <- ws06 /cache/hd2    8 syms, fp16 80-ch LOB streams
                             + first-passage labels H=180/600/1800 (~30 GiB)
  ml-research-midts       <- ws06 /cache/midts  8 syms, per-day {ts,mid} event
                             series (arbitrary-horizon relabeling substrate)
  ml-research-sub60       <- ws08 /cache/sub60  DOGE/ETH/LINK 1s-grid combined
                             cache (71-feat + LOB stream + rH60/y60), ~23 GiB
  ml-research-results     <- results/ of every workspace, subfolder per ws

Both entrypoints run under the virginship06 profile: it holds the `hf-token`
secret AND the fullest volume (8-sym hd2 + 8-sym midts). Other workspaces'
volumes are reached by a bridge container that drives the Modal CLI with that
workspace's token pair taken from the ad-hoc `evac-ws-tokens` secret
(WS03/WS05/WS07/WS08 _ID/_SECRET pairs; created from the operator's local
token store, never committed).

  MODAL_PROFILE=virginship06 modal run -d scripts/modal_evac_hf.py::push_main
  MODAL_PROFILE=virginship06 modal run -d scripts/modal_evac_hf.py::push_bridge
"""
import os
import subprocess

import modal

IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("huggingface_hub[hf_transfer]==0.26.2", "modal==1.4.3"))
VOL = modal.Volume.from_name("hd2-cache")
MNT = "/cache"
app = modal.App("ml-research-evac-hf")

USER = "delmiron27"
DS_HD2 = f"{USER}/ml-research-hd2-streams"
DS_MID = f"{USER}/ml-research-midts"
DS_S60 = f"{USER}/ml-research-sub60"
DS_RES = f"{USER}/ml-research-results"
IGNORE = [".cache/**", ".gitattributes", "**/.DS_Store"]
# volumes are read here, never written: safe to run concurrently with nothing
# else because every other consumer of hd2-cache died with GCP.


def _api():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def _du(path):
    t = 0
    for r, _d, fs in os.walk(path):
        for f in fs:
            t += os.path.getsize(os.path.join(r, f))
    return round(t / 2**30, 2)


def _push_large(api, ds, src):
    api.create_repo(ds, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=ds, repo_type="dataset", folder_path=src,
                            ignore_patterns=IGNORE, num_workers=8)
    return {"ds": ds, "gib": _du(src)}


def _push_results(api, ws, src):
    api.create_repo(DS_RES, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=DS_RES, repo_type="dataset", folder_path=src,
                      path_in_repo=ws, ignore_patterns=IGNORE)
    return {"ds": f"{DS_RES}/{ws}", "gib": _du(src)}


@app.function(image=IMG, cpu=8.0, timeout=86400, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hf-token")])
def push_main():
    """ws06 volume -> HF: hd2 (8 syms), midts (8 syms), results/virginship06."""
    api = _api()
    out = [_push_large(api, DS_HD2, f"{MNT}/hd2"),
           _push_large(api, DS_MID, f"{MNT}/midts"),
           _push_results(api, "virginship06", f"{MNT}/results")]
    print("PUSH_MAIN_DONE", out)
    return out


def _fetch(ws_env, vol, remote, dest):
    """Drive the Modal CLI against ANOTHER workspace using its token pair.
    Inside a container the client IGNORES token env vars and assumes the
    task's own identity (measured: the ws08 volume silently resolved to
    ws06's), so every MODAL_* container variable must be stripped first to
    make the CLI behave as an external client."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MODAL_")}
    env["MODAL_TOKEN_ID"] = os.environ[f"{ws_env}_ID"]
    env["MODAL_TOKEN_SECRET"] = os.environ[f"{ws_env}_SECRET"]
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", vol, remote, dest],
                   env=env, check=True, timeout=14400)


@app.function(image=IMG, cpu=8.0, timeout=86400,
              secrets=[modal.Secret.from_name("hf-token"),
                       modal.Secret.from_name("evac-ws-tokens")])
def push_bridge():
    """Foreign-workspace material -> HF: ws08 sub60 + results of 03/05/07/08."""
    api = _api()
    out = []
    _fetch("WS08", "hd2-cache", "sub60", "/tmp/dl")
    out.append(_push_large(api, DS_S60, "/tmp/dl/sub60"))
    for ws_env, ws in [("WS08", "virginship08"), ("WS05", "virginship05"),
                       ("WS07", "virginship07"), ("WS03", "virgin-ship03")]:
        dest = f"/tmp/res/{ws}"
        _fetch(ws_env, "hd2-cache", "results", dest)
        out.append(_push_results(api, ws, f"{dest}/results"))
    # ship03 also holds the tiny hd2-smoke volume (one checkpoint): keep it.
    try:
        _fetch("WS03", "hd2-smoke", "/", "/tmp/smoke")
        api.upload_folder(repo_id=DS_RES, repo_type="dataset",
                          folder_path="/tmp/smoke",
                          path_in_repo="virgin-ship03/hd2-smoke")
        out.append({"ds": f"{DS_RES}/virgin-ship03/hd2-smoke",
                    "gib": _du("/tmp/smoke")})
    except Exception as e:  # non-fatal: smoke volume is a single .pt
        print("SMOKE_SKIPPED", repr(e))
    print("PUSH_BRIDGE_DONE", out)
    return out
