#!/usr/bin/env python3
"""HD2 full-tier sweep on Modal H100 (HD2 rev1; cost cap $100, user 2026-05-23).

Units = (symbol, L, seed) = 2 x 3 x 3 = 18 multitask-H runs (each emits 3 H
heads -> the 6 (symbol,H) cells x 3 L x 3 seeds surface). Preemption-resilient:
each unit is idempotent (skips if its result JSON exists on the Volume) and
checkpoints atomically every epoch, so a killed container resumes from cursor.

  modal run scripts/hd2_sweep_modal.py              # validate: 1 cell, 2 epochs
  modal run scripts/hd2_sweep_modal.py --full       # 18-unit sweep
"""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent
_CCV = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/"
        "causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
_MAMBA = ("https://github.com/state-spaces/mamba/releases/download/v2.2.2/"
          "mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
IMG = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy==2.2.4", "scipy", "scikit-learn",
                 "einops", "packaging", "transformers==4.43.3")
    .pip_install(_CCV, _MAMBA)
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
)
VOL = modal.Volume.from_name("hd2-cache", create_if_missing=True)
app = modal.App("hd2-sweep")
MNT = "/cache"
SYMS = ["SOL-USDT-PERP", "LTC-USDT-PERP"]
LS = [6000, 36000, 216000]
SEEDS = [0, 1, 2]
# RIGHT-SIZED (2026-05-23): the model footprint is ~5 GB (smoke), so an L4
# 24GB at ~$0.8/hr is the smallest sufficient card (~5x cheaper than H100,
# which ran at ~6% VRAM = wasteful). batch_periods is set per-L from a fixed
# token budget so each forward FILLS the card and VRAM stays ~constant across
# the L axis (L4 de-risk: bp*L=576k -> ~11 GB / 24 GB).
GPU = "L4"
TOKEN_BUDGET = 576_000


@app.function(image=IMG, gpu=GPU, timeout=5400, volumes={MNT: VOL}, retries=2)
def train_unit(task):
    import sys, json, os
    sys.path.insert(0, "/root/scripts")
    import hd2_train_full as T
    import numpy as np

    sym, L, seed, epochs = task["sym"], task["L"], task["seed"], task["epochs"]
    tag = f"{sym}_L{L}_s{seed}"
    rdir = f"{MNT}/results/hd2"; os.makedirs(rdir, exist_ok=True)
    rpath = f"{rdir}/{tag}.json"
    VOL.reload()
    if os.path.exists(rpath) and not task.get("force"):
        with open(rpath) as f:
            return {"tag": tag, "skip": True, **json.load(f).get("by_H_summary", {})}

    bp = max(1, TOKEN_BUDGET // L)        # fill the card; const VRAM across L
    cfg = T.FullCfg(symbol=sym, L=L, seed=seed, epochs=epochs,
                    batch_periods=bp, ckpt_path=f"{rdir}/{tag}.ckpt")
    out, preds = T.train_cell(MNT, cfg, log=lambda s: print(tag, s))

    np.savez(f"{rdir}/{tag}_preds.npz",
             **{f"pred_H{H}": preds[H] for H in cfg.Hs})
    summary = {H: {"all_rank_ic": out["by_H"][H]["all"].get("rank_ic"),
                   "all_auc": out["by_H"][H]["all"].get("auc"),
                   "deep_rank_ic": out["by_H"][H]["deep"].get("rank_ic")}
               for H in cfg.Hs}
    rec = {**out, "by_H_summary": summary}
    tmp = rpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, default=float)
    os.replace(tmp, rpath)
    VOL.commit()
    print(tag, "RESULT " + json.dumps(summary, default=float))
    return {"tag": tag, **summary}


@app.local_entrypoint()
def main(full: bool = False):
    import json
    if not full:
        t = {"sym": "SOL-USDT-PERP", "L": 6000, "seed": 0, "epochs": 2,
             "force": True}
        print("VALIDATE:", t)
        print(json.dumps(train_unit.remote(t), indent=2, default=float))
        return
    tasks = [{"sym": s, "L": L, "seed": sd, "epochs": 15}
             for s in SYMS for L in LS for sd in SEEDS]
    print(f"FULL sweep: {len(tasks)} units")
    res = list(train_unit.map(tasks, order_outputs=False))
    print("SWEEP_DONE " + json.dumps(res, default=float))
