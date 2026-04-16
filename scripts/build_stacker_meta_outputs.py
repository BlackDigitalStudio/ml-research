#!/usr/bin/env python3
"""Train L2 stacker + L3 meta on the first 75% of samples and save their
outputs for all 93k so downstream modules (e.g. IQL policy head) can
consume them as state features.

Uses the same `_walk_forward_stacker_meta` logic as `scripts/grid_live.py`
so the outputs are identical to what the grid already used internally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.grid_live import _walk_forward_stacker_meta   # noqa: E402


CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
SOFTS_PATH = Path("/home/scalper/scalper-bot/models/primary_softs_v4.npz")
OUT = Path("/home/scalper/scalper-bot/models/stacker_meta_v1.npz")
TRAIN_FRAC = 0.75


def _load_cache():
    cand = sorted(CACHE_DIR.glob("samples_v3_*_y.npy"))
    prefix = str(cand[-1])[: -len("_y.npy")]
    return {
        "prefix": prefix,
        "y": np.load(f"{prefix}_y.npy"),
        "pnl": np.load(f"{prefix}_pnl.npy"),
    }


def main():
    c = _load_cache()
    d = np.load(SOFTS_PATH, allow_pickle=False)
    soft_keys = sorted(k for k in d.files if k.startswith("soft_"))
    primary_softs = [d[k] for k in soft_keys]
    arch_keys = [k[len("soft_"):] for k in soft_keys]

    n = c["y"].shape[0]
    n_tr = int(TRAIN_FRAC * n)
    print(f"[smo] N={n:,}  n_tr={n_tr:,}  archs={len(arch_keys)}")

    stacker_soft, meta_prob, meta_metrics = _walk_forward_stacker_meta(
        primary_softs=primary_softs,
        y=c["y"],
        pnl_for_meta=c["pnl"],
        n_tr=n_tr,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        stacker_soft=stacker_soft.astype(np.float32),   # (N, 3)
        meta_prob=meta_prob.astype(np.float32),          # (N,)
        arch_keys=np.array(arch_keys),
        y=c["y"],
        meta_metrics=json.dumps(meta_metrics, default=float),
        n_train=n_tr,
    )
    print(f"[smo] wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"[smo] meta metrics: {meta_metrics}")
    print(f"[smo] stacker class dist argmax: "
          f"UP={(stacker_soft.argmax(-1)==0).sum()} "
          f"DN={(stacker_soft.argmax(-1)==1).sum()} "
          f"FL={(stacker_soft.argmax(-1)==2).sum()}")


if __name__ == "__main__":
    main()
