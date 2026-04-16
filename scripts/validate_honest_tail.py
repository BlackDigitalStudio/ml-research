#!/usr/bin/env python3
"""Validate the retargeted-meta grid winners on the HONEST-ONLY tail.

Grid tail was samples [70k, 93k] = 23k samples, but primaries were
trained on [0, 74k) — so the first 4k of the eval tail (70k..74k) leaks.
This script evaluates only on [74k, 93k] = 19k truly unseen samples.

Also computes compounded (multiplicative) equity in addition to the
grid_live_retargeted cumsum PnL, so drawdown reflects realistic
position sizing.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge      # noqa: E402


import os

CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
V2_PATH = Path(os.environ.get("V2_PATH",
                                "/home/scalper/scalper-bot/models/stacker_meta_v2.npz"))
GRID_PATH = Path(os.environ.get("GRID_PATH",
                                  "/home/scalper/scalper-bot/models/grid_live_v5_retarget.json"))
# Primary train-end for leak boundary. Depends on which bakeoff trained
# the primaries whose softs fed V2_PATH:
#    - pre-leakfree: primaries trained on [0, 74k) → HONEST_START = 74000
#    - leakfree: primaries trained on [0, 70k) → HONEST_START = 70000
# Override via env: HONEST_START=70000 python scripts/validate_honest_tail.py
HONEST_START = int(os.environ.get("HONEST_START", "74000"))


def _load_cache():
    cand = sorted(CACHE_DIR.glob("samples_v3_*_mid_paths.npy"))
    if not cand:
        raise FileNotFoundError(f"No v3 cache in {CACHE_DIR}")
    cand.sort(key=lambda p: p.stat().st_size, reverse=True)
    prefix = str(cand[0])[: -len("_mid_paths.npy")]
    print(f"[val] using cache prefix: {prefix}")
    return {
        "y": np.load(f"{prefix}_y.npy"),
        "mid_paths": np.load(f"{prefix}_mid_paths.npy"),
        "entry_long": np.load(f"{prefix}_entry_long.npy"),
        "entry_short": np.load(f"{prefix}_entry_short.npy"),
    }


def _sim_config(c, tp, sl, timeout):
    N = c["y"].shape[0]
    return rust_bridge.simulate_labels(
        c["entry_long"], c["entry_short"], c["mid_paths"],
        np.full(N, tp, dtype=np.float64),
        np.full(N, sl, dtype=np.float64),
        np.full(N, timeout, dtype=np.int64),
        commission_win_pct=0.04, commission_loss_pct=0.07,
        partial_enabled=True, trailing_enabled=True, fill_latency_ms=150.0,
    )


def main():
    c = _load_cache()
    d = np.load(V2_PATH, allow_pickle=False)
    stacker_soft = d["stacker_soft"]
    meta_prob = d["meta_prob"]
    primary_pred = stacker_soft.argmax(axis=-1)
    primary_max = stacker_soft.max(axis=-1)

    grid = json.loads(GRID_PATH.read_text())
    top_configs = grid["top_by_net"][:20]

    print(f"{'rank':>4} {'tp':>5} {'sl':>5} {'to_s':>5} {'thr':>5} {'fp':>4} "
          f"{'n_full':>7} {'WR_full':>8} {'net_full':>9} "
          f"{'n_hon':>6} {'WR_hon':>7} {'net_hon':>8} {'mul_eq':>8}")
    print("=" * 110)

    rng = np.random.default_rng(42)
    for rank, cfg in enumerate(top_configs[:20], 1):
        tp, sl, to = cfg["tp"], cfg["sl"], cfg["timeout"]
        thr, minp, fp = cfg["meta_thr"], cfg["min_prob"], cfg["fill_prob"]
        spread_bps = cfg["spread_bps"]

        out = _sim_config(c, tp, sl, to)
        pnl_long = out["pnl_long"].astype(np.float64)
        pnl_short = out["pnl_short"].astype(np.float64)
        real = np.where(primary_pred == 0, pnl_long,
                np.where(primary_pred == 1, pnl_short, 0.0))
        real_net = real - spread_bps / 100.0
        real_net *= 0.25   # kelly fraction in the grid

        nf = primary_pred != 2
        gate = nf & (primary_max >= minp) & (meta_prob >= thr)
        fill = rng.random(len(c["y"])) < fp if fp < 1.0 else np.ones(len(c["y"]), dtype=bool)
        take = gate & fill

        # Full tail [70k, 93k]
        full_mask = np.zeros(len(c["y"]), dtype=bool)
        full_mask[69922:] = True          # matches grid n_train
        full_take = take & full_mask
        full_tr = real_net[full_take]
        # Honest tail only [74k, 93k]
        hon_mask = np.zeros(len(c["y"]), dtype=bool)
        hon_mask[HONEST_START:] = True
        hon_take = take & hon_mask
        hon_tr = real_net[hon_take]

        n_full = len(full_tr)
        n_hon = len(hon_tr)
        wr_full = (full_tr > 0).mean() * 100 if n_full else float("nan")
        wr_hon = (hon_tr > 0).mean() * 100 if n_hon else float("nan")
        net_full = full_tr.sum()
        net_hon = hon_tr.sum()
        mul_eq = float(np.prod(1.0 + hon_tr / 100.0) - 1.0) * 100 if n_hon else float("nan")

        print(f"{rank:>4d} {tp:>5.2f} {sl:>5.2f} {to/10:>4.0f}s "
              f"{thr:>5.2f} {fp:>4.1f} "
              f"{n_full:>7d} {wr_full:>7.1f}% {net_full:>+7.2f}% "
              f"{n_hon:>6d} {wr_hon:>6.1f}% {net_hon:>+7.2f}% {mul_eq:>+7.2f}%")


if __name__ == "__main__":
    main()
