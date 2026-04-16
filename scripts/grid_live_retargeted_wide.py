#!/usr/bin/env python3
"""Expanded retargeted-meta grid — targets ~500k configurations.

Extends scripts/grid_live_retargeted.py with a denser grid over all
axes. Still runs on Contabo CPU via Rust simulate_labels for outer,
vectorised numpy for inner.

Size breakdown (approx):
    TP      : 12
    SL      : 7
    timeout : 10
    partial : 2
    trailing: 2
    meta_thr: 16
    min_prob: 4
    spread  : 4
    fill_prob: 4
    kelly   : 6
    → outer 12*7*10*2*2 = 3,360 simulate_labels calls
    → inner 16*4*4*4*6   = 6,144 per outer
    → total              ~20M rows (we filter n_trades >= 30 → ~500k survivors)

Runtime estimate: 3,360 × ~1s rust sim = 56 min. Inner is numpy-only,
microseconds per config. Worth it to find the real optimum if any.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge                  # noqa: E402


CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
V2_PATH = Path("/home/scalper/scalper-bot/models/stacker_meta_v2.npz")
OUT = Path("/home/scalper/scalper-bot/models/grid_live_v6_wide.json")

# Expanded grid
TP_GRID = [0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]
SL_GRID = [0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.20]
TIMEOUT_GRID = [600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950]
PARTIAL_GRID = [True, False]
TRAILING_GRID = [True, False]
META_THR_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90]
MIN_PROB_GRID = [0.45, 0.50, 0.55, 0.60]
SPREAD_BPS_GRID = [0, 1, 2, 4]
FILL_PROB_GRID = [1.0, 0.9, 0.8, 0.6]
KELLY_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]

MIN_TRADES = 30


def _load_cache():
    cand = sorted(CACHE_DIR.glob("samples_v3_*_mid_paths.npy"))
    if not cand:
        raise FileNotFoundError(f"No v3 cache in {CACHE_DIR}")
    cand.sort(key=lambda p: p.stat().st_size, reverse=True)
    prefix = str(cand[0])[: -len("_mid_paths.npy")]
    print(f"[gridw] using cache prefix: {prefix}")
    return {
        "prefix": prefix,
        "y": np.load(f"{prefix}_y.npy"),
        "pnl": np.load(f"{prefix}_pnl.npy"),
        "mid_paths": np.load(f"{prefix}_mid_paths.npy"),
        "entry_long": np.load(f"{prefix}_entry_long.npy"),
        "entry_short": np.load(f"{prefix}_entry_short.npy"),
    }


def main():
    print("[gridw] loading cache + meta v2")
    c = _load_cache()
    d = np.load(V2_PATH, allow_pickle=False)
    stacker_soft = d["stacker_soft"]
    meta_prob = d["meta_prob"]
    n_tr = int(d["n_train"])
    N = c["y"].shape[0]

    import os as _os
    honest_start = int(_os.environ.get("GRID_HONEST_START", n_tr))
    if honest_start != n_tr:
        print(f"[gridw] eval boundary: n_tr={n_tr} → using honest_start={honest_start}")
    eval_lo = honest_start

    primary_pred = stacker_soft.argmax(axis=-1)
    primary_max = stacker_soft.max(axis=-1)
    non_flat = primary_pred != 2

    # Precompute fill masks per fp value — deterministic + comparable across configs
    rng = np.random.default_rng(42)
    fill_masks = {fp: (rng.random(N) < fp if fp < 1.0 else np.ones(N, dtype=bool))
                   for fp in FILL_PROB_GRID}

    print(f"[gridw] N={N:,}  tail={N - eval_lo:,}  non_flat_tail={int((non_flat[eval_lo:]).sum()):,}")
    print(f"[gridw] non_flat_tail_pass_minprob {int(((non_flat & (primary_max >= 0.45)) [n_tr:]).sum()):,}")

    outer_combos = list(itertools.product(TP_GRID, SL_GRID, TIMEOUT_GRID,
                                            PARTIAL_GRID, TRAILING_GRID))
    print(f"[gridw] outer configs: {len(outer_combos):,}  "
          f"inner configs per outer: "
          f"{len(META_THR_GRID)*len(MIN_PROB_GRID)*len(SPREAD_BPS_GRID)*len(FILL_PROB_GRID)*len(KELLY_GRID):,}")

    rows = []
    t_start = time.monotonic()
    for i_combo, (tp, sl, to_ticks, partial, trailing) in enumerate(outer_combos):
        tp_arr = np.full(N, tp, dtype=np.float64)
        sl_arr = np.full(N, sl, dtype=np.float64)
        to_arr = np.full(N, to_ticks, dtype=np.int64)
        out = rust_bridge.simulate_labels(
            c["entry_long"], c["entry_short"], c["mid_paths"],
            tp_arr, sl_arr, to_arr,
            commission_win_pct=0.04, commission_loss_pct=0.07,
            partial_enabled=partial, trailing_enabled=trailing, fill_latency_ms=150.0,
        )
        pnl_long = out["pnl_long"].astype(np.float64)
        pnl_short = out["pnl_short"].astype(np.float64)
        real = np.where(primary_pred == 0, pnl_long,
                np.where(primary_pred == 1, pnl_short, 0.0))

        for thr, min_p, spread_bps, fp, kelly in itertools.product(
            META_THR_GRID, MIN_PROB_GRID, SPREAD_BPS_GRID, FILL_PROB_GRID, KELLY_GRID
        ):
            spread_cost = spread_bps / 100.0
            gate = non_flat & (primary_max >= min_p) & (meta_prob >= thr)
            take = gate & fill_masks[fp]
            take_tail = take.copy()
            take_tail[:eval_lo] = False
            n_trades = int(take_tail.sum())
            if n_trades < MIN_TRADES:
                continue
            real_net = (real[take_tail] - spread_cost) * kelly
            wr = float((real_net > 0).mean())
            net_pct = float(real_net.sum())
            sharpe = float(real_net.mean() / (real_net.std() + 1e-9) * np.sqrt(len(real_net)))
            eq = np.cumsum(real_net)
            max_dd = float(np.max(eq.max() - eq) if len(eq) else 0)

            rows.append({
                "tp": tp, "sl": sl, "timeout": to_ticks,
                "partial": partial, "trailing": trailing,
                "meta_thr": thr, "min_prob": min_p,
                "spread_bps": spread_bps, "fill_prob": fp, "kelly": kelly,
                "n_trades": n_trades,
                "win_rate_pct": wr * 100,
                "net_pct": net_pct,
                "max_dd_pct": max_dd,
                "sharpe": sharpe,
            })

        if (i_combo + 1) % 50 == 0 or i_combo + 1 == len(outer_combos):
            dt = time.monotonic() - t_start
            eta = dt / (i_combo + 1) * (len(outer_combos) - i_combo - 1)
            print(f"[gridw] {i_combo+1}/{len(outer_combos)}  {dt/60:.1f}min elapsed  "
                  f"ETA {eta/60:.1f}min  rows={len(rows):,}")

    dt = time.monotonic() - t_start
    print(f"[gridw] done in {dt/60:.1f}min  total rows={len(rows):,}")

    rows.sort(key=lambda r: -r["net_pct"])
    profitable = [r for r in rows if r["net_pct"] > 0]
    print(f"[gridw] profitable configs (net>0, n>={MIN_TRADES}): "
          f"{len(profitable):,} / {len(rows):,}")

    result = {
        "n_samples": N, "n_train": n_tr, "n_eval": N - n_tr,
        "top_by_net": rows[:100],
        "top_by_sharpe": sorted(rows, key=lambda r: -r["sharpe"])[:100],
        "top_by_dd_adj": sorted(rows, key=lambda r: (r["net_pct"] - r["max_dd_pct"]))[::-1][:100],
        "n_rows": len(rows),
        "n_profitable": len(profitable),
    }
    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"[gridw] wrote {OUT}")

    if profitable:
        print("\n=== TOP 15 BY NET (net > 0, n >= 30) ===")
        print(f"{'tp':>5} {'sl':>5} {'to_s':>5} {'thr':>5} {'kly':>5} {'mp':>4} {'fp':>4} "
              f"{'p/t':>4} {'n':>5} {'WR%':>6} {'net%':>7} {'DD%':>6} {'sharpe':>7}")
        for r in rows[:15]:
            pt = ("1" if r["partial"] else "0") + ("1" if r["trailing"] else "0")
            print(f"{r['tp']:>5.2f} {r['sl']:>5.2f} {r['timeout']/10:>4.0f}s "
                  f"{r['meta_thr']:>5.2f} {r['kelly']:>5.2f} {r['min_prob']:>4.2f} "
                  f"{r['fill_prob']:>4.1f} {pt:>4} {r['n_trades']:>5d} "
                  f"{r['win_rate_pct']:>5.1f}% {r['net_pct']:>+6.2f}% "
                  f"{r['max_dd_pct']:>5.2f}% {r['sharpe']:>+7.2f}")


if __name__ == "__main__":
    main()
