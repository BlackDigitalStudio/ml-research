#!/usr/bin/env python3
"""~100k-config grid на 4-arch ансамбле (xgb + cnn + trans + feat_mlp)
обученном на unfiltered кэше samples_v3_60000h_1777503633.

Расширенный sweep:
  outer (240):
    tp = [0.15, 0.20, 0.25, 0.30, 0.35]      (5)
    sl = [0.08, 0.10, 0.12, 0.15]            (4)
    to = [600, 1200, 1800]                   (3)  60/120/180s @ 100ms
    par × tr = (F,F), (F,T), (T,F), (T,T)    (4)
    = 5 × 4 × 3 × 2 × 2 = 240
  inner (400):
    min_prob = [0.35, 0.40, 0.45, 0.50, 0.55] (5)
    spread   = [0.0, 0.02, 0.04, 0.06]        (4)
    fill     = [1.0, 0.8, 0.6, 0.4, 0.2]      (5)
    kelly    = [0.10, 0.20, 0.35, 0.50]       (4)
    = 5 × 4 × 5 × 4 = 400
Total: 240 outer × 400 inner = 96,000 (~100k)

Estimated wall time: ~3-5 минут (warm cache).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import rust_bridge  # noqa: E402

CACHE_PREFIX = REPO / "data/_cache/samples_v3_60000h_1777503633"
M = REPO / "models/ensemble_oof_b300_v2"

UP, DN, FL = 0, 1, 2
INITIAL_CAPITAL = 100.0
HOLDOUT_FRAC = 0.20


def build_full_ensemble_soft():
    y_full = np.load(str(CACHE_PREFIX) + "_y.npy").astype(np.int64)
    n_total = len(y_full)
    n_h = int(n_total * HOLDOUT_FRAC); n_w = n_total - n_h

    oof = {
        "xgb":      np.load(M/"xgb_oof.npy"),
        "cnn":      np.load(M/"cnn_oof.npy"),
        "trans":    np.load(M/"trans_oof.npy"),
        "feat_mlp": np.load(M/"feat_mlp_oof.npy"),
    }
    hold = {
        "xgb":      np.load(M/"xgb_holdout.npy"),
        "cnn":      np.load(M/"cnn_holdout.npy"),
        "trans":    np.load(M/"trans_holdout.npy"),
        "feat_mlp": np.load(M/"feat_mlp_holdout.npy"),
    }

    full = {}
    for k in oof:
        full[k] = np.concatenate([oof[k], hold[k]], axis=0).astype(np.float32)
        assert len(full[k]) == n_total, f"{k}: {len(full[k])} != {n_total}"

    avg = np.mean([full["xgb"], full["cnn"], full["trans"], full["feat_mlp"]], axis=0)
    return avg, y_full, n_w, n_h


def main():
    print("[grid-100k] === LOADING ensemble ===", flush=True)
    t0 = time.time()
    avg_soft, y_full, n_w, n_h = build_full_ensemble_soft()
    n_total = len(y_full)
    pred = avg_soft.argmax(1)
    max_prob = avg_soft.max(1)
    print(f"[grid-100k] ensemble loaded ({time.time()-t0:.1f}s): n_total={n_total}, n_holdout={n_h}", flush=True)

    entry_long = np.load(str(CACHE_PREFIX)+"_entry_long.npy").astype(np.float64)
    entry_short = np.load(str(CACHE_PREFIX)+"_entry_short.npy").astype(np.float64)
    mid_paths_path = str(CACHE_PREFIX) + "_mid_paths.npy"

    n_eff_days = n_h / 17680.0  # 86400s/day / 5s sample stride

    # ── Расширенный outer sweep ────
    tp_list = [0.15, 0.20, 0.25, 0.30, 0.35]
    sl_list = [0.08, 0.10, 0.12, 0.15]
    to_list = [600, 1200, 1800]
    par_list = [False, True]
    tr_list = [False, True]
    outer = list(itertools.product(tp_list, sl_list, to_list, par_list, tr_list))
    print(f"[grid-100k] outer combos: {len(outer)}", flush=True)

    # ── Расширенный inner sweep ────
    min_prob_list = [0.35, 0.40, 0.45, 0.50, 0.55]
    spread_list   = [0.0, 0.02, 0.04, 0.06]
    fill_list     = [1.0, 0.8, 0.6, 0.4, 0.2]
    kelly_list    = [0.10, 0.20, 0.35, 0.50]
    n_inner = len(min_prob_list) * len(spread_list) * len(fill_list) * len(kelly_list)
    print(f"[grid-100k] inner combos per outer: {n_inner}", flush=True)
    print(f"[grid-100k] TOTAL configs: {len(outer) * n_inner}", flush=True)
    print(f"[grid-100k] holdout effective days: {n_eff_days:.2f}", flush=True)

    cfgs_for_rust = [
        {"tp": tp, "sl": sl, "to": to, "par": par, "tr": tr}
        for (tp, sl, to, par, tr) in outer
    ]

    t0_sim = time.time()
    fused = rust_bridge.simulate_labels_grid(
        entry_long=entry_long, entry_short=entry_short,
        mid_paths=mid_paths_path,
        configs=cfgs_for_rust,
        pred=pred.astype(np.int64),
        max_prob=max_prob.astype(np.float64),
        holdout_start=n_w,
        n_eff_days=n_eff_days,
        inner_min_probs=min_prob_list,
        inner_spreads=spread_list,
        inner_fill_probs=fill_list,
        inner_kelly_fracs=kelly_list,
        inner_kelly_cap=19.0,
        inner_initial_capital=INITIAL_CAPITAL,
        inner_seed=42,
    )
    print(f"[grid-100k] fused outer+inner: {time.time()-t0_sim:.1f}s", flush=True)
    all_results = fused["inner_results"]
    print(f"[grid-100k] inner combos returned: {len(all_results)}", flush=True)

    # Ranking
    profitable = [r for r in all_results if r["n_trades"] >= 30 and r["net_return_pct"] > 0]
    print(f"\n[grid-100k] profitable & n_trades≥30: {len(profitable)}/{len(all_results)}")
    by_net    = sorted(all_results, key=lambda r: r["net_return_pct"], reverse=True)
    by_sharpe = sorted([r for r in all_results if r["n_trades"] >= 30],
                       key=lambda r: r["sharpe"], reverse=True)
    by_ev     = sorted([r for r in all_results if r["n_trades"] >= 30],
                       key=lambda r: r["ev_per_trade_pct"], reverse=True)

    def _print_top(rows, label, k=15):
        print(f"\n=== TOP {k} by {label} ===")
        print(f"{'tp':>4s} {'sl':>4s} {'to':>4s} {'p':>1s}{'t':>1s} {'mp':>4s} {'spr':>4s} {'fp':>4s} {'k':>4s} | "
              f"{'n':>5s} {'wr%':>5s} {'net%':>7s} {'sharpe':>7s} {'dd%':>5s} {'ev%':>6s} {'t/d':>6s}")
        for r in rows[:k]:
            p_flag = "T" if r['partial'] else "F"
            t_flag = "T" if r['trailing'] else "F"
            print(f"{r['tp']:>4.2f} {r['sl']:>4.2f} {r['timeout']:>4d} "
                  f"{p_flag:>1s}{t_flag:>1s} "
                  f"{r['min_prob']:>4.2f} {r['spread_pct']:>4.2f} {r['fill_prob']:>4.2f} {r['kelly_frac']:>4.2f} | "
                  f"{r['n_trades']:>5d} {r['win_rate_pct']:>5.1f} {r['net_return_pct']:>7.2f} "
                  f"{r['sharpe']:>7.3f} {r['max_dd_pct']:>5.1f} {r['ev_per_trade_pct']:>6.3f} {r['trades_per_day']:>6.1f}")

    _print_top(by_net, "net_return_pct", 20)
    _print_top(by_sharpe, "sharpe", 15)
    _print_top(by_ev, "ev_per_trade_pct", 15)

    out = {
        "top_by_net": by_net[:100],
        "top_by_sharpe": by_sharpe[:100],
        "top_by_ev": by_ev[:100],
        "n_configs": len(all_results),
        "n_profitable": len(profitable),
        "holdout_eff_days": n_eff_days,
        "outer_sweep": {
            "tp_list": tp_list, "sl_list": sl_list, "to_list": to_list,
            "par_list": par_list, "tr_list": tr_list,
        },
        "inner_sweep": {
            "min_prob_list": min_prob_list, "spread_list": spread_list,
            "fill_list": fill_list, "kelly_list": kelly_list,
        },
    }
    out_path = REPO/"runs/grid_ensemble_b300_100k_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
