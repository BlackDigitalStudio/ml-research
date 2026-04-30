#!/usr/bin/env python3
"""Smoke-test: PT/TS параметры реально влияют на результат.

Прогоняет 6 outer-configs:
  baseline (par=False, tr=False) — нет partial, нет trailing
  par=True, partial_tp_progress = 0.30 / 0.50 / 0.70
  par=True+tr=True, partial_tp_progress=0.50, trailing_step1_progress = 0.30 / 0.70

И сравнивает n_trades, net_return — чтобы проверить что параметры действительно
дают разные исходы (они не игнорируются Rust бинарем).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import rust_bridge

CACHE_PREFIX = REPO / "data/_cache/samples_v3_60000h_1777503633"
M = REPO / "models/ensemble_oof_b300_v2"


def main():
    print("=== PT/TS sweep smoke test ===", flush=True)
    y_full = np.load(str(CACHE_PREFIX) + "_y.npy").astype(np.int64)
    n_total = len(y_full)
    n_h = int(n_total * 0.20); n_w = n_total - n_h

    def _full(name):
        oof = np.load(M / f"{name}_oof.npy")
        hold = np.load(M / f"{name}_holdout.npy")
        return np.concatenate([oof, hold], axis=0).astype(np.float32)
    avg = np.mean([_full("xgb"), _full("cnn"), _full("trans"), _full("feat_mlp")], axis=0)
    pred = avg.argmax(1)
    max_prob = avg.max(1)

    entry_long = np.load(str(CACHE_PREFIX)+"_entry_long.npy").astype(np.float64)
    entry_short = np.load(str(CACHE_PREFIX)+"_entry_short.npy").astype(np.float64)
    mid_paths_path = str(CACHE_PREFIX) + "_mid_paths.npy"

    base = {"tp": 0.25, "sl": 0.12, "to": 1200}
    configs = [
        # 1. Baseline — нет partial, нет trailing
        {**base, "par": False, "tr": False},
        # 2-4. Только partial — varying partial_tp_progress
        {**base, "par": True, "tr": False, "partial_tp_progress": 0.30},
        {**base, "par": True, "tr": False, "partial_tp_progress": 0.50},
        {**base, "par": True, "tr": False, "partial_tp_progress": 0.70},
        # 5-6. Partial + Trailing — varying trailing_step1_progress
        {**base, "par": True, "tr": True, "partial_tp_progress": 0.50,
         "trailing_step1_progress": 0.30, "trailing_step2_progress": 0.60},
        {**base, "par": True, "tr": True, "partial_tp_progress": 0.50,
         "trailing_step1_progress": 0.70, "trailing_step2_progress": 0.85},
    ]

    t0 = time.time()
    fused = rust_bridge.simulate_labels_grid(
        entry_long=entry_long, entry_short=entry_short,
        mid_paths=mid_paths_path,
        configs=configs,
        pred=pred.astype(np.int64), max_prob=max_prob.astype(np.float64),
        holdout_start=n_w, n_eff_days=n_h/17680.0,
        inner_min_probs=[0.45], inner_spreads=[0.0],
        inner_fill_probs=[1.0], inner_kelly_fracs=[0.10],
    )
    print(f"\nfused {len(configs)} outer × 1 inner: {time.time()-t0:.1f}s")

    print("\n--- Inner results ---")
    print(f"{'idx':>3s} {'par':>3s} {'tr':>3s} {'pt_prog':>7s} {'ts1_prog':>8s} {'ts2_prog':>8s} | "
          f"{'n':>5s} {'wr%':>5s} {'net%':>7s} {'mean_pl_long':>14s}")
    pnl_l_means = []
    for i, r in enumerate(fused["inner_results"]):
        m = float(fused["pnl_long"][i].mean())
        pnl_l_means.append(m)
        print(f"{i:>3d} {str(r['partial'])[0]:>3s} {str(r['trailing'])[0]:>3s} "
              f"{r['partial_tp_progress']:>7.2f} {r['trailing_step1_progress']:>8.2f} "
              f"{r['trailing_step2_progress']:>8.2f} | "
              f"{r['n_trades']:>5d} {r['win_rate_pct']:>5.1f} {r['net_return_pct']:>7.2f} "
              f"{m:>14.6f}")

    # Sanity: pnl_long mean должны различаться между конфигурациями.
    distinct = len(set([round(x, 6) for x in pnl_l_means]))
    print(f"\nPnL_long distinct values across {len(configs)} configs: {distinct}")
    if distinct == 1:
        print("WARN: все конфигурации дали одинаковый mean pnl_long — параметры PT/TS не дошли до simulate_trade!")
    else:
        print("OK: PT/TS параметры реально влияют на симуляцию.")


if __name__ == "__main__":
    main()
