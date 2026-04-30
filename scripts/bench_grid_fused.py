#!/usr/bin/env python3
"""Шаг Б — fused 27-config grid в одном Rust вызове.

Сравнить с baseline (62.6 мин projected sim_labels-only) и Шагом А
(22.6 мин projected). Один проход вместо 27.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CACHE = REPO / "data/_cache/samples_v3_60000h_1777503633"


def main():
    print("[bench-fused] === LOAD entry/mid metadata ===", flush=True)
    t0 = time.time()
    entry_long = np.load(str(CACHE) + "_entry_long.npy")
    entry_short = np.load(str(CACHE) + "_entry_short.npy")
    print(f"  entry arrays: {time.time()-t0:.2f}s ({len(entry_long)} samples)", flush=True)

    mid_path_arg = str(CACHE) + "_mid_paths.npy"

    # Те же 27 outer configs что в grid_ensemble_b300.py
    tp_list = [0.20, 0.25, 0.30]
    sl_list = [0.10, 0.12, 0.15]
    to_list = [600, 1200, 1800]
    configs = []
    for tp in tp_list:
        for sl in sl_list:
            for to in to_list:
                configs.append({"tp": tp, "sl": sl, "to": to, "par": False, "tr": False})

    print(f"[bench-fused] === RUN grid_sim with {len(configs)} configs ===", flush=True)
    from src import rust_bridge

    t0 = time.time()
    out = rust_bridge.simulate_labels_grid(
        entry_long=entry_long, entry_short=entry_short,
        mid_paths=mid_path_arg,
        configs=configs,
    )
    dt = time.time() - t0
    print(f"\n[bench-fused] grid_sim TOTAL: {dt:.2f}s", flush=True)
    print(f"[bench-fused] pnl_long shape: {out['pnl_long'].shape}", flush=True)
    print(f"[bench-fused] pnl_short shape: {out['pnl_short'].shape}", flush=True)

    # Sanity: каждый config должен иметь хотя бы какие-то ненулевые pnl
    for k in range(len(configs)):
        nl = (out['pnl_long'][k] != 0).sum()
        ns_ = (out['pnl_short'][k] != 0).sum()
        c = configs[k]
        if k < 3 or k == len(configs) - 1:
            print(f"  cfg[{k}] tp={c['tp']} sl={c['sl']} to={c['to']}: "
                  f"non-zero long {nl}/{len(out['pnl_long'][k])}, short {ns_}", flush=True)

    print(f"\n[bench-fused] BASELINE 27-outer:    62.6 min")
    print(f"[bench-fused] Шаг А 27-outer:          22.6 min")
    print(f"[bench-fused] Шаг Б (this run):        {dt/60:.2f} min")
    print(f"[bench-fused] Speedup vs baseline:     {62.6*60/dt:.1f}×")
    print(f"[bench-fused] Speedup vs Шаг А:        {22.6*60/dt:.1f}×")

    import resource
    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[bench-fused] peak RSS: {mb:.0f} MB")


if __name__ == "__main__":
    main()
