#!/usr/bin/env python3
"""Микро-бенчмарк одного outer-вызова grid: измерение времени I/O vs compute.

Запускает rust_bridge.simulate_labels один раз на полном 1.45M кеше
и разбивает время на: load .npy в Python → np.save в /tmp → Rust subprocess
(с внутренним таймером в Rust) → np.load результатов.

Это даёт профиль ОДНОГО outer config из grid (которых сейчас 27 в
grid_ensemble_b300.py). Умножение результата на 27 = baseline для grid.

Mode:
  python bench_grid_outer.py             — старый путь (Python np.load + np.save в /tmp)
  python bench_grid_outer.py --fast-path — Шаг А: пробросить путь к mid_paths напрямую
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CACHE = REPO / "data/_cache/samples_v3_60000h_1777503633"


def main():
    fast = "--fast-path" in sys.argv
    print(f"[bench] mode = {'FAST-PATH (Шаг А)' if fast else 'BASELINE'}", flush=True)

    print("[bench] === LOAD CACHE ===", flush=True)
    t0 = time.time()
    entry_long = np.load(str(CACHE) + "_entry_long.npy")
    entry_short = np.load(str(CACHE) + "_entry_short.npy")
    print(f"  entry arrays: {time.time()-t0:.2f}s", flush=True)

    if fast:
        n_total = entry_long.shape[0]
        # mid_paths.npy на диске — пробрасываем путь
        mid_paths_arg = str(CACHE) + "_mid_paths.npy"
        print(f"  mid_paths: PATH MODE → {mid_paths_arg}", flush=True)
    else:
        t0 = time.time()
        mid_paths = np.load(str(CACHE) + "_mid_paths.npy")
        n_total = len(mid_paths)
        size_gb = mid_paths.nbytes / 1e9
        print(f"  mid_paths {mid_paths.shape} dtype={mid_paths.dtype} = {size_gb:.2f} GB | load {time.time()-t0:.2f}s",
              flush=True)
        mid_paths_arg = mid_paths

    print("\n[bench] === ONE simulate_labels CALL ===", flush=True)
    from src import rust_bridge

    tp = np.full(n_total, 0.20, dtype=np.float64)
    sl = np.full(n_total, 0.10, dtype=np.float64)
    to = np.full(n_total, 1200, dtype=np.int64)

    t0 = time.time()
    sim = rust_bridge.simulate_labels(
        entry_long=entry_long, entry_short=entry_short,
        mid_paths=mid_paths_arg,
        tp_pct=tp, sl_pct=sl, timeout_ticks=to,
        partial_enabled=False, trailing_enabled=False,
    )
    dt = time.time() - t0
    print(f"\n[bench] simulate_labels TOTAL: {dt:.2f}s", flush=True)
    print(f"[bench] pnl_long shape: {sim['pnl_long'].shape}, "
          f"non-zero: {(sim['pnl_long']!=0).sum()}", flush=True)
    print(f"\n[bench] PROJECTED 27-outer grid (sim_labels only): {dt*27:.0f}s = {dt*27/60:.1f}min")

    # Memory check
    import resource
    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[bench] peak RSS: {mb:.0f} MB")


if __name__ == "__main__":
    main()
