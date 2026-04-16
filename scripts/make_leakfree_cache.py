#!/usr/bin/env python3
"""Create a leak-free cache slice [0, 70k) from the full 93k v3 cache.

Bakeoff_v3 trained on [0, 74k) (internal 80/20 of 93k). Grid_live then
evaluated on [70k, 93k] — the first 4k overlapped with primary training.
Retraining on a strictly shorter slice [0, 70k) makes the grid's eval
tail [70k, 93k] fully unseen. Once retrained primaries re-infer on the
full 93k, the downstream stacker/meta/grid have a legitimate OOS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
SRC_PREFIX_NAME = "samples_v3_999h_1776165949"
DST_PREFIX_NAME = "samples_v3_leakfree70k_1776165949"
N_KEEP = 70000

def main():
    src = CACHE_DIR / SRC_PREFIX_NAME
    dst = CACHE_DIR / DST_PREFIX_NAME
    files = ["X_lob", "X_feat", "y", "mid", "pnl",
             "mid_paths", "entry_long", "entry_short"]
    for f in files:
        sp = CACHE_DIR / f"{SRC_PREFIX_NAME}_{f}.npy"
        dp = CACHE_DIR / f"{DST_PREFIX_NAME}_{f}.npy"
        if dp.exists():
            print(f"[leakfree] skip existing {dp.name}")
            continue
        arr = np.load(sp, mmap_mode="r")
        sliced = np.asarray(arr[:N_KEEP])
        np.save(dp, sliced)
        print(f"[leakfree] wrote {dp.name}  shape={sliced.shape}  dtype={sliced.dtype}")
    print(f"[leakfree] done. Use cache prefix:\n  {dst}")


if __name__ == "__main__":
    main()
