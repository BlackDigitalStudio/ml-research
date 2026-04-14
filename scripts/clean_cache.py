#!/usr/bin/env python3
"""Clean + robust-normalize a training cache.

Our cache pipeline occasionally emits rare outliers (per-column p99 vs max
ratios >100x) and very rare NaN values — enough to NaN the loss and destroy
training. Rather than train through that, we:

  1. replace NaN / ±inf → 0
  2. clip X_lob to [0, 100] (depth volumes >100 are bad data)
  3. clip each X_feat column to its [0.1, 99.9] percentile range (kills
     the long tail without losing signal in the middle)
  4. robust z-score per column: (x - median) / (1.4826 * MAD)
  5. persist the per-column (clip_lo, clip_hi, median, scale) stats inside
     the cache NPZ so the live executor can apply the SAME transform

Run this once after build_cache.py; it rewrites the NPZ in-place and emits
diagnostics.

Usage:
    python scripts/clean_cache.py --cache data/_cache/cache.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

CLIP_LO_PCT = 1.0    # tight clip percentiles — kill the heavy tail early
CLIP_HI_PCT = 99.0
LOB_CLIP_MAX = 100.0
MIN_SCALE = 1e-6     # prevent div-by-zero when a column is near-constant


def _robust_norm(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Clip + z-score per column. Returns (X_clean, stats_dict).

    Two-layer robustness:
      1. Clip to [p1, p99] to remove heavy-tail outliers.
      2. Scale = max(std_of_clipped, IQR/1.349, MIN_SCALE) — falls back
         gracefully when std=0 (column near-constant) or MAD=0 (bimodal).

    MAD-only normalization (what we tried first) collapses when >50% of
    values are identical — common in binary/flag-like features — because
    MAD is literally 0, scale defaults to 1, and clipped-boundary values
    survive unchanged. Std of clipped values handles this case.
    """
    stats = {}
    X_out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[1]):
        col = X[:, i].astype(np.float64)
        lo = float(np.percentile(col, CLIP_LO_PCT))
        hi = float(np.percentile(col, CLIP_HI_PCT))
        if hi <= lo:  # degenerate: column is (near) constant
            lo -= 1.0
            hi += 1.0
        clipped = np.clip(col, lo, hi)
        med = float(np.median(clipped))
        std = float(clipped.std())
        q25, q75 = np.percentile(clipped, [25, 75])
        iqr_scale = float((q75 - q25) / 1.349)
        scale = max(std, iqr_scale, MIN_SCALE)
        normed = (clipped - med) / scale
        X_out[:, i] = normed.astype(np.float32)
        stats[i] = {"clip_lo": lo, "clip_hi": hi, "median": med, "scale": scale}
    return X_out, stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="path to cache.npz")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERR: {cache_path} not found")
        return 1

    t0 = time.time()
    print(f"[clean] loading {cache_path}...")
    d = np.load(cache_path, allow_pickle=False)

    X_lob = np.ascontiguousarray(d["X_lob"]).astype(np.float32)
    X_feat_raw = np.ascontiguousarray(d["X_feat"]).astype(np.float32)
    y = np.asarray(d["y"]).astype(np.int64)
    target_pnl = np.asarray(d["target_pnl"]).astype(np.float32)

    # 1/2: NaN/inf scrubbing + LOB clip
    lob_nan = int(np.isnan(X_lob).sum()) + int(np.isinf(X_lob).sum())
    feat_nan = int(np.isnan(X_feat_raw).sum()) + int(np.isinf(X_feat_raw).sum())
    pnl_nan = int(np.isnan(target_pnl).sum()) + int(np.isinf(target_pnl).sum())
    print(f"[clean] NaN/inf: X_lob={lob_nan} X_feat={feat_nan} target_pnl={pnl_nan}")

    X_lob = np.nan_to_num(X_lob, nan=0.0, posinf=0.0, neginf=0.0)
    X_lob = np.clip(X_lob, 0.0, LOB_CLIP_MAX)
    print(f"[clean] X_lob range after clean: [{X_lob.min():.3g}, {X_lob.max():.3g}]")

    X_feat_raw = np.nan_to_num(X_feat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    target_pnl = np.nan_to_num(target_pnl, nan=0.0, posinf=0.0, neginf=0.0)

    # 3/4: per-column robust normalization
    pre = {"range": [float(X_feat_raw.min()), float(X_feat_raw.max())],
           "std": float(X_feat_raw.std())}
    X_feat, stats = _robust_norm(X_feat_raw)
    post = {"range": [float(X_feat.min()), float(X_feat.max())],
            "std": float(X_feat.std())}
    print(f"[clean] X_feat pre:  range={pre['range']}  std={pre['std']:.3g}")
    print(f"[clean] X_feat post: range={post['range']}  std={post['std']:.3g}")

    # Worst-preserved columns (highest abs value after normalization)
    col_max = np.abs(X_feat).max(axis=0)
    top5 = np.argsort(-col_max)[:5]
    print(f"[clean] top-5 surviving-extreme columns after norm: "
          f"{[(int(i), float(col_max[i])) for i in top5]}")

    # 5: save stats in a packed format (34 × 4 float32 array + keys)
    stat_arr = np.zeros((X_feat.shape[1], 4), dtype=np.float32)
    for i in range(X_feat.shape[1]):
        s = stats[i]
        stat_arr[i, 0] = s["clip_lo"]
        stat_arr[i, 1] = s["clip_hi"]
        stat_arr[i, 2] = s["median"]
        stat_arr[i, 3] = s["scale"]

    if args.dry_run:
        print("[clean] --dry-run, not writing")
        return 0

    np.savez(
        cache_path,
        X_lob=X_lob, X_feat=X_feat, y=y, target_pnl=target_pnl,
        feat_norm_stats=stat_arr,  # (n_feat, 4): clip_lo, clip_hi, median, scale
    )
    print(f"[clean] rewrote {cache_path} in {time.time()-t0:.1f}s")
    print(f"[clean] cache now carries feat_norm_stats — "
          f"apply to live inputs as: x = clip(x, lo, hi); (x - median) / scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
