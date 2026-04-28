#!/usr/bin/env python3
"""Vol-aware triple-barrier label re-generation.

Reads existing v3 cache, computes per-sample TP/SL пропорциональные realized
vol_120s feature (idx 31), saves new labels + per-sample TP/SL/timeout arrays
+ realized PnL под new labels.

Output cache: {prefix}_volaware_*.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402

UP, DN, FL = 0, 1, 2

# Feature index where rv_120s lives
RV120_IDX = 31


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-prefix", required=True,
                    help="Existing cache prefix (без _suffix)")
    ap.add_argument("--out-suffix", default="volaware",
                    help="Suffix к новому cache, e.g. samples_v3_999h_volaware")
    ap.add_argument("--scale", type=float, default=6.3,
                    help="TP = scale * rv_120s (decimal). Calibrated so median TP ≈ 0.20%%")
    ap.add_argument("--rr", type=float, default=2.0, help="TP / SL ratio")
    ap.add_argument("--tp-min", type=float, default=0.10, help="TP cap min %%")
    ap.add_argument("--tp-max", type=float, default=0.50, help="TP cap max %%")
    ap.add_argument("--sl-min", type=float, default=0.05, help="SL cap min %%")
    ap.add_argument("--sl-max", type=float, default=0.25, help="SL cap max %%")
    ap.add_argument("--timeout-ticks", type=int, default=1300)
    args = ap.parse_args()

    p = args.cache_prefix
    print(f"[volaware] loading cache {p}", flush=True)
    X_feat = np.load(f"{p}_X_feat.npy")
    mid_paths = np.load(f"{p}_mid_paths.npy")
    entry_long = np.load(f"{p}_entry_long.npy")
    entry_short = np.load(f"{p}_entry_short.npy")
    n = len(X_feat)
    print(f"[volaware] n={n}", flush=True)

    # rv_120s in decimal — convert to TP pct
    rv = X_feat[:, RV120_IDX].astype(np.float64)
    rv = np.nan_to_num(rv, nan=np.median(rv[rv > 0]) if (rv > 0).any() else 0.0,
                       posinf=0.0, neginf=0.0)
    rv = np.maximum(rv, 0.0)
    print(f"[volaware] rv_120s median={np.median(rv):.6f} "
          f"p95={np.percentile(rv, 95):.6f}", flush=True)

    # vol-aware TP/SL pct — saturate at caps
    tp_pct = args.scale * rv * 100.0  # decimal → percent
    tp_pct = np.clip(tp_pct, args.tp_min, args.tp_max)
    sl_pct = tp_pct / args.rr
    sl_pct = np.clip(sl_pct, args.sl_min, args.sl_max)
    timeout_ticks = np.full(n, args.timeout_ticks, dtype=np.int64)
    print(f"[volaware] TP%: median={np.median(tp_pct):.3f} "
          f"p25={np.percentile(tp_pct, 25):.3f} p75={np.percentile(tp_pct, 75):.3f}",
          flush=True)
    print(f"[volaware] SL%: median={np.median(sl_pct):.3f}", flush=True)

    t0 = time.time()
    sim = rust_bridge.simulate_labels(
        entry_long=entry_long, entry_short=entry_short,
        mid_paths=mid_paths,
        tp_pct=tp_pct, sl_pct=sl_pct, timeout_ticks=timeout_ticks,
        partial_enabled=True, trailing_enabled=True,
    )
    pnl_long = sim["pnl_long"]
    pnl_short = sim["pnl_short"]
    print(f"[volaware] simulate_labels {n} samples in {time.time()-t0:.1f}s",
          flush=True)
    print(f"[volaware] pnl_long mean={pnl_long.mean():.4f}% "
          f"pos={(pnl_long > 0).mean():.3f}", flush=True)
    print(f"[volaware] pnl_short mean={pnl_short.mean():.4f}% "
          f"pos={(pnl_short > 0).mean():.3f}", flush=True)

    # Triple barrier labels (direction = best-side profitability):
    #   y = UP if pnl_long > 0 and pnl_long > pnl_short
    #   y = DN if pnl_short > 0 and pnl_short > pnl_long
    #   y = FL otherwise
    y = np.full(n, FL, dtype=np.int64)
    up_mask = (pnl_long > 0) & (pnl_long > pnl_short)
    dn_mask = (pnl_short > 0) & (pnl_short > pnl_long)
    y[up_mask] = UP
    y[dn_mask & ~up_mask] = DN
    print(f"[volaware] new y: UP={(y==UP).mean()*100:.1f}%  "
          f"DN={(y==DN).mean()*100:.1f}%  FL={(y==FL).mean()*100:.1f}%",
          flush=True)

    # Save new cache (volaware suffix)
    out_prefix = p + "_" + args.out_suffix
    np.save(f"{out_prefix}_y.npy", y)
    np.save(f"{out_prefix}_pnl_long.npy", pnl_long)
    np.save(f"{out_prefix}_pnl_short.npy", pnl_short)
    np.save(f"{out_prefix}_tp_pct.npy", tp_pct)
    np.save(f"{out_prefix}_sl_pct.npy", sl_pct)
    np.save(f"{out_prefix}_timeout_ticks.npy", timeout_ticks)
    # Symlink existing X_lob, X_feat, mid_paths, entry_long, entry_short to new prefix
    for tail in ["X_lob", "X_feat", "mid_paths", "entry_long", "entry_short"]:
        src = Path(f"{p}_{tail}.npy")
        dst = Path(f"{out_prefix}_{tail}.npy")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.absolute())

    print(f"\n[volaware] saved new cache: {out_prefix}_*.npy", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
