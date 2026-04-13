#!/usr/bin/env python3
"""One-shot cache builder. Calls Trainer.build_samples_cached and reports
RAM peak + wall time + cache file sizes.

Usage:
    SCALPER_USE_RUST=1 python scripts/build_cache.py --hours 76
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=24,
                   help="last N depth files to load (Tardis: 1 file/day, recorder: 1/hour)")
    p.add_argument("--data-dir", default="/workspace/scalper-bot/data")
    p.add_argument("--model-dir", default="/workspace/scalper-bot/models")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    cfg = load_config(env_path=str(Path(args.data_dir).parent / "config.env"))
    # Override paths after load (Config is frozen — use object.__setattr__)
    object.__setattr__(cfg, "data_dir", Path(args.data_dir))
    object.__setattr__(cfg, "model_dir", Path(args.model_dir))

    use_rust = os.environ.get("SCALPER_USE_RUST", "0") in ("1", "true", "yes")
    print(f"[build_cache] hours={args.hours} data_dir={args.data_dir} "
          f"force={args.force} SCALPER_USE_RUST={use_rust}")

    t0 = time.time()
    trainer = Trainer(cfg)
    X_lob, X_feat, y, mid, target_pnl = trainer.build_samples_cached(
        hours=args.hours, force_rebuild=args.force,
    )
    dt = time.time() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / (1024**2)

    cls_counts = np.bincount(y, minlength=3)
    print(f"\n[build_cache] DONE in {dt/60:.1f} min, peak RSS={rss_gb:.2f} GB")
    print(f"  X_lob       {X_lob.shape}  {X_lob.dtype}")
    print(f"  X_feat      {X_feat.shape}  {X_feat.dtype}")
    print(f"  y           {y.shape}  {y.dtype}  classes UP={cls_counts[0]} DN={cls_counts[1]} FL={cls_counts[2]}")
    print(f"  target_pnl  {target_pnl.shape}  {target_pnl.dtype}  "
          f"mean={float(target_pnl.mean()):.4f}% std={float(target_pnl.std()):.4f}%")

    cache_dir = cfg.data_dir / "_cache"
    if cache_dir.exists():
        files = sorted(cache_dir.glob("samples_*"))
        if files:
            total = sum(f.stat().st_size for f in files)
            print(f"\n  cache files ({len(files)}, {total/1e9:.2f} GB):")
            for f in files:
                print(f"    {f.name}  {f.stat().st_size/1e6:.0f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
