#!/usr/bin/env python3
"""Architecture bake-off V2 — full stacked ensemble + meta-labeling.

For each architecture in --archs: train with `train_generic`, collect its
val-set softmax. After all primaries trained, fit XGBoost stacker on
concatenated softmaxes + handcrafted feats → stacked primary prediction.
Then fit a meta-labeling XGBoost on the stacker's output to filter
which non-FLAT decisions to actually trade.

Outputs `runs/bakeoff_v2/`:
    leaderboard.json    — per-arch metrics + stacker + meta metrics
    <tag>.pt            — per-model state_dicts
    stacker.json        — XGBoost stacker
    meta.json           — XGBoost meta-labeler
    val_predictions.npz — val_softmax per arch + y_val + meta decisions

Usage:
    python scripts/bakeoff_v2.py --cache cache.npz \\
        --archs transformer patchtst mamba tcn chronos_bolt_tiny \\
        --epochs 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bakeoff_v1 import build_factory  # noqa: E402
from src.models.train import train_generic  # noqa: E402
from src.models.stacking import train_stacker  # noqa: E402
from src.models.meta_label import (  # noqa: E402
    build_meta_dataset, train_meta, MetaConfig,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True,
                   help="NPZ with X_lob, X_feat, y, target_pnl")
    p.add_argument("--archs", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="runs/bakeoff_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-stacking", action="store_true",
                   help="Only train primaries; don't fit stacker/meta")
    args = p.parse_args()

    data = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    X_lob = data["X_lob"]
    X_feat = data["X_feat"]
    y = data["y"]
    target_pnl = data["target_pnl"]
    print(f"[bakeoff2] loaded {len(y)} samples: X_lob {X_lob.shape} X_feat {X_feat.shape}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Level 1: train each primary, collect val_softmax ===
    primary_softmaxes: list[np.ndarray] = []
    primary_tags: list[str] = []
    y_val: np.ndarray | None = None
    X_feat_val: np.ndarray | None = None
    leaderboard = []

    for arch in args.archs:
        factory, tag = build_factory(arch)
        print(f"\n==== {tag} ====")
        t0 = time.time()
        model, metrics = train_generic(
            factory, X_lob, X_feat, y, target_pnl,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            seed=args.seed, tag=tag,
        )
        dt = time.time() - t0
        torch.save(model.state_dict(), out_dir / f"{tag}.pt")

        # Collect for stacking
        primary_softmaxes.append(metrics["val_softmax"])
        primary_tags.append(tag)
        if y_val is None:
            y_val = metrics["y_val"]
            X_feat_val = metrics["X_feat_val"]

        leaderboard.append({
            "arch": tag, "level": "primary",
            "best_bal_acc": metrics["best_bal_acc"],
            "params_M": metrics["params_M"],
            "train_seconds": dt,
            "epochs_run": len(metrics["history"]),
        })
        print(f"[bakeoff2] {tag} done — bal_acc={metrics['best_bal_acc']:.4f} "
              f"params={metrics['params_M']:.2f}M time={dt:.0f}s")

    if args.skip_stacking or len(primary_softmaxes) < 2:
        print("[bakeoff2] skipping stacker (need ≥2 primaries)")
    else:
        # === Level 2: train stacker on val softmaxes ===
        print("\n==== stacker ====")
        stacker, stk_metrics = train_stacker(
            primary_softmaxes, y_val, X_feat=X_feat_val,
        )
        print(f"[bakeoff2] stacker val_acc={stk_metrics['val_acc']:.4f} "
              f"bal_acc={stk_metrics['val_bal_acc']:.4f} "
              f"logloss={stk_metrics['val_logloss']:.4f}")
        stacker.save_model(str(out_dir / "stacker.json"))
        leaderboard.append({
            "arch": "STACKER", "level": "level2",
            "best_bal_acc": stk_metrics["val_bal_acc"],
            **{f"stacker_{k}": v for k, v in stk_metrics.items() if k != "per_class_recall"},
        })

        # === Level 3: meta-labeling on stacker output ===
        # Use the stacker's softmax as the primary for meta-labeling.
        from src.models.stacking import predict_stacked
        stacker_soft = predict_stacked(
            stacker, primary_softmaxes, X_feat=X_feat_val, use_feats=True,
        )
        # Align pnl with val split: train_generic used [n_train+gap:] as val.
        n = len(y)
        n_val = int(n * 0.2)
        n_train = n - n_val - 650
        pnl_val = target_pnl[n_train + 650:]

        print("\n==== meta-labeler on stacker ====")
        X_m, y_m, w_m = build_meta_dataset(
            stacker_soft, y_val, pnl_val, X_feat=X_feat_val,
            cfg=MetaConfig(pnl_threshold_pct=0.0),
        )
        print(f"[bakeoff2] meta dataset: {X_m.shape}  pos_rate={y_m.mean():.3f}")
        meta, meta_metrics = train_meta(X_m, y_m, w_m, cfg=MetaConfig())
        print(f"[bakeoff2] meta val_auc={meta_metrics['val_auc']:.4f} "
              f"precision={meta_metrics['val_precision']:.4f} "
              f"recall={meta_metrics['val_recall']:.4f}")
        meta.save_model(str(out_dir / "meta.json"))
        leaderboard.append({
            "arch": "META", "level": "level3",
            **{f"meta_{k}": v for k, v in meta_metrics.items()},
        })

        # Save val predictions for further analysis.
        np.savez(
            out_dir / "val_predictions.npz",
            y_val=y_val,
            stacker_soft=stacker_soft,
            pnl_val=pnl_val,
            **{f"soft_{t}": s for t, s in zip(primary_tags, primary_softmaxes)},
        )

    with (out_dir / "leaderboard.json").open("w") as f:
        json.dump({"rows": leaderboard, "args": vars(args)}, f, indent=2)

    print("\n=== Leaderboard ===")
    for r in leaderboard:
        if r.get("level") == "primary":
            print(f"  L1  {r['arch']:22s}  bal_acc={r['best_bal_acc']:.4f}  "
                  f"params={r['params_M']:.2f}M  time={r.get('train_seconds', 0):.0f}s")
        elif r.get("level") == "level2":
            print(f"  L2  {r['arch']:22s}  bal_acc={r['best_bal_acc']:.4f}  "
                  f"logloss={r.get('stacker_val_logloss', 0):.4f}")
        elif r.get("level") == "level3":
            print(f"  L3  {r['arch']:22s}  auc={r.get('meta_val_auc', 0):.4f}  "
                  f"prec={r.get('meta_val_precision', 0):.4f}  "
                  f"rec={r.get('meta_val_recall', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
