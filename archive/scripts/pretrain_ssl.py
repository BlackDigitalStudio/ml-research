#!/usr/bin/env python3
"""Run SSL pretraining on the full Tardis depth corpus.

Usage:
    python scripts/pretrain_ssl.py --data-dir /workspace/scalper-bot/data \\
        --output /workspace/scalper-bot/runs/ssl_pretrain_v1 \\
        --epochs 20 --window 256 --batch 256

Saves backbone weights to <output>/final_backbone.pt — load into
downstream PatchTST classifier via:
    model.backbone.load_state_dict(torch.load("final_backbone.pt"))
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ssl import PretrainConfig, pretrain_loop  # noqa: E402
from src.ssl.model import BackboneConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="root data dir containing depth/*tardis*.parquet")
    p.add_argument("--output", required=True, help="output dir for checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--mask-ratio", type=float, default=0.20)
    p.add_argument("--samples-per-epoch", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--patch-len", type=int, default=16)
    args = p.parse_args()

    depth_dir = Path(args.data_dir) / "depth"
    depth_files = sorted(depth_dir.glob("*tardis*.parquet"))
    if not depth_files:
        print(f"No flat-schema tardis depth files in {depth_dir}", file=sys.stderr)
        return 2
    print(f"[pretrain_ssl] {len(depth_files)} flat depth files in {depth_dir}")

    cfg = PretrainConfig(
        backbone=BackboneConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            patch_len=args.patch_len,
            stride=args.patch_len,
        ),
        window_size=args.window,
        mask_ratio=args.mask_ratio,
        batch_size=args.batch,
        samples_per_epoch=args.samples_per_epoch,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
    )

    result = pretrain_loop(
        depth_paths=depth_files,
        output_dir=args.output,
        cfg=cfg,
    )
    out = Path(args.output)
    with (out / "history.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[pretrain_ssl] DONE → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
