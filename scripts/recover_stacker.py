#!/usr/bin/env python3
"""Recover stacker + meta from saved .pt weights — bake-off-crash fallback.

If the bake-off crashed mid-run (e.g. a chronos/mamba import died and took
out the Python process before val_predictions.npz was written), this
script reconstructs the ensemble from whatever .pt weights survived.

For each *.pt file in --weights-dir: rebuild the arch via build_factory,
load weights, run val inference on the same split train_generic uses,
collect the val_softmax. Then fit XGBoost stacker + meta-labeler on the
collected softmaxes, exactly as bakeoff_v2.py would.

Usage:
    python scripts/recover_stacker.py \\
        --cache /workspace/scalper-bot/data/_cache/cache.npz \\
        --weights-dir /workspace/scalper-bot/runs \\
        --out /workspace/scalper-bot/runs/bakeoff_merged
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
from src.models.stacking import train_stacker, predict_stacked  # noqa: E402
from src.models.meta_label import build_meta_dataset, train_meta, MetaConfig  # noqa: E402


# Must match train_generic defaults
GAP = 650
VAL_FRAC = 0.2


def _val_split(X_lob, X_feat, y, target_pnl):
    """Mirror train_generic's split: last val_frac after gap."""
    n = len(y)
    n_val = int(n * VAL_FRAC)
    n_train = n - n_val - GAP
    lo = n_train + GAP
    return (
        X_lob[lo:], X_feat[lo:],
        y[lo:], target_pnl[lo:],
        n_train,
    )


def _pt_to_arch(path: Path) -> str:
    """Extract arch name from e.g. 'transformer.pt' → 'transformer'."""
    return path.stem


def _infer_val_softmax(model, X_lob_val, X_feat_val, device, batch_size=512):
    """Run val inference and return softmax (N, 3)."""
    model = model.to(device).eval()
    n = len(X_feat_val)
    softs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            lob_batch = np.array(X_lob_val[i:i+batch_size], dtype=np.float32)
            feat_batch = np.array(X_feat_val[i:i+batch_size], dtype=np.float32)
            # nan safety consistent with LOBDataset
            lob_batch = np.nan_to_num(lob_batch, nan=0.0, posinf=0.0, neginf=0.0)
            feat_batch = np.nan_to_num(feat_batch, nan=0.0, posinf=0.0, neginf=0.0)
            xb = torch.from_numpy(lob_batch).to(device)
            fb = torch.from_numpy(feat_batch).to(device)
            logits, _ = model(xb, fb)
            softs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(softs, axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--weights-dir", required=True,
                   help="dir (or parent of dirs) containing *.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[recover] loading cache {args.cache}")
    data = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    X_lob = data["X_lob"]
    X_feat = data["X_feat"]
    y = data["y"]
    target_pnl = data["target_pnl"]

    X_lob_val, X_feat_val, y_val, pnl_val, _ = _val_split(X_lob, X_feat, y, target_pnl)
    # Materialize into RAM for quick multi-model inference
    X_lob_val = np.ascontiguousarray(X_lob_val, dtype=np.float32)
    X_feat_val = np.ascontiguousarray(X_feat_val, dtype=np.float32)
    print(f"[recover] val split: {len(y_val)} samples "
          f"(y UP={int((y_val==0).sum())} DN={int((y_val==1).sum())} FL={int((y_val==2).sum())})")

    # Discover *.pt files (recursive one level)
    wroot = Path(args.weights_dir)
    pt_files = []
    if wroot.is_file() and wroot.suffix == ".pt":
        pt_files = [wroot]
    else:
        for sub in [wroot] + list(wroot.iterdir() if wroot.is_dir() else []):
            if sub.is_dir():
                pt_files.extend(sub.glob("*.pt"))
            elif sub.suffix == ".pt":
                pt_files.append(sub)
    pt_files = sorted(set(pt_files))
    print(f"[recover] found {len(pt_files)} weight files: {[str(p) for p in pt_files]}")

    if not pt_files:
        print("[recover] no .pt files, nothing to do")
        return 1

    primary_softs = []
    primary_tags = []
    perf_rows = []
    for pt in pt_files:
        arch = _pt_to_arch(pt)
        try:
            factory, tag = build_factory(arch)
        except ValueError as e:
            print(f"[recover] skip {pt.name}: {e}")
            continue
        t0 = time.time()
        model = factory(X_feat_val.shape[1])
        model.load_state_dict(torch.load(pt, map_location="cpu"))
        softs = _infer_val_softmax(model, X_lob_val, X_feat_val, args.device)
        dt = time.time() - t0
        pred = softs.argmax(-1)
        non_fl = pred != 2
        acc = float((pred == y_val).mean())
        prec_nonfl = float(((pred == y_val) & non_fl).sum() / max(non_fl.sum(), 1))
        print(f"[recover] {tag}: softmax {softs.shape} acc={acc:.3f} "
              f"prec_nonFL={prec_nonfl:.3f} time={dt:.1f}s")
        primary_softs.append(softs)
        primary_tags.append(tag)
        perf_rows.append({"arch": tag, "acc": acc, "prec_nonfl": prec_nonfl,
                          "n_nonfl_pred": int(non_fl.sum())})

    if len(primary_softs) < 2:
        print("[recover] need ≥2 archs for stacker, got "
              f"{len(primary_softs)}. Saving soft_* only.")
        np.savez(out_dir / "val_predictions.npz",
                 y_val=y_val, pnl_val=pnl_val,
                 **{f"soft_{t}": s for t, s in zip(primary_tags, primary_softs)})
        return 0

    print(f"\n[recover] training stacker on {len(primary_softs)} L1 softmaxes...")
    stacker, stk = train_stacker(primary_softs, y_val, X_feat=None)
    stacker.save_model(str(out_dir / "stacker.json"))
    print(f"[recover] stacker val_acc={stk['val_acc']:.4f} "
          f"bal_acc={stk['val_bal_acc']:.4f} logloss={stk['val_logloss']:.4f}")

    stacker_soft = predict_stacked(stacker, primary_softs, X_feat=None, use_feats=False)

    print(f"\n[recover] training meta-labeler...")
    X_m, y_m, w_m = build_meta_dataset(
        stacker_soft, y_val, pnl_val,
        cfg=MetaConfig(pnl_threshold_pct=0.0),
    )
    meta, meta_m = train_meta(X_m, y_m, w_m, cfg=MetaConfig())
    meta.save_model(str(out_dir / "meta.json"))
    print(f"[recover] meta val_auc={meta_m['val_auc']:.4f} "
          f"prec={meta_m['val_precision']:.4f} rec={meta_m['val_recall']:.4f}")

    # Final output — mirrors bakeoff_v2.py's val_predictions.npz format
    np.savez(
        out_dir / "val_predictions.npz",
        y_val=y_val, pnl_val=pnl_val, stacker_soft=stacker_soft,
        **{f"soft_{t}": s for t, s in zip(primary_tags, primary_softs)},
    )

    with (out_dir / "leaderboard.json").open("w") as f:
        json.dump({
            "source": "recover_stacker",
            "primaries": perf_rows,
            "stacker": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in stk.items() if k != "per_class_recall"},
            "meta": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                     for k, v in meta_m.items()},
        }, f, indent=2)

    print(f"\n[recover] DONE. Outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
