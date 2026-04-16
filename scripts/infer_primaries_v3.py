#!/usr/bin/env python3
"""Re-infer primary softmaxes on the v3 cache using saved .pt checkpoints.

Produces `primary_softs_v3.npz` with keys `soft_<arch>` plus `y` and
`pnl_val`. The file is the input to `grid_live.py` — separating inference
from grid means we can sweep thousands of strategy configs without paying
the forward-pass cost each time.

Checkpoints expected in `--weights-dir` (default recover_v2/):
  transformer.pt, tcn.pt, and optionally chronos_bolt_{tiny,mini,small}.pt.

We skip any weight file whose state-dict doesn't match the current factory
constructor (e.g. num_feat mismatch, renamed layers). Skipped archs get
logged; the script continues on whatever survives — ≥2 archs give a
meaningful stacker, 1 gives a single-primary grid.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bakeoff_v1 import build_factory  # noqa: E402


def _load_v3(cache_dir: Path) -> dict:
    cand = sorted(cache_dir.glob("samples_v3_*_y.npy"))
    if not cand:
        raise FileNotFoundError(f"No v3 cache in {cache_dir}")
    prefix = str(cand[-1])[: -len("_y.npy")]
    return {
        "prefix": prefix,
        "X_lob":  np.load(f"{prefix}_X_lob.npy", mmap_mode="r"),
        "X_feat": np.load(f"{prefix}_X_feat.npy"),
        "y":      np.load(f"{prefix}_y.npy"),
        "pnl":    np.load(f"{prefix}_pnl.npy"),
    }


def _infer(model, X_lob, X_feat, batch_size: int = 512,
            device: str = "cpu") -> np.ndarray:
    model = model.to(device).eval()
    n = len(X_feat)
    out = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            lob = np.array(X_lob[i:i + batch_size], dtype=np.float32)
            feat = np.array(X_feat[i:i + batch_size], dtype=np.float32)
            lob = np.nan_to_num(lob, nan=0.0, posinf=0.0, neginf=0.0)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            xb = torch.from_numpy(lob).to(device)
            fb = torch.from_numpy(feat).to(device)
            logits, _ = model(xb, fb)
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/scalper/scalper-bot/data/_cache")
    ap.add_argument("--weights-dir",
                    default="/home/scalper/backups/pod/recover_v2")
    ap.add_argument("--archs", default="transformer,tcn",
                    help="comma-separated archs to try")
    ap.add_argument("--out",
                    default="/home/scalper/scalper-bot/models/primary_softs_v3.npz")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"[infer] loading v3 cache from {args.cache_dir}")
    c = _load_v3(Path(args.cache_dir))
    n_feat = c["X_feat"].shape[1]
    print(f"[infer] N={len(c['y'])}, X_lob={c['X_lob'].shape}, "
          f"X_feat={c['X_feat'].shape}")

    wdir = Path(args.weights_dir)
    results = {"y": c["y"], "pnl_val": c["pnl"]}
    for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
        # bakeoff_v3 writes `{tag}_best.pt`; fall back to legacy `{tag}.pt`
        # (written by bakeoff_v1/v2) to keep older weight dirs loadable.
        pt = wdir / f"{arch}_best.pt"
        if not pt.exists():
            pt = wdir / f"{arch}.pt"
        if not pt.exists():
            print(f"[infer] skip {arch}: no {pt}")
            continue
        try:
            factory, tag = build_factory(arch)
        except ValueError as e:
            print(f"[infer] skip {arch}: {e}")
            continue
        try:
            model = factory(n_feat)
            ckpt = torch.load(pt, map_location="cpu", weights_only=True)
            # bakeoff_v3 wraps state_dict in {"state_dict": ..., "metrics": ...};
            # legacy bakeoff_v1/v2 saved the state_dict directly.
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            # Some archs saved via PEFT add `base_model.model.` prefix; also
            # some training runs leave `_orig_mod.` from torch.compile. Strip
            # known prefixes so they load into the fresh factory model.
            def _strip(k):
                for p in ("_orig_mod.", "module."):
                    if k.startswith(p):
                        k = k[len(p):]
                return k
            sd = {_strip(k): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[infer] skip {arch}: load failed — {e}")
            continue
        t0 = time.time()
        soft = _infer(model, c["X_lob"], c["X_feat"],
                       batch_size=args.batch_size, device=args.device)
        dt = time.time() - t0
        pred = soft.argmax(-1)
        non_fl = pred != 2
        # Directional accuracy only on samples where y is non-FLAT.
        y_nf = c["y"] != 2
        dir_correct = ((pred == c["y"]) & non_fl & y_nf).sum()
        dir_total = (non_fl & y_nf).sum()
        wrong_dir = ((pred != c["y"]) & non_fl & y_nf).sum()
        print(f"[infer] {tag}: softmax {soft.shape} "
              f"non_fl_pred={int(non_fl.sum())} "
              f"dir_acc_on_nf={dir_correct / max(dir_total, 1):.3f} "
              f"({dir_correct}/{dir_total}, wrong={wrong_dir}) "
              f"time={dt:.1f}s")
        results[f"soft_{tag}"] = soft.astype(np.float32)

    if len([k for k in results if k.startswith("soft_")]) == 0:
        print("[infer] no primaries inferred — abort")
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **results)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
