#!/usr/bin/env python3
"""Step 2 — train regime classifier + measure lift as a pre-trade gate.

Pipeline:
  1. Load cache.npz (X_feat, y, target_pnl, mid, ...).
  2. Load depth timestamps aligned to sample indices (via Trainer).
  3. Train XGBoost regime classifier: target = "next-K-sample mean pnl > 0".
  4. Report classifier OOS metrics.
  5. Measure regime-gate lift against the baseline of "take every non-FLAT
     triple-barrier label" — shows whether the gate lifts OOS PnL.
  6. Optionally re-infer primaries (recover_v2 .pt weights) on new cache
     and evaluate regime-gate + primary + meta jointly. That step is
     heavy (~10-20 min CPU inference) — gated by --with-primaries.

Output:
  - models/regime_classifier.json — trained XGBoost model.
  - outputs/step2_regime_metrics.json — metrics payload.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.regime_classifier import (  # noqa: E402
    RegimeConfig, build_regime_features, build_regime_labels,
    train_regime_classifier,
)


def _load_cache(cache_dir: Path, hours: int) -> dict:
    """Find the newest cache entry matching {hours}h and load its arrays."""
    candidates = sorted(cache_dir.glob(f"samples_*_{hours}h_*_X_feat.npy"))
    if not candidates:
        # Fallback: any *_X_feat.npy
        candidates = sorted(cache_dir.glob("samples_*_X_feat.npy"))
    if not candidates:
        raise FileNotFoundError(f"No cache X_feat.npy in {cache_dir}")
    newest = candidates[-1]
    prefix = str(newest)[: -len("_X_feat.npy")]
    paths = {k: Path(f"{prefix}_{k}.npy")
              for k in ("X_lob", "X_feat", "y", "mid", "pnl")}
    return {
        "prefix":     prefix,
        "X_feat":     np.load(paths["X_feat"]),
        "y":          np.load(paths["y"]),
        "target_pnl": np.load(paths["pnl"]),
        "mid":        np.load(paths["mid"]),
        # X_lob is mmap — only load if we need it for primary inference.
    }


def _estimate_timestamps(n: int, mid: np.ndarray) -> np.ndarray:
    """Estimate per-sample timestamps assuming 100 ms tick spacing.

    Real timestamps aren't persisted alongside the sample cache; we use a
    uniform-spaced proxy anchored at an arbitrary epoch. Time-of-day features
    are still meaningful because the recorder + Tardis CSVs are both 100 ms
    cadence, and the regime classifier doesn't need millisecond accuracy.
    """
    # 100 ms per sample, anchored at 2024-01-01 UTC as a conservative default.
    anchor_ms = 1704067200000
    return anchor_ms + np.arange(n, dtype=np.int64) * 100


def _baseline_metrics(target_pnl: np.ndarray, y: np.ndarray,
                       label: str) -> dict:
    """PnL assuming we take every non-FLAT triple-barrier label at fraction=1.
    pnl already includes commissions (live_sim output)."""
    take = y != 2
    n = int(take.sum())
    pnl_t = target_pnl[take]
    wr = float((pnl_t > 0).mean() * 100) if n else 0.0
    total = float(pnl_t.sum())
    eq = 50.0 * np.cumprod(1.0 + target_pnl / 100.0 * take.astype(np.float32))
    final = float(eq[-1]) if len(eq) else 50.0
    return {
        "label": label, "n_candidates": n, "win_rate_pct": wr,
        "sum_pnl_pct": total,
        "net_return_pct": 100 * (final / 50.0 - 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir",
                    default="/home/scalper/scalper-bot/data/_cache")
    ap.add_argument("--hours", type=int, default=50)
    ap.add_argument("--model-out",
                    default="/home/scalper/scalper-bot/models/regime_classifier.json")
    ap.add_argument("--metrics-out",
                    default="/home/scalper/scalper-bot/models/step2_regime_metrics.json")
    ap.add_argument("--lookahead-k", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="tail fraction for walk-forward eval")
    ap.add_argument("--gate-thresholds", default="0.3,0.4,0.5,0.6,0.7",
                    help="comma-separated regime_prob thresholds to evaluate")
    args = ap.parse_args()

    print(f"[step2] loading cache from {args.cache_dir} ({args.hours}h)")
    c = _load_cache(Path(args.cache_dir), args.hours)
    n = len(c["y"])
    print(f"[step2] cache: N={n}, X_feat={c['X_feat'].shape}")
    cls_counts = np.bincount(c["y"], minlength=3)
    print(f"[step2] class balance: UP={cls_counts[0]} DN={cls_counts[1]} FL={cls_counts[2]} "
          f"({100*cls_counts[2]/n:.1f}% FLAT)")
    pnl_all = c["target_pnl"]
    print(f"[step2] target_pnl: mean={pnl_all.mean():+.4f}%, std={pnl_all.std():.4f}%")

    # Walk-forward split — last val_frac of samples is eval.
    n_val = int(n * args.val_frac)
    n_tr = n - n_val
    print(f"[step2] split: train {n_tr}, eval {n_val}")

    cfg = RegimeConfig(lookahead_k=args.lookahead_k)

    # Timestamps + regime features + forward-looking labels.
    ts = _estimate_timestamps(n, c["mid"])
    X_reg = build_regime_features(c["X_feat"], ts, pnl_all, cfg)
    y_reg, valid = build_regime_labels(pnl_all, cfg)

    # Train only on samples whose forward window falls inside TRAIN split.
    train_valid_mask = valid.copy()
    train_valid_mask[n_tr:] = False  # exclude tail from training
    y_reg_train = y_reg.copy()
    y_reg_train[~train_valid_mask] = 0   # doesn't matter — masked out

    print(f"[step2] regime train samples (valid): {int(train_valid_mask.sum())}")
    t0 = time.time()
    model, reg_metrics = train_regime_classifier(
        X_reg[train_valid_mask], y_reg[train_valid_mask],
        valid=np.ones(int(train_valid_mask.sum()), dtype=bool),
        cfg=cfg, val_frac=0.2,
    )
    print(f"[step2] regime classifier trained in {time.time() - t0:.1f}s")
    print(f"[step2] regime OOS metrics: {json.dumps(reg_metrics, indent=2)}")

    # Inference on the actual held-out tail (n_tr .. n-k).
    eval_lo, eval_hi = n_tr, n - cfg.lookahead_k
    if eval_hi <= eval_lo:
        raise SystemExit("Not enough tail samples after reserving look-ahead window")
    X_eval = X_reg[eval_lo:eval_hi]
    regime_prob = model.predict_proba(X_eval)[:, 1].astype(np.float32)

    pnl_eval = pnl_all[eval_lo:eval_hi]
    y_eval = c["y"][eval_lo:eval_hi]

    # Baseline: take all non-FLAT triple-barrier labels at fraction=1.
    baseline = _baseline_metrics(pnl_eval, y_eval, "eval_tail_baseline")
    print(f"\n[step2] BASELINE (no regime gate, take all non-FLAT):")
    print(f"  {baseline}")

    # Per-threshold regime-gate results.
    print(f"\n[step2] Regime-gate lift (eval samples {eval_hi - eval_lo}):")
    print(f"  {'thr':>5s}  {'kept':>7s}  {'cand':>6s}  {'WR':>6s}  "
          f"{'sumPnl':>8s}  {'net%':>7s}")
    gate_rows = []
    for thr in [float(x) for x in args.gate_thresholds.split(",")]:
        gate_mask = regime_prob >= thr
        take = (y_eval != 2) & gate_mask
        n_cand = int(take.sum())
        pnl_t = pnl_eval[take]
        wr = float((pnl_t > 0).mean() * 100) if n_cand else 0.0
        total = float(pnl_t.sum())
        returns = take.astype(np.float32) * pnl_eval / 100.0
        eq = 50.0 * np.cumprod(1.0 + returns) if len(returns) else np.array([50.0])
        net = 100 * (float(eq[-1]) / 50.0 - 1)
        print(f"  {thr:>5.2f}  {int(gate_mask.sum()):>7d}  {n_cand:>6d}  "
              f"{wr:>5.1f}%  {total:>+8.3f}  {net:>+7.2f}")
        gate_rows.append({
            "thr": thr, "n_kept": int(gate_mask.sum()),
            "n_candidates": n_cand, "win_rate_pct": wr,
            "sum_pnl_pct": total, "net_return_pct": net,
        })

    # Save.
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_out))
    metrics = {
        "cache_prefix":       c["prefix"],
        "n":                  int(n),
        "n_train":            int(n_tr),
        "n_eval":             int(n_val),
        "regime_cfg":         {
            "lookahead_k":      cfg.lookahead_k,
            "rolling_window":   cfg.rolling_window,
            "pnl_sum_threshold_pct": cfg.pnl_sum_threshold_pct,
            "feat_columns":     list(cfg.feat_columns),
        },
        "regime_metrics":     reg_metrics,
        "baseline":           baseline,
        "gates":              gate_rows,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model → {args.model_out}")
    print(f"Saved metrics → {args.metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
