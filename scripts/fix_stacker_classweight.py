#!/usr/bin/env python3
"""C5 — retrain L2 stacker with class-balanced sample weights.

The current stacker (recover_v2/stacker.json) collapsed to near-always-FL
(bal_acc ≈ 0.33). Root cause: XGBoost trained without compensating for
the ~85% FL class imbalance. Fixing this should expose more non-FL
candidates at similar or better precision — more meta-rescuable signal,
more grid-search trades, tighter CI on WR.

This script reuses val_predictions.npz from recover_v2 (5 primary
softmaxes + y_val), retrains the L2 stacker with
``sample_weight = class_count_inv[y]`` normalised to mean 1, saves the
new model, and reports a before/after comparison.

Usage:
    python3 scripts/fix_stacker_classweight.py \
        --val-predictions /home/scalper/backups/pod/recover_v2/val_predictions.npz \
        --out-dir        /home/scalper/backups/pod/recover_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.stacking import StackerConfig, stack_inputs  # noqa: E402


UP, DOWN, FLAT = 0, 1, 2


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """1/class_frequency, normalised so the mean weight is 1."""
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    # Map class → weight inversely proportional to frequency.
    inv = {int(c): 1.0 / f for c, f in zip(classes, freq)}
    w = np.array([inv[int(yi)] for yi in y], dtype=np.float32)
    return w / w.mean()


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, soft: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  f1_score, precision_score, recall_score,
                                  log_loss)
    out = {
        "acc":         float(accuracy_score(y_true, y_pred)),
        "bal_acc":     float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro")),
        "logloss":     float(log_loss(y_true, soft, labels=[0, 1, 2])),
    }
    for c, name in [(UP, "UP"), (DOWN, "DOWN"), (FLAT, "FLAT")]:
        out[f"precision_{name}"] = float(precision_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
        out[f"recall_{name}"]    = float(recall_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
        out[f"count_pred_{name}"] = int((y_pred == c).sum())
        out[f"count_true_{name}"] = int((y_true == c).sum())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-predictions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.25,
                   help="Fraction of val_predictions used as stacker's hold-out tail.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = np.load(args.val_predictions, allow_pickle=False)
    y_val = data["y_val"].astype(np.int64)
    primary_keys = sorted(k for k in data.keys() if k.startswith("soft_"))
    primary_softs = [data[k] for k in primary_keys]
    print(f"Loaded {len(y_val)} samples, {len(primary_keys)} primaries: {primary_keys}")

    # Class distribution.
    uniq, cnt = np.unique(y_val, return_counts=True)
    print("Class distribution:", dict(zip(uniq.tolist(), (cnt / cnt.sum()).tolist())))

    # Build stacker inputs (same as stacking.train_stacker does internally).
    X_stack = stack_inputs(primary_softs, X_feat=None)  # X_feat=None — val_predictions.npz
                                                         # does not store it separately.
    n = len(y_val)
    n_val = int(n * args.val_frac)
    n_tr = n - n_val

    Xtr, Xv = X_stack[:n_tr], X_stack[n_tr:]
    ytr, yv = y_val[:n_tr], y_val[n_tr:]
    w_tr = balanced_sample_weight(ytr)
    print(f"Train: {n_tr}, Val(tail): {n_val}; sample_weight mean={w_tr.mean():.3f}, "
          f"max={w_tr.max():.3f}")

    import xgboost as xgb
    cfg = StackerConfig()

    def _train(sample_weight):
        m = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate, subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha, reg_lambda=cfg.reg_lambda,
            objective="multi:softprob", num_class=3,
            early_stopping_rounds=40,
            random_state=args.seed, n_jobs=-1, verbosity=0,
        )
        m.fit(Xtr, ytr, sample_weight=sample_weight,
              eval_set=[(Xv, yv)], verbose=False)
        return m

    print("\n>> Training balanced stacker ...")
    m_bal = _train(w_tr)
    soft_bal = m_bal.predict_proba(Xv)
    pred_bal = soft_bal.argmax(axis=-1)
    met_bal = per_class_metrics(yv, pred_bal, soft_bal)

    print("\n>> Training un-balanced reference (for direct comparison) ...")
    m_unb = _train(None)
    soft_unb = m_unb.predict_proba(Xv)
    pred_unb = soft_unb.argmax(axis=-1)
    met_unb = per_class_metrics(yv, pred_unb, soft_unb)

    print("\n=== Stacker comparison (tail val) ===")
    cols = ["acc", "bal_acc", "f1_macro", "logloss",
            "precision_UP", "precision_DOWN", "precision_FLAT",
            "recall_UP", "recall_DOWN", "recall_FLAT",
            "count_pred_UP", "count_pred_DOWN", "count_pred_FLAT"]
    print(f"  {'metric':20s}  {'unbalanced':>12s}  {'balanced':>12s}")
    for c in cols:
        u = met_unb[c]; b = met_bal[c]
        fmt = "{:12d}" if isinstance(u, int) else "{:12.4f}"
        print(f"  {c:20s}  " + fmt.format(u) + "  " + fmt.format(b))
    print(f"  {'true_UP':20s}  {met_unb['count_true_UP']:12d}  {met_bal['count_true_UP']:12d}")
    print(f"  {'true_DOWN':20s}  {met_unb['count_true_DOWN']:12d}  {met_bal['count_true_DOWN']:12d}")
    print(f"  {'true_FLAT':20s}  {met_unb['count_true_FLAT']:12d}  {met_bal['count_true_FLAT']:12d}")

    # Persist balanced stacker + a full-dataset balanced stacker_soft for
    # downstream grid-search / meta retraining.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n>> Retraining balanced stacker on ALL val data (for downstream) ...")
    w_all = balanced_sample_weight(y_val)
    m_full = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate, subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha, reg_lambda=cfg.reg_lambda,
        objective="multi:softprob", num_class=3,
        random_state=args.seed, n_jobs=-1, verbosity=0,
    )
    # n_estimators pinned (no early-stop because no held-out) — keep cfg value.
    m_full.fit(X_stack, y_val, sample_weight=w_all, verbose=False)
    m_full.save_model(str(out / "stacker_balanced.json"))

    # Recompute stacker_soft_balanced over FULL val set.
    stacker_soft_balanced = m_full.predict_proba(X_stack).astype(np.float32)

    # Save alongside originals. Preserve every existing array.
    payload = {k: data[k] for k in data.keys()}
    payload["stacker_soft_balanced"] = stacker_soft_balanced
    out_npz = out / "val_predictions.npz"
    np.savez_compressed(out_npz, **payload)

    metrics_path = out / "stacker_balanced_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"balanced_taileval": met_bal,
                   "unbalanced_taileval": met_unb,
                   "val_frac": args.val_frac,
                   "seed": args.seed,
                   "n_samples": n}, f, indent=2)
    print(f"\nSaved:")
    print(f"  {out / 'stacker_balanced.json'}")
    print(f"  {metrics_path}")
    print(f"  {out_npz} (added 'stacker_soft_balanced' key)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
