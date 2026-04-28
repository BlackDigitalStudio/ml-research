#!/usr/bin/env python3
"""Optuna search XGBoost stacker hparams maximizing prec_nonflat × n.

Reads pre-built X_stack from holdout_X_stack.npy + y_w from cache_volaware.
Splits 75/25 walk-forward.
Search: depth, eta, n_estimators, subsample, colsample, min_child_weight, gamma.

Score = (prec_nonflat * sqrt(n_nonflat / n_val)) — поощряет precision but punishes silence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UP, DN, FL = 0, 1, 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="Local OOF cache directory (e.g. data/_oof_cache_combined)")
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--cv-holdout-frac", type=float, default=0.20)
    ap.add_argument("--cv-n-groups", type=int, default=6)
    ap.add_argument("--cv-k-test", type=int, default=1)
    ap.add_argument("--cv-purge-indices", type=int, default=2000)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--out", default="runs/optuna_stacker.json")
    args = ap.parse_args()

    from src.cv import CPCV

    p = args.cache_prefix
    print(f"[opt] loading cache {p}", flush=True)
    y = np.load(f"{p}_y.npy")
    n_total = len(y)
    n_holdout = int(n_total * args.cv_holdout_frac)
    n_working = n_total - n_holdout
    y_w = y[:n_working]

    sample_pseudo_ts = np.arange(n_working).astype(np.int64)
    cpcv = CPCV(n_groups=args.cv_n_groups, k_test=args.cv_k_test,
                sample_ts=sample_pseudo_ts,
                label_horizon_ms=args.cv_purge_indices,
                embargo_pct=0.005)
    fold_splits = list(cpcv.split(n_working))

    cache_dir = Path(args.cache_dir)
    arch_oof = {}
    for arch in args.archs:
        oof = np.zeros((n_working, 3), dtype=np.float32)
        mask = np.zeros(n_working, dtype=bool)
        for f in range(len(fold_splits)):
            local = cache_dir / arch / f"fold_{f}" / "softmax.npy"
            if not local.exists():
                continue
            soft = np.load(local)
            _, test_idx = fold_splits[f]
            if soft.shape[0] != len(test_idx):
                continue
            oof[test_idx] = soft
            mask[test_idx] = True
        cov = mask.mean()
        if cov < 0.5:
            print(f"[opt] skip {arch}: cov={cov:.2f}", flush=True)
            continue
        arch_oof[arch] = oof
    arch_names = sorted(arch_oof.keys())
    print(f"[opt] using {len(arch_names)} archs: {arch_names}", flush=True)

    X_stack = np.concatenate([arch_oof[a] for a in arch_names], axis=1)
    print(f"[opt] X_stack {X_stack.shape}", flush=True)

    n_train = int(n_working * 0.75)
    Xs_tr, ys_tr = X_stack[:n_train], y_w[:n_train]
    Xs_va, ys_va = X_stack[n_train:], y_w[n_train:]

    cls_freq = np.array([(ys_tr == c).sum() for c in (UP, DN, FL)],
                        dtype=np.float64)
    cls_w = 1.0 / np.sqrt(np.maximum(cls_freq, 1.0))
    cls_w = cls_w / cls_w.mean()
    sample_w = cls_w[ys_tr]

    n_val = len(ys_va)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        clf = xgb.XGBClassifier(
            **params, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", early_stopping_rounds=20,
            n_jobs=8, verbosity=0,
        )
        clf.fit(Xs_tr, ys_tr, sample_weight=sample_w,
                eval_set=[(Xs_va, ys_va)], verbose=False)
        pred = clf.predict(Xs_va)
        nf = pred != FL
        n_nf = int(nf.sum())
        if n_nf < 50:
            return -1.0  # punish trivial all-FLAT
        prec_nf = float(((pred == ys_va) & nf).sum() / n_nf)
        # Score: prec * sqrt(coverage)
        cov = n_nf / n_val
        return prec_nf * np.sqrt(cov)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False,
                   n_jobs=4)

    print(f"\n[opt] best trial: {study.best_value:.4f}")
    print(f"[opt] best params: {study.best_params}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "best_score": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "archs": arch_names,
        "trials": [{"value": t.value, "params": t.params}
                   for t in study.trials[:30]],
    }, indent=2, default=float))
    print(f"[opt] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
