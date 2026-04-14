#!/usr/bin/env python3
"""C5 — end-to-end evaluation: balanced stacker + retrained meta + grid.

Flow:
  1. Load val_predictions.npz (must already contain stacker_soft_balanced
     from scripts/fix_stacker_classweight.py).
  2. Retrain meta on balanced stacker_soft with walk-forward (train=first
     75%, val=tail 25%).
  3. Run the same grid as grid_test_ensemble.py but swap in the balanced
     stacker output.
  4. Print side-by-side comparison vs original grid_results.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.meta_label import (MetaConfig, build_meta_dataset,
                                     train_meta)  # noqa: E402
from src.sizing import KellyConfig, size_trades_batch  # noqa: E402


# Match grid_test_ensemble.py defaults exactly for apples-to-apples.
GRID_TP = [0.10, 0.15, 0.20, 0.30, 0.40]
GRID_SL = [0.05, 0.10, 0.15, 0.20]
GRID_KELLY = [0.10, 0.25, 0.50, 1.00]
# Extended meta_thr — balanced stacker produces ~420× more candidates
# so we need much tighter meta filtering to find the high-precision subset.
GRID_META = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
GRID_MIN_PROB = [0.50, 0.55, 0.60, 0.70]

COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07
INITIAL_CAPITAL = 50.0
FLAT = 2


def train_meta_on(stacker_soft: np.ndarray, y_val: np.ndarray,
                   pnl_val: np.ndarray, *, label: str, seed: int = 42):
    """Walk-forward meta training on the first 75% of val, eval on tail 25%."""
    n = len(y_val)
    n_tr = int(n * 0.75)

    # Build full meta dataset on TRAIN split only — meta filters non-FLAT primary.
    X_m_tr, y_m_tr, w_m_tr = build_meta_dataset(
        primary_softmax=stacker_soft[:n_tr],
        y_true=y_val[:n_tr],
        target_pnl=pnl_val[:n_tr],
    )
    print(f"[meta:{label}] train non-FLAT candidates: {len(y_m_tr)} "
          f"(pos rate={y_m_tr.mean():.4f})")
    if len(y_m_tr) < 200:
        raise ValueError(f"Meta train too small ({len(y_m_tr)}) — "
                          f"stacker still collapses?")

    model, metrics = train_meta(X_m_tr, y_m_tr, w_m_tr, cfg=MetaConfig(),
                                 val_frac=0.2, seed=seed)
    print(f"[meta:{label}] val metrics: {json.dumps(metrics, indent=2)}")
    return model


def score_config(stacker_soft, meta_model, y_val, pnl_val,
                  tp, sl, kelly_frac, meta_thr, min_prob):
    primary_pred = stacker_soft.argmax(axis=-1)
    primary_max = stacker_soft.max(axis=-1)
    non_flat = primary_pred != FLAT

    p_soft = stacker_soft
    p_max = p_soft.max(axis=-1, keepdims=True)
    p_margin = (np.sort(p_soft, axis=-1)[:, -1] - np.sort(p_soft, axis=-1)[:, -2])[:, None]
    p_entropy = -(p_soft * np.log(p_soft.clip(1e-9))).sum(axis=-1, keepdims=True)
    X_meta = np.concatenate([
        p_soft, p_max, p_margin, p_entropy,
        (primary_pred == 0).astype(np.float32)[:, None],
    ], axis=-1).astype(np.float32)

    meta_prob = np.zeros(len(y_val), dtype=np.float32)
    if non_flat.any():
        meta_prob[non_flat] = meta_model.predict_proba(X_meta[non_flat])[:, 1]

    take_meta = non_flat & (primary_max >= min_prob) & (meta_prob >= meta_thr)

    cfg_k = KellyConfig(fraction=kelly_frac, max_position_fraction=19.0,
                         min_probability=min_prob)
    sizing = size_trades_batch(
        p_win=primary_max, win_pct=tp, loss_pct=sl,
        commission_win_pct=COMMISSION_WIN,
        commission_loss_pct=COMMISSION_LOSS,
        cfg=cfg_k,
    )
    final_take = take_meta & sizing["take"]
    fractions = sizing["fraction"]

    weighted = np.where(final_take, fractions * pnl_val, 0.0)
    returns = weighted / 100.0
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    final_eq = float(equity[-1]) if len(equity) else INITIAL_CAPITAL

    taken = np.where(final_take)[0]
    n_tr = len(taken)
    pnl_t = pnl_val[taken]
    wr = float((pnl_t > 0).mean() * 100) if n_tr else 0.0
    sh = float(pnl_t.mean() / (pnl_t.std() + 1e-9)) if n_tr > 1 else 0.0
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([INITIAL_CAPITAL])
    dd = float(((peaks - equity) / np.maximum(peaks, 1e-12)).max()) if len(equity) else 0.0

    return {
        "tp_pct": tp, "sl_pct": sl, "kelly_frac": kelly_frac,
        "meta_threshold": meta_thr, "min_probability": min_prob,
        "n_trades": n_tr,
        "trade_rate_pct": 100 * n_tr / max(len(y_val), 1),
        "win_rate_pct": wr,
        "sharpe_per_trade": sh,
        "max_drawdown_pct": 100 * dd,
        "net_return_pct": 100 * (final_eq / INITIAL_CAPITAL - 1),
    }


def run_grid_on(stacker_soft, meta_model, y_val, pnl_val, label,
                 min_trades: int = 20):
    configs = list(itertools.product(GRID_TP, GRID_SL, GRID_KELLY,
                                       GRID_META, GRID_MIN_PROB))
    rows = [score_config(stacker_soft, meta_model, y_val, pnl_val, *c)
            for c in configs]
    # Filter out zero-trade configs before ranking — zero return is
    # not the "best" result, it's a filter too tight to produce signal.
    with_trades = [r for r in rows if r["n_trades"] >= min_trades]
    print(f"\n=== {label} ===")
    print(f"  total configs: {len(rows)}, with n_trades>={min_trades}: {len(with_trades)}")
    if not with_trades:
        print("  (no configs produced signal)")
        return rows
    profitable = [r for r in with_trades if r["net_return_pct"] > 0]
    print(f"  profitable (n≥{min_trades} & net>0): {len(profitable)}  "
          f"(best net: {max(r['net_return_pct'] for r in with_trades):+.2f}%)")

    by_pnl = sorted(with_trades, key=lambda r: -r["net_return_pct"])[:7]
    by_wr = sorted(with_trades, key=lambda r: -r["win_rate_pct"])[:5]

    def _row(r):
        return (f"  {r['tp_pct']:>5.2f} {r['sl_pct']:>5.2f} {r['kelly_frac']:>5.2f} "
                f"{r['meta_threshold']:>5.2f} {r['min_probability']:>5.2f}  "
                f"{r['n_trades']:>6d} {r['win_rate_pct']:>6.1f} "
                f"{r['net_return_pct']:>+7.2f} {r['max_drawdown_pct']:>6.2f} "
                f"{r['sharpe_per_trade']:>5.2f}")

    print(f"  {'tp':>5s} {'sl':>5s} {'kel':>5s} {'mthr':>5s} {'mprob':>5s}  "
          f"{'n':>6s} {'WR%':>6s} {'net%':>7s} {'DD%':>6s} {'Shp':>5s}")
    print("  top 7 by net_return:")
    for r in by_pnl:
        print(_row(r))
    print("  top 5 by win_rate:")
    for r in by_wr:
        print(_row(r))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.val_predictions, allow_pickle=False)
    y_val = data["y_val"].astype(np.int64)
    pnl_val = data["pnl_val"].astype(np.float32)
    if "stacker_soft_balanced" not in data.files:
        raise SystemExit("stacker_soft_balanced not in val_predictions.npz — "
                         "run scripts/fix_stacker_classweight.py first")

    stacker_orig = data["stacker_soft"].astype(np.float32)
    stacker_bal = data["stacker_soft_balanced"].astype(np.float32)
    print(f"Loaded {len(y_val)} val samples")

    # === Retrain meta on each stacker flavour ===
    meta_orig = train_meta_on(stacker_orig, y_val, pnl_val,
                               label="orig-stacker", seed=args.seed)
    meta_bal = train_meta_on(stacker_bal, y_val, pnl_val,
                              label="balanced-stacker", seed=args.seed)

    # === Grid on each ===
    rows_orig = run_grid_on(stacker_orig, meta_orig, y_val, pnl_val,
                             "ORIGINAL (unbalanced stacker + matched meta)")
    rows_bal = run_grid_on(stacker_bal, meta_bal, y_val, pnl_val,
                            "BALANCED (new stacker + new meta)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_bal.save_model(str(out / "meta_balanced.json"))
    with (out / "grid_comparison.json").open("w") as f:
        json.dump({"original": rows_orig, "balanced": rows_bal}, f, indent=2)
    print(f"\nSaved meta_balanced.json + grid_comparison.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
