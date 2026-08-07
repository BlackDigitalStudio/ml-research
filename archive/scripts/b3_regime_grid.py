#!/usr/bin/env python3
"""B3 — regime-conditional analysis on val_predictions.

True regime split (vol bucket, funding regime, time-of-day) requires the
original X_feat + timestamps which were not persisted alongside
val_predictions.npz. As a usable approximation we split the val set into
N equal-size CONTIGUOUS time chunks (val is already time-ordered) — this
captures any time-of-day / week-day cycles + intra-period volatility
shifts.

For each chunk we compute (a) primary's per-class precision and (b) the
profitability of the same grid the global C5 evaluation used. If a chunk
shows materially better/worse behaviour than the global average, that
indicates regime-dependent edge worth investigating with proper feature-
based bucketing later.
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

from src.models.meta_label import MetaConfig, build_meta_dataset, train_meta  # noqa: E402
from src.sizing import KellyConfig, size_trades_batch  # noqa: E402


GRID_TP = [0.10, 0.15, 0.20, 0.30]
GRID_SL = [0.05, 0.10, 0.15, 0.20]
GRID_KELLY = [0.10, 0.25, 0.50]
GRID_META = [0.50, 0.70, 0.85, 0.90, 0.95]
GRID_MIN_PROB = [0.50, 0.55, 0.60]

COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07
INITIAL_CAPITAL = 50.0
FLAT = 2


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
        commission_win_pct=COMMISSION_WIN, commission_loss_pct=COMMISSION_LOSS,
        cfg=cfg_k,
    )
    final_take = take_meta & sizing["take"]
    fractions = sizing["fraction"]
    weighted = np.where(final_take, fractions * pnl_val, 0.0)
    returns = weighted / 100.0
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + returns) if len(returns) else np.array([INITIAL_CAPITAL])
    final_eq = float(equity[-1])
    n_tr = int(final_take.sum())
    pnl_t = pnl_val[final_take]
    wr = float((pnl_t > 0).mean() * 100) if n_tr else 0.0
    return {
        "tp": tp, "sl": sl, "kelly": kelly_frac, "mthr": meta_thr, "mprob": min_prob,
        "n_trades": n_tr, "win_rate": wr,
        "net_return": 100 * (final_eq / INITIAL_CAPITAL - 1),
    }


def per_chunk_summary(stacker_soft, meta_model, y_val, pnl_val,
                       chunk_label: str):
    rows = [score_config(stacker_soft, meta_model, y_val, pnl_val, *c)
            for c in itertools.product(GRID_TP, GRID_SL, GRID_KELLY,
                                          GRID_META, GRID_MIN_PROB)]
    with_trades = [r for r in rows if r["n_trades"] >= 5]
    profitable = [r for r in with_trades if r["net_return"] > 0]
    if not with_trades:
        return {
            "chunk": chunk_label, "n_samples": int(len(y_val)),
            "configs_with_signal": 0, "profitable_configs": 0,
            "best_net": 0.0, "best_config": None,
        }
    best = max(with_trades, key=lambda r: r["net_return"])
    return {
        "chunk": chunk_label, "n_samples": int(len(y_val)),
        "configs_with_signal": len(with_trades),
        "profitable_configs": len(profitable),
        "best_net": best["net_return"], "best_n": best["n_trades"],
        "best_wr": best["win_rate"],
        "best_config": (best["tp"], best["sl"], best["mthr"], best["mprob"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions", required=True)
    ap.add_argument("--n-chunks", type=int, default=10)
    ap.add_argument("--stacker-key", default="stacker_soft_balanced",
                   choices=["stacker_soft", "stacker_soft_balanced"])
    ap.add_argument("--out", default="/home/scalper/backups/pod/recover_v2/b3_regime.json")
    args = ap.parse_args()

    data = np.load(args.val_predictions, allow_pickle=False)
    y_all = data["y_val"].astype(np.int64)
    pnl_all = data["pnl_val"].astype(np.float32)
    stacker_all = data[args.stacker_key].astype(np.float32)

    n = len(y_all)
    print(f"Loaded {n} val rows, splitting into {args.n_chunks} contiguous chunks")
    print(f"Using stacker_key={args.stacker_key}")

    # Train meta on first 75% of FULL val (same protocol as C5)
    n_meta_tr = int(n * 0.75)
    print(f"Training meta on first {n_meta_tr} samples ...")
    X_m, y_m, w_m = build_meta_dataset(
        primary_softmax=stacker_all[:n_meta_tr],
        y_true=y_all[:n_meta_tr], target_pnl=pnl_all[:n_meta_tr],
    )
    meta_model, meta_metrics = train_meta(X_m, y_m, w_m, cfg=MetaConfig(),
                                            val_frac=0.2, seed=42)
    print(f"meta OOS metrics: AUC={meta_metrics['val_auc']:.3f} "
          f"prec={meta_metrics['val_precision']:.3f} rec={meta_metrics['val_recall']:.3f}")

    # Evaluate per chunk on the LAST 25% only (held-out tail), split into N chunks
    tail_lo = n_meta_tr
    tail_hi = n
    tail_n = tail_hi - tail_lo
    chunk_size = tail_n // args.n_chunks
    print(f"\nPer-chunk evaluation on tail [{tail_lo}, {tail_hi}) "
          f"({tail_n} samples; chunk_size={chunk_size})")
    print(f"  {'chunk':>10s}  {'n':>6s}  {'sigCfg':>7s}  {'profit':>7s}  "
          f"{'bestNet':>10s}  {'bestN':>6s}  {'bestWR':>7s}")

    summaries = []
    for ci in range(args.n_chunks):
        lo = tail_lo + ci * chunk_size
        hi = tail_lo + (ci + 1) * chunk_size if ci < args.n_chunks - 1 else tail_hi
        s = per_chunk_summary(stacker_all[lo:hi], meta_model,
                               y_all[lo:hi], pnl_all[lo:hi],
                               f"chunk_{ci}_[{lo},{hi})")
        summaries.append(s)
        bc = s.get("best_config")
        bc_str = f"tp={bc[0]} sl={bc[1]} mthr={bc[2]} mp={bc[3]}" if bc else "—"
        print(f"  {f'chunk{ci}':>10s}  {s['n_samples']:>6d}  "
              f"{s['configs_with_signal']:>7d}  {s['profitable_configs']:>7d}  "
              f"{s['best_net']:>+10.2f}  {s.get('best_n', 0):>6d}  "
              f"{s.get('best_wr', 0):>7.2f}  ({bc_str})")

    # Whole-tail aggregate.
    s_all = per_chunk_summary(stacker_all[tail_lo:tail_hi], meta_model,
                                y_all[tail_lo:tail_hi], pnl_all[tail_lo:tail_hi],
                                "TAIL_ALL")
    print(f"  {'TAIL_ALL':>10s}  {s_all['n_samples']:>6d}  "
          f"{s_all['configs_with_signal']:>7d}  {s_all['profitable_configs']:>7d}  "
          f"{s_all['best_net']:>+10.2f}  {s_all.get('best_n', 0):>6d}  "
          f"{s_all.get('best_wr', 0):>7.2f}")
    summaries.append(s_all)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"meta_metrics": meta_metrics, "chunks": summaries,
                    "stacker_key": args.stacker_key}, f, indent=2)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
