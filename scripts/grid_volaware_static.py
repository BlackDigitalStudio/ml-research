#!/usr/bin/env python3
"""Grid sweep on vol-aware TP/SL (per-sample fixed) + variable timeout/conf/meta/scale.

Использует existing stacker_v3 + meta_v3 + holdout_X_stack.npy для direction.
TP/SL берутся из cache (per-sample, vol-derived) с optional scale factor.

Sweep: scale × timeout × conf_thr × meta_thr.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402

UP, DN, FL = 0, 1, 2


def realise(direction, pnl_long, pnl_short):
    return np.where(direction == +1, pnl_long,
             np.where(direction == -1, pnl_short, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--volaware-prefix", required=True,
                    help="Prefix к volaware-cache (с _tp_pct.npy, _sl_pct.npy)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--summary", default="models/stacker_summary.json")
    ap.add_argument("--cv-holdout-frac", type=float, default=0.20)
    ap.add_argument("--out", default="runs/grid_volaware_static.json")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = args.cache_prefix
    print(f"[gva] loading cache {p}", flush=True)
    y = np.load(f"{p}_y.npy")
    mid_paths = np.load(f"{p}_mid_paths.npy", mmap_mode="r")
    entry_long = np.load(f"{p}_entry_long.npy")
    entry_short = np.load(f"{p}_entry_short.npy")
    n_total = len(y)
    n_holdout = int(n_total * args.cv_holdout_frac)
    n_working = n_total - n_holdout

    vp = args.volaware_prefix
    tp_pct_all = np.load(f"{vp}_tp_pct.npy")
    sl_pct_all = np.load(f"{vp}_sl_pct.npy")
    print(f"[gva] vol-aware TP%: median={np.median(tp_pct_all):.3f} "
          f"p25={np.percentile(tp_pct_all, 25):.3f} "
          f"p75={np.percentile(tp_pct_all, 75):.3f}", flush=True)

    # Load stacker + meta
    stacker = xgb.XGBClassifier()
    stacker.load_model(str(Path(args.models_dir) / "stacker_v3.json"))
    meta = joblib.load(Path(args.models_dir) / "meta_v3.pkl")
    H = np.load(Path(args.models_dir) / "holdout_X_stack.npy")
    summary = json.loads(Path(args.summary).read_text())
    arch_names = summary["archs_used"]
    print(f"[gva] {len(arch_names)} archs, H {H.shape}", flush=True)

    proba = stacker.predict_proba(H)
    pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    direction = np.where(pred == UP, +1,
                  np.where(pred == DN, -1, 0)).astype(np.int8)
    n_h = len(direction)
    print(f"[gva] L2 predictions: UP={int((pred==UP).sum())} "
          f"DN={int((pred==DN).sum())} FL={int((pred==FL).sum())}", flush=True)

    max_prob = proba.max(axis=1, keepdims=True)
    entropy = (-proba * np.log(proba + 1e-12)).sum(axis=1, keepdims=True)
    X_meta = np.hstack([proba, max_prob, entropy])
    meta_proba_all = np.zeros(n_h, dtype=np.float32)
    nf = pred != FL
    if nf.any():
        meta_proba_all[nf] = meta.predict_proba(X_meta[nf])[:, 1]

    holdout_mid_paths = np.array(mid_paths[n_working:])
    holdout_entry_long = entry_long[n_working:]
    holdout_entry_short = entry_short[n_working:]
    holdout_tp = tp_pct_all[n_working:]
    holdout_sl = sl_pct_all[n_working:]

    # Grid: scale × timeout × conf × meta
    scale_grid = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
    rr_grid = [1.5, 2.0, 2.5]
    timeout_grid = [200, 400, 600, 800, 1000, 1300, 1500, 1700, 2000, 2500]
    conf_grid = np.round(np.linspace(0.0, 0.85, 18), 3).tolist()
    meta_grid = np.round(np.linspace(0.0, 0.85, 18), 3).tolist()
    n_total_cfg = (len(scale_grid) * len(rr_grid) * len(timeout_grid)
                   * len(conf_grid) * len(meta_grid))
    print(f"[gva] grid total: {n_total_cfg}", flush=True)

    # Pre-build take_mat per (conf, meta) pair
    n_pairs = len(conf_grid) * len(meta_grid)
    take_mat = np.zeros((n_h, n_pairs), dtype=bool)
    pair_meta = []
    pi = 0
    for conf_thr in conf_grid:
        for meta_thr in meta_grid:
            take_mat[:, pi] = (direction != 0) & (confidence >= conf_thr) & (meta_proba_all >= meta_thr)
            pair_meta.append((conf_thr, meta_thr))
            pi += 1
    print(f"[gva] take_mat {take_mat.shape}", flush=True)

    sqrt_year = np.sqrt(525600 / 60)
    eps = 1e-9
    results = []

    for scale in scale_grid:
        tp_per = np.clip(holdout_tp * scale, 0.05, 1.0).astype(np.float64)
        for rr in rr_grid:
            sl_per = np.clip(tp_per / rr, 0.02, 0.5).astype(np.float64)
            for to_ticks in timeout_grid:
                t0 = time.time()
                sim = rust_bridge.simulate_labels(
                    entry_long=holdout_entry_long,
                    entry_short=holdout_entry_short,
                    mid_paths=holdout_mid_paths,
                    tp_pct=tp_per,
                    sl_pct=sl_per,
                    timeout_ticks=np.full(n_h, to_ticks, dtype=np.int64),
                    partial_enabled=True, trailing_enabled=True,
                )
                pnl_long_h = sim["pnl_long"]
                pnl_short_h = sim["pnl_short"]
                realised = realise(direction, pnl_long_h, pnl_short_h)

                r_per_trade = realised[:, None] * take_mat
                n_per = take_mat.sum(0)
                valid = n_per > 0
                wins = (r_per_trade > 0).sum(0)
                wr = np.divide(wins * 100.0, np.maximum(n_per, 1),
                               out=np.zeros(n_pairs), where=valid)
                ev = np.divide(r_per_trade.sum(0), np.maximum(n_per, 1),
                               out=np.zeros(n_pairs), where=valid)
                sumsq = (r_per_trade ** 2).sum(0)
                var = sumsq / np.maximum(n_per, 1) - ev ** 2
                std = np.sqrt(np.maximum(var, eps))
                sharpe = np.divide(ev * sqrt_year, std + eps,
                                   out=np.zeros(n_pairs), where=valid)

                for pi_local in range(n_pairs):
                    if not valid[pi_local]:
                        continue
                    conf_thr, meta_thr = pair_meta[pi_local]
                    taken = take_mat[:, pi_local]
                    r = realised[taken]
                    n_tr = int(taken.sum())
                    eq = 50.0 * np.cumprod(1.0 + r / 100.0)
                    net = 100 * (float(eq[-1]) / 50.0 - 1)
                    peaks = np.maximum.accumulate(eq)
                    dd = float(((peaks - eq) / np.maximum(peaks, 1e-12)).max()) * 100
                    results.append({
                        "n": n_tr, "wr_pct": float(wr[pi_local]),
                        "ev": float(ev[pi_local]),
                        "net_return_pct": net, "max_dd_pct": dd,
                        "sharpe_ann": float(sharpe[pi_local]),
                        "scale": scale, "rr": rr, "timeout_ticks": to_ticks,
                        "conf_thr": float(conf_thr), "meta_thr": float(meta_thr),
                        "label": f"sc={scale} rr={rr} to={to_ticks} "
                                 f"conf>={conf_thr} meta>={meta_thr}",
                    })
        print(f"[gva] scale={scale} done, results={len(results)}", flush=True)

    nets = np.array([m["net_return_pct"] for m in results])
    n_arr = np.array([m["n"] for m in results])
    sharpes = np.array([m["sharpe_ann"] for m in results])
    pct_profit = float((nets > 0).mean() * 100)

    valid_n = [m for m in results if m["n"] >= 50]
    valid_n.sort(key=lambda r: -r["net_return_pct"])
    by_sharpe = sorted(results, key=lambda r: -r["sharpe_ann"])

    out_path.write_text(json.dumps({
        "n_holdout": n_h,
        "archs_used": arch_names,
        "n_predictions_up": int((pred==UP).sum()),
        "n_predictions_dn": int((pred==DN).sum()),
        "n_predictions_fl": int((pred==FL).sum()),
        "n_configs_total": len(results),
        "pct_profitable": pct_profit,
        "net_return_stats": {
            "min": float(nets.min()), "max": float(nets.max()),
            "median": float(np.median(nets)),
            "p05": float(np.percentile(nets, 5)),
            "p95": float(np.percentile(nets, 95)),
        },
        "trades_stats": {"min": int(n_arr.min()), "max": int(n_arr.max()),
                         "median": int(np.median(n_arr))},
        "sharpe_stats": {"min": float(sharpes.min()), "max": float(sharpes.max()),
                         "median": float(np.median(sharpes))},
        "top_30_net_n_ge_50": valid_n[:30],
        "top_30_sharpe": by_sharpe[:30],
    }, indent=2, default=float))
    print(f"\n[gva] saved {out_path}", flush=True)
    print(f"[gva] {len(results)} configs | profitable={pct_profit:.2f}% | "
          f"net min={nets.min():.2f}% median={np.median(nets):.2f}% max={nets.max():.2f}%",
          flush=True)
    if valid_n:
        print(f"\n=== TOP-15 by net (n>=50) ===", flush=True)
        for m in valid_n[:15]:
            print(f"  n={m['n']:5d} WR={m['wr_pct']:5.1f}% net={m['net_return_pct']:+7.2f}% "
                  f"DD={m['max_dd_pct']:5.1f}% Sh={m['sharpe_ann']:+5.2f} | {m['label']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
