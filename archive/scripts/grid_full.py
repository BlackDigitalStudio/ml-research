#!/usr/bin/env python3
"""Phase 2.2 — Big Grid на full L1+L2+L3 ensemble на 37k holdout.

Reads:
    models/stacker_v3.json     — XGBoost L2 stacker
    models/meta_v3.pkl         — XGBoost L3 meta gate
    models/holdout_X_stack.npy — (n_holdout, K*3) — concat per-arch holdout softmax
                                    в порядке arch_names из stacker_summary.json
    cache/{prefix}_*.npy       — y, mid_paths, entry_long, entry_short

Sweep TP/SL/timeout/conf_thr/meta_thr → direction-aware net return + Sharpe.
Output: runs/grid_full_top_configs.json.
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

UP, DOWN, FLAT = 0, 1, 2


def realise(direction, pnl_long, pnl_short):
    return np.where(direction == +1, pnl_long,
             np.where(direction == -1, pnl_short, 0.0))


def metrics(realised, taken_mask, label):
    r = realised[taken_mask]
    n = int(taken_mask.sum())
    if n == 0:
        return None
    wr = float((r > 0).mean() * 100)
    s = float(r.sum())
    eq = 50.0 * np.cumprod(1.0 + r / 100.0)
    net = 100 * (float(eq[-1]) / 50.0 - 1)
    peaks = np.maximum.accumulate(eq)
    dd = float(((peaks - eq) / np.maximum(peaks, 1e-12)).max()) * 100
    ev = float(r.mean())
    sigma = max(r.std(ddof=1), 1e-9)
    sharpe = float(r.mean() / sigma) * np.sqrt(525600 / 60)
    return {"label": label, "n": n, "wr_pct": wr, "ev": ev,
            "sum_pnl_pct": s, "net_return_pct": net, "max_dd_pct": dd,
            "sharpe_ann": sharpe}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--summary", default="models/stacker_summary.json")
    ap.add_argument("--cv-holdout-frac", type=float, default=0.20)
    ap.add_argument("--out", default="runs/grid_full_top_configs.json")
    ap.add_argument("--validation-net-min", type=float, default=-90.0)
    ap.add_argument("--validation-net-max", type=float, default=50.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = args.cache_prefix
    print(f"[grid_full] loading cache {p}", flush=True)
    y = np.load(f"{p}_y.npy")
    mid_paths = np.load(f"{p}_mid_paths.npy")
    entry_long = np.load(f"{p}_entry_long.npy")
    entry_short = np.load(f"{p}_entry_short.npy")
    n_total = len(y)
    n_holdout = int(n_total * args.cv_holdout_frac)
    n_working = n_total - n_holdout
    print(f"[grid_full] holdout idx [{n_working}, {n_total}) = {n_holdout}",
          flush=True)

    # Load stacker + meta
    stacker_path = Path(args.models_dir) / "stacker_v3.json"
    meta_path = Path(args.models_dir) / "meta_v3.pkl"
    holdout_X_path = Path(args.models_dir) / "holdout_X_stack.npy"
    summary = json.loads(Path(args.summary).read_text())
    arch_names = summary["archs_used"]
    print(f"[grid_full] L1 archs: {arch_names}", flush=True)

    if not (stacker_path.exists() and meta_path.exists() and holdout_X_path.exists()):
        print("[grid_full] missing models or holdout_X_stack.npy — abort",
              flush=True)
        return 1

    stacker = xgb.XGBClassifier()
    stacker.load_model(str(stacker_path))
    meta = joblib.load(meta_path)

    H = np.load(holdout_X_path)
    print(f"[grid_full] H {H.shape}", flush=True)
    expected_cols = len(arch_names) * 3
    if H.shape[1] != expected_cols:
        print(f"[grid_full] H cols {H.shape[1]} != {expected_cols} — abort",
              flush=True)
        return 1

    proba = stacker.predict_proba(H)
    pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    direction = np.where(pred == UP, +1,
                  np.where(pred == DOWN, -1, 0)).astype(np.int8)
    n_h = len(direction)
    print(f"[grid_full] L2 predictions: UP={int((pred==UP).sum())} "
          f"DN={int((pred==DOWN).sum())} FL={int((pred==FLAT).sum())}",
          flush=True)

    # Meta gate for non-FLAT
    max_prob = proba.max(axis=1, keepdims=True)
    entropy = (-proba * np.log(proba + 1e-12)).sum(axis=1, keepdims=True)
    X_meta = np.hstack([proba, max_prob, entropy])
    meta_proba_all = np.zeros(n_h, dtype=np.float32)
    nonflat = pred != FLAT
    if nonflat.any():
        meta_proba_all[nonflat] = meta.predict_proba(X_meta[nonflat])[:, 1]
    print(f"[grid_full] L3 meta median={np.median(meta_proba_all[nonflat]):.3f} "
          f"if non-FLAT", flush=True)

    holdout_mid_paths = mid_paths[n_working:]
    holdout_entry_long = entry_long[n_working:]
    holdout_entry_short = entry_short[n_working:]
    if len(holdout_mid_paths) != n_h:
        print(f"[grid_full] holdout mid_paths len {len(holdout_mid_paths)} "
              f"!= n_h {n_h}", flush=True)
        return 1

    # Wide grid (~500K configs) — broad timing zone и около-пограничные точки
    tp_grid = [0.10, 0.125, 0.15, 0.175, 0.20, 0.22, 0.25, 0.27,
               0.30, 0.32, 0.35, 0.40, 0.45, 0.50]                       # 14
    sl_grid = [0.05, 0.07, 0.08, 0.10, 0.11, 0.12, 0.14, 0.15,
               0.18, 0.20, 0.25]                                          # 11
    timeout_grid_ticks = [100, 200, 300, 500, 700, 900, 1100,
                          1300, 1500, 1700, 2000, 2500, 3000]             # 13
    confidence_thresholds = np.round(np.linspace(0.0, 0.85, 18), 3).tolist()  # 18
    meta_thresholds = np.round(np.linspace(0.0, 0.85, 18), 3).tolist()       # 18

    n_pre = (len(tp_grid) * len(sl_grid) * len(timeout_grid_ticks)
             * len(confidence_thresholds) * len(meta_thresholds))

    # R:R filter (расширенный — 1.0..4.0)
    rr_pairs = [(tp, sl) for tp in tp_grid for sl in sl_grid
                if 1.0 <= tp / sl <= 4.0]
    n_post = (len(rr_pairs) * len(timeout_grid_ticks)
              * len(confidence_thresholds) * len(meta_thresholds))
    print(f"[grid_full] grid: pre={n_pre} post-R:R={n_post}", flush=True)

    # Pre-build vectorized take masks for all (conf_thr, meta_thr) pairs
    # take_mat: (n_h, n_conf*n_meta) bool. Heavy memory ~37k*324 = ~12MB OK.
    nf = direction != 0
    n_pairs = len(confidence_thresholds) * len(meta_thresholds)
    take_mat = np.zeros((n_h, n_pairs), dtype=bool)
    pair_meta = []
    pi = 0
    for conf_thr in confidence_thresholds:
        for meta_thr in meta_thresholds:
            take_mat[:, pi] = nf & (confidence >= conf_thr) & (meta_proba_all >= meta_thr)
            pair_meta.append((conf_thr, meta_thr))
            pi += 1
    print(f"[grid_full] take_mat built {take_mat.shape}", flush=True)

    results = []
    cfg_idx = 0
    sqrt_year = np.sqrt(525600 / 60)
    eps = 1e-9

    for tp, sl in rr_pairs:
        rr = round(tp / sl, 2)
        for to_ticks in timeout_grid_ticks:
            sim = rust_bridge.simulate_labels(
                entry_long=holdout_entry_long,
                entry_short=holdout_entry_short,
                mid_paths=holdout_mid_paths,
                tp_pct=np.full(n_h, tp, dtype=np.float64),
                sl_pct=np.full(n_h, sl, dtype=np.float64),
                timeout_ticks=np.full(n_h, to_ticks, dtype=np.int64),
                partial_enabled=True, trailing_enabled=True,
            )
            pnl_long_h = sim["pnl_long"]
            pnl_short_h = sim["pnl_short"]
            realised = realise(direction, pnl_long_h, pnl_short_h)
            # vectorized metrics across pairs
            r_per_trade = realised[:, None] * take_mat  # (n_h, n_pairs)
            n_per_pair = take_mat.sum(0)  # (n_pairs,)
            valid_pair = n_per_pair > 0
            # Sum & WR vectorized
            sum_pnl = r_per_trade.sum(0)  # (n_pairs,)
            wins = (r_per_trade > 0).sum(0)  # only non-zero entries are taken trades
            wr = np.divide(wins * 100.0, np.maximum(n_per_pair, 1),
                           out=np.zeros(n_pairs), where=valid_pair)
            # EV — mean of taken trades (zero-mask sum / count)
            ev = np.divide(sum_pnl, np.maximum(n_per_pair, 1),
                           out=np.zeros(n_pairs), where=valid_pair)
            # Std для Sharpe — нужно вторичный момент только по taken
            sumsq = (r_per_trade ** 2).sum(0)
            mean = ev
            var = sumsq / np.maximum(n_per_pair, 1) - mean ** 2
            std = np.sqrt(np.maximum(var, eps))
            sharpe = np.divide(mean * sqrt_year, std + eps,
                               out=np.zeros(n_pairs), where=valid_pair)

            # net_return и DD требуют per-trade cumprod — это дороже,
            # делаем для valid pairs only через цикл (vector в pandas был бы лучше)
            for pi_local in range(n_pairs):
                cfg_idx += 1
                if not valid_pair[pi_local]:
                    continue
                conf_thr, meta_thr = pair_meta[pi_local]
                taken = take_mat[:, pi_local]
                r = realised[taken]
                n = int(taken.sum())
                eq = 50.0 * np.cumprod(1.0 + r / 100.0)
                net = 100 * (float(eq[-1]) / 50.0 - 1)
                peaks = np.maximum.accumulate(eq)
                dd = float(((peaks - eq) / np.maximum(peaks, 1e-12)).max()) * 100
                results.append({
                    "n": n, "wr_pct": float(wr[pi_local]),
                    "ev": float(ev[pi_local]),
                    "sum_pnl_pct": float(sum_pnl[pi_local]),
                    "net_return_pct": net, "max_dd_pct": dd,
                    "sharpe_ann": float(sharpe[pi_local]),
                    "tp": tp, "sl": sl, "timeout_ticks": to_ticks,
                    "conf_thr": float(conf_thr),
                    "meta_thr": float(meta_thr), "rr": rr,
                    "label": f"tp={tp} sl={sl} to={to_ticks} "
                             f"conf>={conf_thr} meta>={meta_thr}",
                })
        if cfg_idx > 0 and len(results) % 5000 < 400:
            print(f"[grid_full] processed {cfg_idx} cfgs (results={len(results)})",
                  flush=True)

    if not results:
        print("[grid_full] no valid configs", flush=True)
        return 2

    for m in results:
        m["score"] = m["sharpe_ann"] - 0.1 * m["max_dd_pct"]
    results.sort(key=lambda r: -r["score"])

    flagged = []
    for m in results:
        if (m["net_return_pct"] < args.validation_net_min or
            m["net_return_pct"] > args.validation_net_max):
            flagged.append(m)

    print(f"\n=== TOP-15 by Sharpe - 0.1*DD ===", flush=True)
    print(f"{'cfg':70s} n      WR     EV       net    DD     Sh    score",
          flush=True)
    for m in results[:15]:
        print(f"{m['label']:70s} {m['n']:>5d} {m['wr_pct']:5.1f}% "
              f"{m['ev']:+5.3f}% {m['net_return_pct']:+7.2f}% "
              f"{m['max_dd_pct']:5.1f}% {m['sharpe_ann']:+5.2f} "
              f"{m['score']:+5.2f}", flush=True)

    if flagged:
        print(f"\n[grid_full] WARN: {len(flagged)} configs outside validation "
              f"range [{args.validation_net_min}, {args.validation_net_max}] — "
              f"possible leakage", flush=True)
        for m in flagged[:5]:
            print(f"  {m['label']}  net={m['net_return_pct']:+7.2f}%",
                  flush=True)

    # Distribution stats over всех configs (для оценки шума без сохранения 500K)
    nets = np.array([m["net_return_pct"] for m in results])
    n_arr = np.array([m["n"] for m in results])
    sharpes = np.array([m["sharpe_ann"] for m in results])
    pct_profit = float((nets > 0).mean() * 100)

    # ranking by net_return при min n>=50 (более надёжные)
    valid_n = [m for m in results if m["n"] >= 50]
    valid_n.sort(key=lambda r: -r["net_return_pct"])

    out_path.write_text(json.dumps({
        "n_holdout": n_h,
        "archs_used": arch_names,
        "n_predictions_up": int((pred==UP).sum()),
        "n_predictions_dn": int((pred==DOWN).sum()),
        "n_predictions_fl": int((pred==FLAT).sum()),
        "n_configs_total": len(results),
        "pct_profitable": pct_profit,
        "net_return_stats": {
            "min": float(nets.min()), "max": float(nets.max()),
            "median": float(np.median(nets)),
            "p05": float(np.percentile(nets, 5)),
            "p95": float(np.percentile(nets, 95)),
        },
        "trades_stats": {
            "min": int(n_arr.min()), "max": int(n_arr.max()),
            "median": int(np.median(n_arr)),
        },
        "sharpe_stats": {
            "min": float(sharpes.min()), "max": float(sharpes.max()),
            "median": float(np.median(sharpes)),
        },
        "top_30_score": results[:30],
        "top_30_net_return_n_ge_50": valid_n[:30],
        "flagged_configs": flagged[:20],
    }, indent=2, default=float))
    print(f"\n[grid_full] saved {out_path}", flush=True)
    print(f"[grid_full] {len(results)} configs total | "
          f"profitable={pct_profit:.2f}% | "
          f"net min={nets.min():.2f}% median={np.median(nets):.2f}% max={nets.max():.2f}%",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
