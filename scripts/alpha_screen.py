#!/usr/bin/env python3
"""HA1 alpha screen — pure prediction, cost/execution-agnostic.

Pre-registered protocol (see research/PLAN.md "CURRENT DIRECTION" and the
chat spec 2026-05-17). Per symbol x forward-horizon h:

  X  = features_v1 (59 cols) at decision point k
  y  = log( mid[t_k + h] / mid[t_k] )   wall-clock h (gaps! via timestamp)
  XGB regressor (fixed hp, seed) — honest time split: train 70% |
  embargo (>= h coverage) | OOS 30%.

Measured on OOS only, per (symbol, h):
  rank_IC (Spearman), Pearson, R2, sign-AUC; decile monotonicity;
  block-bootstrap CI of rank_IC (block ~ h/cadence) -> n_eff decorrelated;
  top/bot predicted-decile mean |move|%; economic_pass_{loose=0.08%,
  strict=0.13%}; PLACEBO (shuffled y) rank_IC must be ~0 (features_v1 is
  a reconstructed pipeline — leak sentinel).

Emits ledger-ready kind='alpha' records to GCS. status defaults
'exploratory'; only 'confirmed' if CI excludes 0 AND decile_monotonic AND
economic_pass_strict AND placebo clean — and the ledger gate re-checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_cryptolake_cache import (  # noqa: E402
    _gcs_bucket, _list_days, _load_npy, _load_book_l1)

HORIZONS = (30, 60, 120, 180)
CADENCE_S = 24                 # features_v1 decision-point step
FLOOR_LOOSE = 0.08             # % maker round-trip, idealised fills
FLOOR_STRICT = 0.13            # % taker 0.10 + slippage/latency haircut
SEED = 42


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        random_state=SEED, objective="reg:squarederror")


def _rank(a):
    o = a.argsort(); r = np.empty(len(a), np.float64); r[o] = np.arange(len(a))
    return r


def _spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _collect_symbol(bk, sym, days):
    """-> X (n,59) f32, per-h y dict, day-id (n,). One book read per day,
    all horizons reuse it."""
    Xs, dayids = [], []
    ys = {h: [] for h in HORIZONS}
    for di, day in enumerate(days):
        try:
            X = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/features.npy")
            idx = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/indices.npy"
                            ).astype(np.int64)
            ts, bid0, ask0 = _load_book_l1(bk, sym, day)
        except Exception as e:
            print(f"  {sym} {day}: skip ({type(e).__name__})", flush=True)
            continue
        if X.shape[0] != idx.shape[0]:
            continue
        n_book = ts.shape[0]
        mid = (bid0 + ask0) * 0.5
        i = idx
        ok = (i > 0) & (i < n_book - 1)
        i = i[ok]; Xd = X[ok]
        if i.size == 0:
            continue
        t0 = ts[i]; m0 = mid[i]
        keep = np.ones(i.size, bool)
        yh = {}
        for h in HORIZONS:
            j = np.searchsorted(ts, t0 + h * 1_000_000_000, side="left")
            vj = j < n_book
            r = np.full(i.size, np.nan)
            r[vj] = np.log(mid[j[vj]] / m0[vj])
            yh[h] = r
            keep &= vj & np.isfinite(r) & (m0 > 0)
        if keep.sum() == 0:
            continue
        Xs.append(Xd[keep].astype(np.float32))
        dayids.append(np.full(int(keep.sum()), di, np.int32))
        for h in HORIZONS:
            ys[h].append(yh[h][keep])
    if not Xs:
        return None
    return (np.concatenate(Xs),
            {h: np.concatenate(ys[h]) for h in HORIZONS},
            np.concatenate(dayids))


def _screen(sym, h, X, y, run_id, git, days, n_total):
    n = X.shape[0]
    n_tr = int(n * 0.70)
    emb = max(32, int(np.ceil(h / CADENCE_S)) * 4)   # >= horizon coverage
    tr = slice(0, n_tr); te = slice(n_tr + emb, n)
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    m = _xgb(); m.fit(Xtr, ytr)
    p = m.predict(Xte)

    ric = _spearman(p, yte)
    pear = float(np.corrcoef(p, yte)[0, 1])
    ss = ((yte - yte.mean()) ** 2).sum()
    r2 = float(1 - ((yte - p) ** 2).sum() / ss) if ss > 0 else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score((yte > 0).astype(int), p)) \
            if (yte > 0).any() and (yte <= 0).any() else 0.5
    except Exception:
        auc = None

    # deciles by prediction
    q = np.quantile(p, np.linspace(0, 1, 11))
    q[-1] += 1e-12
    b = np.clip(np.digitize(p, q[1:-1]), 0, 9)
    dmean = np.array([yte[b == k].mean() if (b == k).any() else np.nan
                      for k in range(10)])
    dm_ic = _spearman(np.arange(10)[~np.isnan(dmean)],
                      dmean[~np.isnan(dmean)])
    decile_monotonic = int(abs(dm_ic) >= 0.9)
    top = yte[b == 9]; bot = yte[b == 0]
    top_abs = float(np.abs(top).mean() * 100) if top.size else 0.0
    bot_abs = float(np.abs(bot).mean() * 100) if bot.size else 0.0
    top_signed = float(top.mean() * 100) if top.size else 0.0
    sign_ok = (top_signed > 0)            # top decile should be net up
    e_loose = int(sign_ok and top_abs > FLOOR_LOOSE)
    e_strict = int(sign_ok and top_abs > FLOOR_STRICT)

    # block bootstrap CI of rank-IC (autocorr-aware)
    blk = max(1, int(np.ceil(h / CADENCE_S)))
    nb = max(1, len(p) // blk)
    rng = np.random.default_rng(SEED)
    boot = []
    starts = np.arange(0, len(p) - blk + 1)
    for _ in range(300):
        s = rng.choice(starts, nb, replace=True)
        ix = np.concatenate([np.arange(x, x + blk) for x in s])
        boot.append(_spearman(p[ix], yte[ix]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ci_excl0 = int(lo > 0 or hi < 0)
    n_eff = int(len(p) * CADENCE_S / h)

    # placebo: shuffled labels must give ~0 IC (leak sentinel)
    placebo = _spearman(p, rng.permutation(yte))

    status = "exploratory"
    if ci_excl0 and decile_monotonic and e_strict and abs(placebo) < 0.01:
        status = "confirmed"
    elif ci_excl0 and abs(placebo) < 0.01:
        status = "suspect"

    return {
        "experiment_id": f"{run_id}_HA1_{sym}_h{h}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git, "author": "claude(vm)", "hypothesis_id": "HA1",
        "status": status, "kind": "alpha",
        "setup": f"HA1 horizon screen XGB ({sym}, h={h}s)",
        "model_family": "xgb",
        "params": {"h": h, "n_est": 300, "max_depth": 5, "lr": 0.05,
                   "seed": SEED, "embargo": emb, "deciles": 10,
                   "cadence_s": CADENCE_S, "split": "70/emb/30",
                   "floor_loose": FLOOR_LOOSE, "floor_strict": FLOOR_STRICT,
                   "ci95": [round(float(lo), 5), round(float(hi), 5)],
                   "pearson": round(pear, 5), "placebo_ic": round(placebo, 5),
                   "top_decile_signed_pct": round(top_signed, 4),
                   "bot_decile_absmove_pct": round(bot_abs, 4),
                   "n_oos_raw": int(len(p))},
        "data_source": "cryptolake",
        "cache_id": f"cryptolake_{sym}_features_v1_{days[0]}_{days[-1]}",
        "symbols": [sym], "date_range_start": days[0],
        "date_range_end": days[-1], "n_samples": int(n_total),
        "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
        "commission_loss_pct": 0.07, "split_method": "honest_val_test",
        "embargo": str(emb),
        "label_def": (f"y = log(mid[t+{h}s]/mid[t]); mid=(bid0+ask0)/2; "
                      f"wall-clock horizon; execution-neutral"),
        "alpha_target": "fwd_mid_logret", "horizon_sec": h,
        "rank_ic_oos": round(ric, 5), "r2_oos": round(r2, 5),
        "auc_oos": None if auc is None else round(auc, 4),
        "top_decile_absmove_pct": round(top_abs, 4),
        "bot_decile_absmove_pct": round(bot_abs, 4),
        "cost_floor_pct": FLOOR_STRICT,
        "decile_monotonic": decile_monotonic,
        "economic_pass_loose": e_loose,
        "economic_pass_strict": e_strict, "n_eff": n_eff,
        "repro_cmd": (f"python scripts/alpha_screen.py --symbols {sym} "
                      f"--days {len(days)} (run {run_id})"),
        "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["LINK-USDT-PERP", "SOL-USDT-PERP"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--git-commit", default="unknown")
    a = ap.parse_args(argv)
    bk = _gcs_bucket()
    out = {"run_id": a.run_id, "records": [],
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for sym in a.symbols:
        try:
            days = _list_days(bk, sym)[-a.days:]
            c = _collect_symbol(bk, sym, days)
            if c is None:
                out["records"].append({"symbol": sym, "error": "no samples"})
                continue
            X, ys, _ = c
            print(f"[{sym}] n={X.shape[0]} days={len(days)}", flush=True)
            for h in HORIZONS:
                rec = _screen(sym, h, X, ys[h], a.run_id, a.git_commit,
                              days, X.shape[0])
                out["records"].append(rec)
                print(f"  h={h}s ric={rec['rank_ic_oos']} "
                      f"top|mv|={rec['top_decile_absmove_pct']}% "
                      f"eL={rec['economic_pass_loose']} "
                      f"eS={rec['economic_pass_strict']} "
                      f"mono={rec['decile_monotonic']} "
                      f"placebo={rec['params']['placebo_ic']} "
                      f"{rec['status']}", flush=True)
        except Exception as e:
            out["records"].append({"symbol": sym, "error": repr(e),
                                    "trace": traceback.format_exc()})
            print(f"[{sym}] ERROR {e}", flush=True)
        bk.blob(f"research_runs/{a.run_id}/results.json").upload_from_string(
            json.dumps(out, indent=2, default=str))
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bk.blob(f"research_runs/{a.run_id}/results.json").upload_from_string(
        json.dumps(out, indent=2, default=str))
    print("PHASE_B_DONE", a.run_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
