#!/usr/bin/env python3
"""HBV1 analysis (non-frozen): is there ONE optimal HP region — do seeds whose
chosen hyperparameters sit closer to it perform better?

Data: MODELS_S{s}_DOGE_f{k}_hp.json (chosen hpA/hpB + val scores) x per-(seed,fold)
outcome EV (per-seed causal t5 selection on that fold, per-trade EV, own PERFOLD).
Method (fold identity is a huge confounder — burst folds dominate EV):
  1. within-fold DEMEANED Spearman corr of each HP dim vs fold EV, pooled;
  2. per-seed: mean HP position (log-scale for log-searched dims) vs seed EV;
  3. distance-to-centroid: EV-weighted HP centroid; corr(dist, EV) within fold.
Pools via env POOLS "name=sub:seeds;..." (v2-family only — v1 has no dumps).
"""
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = "DOGE"
KDAYS = 30

POOLS = {}
for spec in os.environ.get("POOLS", "v2-base=maker_labels_tb3s_h150anch:0,1,2,3;"
                                    "v2-nooi=maker_labels_tb3s_h150anch_v2_nooi:0,1,2,3,4,5,6,7").split(";"):
    name, rest = spec.split("=")
    sub, seeds = rest.split(":")
    POOLS[name] = (sub, [int(x) for x in seeds.split(",")])

HP_B = ["max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight",
        "reg_lambda", "reg_alpha"]
LOG_DIMS = {"eta", "min_child_weight", "reg_lambda", "reg_alpha"}


def causal_ev(z, tgt=5.0):
    sc_tr = z["axb_tr"].astype(np.float64); sc_te = z["axb_te"].astype(np.float64)
    day_tr = z["day_tr"]; day_te = z["day_te"]
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.nan
    side = z["side"][sel]
    net = np.where(side, z["netl"].astype(np.float64)[sel], z["nets"].astype(np.float64)[sel])
    fc = np.where(side, z["fl"][sel], z["fs"][sel])
    ex = fc & np.isfinite(net)
    return float(net[ex].mean()) if ex.any() else np.nan


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)


for name, (sub, seeds) in POOLS.items():
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"research_runs/{sub}/PERFOLD_S{seeds[0]}_{SYM}_qm0_f")
             if b.name.endswith(".npz"))
    rows = []  # (seed, fold, ev, hpB dict, icB, aucA, nrB)
    for s in seeds:
        for f in range(nf):
            try:
                hp = json.loads(bk.blob(f"research_runs/{sub}/MODELS_S{s}_{SYM}_f{f}_hp.json").download_as_bytes())
                z = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
            except Exception:
                continue
            rows.append((s, f, causal_ev(z), hp["hpB"], hp.get("icB"), hp.get("aucA"), hp.get("biB")))
    if not rows:
        print(f"### {name}: no hp dumps", flush=True)
        continue
    print(f"\n### {name}: {len(rows)} (seed,fold) points, {nf} folds, seeds {seeds}", flush=True)
    # 1. within-fold demeaned correlations
    ev_dm = {}
    by_fold = defaultdict(list)
    for i, (s, f, ev, hpB, icB, aucA, nrB) in enumerate(rows):
        by_fold[f].append(i)
    ev_arr = np.array([r[2] for r in rows], float)
    dm = np.full(len(rows), np.nan)
    for f, idxs in by_fold.items():
        vals = ev_arr[idxs]
        if np.isfinite(vals).sum() >= 2:
            dm[idxs] = vals - np.nanmean(vals)
    print("  HP dim -> Spearman(within-fold-demeaned EV):", flush=True)
    for d in HP_B + ["num_boost_round", "icB"]:
        if d == "num_boost_round":
            x = [r[6] for r in rows]
        elif d == "icB":
            x = [r[4] for r in rows]
        else:
            x = [np.log(r[3][d]) if d in LOG_DIMS else r[3][d] for r in rows]
        print(f"    {d:18s} {spearman(x, dm):+.3f}", flush=True)
    # 2. per-seed mean EV vs per-seed HP means
    print("  per-seed: EV | depth eta(log) mcw(log) lambda(log) nr:", flush=True)
    for s in seeds:
        ri = [r for r in rows if r[0] == s]
        evs = np.nanmean([r[2] for r in ri])
        md = np.mean([r[3]["max_depth"] for r in ri]); et = np.mean([np.log(r[3]["eta"]) for r in ri])
        mc = np.mean([np.log(r[3]["min_child_weight"]) for r in ri])
        rl = np.mean([np.log(r[3]["reg_lambda"]) for r in ri]); nr = np.mean([r[6] for r in ri])
        print(f"    S{s}: {evs:+6.2f} | {md:4.1f} {et:+5.2f} {mc:+5.2f} {rl:+5.2f} {nr:5.0f}", flush=True)
    # 3. distance to EV-weighted centroid (z-scored dims), within-fold corr
    X = np.array([[np.log(r[3][d]) if d in LOG_DIMS else float(r[3][d]) for d in HP_B] for r in rows])
    Xz = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    w = np.clip(ev_arr - np.nanmin(ev_arr), 0, None)
    w = np.where(np.isfinite(w), w, 0)
    cent = (Xz * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    dist = np.linalg.norm(Xz - cent[None, :], axis=1)
    print(f"  Spearman(dist-to-EV-centroid, demeaned EV): {spearman(dist, dm):+.3f} "
          f"(negative = closer is better)", flush=True)
print("\n[done]", flush=True)
