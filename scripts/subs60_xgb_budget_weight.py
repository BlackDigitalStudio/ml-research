#!/usr/bin/env python3
"""DOGE apred walk-forward: BUDGET (trades/day) x A/B-confidence WEIGHTING surface,
with a PERSISTENT per-fold cache (train ONCE, re-sweep selection forever for free).

Regime-ADAPTIVE vol-gate threshold (the winning config). The deploy SELECTION does not
depend on the models, so we train A+B per fold ONCE, SAVE per-fold weights + test-row
predictions to GCS, and sweep the selection cheaply:
    score = pct_rank(pA)**wA * pct_rank(|pB-0.5|)**wB ,  daily top-`budget`.
On re-run the cache is loaded and NO training happens. Saves:
  research_runs/wf_cache/DOGE_adaptive_W200T30_preds.npz   (per-fold test preds)
  research_runs/wf_models/DOGE_adaptive_W200T30/fold{i}_{A,B}.json  (per-fold weights)
  research_runs/maker_labels_rr/BUDGET_WEIGHT_RESULT.json  (the surface)
"""
import io, json, os, tempfile
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
NF_RATE = 0.05; GATE_PCT = 5.0
W, T, EMB = 200, 30, 2
TAG = f"DOGE_adaptive_W{W}T{T}"
CACHE_PREDS = f"research_runs/wf_cache/{TAG}_preds.npz"
MODELS_DIR = f"research_runs/wf_models/{TAG}"
BUDGETS = [1, 2, 5, 10]
WEIGHTS = [("A=B(1,1)", 1.0, 1.0), ("A2(2,1)", 2.0, 1.0), ("A3(3,1)", 3.0, 1.0),
           ("A-only(1,0)", 1.0, 0.0), ("B-only(0,1)", 0.0, 1.0)]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_rr(sub, symk):
    d = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pl": d["pnl_long"].astype(np.float32), "ps": d["pnl_short"].astype(np.float32),
            "fl": d["fill_long"].astype(bool), "fs": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def pct_rank(x): o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def fit(hp, niter, X, y, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def oof_pA(F, yA, trn, day, hpA, k=5):
    tdays = sorted(set(day[trn].tolist())); fold = {d: i % k for i, d in enumerate(tdays)}
    fday = np.array([fold.get(int(d), -1) for d in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk)
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


def save_model_gcs(bst, blobname):
    p = tempfile.mktemp(suffix=".json"); bst.save_model(p)
    bk.blob(blobname).upload_from_filename(p); os.remove(p)


SYMK = "DOGE"
hpA = jload(f"{MAIN}/A_{SYMK}.json"); hpB = jload(f"{MAIN}/B_pool.json")
E = load_rr("maker_labels_rr", SYMK)
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]; ndays = E["ndays"]
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee


def train_fold(trn, thr, fi):
    yA = (np.abs(rH) >= thr).astype(int)
    spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
    bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
    oof = oof_pA(F, yA, trn, day, hpA); valid = trn & np.isfinite(oof)
    thr_oof = float(np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0)); gate = valid & (oof >= thr_oof)
    keep = gate & (fl | fs); nl = netl[0]; ns = nets[0]
    yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int); both = fl & fs
    w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
    wc = np.clip(w[keep], 0, np.quantile(w[keep][w[keep] > 0], 0.99))
    bstB = fit(hpB["best_params"], hpB["best_iter"], F[keep], yB[keep], w=wc)
    save_model_gcs(bstA, f"{MODELS_DIR}/fold{fi}_A.json")
    save_model_gcs(bstB, f"{MODELS_DIR}/fold{fi}_B.json")
    return bstA, bstB


# ---------- load cache OR train-once-and-cache ----------
if bk.blob(CACHE_PREDS).exists():
    z = np.load(io.BytesIO(bk.blob(CACHE_PREDS).download_as_bytes()), allow_pickle=True)
    fid = z["fold_id"]
    print(f"[cache HIT] {CACHE_PREDS} -> NO training; {len(set(fid.tolist()))} folds, {len(fid)} test rows", flush=True)
    folds = []
    for i in sorted(set(fid.tolist())):
        m = fid == i
        folds.append({"day": z["day"][m], "pA": z["pA"][m], "pB": z["pB"][m], "fl": z["fl"][m],
                      "fs": z["fs"][m], "nl": z["nl"][m], "ns": z["ns"][m], "ndays": len(set(z["day"][m].tolist()))})
else:
    print(f"[cache MISS] training per fold ONCE, saving weights -> {MODELS_DIR}/ + preds -> {CACHE_PREDS}", flush=True)
    folds = []; acc = {k: [] for k in ("fold_id", "day", "pA", "pB", "fl", "fs", "nl", "ns")}
    ts = W + EMB; fi = 0
    while ts < ndays:
        te = min(ts + T, ndays)
        trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
        if tst.sum() < 50 or trn.sum() < 5000:
            ts += T; continue
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE))
        bstA, bstB = train_fold(trn, thr, fi); ti = np.where(tst)[0]
        pA = bstA.predict(xgb.DMatrix(F[ti])); pB = bstB.predict(xgb.DMatrix(F[ti]))
        folds.append({"day": day[ti], "pA": pA, "pB": pB, "fl": fl[ti], "fs": fs[ti],
                      "nl": netl[0][ti], "ns": nets[0][ti], "ndays": len(set(day[ti].tolist()))})
        acc["fold_id"].append(np.full(len(ti), fi)); acc["day"].append(day[ti]); acc["pA"].append(pA)
        acc["pB"].append(pB); acc["fl"].append(fl[ti]); acc["fs"].append(fs[ti])
        acc["nl"].append(netl[0][ti]); acc["ns"].append(nets[0][ti])
        print(f"  fold{fi} [{ts}-{te}) trained+saved, test rows={len(ti)} days={folds[-1]['ndays']} thr={thr:.1f}", flush=True)
        ts += T; fi += 1
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.concatenate(v) for k, v in acc.items()},
                        meta=np.array(json.dumps({"W": W, "T": T, "EMB": EMB, "tag": TAG})))
    bk.blob(CACHE_PREDS).upload_from_string(buf.getvalue())
    print(f"[cached] {CACHE_PREDS} + {fi} folds of weights", flush=True)

tot_days = sum(f["ndays"] for f in folds)


def sel_nets(fd, wA, wB, budget):
    a = pct_rank(fd["pA"]); b = pct_rank(np.abs(fd["pB"] - 0.5))
    score = (a ** wA) * (b ** wB) if (wA or wB) else np.ones_like(a)
    sel = daily_pick(fd["day"], score, budget)
    side = fd["pB"][sel] >= 0.5; net = np.where(side, fd["nl"][sel], fd["ns"][sel])
    fc = np.where(side, fd["fl"][sel], fd["fs"][sel]); ex = fc & np.isfinite(net)
    return net[ex]


print(f"\n{'weighting':>12} | " + " ".join(f"{'b=' + str(b):>17}" for b in BUDGETS) + "   net EV/trd bp (xTrd/day /totBp/day)", flush=True)
out = {"W": W, "T": T, "EMB": EMB, "tot_test_days": tot_days, "cells": {}}
for wlab, wA, wB in WEIGHTS:
    cells = []
    for bud in BUDGETS:
        pool = np.concatenate([sel_nets(fd, wA, wB, bud) for fd in folds]) if folds else np.array([])
        n = len(pool); ev = float(pool.mean()) if n else float("nan")
        se = float(pool.std() / max(np.sqrt(n), 1)) if n else float("nan")
        tpd = n / max(tot_days, 1)
        cells.append(f"{ev:+6.2f}+-{se:3.1f}(x{tpd:.1f}/{ev*tpd:+5.1f})")
        out["cells"][f"{wlab}|b{bud}"] = {"net_ev": ev, "se": se, "n": n, "trades_per_day": tpd, "tot_per_day": ev * tpd}
    print(f"{wlab:>12} | " + " ".join(f"{c:>17}" for c in cells), flush=True)

bk.blob("research_runs/maker_labels_rr/BUDGET_WEIGHT_RESULT.json").upload_from_string(json.dumps(out, default=float))
print("\n[saved] research_runs/maker_labels_rr/BUDGET_WEIGHT_RESULT.json", flush=True)
print("legend: cell = net_EV/trade +- SE  (xTRADES/day / TOTAL_bp_per_day=EVxtrd)", flush=True)
