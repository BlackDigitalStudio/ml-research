#!/usr/bin/env python3
"""DOGE apred-cascade WALK-FORWARD: regime-ADAPTIVE vs FIXED Model-A vol-gate threshold.

Rolling train (W days) -> test (T days), embargo EMB, stepped over the year on the
existing clean maker_labels_rr/DOGE. Both branches RETRAIN A+B per fold (apred-5%
gate, hold-60s B, 1 trade/day, maker 4bp). ONLY the A vol-gate LABEL threshold differs:
  adaptive : thr = p95(|rH60|) on THIS fold's train window  (tracks the regime)
  fixed    : thr = p95(|rH60|) on the FIRST fold's train     (frozen, the hardcoded way)
Headline = pooled OOS net maker EV per branch (+ per-fold breakdown & per-fold thr).
Saves -> research_runs/maker_labels_rr/WALKFORWARD_ADAPTIVE.json
"""
import io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
NF_RATE = 0.05; GATE_PCT = 5.0
W, T, EMB = 200, 30, 2
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


SYMK = "DOGE"
hpA = jload(f"{MAIN}/A_{SYMK}.json"); hpB = jload(f"{MAIN}/B_pool.json")
E = load_rr("maker_labels_rr", SYMK)
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]; ndays = E["ndays"]
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee


def train_eval(trn, tst, thr):
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
    ti = np.where(tst)[0]
    pA = bstA.predict(xgb.DMatrix(F[ti])); pB = bstB.predict(xgb.DMatrix(F[ti]))
    score = pct_rank(pA) * pct_rank(np.abs(pB - 0.5)); sel = daily_pick(day[ti], score, 1)
    side = pB[sel] >= 0.5; net = np.where(side, netl[0][ti][sel], nets[0][ti][sel])
    fc = np.where(side, fl[ti][sel], fs[ti][sel]); ex = fc & np.isfinite(net)
    return net[ex]


def stats(a):
    a = np.asarray(a)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "net": float(a.mean()), "se": float(a.std() / max(np.sqrt(len(a)), 1)),
            "median": float(np.median(a)), "pos": int((a > 0).sum())}


thr_fixed = float(np.quantile(np.abs(rH[(day >= 0) & (day < W)]), 1 - NF_RATE))
print(f"[walk-forward] W={W} T={T} EMB={EMB} | fixed thr (fold0)={thr_fixed:.2f}bp | ndays={ndays}", flush=True)
print(f"{'fold':>4} {'test-days':>11} {'thr_ad':>7} | {'n':>3} {'adaptive':>9} {'fixed':>9}", flush=True)
pool_ad, pool_fx, recs = [], [], []
ts = W + EMB; fi = 0
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() < 50 or trn.sum() < 5000:
        ts += T; continue
    thr_ad = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE))
    na = train_eval(trn, tst, thr_ad); nf = train_eval(trn, tst, thr_fixed)
    pool_ad += na.tolist(); pool_fx += nf.tolist()
    sa, sf = stats(na), stats(nf)
    recs.append({"fold": fi, "test": [int(ts), int(te)], "thr_ad": thr_ad, "n": sa["n"],
                 "adaptive": sa.get("net"), "fixed": sf.get("net")})
    print(f"{fi:>4} [{ts:>3}-{te:>3})    {thr_ad:>6.1f} | {sa['n']:>3} "
          f"{sa.get('net', float('nan')):>+9.2f} {sf.get('net', float('nan')):>+9.2f}", flush=True)
    ts += T; fi += 1

PA, PF = stats(pool_ad), stats(pool_fx)
print(f"\n[POOLED OOS]  adaptive: n={PA['n']} net={PA['net']:+.2f}±{PA['se']:.1f}bp median={PA['median']:+.2f} pos={PA['pos']}/{PA['n']}", flush=True)
print(f"[POOLED OOS]  fixed   : n={PF['n']} net={PF['net']:+.2f}±{PF['se']:.1f}bp median={PF['median']:+.2f} pos={PF['pos']}/{PF['n']}", flush=True)
out = {"W": W, "T": T, "EMB": EMB, "thr_fixed": thr_fixed, "folds": recs,
       "pooled_adaptive": PA, "pooled_fixed": PF}
bk.blob(f"research_runs/maker_labels_rr/WALKFORWARD_ADAPTIVE.json").upload_from_string(json.dumps(out, default=float))
print("[saved] research_runs/maker_labels_rr/WALKFORWARD_ADAPTIVE.json", flush=True)
