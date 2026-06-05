#!/usr/bin/env python3
"""Hold-HORIZON comparison (15 / 30 / 60s) walk-forward, ZERO maker fee (USDC). DOGE.

For each hold horizon H: Model-A vol-gate on |rH_H|, Model-B maker-payoff for hold-H,
deploy 1 & 5 trades/day (score = pct_rank(pA)*pct_rank(|pB-0.5|)); net = GROSS (zero fee).
Reuses frozen A_DOGE/B_pool HP (same as the original WF). Reports per-horizon pooled net
bp/trade (+ SE + per-fold) and Model-A vol-AUC, to test 'shorter window -> stronger signal'.
Reads research_runs/maker_labels_h/DOGE.npz (cfgs = holds [15,30,60]s).
"""
import io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
W, T, EMB = 200, 30, 2; NF_RATE = 0.05; GATE_PCT = 5.0
HORIZONS = [("15s", 0, "rH15"), ("30s", 1, "rH30"), ("60s", 2, "rH60")]  # (label, cfg index, rH key)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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


def oof_pA(F, yA, trn, day, hpA, niter, k=4):
    tdays = sorted(set(day[trn].tolist())); fold = {dd: i % k for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], niter, F[trk], yA[trk], spw=spwk); oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


hpA = jload(f"{MAIN}/A_DOGE.json"); hpB = jload(f"{MAIN}/B_pool.json")
d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_h/DOGE.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float32); day = d["day"]
rHk = {"rH15": d["rH15"].astype(np.float64), "rH30": d["rH30"].astype(np.float64), "rH60": d["rH60"].astype(np.float64)}
PL = d["pnl_long"].astype(np.float64); PS = d["pnl_short"].astype(np.float64)   # (3,2,N)
FLa = d["fill_long"].astype(bool); FSa = d["fill_short"].astype(bool)            # (2,N)
fl = FLa[0]; fs = FSa[0]  # touch qm0

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)
print(f"[horizon WF | ZERO maker fee] {len(FOLDS)} folds, OOS={tot_days} days | N={len(F)}\n", flush=True)


def run_horizon(cfgidx, rhkey):
    rH = rHk[rhkey]; netl = PL[cfgidx, 0, :] * 100.0; nets = PS[cfgidx, 0, :] * 100.0   # ZERO fee = gross
    ev1, ev5, aucs = [], [], []
    for trn, tst in FOLDS:
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
        ti = np.where(tst)[0]; pA = bstA.predict(xgb.DMatrix(F[ti])); aucs.append(auc(yA[ti], pA))
        oof = oof_pA(F, yA, trn, day, hpA, hpA["best_iter"]); valid = trn & np.isfinite(oof)
        thr_oof = float(np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0)); gate = valid & (oof >= thr_oof)
        keep = gate & (fl | fs); nl = netl; ns = nets
        yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int); both = fl & fs
        wq = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
        wc = np.clip(wq[keep], 0, np.quantile(wq[keep][wq[keep] > 0], 0.99) if (wq[keep] > 0).any() else 1.0)
        bstB = fit(hpB["best_params"], hpB["best_iter"], F[keep], yB[keep], w=wc)
        pB = bstB.predict(xgb.DMatrix(F[ti]))
        score = pct_rank(pA) * pct_rank(np.abs(pB - 0.5)); dyt = day[ti]
        for bud, store in ((1, ev1), (5, ev5)):
            sel = daily_pick(dyt, score, bud); side = pB[sel] >= 0.5
            net = np.where(side, netl[ti][sel], nets[ti][sel]); fc = np.where(side, fl[ti][sel], fs[ti][sel])
            ex = fc & np.isfinite(net); store.append(net[ex])
    return ev1, ev5, aucs


def summ(pf):
    a = np.concatenate(pf) if pf else np.array([]); n = len(a)
    ev = float(a.mean()) if n else float("nan"); se = float(a.std() / max(np.sqrt(n), 1)) if n else 0
    return ev, se, n, [round(float(p.sum() * 0.01), 1) for p in pf]


print(f"{'horizon':>8} | {'A-AUC':>6} | {'1/day net bp/trd':>26} | {'5/day net bp/trd':>20}")
RES = {}
for hlabel, cfgidx, rhkey in HORIZONS:
    ev1, ev5, aucs = run_horizon(cfgidx, rhkey)
    e1, s1, n1, pf1 = summ(ev1); e5, s5, n5, pf5 = summ(ev5)
    print(f"{hlabel:>8} | {np.nanmean(aucs):.3f}  | {e1:+6.2f}+-{s1:.2f} (n{n1}) pf{pf1} | {e5:+6.2f}+-{s5:.2f} (n{n5})", flush=True)
    RES[hlabel] = {"A_AUC": float(np.nanmean(aucs)), "ev1_bp": e1, "se1": s1, "n1": n1, "perfold1": pf1,
                   "ev5_bp": e5, "se5": s5, "n5": n5}
bk.blob("research_runs/maker_labels_h/HORIZON_WF_RESULT.json").upload_from_string(json.dumps(RES, default=float))
print("\n[saved] research_runs/maker_labels_h/HORIZON_WF_RESULT.json  | net = GROSS (zero maker fee)", flush=True)
