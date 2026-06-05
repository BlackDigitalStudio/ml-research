#!/usr/bin/env python3
"""CAUSAL (realistic, NOT post-hoc) zero-fee baseline: AxB vs noA at 5 & 10 trades/day,
horizons 30s & 60s, DOGE. Rolling-adaptive deploy threshold (recalibrated daily to target
rate from a trailing buffer -> regime-adaptive, deployable). net = GROSS (zero maker fee).

Per fold: train A (vol on |rH_H|) + B-gated (apred-5%) + B-full (no gate); predict train+test;
CACHE per-horizon preds (re-sweep deploy for free). Reports per (horizon,policy,budget):
actual trd/day, net bp/trade (+SE), total bp/day, per-fold. Best -> baseline.
Reads research_runs/maker_labels_h/DOGE.npz. EV is the metric (AUC is size-blind).
"""
import io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
W, T, EMB = 200, 30, 2; NF_RATE = 0.05; GATE_PCT = 5.0; KDAYS = 30
HORIZONS = [("30s", 1, "rH30"), ("60s", 2, "rH60")]
BUDGETS = [5, 10]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def fit(hp, niter, X, y, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def oof_pA(F, yA, trn, day, hpA, k=4):
    tdays = sorted(set(day[trn].tolist())); fold = {dd: i % k for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk); oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


def trainB(F_, fl, fs, netl, nets, mask, hpB):
    nl = netl; ns = nets
    yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int); both = fl & fs
    wq = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
    keep = mask & (fl | fs); pos = wq[keep][wq[keep] > 0]
    wc = np.clip(wq[keep], 0, np.quantile(pos, 0.99) if len(pos) else 1.0)
    return fit(hpB["best_params"], hpB["best_iter"], F_[keep], yB[keep], w=wc)


def cdf_map(x, ref): return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def causal_rolling(sc_tr, sc_te, day_tr, day_te, target_tpd, sideB, fl, fs, nl, ns):
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - target_tpd / max(wpd, 1.0))
    tr_days = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, tr_days[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist())
        buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl[sel], ns[sel]); fc = np.where(side, fl[sel], fs[sel])
    ex = fc & np.isfinite(net); return net[ex]


hpA = jload(f"{MAIN}/A_DOGE.json"); hpB = jload(f"{MAIN}/B_pool.json")
d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_h/DOGE.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float32); day = d["day"]
rHk = {"rH30": d["rH30"].astype(np.float64), "rH60": d["rH60"].astype(np.float64)}
PL = d["pnl_long"].astype(np.float64); PS = d["pnl_short"].astype(np.float64)
fl = d["fill_long"].astype(bool)[0]; fs = d["fill_short"].astype(bool)[0]

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)
print(f"[CAUSAL baseline | ZERO fee | rolling deploy] {len(FOLDS)} folds OOS={tot_days}d", flush=True)


def summ(pf):
    a = np.concatenate(pf) if pf else np.array([]); n = len(a)
    ev = float(a.mean()) if n else float("nan"); se = float(a.std() / max(np.sqrt(n), 1)) if n else 0
    tpd = n / max(tot_days, 1); return ev, se, tpd, ev * tpd, [round(float(p.sum() * 0.01), 1) for p in pf]


RES = {}
for hlabel, cfgidx, rhkey in HORIZONS:
    rH = rHk[rhkey]; netl = PL[cfgidx, 0, :] * 100.0; nets = PS[cfgidx, 0, :] * 100.0   # zero fee
    perfold = []  # (sc_axb_tr, sc_axb_te, sc_noa_tr, sc_noa_te, day_tr, day_te, sB_te, fl_te, fs_te, nl_te, ns_te)
    for trn, tst in FOLDS:
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        A = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
        oof = oof_pA(F, yA, trn, day, hpA); valid = trn & np.isfinite(oof)
        gate = valid & (oof >= np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0))
        Bg = trainB(F, fl, fs, netl, nets, gate, hpB); Bf = trainB(F, fl, fs, netl, nets, trn, hpB)
        tri = np.where(trn)[0]; tei = np.where(tst)[0]
        pA_tr = A.predict(xgb.DMatrix(F[tri])); pA_te = A.predict(xgb.DMatrix(F[tei]))
        pBg_tr = Bg.predict(xgb.DMatrix(F[tri])); pBg_te = Bg.predict(xgb.DMatrix(F[tei]))
        pBf_tr = Bf.predict(xgb.DMatrix(F[tri])); pBf_te = Bf.predict(xgb.DMatrix(F[tei]))
        # scores (causal: train cdf maps)
        sA = np.sort(pA_tr); sBg = np.sort(np.abs(pBg_tr - 0.5)); sBf = np.sort(np.abs(pBf_tr - 0.5))
        axb_tr = (np.searchsorted(sA, pA_tr, "right") / len(sA)) * (np.searchsorted(sBg, np.abs(pBg_tr - 0.5), "right") / len(sBg))
        axb_te = cdf_map(pA_te, sA) * cdf_map(np.abs(pBg_te - 0.5), sBg)
        noa_tr = np.searchsorted(sBf, np.abs(pBf_tr - 0.5), "right") / len(sBf)
        noa_te = cdf_map(np.abs(pBf_te - 0.5), sBf)
        perfold.append((axb_tr, axb_te, noa_tr, noa_te, day[tri], day[tei], pBg_te >= 0.5, pBf_te >= 0.5,
                        fl[tei], fs[tei], netl[tei], nets[tei]))
    print(f"\n=== {hlabel} | net=GROSS(zero fee) | rolling causal ===", flush=True)
    print(f"  {'policy':>10} {'tgt':>4} {'trd/day':>8} {'net bp/trd':>11} {'tot bp/day':>11}  per-fold OOS%")
    for tgt in BUDGETS:
        for pol in ("AxB", "noA"):
            pf = []
            for (axb_tr, axb_te, noa_tr, noa_te, dtr, dte, sBg, sBf, flt, fst, nlt, nst) in perfold:
                if pol == "AxB":
                    pf.append(causal_rolling(axb_tr, axb_te, dtr, dte, tgt, sBg, flt, fst, nlt, nst))
                else:
                    pf.append(causal_rolling(noa_tr, noa_te, dtr, dte, tgt, sBf, flt, fst, nlt, nst))
            ev, se, tpd, tot, perf = summ(pf)
            print(f"  {pol:>10} {tgt:>4} {tpd:>8.1f} {ev:>+10.2f}+-{se:.2f} {tot:>+10.2f}   {perf}", flush=True)
            RES[f"{hlabel}_{pol}_tgt{tgt}"] = {"trd_day": tpd, "net_bp_trd": ev, "se": se, "tot_bp_day": tot, "perfold": perf}
bk.blob("research_runs/maker_labels_h/BASELINE_CAUSAL_RESULT.json").upload_from_string(json.dumps(RES, default=float))
print("\n[saved] research_runs/maker_labels_h/BASELINE_CAUSAL_RESULT.json", flush=True)
