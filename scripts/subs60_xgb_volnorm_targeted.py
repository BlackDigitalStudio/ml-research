#!/usr/bin/env python3
"""TARGETED vol-normalization: causal trailing per-feature z-score applied ONLY to the
vol-SCALED features (data-driven: a feature is vol-scaled iff its per-day std correlates with
the day's realized vol, corr(day_std_j, day_vol) > THR). Dimensionless features (OBI ratios,
ToD sin/cos) are left RAW -> preserve their signal while removing the vol covariate shift from
the magnitude features. Walk-forward 30s, ZERO fee, causal rolling, noA & AxB, 5/10 trades/day.
Compare to blanket-norm (annS +2.42) and non-norm (annS +5.39, f2-spike). Reads maker_labels_h/DOGE.npz.
"""
import io, json, sys
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
W, T, EMB = 200, 30, 2; NF_RATE = 0.05; GATE_PCT = 5.0; KDAYS = 30; KNORM = 20
CFGIDX, RHKEY = 1, "rH30"; BUDGETS = [5, 10]
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
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
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl[sel], ns[sel]); fc = np.where(side, fl[sel], fs[sel])
    ex = fc & np.isfinite(net); return net[ex]


hpA = jload(f"{MAIN}/A_DOGE.json"); hpB = jload(f"{MAIN}/B_pool.json")
d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_h/DOGE.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int)
rH = d[RHKEY].astype(np.float64)
netl = d["pnl_long"][CFGIDX, 0, :].astype(np.float64) * 100.0; nets = d["pnl_short"][CFGIDX, 0, :].astype(np.float64) * 100.0
fl = d["fill_long"].astype(bool)[0]; fs = d["fill_short"].astype(bool)[0]
nfeat = F.shape[1]

# ---- per-day stats + day vol ----
day_mean = np.zeros((ndays, nfeat)); day_var = np.zeros((ndays, nfeat)); day_vol = np.zeros(ndays)
for dd in range(ndays):
    mk = day == dd
    if mk.sum() > 1:
        sub = F[mk]; day_mean[dd] = sub.mean(0); day_var[dd] = sub.var(0)
        day_vol[dd] = np.quantile(np.abs(rH[mk]), 0.95)
day_std = np.sqrt(day_var); gstd = F.std(0)
# data-driven vol-scaled mask: corr(day_std_j, day_vol)
valid_d = day_vol > 0
corr = np.array([np.corrcoef(day_std[valid_d, j], day_vol[valid_d])[0, 1] if day_std[valid_d, j].std() > 0 else 0.0
                 for j in range(nfeat)])
corr = np.nan_to_num(corr)
vol_mask = corr > THR
print(f"[targeted vol-norm | THR={THR}] vol-scaled feats normalized: {int(vol_mask.sum())}/{nfeat} "
      f"| left raw: {int((~vol_mask).sum())}", flush=True)
print(f"  corr quartiles: {np.round(np.percentile(corr, [10,25,50,75,90]),2).tolist()} | "
      f"normalized idx (first 25): {np.where(vol_mask)[0][:25].tolist()}", flush=True)
# ---- causal trailing z-score, applied only to vol_mask cols ----
mu_ref = np.zeros((ndays, nfeat)); sd_ref = np.zeros((ndays, nfeat))
for dd in range(ndays):
    sl = slice(max(0, dd - KNORM), dd) if dd > 0 else slice(0, 1)
    mu_ref[dd] = day_mean[sl].mean(0); sd_ref[dd] = np.sqrt(np.maximum(day_var[sl].mean(0), 0))
sd_ref = np.maximum(sd_ref, 0.2 * gstd[None, :] + 1e-9)
Fn = F.astype(np.float32).copy()
z = ((F - mu_ref[day]) / sd_ref[day]).astype(np.float32)
Fn[:, vol_mask] = z[:, vol_mask]
print(f"[done] Fn shape={Fn.shape}", flush=True)

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)


def metrics(pf):
    a = np.concatenate(pf) if pf else np.array([]); n = len(a)
    if not n:
        return dict(n=0, tpd=0, ev=float("nan"), sharpe=float("nan"), ann=float("nan"), hit=float("nan"), tot=float("nan"), perfold=[])
    ev = float(a.mean()); std = float(a.std()); tpd = n / max(tot_days, 1); sh = ev / std if std > 0 else 0.0
    return dict(n=n, tpd=tpd, ev=ev, sharpe=sh, ann=sh * np.sqrt(tpd * 365.0), hit=float((a > 0).mean()),
                tot=ev * tpd, perfold=[round(float(p.sum() * 0.01), 1) for p in pf])


perfold = []
for trn, tst in FOLDS:
    thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
    spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
    A = fit(hpA["best_params"], hpA["best_iter"], Fn[trn], yA[trn], spw=spw)
    oof = oof_pA(Fn, yA, trn, day, hpA); valid = trn & np.isfinite(oof)
    gate = valid & (oof >= np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0))
    Bg = trainB(Fn, fl, fs, netl, nets, gate, hpB); Bf = trainB(Fn, fl, fs, netl, nets, trn, hpB)
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    pA_tr = A.predict(xgb.DMatrix(Fn[tri])); pA_te = A.predict(xgb.DMatrix(Fn[tei]))
    pBg_tr = Bg.predict(xgb.DMatrix(Fn[tri])); pBg_te = Bg.predict(xgb.DMatrix(Fn[tei]))
    pBf_tr = Bf.predict(xgb.DMatrix(Fn[tri])); pBf_te = Bf.predict(xgb.DMatrix(Fn[tei]))
    sA = np.sort(pA_tr); sBg = np.sort(np.abs(pBg_tr - 0.5)); sBf = np.sort(np.abs(pBf_tr - 0.5))
    axb_tr = (np.searchsorted(sA, pA_tr, "right") / len(sA)) * (np.searchsorted(sBg, np.abs(pBg_tr - 0.5), "right") / len(sBg))
    axb_te = cdf_map(pA_te, sA) * cdf_map(np.abs(pBg_te - 0.5), sBg)
    noa_tr = np.searchsorted(sBf, np.abs(pBf_tr - 0.5), "right") / len(sBf)
    noa_te = cdf_map(np.abs(pBf_te - 0.5), sBf)
    perfold.append((axb_tr, axb_te, noa_tr, noa_te, day[tri], day[tei], pBg_te >= 0.5, pBf_te >= 0.5, fl[tei], fs[tei], netl[tei], nets[tei]))

print(f"\n  {'pol':>4} {'tgt':>4} {'trd/d':>6} {'EV/trd':>8} {'Shrp':>6} {'annS':>6} {'hit%':>6} {'tot/d':>7}  per-fold", flush=True)
RES = {"THR": THR, "n_normalized": int(vol_mask.sum()), "cells": {}}
for tgt in BUDGETS:
    for pol in ("AxB", "noA"):
        pf = []
        for (axb_tr, axb_te, noa_tr, noa_te, dtr, dte, sBg, sBf, flt, fst, nlt, nst) in perfold:
            a = (axb_tr, axb_te, sBg) if pol == "AxB" else (noa_tr, noa_te, sBf)
            pf.append(causal_rolling(a[0], a[1], dtr, dte, tgt, a[2], flt, fst, nlt, nst))
        x = metrics(pf)
        print(f"  {pol:>4} {tgt:>4} {x['tpd']:>6.1f} {x['ev']:>+7.2f} {x['sharpe']:>+6.3f} {x['ann']:>+6.2f} "
              f"{100*x['hit']:>5.1f} {x['tot']:>+7.2f}  {x['perfold']}", flush=True)
        RES["cells"][f"{pol}_t{tgt}"] = x
bk.blob("research_runs/maker_labels_h/VOLNORM_TARGETED_RESULT.json").upload_from_string(json.dumps(RES, default=float))
print("\n[saved] VOLNORM_TARGETED_RESULT.json | blanket noA_t5 annS+2.42 ; non-norm annS+5.39 (f2-spike)", flush=True)
