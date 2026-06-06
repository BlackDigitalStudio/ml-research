#!/usr/bin/env python3
"""Per-SYMBOL noA baseline (the declared cell): B-full direction, blanket vol-norm, per-fold Optuna
tuned on SIZE-AWARE IC=corr(pB-0.5, netl-nets), causal-rolling deploy at target-10/day, ZERO fee.
noA only -> NO model A at all (no vol-gate, no apred). Each symbol gets its OWN weights & HP.
Saves per-test-trade (calendar_day, net_bp) for portfolio aggregation + per-symbol surface.
Usage: python3 subs60_xgb_sym_baseline.py SYM [nthread]   (SYM e.g. BTC, reads maker_labels_h/SYM.npz)
"""
import io, json, sys
import numpy as np
from google.cloud import storage
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYM = sys.argv[1]
NTHREAD = int(sys.argv[2]) if len(sys.argv) > 2 else 2
LABELSUB = sys.argv[3] if len(sys.argv) > 3 else "maker_labels_h"
QMIDX = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # entry queue-mult index (0=touch/front, 1=always-last)
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
W, T, EMB = 200, 30, 2; KDAYS = 30; KNORM = 20
SUBVAL_D = 30; N_TRIALS = 25; TUNE_SUB = 200000; TGT = 10
CFGIDX, RHKEY = 1, "rH30"
DAY_NS = 86_400_000_000_000
bk = storage.Client(project=PROJ).bucket(BUCKET)


def _space(t):
    return {"max_depth": t.suggest_int("max_depth", 3, 9),
            "eta": t.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True)}


def tune_B_ic(Xst, yst, wst, Xv, pdiff_v):
    if len(Xst) > TUNE_SUB:
        ix = np.random.RandomState(0).choice(len(Xst), TUNE_SUB, replace=False); Xst, yst, wst = Xst[ix], yst[ix], wst[ix]
    dst = xgb.DMatrix(Xst, label=yst, weight=wst); dv = xgb.DMatrix(Xv)
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0}

    def obj(t):
        hp = _space(t); nr = t.suggest_int("num_boost_round", 50, 400)
        b = xgb.train(dict(base, **hp), dst, num_boost_round=nr)
        pv = b.predict(dv) - 0.5
        if pv.std() < 1e-9:
            return -1.0
        ic = float(np.corrcoef(pv, pdiff_v)[0, 1])
        return ic if np.isfinite(ic) else -1.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=N_TRIALS)
    bp = dict(st.best_params); nr = bp.pop("num_boost_round"); return bp, int(nr), float(st.best_value)


def fith(hp, niter, X, y, w):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0}
    return xgb.train(dict(p, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def cdf_map(x, ref): return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def causal_rolling_trades(sc_tr, sc_te, day_tr, day_te, target_tpd, sideB, fl, fs, nl, ns, calday_te):
    """noA causal rolling deploy -> returns (net_bp, calendar_day) of executed trades."""
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
        return np.array([]), np.array([])
    side = sideB[sel]; net = np.where(side, nl[sel], ns[sel]); fc = np.where(side, fl[sel], fs[sel])
    ex = fc & np.isfinite(net); return net[ex], calday_te[sel][ex]


d = np.load(io.BytesIO(bk.blob(f"research_runs/{LABELSUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int); rH = d[RHKEY].astype(np.float64); ts = d["ts"].astype(np.int64)
calday = (ts // DAY_NS).astype(np.int64)
netl = d["pnl_long"][CFGIDX, QMIDX, :].astype(np.float64) * 100.0; nets = d["pnl_short"][CFGIDX, QMIDX, :].astype(np.float64) * 100.0
fl = d["fill_long"].astype(bool)[QMIDX]; fs = d["fill_short"].astype(bool)[QMIDX]; nfeat = F.shape[1]
day_mean = np.zeros((ndays, nfeat)); day_var = np.zeros((ndays, nfeat))
for dd in range(ndays):
    mk = day == dd
    if mk.sum() > 1:
        day_mean[dd] = F[mk].mean(0); day_var[dd] = F[mk].var(0)
gstd = F.std(0); mu_ref = np.zeros((ndays, nfeat)); sd_ref = np.zeros((ndays, nfeat))
for dd in range(ndays):
    sl = slice(max(0, dd - KNORM), dd) if dd > 0 else slice(0, 1)
    mu_ref[dd] = day_mean[sl].mean(0); sd_ref[dd] = np.sqrt(np.maximum(day_var[sl].mean(0), 0))
sd_ref = np.maximum(sd_ref, 0.2 * gstd[None, :] + 1e-9)
Fn = ((F - mu_ref[day]) / sd_ref[day]).astype(np.float32)
print(f"[{SYM} noA-IC baseline] Fn={Fn.shape} ndays={ndays} nthread={NTHREAD}", flush=True)

FOLDS = []; tsd = W + EMB
while tsd < ndays:
    te = min(tsd + T, ndays); trn = (day >= tsd - EMB - W) & (day < tsd - EMB); tst = (day >= tsd) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    tsd += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)

all_net, all_cd, perfold_pct, ic_val = [], [], [], []
for fi, (trn, tst) in enumerate(FOLDS):
    tr_days = sorted(set(day[trn].tolist())); sv = set(tr_days[-SUBVAL_D:])
    sub = trn & np.isin(day, list(tr_days[:-(SUBVAL_D + EMB)])); val = trn & np.isin(day, list(sv))
    yB = (np.where(fl, netl, -np.inf) > np.where(fs, nets, -np.inf)).astype(int); both = fl & fs
    wq = np.where(both, np.abs(netl - nets), np.where(fl, np.abs(netl), np.where(fs, np.abs(nets), 0.0)))
    wcl = lambda mk: np.clip(wq[mk], 0, np.quantile(wq[mk][wq[mk] > 0], 0.99) if (wq[mk] > 0).any() else 1.0)
    sb = sub & (fl | fs); vb = val & both
    hpB, biB, icB = tune_B_ic(Fn[sb], yB[sb], wcl(sb), Fn[vb], (netl[vb] - nets[vb]))
    gfull = trn & (fl | fs)
    Bf = fith(hpB, biB, Fn[gfull], yB[gfull], wcl(gfull))
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    pBf_tr = Bf.predict(xgb.DMatrix(Fn[tri])); pBf_te = Bf.predict(xgb.DMatrix(Fn[tei]))
    sBf = np.sort(np.abs(pBf_tr - 0.5))
    noa_tr = np.searchsorted(sBf, np.abs(pBf_tr - 0.5), "right") / len(sBf); noa_te = cdf_map(np.abs(pBf_te - 0.5), sBf)
    net, cd = causal_rolling_trades(noa_tr, noa_te, day[tri], day[tei], TGT, pBf_te >= 0.5, fl[tei], fs[tei], netl[tei], nets[tei], calday[tei])
    all_net.append(net); all_cd.append(cd); ic_val.append(round(icB, 4))
    perfold_pct.append(round(float(net.sum() * 0.01), 1))
    print(f"  fold{fi}: B IC(val)={icB:+.4f} (d{hpB['max_depth']},nr{biB}) | trades={len(net)} fold_pct={perfold_pct[-1]:+.1f}", flush=True)

NET = np.concatenate(all_net) if all_net else np.array([]); CD = np.concatenate(all_cd) if all_cd else np.array([])
n = len(NET); ev = float(NET.mean()) if n else float("nan"); std = float(NET.std()) if n else float("nan")
tpd = n / max(tot_days, 1); annS = (ev / std) * np.sqrt(tpd * 365.0) if std > 0 else float("nan")
hit = float((NET > 0).mean()) if n else float("nan")
summ = {"symbol": SYM, "n_trades": n, "tot_days": tot_days, "trd_per_day": round(tpd, 2), "ev_bp": round(ev, 3),
        "ann_sharpe": round(annS, 3), "hit_pct": round(100 * hit, 1), "perfold_pct": perfold_pct, "ic_val": ic_val, "n_folds": len(FOLDS)}
buf = io.BytesIO(); np.savez_compressed(buf, net=NET, calday=CD, perfold_pct=np.array(perfold_pct), ic_val=np.array(ic_val))
TAG = f"_qm{QMIDX}" if QMIDX else ""
bk.blob(f"research_runs/{LABELSUB}/portfolio/{SYM}_noA_t10{TAG}.npz").upload_from_string(buf.getvalue())
bk.blob(f"research_runs/{LABELSUB}/portfolio/{SYM}_noA_t10{TAG}.json").upload_from_string(json.dumps(summ, default=float))
print(f"\n[{SYM}] DONE trd/d={tpd:.1f} EV={ev:+.2f}bp annS={annS:+.2f} hit={100*hit:.1f}% perfold={perfold_pct} IC={ic_val}", flush=True)
print(f"[saved] portfolio/{SYM}_noA_t10.{{npz,json}}", flush=True)
