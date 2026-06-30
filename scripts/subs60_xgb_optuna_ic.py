#!/usr/bin/env python3
"""Per-fold Optuna on blanket vol-norm feats, but tune B (direction) by a SIZE-AWARE metric:
IC = corr(pB-0.5, netl-nets) on the sub-val filled windows (not AUC, which is size-blind and
collapsed to ~0.52). IC is magnitude-weighted AND computed on ~170k windows (stable -> less
val-overfit than AUC/EV). A tuned on AUC (vol well-predicted). DOGE 30s, ZERO fee, causal rolling
deploy noA & AxB at 5/10. Compare to frozen-HP (noA annS +2.42) and AUC-Optuna (noA +0.30).
Reads research_runs/maker_labels_h/DOGE.npz.
"""
import io, json, sys, os
import numpy as np
from google.cloud import storage
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
LABELSUB = sys.argv[2] if len(sys.argv) > 2 else "maker_labels_h"
QMIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NTHREAD = int(sys.argv[4]) if len(sys.argv) > 4 else 8
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
W, T, EMB = 200, 30, 2; NF_RATE = 0.05; GATE_PCT = 5.0; KDAYS = 30; KNORM = 20
SUBVAL_D = 30; N_TRIALS = int(os.environ.get("N_TRIALS", "25")); TUNE_SUB = 200000; SEED = int(os.environ.get("SEED", "0"))
CFGIDX, RHKEY = 1, "rH30"; BUDGETS = [5, 10]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def _space(t):
    return {"max_depth": t.suggest_int("max_depth", 3, 9),
            "eta": t.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True)}


def subsample(X, y, w=None):
    if len(X) > TUNE_SUB:
        ix = np.random.RandomState(SEED).choice(len(X), TUNE_SUB, replace=False)
        return X[ix], y[ix], (w[ix] if w is not None else None)
    return X, y, w


def tune_A(Xst, yst, Xv, yv, spw):
    Xst, yst, _ = subsample(Xst, yst)
    dst = xgb.DMatrix(Xst, label=yst); dv = xgb.DMatrix(Xv, label=yv)
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": SEED, "eval_metric": "auc", "scale_pos_weight": spw}

    def obj(t):
        b = xgb.train(dict(base, **_space(t)), dst, num_boost_round=400, evals=[(dv, "v")], early_stopping_rounds=30, verbose_eval=False)
        t.set_user_attr("bi", int(b.best_iteration)); return float(b.best_score)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj, n_trials=N_TRIALS)
    return st.best_params, int(st.best_trial.user_attrs["bi"]), float(st.best_value)


def tune_B_ic(Xst, yst, wst, Xv, pdiff_v):
    Xst, yst, wst = subsample(Xst, yst, wst)
    dst = xgb.DMatrix(Xst, label=yst, weight=wst); dv = xgb.DMatrix(Xv)
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": SEED}

    def obj(t):
        hp = _space(t); nr = t.suggest_int("num_boost_round", 50, 400)
        b = xgb.train(dict(base, **hp), dst, num_boost_round=nr)
        pv = b.predict(dv) - 0.5
        if pv.std() < 1e-9:
            return -1.0
        ic = float(np.corrcoef(pv, pdiff_v)[0, 1])
        return ic if np.isfinite(ic) else -1.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj, n_trials=N_TRIALS)
    bp = dict(st.best_params); nr = bp.pop("num_boost_round"); return bp, int(nr), float(st.best_value)


def fith(hp, niter, X, y, w=None, spw=None):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": SEED}
    if spw is not None:
        p["scale_pos_weight"] = spw
    return xgb.train(dict(p, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def oof_pA(F, yA, trn, day, hp, niter, k=4):
    tdays = sorted(set(day[trn].tolist())); fold = {dd: i % k for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fith(hp, niter, F[trk], yA[trk], spw=spwk); oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


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


d = np.load(io.BytesIO(bk.blob(f"research_runs/{LABELSUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int); rH = d[RHKEY].astype(np.float64)
_drop = os.environ.get("DROP_COLS", "")
if _drop:
    dc = sorted(int(x) for x in _drop.split(","))
    keep = [i for i in range(F.shape[1]) if i not in dc]
    F = F[:, keep]; print(f"*** DROPPED cols {dc} -> F now {F.shape[1]} cols ***", flush=True)
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
print(f"[{SYM} {LABELSUB} qm{QMIDX} | Optuna A-AUC + B-IC, AxB&noA | trials={N_TRIALS}] Fn={Fn.shape}", flush=True)

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays); trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)


def metrics(pf):
    a = np.concatenate(pf) if pf else np.array([]); n = len(a)
    if not n:
        return dict(tpd=0, ev=float("nan"), ann=float("nan"), hit=float("nan"), tot=float("nan"), perfold=[])
    ev = float(a.mean()); std = float(a.std()); tpd = n / max(tot_days, 1); sh = ev / std if std > 0 else 0.0
    return dict(tpd=tpd, ev=ev, ann=sh * np.sqrt(tpd * 365.0), hit=float((a > 0).mean()), tot=ev * tpd,
               perfold=[round(float(p.sum() * 0.01), 1) for p in pf])


perfold = []
for fi, (trn, tst) in enumerate(FOLDS):
    dtr = day[trn]; tr_days = sorted(set(dtr.tolist())); sv = set(tr_days[-SUBVAL_D:])
    sub = trn & np.isin(day, list(tr_days[:-(SUBVAL_D + EMB)])); val = trn & np.isin(day, list(sv))
    thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
    spwA = float((yA[sub] == 0).sum() / max((yA[sub] == 1).sum(), 1))
    hpA, biA, aucA = tune_A(Fn[sub], yA[sub], Fn[val], yA[val], spwA)
    spwAf = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
    A = fith(hpA, biA, Fn[trn], yA[trn], spw=spwAf)
    oof = oof_pA(Fn, yA, trn, day, hpA, biA); valid = trn & np.isfinite(oof)
    gate = valid & (oof >= np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0))
    yB = (np.where(fl, netl, -np.inf) > np.where(fs, nets, -np.inf)).astype(int); both = fl & fs
    wq = np.where(both, np.abs(netl - nets), np.where(fl, np.abs(netl), np.where(fs, np.abs(nets), 0.0)))
    wcl = lambda mk: np.clip(wq[mk], 0, np.quantile(wq[mk][wq[mk] > 0], 0.99) if (wq[mk] > 0).any() else 1.0)
    sb = sub & (fl | fs); vb = val & both   # IC target on both-filled sub-val
    hpB, biB, icB = tune_B_ic(Fn[sb], yB[sb], wcl(sb), Fn[vb], (netl[vb] - nets[vb]))
    gfull = trn & (fl | fs); ggate = gate & (fl | fs)
    Bf = fith(hpB, biB, Fn[gfull], yB[gfull], w=wcl(gfull)); Bg = fith(hpB, biB, Fn[ggate], yB[ggate], w=wcl(ggate))
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    pA_tr = A.predict(xgb.DMatrix(Fn[tri])); pA_te = A.predict(xgb.DMatrix(Fn[tei]))
    pBg_tr = Bg.predict(xgb.DMatrix(Fn[tri])); pBg_te = Bg.predict(xgb.DMatrix(Fn[tei]))
    pBf_tr = Bf.predict(xgb.DMatrix(Fn[tri])); pBf_te = Bf.predict(xgb.DMatrix(Fn[tei]))
    sA = np.sort(pA_tr); sBg = np.sort(np.abs(pBg_tr - 0.5)); sBf = np.sort(np.abs(pBf_tr - 0.5))
    axb_tr = (np.searchsorted(sA, pA_tr, "right") / len(sA)) * (np.searchsorted(sBg, np.abs(pBg_tr - 0.5), "right") / len(sBg))
    axb_te = cdf_map(pA_te, sA) * cdf_map(np.abs(pBg_te - 0.5), sBg)
    noa_tr = np.searchsorted(sBf, np.abs(pBf_tr - 0.5), "right") / len(sBf); noa_te = cdf_map(np.abs(pBf_te - 0.5), sBf)
    perfold.append((axb_tr, axb_te, noa_tr, noa_te, day[tri], day[tei], pBg_te >= 0.5, pBf_te >= 0.5, fl[tei], fs[tei], netl[tei], nets[tei]))
    print(f"  fold{fi}: A AUC={aucA:.3f} | B IC(val)={icB:+.4f} (d{hpB['max_depth']},nr{biB})", flush=True)

print(f"\n  {'pol':>4} {'tgt':>4} {'trd/d':>6} {'EV/trd':>8} {'annS':>6} {'hit%':>6} {'tot/d':>7}  per-fold", flush=True)
RES = {}
for tgt in BUDGETS:
    for pol in ("AxB", "noA"):
        pf = []
        for (axb_tr, axb_te, noa_tr, noa_te, dt, de, sBg, sBf, flt, fst, nlt, nst) in perfold:
            a = (axb_tr, axb_te, sBg) if pol == "AxB" else (noa_tr, noa_te, sBf)
            pf.append(causal_rolling(a[0], a[1], dt, de, tgt, a[2], flt, fst, nlt, nst))
        x = metrics(pf)
        print(f"  {pol:>4} {tgt:>4} {x['tpd']:>6.1f} {x['ev']:>+7.2f} {x['ann']:>+6.2f} {100*x['hit']:>5.1f} {x['tot']:>+7.2f}  {x['perfold']}", flush=True)
        RES[f"{pol}_t{tgt}"] = x
bk.blob(f"research_runs/{LABELSUB}/OPTUNA_IC_{SYM}_qm{QMIDX}.json").upload_from_string(json.dumps(RES, default=float))
print(f"\n[saved] {LABELSUB}/OPTUNA_IC_{SYM}_qm{QMIDX}.json (AxB & noA at 5/10)", flush=True)
