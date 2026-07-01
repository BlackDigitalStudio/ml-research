#!/usr/bin/env python3
"""Train DEPLOYABLE single A+B models for live (NOT per-fold). Config = AxB on the honest
always-last pegged-exit labels (maker_labels_pegexit_qm1), 30s. Produces the deploy bundle for one
symbol: A (vol-gate), Bg (direction on the A-OOF-top5% gate, for AxB), Bf (direction on full, for
noA), the blanket vol-norm state (so live can normalize new days), the train-CDF references (for the
cdf-rank deploy score), and the causal-rolling threshold seed. HP tuned on a recent sub-val holdout,
FINAL models fit on ALL history. Saves to research_runs/deploy/{SYM}/.
Usage: python3 subs60_train_deploy.py SYM [nthread]   (reads maker_labels_pegexit_qm1/{SYM}.npz)
"""
import io, json, sys, os
import numpy as np
from google.cloud import storage
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
NTHREAD = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LABELSUB = os.environ.get("LABELSUB", "maker_labels_pegexit_qm1")
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
NF_RATE = 0.05; GATE_PCT = 5.0; KNORM = 20; KDAYS = 30; SUBVAL_D = 30
N_TRIALS = 20; TUNE_SUB = 200000; CFGIDX, QMIDX, RHKEY = 1, 0, "rH30"
bk = storage.Client(project=PROJ).bucket(BUCKET)


def _space(t):
    return {"max_depth": t.suggest_int("max_depth", 3, 9), "eta": t.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True), "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True)}


def sub(X, y, w=None):
    if len(X) > TUNE_SUB:
        ix = np.random.RandomState(0).choice(len(X), TUNE_SUB, replace=False); return X[ix], y[ix], (w[ix] if w is not None else None)
    return X, y, w


def tuneA(Xs, ys, Xv, yv, spw):
    Xs, ys, _ = sub(Xs, ys); dst = xgb.DMatrix(Xs, label=ys); dv = xgb.DMatrix(Xv, label=yv)
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0, "eval_metric": "auc", "scale_pos_weight": spw}

    def obj(t):
        b = xgb.train(dict(base, **_space(t)), dst, num_boost_round=400, evals=[(dv, "v")], early_stopping_rounds=30, verbose_eval=False)
        t.set_user_attr("bi", int(b.best_iteration)); return float(b.best_score)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=N_TRIALS)
    return st.best_params, int(st.best_trial.user_attrs["bi"]), float(st.best_value)


def tuneB(Xs, ys, ws, Xv, pdiff):
    Xs, ys, ws = sub(Xs, ys, ws); dst = xgb.DMatrix(Xs, label=ys, weight=ws); dv = xgb.DMatrix(Xv)
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0}

    def obj(t):
        hp = _space(t); nr = t.suggest_int("num_boost_round", 50, 400)
        b = xgb.train(dict(base, **hp), dst, num_boost_round=nr); pv = b.predict(dv) - 0.5
        if pv.std() < 1e-9:
            return -1.0
        ic = float(np.corrcoef(pv, pdiff)[0, 1]); return ic if np.isfinite(ic) else -1.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=N_TRIALS)
    bp = dict(st.best_params); nr = bp.pop("num_boost_round"); return bp, int(nr), float(st.best_value)


def fitA(hp, ni, X, y, spw):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0, "scale_pos_weight": spw}
    return xgb.train(dict(p, **hp), xgb.DMatrix(X, label=y), num_boost_round=max(1, ni + 1))


def fitB(hp, ni, X, y, w):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0}
    return xgb.train(dict(p, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, ni + 1))


def oofA(F, yA, hpA, biA, day, k=4):
    tdays = sorted(set(day.tolist())); fold = {dd: i % k for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        tr = fday != kk; va = fday == kk
        if va.sum() < 50 or (yA[tr] == 1).sum() < 20:
            continue
        spw = float((yA[tr] == 0).sum() / max((yA[tr] == 1).sum(), 1))
        b = fitA(hpA, biA, F[tr], yA[tr], spw); oof[np.where(va)[0]] = b.predict(xgb.DMatrix(F[va]))
    return oof


d = np.load(io.BytesIO(bk.blob(f"research_runs/{LABELSUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int); rH = d[RHKEY].astype(np.float64)
netl = d["pnl_long"][CFGIDX, QMIDX, :].astype(np.float64) * 100.0; nets = d["pnl_short"][CFGIDX, QMIDX, :].astype(np.float64) * 100.0
fl = d["fill_long"].astype(bool)[QMIDX]; fs = d["fill_short"].astype(bool)[QMIDX]
feat_names = [str(x) for x in d["feat_names"]]; nfeat = F.shape[1]
# blanket causal vol-norm + SAVE the state (live rolls day_mean/day_var forward)
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
print(f"[{SYM} train-deploy | {LABELSUB} | AxB] Fn={Fn.shape} ndays={ndays}", flush=True)

# sub-val = last SUBVAL_D days (HP tuning only); FINAL models fit on ALL
tdays = sorted(set(day.tolist())); sv = set(tdays[-SUBVAL_D:])
subm = ~np.isin(day, list(sv)) & ~np.isin(day, tdays[-(SUBVAL_D + 2):-SUBVAL_D]); valm = np.isin(day, list(sv))

thr = float(np.quantile(np.abs(rH), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
spwA = float((yA[subm] == 0).sum() / max((yA[subm] == 1).sum(), 1))
hpA, biA, aucA = tuneA(Fn[subm], yA[subm], Fn[valm], yA[valm], spwA)
spwAf = float((yA == 0).sum() / max((yA == 1).sum(), 1)); A = fitA(hpA, biA, Fn, yA, spwAf)
print(f"  A: val-AUC={aucA:.3f} (d{hpA['max_depth']},bi{biA})", flush=True)

oof = oofA(Fn, yA, hpA, biA, day); valid = np.isfinite(oof)
gate = valid & (oof >= np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0))
yB = (np.where(fl, netl, -np.inf) > np.where(fs, nets, -np.inf)).astype(int); both = fl & fs
wq = np.where(both, np.abs(netl - nets), np.where(fl, np.abs(netl), np.where(fs, np.abs(nets), 0.0)))
wcl = lambda mk: np.clip(wq[mk], 0, np.quantile(wq[mk][wq[mk] > 0], 0.99) if (wq[mk] > 0).any() else 1.0)
sbm = subm & (fl | fs); vbm = valm & both
hpB, biB, icB = tuneB(Fn[sbm], yB[sbm], wcl(sbm), Fn[vbm], (netl[vbm] - nets[vbm]))
gfull = (fl | fs); ggate = gate & (fl | fs)
Bg = fitB(hpB, biB, Fn[ggate], yB[ggate], wcl(ggate)); Bf = fitB(hpB, biB, Fn[gfull], yB[gfull], wcl(gfull))
print(f"  B: val-IC={icB:+.4f} (d{hpB['max_depth']},nr{biB}) | gate n={int(ggate.sum())} full n={int(gfull.sum())}", flush=True)

# train-CDF refs + rolling-threshold seed (last KDAYS of deploy scores)
pA = A.predict(xgb.DMatrix(Fn)); pBg = Bg.predict(xgb.DMatrix(Fn)); pBf = Bf.predict(xgb.DMatrix(Fn))
sA = np.sort(pA).astype(np.float32); sBg = np.sort(np.abs(pBg - 0.5)).astype(np.float32); sBf = np.sort(np.abs(pBf - 0.5)).astype(np.float32)
cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)
axb_sc = (cdf(pA, sA) * cdf(np.abs(pBg - 0.5), sBg)).astype(np.float32)
noa_sc = cdf(np.abs(pBf - 0.5), sBf).astype(np.float32)
seedm = np.isin(day, tdays[-KDAYS:])

OUT = "research_runs/" + os.environ.get('DEPLOY_DIR', 'deploy') + f"/{SYM}"
for nm, mdl in [("A", A), ("Bg", Bg), ("Bf", Bf)]:
    p = f"/tmp/_{SYM}_{nm}.json"; mdl.save_model(p)
    bk.blob(f"{OUT}/{nm}.json").upload_from_filename(p)
nb = io.BytesIO(); np.savez_compressed(nb, day_mean=day_mean.astype(np.float32), day_var=day_var.astype(np.float32),
                                       gstd=gstd.astype(np.float32), sA=sA, sBg=sBg, sBf=sBf,
                                       axb_seed=axb_sc[seedm], noa_seed=noa_sc[seedm])
bk.blob(f"{OUT}/refs.npz").upload_from_string(nb.getvalue())
meta = {"symbol": SYM, "labels": LABELSUB, "cfgidx": CFGIDX, "qmidx": QMIDX, "horizon_s": 30, "config": "AxB (t5 argmax)",
        "KNORM": KNORM, "KDAYS": KDAYS, "NF_RATE": NF_RATE, "GATE_PCT": GATE_PCT, "vol_thr_p95_rH30": thr,
        "hpA": hpA, "biA": biA, "valAUC_A": aucA, "hpB": hpB, "biB": biB, "valIC_B": icB,
        "feat_names": feat_names, "ndays": ndays, "n_train": int(len(F)),
        "deploy": "score AxB = cdf(pA)*cdf(|pBg-0.5|), side=sign(pBg-0.5); noA = cdf(|pBf-0.5|); causal-rolling tau from refs seed, target N/day; features blanket causal-vol-normed (roll day_mean/day_var)",
        "WARNING": "OOS edge is CONDITIONAL (see HD3 rev5): AxB t5 DOGE annS~+3.4 under maker-SIM always-last fills; live needs feature-parity + execution validation before real money."}
bk.blob(f"{OUT}/meta.json").upload_from_string(json.dumps(meta, default=float))
print(f"\n[saved] gs://{BUCKET}/{OUT}/ : A.json Bg.json Bf.json refs.npz meta.json", flush=True)
print(f"  feat order saved ({nfeat} feats) for live parity; vol-norm state (day_mean/day_var) for live rolling.", flush=True)
