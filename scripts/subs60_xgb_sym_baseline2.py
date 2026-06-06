#!/usr/bin/env python3
"""Per-symbol noA baseline (B-full, IC-tuned, vol-norm) with 3 SELECTIVITY schemes compared
(10/day is NOT forced -- profitable symbols may trade more):
  rolling   = current 30d train-seeded buffer, target-10 (reference)
  frozen    = confidence threshold FROZEN at train (train-10/day bar); test rate FLOATS
  fastadapt = 1-day-lagged daily quantile (today's bar = yesterday's distribution), adapts fast
Saves per-fold TEST preds + train conf to portfolio2/{SYM}_preds.npz (future deploy sweeps = free).
Metric = DAILY-series annualized Sharpe (honest) + total bp. Each symbol = own weights/HP.
Usage: python3 subs60_xgb_sym_baseline2.py SYM [nthread]
"""
import io, json, sys
import numpy as np
from google.cloud import storage
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYM = sys.argv[1]; NTHREAD = int(sys.argv[2]) if len(sys.argv) > 2 else 2
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
W, T, EMB, KDAYS, KNORM = 200, 30, 2, 30, 20
SUBVAL_D, N_TRIALS, TUNE_SUB, TGT = 30, 25, 200000, 10
CFGIDX, RHKEY, DAY_NS = 1, "rH30", 86_400_000_000_000
bk = storage.Client(project=PROJ).bucket(BUCKET)


def _space(t):
    return {"max_depth": t.suggest_int("max_depth", 3, 9), "eta": t.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True), "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True)}


def tune_B_ic(Xst, yst, wst, Xv, pdiff_v):
    if len(Xst) > TUNE_SUB:
        ix = np.random.RandomState(0).choice(len(Xst), TUNE_SUB, replace=False); Xst, yst, wst = Xst[ix], yst[ix], wst[ix]
    dst = xgb.DMatrix(Xst, label=yst, weight=wst); dv = xgb.DMatrix(Xv)
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0}

    def obj(t):
        hp = _space(t); nr = t.suggest_int("num_boost_round", 50, 400)
        b = xgb.train(dict(base, **hp), dst, num_boost_round=nr); pv = b.predict(dv) - 0.5
        if pv.std() < 1e-9:
            return -1.0
        ic = float(np.corrcoef(pv, pdiff_v)[0, 1]); return ic if np.isfinite(ic) else -1.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=N_TRIALS)
    bp = dict(st.best_params); nr = bp.pop("num_boost_round"); return bp, int(nr), float(st.best_value)


def fith(hp, niter, X, y, w):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0}
    return xgb.train(dict(p, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def exec_trades(sel, fd):
    side = fd["side"][sel]; net = np.where(side, fd["netl"][sel], fd["nets"][sel]); fc = np.where(side, fd["fl"][sel], fd["fs"][sel])
    ex = fc & np.isfinite(net); return net[ex], fd["cd"][sel][ex]


def dep_rolling(FD, target):
    NA, CA = [], []
    for fd in FD:
        sc_tr, sc_te = fd["cdf_tr"], fd["cdf_te"]; day_te, day_tr = fd["day_te"], fd["day_tr"]
        days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1); q = max(0.0, 1.0 - target / max(wpd, 1.0))
        tr_days = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, tr_days[-KDAYS:])
        buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
        for d in days:
            idx = np.where(day_te == d)[0]; tau = float(np.quantile(buf, q)) if buf else 0.0
            sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
        n, c = exec_trades(np.array(sel, int), fd) if sel else (np.array([]), np.array([])); NA.append(n); CA.append(c)
    return np.concatenate(NA) if NA else np.array([]), np.concatenate(CA) if CA else np.array([])


def dep_frozen(FD, target):
    NA, CA = [], []
    for fd in FD:
        tau = float(np.quantile(fd["conf_tr"], 1.0 - target / max(fd["wpd_tr"], 1.0)))
        sel = np.where(fd["conf_te"] >= tau)[0]; n, c = exec_trades(sel, fd); NA.append(n); CA.append(c)
    return np.concatenate(NA) if NA else np.array([]), np.concatenate(CA) if CA else np.array([])


def dep_fastadapt(FD, target):
    NA, CA = [], []
    for fd in FD:
        day_te = fd["day_te"]; days = sorted(set(day_te.tolist()))
        tau = float(np.quantile(fd["conf_tr"], 1.0 - target / max(fd["wpd_tr"], 1.0)))  # seed from train day-0
        sel = []
        for d in days:
            idx = np.where(day_te == d)[0]
            sel.extend(idx[fd["conf_te"][idx] >= tau].tolist())
            dc = fd["conf_te"][idx]
            if len(idx) > 10:
                tau = float(np.quantile(dc, 1.0 - target / max(len(idx), 1.0)))  # tomorrow's bar = today's distn
        n, c = exec_trades(np.array(sel, int), fd) if sel else (np.array([]), np.array([])); NA.append(n); CA.append(c)
    return np.concatenate(NA) if NA else np.array([]), np.concatenate(CA) if CA else np.array([])


def metrics(net, cd, tot_days):
    n = len(net)
    if not n:
        return dict(n=0, trd_d=0, ev=float("nan"), daily_annS=float("nan"), tot_pct=0.0, hit=float("nan"))
    cd = cd.astype(np.int64); lo, hi = int(cd.min()), int(cd.max()); g = np.zeros(hi - lo + 1); np.add.at(g, cd - lo, net)
    da = float(g.mean() / g.std() * np.sqrt(365.0)) if g.std() > 0 else float("nan")
    return dict(n=n, trd_d=round(n / max(tot_days, 1), 1), ev=round(float(net.mean()), 3),
                daily_annS=round(da, 2), tot_pct=round(float(net.sum() * 0.01), 1), hit=round(float((net > 0).mean()) * 100, 1))


d = np.load(io.BytesIO(bk.blob(f"research_runs/maker_labels_h/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int); rH = d[RHKEY].astype(np.float64); ts = d["ts"].astype(np.int64)
calday = (ts // DAY_NS).astype(np.int64)
netl = d["pnl_long"][CFGIDX, 0, :].astype(np.float64) * 100.0; nets = d["pnl_short"][CFGIDX, 0, :].astype(np.float64) * 100.0
fl = d["fill_long"].astype(bool)[0]; fs = d["fill_short"].astype(bool)[0]; nfeat = F.shape[1]
day_mean = np.zeros((ndays, nfeat)); day_var = np.zeros((ndays, nfeat))
for dd in range(ndays):
    mk = day == dd
    if mk.sum() > 1:
        day_mean[dd] = F[mk].mean(0); day_var[dd] = F[mk].var(0)
gstd = F.std(0); mu = np.zeros((ndays, nfeat)); sd = np.zeros((ndays, nfeat))
for dd in range(ndays):
    sl = slice(max(0, dd - KNORM), dd) if dd > 0 else slice(0, 1)
    mu[dd] = day_mean[sl].mean(0); sd[dd] = np.sqrt(np.maximum(day_var[sl].mean(0), 0))
sd = np.maximum(sd, 0.2 * gstd[None, :] + 1e-9)
Fn = ((F - mu[day]) / sd[day]).astype(np.float32)
print(f"[{SYM} baseline2: 3 selectivity schemes] Fn={Fn.shape} nthread={NTHREAD}", flush=True)

FOLDS = []; tsd = W + EMB
while tsd < ndays:
    te = min(tsd + T, ndays); trn = (day >= tsd - EMB - W) & (day < tsd - EMB); tst = (day >= tsd) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    tsd += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)

FD = []; ic_val = []
for fi, (trn, tst) in enumerate(FOLDS):
    tr_days = sorted(set(day[trn].tolist())); sv = set(tr_days[-SUBVAL_D:])
    sub = trn & np.isin(day, list(tr_days[:-(SUBVAL_D + EMB)])); val = trn & np.isin(day, list(sv))
    yB = (np.where(fl, netl, -np.inf) > np.where(fs, nets, -np.inf)).astype(int); both = fl & fs
    wq = np.where(both, np.abs(netl - nets), np.where(fl, np.abs(netl), np.where(fs, np.abs(nets), 0.0)))
    wcl = lambda mk: np.clip(wq[mk], 0, np.quantile(wq[mk][wq[mk] > 0], 0.99) if (wq[mk] > 0).any() else 1.0)
    sb = sub & (fl | fs); vb = val & both
    hpB, biB, icB = tune_B_ic(Fn[sb], yB[sb], wcl(sb), Fn[vb], (netl[vb] - nets[vb]))
    gfull = trn & (fl | fs); Bf = fith(hpB, biB, Fn[gfull], yB[gfull], wcl(gfull))
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    conf_tr = np.abs(Bf.predict(xgb.DMatrix(Fn[tri])) - 0.5); pte = Bf.predict(xgb.DMatrix(Fn[tei])); conf_te = np.abs(pte - 0.5)
    sref = np.sort(conf_tr)
    n_tr_days = len(set(day[tri].tolist())); n_te_days = len(set(day[tei].tolist()))
    FD.append(dict(conf_tr=conf_tr, conf_te=conf_te, cdf_tr=np.searchsorted(sref, conf_tr, "right") / len(sref),
                   cdf_te=np.searchsorted(sref, conf_te, "right") / len(sref), side=pte >= 0.5,
                   day_tr=day[tri], day_te=day[tei], cd=calday[tei], netl=netl[tei], nets=nets[tei], fl=fl[tei], fs=fs[tei],
                   wpd_tr=len(tri) / max(n_tr_days, 1), wpd_te=len(tei) / max(n_te_days, 1)))
    ic_val.append(round(icB, 4))
    print(f"  fold{fi}: IC(val)={icB:+.4f} (d{hpB['max_depth']},nr{biB})", flush=True)

SCHEMES = {"rolling": dep_rolling, "frozen": dep_frozen, "fastadapt": dep_fastadapt}
RES = {"symbol": SYM, "ic_val": ic_val, "schemes": {}}
print(f"\n  {'scheme':>10} {'trd/d':>6} {'EV/tr':>7} {'dailyS':>7} {'hit%':>6} {'tot%':>7}", flush=True)
for nm, fn in SCHEMES.items():
    net, cd = fn(FD, TGT); x = metrics(net, cd, tot_days); RES["schemes"][nm] = x
    print(f"  {nm:>10} {x['trd_d']:>6.1f} {x['ev']:>+7.2f} {x['daily_annS']:>+7.2f} {x['hit']:>6.1f} {x['tot_pct']:>+7.1f}", flush=True)
    b = io.BytesIO(); np.savez_compressed(b, net=net, calday=cd); bk.blob(f"research_runs/maker_labels_h/portfolio2/{SYM}_{nm}.npz").upload_from_string(b.getvalue())

# preds cache (offline sweep of any future scheme)
cb = io.BytesIO()
np.savez_compressed(cb, conf_te=np.concatenate([fd["conf_te"] for fd in FD]).astype(np.float32),
                    cd_te=np.concatenate([fd["cd"] for fd in FD]).astype(np.int64),
                    day_te=np.concatenate([fd["day_te"] for fd in FD]).astype(np.int32),
                    netl=np.concatenate([fd["netl"] for fd in FD]).astype(np.float32),
                    nets=np.concatenate([fd["nets"] for fd in FD]).astype(np.float32),
                    fl=np.concatenate([fd["fl"] for fd in FD]), fs=np.concatenate([fd["fs"] for fd in FD]),
                    side=np.concatenate([fd["side"] for fd in FD]),
                    fold=np.concatenate([np.full(len(fd["cd"]), i, np.int16) for i, fd in enumerate(FD)]),
                    conf_tr=np.concatenate([fd["conf_tr"] for fd in FD]).astype(np.float32),
                    fold_tr=np.concatenate([np.full(len(fd["conf_tr"]), i, np.int16) for i, fd in enumerate(FD)]))
bk.blob(f"research_runs/maker_labels_h/portfolio2/{SYM}_preds.npz").upload_from_string(cb.getvalue())
bk.blob(f"research_runs/maker_labels_h/portfolio2/{SYM}_schemes.json").upload_from_string(json.dumps(RES, default=float))
print(f"\n[{SYM}] saved portfolio2/{SYM}_{{rolling,frozen,fastadapt}}.npz + _preds.npz + _schemes.json", flush=True)
