#!/usr/bin/env python3
"""HX1 rev5 — A0/A1 retrain arms (exploratory trainer; the byte-frozen year
protocol is NOT touched — its W200/T30 folds cannot fit a 17-day window).

Per (SYM, ARM, FOLD, SEED): two XGBoost heads mirroring the deployed
structure — A = activity classifier (top-5% |amp|, AUC-tuned), Bg = direction
classifier (amp>0, |amp|-weighted, val rank-IC-tuned), Optuna 25 trials each
(sampler seed = SEED), refit on full train with tuned params; per-seed score
= rankCDF_train(pA) * rankCDF_train(|pBg-0.5|). ENSEMBLE = mean 4-seed score,
side = mean pBg >= 0.5 (deployed scoring).

amp = (netl - nets)/2 (signed-move proxy from the maker sim, both sides
filled); ARM A0 -> F71, A1 -> [F71 | CV11] (NaN->0).

FOLDS (frozen): f1 train 0628-0704 test 0705-0709; f2 train 0628-0709
test 0710-0714. Inner val = last 2 train days (tuning only).

STAGE=train  -> per-(sym,arm,fold,seed) artifacts {OUT}/train/{...}.npz
STAGE=eval   -> ensemble cells: rank-IC(ens, amp), EV t5/t10 top-K/day
                (y = side-consistent maker pnl; NaN = unfilled = no trade),
                LOO-day, jitter sd {.017,.02,.05} x 200 (selection only),
                per-seed spread, delta(A1-A0) -> {OUT}/TRAIN_RESULT.json

Env: SYMS(DOGE,XRP,BTC,ETH) STAGE(all) NPROC(14) NTHREAD(2)
     OUT(gs://market-data-0998ac51/research_runs/hx1_stack)
"""
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("DAY0", "20260628")
os.environ.setdefault("DAYN", "20260714")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx1_oos import day_list  # noqa: E402

MKT = "gs://market-data-0998ac51"
OUT = os.environ.get("OUT", f"{MKT}/research_runs/hx1_stack")
SYMS = os.environ.get("SYMS", "DOGE,XRP,BTC,ETH").split(",")
NTHREAD = int(os.environ.get("NTHREAD", "2"))
SEEDS = [0, 1, 2, 3]
FOLDS = {"f1": ("20260628", "20260704", "20260705", "20260709"),
         "f2": ("20260628", "20260709", "20260710", "20260714")}
N_TRIALS = 25
BUDGETS = [5, 10]
JITTERS = [0.017, 0.02, 0.05]
NDRAW = 200
STEP_S = 3


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-800:]}")


def dl(url, lf):
    if not os.path.exists(lf):
        sh(f"gcloud storage cp {url} {lf} -q")
    return lf


_daycache = {}


def load_sym(sym, workdir=None):
    if sym in _daycache:
        return _daycache[sym]
    workdir = workdir or f"hx1_train_local/w{os.getpid()}"  # no cross-process races
    os.makedirs(workdir, exist_ok=True)
    days = []
    for day in day_list():
        try:
            d = np.load(dl(f"{OUT}/days/{sym}/{day}.npz",
                           f"{workdir}/days_{sym}_{day}.npz"))
            cv = np.load(dl(f"{OUT}/cv/{sym}/{day}.npy",
                            f"{workdir}/cv_{sym}_{day}.npy"))
        except RuntimeError:
            continue
        row = d["sec"] // STEP_S
        days.append(dict(
            day=day, F=d["F"].astype(np.float32),
            CV=np.nan_to_num(cv[row], nan=0.0).astype(np.float32),
            netl=d["netl"].astype(np.float64), nets=d["nets"].astype(np.float64)))
    _daycache[sym] = days
    return days


def xmat(dd, arm):
    return dd["F"] if arm == "A0" else np.column_stack([dd["F"], dd["CV"]])


def rank_ic(x, y):
    from scipy.stats import rankdata
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 500:
        return np.nan
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def train_job(args):
    sym, arm, fold, seed = args
    tag = f"{sym}_{arm}_{fold}_s{seed}"
    out_url = f"{OUT}/train/{tag}.npz"
    if subprocess.run(f"gcloud storage ls {out_url}", shell=True,
                      capture_output=True).returncode == 0:
        return f"{tag} SKIP"
    try:
        import optuna
        import xgboost as xgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t0, t1, e0, e1 = FOLDS[fold]
        days = load_sym(sym)
        tr = [d for d in days if t0 <= d["day"] <= t1]
        te = [d for d in days if e0 <= d["day"] <= e1]
        if len(tr) < 5 or not te:
            return f"{tag} SKIP thin ({len(tr)}/{len(te)})"
        val_days = {d["day"] for d in tr[-2:]}
        amp = lambda ds: np.concatenate(
            [(d["netl"] - d["nets"]) / 2.0 for d in ds])
        X_all = lambda ds: np.vstack([xmat(d, arm) for d in ds])
        a_tr, a_te = amp(tr), amp(te)
        X_tr, X_te = X_all(tr), X_all(te)
        isval = np.concatenate(
            [np.full(len(d["netl"]), d["day"] in val_days) for d in tr])
        ok_tr = np.isfinite(a_tr)
        fitm, valm = ok_tr & ~isval, ok_tr & isval

        def tune(objective_label, weights, metric):
            def obj(trial):
                p = dict(
                    max_depth=trial.suggest_int("max_depth", 3, 8),
                    learning_rate=trial.suggest_float("eta", 0.03, 0.3, log=True),
                    min_child_weight=trial.suggest_float("mcw", 1, 50, log=True),
                    subsample=trial.suggest_float("sub", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("col", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("lam", 0.01, 10, log=True),
                    n_estimators=400, tree_method="hist", n_jobs=NTHREAD,
                    random_state=seed, eval_metric="logloss",
                    early_stopping_rounds=30)
                m = xgb.XGBClassifier(**p)
                m.fit(X_tr[fitm], objective_label[fitm],
                      sample_weight=(weights[fitm] if weights is not None else None),
                      eval_set=[(X_tr[valm], objective_label[valm])], verbose=False)
                pv = m.predict_proba(X_tr[valm])[:, 1]
                trial.set_user_attr("best_iter", int(m.best_iteration or 400))
                return metric(pv)
            st = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed))
            st.optimize(obj, n_trials=N_TRIALS, n_jobs=1)
            bp = st.best_params
            ne = st.best_trial.user_attrs.get("best_iter", 400)
            params = dict(
                max_depth=bp["max_depth"], learning_rate=bp["eta"],
                min_child_weight=bp["mcw"], subsample=bp["sub"],
                colsample_bytree=bp["col"], reg_lambda=bp["lam"],
                n_estimators=max(ne, 20), tree_method="hist", n_jobs=NTHREAD,
                random_state=seed, eval_metric="logloss")
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr[ok_tr], objective_label[ok_tr],
                  sample_weight=(weights[ok_tr] if weights is not None else None),
                  verbose=False)
            return m, bp, float(st.best_value)

        from sklearn.metrics import roc_auc_score
        # A head: activity = top-5% |amp| on fit days
        thrA = np.quantile(np.abs(a_tr[fitm]), 0.95)
        labA = (np.abs(a_tr) >= thrA).astype(int)
        av = a_tr[valm]
        mA, bpA, scA = tune(labA, None,
                            lambda pv: roc_auc_score(labA[valm], pv))
        # Bg head: direction, |amp|-weighted, val rank-IC objective
        labB = (a_tr > 0).astype(int)
        def _icm(pv):
            v = rank_ic(pv, av)
            return -1.0 if (v is None or np.isnan(v)) else v
        mB, bpB, scB = tune(labB, np.abs(a_tr), _icm)
        pA_tr = mA.predict_proba(X_tr[ok_tr])[:, 1]
        pB_tr = mB.predict_proba(X_tr[ok_tr])[:, 1]
        pA_te = mA.predict_proba(X_te)[:, 1]
        pB_te = mB.predict_proba(X_te)[:, 1]
        refA = np.sort(pA_tr)
        refB = np.sort(np.abs(pB_tr - 0.5))
        cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)
        score_te = cdf(pA_te, refA) * cdf(np.abs(pB_te - 0.5), refB)
        buf = io.BytesIO()
        np.savez_compressed(
            buf, score=score_te.astype(np.float32), pbg=pB_te.astype(np.float32),
            te_days=np.array([d["day"] for d in te]),
            te_len=np.array([len(d["netl"]) for d in te]),
            bpA=json.dumps(bpA), bpB=json.dumps(bpB),
            valA=scA, valB=scB)
        lf = f"hx1_train_local/{tag}.npz"
        open(lf, "wb").write(buf.getvalue())
        sh(f"gcloud storage cp {lf} {out_url} -q")
        return f"{tag} OK valAUC_A={scA:.4f} valIC_B={scB:+.4f}"
    except Exception as ex:  # noqa: BLE001
        return f"{tag} FAIL {type(ex).__name__}: {str(ex)[:250]}"


# ------------------------------------------------------------------- eval
def ev_topk(score_by_day, y_by_day, budget, rng=None, jit=0.0):
    bp_days, ntr = [], 0
    for day in score_by_day:
        s = score_by_day[day]
        if jit:
            s = s + rng.normal(0, jit, len(s))
        k = min(budget, len(s))
        thr = np.partition(s, -k)[-k]
        take = s >= thr
        y = y_by_day[day][take]
        y = y[np.isfinite(y)]
        bp_days.append(float(y.sum()))
        ntr += len(y)
    tot = sum(bp_days)
    return dict(bpd=tot / len(bp_days) if bp_days else 0.0,
                ev_tr=tot / ntr if ntr else 0.0, ntr=ntr, bp_days=bp_days)


def evaluate():
    res = {}
    for sym in SYMS:
        days = {d["day"]: d for d in load_sym(sym)}
        for fold in FOLDS:
            for arm in ("A0", "A1"):
                seeds = []
                for seed in SEEDS:
                    tag = f"{sym}_{arm}_{fold}_s{seed}"
                    try:
                        z = np.load(dl(f"{OUT}/train/{tag}.npz",
                                       f"hx1_train_local/{tag}.npz"),
                                    allow_pickle=True)
                    except RuntimeError:
                        continue
                    seeds.append(z)
                if len(seeds) < len(SEEDS):
                    res[f"{sym}_{arm}_{fold}"] = dict(error="missing seeds")
                    continue
                te_days = [str(x) for x in seeds[0]["te_days"]]
                te_len = list(seeds[0]["te_len"])
                sc = np.mean([z["score"] for z in seeds], axis=0)
                pbg = np.mean([z["pbg"] for z in seeds], axis=0)
                side = pbg >= 0.5
                score_by_day, y_by_day, amp_by_day = {}, {}, {}
                off = 0
                for day, ln in zip(te_days, te_len):
                    dd = days[day]
                    sl = slice(off, off + ln)
                    score_by_day[day] = sc[sl]
                    y_by_day[day] = np.where(side[sl], dd["netl"], dd["nets"])
                    amp_by_day[day] = (dd["netl"] - dd["nets"]) / 2.0
                    off += ln
                amp_all = np.concatenate([amp_by_day[d] for d in te_days])
                ic = rank_ic(sc, amp_all)
                per_seed_ic = [rank_ic(z["score"], amp_all) for z in seeds]
                cell = dict(ic_ens=ic, ic_seeds=per_seed_ic,
                            valA=[float(z["valA"]) for z in seeds],
                            valB=[float(z["valB"]) for z in seeds])
                for b in BUDGETS:
                    base = ev_topk(score_by_day, y_by_day, b)
                    loo = []
                    for d in te_days:
                        sub = {k: v for k, v in score_by_day.items() if k != d}
                        loo.append(ev_topk(sub, y_by_day, b)["bpd"])
                    rng = np.random.default_rng(11)
                    jd = {}
                    for j in JITTERS:
                        draws = [ev_topk(score_by_day, y_by_day, b, rng, j)["bpd"]
                                 for _ in range(NDRAW)]
                        jd[str(j)] = dict(p50=float(np.median(draws)),
                                          p_pos=float(np.mean([x > 0 for x in draws])))
                    cell[f"t{b}"] = dict(ev_tr=base["ev_tr"], bpd=base["bpd"],
                                         ntr=base["ntr"], bp_days=base["bp_days"],
                                         loo_bpd_min=float(np.min(loo)),
                                         jitter=jd)
                res[f"{sym}_{arm}_{fold}"] = cell
                print(f"{sym} {arm} {fold}: ic={ic:+.4f} "
                      f"t5 ev/tr {cell['t5']['ev_tr']:+.2f} ({cell['t5']['ntr']}tr) "
                      f"t10 {cell['t10']['ev_tr']:+.2f}", flush=True)
        for fold in FOLDS:
            for b in BUDGETS:
                k0, k1 = f"{sym}_A0_{fold}", f"{sym}_A1_{fold}"
                if "error" in res.get(k0, {}) or "error" in res.get(k1, {}):
                    continue
                res[f"{sym}_DELTA_{fold}_t{b}"] = dict(
                    d_ev_tr=res[k1][f"t{b}"]["ev_tr"] - res[k0][f"t{b}"]["ev_tr"],
                    d_bpd=res[k1][f"t{b}"]["bpd"] - res[k0][f"t{b}"]["bpd"],
                    d_ic=res[k1]["ic_ens"] - res[k0]["ic_ens"])
    with open("hx1_train_local/TRAIN_RESULT.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    sh(f"gcloud storage cp hx1_train_local/TRAIN_RESULT.json {OUT}/ -q")
    print("saved", f"{OUT}/TRAIN_RESULT.json")


def main():
    os.makedirs("hx1_train_local", exist_ok=True)
    stage = os.environ.get("STAGE", "all")
    if stage in ("train", "all"):
        jobs = [(s, a, f, sd) for s in SYMS for a in ("A0", "A1")
                for f in FOLDS for sd in SEEDS]
        nproc = int(os.environ.get("NPROC", "14"))
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for msg in ex.map(train_job, jobs):
                print(msg, flush=True)
    if stage in ("eval", "all"):
        evaluate()


if __name__ == "__main__":
    main()
