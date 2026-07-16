#!/usr/bin/env python3
"""HX1 rev5 — R arm (PRIMARY): re-ranker on [deployed ensemble score, CV11]
vs the score alone, on the stream-complete recorder window.

Decisions/labels come from the deployed recorder-EV artifacts
(_recev_h150anch2_{SYM}/D_{day}.npz: score/side/netl/nets/FL/FS — bit-identical
scoring path to live). D npz carries no timestamps, so stage `map` replicates
the recorder-EV decision-grid construction exactly (calendar-UTC-midnight 3s
grid over sorted exchange_event_ts_us book ticks; ends = unique last-tick<=grid,
W-1 <= e < n-H-1) from ts-only reads of the depth_snapshot parquets, VERIFIES
len(ends) == len(D.score), and stores the FIRST grid ts per decision.

CV11 join: decision at grid ts T -> CV row T//3 (hx1_stack_cv.py rows are
"value of second T-1"), NaN -> 0.0 (rev5 frozen fill policy).

Evaluation (frozen): per OOS day d (expanding walk-forward, first OOS day =
day 8 of the window), threshold = train-pooled score quantile at K/day
(FIXQ-style), takes = score >= thr, trade side = D.side, y = netl if long else
nets (NaN = entry unfilled -> no trade). EV/tr and bp/day at K in {5,10};
delta(arm - base) with LOO-day and base-score jitter sd {.017,.02,.05}
(200 draws; jitter applies to the ens-score input of BOTH arms' selection).

Env: SYMS(DOGE,XRP) STAGE(map|rerank|all) DAY0(20260628) DAYN(20260714)
     OOS0(20260705) OUT(gs://market-data-0998ac51/research_runs/hx1_stack)
"""
import io
import json
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("DAY0", "20260628")
os.environ.setdefault("DAYN", "20260714")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx1_oos import day_list  # noqa: E402

MKT = "gs://market-data-0998ac51"
REC = "gs://recorder-data-asia-0998ac51/chronos/scalper-recorder/binance_futures"
OUT = os.environ.get("OUT", f"{MKT}/research_runs/hx1_stack")
SYMS = os.environ.get("SYMS", "DOGE,XRP").split(",")
OOS0 = os.environ.get("OOS0", "20260705")
NS = 1_000_000_000
W, H, STEP_S = 50, 6000, 3
BUDGETS = [5, 10]
JITTERS = [0.017, 0.02, 0.05]
NDRAW = 200
CV_NCOL = 11


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1000:]}")


def dl(url, lf):
    if not os.path.exists(lf):
        sh(f"gcloud storage cp {url} {lf} -q")
    return lf


# ------------------------------------------------------------------ stage map
def build_map(sym, day, workdir):
    """Replicate recorder-EV decision construction; return grid ts (ns) per
    decision, verified against the D npz length."""
    import pyarrow.parquet as pq
    import tempfile
    from datetime import datetime, timezone
    with tempfile.TemporaryDirectory() as td:
        sh(f"gcloud storage cp '{REC}/{sym}USDT/depth_snapshot/{day}_*.parquet' {td}/ -q")
        ts_all = []
        for f in sorted(os.listdir(td)):
            t = pq.read_table(os.path.join(td, f), columns=["exchange_event_ts_us"])
            a = t["exchange_event_ts_us"].to_numpy()
            ts_all.append(a[~np.isnan(a.astype(np.float64))].astype(np.int64))
    bt = np.sort(np.concatenate(ts_all), kind="stable") * 1000  # us -> ns
    n = len(bt)
    mid0 = int(datetime.strptime(day, "%Y%m%d")
               .replace(tzinfo=timezone.utc).timestamp()) * NS
    grid = np.arange(mid0, bt[-1], STEP_S * NS)
    grid = grid[grid >= bt[0]]
    e = np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1)
    ends, first_i = np.unique(e, return_index=True)
    keep = (ends >= W - 1) & (ends < n - H - 1)
    ends, first_i = ends[keep], first_i[keep]
    grid_ts = grid[first_i]  # first grid point that produced each decision
    d = np.load(dl(f"{MKT}/research_runs/_recev_h150anch2_{sym}/D_{day}.npz",
                   f"{workdir}/D_{sym}_{day}.npz"))
    if len(ends) != len(d["score"]):
        raise RuntimeError(
            f"{sym} {day}: replicated ends {len(ends)} != D rows {len(d['score'])}")
    sec = ((grid_ts - mid0) // NS).astype(np.int64)
    np.save(f"{workdir}/map_{sym}_{day}.npy", sec)
    sh(f"gcloud storage cp {workdir}/map_{sym}_{day}.npy {OUT}/map/{sym}/{day}.npy -q")
    return len(ends)


# --------------------------------------------------------------- stage rerank
def load_day(sym, day, workdir):
    d = np.load(dl(f"{MKT}/research_runs/_recev_h150anch2_{sym}/D_{day}.npz",
                   f"{workdir}/D_{sym}_{day}.npz"))
    sec = np.load(dl(f"{OUT}/map/{sym}/{day}.npy", f"{workdir}/map_{sym}_{day}.npy"))
    cv = np.load(dl(f"{OUT}/cv/{sym}/{day}.npy", f"{workdir}/cv_{sym}_{day}.npy"))
    row = sec // STEP_S
    X = np.nan_to_num(cv[row], nan=0.0).astype(np.float64)
    y = np.where(d["side"], d["netl"], d["nets"]).astype(np.float64)
    filled = np.where(d["side"], d["FL"], d["FS"]).astype(bool)
    return dict(score=d["score"].astype(np.float64), cv=X, y=y,
                filled=filled & ~np.isnan(y), day=day)


def ev_at_budget(score, y, tradeable, thr):
    take = score >= thr
    tr = take & tradeable
    if not tr.any():
        return 0.0, 0
    return float(np.nansum(y[tr])), int(tr.sum())


def evaluate(days_data, scores_by_day, budget):
    """FIXQ-style: per OOS day, thr = train-pooled quantile at budget/day."""
    res = []
    for i, dd in enumerate(days_data):
        if dd["day"] < OOS0:
            continue
        train = [d for d in days_data if d["day"] < dd["day"]]
        pool = np.concatenate([scores_by_day[t["day"]] for t in train])
        q = 1.0 - budget / (len(pool) / len(train))
        thr = np.quantile(pool, min(max(q, 0.0), 1.0))
        s = scores_by_day[dd["day"]]
        bp, ntr = ev_at_budget(s, dd["y"], dd["filled"], thr)
        res.append(dict(day=dd["day"], bp=bp, ntr=ntr))
    tot = sum(r["bp"] for r in res)
    n = sum(r["ntr"] for r in res)
    return dict(bp_sum=tot, ntr=n, ev_tr=(tot / n if n else 0.0),
                bpd=tot / len(res) if res else 0.0, days=res)


def rerank(sym, workdir):
    """Arms are FIT ONCE per OOS day on unjittered scores (deployed semantics:
    the model is frozen; live jitter perturbs the score INPUT). Jitter passes
    only re-PREDICT with the fitted models. Selection rule for all arms =
    within-day rank, train-pooled quantile at K/day (~top-K/day)."""
    days_data = []
    for day in day_list():
        try:
            days_data.append(load_day(sym, day, workdir))
        except RuntimeError as e:
            print(f"{sym} {day}: missing ({e})", flush=True)
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    # fit per OOS day (expanding) on clean scores
    fitted = {}  # day -> dict(arm -> predict_fn)
    for dd in days_data:
        if dd["day"] < OOS0:
            continue
        tr = [t for t in days_data if t["day"] < dd["day"]]
        Xtr = np.vstack([np.column_stack([t["score"], t["cv"]]) for t in tr])
        ftr = np.concatenate([t["filled"] for t in tr])
        lab = (np.concatenate([t["y"] for t in tr])[ftr] > 0).astype(int)
        mu, sd = Xtr[ftr].mean(0), Xtr[ftr].std(0) + 1e-12
        lr = LogisticRegression(max_iter=300, C=1.0)
        lr.fit((Xtr[ftr] - mu) / sd, lab)
        arms = {"logreg": (lambda X, lr=lr, mu=mu, sd=sd:
                           lr.predict_proba((X - mu) / sd)[:, 1])}
        for sdd in (0, 1):
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                random_state=sdd, n_jobs=4, eval_metric="logloss")
            m.fit(Xtr[ftr], lab)
            arms[f"gbm_s{sdd}"] = (lambda X, m=m: m.predict_proba(X)[:, 1])
        fitted[dd["day"]] = arms
    rank = lambda v: (np.argsort(np.argsort(v)) + 1) / len(v)
    out = {}
    for jit in [0.0] + JITTERS:
        rng = np.random.default_rng(7)
        draws = 1 if jit == 0.0 else NDRAW
        acc = {}
        for _ in range(draws):
            base_by_day = {
                dd["day"]: dd["score"] + (rng.normal(0, jit, len(dd["score"]))
                                          if jit else 0.0)
                for dd in days_data}
            base_rank = {d: rank(v) for d, v in base_by_day.items()}
            armscores = {a: dict(base_rank) for a in ("logreg", "gbm_s0", "gbm_s1")}
            for dd in days_data:
                if dd["day"] < OOS0:
                    continue
                Xte = np.column_stack([base_by_day[dd["day"]], dd["cv"]])
                for a, fn in fitted[dd["day"]].items():
                    armscores[a][dd["day"]] = rank(fn(Xte))
            for b in BUDGETS:
                base_ev = evaluate(days_data, base_rank, b)
                for a in armscores:
                    arm_ev = evaluate(days_data, armscores[a], b)
                    acc.setdefault((a, b), []).append((base_ev, arm_ev))
        for (a, b), lst in acc.items():
            d_evtr = [x[1]["ev_tr"] - x[0]["ev_tr"] for x in lst]
            d_bpd = [x[1]["bpd"] - x[0]["bpd"] for x in lst]
            k = f"{a}_t{b}_jit{jit}"
            out[k] = dict(
                arm_ev_tr=float(np.mean([x[1]["ev_tr"] for x in lst])),
                base_ev_tr=float(np.mean([x[0]["ev_tr"] for x in lst])),
                arm_bpd=float(np.mean([x[1]["bpd"] for x in lst])),
                base_bpd=float(np.mean([x[0]["bpd"] for x in lst])),
                arm_ntr=float(np.mean([x[1]["ntr"] for x in lst])),
                base_ntr=float(np.mean([x[0]["ntr"] for x in lst])),
                d_ev_tr=float(np.mean(d_evtr)), d_bpd=float(np.mean(d_bpd)),
                p_d_bpd_pos=float(np.mean([x > 0 for x in d_bpd])),
                draws=len(lst),
                arm_days=(lst[0][1]["days"] if jit == 0.0 else None),
                base_days=(lst[0][0]["days"] if jit == 0.0 else None))
            print(f"[{sym}] {a} t{b} jit={jit}: base ev/tr "
                  f"{out[k]['base_ev_tr']:+.2f} ({out[k]['base_ntr']:.0f}tr) arm "
                  f"{out[k]['arm_ev_tr']:+.2f} ({out[k]['arm_ntr']:.0f}tr) "
                  f"d_bpd {out[k]['d_bpd']:+.2f} P(d>0) {out[k]['p_d_bpd_pos']:.2f}",
                  flush=True)
    lf = f"{workdir}/RERANK_{sym}.json"
    with open(lf, "w") as f:
        json.dump(out, f, indent=1, default=float)
    sh(f"gcloud storage cp {lf} {OUT}/RERANK_{sym}.json -q")


def main():
    stage = os.environ.get("STAGE", "all")
    workdir = os.environ.get("WORKDIR", "hx1_rerank_local")
    os.makedirs(workdir, exist_ok=True)
    if stage in ("map", "all"):
        for sym in SYMS:
            for day in day_list():
                try:
                    n = build_map(sym, day, workdir)
                    print(f"map {sym} {day}: {n} decisions OK", flush=True)
                except RuntimeError as e:
                    print(f"map {sym} {day}: FAIL {e}", flush=True)
    if stage in ("rerank", "all"):
        for sym in SYMS:
            rerank(sym, workdir)


if __name__ == "__main__":
    main()
