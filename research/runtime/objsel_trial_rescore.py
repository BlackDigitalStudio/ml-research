#!/usr/bin/env python3
"""Is the B-search selecting on the population the policy actually trades?

OBJSEL rev1. The frozen trainer picks B's hyperparameters by

    ic = corr(pB - 0.5, netl - nets)   over  vb = val & both      (~1e5-1e6 rows)

while the cell is produced by the tau-selected top ~0.04% of the AxB composite. This
script measures, on the ALREADY-CAPTURED trials with NO retraining, whether a
selection metric computed at the OPERATING POINT would pick different trials, and
whether that pick is better or merely noisier.

Per (symbol, seed, fold) it rescores all 25 captured B-trial boosters and reports, for
each candidate metric M:

  RELEVANCE   Spearman over the 25 trials between M (computed on val) and the trial's
              realised TEST cell EV (causal_rolling at the deployed budget).
  RELIABILITY Spearman between the trial-rankings M produces on the first and second
              temporal half of the val window. A metric that cannot reproduce its own
              ranking within val cannot be selecting on signal.
  SWITCH      test EV of the trial M would have picked, minus test EV of the trial the
              incumbent bulk IC actually picked.

Metric family, all on the same val rows, indexed by selectivity q (q=1.0 IS the
incumbent):  M_ic(q) = the same correlation restricted to the top-q of the composite;
M_ev(q) = mean realised net of the top-q. Plus the per-day budget form (top-K per val
day), which is the deployed selection semantics rather than a pooled quantile.

BINDING PARITY GATE, run before any metric is emitted: the recomputed q=1.0 IC must
reproduce the `ic` stored in TRIALS_*_B_index.json for every trial (tol 1e-6). That
validates the val-window reconstruction, the vol-norm and the row join in one shot -
if it fails, nothing this script prints may be quoted.

WHAT THIS IS NOT: the trial boosters are fit on the subsampled tuning set `sb`, while
the deployed Bg is a REFIT on the gated train window carrying the winning trial's HP.
So this measures which trial the search should have preferred, not what the protocol
would have deployed. The refit arm is a separate cell and needs training.

Env: SYM, DATA_SUB, MODEL_SUB, SEEDS (csv), FOLDS_ONLY (csv, default all), DROP_COLS,
     BUDGET, QMIDX, CFGIDX, NTRIALS, OUT, LOCAL_CACHE.
"""
import io, json, os

import numpy as np
import xgboost as xgb
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
DATA_SUB = os.environ.get("DATA_SUB", "research_runs/maker_labels_tb3s_h150anch")
MODEL_SUB = os.environ.get("MODEL_SUB", "modelcap_h150anch")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3").split(",") if x != ""]
FOLDS_ONLY = [int(x) for x in os.environ.get("FOLDS_ONLY", "").split(",") if x != ""]
DROP = [int(x) for x in os.environ.get("DROP_COLS", "").split(",") if x != ""]
BUDGET = int(os.environ.get("BUDGET", "5"))
QMIDX = int(os.environ.get("QMIDX", "0")); CFGIDX = int(os.environ.get("CFGIDX", "1"))
NTRIALS = int(os.environ.get("NTRIALS", "25"))
CACHE = os.environ.get("LOCAL_CACHE", "/tmp/objsel_cache")
OUT = os.environ.get("OUT", f"research_runs/objsel/OBJSEL_{SYM}.json")
# frozen protocol constants (subs60_xgb_optuna_ic.py) - do not tune here
W, T, EMB, KNORM, SUBVAL_D, KDAYS = 200, 30, 2, 20, 30, 30
QGRID = [1.0, 0.1, 0.03, 0.01, 0.003, 0.001]
PARITY_TOL = 1e-6
bk = storage.Client(project=PROJ).bucket(BUCKET)
os.makedirs(CACHE, exist_ok=True)


def log(s):
    print(s, flush=True)


def cdf_map(x, ref):
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def causal_rolling(sc_tr, sc_te, day_tr, day_te, target_tpd, sideB, fl_, fs_, nl_, ns_):
    """subs60_xgb_optuna_ic.causal_rolling, verbatim - this is measurement semantics."""
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - target_tpd / max(wpd, 1.0))
    tr_days = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, tr_days[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for dd in days:
        idx = np.where(day_te == dd)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl_[sel], ns_[sel]); fc = np.where(side, fl_[sel], fs_[sel])
    ex = fc & np.isfinite(net); return net[ex]


def blob_file(name):
    """Download once into LOCAL_CACHE; boosters are re-read across folds/metrics."""
    p = os.path.join(CACHE, name.replace("/", "_"))
    if not os.path.exists(p):
        tmp = p + ".part"
        bk.blob(name).download_to_filename(tmp); os.replace(tmp, p)
    return p


def booster(name):
    b = xgb.Booster(); b.load_model(blob_file(name)); return b


# ---------------------------------------------------------------- dataset + vol-norm
# (identical to subs60_xgb_optuna_ic.py / logit_anatomy.py: DROP first, then the
#  day-wise trailing-KNORM norm in float64, cast to float32)
log(f"[load] {DATA_SUB}/{SYM}.npz")
d = np.load(io.BytesIO(bk.blob(f"{DATA_SUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int)
if DROP:
    keep = [i for i in range(F.shape[1]) if i not in DROP]
    F = F[:, keep]; log(f"  dropped {DROP} -> {F.shape[1]} cols")
netl = d["pnl_long"][CFGIDX, QMIDX, :].astype(np.float64) * 100.0
nets = d["pnl_short"][CFGIDX, QMIDX, :].astype(np.float64) * 100.0
fl = d["fill_long"].astype(bool)[QMIDX]; fs = d["fill_short"].astype(bool)[QMIDX]
nfeat = F.shape[1]
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
del F, day_mean, day_var, mu_ref, sd_ref
log(f"  Fn={Fn.shape}")

yB = (np.where(fl, netl, -np.inf) > np.where(fs, nets, -np.inf)).astype(int)
both = fl & fs
pdiff = netl - nets

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays); trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
log(f"  folds={len(FOLDS)} budget=t{BUDGET}")


# ---------------------------------------------------------------- metric family
def _corr(x, y):
    if len(x) < 20 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    v = float(np.corrcoef(x, y)[0, 1])
    return v if np.isfinite(v) else float("nan")


def metrics_on(score, pv, pdif, net, filled, dayv, tag_days):
    """Candidate metrics for one trial on one row set.

    score  composite AxB rank score (A pinned to the fold's captured A model)
    pv     pB - 0.5 of this trial            pdif  netl - nets  (the incumbent's target)
    net    realised net of the chosen side   filled  did that side fill
    ic_q1 is the incumbent search objective and is the parity-gated quantity.
    """
    out = {}
    n = len(score)
    ok = filled & np.isfinite(net)
    for q in QGRID:
        if q >= 1.0:
            sel = np.ones(n, dtype=bool)
        else:
            k = max(int(round(q * n)), 20)
            sel = np.ones(n, dtype=bool) if k >= n else score >= np.partition(score, n - k)[n - k]
        s_ok = sel & ok
        out[f"ev_q{q:g}"] = float(net[s_ok].mean()) if s_ok.sum() >= 20 else float("nan")
        out[f"ic_q{q:g}"] = _corr(pv[sel], pdif[sel])
    # deployed selection semantics: top-BUDGET per day, not a pooled quantile
    selb = np.zeros(n, dtype=bool)
    for dd in tag_days:
        idx = np.where(dayv == dd)[0]
        if len(idx):
            selb[idx[np.argsort(score[idx])[-min(BUDGET, len(idx)):]]] = True
    s_ok = selb & ok
    out["ev_budget"] = float(net[s_ok].mean()) if s_ok.sum() >= 20 else float("nan")
    out["n_budget"] = int(s_ok.sum())
    return out


res = {"sym": SYM, "data_sub": DATA_SUB, "model_sub": MODEL_SUB, "budget": BUDGET,
       "drop_cols": DROP, "qgrid": QGRID, "cells": []}
parity_max = 0.0; parity_n = 0

for s in SEEDS:
    for fi, (trn, tst) in enumerate(FOLDS):
        if FOLDS_ONLY and fi not in FOLDS_ONLY:
            continue
        idx_name = f"research_runs/{MODEL_SUB}/TRIALS_S{s}_{SYM}_f{fi}_B_index.json"
        try:
            index = json.load(open(blob_file(idx_name)))
        except Exception as e:
            log(f"  [skip] S{s} f{fi}: no trial index ({type(e).__name__})"); continue

        # trainer's sub/val split, verbatim
        tr_days = sorted(set(day[trn].tolist())); sv = set(tr_days[-SUBVAL_D:])
        sub = trn & np.isin(day, list(tr_days[:-(SUBVAL_D + EMB)])); val = trn & np.isin(day, list(sv))
        vb = val & both
        vbi = np.where(vb)[0]; subi = np.where(sub)[0]; tri = np.where(trn)[0]; tei = np.where(tst)[0]
        if len(vbi) < 1000:
            log(f"  [skip] S{s} f{fi}: vb={len(vbi)}"); continue

        # materialise the four row-slices ONCE per (seed,fold) - fancy indexing copies,
        # and Fn[tri] is ~0.4GB, so doing it inside the 25-trial loop is 25x the cost
        Xsub = Fn[subi]; Xvb = Fn[vbi]; Xtr = Fn[tri]; Xte = Fn[tei]

        A = booster(f"research_runs/{MODEL_SUB}/MODELS_S{s}_{SYM}_f{fi}_A.json")
        pA_sub = A.inplace_predict(Xsub, validate_features=False).astype(np.float64)
        pA_vb = A.inplace_predict(Xvb, validate_features=False).astype(np.float64)
        pA_tr = A.inplace_predict(Xtr, validate_features=False).astype(np.float64)
        pA_te = A.inplace_predict(Xte, validate_features=False).astype(np.float64)
        sA_sub = np.sort(pA_sub); sA_tr = np.sort(pA_tr)
        cdfA_vb = cdf_map(pA_vb, sA_sub); cdfA_te = cdf_map(pA_te, sA_tr)
        cdfA_tr = np.searchsorted(sA_tr, pA_tr, "right") / len(sA_tr)

        dv = day[vbi]; vdays = sorted(set(dv.tolist()))
        half = vdays[: len(vdays) // 2]; half2 = vdays[len(vdays) // 2:]
        h1 = np.isin(dv, half); h2 = np.isin(dv, half2)
        net_vb_l = netl[vbi]; net_vb_s = nets[vbi]
        fl_vb = fl[vbi]; fs_vb = fs[vbi]; pd_vb = pdiff[vbi]

        rows = []
        for t in range(min(NTRIALS, len(index))):
            ent = index[t]
            Bt = booster(f"research_runs/{MODEL_SUB}/TRIALS_S{s}_{SYM}_f{fi}_B_t{t}.json")
            pv = Bt.inplace_predict(Xvb, validate_features=False).astype(np.float64) - 0.5

            # --- PARITY GATE: reproduce the stored search objective exactly
            ic_full = float(np.corrcoef(pv, pd_vb)[0, 1])
            dif = abs(ic_full - float(ent["ic"]))
            parity_max = max(parity_max, dif); parity_n += 1
            if dif > PARITY_TOL:
                raise SystemExit(f"PARITY FAIL S{s} f{fi} t{t}: recomputed {ic_full:.12f} "
                                 f"vs stored {ent['ic']:.12f} (d={dif:.2e})")

            # --- val-side composite at the operating point (A pinned: isolates the B knob)
            p_sub = Bt.inplace_predict(Xsub, validate_features=False).astype(np.float64)
            sB_sub = np.sort(np.abs(p_sub - 0.5))
            score_vb = cdfA_vb * cdf_map(np.abs(pv), sB_sub)
            side_vb = pv >= 0.0
            net_vb = np.where(side_vb, net_vb_l, net_vb_s)
            filled_vb = np.where(side_vb, fl_vb, fs_vb)
            mall = metrics_on(score_vb, pv, pd_vb, net_vb, filled_vb, dv, vdays)
            m1 = metrics_on(score_vb[h1], pv[h1], pd_vb[h1], net_vb[h1], filled_vb[h1], dv[h1], half)
            m2 = metrics_on(score_vb[h2], pv[h2], pd_vb[h2], net_vb[h2], filled_vb[h2], dv[h2], half2)

            # --- test-side ground truth: this trial as Bg, deployed selection
            p_tr = Bt.inplace_predict(Xtr, validate_features=False).astype(np.float64)
            p_te = Bt.inplace_predict(Xte, validate_features=False).astype(np.float64)
            sB_tr = np.sort(np.abs(p_tr - 0.5))
            sc_tr = cdfA_tr * (np.searchsorted(sB_tr, np.abs(p_tr - 0.5), "right") / len(sB_tr))
            sc_te = cdfA_te * cdf_map(np.abs(p_te - 0.5), sB_tr)
            a = causal_rolling(sc_tr, sc_te, day[tri], day[tei], BUDGET,
                               p_te >= 0.5, fl[tei], fs[tei], netl[tei], nets[tei])
            rows.append(dict(trial=t, params=ent["params"], ic_stored=float(ent["ic"]),
                             ev_test=float(a.mean()) if len(a) else float("nan"),
                             n_test=int(len(a)), val=mall, val_h1=m1, val_h2=m2))

        if not rows:
            continue
        ev_test = np.array([r["ev_test"] for r in rows])
        names = [k for k in rows[0]["val"] if k.startswith(("ev_", "ic_")) and k != "n_budget"]
        inc = int(np.nanargmax([r["ic_stored"] for r in rows]))
        summ = {}
        for nm in names:
            v = np.array([r["val"][nm] for r in rows])
            v1 = np.array([r["val_h1"][nm] for r in rows]); v2 = np.array([r["val_h2"][nm] for r in rows])
            good = np.isfinite(v) & np.isfinite(ev_test)
            pick = int(np.argmax(np.where(np.isfinite(v), v, -np.inf))) if np.isfinite(v).any() else inc
            summ[nm] = dict(relevance=spearman(v[good], ev_test[good]) if good.sum() >= 3 else float("nan"),
                            reliability=spearman(v1[np.isfinite(v1) & np.isfinite(v2)],
                                                 v2[np.isfinite(v1) & np.isfinite(v2)]),
                            pick=pick, ev_pick=float(ev_test[pick]),
                            switch=float(ev_test[pick] - ev_test[inc]))
        cell = dict(seed=s, fold=fi, n_trials=len(rows), n_vb=int(len(vbi)),
                    incumbent_trial=inc, ev_incumbent=float(ev_test[inc]),
                    ev_test_spread=[float(np.nanmin(ev_test)), float(np.nanmax(ev_test))],
                    summary=summ, trials=rows)
        res["cells"].append(cell)
        log(f"  S{s} f{fi}: vb={len(vbi)} inc=t{inc} EV_test={ev_test[inc]:+.2f} "
            f"spread [{np.nanmin(ev_test):+.2f},{np.nanmax(ev_test):+.2f}] | "
            + " ".join(f"{nm}:rel{summ[nm]['relevance']:+.2f}/rlb{summ[nm]['reliability']:+.2f}"
                       for nm in ("ic_q1", "ev_budget", "ev_q0.01")))

res["parity"] = dict(n=parity_n, max_abs_diff=float(parity_max), tol=PARITY_TOL)
log(f"\n[parity] {parity_n} trials reproduced, max |d| = {parity_max:.3e} (tol {PARITY_TOL:g})")

if res["cells"]:
    names = list(res["cells"][0]["summary"].keys())
    log(f"\n[{SYM}] pooled over {len(res['cells'])} (seed,fold) cells:")
    log(f"  {'metric':>12} {'relevance':>10} {'reliability':>12} {'switch bp':>10}")
    res["pooled"] = {}
    for nm in names:
        rel = np.array([c["summary"][nm]["relevance"] for c in res["cells"]], dtype=float)
        rlb = np.array([c["summary"][nm]["reliability"] for c in res["cells"]], dtype=float)
        sw = np.array([c["summary"][nm]["switch"] for c in res["cells"]], dtype=float)
        res["pooled"][nm] = dict(relevance=float(np.nanmean(rel)), relevance_sd=float(np.nanstd(rel)),
                                 reliability=float(np.nanmean(rlb)), switch=float(np.nanmean(sw)))
        log(f"  {nm:>12} {np.nanmean(rel):>+10.3f} {np.nanmean(rlb):>+12.3f} {np.nanmean(sw):>+10.2f}")

bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"\n[saved] gs://{BUCKET}/{OUT}")
