#!/usr/bin/env python3
"""Per-fold EARLY-STOPPING (best-epoch) + Model-A vol-target (NF_RATE) sweep, DOGE WF.

Tests three hypotheses about the per-fold capture variation (f2 dominates):
 (1) the A vol-gate window is mis-sized -> sweep NF_RATE in {1%,5%,10%} (A label = |rH60|>=
     train-p(1-NF) ; apred gate coupled = top-NF% by OOF pA);
 (3) no proper early-stop / best-epoch -> per fold, hold out the last SUBVAL_DAYS of the train
     window as a sub-val, fit A and B with early_stopping_rounds, pick the single best_iteration
     per fold, then REFIT on the full train at that best_iter. (HP params stay the frozen
     A_DOGE/B_pool best_params; only the #rounds is re-chosen per fold.)
Reports per-fold: chosen best_iter (A,B), A vol-AUC(test), deploy OOS net EV (post-hoc top-1 and
top-5/day). Saves per-(NF) per-fold preds cache (re-sweep deploy for free) + ES_NFSWEEP_RESULT.json.
"""
import io, json, os, tempfile
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
W, T, EMB = 200, 30, 2
SUBVAL_DAYS = 30; ESR = 30; MAXR = 400; OOF_K = 3
NF_RATES = [0.01, 0.05, 0.10]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_rr(symk):
    d = np.load(io.BytesIO(bk.blob(f"research_runs/maker_labels_rr/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pl": d["pnl_long"].astype(np.float32), "ps": d["pnl_short"].astype(np.float32),
            "fl": d["fill_long"].astype(bool), "fs": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="stable"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct_rank(x): o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def es_best_iter(params, Xtr, ytr, Xv, yv, wtr=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0, "eval_metric": "auc"}
    if spw is not None:
        base["scale_pos_weight"] = spw
    b = xgb.train(dict(base, **params), xgb.DMatrix(Xtr, label=ytr, weight=wtr), num_boost_round=MAXR,
                  evals=[(xgb.DMatrix(Xv, label=yv), "v")], early_stopping_rounds=ESR, verbose_eval=False)
    return int(b.best_iteration)


def refit(params, X, y, n, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **params), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, n + 1))


def oof_pA(F, yA, trn, day, params, niter, k=OOF_K):
    tdays = sorted(set(day[trn].tolist())); fold = {d: i % k for i, d in enumerate(tdays)}
    fday = np.array([fold.get(int(d), -1) for d in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = refit(params, F[trk], yA[trk], niter, spw=spwk)
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


hpA = jload(f"{MAIN}/A_DOGE.json")["best_params"]; hpB = jload(f"{MAIN}/B_pool.json")["best_params"]
E = load_rr("DOGE")
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]; ndays = E["ndays"]
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee

FOLDS = []
ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((ts, te, trn, tst))
    ts += T
print(f"[ES + NF sweep] {len(FOLDS)} folds | NF_RATES={NF_RATES} | SUBVAL={SUBVAL_DAYS}d ESR={ESR} OOF_K={OOF_K}", flush=True)


def deploy(pA, pB, dy, flt, fst, nlt, nst, budget):
    score = pct_rank(pA) * pct_rank(np.abs(pB - 0.5)); sel = daily_pick(dy, score, budget)
    side = pB[sel] >= 0.5; net = np.where(side, nlt[sel], nst[sel]); fc = np.where(side, flt[sel], fst[sel])
    ex = fc & np.isfinite(net); return net[ex]


RES = {}
for nf in NF_RATES:
    print(f"\n=== NF_RATE={nf:.0%} (A target top-{nf:.0%} vol; apred gate top-{nf:.0%}) ===", flush=True)
    print(f"{'fold':>4} {'biA':>4} {'biB':>4} {'A-AUC':>6} {'ev1':>7} {'n1':>4} {'ev5':>7} {'n5':>5}", flush=True)
    acc = {k: [] for k in ("fold_id", "day", "pA", "pB", "fl", "fs", "nl", "ns")}
    perfold = []
    for fi, (t0, t1, trn, tst) in enumerate(FOLDS):
        tr_days = sorted(set(day[trn].tolist()))
        sv_days = set(tr_days[-SUBVAL_DAYS:]); st_days = set(tr_days[:-(SUBVAL_DAYS + EMB)])
        subtr = trn & np.isin(day, list(st_days)); subval = trn & np.isin(day, list(sv_days))
        thr = float(np.quantile(np.abs(rH[trn]), 1 - nf)); yA = (np.abs(rH) >= thr).astype(int)
        spwA = float((yA[subtr] == 0).sum() / max((yA[subtr] == 1).sum(), 1))
        biA = es_best_iter(hpA, F[subtr], yA[subtr], F[subval], yA[subval], spw=spwA)
        spwAf = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = refit(hpA, F[trn], yA[trn], biA, spw=spwAf)
        ti = np.where(tst)[0]; pA_te = bstA.predict(xgb.DMatrix(F[ti]))
        aucA = auc(yA[ti], pA_te)
        # apred gate (top-nf% by OOF pA) using biA
        oof = oof_pA(F, yA, trn, day, hpA, biA); valid = trn & np.isfinite(oof)
        thr_oof = float(np.nanquantile(oof[valid], 1 - nf)); gate = valid & (oof >= thr_oof)
        nl = netl[0]; ns = nets[0]
        yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int); both = fl & fs
        wgt = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
        gsub = gate & subtr & (fl | fs); gsv = gate & subval & (fl | fs); gfull = gate & (fl | fs)
        def clipw(mask):
            w = wgt[mask]; pos = w[w > 0]; return np.clip(w, 0, np.quantile(pos, 0.99) if len(pos) else 1.0)
        biB = es_best_iter(hpB, F[gsub], yB[gsub], F[gsv], yB[gsv], wtr=clipw(gsub)) if gsv.sum() > 20 else biA
        bstB = refit(hpB, F[gfull], yB[gfull], biB, w=clipw(gfull))
        pB_te = bstB.predict(xgb.DMatrix(F[ti]))
        dy = day[ti]; flt = fl[ti]; fst = fs[ti]; nlt = nl[ti]; nst = ns[ti]
        ev1 = deploy(pA_te, pB_te, dy, flt, fst, nlt, nst, 1)
        ev5 = deploy(pA_te, pB_te, dy, flt, fst, nlt, nst, 5)
        perfold.append({"fold": fi, "biA": biA, "biB": biB, "aucA": aucA,
                        "ev1_oos": float(ev1.sum() * 0.01), "n1": len(ev1),
                        "ev5_oos": float(ev5.sum() * 0.01), "n5": len(ev5)})
        for k, v in [("fold_id", np.full(len(ti), fi)), ("day", dy), ("pA", pA_te), ("pB", pB_te),
                     ("fl", flt), ("fs", fst), ("nl", nlt), ("ns", nst)]:
            acc[k].append(v)
        print(f"{fi:>4} {biA:>4} {biB:>4} {aucA:>6.3f} {ev1.sum()*0.01:>+6.1f}% {len(ev1):>4} "
              f"{ev5.sum()*0.01:>+6.1f}% {len(ev5):>5}", flush=True)
    tot1 = sum(p["ev1_oos"] for p in perfold); tot5 = sum(p["ev5_oos"] for p in perfold)
    print(f"  POOLED OOS: top-1 {tot1:+.1f}%  top-5 {tot5:+.1f}%  | per-fold ev1: {[round(p['ev1_oos'],1) for p in perfold]}", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.concatenate(v) for k, v in acc.items()},
                        meta=np.array(json.dumps({"nf": nf, "W": W, "T": T, "es": True})))
    bk.blob(f"research_runs/wf_cache/DOGE_es_nf{int(nf*100)}_preds.npz").upload_from_string(buf.getvalue())
    RES[f"nf{nf}"] = {"perfold": perfold, "pooled_ev1": tot1, "pooled_ev5": tot5}

bk.blob("research_runs/maker_labels_rr/ES_NFSWEEP_RESULT.json").upload_from_string(json.dumps(RES, default=float))
print("\n[saved] ES_NFSWEEP_RESULT.json + per-nf preds caches (wf_cache/DOGE_es_nf{1,5,10}_preds.npz)", flush=True)
