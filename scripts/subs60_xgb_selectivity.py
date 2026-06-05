#!/usr/bin/env python3
"""SELECTIVITY MODEL (C): learned trade-quality selector vs the pA*pB rank heuristic, DOGE WF.

C = XGB regressor on [F(71), pA, pB] -> executed net maker EV (B's side, hold-60s, touch, MISS->0).
Its output is a calibrated EV in bp, so a frozen deploy threshold transfers train->test better than
pct_rank (which suffers distribution shift). Reuses SAVED per-fold A/B weights (NO A/B retrain); C
trained per fold with early-stop. Compares per-fold (causal-frozen + post-hoc) C vs the heuristic.
Saves per-fold C preds cache + SELECTIVITY_RESULT.json.
"""
import io, json, os, tempfile
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
MODELS_DIR = "research_runs/wf_models/DOGE_adaptive_W200T30"
W, T, EMB = 200, 30, 2; SUBVAL_DAYS = 30; ESR = 30; MAXR = 400
bk = storage.Client(project=PROJ).bucket(BUCKET)


def load_rr(symk):
    d = np.load(io.BytesIO(bk.blob(f"research_runs/maker_labels_rr/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pl": d["pnl_long"].astype(np.float32), "ps": d["pnl_short"].astype(np.float32),
            "fl": d["fill_long"].astype(bool), "fs": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def load_model(blobname):
    p = tempfile.mktemp(suffix=".json"); bk.blob(blobname).download_to_filename(p)
    b = xgb.Booster(); b.load_model(p); os.remove(p); return b


def pct_rank(x): o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def cdf_map(x, ref): return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


E = load_rr("DOGE")
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]; ndays = E["ndays"]
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee

FOLDS = []; ts = W + EMB; fi = 0
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((fi, trn, tst)); fi += 1
    ts += T
tot_te_days = sum(len(set(day[tst].tolist())) for _, _, tst in FOLDS)
print(f"[selectivity-C vs pA*pB heuristic] {len(FOLDS)} folds, OOS={tot_te_days} days\n", flush=True)


def realize(sel, sideB, fl_, fs_, nl_, ns_):
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl_[sel], ns_[sel]); fc = np.where(side, fl_[sel], fs_[sel])
    ex = fc & np.isfinite(net); return net[ex]


def causal_frozen(score_tr, score_te, day_tr, target_tpd, sideB, fl_, fs_, nl_, ns_):
    nd = len(set(day_tr.tolist())); k = int(round(target_tpd * nd))
    tau = np.partition(score_tr, -k)[-k] if 0 < k <= len(score_tr) else -np.inf
    return realize(np.where(score_te >= tau)[0], sideB, fl_, fs_, nl_, ns_)


def st(nets_):
    nets_ = np.asarray(nets_); n = len(nets_)
    return (len(nets_), float(nets_.sum() * 0.01))   # (n_trades, OOS%)


rows = {"C_frozen1": [], "C_frozen5": [], "P_frozen1": [], "P_frozen5": [], "C_post1": []}
nC = {k: [] for k in rows}
for fi, trn, tst in FOLDS:
    A = load_model(f"{MODELS_DIR}/fold{fi}_A.json"); B = load_model(f"{MODELS_DIR}/fold{fi}_B.json")
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    pA_tr = A.predict(xgb.DMatrix(F[tri])); pB_tr = B.predict(xgb.DMatrix(F[tri]))
    pA_te = A.predict(xgb.DMatrix(F[tei])); pB_te = B.predict(xgb.DMatrix(F[tei]))
    # C inputs + target (executed net of B's side, MISS->0)
    Xc_tr = np.column_stack([F[tri], pA_tr, pB_tr]).astype(np.float32)
    Xc_te = np.column_stack([F[tei], pA_te, pB_te]).astype(np.float32)
    sB_tr = pB_tr >= 0.5; sB_te = pB_te >= 0.5
    sfill_tr = np.where(sB_tr, fl[tri], fs[tri]); snet_tr = np.where(sB_tr, netl[0][tri], nets[0][tri])
    yC = np.where(sfill_tr, snet_tr, 0.0); yC = np.clip(yC, -40, 40)
    # sub-val split for early stop (last SUBVAL_DAYS train days)
    dtr = day[tri]; tr_days = sorted(set(dtr.tolist())); sv = set(tr_days[-SUBVAL_DAYS:])
    msv = np.isin(dtr, list(sv)); mst = ~msv
    par = {"objective": "reg:squarederror", "tree_method": "hist", "nthread": 8, "seed": 0,
           "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "eval_metric": "rmse"}
    bst = xgb.train(par, xgb.DMatrix(Xc_tr[mst], label=yC[mst]), num_boost_round=MAXR,
                    evals=[(xgb.DMatrix(Xc_tr[msv], label=yC[msv]), "v")], early_stopping_rounds=ESR, verbose_eval=False)
    cpred_tr = bst.predict(xgb.DMatrix(Xc_tr)); cpred_te = bst.predict(xgb.DMatrix(Xc_te))
    # heuristic product score (causal via train cdf maps)
    sA = np.sort(pA_tr); sBb = np.sort(np.abs(pB_tr - 0.5))
    pscore_tr = pct_rank(pA_tr) * pct_rank(np.abs(pB_tr - 0.5))
    pscore_te = cdf_map(pA_te, sA) * cdf_map(np.abs(pB_te - 0.5), sBb)
    flt = fl[tei]; fst = fs[tei]; nlt = netl[0][tei]; nst = nets[0][tei]; dyt = day[tei]; dytr = day[tri]
    # C causal-frozen
    rows["C_frozen1"].append(causal_frozen(cpred_tr, cpred_te, dytr, 1, sB_te, flt, fst, nlt, nst))
    rows["C_frozen5"].append(causal_frozen(cpred_tr, cpred_te, dytr, 5, sB_te, flt, fst, nlt, nst))
    # product causal-frozen
    rows["P_frozen1"].append(causal_frozen(pscore_tr, pscore_te, dytr, 1, sB_te, flt, fst, nlt, nst))
    rows["P_frozen5"].append(causal_frozen(pscore_tr, pscore_te, dytr, 5, sB_te, flt, fst, nlt, nst))
    # C post-hoc top-1 (upper bound)
    rows["C_post1"].append(realize(daily_pick(dyt, cpred_te, 1), sB_te, flt, fst, nlt, nst))
    print(f"  fold{fi}: best_iterC={bst.best_iteration} "
          f"Cfz1={rows['C_frozen1'][-1].sum()*0.01:+.1f}%(n{len(rows['C_frozen1'][-1])}) "
          f"Cfz5={rows['C_frozen5'][-1].sum()*0.01:+.1f}%(n{len(rows['C_frozen5'][-1])}) "
          f"Pfz5={rows['P_frozen5'][-1].sum()*0.01:+.1f}%(n{len(rows['P_frozen5'][-1])})", flush=True)


def report(key):
    pf = rows[key]; alln = np.concatenate(pf) if pf else np.array([])
    n = len(alln); ev = float(alln.mean()) if n else float("nan"); se = float(alln.std() / max(np.sqrt(n), 1)) if n else 0
    tpd = n / max(tot_te_days, 1)
    print(f"  {key:11} | trd/day={tpd:4.1f} EV/trd={ev*0.01:+.4f}%+-{se*0.01:.4f} OOS={float(alln.sum())*0.01:+6.1f}%  "
          f"per-fold: {[round(p.sum()*0.01,1) for p in pf]}")


print("\n=== SELECTIVITY-C vs heuristic (per-fold OOS%) ===")
for k in ["C_post1", "C_frozen1", "C_frozen5", "P_frozen1", "P_frozen5"]:
    report(k)
RES = {k: {"perfold": [float(p.sum() * 0.01) for p in rows[k]], "n": [int(len(p)) for p in rows[k]],
           "pooled_oos": float(np.concatenate(rows[k]).sum() * 0.01) if rows[k] else 0.0} for k in rows}
bk.blob("research_runs/maker_labels_rr/SELECTIVITY_RESULT.json").upload_from_string(json.dumps(RES, default=float))
print("\n[saved] research_runs/maker_labels_rr/SELECTIVITY_RESULT.json", flush=True)
