"""Model C: FILL PREDICTOR — does a resting maker limit get HIT or MISS? (the near-term order-flow signal)

Premise: B (60s-direction) and abstention failed because the FORWARD PRICE OUTCOME is noise (AUC ~0.51).
But FILL (will my passive bid/ask be hit within the ~12s entry window) is a NEAR-TERM order-flow event ->
structurally more predictable (flow autocorrelation). And fill ANTI-correlates with favorable direction
(the eager-to-fill side is adverse). So a fill model may extract the direction B couldn't.

Two decisive tests on the FIXED A-top-5% pool:
  1. PREDICTABILITY: AUC of C_long vs fill_long, C_short vs fill_short. (Is fill predictable at all?)
  2. DIRECTIONALITY: rank-IC of fill-asymmetry (pC_long - pC_short) vs maker-pnl-asymmetry (nl - ns).
Plus a monetization probe: maker EV using the fill-derived side vs B's side.

Reuses SAVED A + B_g5 (no retrain). Trains C_long, C_short (Optuna). Persists weights + preds + metrics.
"""
import argparse, io, json, os, tempfile
import numpy as np, xgboost as xgb, optuna
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; MAIN = "research_runs/xgb_maker"; BU = "research_runs/b_universe"
SAVE = "research_runs/fill_model"
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_booster(path):
    b = xgb.Booster()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    bk.blob(path).download_to_filename(tmp); b.load_model(tmp); os.remove(tmp); return b


def save_booster(b, name):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    b.save_model(tmp); bk.blob(f"{SAVE}/{name}").upload_from_filename(tmp); os.remove(tmp)


def load_rr(symk):
    d = np.load(io.BytesIO(bk.blob(f"{RR}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pnl_long": d["pnl_long"].astype(np.float32), "pnl_short": d["pnl_short"].astype(np.float32),
            "fill_long": d["fill_long"].astype(bool), "fill_short": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def split(day, ndays):
    cut = int(ndays * SPLIT[0]); emb = int(ndays * SPLIT[1]); tr = day < cut
    td = sorted(set(day[tr].tolist())); vcut = td[int(len(td) * SPLIT[2])] if td else cut
    return (tr & (day < vcut)), (tr & (day >= vcut)), (day >= emb)


def daily_pick(day, score, n_per_day=1):
    order = np.lexsort((-score, day)); ds = day[order]
    starts = np.zeros(len(order), bool); starts[0] = True; starts[1:] = ds[1:] != ds[:-1]
    start_idx = np.where(starts)[0]
    within = np.arange(len(order)) - np.repeat(start_idx, np.diff(np.append(start_idx, len(order))))
    return order[within < n_per_day]


def fit(hp, niter, X, y, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y), num_boost_round=max(1, niter + 1))


def auc(score, lab):
    lab = np.asarray(lab).astype(int); o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def rank_ic(x, y):
    if len(x) < 20:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def oof_pA(F, yA, trn, day, hpA, kfolds=5):
    tdays = sorted(set(day[trn].tolist()))
    fold = {dd: i % kfolds for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day])
    oof = np.full(len(F), np.nan)
    for k in range(kfolds):
        trk = trn & (fday != k); vak = trn & (fday == k)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk)
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


def tune(Xtr, ytr, Xiv, yiv, trials, spw, seed=0):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": seed,
            "eval_metric": "auc", "scale_pos_weight": spw}
    dtr = xgb.DMatrix(Xtr, label=ytr); div = xgb.DMatrix(Xiv, label=yiv)
    def obj(t):
        p = dict(base, max_depth=t.suggest_int("max_depth", 3, 9),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 300, log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        b = xgb.train(p, dtr, num_boost_round=300, evals=[(div, "iv")], early_stopping_rounds=20, verbose_eval=False)
        return float(b.best_score)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=trials, show_progress_bar=False)
    bp = dict(base, **st.best_params)
    bf = xgb.train(bp, dtr, num_boost_round=300, evals=[(div, "iv")], early_stopping_rounds=20, verbose_eval=False)
    return bp, max(1, bf.best_iteration + 1)


def train_C(F, y, gate, day, trials):
    gdays = sorted(set(day[gate].tolist())); vc = gdays[int(len(gdays) * 0.85)] if gdays else 0
    innr = gate & (day < vc); innv = gate & (day >= vc)
    spw = float((y[innr] == 0).sum() / max((y[innr] == 1).sum(), 1))
    bp, nr = tune(F[innr], y[innr], F[innv], y[innv], trials, spw)
    b = xgb.train(bp, xgb.DMatrix(F[gate], label=y[gate]), num_boost_round=nr)
    return b, bp, nr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["BTC", "LINK"])
    ap.add_argument("--trials", type=int, default=25); ap.add_argument("--kfolds", type=int, default=5)
    a = ap.parse_args()
    def log(s): print(s, flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log(f"[fill-model C] predict fill_long/short on A-non-flat (Optuna {a.trials}); reuse saved A+B_g5; maker 4bp")
    res = {}
    for symk in a.symbols:
        d = load_rr(symk); hpA = jload(f"{MAIN}/A_{symk}.json")
        F = d["F"]; rH = d["rH"]; day = d["day"]; fee = d["fee"]
        trn, val, te = split(day, d["ndays"])
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        fl = d["fill_long"][0]; fs = d["fill_short"][0]
        nl = d["pnl_long"][0, 0].astype(np.float64) * 100.0 - fee
        ns = d["pnl_short"][0, 0].astype(np.float64) * 100.0 - fee
        ti = np.where(te)[0]
        bstA = load_booster(f"{BU}/A_{symk}.xgb.json")
        pA_t = bstA.predict(xgb.DMatrix(F[ti]))
        oof = oof_pA(F, yA, trn, day, hpA, a.kfolds); valid = trn & np.isfinite(oof)
        thrA = float(np.nanquantile(oof[valid], 1 - NF_RATE))
        amask = pA_t >= thrA; pool = ti[amask]
        gate5 = valid & (oof >= thrA) & (fl | fs)
        # ---- C: fill predictors ----
        cl, bpl, nrl = train_C(F, fl.astype(int), gate5, day, a.trials)
        cs, bps, nrs = train_C(F, fs.astype(int), gate5, day, a.trials)
        pCL = cl.predict(xgb.DMatrix(F[pool])); pCS = cs.predict(xgb.DMatrix(F[pool]))
        aucL = auc(pCL, fl[pool]); aucS = auc(pCS, fs[pool])
        # ---- directionality: fill-asymmetry vs maker-pnl-asymmetry, on filled-both pool windows ----
        fb = fl[pool] & fs[pool]
        fill_sig = pCL - pCS
        ic = rank_ic(fill_sig[fb], (nl[pool] - ns[pool])[fb])
        # also: train-side IC to fix the side sign without test fishing
        pCL_tr = cl.predict(xgb.DMatrix(F[gate5])); pCS_tr = cs.predict(xgb.DMatrix(F[gate5]))
        fbt = fl[gate5] & fs[gate5]
        ic_tr = rank_ic((pCL_tr - pCS_tr)[fbt], (nl[gate5] - ns[gate5])[fbt])
        sgn = 1.0 if (np.isnan(ic_tr) or ic_tr >= 0) else -1.0   # sign of how fill_sig maps to long-better
        side_C = (sgn * fill_sig) >= 0                            # True=long
        # ---- monetization probe: maker EV side_C vs side_B on the pool ----
        bB = load_booster(f"{BU}/B_{symk}_g5.xgb.json")
        pBp = bB.predict(xgb.DMatrix(F[pool])); side_B = pBp >= 0.5

        def ev_side(side, sel=None):
            idx = np.arange(len(pool)) if sel is None else sel
            net = np.where(side[idx], nl[pool][idx], ns[pool][idx]); fc = np.where(side[idx], fl[pool][idx], fs[pool][idx])
            ex = fc & np.isfinite(net); return (float(net[ex].mean()) if ex.any() else float("nan")), int(ex.sum())
        evB_full, nB = ev_side(side_B); evC_full, nC = ev_side(side_C)
        # 1/day (positive regime): B by |pB-0.5|, C by |fill_sig|
        selB = daily_pick(day[pool], np.abs(pBp - 0.5), 1); selC = daily_pick(day[pool], np.abs(fill_sig), 1)
        evB_1d, _ = ev_side(side_B, selB); evC_1d, _ = ev_side(side_C, selC)
        agree = float((side_C == side_B).mean())
        res[symk] = {"pool": int(amask.sum()), "fillrate_L": float(fl[pool].mean()), "fillrate_S": float(fs[pool].mean()),
                     "aucL": aucL, "aucS": aucS, "ic_fill_dir": ic, "ic_train": ic_tr, "sign": sgn,
                     "evB_full": evB_full, "evC_full": evC_full, "evB_1day": evB_1d, "evC_1day": evC_1d,
                     "C_vs_B_side_agree": agree}
        log(f"=== {symk}: pool={int(amask.sum())} fillrate L/S={fl[pool].mean():.2f}/{fs[pool].mean():.2f} ===")
        log(f"  [1] FILL AUC: long {aucL:.3f} | short {aucS:.3f}   <- is fill predictable?")
        log(f"  [2] fill-asym -> dir rank-IC (pool) {ic:+.3f} (train {ic_tr:+.3f})   <- does fill carry direction?")
        log(f"  [3] maker EV full-pool: side_B {evB_full:+.2f} vs side_C {evC_full:+.2f} | 1/day: B {evB_1d:+.2f} vs C {evC_1d:+.2f} | agree {agree:.2f}")
        save_booster(cl, f"Cfill_long_{symk}.xgb.json"); save_booster(cs, f"Cfill_short_{symk}.xgb.json")
        buf = io.BytesIO()
        np.savez_compressed(buf, pool_idx=pool.astype(np.int64), pCL=pCL.astype(np.float32), pCS=pCS.astype(np.float32),
                            pB=pBp.astype(np.float32), fl=fl[pool], fs=fs[pool], nl=nl[pool].astype(np.float32),
                            ns=ns[pool].astype(np.float32), day=day[pool].astype(np.int32), rH=rH[pool].astype(np.float32),
                            meta=np.array(json.dumps({"aucL": aucL, "aucS": aucS, "ic": ic, "bp_long": {k: v for k, v in bpl.items() if k not in ("objective","tree_method","nthread","seed","eval_metric","scale_pos_weight")}, "bp_short": {k: v for k, v in bps.items() if k not in ("objective","tree_method","nthread","seed","eval_metric","scale_pos_weight")}}, default=float)))
        bk.blob(f"{SAVE}/preds_{symk}.npz").upload_from_string(buf.getvalue())
        log(f"  [saved] gs://{BUCKET}/{SAVE}/{{Cfill_long/short_{symk}.xgb.json, preds_{symk}.npz}}")
    log("--- POOLED ---")
    for k, lab in [("aucL", "fillAUC_long"), ("aucS", "fillAUC_short"), ("ic_fill_dir", "fill->dir rank-IC"),
                   ("evB_full", "EV side_B (full)"), ("evC_full", "EV side_C (full)"),
                   ("evB_1day", "EV side_B (1/day)"), ("evC_1day", "EV side_C (1/day)")]:
        v = [res[s][k] for s in a.symbols if s in res and np.isfinite(res[s][k])]
        if v:
            log(f"  {lab:20s} = {np.mean(v):+.3f}")
    bk.blob(f"{SAVE}/FILL_RESULT.json").upload_from_string(json.dumps(res, default=float))
    log(f"[saved] gs://{BUCKET}/{SAVE}/FILL_RESULT.json")


if __name__ == "__main__":
    main()
