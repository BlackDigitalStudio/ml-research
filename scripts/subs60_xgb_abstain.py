"""Smarter-maker (B): TOXICITY-GATED ABSTENTION — stay maker (4bp), SKIP predicted-adverse fills.

Lesson from the maker<->taker switch: adverse passive fills ARE predictable (tox AUC ~0.75 BTC) but
crossing to taker costs too much (fee wall). Fix: use the same signal to ABSTAIN (don't post) on toxic
windows -- abstention is free. Trade only benign-flow windows, all maker-maker 4bp.

Reuses SAVED models (no retrain of the expensive ones): A_{SYM}.xgb.json + the optimal B_{SYM}_g5.xgb.json
from research_runs/b_universe/. Trains ONLY the new abstention model (Optuna). A (vol-gate) re-derives OOF
for the gate/pool definition (intrinsic). Persists abstention weights + P(adv) preds + curves + trials.

Abstention target (train, A-non-flat apred-5% windows, better-maker side filled): y_adv = 1{maker_pnl < 0}.
Core test on the FIXED A-top-5% test pool: does ranking trades by LOW P(adv) beat ranking by B-confidence
at equal retained-trade count (orthogonality)? Plus deploy: net maker EV + trades/day vs no-abstention.
"""
import argparse, io, json, os, tempfile
import numpy as np, xgboost as xgb, optuna
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; MAIN = "research_runs/xgb_maker"; BU = "research_runs/b_universe"
SAVE = "research_runs/abstain"
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


def pct_rank(x):
    o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


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
    lab = lab.astype(int); o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


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


def ev_topf(rank_desc, traded_pnl, filled, f):
    """EV over the top-f fraction of windows by rank_desc (higher=keep), filled-only mean."""
    n = max(1, int(round(len(rank_desc) * f)))
    keep = np.argsort(-rank_desc)[:n]
    ex = filled[keep] & np.isfinite(traded_pnl[keep])
    return (float(traded_pnl[keep][ex].mean()) if ex.any() else float("nan")), int(ex.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["BTC", "LINK"])
    ap.add_argument("--trials", type=int, default=25); ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1]); a = ap.parse_args()
    def log(s): print(s, flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log(f"[abstain] reuse saved A + B_g5; train abstention model (trials={a.trials}); maker-maker 4bp; "
        f"orthogonality EV(top-f) conf vs benign vs combo, fracs={a.fracs}")
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
        # A: reuse saved booster for pA; OOF for gate/pool threshold (intrinsic)
        bstA = load_booster(f"{BU}/A_{symk}.xgb.json")
        pA_t = bstA.predict(xgb.DMatrix(F[ti]))
        oof = oof_pA(F, yA, trn, day, hpA, a.kfolds); valid = trn & np.isfinite(oof)
        thrA = float(np.nanquantile(oof[valid], 1 - NF_RATE))
        amask = pA_t >= thrA; pool = ti[amask]
        gate5 = valid & (oof >= thrA) & (fl | fs)              # abstention train universe = apred 5% (matched)
        # B: reuse saved optimal B_g5 -> side + confidence on the pool
        bB = load_booster(f"{BU}/B_{symk}_g5.xgb.json")
        pBp = bB.predict(xgb.DMatrix(F[pool]))
        side = pBp >= 0.5
        traded = np.where(side, nl[pool], ns[pool]); tfill = np.where(side, fl[pool], fs[pool])
        conf = np.abs(pBp - 0.5)
        # Abstention model: target on train A-non-flat better-side filled windows
        better = (nl > ns); bpnl = np.where(better, nl, ns); bfill = np.where(better, fl, fs)
        amask_tr = gate5 & bfill
        y_adv = (bpnl < 0).astype(int)
        adv_days = sorted(set(day[amask_tr].tolist())); vc = adv_days[int(len(adv_days) * 0.85)] if adv_days else 0
        innr = amask_tr & (day < vc); innv = amask_tr & (day >= vc)
        spw = float((y_adv[innr] == 0).sum() / max((y_adv[innr] == 1).sum(), 1))
        bp, nr = tune(F[innr], y_adv[innr], F[innv], y_adv[innv], a.trials, spw)
        bAdv = xgb.train(bp, xgb.DMatrix(F[amask_tr], label=y_adv[amask_tr]), num_boost_round=nr)
        padv = bAdv.predict(xgb.DMatrix(F[pool]))              # P(adverse) on test pool
        # diagnostic: does P(adv) predict losing trades on the pool (B's side)?
        y_adv_pool = (traded < 0).astype(int)
        adv_auc = auc(padv[tfill], y_adv_pool[tfill]) if tfill.any() else float("nan")  # among traded(filled)
        benign = 1.0 - padv
        combo = pct_rank(conf) * pct_rank(benign)
        log(f"=== {symk}: pool={int(amask.sum())} (~{100*amask.mean():.1f}%) | abstAUC(loser) {adv_auc:.3f} | "
            f"base EV(all,B-side,filled)={float(traded[tfill][np.isfinite(traded[tfill])].mean()):+.2f}bp ===")
        log(f"{'frac':>5s} {'n':>5s} | {'byConf':>7s} {'byBenign':>8s} {'byCombo':>7s}")
        save_curves = {}
        for f in a.fracs:
            ev_c, nc = ev_topf(conf, traded, tfill, f)
            ev_b, nb = ev_topf(benign, traded, tfill, f)
            ev_x, nx = ev_topf(combo, traded, tfill, f)
            res[(symk, f)] = {"ev_conf": ev_c, "ev_benign": ev_b, "ev_combo": ev_x, "n": nc, "adv_auc": adv_auc}
            save_curves[f] = res[(symk, f)]
            log(f"{f:5.2f} {nc:5d} | {ev_c:+7.2f} {ev_b:+8.2f} {ev_x:+7.2f}")
        # persist: abstention model + pool preds + curves
        save_booster(bAdv, f"Adv_{symk}.xgb.json")
        buf = io.BytesIO()
        np.savez_compressed(buf, pool_idx=pool.astype(np.int64), padv=padv.astype(np.float32),
                            pB=pBp.astype(np.float32), traded=traded.astype(np.float32), tfill=tfill,
                            day=day[pool].astype(np.int32), rH=rH[pool].astype(np.float32),
                            meta=np.array(json.dumps({"best_params": {k: v for k, v in bp.items()
                                if k not in ("objective", "tree_method", "nthread", "seed", "eval_metric", "scale_pos_weight")},
                                "best_iter": int(nr), "adv_auc": adv_auc, "curves": save_curves}, default=float)))
        bk.blob(f"{SAVE}/preds_{symk}.npz").upload_from_string(buf.getvalue())
        log(f"  [saved] gs://{BUCKET}/{SAVE}/{{Adv_{symk}.xgb.json, preds_{symk}.npz}}")
    log("--- POOLED EV(top-f) conf vs benign vs combo ---")
    for f in a.fracs:
        rows = [res[(s, f)] for s in a.symbols if (s, f) in res]
        if rows:
            log(f"  f={f:4.2f}: conf {np.mean([r['ev_conf'] for r in rows]):+.2f} | "
                f"benign {np.mean([r['ev_benign'] for r in rows]):+.2f} | combo {np.mean([r['ev_combo'] for r in rows]):+.2f}")
    bk.blob(f"{SAVE}/ABSTAIN_RESULT.json").upload_from_string(json.dumps({f"{k[0]}|{k[1]}": v for k, v in res.items()}, default=float))
    log(f"[saved] gs://{BUCKET}/{SAVE}/ABSTAIN_RESULT.json")


if __name__ == "__main__":
    main()
