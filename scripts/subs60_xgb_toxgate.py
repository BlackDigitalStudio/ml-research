"""Toxicity-gated maker<->taker switching + TOX-TRAINING-UNIVERSE ablation (Optuna).

Decision per traded window: REST passive (maker @bid, 4bp RT, may MISS/adverse) vs CROSS (taker @ask,
7bp RT = 5 taker-entry + 2 maker-exit, always fills). A toxicity model predicts P(cross beats rest) from
71 decision-time feats; a val-tuned tau gates maker vs taker. Reuses apred A^B daily-budget deploy.

ABLATION (this version): vary ONLY the TOX model's TRAINING universe, Optuna-tuned each:
  * "5"/"25": top-q% of train by OOF-A prediction (A-gated, q%).
  * "100":   ALL train windows = tox trained WITHOUT any A-gating ("без A" for tox training).
B (direction, apred 5%) AND the DEPLOY selection (A^B daily-budget, A-top windows) stay FIXED across
ALL variants -> isolates purely the tox-training axis. Deploy always uses A on 5% windows.

v0 approximations (smoke): per-symbol CONSTANT half-spread; taker pnl from rH60 (entry@ask, exit@mid
timeout, same hold-60s exit as maker -> isolates ENTRY decision); tox label uses realized better-maker
side; EV basis = mean over the same selected 1/day windows, MISS->0.
"""
import argparse, io, json
import numpy as np, xgboost as xgb, optuna
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; MAIN = "research_runs/xgb_maker"
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05
TAKER_ENTRY_FEE = 5.0; MAKER_SIDE_FEE = 2.0
HS_BP = {"BTC": 0.006, "LINK": 0.30, "ETH": 0.02, "BNB": 0.05, "SOL": 0.10, "LTC": 0.20, "XRP": 0.30, "DOGE": 0.40}
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


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


def fit(hp, niter, X, y, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


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


def tune_tox(Xtr, ytr, Xiv, yiv, trials, seed=0):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": seed, "eval_metric": "auc"}
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["BTC", "LINK"])
    ap.add_argument("--variants", nargs="+", default=["5", "25", "100"])  # tox-training universe (100 = no A-gating)
    ap.add_argument("--gate-pct", type=float, default=5.0)        # B's apred gate (FIXED)
    ap.add_argument("--tox-trials", type=int, default=25)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--kfolds", type=int, default=5); a = ap.parse_args()
    def log(s): print(s, flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TAKER_RT = TAKER_ENTRY_FEE + MAKER_SIDE_FEE
    hpB = jload(f"{MAIN}/B_pool.json")
    log(f"[tox-ablation] taker_RT={TAKER_RT}bp | maker_RT=4bp | B_gate=apred{a.gate_pct}% FIXED | "
        f"variants={a.variants} tox_trials={a.tox_trials} budgets={a.budgets}")
    res = {}
    for symk in a.symbols:
        d = load_rr(symk); hpA = jload(f"{MAIN}/A_{symk}.json")
        F = d["F"]; rH = d["rH"]; day = d["day"]; fee = d["fee"]; HS = HS_BP[symk]
        trn, val, te = split(day, d["ndays"])
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        fl = d["fill_long"][0]; fs = d["fill_short"][0]
        nl = d["pnl_long"][0, 0].astype(np.float64) * 100.0 - fee
        ns = d["pnl_short"][0, 0].astype(np.float64) * 100.0 - fee
        tk_l = rH - HS - TAKER_RT; tk_s = -rH - HS - TAKER_RT
        vi = np.where(val)[0]; ti = np.where(te)[0]
        # A
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
        pA_v = bstA.predict(xgb.DMatrix(F[vi])); pA_t = bstA.predict(xgb.DMatrix(F[ti]))
        # B's apred gate (FIXED 5%) + B
        oof = oof_pA(F, yA, trn, day, hpA, a.kfolds)
        valid = trn & np.isfinite(oof)
        thrB = float(np.nanquantile(oof[valid], 1 - a.gate_pct / 100.0)) if valid.any() else np.inf
        b_gate = valid & (oof >= thrB) & (fl | fs)
        yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
        both = fl & fs
        wB = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
        wc = np.clip(wB[b_gate], 0, np.quantile(wB[b_gate][wB[b_gate] > 0], 0.99) if (wB[b_gate] > 0).any() else 1.0)
        bB = fit(hpB["best_params"], hpB["best_iter"], F[b_gate], yB[b_gate], w=wc)
        pB_v = bB.predict(xgb.DMatrix(F[vi])); pB_t = bB.predict(xgb.DMatrix(F[ti]))
        # tox label (all windows): realized better-maker side, MISS->0 maker vs always-fill taker
        side_star = (nl > ns)
        maker_star0 = np.where(side_star, np.where(fl, nl, 0.0), np.where(fs, ns, 0.0))
        taker_star = np.where(side_star, tk_l, tk_s)
        y_tox = (taker_star > maker_star0).astype(int)
        log(f"=== {symk} (train={int(trn.sum())} test={int(te.sum())}, y_tox base-rate train={y_tox[trn].mean():.3f}) ===")
        log(f"{'var':>4s} {'tox_n':>7s} {'bud':>3s} {'cross%':>6s} {'tau':>4s} {'toxAUC':>6s} | "
            f"{'val mk->gat':>11s} | {'TEST mk':>7s} {'taker':>7s} {'GATED':>7s}")
        for var in a.variants:
            if var in ("100", "noa"):
                tox_tr = trn.copy()
            else:
                q = float(var); thrT = float(np.nanquantile(oof[valid], 1 - q / 100.0))
                tox_tr = valid & (oof >= thrT)
            tox_tr = tox_tr & (fl | fs)
            # inner split by day for Optuna
            tdays = sorted(set(day[tox_tr].tolist())); vc = tdays[int(len(tdays) * 0.85)] if tdays else 0
            innr = tox_tr & (day < vc); innv = tox_tr & (day >= vc)
            if innr.sum() < 300 or innv.sum() < 100:
                log(f"{var:>4s}: too few tox windows"); continue
            bp, nr = tune_tox(F[innr], y_tox[innr], F[innv], y_tox[innv], a.tox_trials)
            btox = xgb.train(bp, xgb.DMatrix(F[tox_tr], label=y_tox[tox_tr]), num_boost_round=nr)
            pc_v = btox.predict(xgb.DMatrix(F[vi])); pc_t = btox.predict(xgb.DMatrix(F[ti]))
            tox_auc = auc(pc_v, y_tox[vi])

            def select(islice, pA_s, pB_s, pc_s, budget):           # deploy selection: A^B (FIXED, A always on)
                score = pct_rank(pA_s) * pct_rank(np.abs(pB_s - 0.5))
                sel = daily_pick(day[islice], score, budget); gi = islice[sel]; side = pB_s[sel] >= 0.5
                mk0 = np.where(side, np.where(fl[gi], nl[gi], 0.0), np.where(fs[gi], ns[gi], 0.0))
                tk = np.where(side, tk_l[gi], tk_s[gi])
                return mk0, tk, pc_s[sel]
            for bud in a.budgets:
                mk0_v, tk_v, pcv = select(vi, pA_v, pB_v, pc_v, bud)
                mk0_t, tk_t, pct = select(ti, pA_t, pB_t, pc_t, bud)
                taus = np.unique(np.round(np.quantile(pcv, np.linspace(0, 1, 21)), 4))
                best_tau, best_ev = 1.01, -1e9
                for tau in taus:
                    ev = float(np.where(pcv >= tau, tk_v, mk0_v).mean())
                    if ev > best_ev:
                        best_ev, best_tau = ev, float(tau)
                gated_t = np.where(pct >= best_tau, tk_t, mk0_t)
                r = {"tox_n": int(tox_tr.sum()), "cross_frac": float((pct >= best_tau).mean()),
                     "tau": best_tau, "tox_auc": tox_auc, "val_maker": float(mk0_v.mean()), "val_gated": best_ev,
                     "ev_maker": float(mk0_t.mean()), "ev_taker": float(tk_t.mean()), "ev_gated": float(gated_t.mean())}
                res[(symk, var, bud)] = r
                log(f"{var:>4s} {r['tox_n']:7d} {bud:3d} {r['cross_frac']*100:5.0f}% {best_tau:4.2f} "
                    f"{tox_auc:6.3f} | {r['val_maker']:+5.2f}->{best_ev:+5.2f} | {r['ev_maker']:+7.2f} "
                    f"{r['ev_taker']:+7.2f} {r['ev_gated']:+7.2f}")
    # pooled per (variant,budget)
    log("--- POOLED across symbols (maker / taker / gated) ---")
    for var in a.variants:
        for bud in a.budgets:
            rows = [res[(s, var, bud)] for s in a.symbols if (s, var, bud) in res]
            if not rows:
                continue
            mk = np.mean([r["ev_maker"] for r in rows]); tk = np.mean([r["ev_taker"] for r in rows])
            gt = np.mean([r["ev_gated"] for r in rows]); ta = np.mean([r["tox_auc"] for r in rows])
            log(f"  var={var:>4s} bud={bud:2d}: toxAUC {ta:.3f} | maker {mk:+.2f} | taker {tk:+.2f} | GATED {gt:+.2f}")
    bk.blob(f"{RR}/TOXGATE_ABLATION.json").upload_from_string(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, default=float))
    log(f"[saved] gs://{BUCKET}/{RR}/TOXGATE_ABLATION.json")


if __name__ == "__main__":
    main()
