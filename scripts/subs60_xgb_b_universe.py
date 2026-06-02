"""B (direction head) TRAINING-UNIVERSE ablation -> pure-maker deploy EV (Optuna-tuned B per variant).

Question (user): B currently trains only on the top-5% A-predicted-non-flat windows (apred gate). Does
training B on a WIDER universe improve the PURE-MAKER deploy result (the apred A^B 1/day +3.00bp), or is
5% (matching the deploy distribution) optimal?

Variants = B's training universe:  "5" / "25" = top-q% of train by OOF-A prediction ; "100" = ALL train
windows (B trained WITHOUT any A-gating, "без A").  B Optuna-tuned per variant (inner-val AUC of the
better-maker-side label).  DEPLOY is unchanged across variants: A^B daily-budget(b/day), hold-60s,
pure-maker, maker-maker 4bp RT, filled-only EV (the metric the +3.00 baseline used). A (vol-gate) and
the selection stay FIXED -> isolates B's training universe. No taker / no toxicity here.
"""
import argparse, io, json, os, tempfile
import numpy as np, xgboost as xgb, optuna
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; MAIN = "research_runs/xgb_maker"
SAVE = "research_runs/b_universe"          # all weights + preds + trials persisted here
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05
bk = storage.Client(project=PROJ).bucket(BUCKET)


def save_booster(b, name):                  # persist model weights to GCS (information asset)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    b.save_model(tmp); bk.blob(f"{SAVE}/{name}").upload_from_filename(tmp); os.remove(tmp)


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


def tune_b(Xtr, ytr, wtr, Xiv, yiv, trials, seed=0):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": seed, "eval_metric": "auc"}
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr); div = xgb.DMatrix(Xiv, label=yiv)
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
    ap.add_argument("--gate-pcts", nargs="+", default=["5", "25", "100"])   # B training universe
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--b-trials", type=int, default=25); ap.add_argument("--kfolds", type=int, default=5)
    a = ap.parse_args()
    def log(s): print(s, flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log(f"[B-universe ablation -> PURE-MAKER] deploy A^B hold-60s maker 4bp | gate_pcts={a.gate_pcts} "
        f"b_trials={a.b_trials} budgets={a.budgets}")
    res = {}
    for symk in a.symbols:
        d = load_rr(symk); hpA = jload(f"{MAIN}/A_{symk}.json")
        F = d["F"]; rH = d["rH"]; day = d["day"]; fee = d["fee"]
        trn, val, te = split(day, d["ndays"])
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        fl = d["fill_long"][0]; fs = d["fill_short"][0]
        nl = d["pnl_long"][0, 0].astype(np.float64) * 100.0 - fee
        ns = d["pnl_short"][0, 0].astype(np.float64) * 100.0 - fee
        vi = np.where(val)[0]; ti = np.where(te)[0]
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
        save_booster(bstA, f"A_{symk}.xgb.json")            # persist A (trained on _rr) for reuse
        pA_t = bstA.predict(xgb.DMatrix(F[ti]))
        oof = oof_pA(F, yA, trn, day, hpA, a.kfolds); valid = trn & np.isfinite(oof)
        # FIXED test pool = A-top-5% (A-predicted non-flat); threshold from OOF-A on train (calibrated,
        # same definition as the apred 5% B-gate). Identical across ALL B-variants (A fixed) -> isolates B-universe.
        thrA = float(np.nanquantile(oof[valid], 1 - NF_RATE))
        amask = pA_t >= thrA; pool_idx = ti[amask]
        # B target + weights (all windows; gate only selects which rows B trains on)
        yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
        both = fl & fs
        wB = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
        log(f"=== {symk} (train={int(trn.sum())} test={int(te.sum())} | A-top5% test pool={int(amask.sum())} "
            f"~{100*amask.mean():.1f}%) ===")
        log(f"{'gate':>4s} {'B_n':>7s} {'B_dirAUC':>8s} | " + " ".join(f"b{b}".rjust(7) for b in a.budgets) + "   poolEV")
        gate_meta = {}; pB_by_gate = {}
        for gp in a.gate_pcts:
            if gp in ("100", "all"):
                b_gate = trn & (fl | fs)
            else:
                q = float(gp); thrB = float(np.nanquantile(oof[valid], 1 - q / 100.0))
                b_gate = valid & (oof >= thrB) & (fl | fs)
            tdays = sorted(set(day[b_gate].tolist())); vc = tdays[int(len(tdays) * 0.85)] if tdays else 0
            innr = b_gate & (day < vc); innv = b_gate & (day >= vc)
            if innr.sum() < 300 or innv.sum() < 100:
                log(f"{gp:>4s}: too few B windows"); continue
            clip = lambda m: np.clip(wB[m], 0, np.quantile(wB[m][wB[m] > 0], 0.99) if (wB[m] > 0).any() else 1.0)
            bp, nr = tune_b(F[innr], yB[innr], clip(innr), F[innv], yB[innv], a.b_trials)
            bB = xgb.train(bp, xgb.DMatrix(F[b_gate], label=yB[b_gate], weight=clip(b_gate)), num_boost_round=nr)
            save_booster(bB, f"B_{symk}_g{gp}.xgb.json")       # persist each Optuna-tuned B
            gate_meta[gp] = {"best_params": {k: v for k, v in bp.items() if k not in ("objective", "tree_method", "nthread", "seed", "eval_metric")},
                             "best_iter": int(nr), "B_n": int(b_gate.sum())}
            pB_t = bB.predict(xgb.DMatrix(F[ti]))
            pBp = pB_t[amask]                                   # B on the FIXED A-top-5% pool (same for all variants)
            pB_by_gate[gp] = pBp
            dir_auc = auc(pBp, yB[pool_idx]); confp = np.abs(pBp - 0.5)
            evs = {}
            for bud in a.budgets:                              # daily-budget by B-confidence WITHIN the A-pool
                sel = daily_pick(day[pool_idx], confp, bud); gi = pool_idx[sel]; side = pBp[sel] >= 0.5
                net = np.where(side, nl[gi], ns[gi]); fc = np.where(side, fl[gi], fs[gi]); ex = fc & np.isfinite(net)
                evs[bud] = float(net[ex].mean()) if ex.any() else float("nan")
                res[(symk, gp, bud)] = {"B_n": int(b_gate.sum()), "dir_auc": dir_auc, "ev_maker": evs[bud], "n": int(ex.sum())}
            sidew = pBp >= 0.5; netw = np.where(sidew, nl[pool_idx], ns[pool_idx]); fcw = np.where(sidew, fl[pool_idx], fs[pool_idx])
            exw = fcw & np.isfinite(netw); pool_ev = float(netw[exw].mean()) if exw.any() else float("nan")   # all A-pool, B's side
            res[(symk, gp, "pool")] = {"B_n": int(b_gate.sum()), "dir_auc": dir_auc, "ev_maker": pool_ev, "n": int(exw.sum())}
            log(f"{gp:>4s} {int(b_gate.sum()):7d} {dir_auc:8.3f} | " + " ".join(f"{evs[b]:+7.2f}" for b in a.budgets) + f"  {pool_ev:+7.2f}")
        # persist test-pool predictions + payoffs + per-gate B preds + Optuna params -> recompute any EV offline, no retrain
        preds = {"pool_idx": pool_idx.astype(np.int64), "day": day[pool_idx].astype(np.int32),
                 "pA": pA_t[amask].astype(np.float32), "yB": yB[pool_idx].astype(np.int8),
                 "nl": nl[pool_idx].astype(np.float32), "ns": ns[pool_idx].astype(np.float32),
                 "fl": fl[pool_idx], "fs": fs[pool_idx], "rH": rH[pool_idx].astype(np.float32),
                 "thrA": np.float64(thrA)}
        for gp, pb in pB_by_gate.items():
            preds[f"pB_g{gp}"] = pb.astype(np.float32)
        buf = io.BytesIO(); np.savez_compressed(buf, meta=np.array(json.dumps(gate_meta)), **preds)
        bk.blob(f"{SAVE}/preds_{symk}.npz").upload_from_string(buf.getvalue())
        log(f"  [saved] gs://{BUCKET}/{SAVE}/{{A_{symk}.xgb.json, B_{symk}_g*.xgb.json, preds_{symk}.npz}}")
    log("--- POOLED pure-maker EV on FIXED A-top-5% pool (by gate_pct x budget) ---")
    for gp in a.gate_pcts:
        line = f"  gate={gp:>4s}:"
        for bud in list(a.budgets) + ["pool"]:
            rows = [res[(s, gp, bud)] for s in a.symbols if (s, gp, bud) in res]
            if rows:
                line += f"  {('b'+str(bud)) if bud!='pool' else 'POOL'}={np.mean([r['ev_maker'] for r in rows]):+.2f}"
        line += f"   (B_dirAUC {np.mean([res[(s,gp,a.budgets[0])]['dir_auc'] for s in a.symbols if (s,gp,a.budgets[0]) in res]):.3f})"
        log(line)
    bk.blob(f"{RR}/B_UNIVERSE_ABLATION.json").upload_from_string(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, default=float))
    log(f"[saved] gs://{BUCKET}/{RR}/B_UNIVERSE_ABLATION.json")


if __name__ == "__main__":
    main()
