"""A/B/C trio: add fill-predictor C as a THIRD selection factor to the winning apred A^B 1/day deploy.

Baseline = HD3 best result (xgb-20260601_makerlabels_b2_apred): apred-gate B + hold-60s + A^B daily-budget
1/symbol/day -> pooled +3.00 bp, 7/8 net-positive (BNB the only negative). Reproduced here with reused-HP B.

C predicts whether the maker order FILLS (validated: fill AUC BTC 0.73 / LINK 0.64). Use C as the third
factor in the per-day window pick: score = rank(pA) * rank(|pB-0.5|) * rank(Cfac), Cfac = P(fill on B's
chosen side). Test BOTH signs (prefer-fill vs avoid-fill, since fill anti-correlates with favorable move).
Question: does the third factor beat the 2-factor +3.00 baseline (more bp / BNB->positive / 8-of-8)?

Reuses SAVED A (b_universe) + C (fill_model) when present, else trains+saves. Persists all weights/preds.
"""
import argparse, io, json, os, tempfile
import numpy as np, xgboost as xgb, optuna
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; MAIN = "research_runs/xgb_maker"
BU = "research_runs/b_universe"; FM = "research_runs/fill_model"; SAVE = "research_runs/abc"
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def try_load(path):
    try:
        b = xgb.Booster()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        bk.blob(path).download_to_filename(tmp); b.load_model(tmp); os.remove(tmp); return b
    except Exception:
        return None


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


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    starts = np.zeros(len(order), bool); starts[0] = True; starts[1:] = ds[1:] != ds[:-1]
    si = np.where(starts)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def fit(hp, niter, X, y, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def auc(score, lab):
    lab = np.asarray(lab).astype(int); o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def oof_pA(F, yA, trn, day, hpA, kfolds=5):
    tdays = sorted(set(day[trn].tolist())); fold = {dd: i % kfolds for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
    for k in range(kfolds):
        trk = trn & (fday != k); vak = trn & (fday == k)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk)
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


def tune_fill(F, y, gate, day, trials):
    gd = sorted(set(day[gate].tolist())); vc = gd[int(len(gd) * 0.85)] if gd else 0
    innr = gate & (day < vc); innv = gate & (day >= vc)
    spw = float((y[innr] == 0).sum() / max((y[innr] == 1).sum(), 1))
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0,
            "eval_metric": "auc", "scale_pos_weight": spw}
    dtr = xgb.DMatrix(F[innr], label=y[innr]); div = xgb.DMatrix(F[innv], label=y[innv])
    def obj(t):
        p = dict(base, max_depth=t.suggest_int("max_depth", 3, 9),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 300, log=True), reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        b = xgb.train(p, dtr, num_boost_round=300, evals=[(div, "iv")], early_stopping_rounds=20, verbose_eval=False)
        return float(b.best_score)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(obj, n_trials=trials, show_progress_bar=False)
    bp = dict(base, **st.best_params)
    bf = xgb.train(bp, dtr, num_boost_round=300, evals=[(div, "iv")], early_stopping_rounds=20, verbose_eval=False)
    return xgb.train(bp, xgb.DMatrix(F[gate], label=y[gate]), num_boost_round=max(1, bf.best_iteration + 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=SYMS)
    ap.add_argument("--trials", type=int, default=25); ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=1); a = ap.parse_args()
    def log(s): print(s, flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hpB = jload(f"{MAIN}/B_pool.json")
    log(f"[A/B/C trio] baseline=apred A^B {a.budget}/day (reproduce +3.00); +C 3rd factor P(fill on B-side), both signs")
    log(f"{'SYM':5s} {'fillAUC':>7s} | {'base':>6s} {'+C_hi':>6s} {'+C_lo':>6s}  (bp, 1/day filled-only)")
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
        # A: reuse saved (b_universe) or train
        bstA = try_load(f"{BU}/A_{symk}.xgb.json")
        if bstA is None:
            spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
            bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw); save_booster(bstA, f"A_{symk}.xgb.json")
        pA_t = bstA.predict(xgb.DMatrix(F[ti]))
        oof = oof_pA(F, yA, trn, day, hpA, a.kfolds); valid = trn & np.isfinite(oof)
        thrG = float(np.nanquantile(oof[valid], 1 - 0.05)); gate5 = valid & (oof >= thrG) & (fl | fs)
        # B: reproduce b2.py apred B (reused HP)
        yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
        both = fl & fs; wB = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
        wc = np.clip(wB[gate5], 0, np.quantile(wB[gate5][wB[gate5] > 0], 0.99) if (wB[gate5] > 0).any() else 1.0)
        bB = fit(hpB["best_params"], hpB["best_iter"], F[gate5], yB[gate5], w=wc); save_booster(bB, f"B_{symk}.xgb.json")
        pB_t = bB.predict(xgb.DMatrix(F[ti]))
        # C: reuse saved fill models (fill_model) or train
        cl = try_load(f"{FM}/Cfill_long_{symk}.xgb.json") or tune_fill(F, fl.astype(int), gate5, day, a.trials)
        cs = try_load(f"{FM}/Cfill_short_{symk}.xgb.json") or tune_fill(F, fs.astype(int), gate5, day, a.trials)
        if try_load(f"{FM}/Cfill_long_{symk}.xgb.json") is None:
            save_booster(cl, f"Cfill_long_{symk}.xgb.json"); save_booster(cs, f"Cfill_short_{symk}.xgb.json")
        pCL = cl.predict(xgb.DMatrix(F[ti])); pCS = cs.predict(xgb.DMatrix(F[ti]))
        fauc = (auc(pCL, fl[ti]) + auc(pCS, fs[ti])) / 2

        def deploy(score):
            sel = daily_pick(day[ti], score, a.budget); side = pB_t[sel] >= 0.5
            net = np.where(side, nl[ti][sel], ns[ti][sel]); fc = np.where(side, fl[ti][sel], fs[ti][sel])
            ex = fc & np.isfinite(net); return (float(net[ex].mean()) if ex.any() else float("nan")), int(ex.sum())
        r2 = pct_rank(pA_t) * pct_rank(np.abs(pB_t - 0.5))
        Cfill = np.where(pB_t >= 0.5, pCL, pCS)               # P(fill on B's chosen side), per test window
        ev_base, nb = deploy(r2)
        ev_hi, _ = deploy(r2 * pct_rank(Cfill))               # prefer high fill prob
        ev_lo, _ = deploy(r2 * pct_rank(1.0 - Cfill))         # prefer low fill prob (avoid adverse)
        res[symk] = {"fillAUC": fauc, "ev_base": ev_base, "ev_C_hi": ev_hi, "ev_C_lo": ev_lo, "n": nb}
        log(f"{symk:5s} {fauc:7.3f} | {ev_base:+6.2f} {ev_hi:+6.2f} {ev_lo:+6.2f}")
        # persist preds for offline recompute
        buf = io.BytesIO()
        np.savez_compressed(buf, ti=ti.astype(np.int64), pA=pA_t.astype(np.float32), pB=pB_t.astype(np.float32),
                            pCL=pCL.astype(np.float32), pCS=pCS.astype(np.float32), nl=nl[ti].astype(np.float32),
                            ns=ns[ti].astype(np.float32), fl=fl[ti], fs=fs[ti], day=day[ti].astype(np.int32))
        bk.blob(f"{SAVE}/preds_{symk}.npz").upload_from_string(buf.getvalue())
    log("--- POOLED (bp) ---")
    for k in ("ev_base", "ev_C_hi", "ev_C_lo"):
        v = [res[s][k] for s in a.symbols if s in res and np.isfinite(res[s][k])]
        npos = sum(1 for s in a.symbols if s in res and res[s][k] > 0)
        log(f"  {k:9s} = {np.mean(v):+.2f}  ({npos}/{len(v)} symbols positive)")
    bk.blob(f"{SAVE}/ABC_RESULT.json").upload_from_string(json.dumps(res, default=float))
    log(f"[saved] gs://{BUCKET}/{SAVE}/ABC_RESULT.json")


if __name__ == "__main__":
    main()
