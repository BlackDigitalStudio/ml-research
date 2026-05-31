#!/usr/bin/env python3
"""XGBoost A/B cascade on MAKER-REALISTIC (adverse-selection) labels — TRAIN + full artifact capture.

Reads gs://.../research_runs/maker_labels/{SYM}.npz (subs60_makerlabel_build.py):
  F(N,71) feats | rH60(N) | day(N) | ts(N) | pnl_long/pnl_short (NC,QM,N) % NaN=miss |
  fill_long/fill_short (QM,N) u8 | meta.

This script ONLY trains + SAVES EVERYTHING (capture-all-information). It does NOT compute the
final surface — that is done OFFLINE by subs60_xgb_surface.py from the saved per-sample test
predictions, so any operating point / config / honest-cascade cut is recomputable WITHOUT
retraining (and the honest A-gated cascade needs pA & pB on the same test rows, which we save).

Model A (per-symbol vol-gate): target |rH60| >= per-symbol TRAIN p(1-nf_rate) quantile
  (vol-adaptive: 13bp is 2.4 sigma for BTC vs 1.25 for LINK). XGBClassifier; Optuna(max val-AUC)
  on a train SUBSAMPLE (all pos + capped neg) then FINAL fit on FULL train (nthread=8).
Model B (pooled direction): target 1{net maker pnl_long > pnl_short} on chosen cfg (hold-60s,
  qm=1 default); net=gross_bp-fee; on >=1-side-fill windows; trains on per-symbol non-flat
  windows + symbol-id feat; sample_weight = econ |Δnet| (executable-only, HM5 R1). Optuna max
  val executed-EV.

SAVES -> research_runs/xgb_maker/:
  A_{SYM}.json / B_pool.json   : metrics + best_params + ALL Optuna trials + ALL importances
                                 (gain/weight/cover/total_gain/total_cover) + per-round val curve + manifest
  A_{SYM}.xgb.json / B_pool.xgb.json : boosters
  preds_{SYM}.npz              : per TEST sample -> ts, day, sid, rH60, yA, pA, pB,
                                 pnl_long/short (NC,QM,n), fill_long/short (QM,n)  [ALL cfg x qm]
  MANIFEST.json                : run-level args/splits/seed/provenance/code tag
Run: python3 subs60_xgb_makerlabel.py --symbols ALL --trials 40 --seed 0
"""
import argparse, io, json, time
import numpy as np
from google.cloud import storage
import xgboost as xgb
import optuna

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SRC = "research_runs/maker_labels"; OUT = "research_runs/xgb_maker"
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
SCRIPT_TAG = "subs60_xgb_makerlabel.py@2026-05-31-rev2-instrumented"
SPLIT = (0.65, 0.68, 0.85)   # train<0.65 ndays ; embargo gap [0.65,0.68) ; test>=0.68 ; val=last 15% of train days
IMP_TYPES = ["gain", "weight", "cover", "total_gain", "total_cover"]
bk = storage.Client(project=PROJ).bucket(BUCKET)
optuna.logging.set_verbosity(optuna.logging.WARNING)
NAMES = ([f"x{c}" for c in range(64)] +
         ["btc_ret5", "btc_ret30", "btc_ret60", "sin_h", "cos_h", "sin_f8", "cos_f8"])


def load(sym):
    d = np.load(io.BytesIO(bk.blob(f"{SRC}/{sym.split('-')[0]}.npz").download_as_bytes()), allow_pickle=True)
    return d, json.loads(str(d["meta"]))


def split(day, ndays):
    cut = int(ndays * SPLIT[0]); emb = int(ndays * SPLIT[1])
    tr = day < cut; te = day >= emb
    tr_days = sorted(set(day[tr].tolist()))
    vcut = tr_days[int(len(tr_days) * SPLIT[2])] if tr_days else cut
    val = tr & (day >= vcut); trn = tr & (day < vcut)
    return trn, val, te


def auc(score, lab):
    o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def all_importances(bst, names):
    out = {}
    for t in IMP_TYPES:
        sc = bst.get_score(importance_type=t)
        out[t] = {(names[int(k[1:])] if (k[0] == "f" and int(k[1:]) < len(names)) else k): float(v)
                  for k, v in sorted(sc.items(), key=lambda z: -z[1])}
    return out


def optuna_search(base, dtr, dval, val_metric, n_trials, seed):
    trials = []
    def objective(trial):
        p = dict(base, max_depth=trial.suggest_int("max_depth", 3, 9),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 min_child_weight=trial.suggest_int("min_child_weight", 1, 200, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True))
        bst = xgb.train(p, dtr, num_boost_round=600, evals=[(dval, "val")],
                        early_stopping_rounds=30, verbose_eval=False)
        pv = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        v = val_metric(pv); trials.append({"params": trial.params, "val": float(v), "best_iter": int(bst.best_iteration)})
        return v
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study, trials


def fit_final(params, dtr, dval, eval_metric):
    ev = {}
    bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dval, "val")],
                    early_stopping_rounds=30, verbose_eval=False, evals_result=ev)
    return bst, ev.get("val", {}).get(eval_metric, [])


# ------------------------------------------------------------------ Model A + per-symbol bundle
def process_symbol(sym, si, nf_rate, cfg_idx, qm_val, n_trials, seed, search_cap, log):
    d, meta = load(sym)
    F = d["F"].astype(np.float32); rH = d["rH60"].astype(np.float64); day = d["day"]; ts = d["ts"]
    PL = d["pnl_long"].astype(np.float32); PS = d["pnl_short"].astype(np.float32)   # (NC,QM,N) %
    FLa = d["fill_long"].astype(np.uint8); FSa = d["fill_short"].astype(np.uint8)   # (QM,N)
    qm_idx = list(meta["queue_mults"]).index(qm_val); fee = meta["maker_rt_fee_pct"] * 100.0
    ndays = meta["n_days"]; trn, val, te = split(day, ndays)
    thr = float(np.quantile(np.abs(rH[trn]), 1 - nf_rate))
    y = (np.abs(rH) >= thr).astype(int)

    # ---- Model A: Optuna on subsample (all pos + capped neg), final fit on FULL train ----
    base = {"objective": "binary:logistic", "tree_method": "hist", "eval_metric": "auc", "nthread": 8, "seed": seed}
    tr_idx = np.where(trn)[0]
    if len(tr_idx) > search_cap:
        rng = np.random.default_rng(seed)
        pos = tr_idx[y[tr_idx] == 1]; neg = tr_idx[y[tr_idx] == 0]
        n_neg = min(len(neg), max(search_cap - len(pos), len(pos) * 5))
        srch = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    else:
        srch = tr_idx
    spw_s = float((y[srch] == 0).sum() / max((y[srch] == 1).sum(), 1))
    spw_f = float((y[trn] == 0).sum() / max((y[trn] == 1).sum(), 1))
    dval = xgb.DMatrix(F[val], label=y[val])
    study, trials = optuna_search(dict(base, scale_pos_weight=spw_s), xgb.DMatrix(F[srch], label=y[srch]),
                                  dval, lambda p: auc(p, y[val]), n_trials, seed)
    bstA, valcurve = fit_final(dict(base, scale_pos_weight=spw_f, **study.best_params),
                               xgb.DMatrix(F[trn], label=y[trn]), dval, "auc")
    itA = (0, bstA.best_iteration + 1)
    pA_te = bstA.predict(xgb.DMatrix(F[te]), iteration_range=itA)
    yte = y[te]; order = np.argsort(-pA_te)
    resA = {"head": "A", "symbol": sym, "nf_rate": nf_rate, "vol_thr_bp": thr,
            "nf_base_train": float(y[trn].mean()), "nf_base_test": float(yte.mean()),
            "best_val_auc": float(study.best_value), "auc": auc(pA_te, yte), "best_params": study.best_params,
            "best_iter": int(bstA.best_iteration), "n_train": int(trn.sum()), "n_val": int(val.sum()),
            "n_test": int(te.sum()), "search_n": int(len(srch)), "seed": seed,
            "val_auc_curve": [float(x) for x in valcurve], "importances": all_importances(bstA, NAMES),
            "trials": trials, "script": SCRIPT_TAG}
    for q in (1.0, 0.5, 0.2):
        k = max(20, int(len(pA_te) * q / 100)); resA[f"prec@{q}%"] = float(yte[order[:k]].mean())
    log(f"[A {sym.split('-')[0]:4s}] thr={thr:5.1f}bp nf_tr/te={y[trn].mean()*100:.1f}/{yte.mean()*100:.1f}% "
        f"AUC={resA['auc']:.3f} prec@1/.5/.2={resA['prec@1.0%']:.2f}/{resA['prec@0.5%']:.2f}/{resA['prec@0.2%']:.2f}")

    # ---- B training rows (this symbol, non-flat & >=1-side-fill on chosen cfg/qm) ----
    nl = PL[cfg_idx, qm_idx].astype(np.float64) * 100.0 - fee
    ns = PS[cfg_idx, qm_idx].astype(np.float64) * 100.0 - fee
    fl = FLa[qm_idx].astype(bool); fs = FSa[qm_idx].astype(bool)
    nf = (np.abs(rH) >= thr) & np.isfinite(rH)
    keep = nf & (fl | fs)
    yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
    both = fl & fs                                              # C2: econ weight on EXECUTABLE outcomes only
    w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
    Btr = keep & trn; Bvl = keep & val
    bundleB = {"Ftr": F[Btr], "ytr": yB[Btr], "wtr": w[Btr],
               "Fvl": F[Bvl], "yvl": yB[Bvl], "nlvl": nl[Bvl], "nsvl": ns[Bvl], "flvl": fl[Bvl], "fsvl": fs[Bvl]}

    # ---- TEST bundle (ALL cfg x qm payoffs saved -> any surface recomputable offline) ----
    test = {"ts": ts[te], "day": day[te], "sid": np.full(int(te.sum()), si, np.int16),
            "rH60": rH[te].astype(np.float32), "yA": yte.astype(np.uint8), "pA": pA_te.astype(np.float32),
            "F_te": F[te], "pnl_long": PL[:, :, te], "pnl_short": PS[:, :, te],
            "fill_long": FLa[:, te], "fill_short": FSa[:, te], "thr": thr,
            "cfgs": meta["cfgs"], "queue_mults": list(meta["queue_mults"]), "fee_bp": fee}
    return resA, bstA, bundleB, test, meta


def save_json(obj, tag):
    bk.blob(f"{OUT}/{tag}.json").upload_from_string(json.dumps(obj, default=float))


def save_booster(bst, tag):
    bst.save_model(f"/tmp/_xgbm_{tag}.json"); bk.blob(f"{OUT}/{tag}.xgb.json").upload_from_filename(f"/tmp/_xgbm_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ALL"])
    ap.add_argument("--nf-rate", type=float, default=0.05)
    ap.add_argument("--cfg-idx", type=int, default=0)     # 0=hold-60s
    ap.add_argument("--qm", type=float, default=1.0)      # 1.0 = queue (more adverse)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-cap", type=int, default=600000)
    a = ap.parse_args()
    syms = SYMS if a.symbols == ["ALL"] else a.symbols
    t0 = time.time()
    def log(s): print(s, flush=True)
    log(f"XGB maker-label TRAIN | syms={len(syms)} nf={a.nf_rate} cfg={a.cfg_idx} qm={a.qm} trials={a.trials} seed={a.seed}")

    # ---- Phase 1: per-symbol Model A + collect B-rows + test bundles ----
    bundlesB, tests, manifest_syms = [], {}, {}
    for si, sym in enumerate(syms):
        resA, bstA, bB, test, meta = process_symbol(sym, si, a.nf_rate, a.cfg_idx, a.qm,
                                                    a.trials, a.seed, a.search_cap, log)
        save_json(resA, f"A_{sym.split('-')[0]}"); save_booster(bstA, f"A_{sym.split('-')[0]}")
        bundlesB.append(bB); tests[sym] = test
        manifest_syms[sym] = {"vol_thr_bp": resA["vol_thr_bp"], "n_train": resA["n_train"],
                              "n_test": resA["n_test"], "A_auc": resA["auc"], "n_days": meta["n_days"]}

    # ---- Phase 2: pooled Model B (non-flat fillable, chosen cfg/qm) ----
    Ftr = np.concatenate([np.concatenate([b["Ftr"], np.full((len(b["Ftr"]), 1), i, np.float32)], 1)
                          for i, b in enumerate(bundlesB)])
    ytr = np.concatenate([b["ytr"] for b in bundlesB]); wtr = np.concatenate([b["wtr"] for b in bundlesB])
    Fvl = np.concatenate([np.concatenate([b["Fvl"], np.full((len(b["Fvl"]), 1), i, np.float32)], 1)
                          for i, b in enumerate(bundlesB)])
    yvl = np.concatenate([b["yvl"] for b in bundlesB])
    NLv = np.concatenate([b["nlvl"] for b in bundlesB]); NSv = np.concatenate([b["nsvl"] for b in bundlesB])
    FLv = np.concatenate([b["flvl"] for b in bundlesB]); FSv = np.concatenate([b["fsvl"] for b in bundlesB])
    wc = np.clip(wtr, 0, np.quantile(wtr[wtr > 0], 0.99) if (wtr > 0).any() else 1.0)
    baseB = {"objective": "binary:logistic", "tree_method": "hist", "eval_metric": "logloss", "nthread": 8, "seed": a.seed}
    dtrB = xgb.DMatrix(Ftr, label=ytr, weight=wc); dvlB = xgb.DMatrix(Fvl, label=yvl)
    def val_execEV(p):
        pl = p >= 0.5; cn = np.where(pl, NLv, NSv); cf = np.where(pl, FLv, FSv)
        m = cf & np.isfinite(cn); return float(cn[m].mean()) if m.any() else -99.0
    studyB, trialsB = optuna_search(baseB, dtrB, dvlB, val_execEV, a.trials, a.seed)
    bstB, valcurveB = fit_final(dict(baseB, **studyB.best_params), dtrB, dvlB, "logloss")
    itB = (0, bstB.best_iteration + 1)
    fee_bp = meta["maker_rt_fee_pct"] * 100.0   # constant from build (last meta in scope)
    resB = {"head": "B", "symbols": syms, "cfg_idx": a.cfg_idx, "qm": a.qm, "fee_bp": fee_bp,
            "nf_rate": a.nf_rate, "best_val_execEV_bp": float(studyB.best_value), "best_params": studyB.best_params,
            "best_iter": int(bstB.best_iteration), "n_train": int(len(ytr)), "n_val": int(len(yvl)),
            "val_logloss_curve": [float(x) for x in valcurveB], "importances": all_importances(bstB, NAMES + ["sym_id"]),
            "trials": trialsB, "seed": a.seed, "long_better_rate_train": float(ytr.mean()), "script": SCRIPT_TAG}
    save_json(resB, "B_pool"); save_booster(bstB, "B_pool")
    log(f"[B pooled] best_val_execEV={studyB.best_value:+.2f}bp n_train={len(ytr)} long-better={ytr.mean():.3f}")

    # ---- Phase 3: predict pB on EACH symbol's FULL test set; save per-symbol preds npz ----
    for sym, test in tests.items():
        sid_col = np.full((len(test["F_te"]), 1), syms.index(sym), np.float32)
        pB = bstB.predict(xgb.DMatrix(np.concatenate([test["F_te"], sid_col], 1)), iteration_range=itB)
        buf = io.BytesIO()
        np.savez_compressed(buf, ts=test["ts"], day=test["day"], sid=test["sid"], rH60=test["rH60"],
                            yA=test["yA"], pA=test["pA"], pB=pB.astype(np.float32),
                            pnl_long=test["pnl_long"], pnl_short=test["pnl_short"],
                            fill_long=test["fill_long"], fill_short=test["fill_short"],
                            meta=np.array(json.dumps({"symbol": sym, "thr": test["thr"], "cfg_idx": a.cfg_idx,
                                                      "qm": a.qm, "seed": a.seed, "cfgs": test["cfgs"],
                                                      "queue_mults": test["queue_mults"], "fee_bp": test["fee_bp"]})))
        bk.blob(f"{OUT}/preds_{sym.split('-')[0]}.npz").upload_from_string(buf.getvalue())
        log(f"  [preds {sym.split('-')[0]:4s}] saved n_test={len(pB)}")

    manifest = {"script": SCRIPT_TAG, "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": vars(a), "split_fracs": SPLIT, "src": SRC, "symbols": manifest_syms,
                "B_cfg": {"cfg_idx": a.cfg_idx, "qm": a.qm}, "wall_s": round(time.time() - t0, 0)}
    save_json(manifest, "MANIFEST")
    log(f"[done] {time.time()-t0:.0f}s | artifacts -> gs://{BUCKET}/{OUT}/")


if __name__ == "__main__":
    main()
