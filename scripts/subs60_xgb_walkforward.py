#!/usr/bin/env python3
"""WALK-FORWARD OOS confirmation of the cascade maker-EV argmax cells (HD3).

The single-split SURFACE.json argmax (LTC +2.0bp, LINK +1.2bp @ hold/touch/A-top0.2%) is a MAX
over ~30 cells (cfg x qm x A-selectivity) CHOSEN ON TEST -> selection-over-conditions optimism.
This script removes that bias: expanding-window walk-forward where, per fold, the operating point
(cfg, qm, A-selectivity) is SELECTED on a VALIDATION slice and MEASURED on a later, disjoint TEST
slice. HPs are REUSED from the main run (A_{SYM}.json / B_pool.json best_params) -> no per-fold
Optuna (cheap). Dataset = existing research_runs/maker_labels (no rebuild).

Per fold f (fractions of each symbol's day-index, expanding train):
  per symbol: thr=p95(|rH60|) on train; train A(hp_A[sym]) early-stop on val -> pA(val,test).
  pooled: train B(hp_B) on non-flat-fillable train (cfg0/qm=1 target)+sym_id -> pB(val,test).
  per symbol: argmax executed-net-maker-EV cell over {cfg x qm x sel} on VAL (n_exec>=MIN_N),
              then MEASURE that SAME cell on TEST. Record (cell, val_EV, test_EV, n, dir_acc, fill).
Aggregate test_EV across folds per symbol -> the OOS-confirmed (or not) number.
Saves -> research_runs/xgb_maker/WALKFORWARD.json.
Run: python3 subs60_xgb_walkforward.py --symbols ALL --folds 4 --seed 0
"""
import argparse, io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SRC = "research_runs/maker_labels"; OUT = "research_runs/xgb_maker"
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
A_SEL = [5.0, 2.0, 1.0, 0.5, 0.2]
MIN_N = 100          # min executed trades on VAL for a cell to be selectable (avoid tiny-sample argmax)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(tag):
    return json.loads(bk.blob(f"{OUT}/{tag}.json").download_as_bytes())


def load_sym(sym):
    d = np.load(io.BytesIO(bk.blob(f"{SRC}/{sym.split('-')[0]}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pnl_long": d["pnl_long"].astype(np.float32), "pnl_short": d["pnl_short"].astype(np.float32),
            "fill_long": d["fill_long"].astype(bool), "fill_short": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "meta": m}


def sides(D, c, q, fee):
    nl = D["pnl_long"][c, q].astype(np.float64) * 100.0 - fee
    ns = D["pnl_short"][c, q].astype(np.float64) * 100.0 - fee
    return nl, ns, D["fill_long"][q], D["fill_short"][q]


def exec_ev(pB, nl, ns, fl, fs, idx):
    pl = pB[idx] >= 0.5
    cn = np.where(pl, nl[idx], ns[idx]); cf = np.where(pl, fl[idx], fs[idx])
    ex = cf & np.isfinite(cn)
    nle = np.where(fl[idx], nl[idx], -np.inf); nse = np.where(fs[idx], ns[idx], -np.inf)
    one = fl[idx] | fs[idx]; better = nle > nse
    return {"execEV_bp": float(cn[ex].mean()) if ex.any() else float("nan"),
            "dir_acc": float((pl[one] == better[one]).mean()) if one.any() else float("nan"),
            "fill": float(cf.mean()), "n_exec": int(ex.sum())}


def fit(hp, base, Xtr, ytr, Xvl, yvl, wtr=None):
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr); dvl = xgb.DMatrix(Xvl, label=yvl)
    bst = xgb.train(dict(base, **hp), dtr, num_boost_round=600, evals=[(dvl, "val")],
                    early_stopping_rounds=30, verbose_eval=False)
    return bst, (0, bst.best_iteration + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ALL"])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--nf-rate", type=float, default=0.05)
    ap.add_argument("--cfg-idx", type=int, default=0)
    ap.add_argument("--qm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    syms = SYMS if a.symbols == ["ALL"] else a.symbols
    def log(s): print(s, flush=True)
    log(f"WALK-FORWARD OOS | syms={len(syms)} folds={a.folds} nf={a.nf_rate} cfg={a.cfg_idx} qm={a.qm} seed={a.seed}")
    hpA = {s: jload(f"A_{s.split('-')[0]}")["best_params"] for s in syms}
    hpB = jload("B_pool")["best_params"]
    D = {s: load_sym(s) for s in syms}
    fee = D[syms[0]]["meta"]["maker_rt_fee_pct"] * 100.0; qms = D[syms[0]]["meta"]["queue_mults"]
    cfgs = D[syms[0]]["meta"]["cfgs"]; qm_idx = list(qms).index(a.qm)
    NC, QM = len(cfgs), len(qms)
    cfg_lab = [("hold" if c["tp"] >= 1 else "RR" + str(round(c["tp"] / c["sl"]))) for c in cfgs]
    qm_lab = [("touch" if q == 0 else f"queue{int(q)}") for q in qms]
    baseA = {"objective": "binary:logistic", "tree_method": "hist", "eval_metric": "auc", "nthread": 8, "seed": a.seed}
    baseB = {"objective": "binary:logistic", "tree_method": "hist", "eval_metric": "logloss", "nthread": 8, "seed": a.seed}

    # expanding-window folds in day-INDEX fraction space (mirror main split style)
    # fold f: train[0,tr), val[tr,vl), test[vl,te)
    base_tr = 0.55; step = (1.0 - 0.62) / a.folds   # test windows tile the last ~38%
    folds = []
    for f in range(a.folds):
        tr = base_tr + f * step; vl = tr + 0.07; te = min(vl + step, 1.0)
        if te - vl < 0.02:
            continue
        folds.append((round(tr, 3), round(vl, 3), round(te, 3)))
    log(f"folds (train_end,val_end,test_end fracs): {folds}")

    results = {s.split("-")[0]: {"folds": [], "cfg_labels": cfg_lab, "qm_labels": qm_lab} for s in syms}
    for fi, (trf, vlf, tef) in enumerate(folds):
        # ---- per-symbol A + collect B rows + per-sym val/test slices ----
        Btr_F, Btr_y, Btr_w, Btr_sid = [], [], [], []
        Bvl_F, Bvl_y, Bvl_NL, Bvl_NS, Bvl_FL, Bvl_FS, Bvl_sid = [], [], [], [], [], [], []
        per = {}
        for si, s in enumerate(syms):
            d = D[s]; nd = d["ndays"]; day = d["day"]; rH = d["rH"]; F = d["F"]
            trn = day < int(nd * trf); val = (day >= int(nd * trf)) & (day < int(nd * vlf))
            te = (day >= int(nd * vlf)) & (day < int(nd * tef))
            if trn.sum() < 5000 or val.sum() < 500 or te.sum() < 500:
                per[s] = None; continue
            thr = float(np.quantile(np.abs(rH[trn]), 1 - a.nf_rate)); y = (np.abs(rH) >= thr).astype(int)
            spw = float((y[trn] == 0).sum() / max((y[trn] == 1).sum(), 1))
            bstA, itA = fit(hpA[s], dict(baseA, scale_pos_weight=spw), F[trn], y[trn], F[val], y[val])
            pA_val = bstA.predict(xgb.DMatrix(F[val]), iteration_range=itA)
            pA_te = bstA.predict(xgb.DMatrix(F[te]), iteration_range=itA)
            # B rows for chosen cfg/qm
            nl, ns, fl, fs = sides(d, a.cfg_idx, qm_idx, fee)
            nf = (np.abs(rH) >= thr) & np.isfinite(rH); keep = nf & (fl | fs)
            yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
            both = fl & fs
            w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
            Btr = keep & trn; Bvl = keep & val
            Btr_F.append(F[Btr]); Btr_y.append(yB[Btr]); Btr_w.append(w[Btr]); Btr_sid.append(np.full(int(Btr.sum()), si, np.float32))
            Bvl_F.append(F[Bvl]); Bvl_y.append(yB[Bvl]); Bvl_sid.append(np.full(int(Bvl.sum()), si, np.float32))
            Bvl_NL.append(nl[Bvl]); Bvl_NS.append(ns[Bvl]); Bvl_FL.append(fl[Bvl]); Bvl_FS.append(fs[Bvl])
            per[s] = {"si": si, "thr": thr, "val": val, "te": te, "pA_val": pA_val, "pA_te": pA_te}
        # ---- pooled B (fixed HP) ----
        Xtr = np.concatenate([np.concatenate([f, s[:, None]], 1) for f, s in zip(Btr_F, Btr_sid)])
        ytr = np.concatenate(Btr_y); wtr = np.concatenate(Btr_w)
        wc = np.clip(wtr, 0, np.quantile(wtr[wtr > 0], 0.99) if (wtr > 0).any() else 1.0)
        Xvl = np.concatenate([np.concatenate([f, s[:, None]], 1) for f, s in zip(Bvl_F, Bvl_sid)])
        yvl = np.concatenate(Bvl_y)
        bstB, itB = fit(hpB, baseB, Xtr, ytr, Xvl, yvl, wtr=wc)
        # ---- per symbol: SELECT op-point on VAL, MEASURE on TEST ----
        for s in syms:
            p = per[s]
            if p is None:
                results[s.split("-")[0]]["folds"].append(None); continue
            si = p["si"]; d = D[s]; val = p["val"]; te = p["te"]
            sidv = np.full((int(val.sum()), 1), si, np.float32); sidt = np.full((int(te.sum()), 1), si, np.float32)
            pB_val = bstB.predict(xgb.DMatrix(np.concatenate([d["F"][val], sidv], 1)), iteration_range=itB)
            pB_te = bstB.predict(xgb.DMatrix(np.concatenate([d["F"][te], sidt], 1)), iteration_range=itB)
            # SELECT (cfg,qm) on VAL by EV at a stable selectivity (top-1%, n>=MIN_N); separate the
            # config choice from the selectivity choice. Then MEASURE the OOS test-EV CURVE across ALL
            # selectivities for that (cfg,qm) -> shows if ANY selectivity is OOS-robust (not just argmax).
            ov = np.argsort(-p["pA_val"]); ot = np.argsort(-p["pA_te"]); nd_te = len(set(d["day"][te].tolist()))
            G_SEL = 1.0; best = None
            for c in range(NC):
                for q in range(QM):
                    nl, ns, fl, fs = sides(d, c, q, fee)
                    kv = max(20, int(len(pB_val) * G_SEL / 100))
                    rv = exec_ev(pB_val, nl[val], ns[val], fl[val], fs[val], ov[:kv])
                    if rv["n_exec"] >= MIN_N and np.isfinite(rv["execEV_bp"]) and (best is None or rv["execEV_bp"] > best["val_EV"]):
                        best = {"cfg": cfg_lab[c], "qm": qm_lab[q], "c": c, "q": q, "val_EV": rv["execEV_bp"]}
            if best is None:
                results[s.split("-")[0]]["folds"].append({"selectable": False}); continue
            nl, ns, fl, fs = sides(d, best["c"], best["q"], fee)
            curve = {}
            for g in A_SEL:
                kt = max(20, int(len(pB_te) * g / 100))
                rt = exec_ev(pB_te, nl[te], ns[te], fl[te], fs[te], ot[:kt])
                curve[f"g{g}"] = {"test_EV_bp": rt["execEV_bp"], "dir_acc": rt["dir_acc"],
                                  "fill": rt["fill"], "n_exec": rt["n_exec"], "trd_day": rt["n_exec"] / max(nd_te, 1)}
            rec = {"fold": fi, "sel_cfg_qm": f"{best['cfg']}/{best['qm']}", "val_EV_top1pct_bp": best["val_EV"],
                   "test_curve": curve}
            results[s.split("-")[0]]["folds"].append(rec)
        log(f"[fold {fi}] {folds[fi]} done")

    # ---- aggregate (mean OOS test-EV per selectivity across folds) + print ----
    log(f"\n=== WALK-FORWARD OOS: mean test-EV(bp) per A-selectivity (cfg/qm picked on VAL@top1%) ===")
    log(f"{'SYM':5s} {'nf':>3s}  selVAL(cfg/qm)    test-EV @ A-top 5/2/1/0.5/0.2%            dir@1%  trd/d@1%")
    for s in syms:
        sk = s.split("-")[0]; fr = [f for f in results[sk]["folds"] if f and "test_curve" in f]
        if not fr:
            log(f"{sk:5s}  (no valid folds)"); continue
        agg = {}
        for g in A_SEL:
            evs = [f["test_curve"][f"g{g}"]["test_EV_bp"] for f in fr if np.isfinite(f["test_curve"][f"g{g}"]["test_EV_bp"])]
            agg[f"g{g}"] = float(np.mean(evs)) if evs else float("nan")
        d1 = [f["test_curve"]["g1.0"]["dir_acc"] for f in fr if np.isfinite(f["test_curve"]["g1.0"]["dir_acc"])]
        t1 = [f["test_curve"]["g1.0"]["trd_day"] for f in fr]
        cfgsel = ",".join(sorted(set(f["sel_cfg_qm"] for f in fr)))
        results[sk]["n_valid_folds"] = len(fr); results[sk]["mean_test_EV_by_sel"] = agg
        results[sk]["mean_dir_acc_top1"] = float(np.mean(d1)) if d1 else float("nan")
        log(f"{sk:5s} {len(fr):3d}  {cfgsel:15s}  {agg['g5.0']:+5.1f}/{agg['g2.0']:+5.1f}/{agg['g1.0']:+5.1f}/"
            f"{agg['g0.5']:+5.1f}/{agg['g0.2']:+5.1f}bp   {results[sk]['mean_dir_acc_top1']:.3f}   {np.mean(t1):.1f}")
    bk.blob(f"{OUT}/WALKFORWARD.json").upload_from_string(json.dumps(results, default=float))
    log(f"\n[saved] gs://{BUCKET}/{OUT}/WALKFORWARD.json")


if __name__ == "__main__":
    main()
