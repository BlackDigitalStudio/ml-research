#!/usr/bin/env python3
"""RECENCY / regime-drift test: does test performance depend on HOW RECENT the training data is,
holding hyperparameters + #rounds + per-symbol vol-threshold FIXED (reused from the main run)?

Hypothesis (user): the single-split degradation is NOT HP overfit but MARKET DRIFT — the closer
the training window is to the test period, the better test results, even with no HP change / no leak.

Protocol (zero new selection, test untouched for any choice):
  - HPs, num_boost_round (=best_iter), and per-symbol A vol-threshold are LOADED from the main run
    (A_{SYM}.json / B_pool.json) and held FIXED.
  - Only the TRAINING-DATA WINDOW varies (same-size old vs recent isolates recency from volume):
      old10   [0.000,0.0975)  | train [0.000,0.5525) | val10 [0.5525,0.65) | trainval [0.000,0.65)
  - Each window: retrain A(per-sym) + B(pooled) with fixed HP & fixed rounds (NO early-stop ->
    no eval set -> no leakage), predict the SAME clean TEST [0.68,1.0).
  - Measure on TEST: A AUC + prec@{1,0.2}% ; B oracle dir-acc@10% ; cascade executed maker-EV
    (hold-60s/TOUCH, A-pred-gated top-1% & top-5%) + dir-acc.
If val10 >= old10 (same size) and trainval >= train -> recency/drift dominates, not overfit.
Saves -> research_runs/xgb_maker/RECENCY.json.
Run: python3 subs60_xgb_recency.py --symbols ALL --seed 0
"""
import argparse, io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SRC = "research_runs/maker_labels"; OUT = "research_runs/xgb_maker"
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
WINDOWS = [("old10", 0.0, 0.0975), ("train", 0.0, 0.5525), ("val10", 0.5525, 0.65), ("trainval", 0.0, 0.65)]
TEST = (0.68, 1.0)
CASC_CFG, CASC_QM = 0, 0          # hold-60s / TOUCH (least-adverse cell, where the illusory edge was)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(tag): return json.loads(bk.blob(f"{OUT}/{tag}.json").download_as_bytes())


def load_sym(sym):
    d = np.load(io.BytesIO(bk.blob(f"{SRC}/{sym.split('-')[0]}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pnl_long": d["pnl_long"].astype(np.float32), "pnl_short": d["pnl_short"].astype(np.float32),
            "fill_long": d["fill_long"].astype(bool), "fill_short": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "qms": list(m["queue_mults"]), "fee": m["maker_rt_fee_pct"] * 100.0}


def auc(score, lab):
    o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def win_mask(day, nd, lo, hi): return (day >= int(nd * lo)) & (day < int(nd * hi))


def exec_ev(pl_pred, nl, ns, fl, fs, idx):
    pl = pl_pred[idx] >= 0.5; cn = np.where(pl, nl[idx], ns[idx]); cf = np.where(pl, fl[idx], fs[idx])
    ex = cf & np.isfinite(cn)
    nle = np.where(fl[idx], nl[idx], -np.inf); nse = np.where(fs[idx], ns[idx], -np.inf)
    one = fl[idx] | fs[idx]; better = nle > nse
    return (float(cn[ex].mean()) if ex.any() else float("nan"),
            float((pl[one] == better[one]).mean()) if one.any() else float("nan"), int(ex.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ALL"])
    ap.add_argument("--nf-rate", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    syms = SYMS if a.symbols == ["ALL"] else a.symbols
    def log(s): print(s, flush=True)
    log(f"RECENCY/drift test | fixed HP+rounds+thr from main run | windows={[w[0] for w in WINDOWS]} test={TEST}")
    hpA = {s: jload(f"A_{s.split('-')[0]}") for s in syms}; hpB = jload("B_pool")
    nrB = max(1, hpB["best_iter"] + 1)
    D = {s: load_sym(s) for s in syms}
    qm_idx = D[syms[0]]["qms"].index(1.0)        # B target cfg/qm = hold-60s / queue (as main run)
    baseA = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": a.seed}
    baseB = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": a.seed}
    res = {"windows": {w[0]: {"lo": w[1], "hi": w[2]} for w in WINDOWS}, "per_symbol": {}, "test": TEST}

    # precompute per-symbol fixed threshold + test masks/labels (threshold FIXED from main A run)
    fix = {}
    for s in syms:
        d = D[s]; nd = d["ndays"]; thr = hpA[s]["vol_thr_bp"]
        te = win_mask(d["day"], nd, *TEST); y = (np.abs(d["rH"]) >= thr).astype(int)
        fix[s] = {"thr": thr, "te": te, "y": y, "nd": nd}

    rows = {w[0]: {} for w in WINDOWS}
    for wlab, lo, hi in WINDOWS:
        # ---- per-symbol A (fixed HP+rounds), collect B-train rows for this window ----
        BF, BY, BW, BSID = [], [], [], []
        pAte = {}
        for si, s in enumerate(syms):
            d = D[s]; nd = fix[s]["nd"]; thr = fix[s]["thr"]; y = fix[s]["y"]
            trn = win_mask(d["day"], nd, lo, hi)
            spw = float((y[trn] == 0).sum() / max((y[trn] == 1).sum(), 1))
            pa = dict(baseA, scale_pos_weight=spw, **{k: hpA[s]["best_params"][k] for k in hpA[s]["best_params"]})
            nr = max(1, hpA[s]["best_iter"] + 1)
            bstA = xgb.train(pa, xgb.DMatrix(d["F"][trn], label=y[trn]), num_boost_round=nr)
            pAte[s] = bstA.predict(xgb.DMatrix(d["F"][fix[s]["te"]]))
            # B rows (this window, non-flat & >=1-side-fill, cfg hold/qm=1)
            nl = d["pnl_long"][0, qm_idx].astype(np.float64) * 100.0 - d["fee"]
            ns = d["pnl_short"][0, qm_idx].astype(np.float64) * 100.0 - d["fee"]
            fl = d["fill_long"][qm_idx]; fs = d["fill_short"][qm_idx]
            nf = (np.abs(d["rH"]) >= thr) & np.isfinite(d["rH"]); keep = trn & nf & (fl | fs)
            yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
            both = fl & fs
            w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
            BF.append(d["F"][keep]); BY.append(yB[keep]); BW.append(w[keep]); BSID.append(np.full(int(keep.sum()), si, np.float32))
        # ---- pooled B (fixed HP+rounds) ----
        Xb = np.concatenate([np.concatenate([f, s[:, None]], 1) for f, s in zip(BF, BSID)])
        yb = np.concatenate(BY); wb = np.concatenate(BW)
        wc = np.clip(wb, 0, np.quantile(wb[wb > 0], 0.99) if (wb > 0).any() else 1.0)
        pb = dict(baseB, **hpB["best_params"])
        bstB = xgb.train(pb, xgb.DMatrix(Xb, label=yb, weight=wc), num_boost_round=nrB)
        # ---- measure on TEST ----
        for si, s in enumerate(syms):
            d = D[s]; te = fix[s]["te"]; yte = fix[s]["y"][te]; pa = pAte[s]
            sidt = np.full((int(te.sum()), 1), si, np.float32)
            pBte = bstB.predict(xgb.DMatrix(np.concatenate([d["F"][te], sidt], 1)))
            order = np.argsort(-pa)
            prec = {q: float(yte[order[:max(20, int(len(pa) * q / 100))]].mean()) for q in (1.0, 0.2)}
            # oracle-gated B dir-acc @ conv-top10% on realized non-flat
            nl0 = d["pnl_long"][CASC_CFG, CASC_QM].astype(np.float64) * 100.0 - d["fee"]
            ns0 = d["pnl_short"][CASC_CFG, CASC_QM].astype(np.float64) * 100.0 - d["fee"]
            fl0 = d["fill_long"][CASC_QM]; fs0 = d["fill_short"][CASC_QM]
            nf_te = yte.astype(bool); nfi = np.where(nf_te)[0]
            conv = np.abs(pBte[nfi] - 0.5); oo = nfi[np.argsort(-conv)]
            _, dor, _ = exec_ev(pBte, nl0[te], ns0[te], fl0[te], fs0[te], oo[:max(20, int(len(nfi) * 0.10))])
            # cascade executed maker-EV @ A-pred top-1% & top-5% (hold/TOUCH)
            casc = {}
            for g in (5.0, 1.0):
                k = max(20, int(len(pa) * g / 100)); ev, da, n = exec_ev(pBte, nl0[te], ns0[te], fl0[te], fs0[te], order[:k])
                casc[f"g{g}"] = {"EV_bp": ev, "dir_acc": da, "n": n}
            rows[wlab][s.split("-")[0]] = {"auc": auc(pa, yte), "prec@1%": prec[1.0], "prec@0.2%": prec[0.2],
                                           "oracle_dir@10%": dor, "casc": casc, "n_train": int(win_mask(d["day"], fix[s]["nd"], lo, hi).sum())}
        log(f"[window {wlab}] trained+measured")

    res["per_symbol"] = {s.split("-")[0]: {w: rows[w][s.split("-")[0]] for w in (x[0] for x in WINDOWS)} for s in syms}
    # ---- print: pooled means per window (the drift signal) ----
    log(f"\n=== RECENCY: mean over symbols, on the SAME clean TEST (fixed HP/rounds/thr) ===")
    log(f"{'window':9s} {'A_AUC':>6s} {'prec@1%':>7s} {'prec@.2%':>8s} {'oracleDir@10':>12s} {'cascEV@5%':>9s} {'cascEV@1%':>9s} {'cascDir@1%':>10s}")
    for wlab, lo, hi in WINDOWS:
        r = [rows[wlab][s.split("-")[0]] for s in syms]
        mn = lambda f: float(np.nanmean([f(x) for x in r]))
        log(f"{wlab:9s} {mn(lambda x: x['auc']):6.3f} {mn(lambda x: x['prec@1%']):7.3f} {mn(lambda x: x['prec@0.2%']):8.3f} "
            f"{mn(lambda x: x['oracle_dir@10%']):12.3f} {mn(lambda x: x['casc']['g5.0']['EV_bp']):+9.2f} "
            f"{mn(lambda x: x['casc']['g1.0']['EV_bp']):+9.2f} {mn(lambda x: x['casc']['g1.0']['dir_acc']):10.3f}")
    bk.blob(f"{OUT}/RECENCY.json").upload_from_string(json.dumps(res, default=float))
    log(f"\n[saved] gs://{BUCKET}/{OUT}/RECENCY.json")


if __name__ == "__main__":
    main()
