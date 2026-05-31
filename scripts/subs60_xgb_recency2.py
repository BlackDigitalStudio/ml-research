#!/usr/bin/env python3
"""RECENCY v2 — corrected drift test: EQUAL-SIZE LARGE windows differing only in RECENCY, clean
per-model metrics. Fixes two flaws of subs60_xgb_recency.py (user catch):
  (1) v1's large windows were anchored at the START (lots of OLD data) -> never gave the model a
      large RECENT window; the only recency-isolating compare was at 10% (too small).
  (2) v1 measured B's dir-acc cross-config (trained qm=1, eval touch qm=0) -> ~0.49, masked B's skill.

Design: three EQUAL-SIZE (45% of days) training windows sliding old->recent, same clean test
[0.68,1.0); HP + #rounds + per-symbol vol-threshold FIXED (reused from the main run):
  old_half [0.00,0.45] | mid_half [0.10,0.55] | recent_half [0.20,0.65]
Per-MODEL clean metrics on TEST (so recency is visible per model, not masked by the maker-EV floor):
  A : AUC, prec@{1,0.2}%
  B : rank-IC = spearman(pB, rH) on non-flat ; dir-acc vs raw sign(rH) @conv-top10% ; dir-acc vs
      maker-better-side (qm=1, B's OWN target) @conv-top10%  [all on the trained config, not cross-config]
  + secondary: cascade executed maker-EV (qm=1, A-pred top-1%/5%).
If recent_half > old_half at equal volume -> recency/drift matters (and on which model is explicit).
Saves -> research_runs/xgb_maker/RECENCY2.json.
Run: python3 subs60_xgb_recency2.py --symbols ALL --seed 0
"""
import argparse, io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SRC = "research_runs/maker_labels"; OUT = "research_runs/xgb_maker"
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
WINDOWS = [("old_half", 0.0, 0.45), ("mid_half", 0.10, 0.55), ("recent_half", 0.20, 0.65)]
TEST = (0.68, 1.0)
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


def spearman(a, b):
    if len(a) < 50:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float); br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean(); den = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / den) if den > 0 else 0.0


def win_mask(day, nd, lo, hi): return (day >= int(nd * lo)) & (day < int(nd * hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ALL"]); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    syms = SYMS if a.symbols == ["ALL"] else a.symbols
    def log(s): print(s, flush=True)
    log(f"RECENCY v2 | equal-size 45% windows old->recent | per-model clean metrics | windows={[w[0] for w in WINDOWS]} test={TEST}")
    hpA = {s: jload(f"A_{s.split('-')[0]}") for s in syms}; hpB = jload("B_pool"); nrB = max(1, hpB["best_iter"] + 1)
    D = {s: load_sym(s) for s in syms}
    qi = D[syms[0]]["qms"].index(1.0)         # B trained on hold-60s / qm=1
    baseA = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": a.seed}
    baseB = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": a.seed}
    fix = {}
    for s in syms:
        d = D[s]; thr = hpA[s]["vol_thr_bp"]
        fix[s] = {"thr": thr, "te": win_mask(d["day"], d["ndays"], *TEST), "y": (np.abs(d["rH"]) >= thr).astype(int)}

    rows = {w[0]: {} for w in WINDOWS}
    for wlab, lo, hi in WINDOWS:
        BF, BY, BW, BSID = [], [], [], []; pAte = {}
        for si, s in enumerate(syms):
            d = D[s]; thr = fix[s]["thr"]; y = fix[s]["y"]; trn = win_mask(d["day"], d["ndays"], lo, hi)
            spw = float((y[trn] == 0).sum() / max((y[trn] == 1).sum(), 1))
            pa = dict(baseA, scale_pos_weight=spw, **hpA[s]["best_params"])
            bstA = xgb.train(pa, xgb.DMatrix(d["F"][trn], label=y[trn]), num_boost_round=max(1, hpA[s]["best_iter"] + 1))
            pAte[s] = bstA.predict(xgb.DMatrix(d["F"][fix[s]["te"]]))
            nl = d["pnl_long"][0, qi].astype(np.float64) * 100.0 - d["fee"]
            ns = d["pnl_short"][0, qi].astype(np.float64) * 100.0 - d["fee"]
            fl = d["fill_long"][qi]; fs = d["fill_short"][qi]
            nf = (np.abs(d["rH"]) >= thr) & np.isfinite(d["rH"]); keep = trn & nf & (fl | fs)
            yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
            both = fl & fs; w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
            BF.append(d["F"][keep]); BY.append(yB[keep]); BW.append(w[keep]); BSID.append(np.full(int(keep.sum()), si, np.float32))
        Xb = np.concatenate([np.concatenate([f, s[:, None]], 1) for f, s in zip(BF, BSID)])
        yb = np.concatenate(BY); wb = np.concatenate(BW)
        wc = np.clip(wb, 0, np.quantile(wb[wb > 0], 0.99) if (wb > 0).any() else 1.0)
        bstB = xgb.train(dict(baseB, **hpB["best_params"]), xgb.DMatrix(Xb, label=yb, weight=wc), num_boost_round=nrB)
        for si, s in enumerate(syms):
            d = D[s]; te = fix[s]["te"]; yte = fix[s]["y"][te]; pa = pAte[s]
            sidt = np.full((int(te.sum()), 1), si, np.float32)
            pB = bstB.predict(xgb.DMatrix(np.concatenate([d["F"][te], sidt], 1)))
            order = np.argsort(-pa)
            prec = {q: float(yte[order[:max(20, int(len(pa) * q / 100))]].mean()) for q in (1.0, 0.2)}
            rH_te = d["rH"][te]; nf = yte.astype(bool)
            nl = d["pnl_long"][0, qi][te].astype(np.float64) * 100.0 - d["fee"]
            ns = d["pnl_short"][0, qi][te].astype(np.float64) * 100.0 - d["fee"]
            fl = d["fill_long"][qi][te]; fs = d["fill_short"][qi][te]
            # B clean direction metrics on NON-FLAT (trained config qm=1):
            ric = spearman(pB[nf], rH_te[nf])                                   # rank-IC pB vs forward move
            nfi = np.where(nf)[0]; conv = np.abs(pB[nfi] - 0.5); oo = nfi[np.argsort(-conv)]
            k10 = max(20, int(len(nfi) * 0.10)); top = oo[:k10]
            dir_raw = float(((pB[top] >= 0.5) == (rH_te[top] > 0)).mean())       # vs raw sign(rH)
            fkeep = (fl | fs)
            mside = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf))      # maker-better side (B's target)
            tf = top[fkeep[top]]
            dir_mk = float(((pB[tf] >= 0.5) == mside[tf]).mean()) if len(tf) > 20 else float("nan")
            # secondary cascade EV (qm=1), A-pred top g%
            casc = {}
            for g in (5.0, 1.0):
                k = max(20, int(len(pa) * g / 100)); idx = order[:k]
                pl = pB[idx] >= 0.5; cn = np.where(pl, nl[idx], ns[idx]); cf = np.where(pl, fl[idx], fs[idx])
                ex = cf & np.isfinite(cn); casc[f"g{g}"] = float(cn[ex].mean()) if ex.any() else float("nan")
            rows[wlab][s.split("-")[0]] = {"A_auc": auc(pa, yte), "A_prec@1%": prec[1.0], "A_prec@.2%": prec[0.2],
                                           "B_rankIC": ric, "B_dir_raw@10%": dir_raw, "B_dir_makerside@10%": dir_mk,
                                           "cascEV@5%": casc["g5.0"], "cascEV@1%": casc["g1.0"],
                                           "n_train": int(win_mask(d["day"], d["ndays"], lo, hi).sum())}
        log(f"[window {wlab}] done")

    res = {"windows": {w[0]: [w[1], w[2]] for w in WINDOWS}, "test": TEST,
           "per_symbol": {s.split("-")[0]: {w[0]: rows[w[0]][s.split("-")[0]] for w in WINDOWS} for s in syms}}
    log(f"\n=== RECENCY v2: mean over 8 symbols, SAME clean test, equal 45% windows (fixed HP/rounds/thr) ===")
    log(f"{'window':12s} {'A_AUC':>6s} {'A_p@1%':>6s} {'A_p@.2%':>7s} | {'B_rankIC':>8s} {'B_dirRaw@10':>11s} {'B_dirMk@10':>10s} | {'cascEV@1%':>9s}")
    for wlab, lo, hi in WINDOWS:
        r = [rows[wlab][s.split("-")[0]] for s in syms]; mn = lambda f: float(np.nanmean([f(x) for x in r]))
        log(f"{wlab:12s} {mn(lambda x: x['A_auc']):6.3f} {mn(lambda x: x['A_prec@1%']):6.3f} {mn(lambda x: x['A_prec@.2%']):7.3f} | "
            f"{mn(lambda x: x['B_rankIC']):+8.4f} {mn(lambda x: x['B_dir_raw@10%']):11.3f} {mn(lambda x: x['B_dir_makerside@10%']):10.3f} | "
            f"{mn(lambda x: x['cascEV@1%']):+9.2f}")
    bk.blob(f"{OUT}/RECENCY2.json").upload_from_string(json.dumps(res, default=float))
    log(f"\n[saved] gs://{BUCKET}/{OUT}/RECENCY2.json")


if __name__ == "__main__":
    main()
