#!/usr/bin/env python3
"""B2-style per-symbol optimal-R:R: like the GRU s14 Stage-2 but for XGBoost.

Reads the R:R-grid dataset research_runs/maker_labels_rr/{SYM}.npz (pnl_long/short (NC,1,N) over a 32-cfg
R:R grid, touch qm=0). Per symbol:
  1. DISCOVER optimal R:R on TRAIN: c* = argmax over configs of the ORACLE best-side mean net maker EV on
     non-flat-fillable train (the per-config ceiling; B-independent; mirrors s14 grid_sim R:R discovery).
  2. Train A (reuse main A_{SYM} HP+rounds, fixed) -> pA_test ; train B on c* better-side AND, as baseline,
     on hold-60s better-side (reuse pooled B HP+rounds, fixed; per-symbol, no sym_id).
  3. Compare on TEST via the daily-budget A^B policy (1 trade/sym/day, score=pctrank(pA)*pctrank(|pB-.5|)):
     executed net maker EV at 1/day(all days) and top-25% days -- B2(c*) vs baseline(hold-60s).
Saves -> research_runs/maker_labels_rr/B2_RESULT.json.
Run: python3 subs60_xgb_b2.py --symbols BNB BTC DOGE ETH LINK LTC SOL XRP
"""
import argparse, io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; MAIN = "research_runs/xgb_maker"
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_rr(symk):
    d = np.load(io.BytesIO(bk.blob(f"{RR}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"], "ts": d["ts"],
            "pnl_long": d["pnl_long"].astype(np.float32), "pnl_short": d["pnl_short"].astype(np.float32),
            "fill_long": d["fill_long"].astype(bool), "fill_short": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "cfgs": m["cfgs"], "fee": m["maker_rt_fee_pct"] * 100.0}


def split(day, ndays):
    cut = int(ndays * SPLIT[0]); emb = int(ndays * SPLIT[1]); tr = day < cut
    td = sorted(set(day[tr].tolist())); vcut = td[int(len(td) * SPLIT[2])] if td else cut
    return (tr & (day < vcut)), (tr & (day >= vcut)), (day >= emb)


def pct_rank(x):
    o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score):
    order = np.lexsort((-score, day)); ds = day[order]
    first = np.ones(len(order), bool); first[1:] = ds[1:] != ds[:-1]; return order[first]


def fit(hp, niter, X, y, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def rr_lab(cfgs, c):
    cf = cfgs[c]
    return "hold" if cf["tp"] >= 1 else f"RR{cf['tp']/cf['sl']:.1f}({cf['tp']}/{cf['sl']})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=SYMS); a = ap.parse_args()
    def log(s): print(s, flush=True)
    hpB = jload(f"{MAIN}/B_pool.json")
    res = {"per_symbol": {}, "policy": "discover c* by oracle-best-side EV on train; B2 on c* vs hold-60s baseline; daily-budget A^B eval"}
    log(f"{'SYM':5s} {'c*_RR':>16s} {'oracleEV*':>9s} {'oracleEV_hold':>13s} | EV/tr 1/day: hold->c*   | top25%: hold->c*")
    for symk in a.symbols:
        d = load_rr(symk); hpA = jload(f"{MAIN}/A_{symk}.json")
        F = d["F"]; rH = d["rH"]; day = d["day"]; fee = d["fee"]; cfgs = d["cfgs"]; NC = len(cfgs)
        trn, val, te = split(day, d["ndays"])
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        nf = (np.abs(rH) >= thr) & np.isfinite(rH)
        fl = d["fill_long"][0]; fs = d["fill_short"][0]
        netl = d["pnl_long"][:, 0, :].astype(np.float64) * 100.0 - fee   # (NC,N)
        nets = d["pnl_short"][:, 0, :].astype(np.float64) * 100.0 - fee
        # ---- (1) discover c* : oracle best-side mean net EV on non-flat-fillable TRAIN ----
        m = nf & trn & (fl | fs); mi = np.where(m)[0]
        oev = np.array([np.nanmean(np.maximum(np.where(fl[mi], netl[c, mi], -np.inf),
                                              np.where(fs[mi], nets[c, mi], -np.inf))) for c in range(NC)])
        cstar = int(np.argmax(oev)); chold = 0
        # ---- (2) train A (reuse main HP) + B for hold & c* (reuse pooled B HP) ----
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
        pA = bstA.predict(xgb.DMatrix(F[te]))

        def trainB_pred(c):
            keep = nf & (fl | fs) & trn
            nl = netl[c]; ns = nets[c]
            yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
            both = fl & fs
            w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
            wc = np.clip(w[keep], 0, np.quantile(w[keep][w[keep] > 0], 0.99) if (w[keep] > 0).any() else 1.0)
            b = fit(hpB["best_params"], hpB["best_iter"], F[keep], yB[keep], w=wc)
            return b.predict(xgb.DMatrix(F[te]))

        # ---- (3) daily-budget A^B EV on TEST for a given config + its B preds ----
        def dailyEV(c, pB):
            nl = netl[c][te]; ns = nets[c][te]; flt = fl[te]; fst = fs[te]
            score = pct_rank(pA) * pct_rank(np.abs(pB - 0.5))
            sel = daily_pick(day[te], score)
            pl = pB[sel] >= 0.5; net = np.where(pl, nl[sel], ns[sel]); fc = np.where(pl, flt[sel], fst[sel])
            ex = fc & np.isfinite(net); ev_all = float(net[ex].mean()) if ex.any() else float("nan")
            ssc = score[sel]; top = np.argsort(-ssc)[:max(5, len(sel) // 4)]
            ex2 = fc[top] & np.isfinite(net[top]); ev25 = float(net[top][ex2].mean()) if ex2.any() else float("nan")
            return ev_all, ev25, int(ex.sum())

        evh_all, evh_25, nh = dailyEV(chold, trainB_pred(chold))
        evs_all, evs_25, ns_ = dailyEV(cstar, trainB_pred(cstar))
        res["per_symbol"][symk] = {"c_star": rr_lab(cfgs, cstar), "c_star_idx": cstar,
                                   "oracleEV_cstar": float(oev[cstar]), "oracleEV_hold": float(oev[chold]),
                                   "hold_EV_1day": evh_all, "cstar_EV_1day": evs_all,
                                   "hold_EV_top25": evh_25, "cstar_EV_top25": evs_25, "n_1day": nh}
        log(f"{symk:5s} {rr_lab(cfgs, cstar):>16s} {oev[cstar]:+9.2f} {oev[chold]:+13.2f} | "
            f"{evh_all:+6.2f} -> {evs_all:+6.2f}        | {evh_25:+6.2f} -> {evs_25:+6.2f}")
    # pooled
    P = res["per_symbol"]
    for k, lab in [("hold_EV_1day", "hold 1/day"), ("cstar_EV_1day", "c* 1/day"),
                   ("hold_EV_top25", "hold top25%"), ("cstar_EV_top25", "c* top25%")]:
        v = [P[s][k] for s in P if np.isfinite(P[s][k])]
        log(f"  POOLED {lab:14s} = {np.mean(v):+.2f}bp" if v else f"  {lab}: n/a")
    bk.blob(f"{RR}/B2_RESULT.json").upload_from_string(json.dumps(res, default=float))
    log(f"\n[saved] gs://{BUCKET}/{RR}/B2_RESULT.json")


if __name__ == "__main__":
    main()
