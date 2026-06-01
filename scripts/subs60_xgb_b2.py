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


def daily_pick(day, score, n_per_day=1):
    """Top-n_per_day windows per day by score (budget). Returns selected indices."""
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


def rr_lab(cfgs, c):
    cf = cfgs[c]
    return "hold" if cf["tp"] >= 1 else f"RR{cf['tp']/cf['sl']:.1f}({cf['tp']}/{cf['sl']})"


def oof_pA(F, yA, trn, day, hpA, kfolds=5):
    """Out-of-fold A predictions on the TRAIN rows (K-fold by day) -> realistic A-predicted gate
    that includes A's false positives (not in-sample-overfit)."""
    tdays = sorted(set(day[trn].tolist()))
    fold = {d: i % kfolds for i, d in enumerate(tdays)}
    fday = np.array([fold.get(int(d), -1) for d in day])
    oof = np.full(len(F), np.nan)
    for k in range(kfolds):
        trk = trn & (fday != k); vak = trn & (fday == k)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk)
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=SYMS)
    ap.add_argument("--gate", choices=["oracle", "apred"], default="apred")  # B/c* training universe
    ap.add_argument("--gate-pct", type=float, default=5.0)   # apred: top-q% by OOF pA = "A predicts non-flat"
    ap.add_argument("--budget", type=int, default=1)         # trades/symbol/day (higher = more samples = less selection noise)
    ap.add_argument("--kfolds", type=int, default=5); a = ap.parse_args()
    def log(s): print(s, flush=True)
    log(f"[gate={a.gate} gate_pct={a.gate_pct}] (B trains on "
        f"{'A-OOF-predicted non-flat top-' + str(a.gate_pct) + '%' if a.gate == 'apred' else 'oracle realized non-flat'} windows)")
    hpB = jload(f"{MAIN}/B_pool.json")
    res = {"per_symbol": {}, "policy": "Stage1 base-B (hold) -> GRID on VAL (model-sided, A-gated) per config -> c* -> Stage2 B on c*; daily-budget A^B eval on test"}
    log(f"{'SYM':5s} {'c*_RR':>16s}  grid valEV*/hold  | 1day hold->c*      | top25 hold->c*")
    for symk in a.symbols:
        d = load_rr(symk); hpA = jload(f"{MAIN}/A_{symk}.json")
        F = d["F"]; rH = d["rH"]; day = d["day"]; fee = d["fee"]; cfgs = d["cfgs"]; NC = len(cfgs)
        trn, val, te = split(day, d["ndays"])
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        nf = (np.abs(rH) >= thr) & np.isfinite(rH)
        fl = d["fill_long"][0]; fs = d["fill_short"][0]
        netl = d["pnl_long"][:, 0, :].astype(np.float64) * 100.0 - fee   # (NC,N)
        nets = d["pnl_short"][:, 0, :].astype(np.float64) * 100.0 - fee
        vi = np.where(val)[0]; ti = np.where(te)[0]; chold = 0
        # ---- A (reuse main HP): train on train, predict val + test ----
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
        pA_v = bstA.predict(xgb.DMatrix(F[vi])); pA_t = bstA.predict(xgb.DMatrix(F[ti]))
        # ---- B training gate (which TRAIN windows B sees) ----
        if a.gate == "apred":                              # A-OOF-predicted non-flat (top gate_pct%) on train
            oof = oof_pA(F, yA, trn, day, hpA, a.kfolds)
            valid = trn & np.isfinite(oof)
            thr_oof = float(np.nanquantile(oof[valid], 1 - a.gate_pct / 100.0)) if valid.any() else np.inf
            gate_train = valid & (oof >= thr_oof)
        else:
            gate_train = nf & trn
        gate_rate = float(gate_train.sum() / max(trn.sum(), 1))

        def trainB(c, idx):                                # train B on gate_train (config-c better-side), predict F[idx]
            keep = gate_train & (fl | fs); nl = netl[c]; ns = nets[c]
            yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
            both = fl & fs
            w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
            wc = np.clip(w[keep], 0, np.quantile(w[keep][w[keep] > 0], 0.99) if (w[keep] > 0).any() else 1.0)
            b = fit(hpB["best_params"], hpB["best_iter"], F[keep], yB[keep], w=wc)
            return b.predict(xgb.DMatrix(F[idx]))

        # ---- STAGE 1: base B (hold), predict val (for grid) + test (baseline) ----
        pB1_v = trainB(chold, vi); pB1_t = trainB(chold, ti)
        # ---- GRID on VAL using the SAME daily-budget policy as deploy (1 trade/day), model-sided ----
        #      (grid metric MUST match the deploy metric; the config selection is identical across c
        #       since the daily-pick score = pA x |pB1-0.5| is config-independent.)
        score_v = pct_rank(pA_v) * pct_rank(np.abs(pB1_v - 0.5))
        selv = daily_pick(day[vi], score_v, a.budget)  # budget trades/day on val
        side1 = pB1_v[selv] >= 0.5; flv = fl[vi][selv]; fsv = fs[vi][selv]
        gridev = []
        for c in range(NC):
            net = np.where(side1, netl[c][vi][selv], nets[c][vi][selv]); fc = np.where(side1, flv, fsv)
            ex = fc & np.isfinite(net)
            gridev.append(float(net[ex].mean()) if ex.sum() >= 10 else -1e9)
        cstar = int(np.argmax(gridev))
        # ---- STAGE 2: B on c* (== stage1 if c*==hold), predict test ----
        pB2_t = pB1_t if cstar == chold else trainB(cstar, ti)

        def dailyEV(c, pB_t):                              # daily-budget A^B on TEST (1 trade/sym/day)
            nl = netl[c][ti]; ns = nets[c][ti]; flt = fl[ti]; fst = fs[ti]
            score = pct_rank(pA_t) * pct_rank(np.abs(pB_t - 0.5))
            sel = daily_pick(day[ti], score, a.budget)
            pl = pB_t[sel] >= 0.5; net = np.where(pl, nl[sel], ns[sel]); fc = np.where(pl, flt[sel], fst[sel])
            ex = fc & np.isfinite(net); ev_all = float(net[ex].mean()) if ex.any() else float("nan")
            ssc = score[sel]; top = np.argsort(-ssc)[:max(5, len(sel) // 4)]
            ex2 = fc[top] & np.isfinite(net[top]); ev25 = float(net[top][ex2].mean()) if ex2.any() else float("nan")
            return ev_all, ev25, int(ex.sum())

        evh_all, evh_25, nh = dailyEV(chold, pB1_t)
        evs_all, evs_25, ns_ = dailyEV(cstar, pB2_t)
        res["per_symbol"][symk] = {"c_star": rr_lab(cfgs, cstar), "c_star_idx": cstar,
                                   "grid_valEV_cstar": float(gridev[cstar]), "grid_valEV_hold": float(gridev[chold]),
                                   "hold_EV_1day": evh_all, "cstar_EV_1day": evs_all,
                                   "hold_EV_top25": evh_25, "cstar_EV_top25": evs_25, "n_1day": nh,
                                   "gate": a.gate, "gate_pct": a.gate_pct, "gate_train_rate": gate_rate}
        log(f"{symk:5s} {rr_lab(cfgs, cstar):>16s}  {gridev[cstar]:+6.2f}/{gridev[chold]:+6.2f}  | "
            f"{evh_all:+6.2f}->{evs_all:+6.2f}  | {evh_25:+6.2f}->{evs_25:+6.2f}")
    # pooled
    P = res["per_symbol"]
    for k, lab in [("hold_EV_1day", "hold 1/day"), ("cstar_EV_1day", "c* 1/day"),
                   ("hold_EV_top25", "hold top25%"), ("cstar_EV_top25", "c* top25%")]:
        v = [P[s][k] for s in P if np.isfinite(P[s][k])]
        log(f"  POOLED {lab:14s} = {np.mean(v):+.2f}bp" if v else f"  {lab}: n/a")
    tag = f"B2_RESULT_{a.gate}" if (a.gate == "oracle" or a.gate_pct == 5.0) else f"B2_RESULT_{a.gate}{int(a.gate_pct)}"
    if a.budget != 1:
        tag += f"_b{a.budget}"
    bk.blob(f"{RR}/{tag}.json").upload_from_string(json.dumps(res, default=float))
    log(f"\n[saved] gs://{BUCKET}/{RR}/{tag}.json")


if __name__ == "__main__":
    main()
