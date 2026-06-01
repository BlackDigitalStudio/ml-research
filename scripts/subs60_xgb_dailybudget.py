#!/usr/bin/env python3
"""Daily-budget policy: filter by BOTH models' confidence, >=1 trade per symbol per day.

OFFLINE from saved preds_{SYM}.npz (ts, day, pA, pB, pnl_long/short [NC,QM,N], fill_*[QM,N]).
No retraining. Policy:
  - combined confidence per window = pct_rank(pA) * pct_rank(|pB-0.5|)  -> high only when BOTH A
    (vol-gate) and B (direction) are confident.
  - budget 1 trade/symbol/day: per (symbol,day) take the SINGLE max-combined-confidence window,
    trade B's side, executed NET maker EV (filled-chosen) on the chosen cfg/qm.
  - cross-day selectivity sweep: rank the daily picks by their combined score, keep the top f% of
    days (f=100 -> trade every day; smaller f -> only the most-confident days).
Reports per symbol + pooled: EV/trade(bp), WR, dir-acc, fill, trades/day, n_trades. cfg = hold-60s,
both touch(qm0) & queue(qm1). Saves -> research_runs/<sub>/DAILYBUDGET.json.
Run: python3 subs60_xgb_dailybudget.py --symbols ALL --sub xgb_maker
"""
import argparse, io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
F_DAYS = [100.0, 50.0, 25.0, 10.0]    # cross-day selectivity (% of days kept, by combined score)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def pct_rank(x):
    o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def load(sub, symk):
    try:
        d = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/preds_{symk}.npz").download_as_bytes()), allow_pickle=True)
    except Exception:
        return None
    return {"day": d["day"], "pA": d["pA"].astype(np.float64), "pB": d["pB"].astype(np.float64),
            "pnl_long": d["pnl_long"], "pnl_short": d["pnl_short"],
            "fill_long": d["fill_long"].astype(bool), "fill_short": d["fill_short"].astype(bool),
            "meta": json.loads(str(d["meta"]))}


def daily_pick(day, score):
    """One window per day = the max-score window. Returns indices (vectorized)."""
    order = np.lexsort((-score, day))            # day asc, score desc within day
    ds = day[order]; first = np.ones(len(order), bool); first[1:] = ds[1:] != ds[:-1]
    return order[first]


def policy(P, cfg, qm):
    fee = P["meta"]["fee_bp"]
    nl = P["pnl_long"][cfg, qm].astype(np.float64) * 100.0 - fee
    ns = P["pnl_short"][cfg, qm].astype(np.float64) * 100.0 - fee
    fl = P["fill_long"][qm]; fs = P["fill_short"][qm]
    score = pct_rank(P["pA"]) * pct_rank(np.abs(P["pB"] - 0.5))    # BOTH-confident
    sel = daily_pick(P["day"], score)                              # 1 window/day (budget)
    pl = P["pB"][sel] >= 0.5
    net = np.where(pl, nl[sel], ns[sel]); fill = np.where(pl, fl[sel], fs[sel])
    better = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf))[sel]
    ssc = score[sel]; ndays = len(sel)
    out = {}
    for f in F_DAYS:                                               # keep top f% of days by combined score
        k = max(5, int(ndays * f / 100)); top = np.argsort(-ssc)[:k]
        ex = fill[top] & np.isfinite(net[top])
        out[f"top{f}d"] = {"EV_bp": float(net[top][ex].mean()) if ex.any() else float("nan"),
                           "WR": float((net[top][ex] > 0).mean()) if ex.any() else float("nan"),
                           "dir_acc": float((pl[top] == better[top]).mean()),
                           "fill": float(fill[top].mean()), "n_trades": int(ex.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=SYMS); ap.add_argument("--sub", default="xgb_maker")
    ap.add_argument("--cfg-idx", type=int, default=0); a = ap.parse_args()
    def log(s): print(s, flush=True)
    P = {s: load(a.sub, s) for s in a.symbols}; P = {k: v for k, v in P.items() if v}
    qms = P[list(P)[0]]["meta"]["queue_mults"]
    res = {"sub": a.sub, "cfg_idx": a.cfg_idx, "policy": "score=pctrank(pA)*pctrank(|pB-.5|); 1 trade/sym/day",
           "by_qm": {}}
    for qm_name, qm in [("touch", list(qms).index(0.0)), ("queue", list(qms).index(1.0))]:
        log(f"\n=== DAILY BUDGET (1 trade/sym/day, BOTH-confident) | hold-60s / {qm_name} ===")
        log(f"{'SYM':5s}  EV/tr @ days top 100/50/25/10%            WR@all  dir@all  fill  trd/all(n)")
        rows = {}
        for s in P:
            r = policy(P[s], a.cfg_idx, qm); rows[s] = r; a100 = r["top100.0d"]
            log(f"{s:5s}  {r['top100.0d']['EV_bp']:+5.1f}/{r['top50.0d']['EV_bp']:+5.1f}/{r['top25.0d']['EV_bp']:+5.1f}/"
                f"{r['top10.0d']['EV_bp']:+5.1f}bp   {a100['WR']:.2f}   {a100['dir_acc']:.2f}   {a100['fill']:.2f}   {a100['n_trades']}")
        # pooled mean over symbols
        for f in F_DAYS:
            evs = [rows[s][f"top{f}d"]["EV_bp"] for s in rows if np.isfinite(rows[s][f"top{f}d"]["EV_bp"])]
            log(f"  POOLED mean EV @ top{f}d = {np.mean(evs):+.2f}bp" if evs else f"  top{f}d: n/a")
        res["by_qm"][qm_name] = rows
    bk.blob(f"research_runs/{a.sub}/DAILYBUDGET.json").upload_from_string(json.dumps(res, default=float))
    log(f"\n[saved] gs://{BUCKET}/research_runs/{a.sub}/DAILYBUDGET.json")


if __name__ == "__main__":
    main()
