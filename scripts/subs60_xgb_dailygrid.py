#!/usr/bin/env python3
"""Daily-budget with EXPLICIT separate confidence windows for A and B (2D grid).

OFFLINE from preds_{SYM}.npz. A window qualifies iff pA is in the top-qA% (vol-gate) AND |pB-0.5| is in
the top-qB% (direction conviction) -- a STRICT conjunction (not the soft product of dailybudget.py).
Budget 1 trade/symbol/day: among qualifying windows that day take the single highest-B-conviction one,
trade B's side, executed NET maker EV (hold-60s). Reports the pooled EV(bp) and trades/day matrix over
qA x qB so asymmetric cells (e.g. A top-10% x B top-1% vs A top-1% x B top-10%) are directly visible.
Saves -> research_runs/<sub>/DAILYGRID.json.
Run: python3 subs60_xgb_dailygrid.py --symbols ALL --sub xgb_maker --qm touch
"""
import argparse, io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
QA = [1.0, 2.0, 5.0, 10.0, 20.0]            # A vol-gate top-q%
QB = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]      # B direction-conviction top-q%
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
    order = np.lexsort((-score, day)); ds = day[order]
    first = np.ones(len(order), bool); first[1:] = ds[1:] != ds[:-1]
    return order[first]


def cell(P, cfg, qm, qA, qB):
    fee = P["meta"]["fee_bp"]
    nl = P["pnl_long"][cfg, qm].astype(np.float64) * 100.0 - fee
    ns = P["pnl_short"][cfg, qm].astype(np.float64) * 100.0 - fee
    fl = P["fill_long"][qm]; fs = P["fill_short"][qm]
    rA = pct_rank(P["pA"]); bconv = np.abs(P["pB"] - 0.5); rB = pct_rank(bconv)
    qual = (rA >= 1 - qA / 100.0) & (rB >= 1 - qB / 100.0)        # STRICT A AND B
    qi = np.where(qual)[0]
    ndays = len(set(P["day"].tolist()))
    if len(qi) < 3:
        return {"EV_bp": float("nan"), "WR": float("nan"), "dir_acc": float("nan"),
                "n": int(len(qi)), "trd_day": len(qi) / max(ndays, 1)}
    sel = qi[daily_pick(P["day"][qi], bconv[qi])]                 # 1/day: highest B-conviction qualifier
    pl = P["pB"][sel] >= 0.5
    net = np.where(pl, nl[sel], ns[sel]); fill = np.where(pl, fl[sel], fs[sel])
    better = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf))[sel]
    ex = fill & np.isfinite(net)
    return {"EV_bp": float(net[ex].mean()) if ex.any() else float("nan"),
            "WR": float((net[ex] > 0).mean()) if ex.any() else float("nan"),
            "dir_acc": float((pl == better).mean()), "n": int(ex.sum()), "trd_day": int(ex.sum()) / max(ndays, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=SYMS); ap.add_argument("--sub", default="xgb_maker")
    ap.add_argument("--cfg-idx", type=int, default=0); ap.add_argument("--qm", default="touch")
    a = ap.parse_args()
    def log(s): print(s, flush=True)
    P = {s: load(a.sub, s) for s in a.symbols}; P = {k: v for k, v in P.items() if v}
    qms = P[list(P)[0]]["meta"]["queue_mults"]; qm = list(qms).index(0.0 if a.qm == "touch" else 1.0)
    grid = {s: {f"qA{qa}_qB{qb}": cell(P[s], a.cfg_idx, qm, qa, qb) for qa in QA for qb in QB} for s in P}
    res = {"sub": a.sub, "qm": a.qm, "cfg_idx": a.cfg_idx, "QA": QA, "QB": QB,
           "policy": "STRICT pA in top-qA% AND |pB-0.5| in top-qB%; 1 trade/sym/day (max B-conviction qualifier)",
           "per_symbol": grid}
    log(f"\n=== DAILY-GRID: strict A AND B confidence windows | hold-60s / {a.qm} | POOLED mean EV(bp) ===")
    log("  (rows = A vol-gate top-qA% ; cols = B direction-conviction top-qB%)")
    log("qA\\qB   " + "  ".join(f"{qb:>6.0f}%" for qb in QB))
    for qa in QA:
        cells = []
        for qb in QB:
            evs = [grid[s][f"qA{qa}_qB{qb}"]["EV_bp"] for s in grid if np.isfinite(grid[s][f"qA{qa}_qB{qb}"]["EV_bp"])]
            cells.append(f"{np.mean(evs):+6.1f}" if evs else "   nan")
        log(f"{qa:>4.0f}%  " + "  ".join(cells))
    log(f"\n=== POOLED mean TRADES/DAY (feasibility of each cell) ===")
    log("qA\\qB   " + "  ".join(f"{qb:>6.0f}%" for qb in QB))
    for qa in QA:
        cells = []
        for qb in QB:
            td = [grid[s][f"qA{qa}_qB{qb}"]["trd_day"] for s in grid]
            cells.append(f"{np.mean(td):6.2f}")
        log(f"{qa:>4.0f}%  " + "  ".join(cells))
    bk.blob(f"research_runs/{a.sub}/DAILYGRID.json").upload_from_string(json.dumps(res, default=float))
    log(f"\n[saved] gs://{BUCKET}/research_runs/{a.sub}/DAILYGRID.json")


if __name__ == "__main__":
    main()
