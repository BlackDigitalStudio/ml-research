#!/usr/bin/env python3
"""Express the DOGE walk-forward OOS result as PnL in PERCENT (cache HIT, no training).

Loads the cached per-fold predictions (research_runs/wf_cache/DOGE_adaptive_W200T30_preds.npz),
applies the deploy selection per (weighting, budget), and reports returns in %:
  EV/trade, daily, cumulative over the OOS period (simple + compounded), monthly, annualized.
Sizing assumption: 1x capital per trade (return on deployed capital), trades intraday
non-overlapping at b<=10, hold-60s. EV is net of 4bp maker RT. maker-SIM fills.
"""
import io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
CACHE_PREDS = "research_runs/wf_cache/DOGE_adaptive_W200T30_preds.npz"
CONFIGS = [("A=B", 1, 1, 1), ("A2", 2, 1, 1), ("A2", 2, 1, 5), ("A2", 2, 1, 10),
           ("A3", 3, 1, 1), ("A3", 3, 1, 5), ("A3", 3, 1, 10)]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def pct_rank(x): o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


z = np.load(io.BytesIO(bk.blob(CACHE_PREDS).download_as_bytes()), allow_pickle=True)
fid = z["fold_id"]
folds = []
for i in sorted(set(fid.tolist())):
    m = fid == i
    folds.append({"day": z["day"][m], "pA": z["pA"][m], "pB": z["pB"][m], "fl": z["fl"][m],
                  "fs": z["fs"][m], "nl": z["nl"][m], "ns": z["ns"][m], "ndays": len(set(z["day"][m].tolist()))})
tot_days = sum(f["ndays"] for f in folds)
print(f"[cache HIT] {len(folds)} folds, OOS test span = {tot_days} days (crypto 24/7 ~ {tot_days/30.4:.1f} months)\n")


def sel_nets(fd, wA, wB, budget):
    a = pct_rank(fd["pA"]); b = pct_rank(np.abs(fd["pB"] - 0.5))
    score = (a ** wA) * (b ** wB) if (wA or wB) else np.ones_like(a)
    sel = daily_pick(fd["day"], score, budget)
    side = fd["pB"][sel] >= 0.5; net = np.where(side, fd["nl"][sel], fd["ns"][sel])
    fc = np.where(side, fd["fl"][sel], fd["fs"][sel]); ex = fc & np.isfinite(net)
    return net[ex]


print(f"{'config':>10} {'EV/trade':>9} {'trd/day':>7} {'daily':>7} | {'OOS cum':>9} {'(comp)':>8} {'monthly':>8} {'annual':>8} {'(comp)':>9}")
for lab, wA, wB, bud in CONFIGS:
    nets = np.concatenate([sel_nets(fd, wA, wB, bud) for fd in folds])  # net bp per trade
    n = len(nets); ev_bp = float(nets.mean()); sum_bp = float(nets.sum())
    ev_pct = ev_bp * 0.01
    tpd = n / tot_days
    daily_pct = ev_bp * tpd * 0.01
    cum_pct = sum_bp * 0.01                                  # simple, 1x notional
    d = daily_pct / 100.0
    cum_comp = ((1 + d) ** tot_days - 1) * 100
    monthly = daily_pct * 30.4
    ann_s = daily_pct * 365
    ann_c = ((1 + d) ** 365 - 1) * 100
    print(f"{lab + '/b' + str(bud):>10} {ev_pct:>+8.4f}% {tpd:>7.1f} {daily_pct:>+6.3f}% | "
          f"{cum_pct:>+8.1f}% {cum_comp:>+7.1f}% {monthly:>+7.1f}% {ann_s:>+7.0f}% {ann_c:>+8.0f}%")
print("\nsimple = sum of per-trade returns (constant notional); comp = compounded daily; "
      "annual = simple x365 / compounded^365 (crypto 24/7). 1x capital/trade, net 4bp maker, maker-SIM fills.")
