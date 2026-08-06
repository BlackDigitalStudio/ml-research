#!/usr/bin/env python3
"""OBJSEL rev2 readout: does the rev1 trial spread survive the REFIT to deployed semantics?

rev1 scored the 25 captured B-trials as TRIAL BOOSTERS (fit on the subsampled tuning set
sb) and found a 17-27bp peak-to-peak in test EV across trials. The deployed Bg/Bf are a
REFIT on the gated/full train window carrying the winning trial's hyperparameters, so
that spread was established for the search's own object, not for what the protocol
deploys. rev2 refit three arms end-to-end; this script turns them into cells.

ARMS. INC is not recomputed: the rev2 parity gate proved that injecting the incumbent's
own hyperparameters reproduces the stored PERFOLD bit-exactly, so PERFOLD_S{s} IS the INC
arm. MAXEV / MINEV are the trials with the highest / lowest rev1 test EV -- selected ON
the outcome, so their rev1 gap is 100% selection-on-noise and regression to the mean is
guaranteed; the question is only whether it goes all the way to zero. EVBUD is the trial
ev_budget would have picked.

PRIMARY: MAXEV - MINEV, paired per (seed, fold), against the ~20bp it holds by
construction in rev1. SECONDARY: EVBUD - INC, the head-to-head.

causal_rolling is copied VERBATIM from the frozen trainer, and the budgets are the
deployed ones, so these cells are on the same footing as rev1's.

Env: SYMS, SEEDS, OUT.
"""
import io, json, os

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = [s for s in os.environ.get("SYMS", "DOGE,XRP,BTC,ETH").split(",") if s]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3").split(",") if x != ""]
ARMS = ["MAXEV", "MINEV", "EVBUD"]
OUT = os.environ.get("OUT", "research_runs/objsel/ARMCELLS.json")
KDAYS = 30
# symbol -> (artifact sub, deployed budget)
CFG = {"DOGE": ("research_runs/maker_labels_tb3s_h150anch", 10),
       "XRP": ("research_runs/maker_labels_tb3s_h150anch", 5),
       "BTC": ("research_runs/maker_labels_tb3s_h150d", 5),
       "ETH": ("research_runs/maker_labels_tb3s_h150danch_v2notod", 5)}
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


def causal_rolling(sc_tr, sc_te, day_tr, day_te, target_tpd, sideB, fl_, fs_, nl_, ns_):
    """subs60_xgb_optuna_ic.causal_rolling, verbatim."""
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - target_tpd / max(wpd, 1.0))
    tr_days = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, tr_days[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for dd in days:
        idx = np.where(day_te == dd)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl_[sel], ns_[sel]); fc = np.where(side, fl_[sel], fs_[sel])
    ex = fc & np.isfinite(net); return net[ex]


def cell_ev(sub, tag, sym, seed, fold, budget):
    key = f"{sub}/PERFOLD{tag}_{sym}_qm0_f{fold}.npz"
    try:
        z = np.load(io.BytesIO(bk.blob(key).download_as_bytes()))
    except Exception:
        return None
    a = causal_rolling(z["axb_tr"].astype(np.float64), z["axb_te"].astype(np.float64),
                       z["day_tr"].astype(int), z["day_te"].astype(int), budget,
                       z["side"].astype(bool), z["fl"].astype(bool), z["fs"].astype(bool),
                       z["netl"].astype(np.float64), z["nets"].astype(np.float64))
    return (float(a.mean()), int(len(a))) if len(a) else (float("nan"), 0)


res = {"syms": SYMS, "seeds": SEEDS, "arms": ARMS, "cells": []}
for sym in SYMS:
    sub, budget = CFG[sym]
    for s in SEEDS:
        for f in range(6):
            row = {"sym": sym, "seed": s, "fold": f}
            inc = cell_ev(sub, f"_S{s}", sym, s, f, budget)
            if inc is None:
                continue
            row["INC"], row["INC_n"] = inc
            ok = True
            for arm in ARMS:
                v = cell_ev(sub, f"_{arm}_S{s}", sym, s, f, budget)
                if v is None:
                    ok = False; break
                row[arm], row[f"{arm}_n"] = v
            if ok:
                res["cells"].append(row)
    log(f"  {sym}: {sum(1 for c in res['cells'] if c['sym'] == sym)} cells")


def paired(sym, a, b):
    d = np.array([c[a] - c[b] for c in res["cells"]
                  if c["sym"] == sym and np.isfinite(c[a]) and np.isfinite(c[b])])
    if len(d) < 3:
        return None
    return dict(n=len(d), mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))


log("\n=== PRIMARY: does the trial spread survive the refit?")
log("    MAXEV - MINEV, paired per (seed,fold). rev1 held ~20bp BY CONSTRUCTION.")
log(f"    {'sym':>5} {'n':>3} {'refit gap':>10} {'+-SE':>6} {'t':>6}")
res["primary"] = {}
for sym in SYMS:
    p = paired(sym, "MAXEV", "MINEV")
    res["primary"][sym] = p
    if p:
        log(f"    {sym:>5} {p['n']:>3} {p['mean']:>+10.2f} {p['se']:>6.2f} {p['mean']/p['se']:>+6.2f}")

log("\n=== SECONDARY: the head-to-head you asked for")
log("    EVBUD - INC, paired per (seed,fold)")
log(f"    {'sym':>5} {'n':>3} {'delta bp':>10} {'+-SE':>6} {'t':>6}")
res["secondary"] = {}
for sym in SYMS:
    p = paired(sym, "EVBUD", "INC")
    res["secondary"][sym] = p
    if p:
        log(f"    {sym:>5} {p['n']:>3} {p['mean']:>+10.2f} {p['se']:>6.2f} {p['mean']/p['se']:>+6.2f}")

log("\n=== arm means per symbol (EV/tr over cells)")
log(f"    {'sym':>5} {'INC':>8} {'MAXEV':>8} {'MINEV':>8} {'EVBUD':>8}")
res["arm_means"] = {}
for sym in SYMS:
    cs = [c for c in res["cells"] if c["sym"] == sym]
    m = {k: float(np.nanmean([c[k] for c in cs])) for k in ["INC"] + ARMS}
    res["arm_means"][sym] = m
    log(f"    {sym:>5} {m['INC']:>+8.2f} {m['MAXEV']:>+8.2f} {m['MINEV']:>+8.2f} {m['EVBUD']:>+8.2f}")

bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"\n[saved] gs://{BUCKET}/{OUT}")
