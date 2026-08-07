#!/usr/bin/env python3
"""HBV1 rev18A (non-frozen analysis): CONSENSUS-SIZED UNION.

The rev14 K-ladder is monotone in EV => a binary k>=K gate throws information
away. Policy: trade EVERY union decision, position weight w = (k/N)^gamma where
k = number of members whose own causal threshold the decision passed. gamma=0
reproduces union; gamma->inf approaches the k=N intersection. Fee scales with
notional (net-FEE applied per unit, then weighted).

Weighted battery per (pool, T_s, gamma): per-unit EV, exposure/day (sum w),
weighted hit, daily-compounded Sharpe/Sortino/maxDD/worst-day at F=1, per-fold
unit-EV + LOFO + neg-fold count, day-block BOOT L=7 CI90 of unit-EV and bpd.
Env: POOLS "name=sub:seed,..." (defaults below), TGTS, GAMMAS, FEE_BP.
"""
import io
import json
import os

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = os.environ.get("SYM", "DOGE")
KDAYS = 30
FEE_BP = float(os.environ.get("FEE_BP", "4"))
TGTS = [float(x) for x in os.environ.get("TGTS", "2.5,5,10").split(",")]
GAMMAS = [float(x) for x in os.environ.get("GAMMAS", "0,0.5,1,2,3").split(",")]
L, REPS = 7, 1000
B = "maker_labels_tb3s_h150anch"

POOLS = {
    "rf4": [(B + f"_v2_nooi_rf{j}", j) for j in range(4)],
    "fbag-mix8": [(B + "_v2_nooi", s) for s in range(4)] + [(B + f"_v2_nooi_fb{j}", j) for j in range(4)],
    "rf-mix8": [(B + "_v2_nooi", s) for s in range(4)] + [(B + f"_v2_nooi_rf{j}", j) for j in range(4)],
    "v2-nooi8": [(B + "_v2_nooi", s) for s in range(8)],
    "v1-nooi8": [(B + "_v1_nooi", s) for s in range(8)],
}
_env_pools = os.environ.get("POOLS", "")
if _env_pools:
    POOLS = {}
    for spec in _env_pools.split(";"):
        name, members = spec.split("=")
        POOLS[name] = [(p.split(":")[0], int(p.split(":")[1])) for p in members.split(",")]

_cache = {}


def load(sub, seed, f):
    k = (sub, seed, f)
    if k not in _cache:
        _cache[k] = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/PERFOLD_S{seed}_{SYM}_qm0_f{f}.npz")
                                       .download_as_bytes()))
    return _cache[k]


def causal_sel(z, tgt):
    sc_tr = z["axb_tr"].astype(np.float64); sc_te = z["axb_te"].astype(np.float64)
    day_tr = z["day_tr"]; day_te = z["day_te"]
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return set(sel)


def weighted_battery(trades, fold_ids, days_sorted, nf):
    """trades: list of (w, net, day). Sequential compounding at F=1, fraction w."""
    if not trades:
        return {"n": 0}
    w = np.array([t[0] for t in trades]); net = np.array([t[1] for t in trades])
    day = np.array([t[2] for t in trades]); fid = np.array(fold_ids)
    span = max(len(days_sorted), 1)
    unit_ev = float((w * net).sum() / w.sum())
    out = dict(n=len(w), exposure_pd=float(w.sum() / span), unit_ev=unit_ev,
               bpd=float((w * net).sum() / span),
               hit_w=float((w * (net > 0)).sum() / w.sum()))
    eq = np.cumprod(1.0 + w * net * 1e-4)
    curve = np.concatenate([[1.0], eq])
    out["roi_monthly"] = float(eq[-1] ** (30.0 / span) - 1.0)
    out["maxdd"] = float((1.0 - curve / np.maximum.accumulate(curve)).max())
    by_day = {}
    for wi, ni, di in zip(w, net, day):
        by_day.setdefault(int(di), []).append((wi, ni))
    dret = np.array([float(np.prod([1.0 + wi * ni * 1e-4 for wi, ni in by_day.get(int(d), [])]) - 1.0)
                     for d in days_sorted])
    out["worst_day"] = float(dret.min())
    out["sharpe"] = float(dret.mean() / dret.std() * np.sqrt(365.0)) if dret.std() > 0 else 0.0
    dn = dret[dret < 0]
    out["sortino"] = float(dret.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else float("inf")
    # per-fold unit EV + LOFO
    pf = []
    for f in range(nf):
        m = fid == f
        pf.append(float((w[m] * net[m]).sum() / w[m].sum()) if w[m].sum() > 0 else None)
    out["perfold_unit_ev"] = [round(x, 2) if x is not None else None for x in pf]
    out["neg_folds"] = sum(1 for x in pf if x is not None and x < 0)
    lofo = []
    for f in range(nf):
        m = fid != f
        lofo.append(float((w[m] * net[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float("nan"))
    out["lofo_min"] = float(np.nanmin(lofo)) if lofo else float("nan")
    # day-block bootstrap of unit_ev and bpd
    rng = np.random.default_rng(1)
    b_ev, b_bpd = [], []
    for _ in range(REPS):
        picked = []
        while len(picked) < span:
            i0 = rng.integers(0, max(span - L, 1))
            picked.extend(days_sorted[i0:i0 + L])
        ws, wn = 0.0, 0.0
        for d in picked[:span]:
            for wi, ni in by_day.get(int(d), []):
                ws += wi; wn += wi * ni
        if ws > 0:
            b_ev.append(wn / ws); b_bpd.append(wn / span)
    out["boot_ev_p5"] = float(np.quantile(b_ev, .05)) if b_ev else float("nan")
    out["boot_ev_p95"] = float(np.quantile(b_ev, .95)) if b_ev else float("nan")
    out["boot_bpd_p5"] = float(np.quantile(b_bpd, .05)) if b_bpd else float("nan")
    out["Ppos"] = float(100 * np.mean(np.array(b_ev) > 0)) if b_ev else float("nan")
    out["gate_pass"] = bool(out["neg_folds"] == 0 and out["boot_ev_p5"] > 0)
    return out


allout = {}
for pname, members in POOLS.items():
    N = len(members)
    nf = sum(1 for b in bk.client.list_blobs(
        bk, prefix=f"research_runs/{members[0][0]}/PERFOLD_S{members[0][1]}_{SYM}_qm0_f")
        if b.name.endswith(".npz"))
    Z = {m: [load(m[0], m[1], f) for f in range(nf)] for m in members}
    days_sorted = sorted({int(d) for f in range(nf) for d in np.unique(Z[members[0]][f]["day_te"])})
    print(f"\n### pool {pname}: N={N}, folds={nf}, {len(days_sorted)} days", flush=True)
    allout[pname] = {}
    for tgt in TGTS:
        base = []  # (k, net, day, fold)
        for f in range(nf):
            sets = {m: causal_sel(Z[m][f], tgt) for m in members}
            z0 = Z[members[0]][f]
            for i in sorted(set().union(*sets.values())):
                ks = [m for m in members if i in sets[m]]
                sides = [bool(Z[m][f]["side"][i]) for m in ks]
                nl = sum(sides)
                if nl * 2 == len(sides):
                    continue
                s_ = nl * 2 > len(sides)
                net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
                fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
                if fill and np.isfinite(net):
                    base.append((len(ks), net - FEE_BP, int(z0["day_te"][i]), f))
        for gamma in GAMMAS:
            trades = [((k / N) ** gamma if gamma > 0 else 1.0, net, d) for k, net, d, _ in base]
            m = weighted_battery(trades, [f for _, _, _, f in base], days_sorted, nf)
            allout[pname][f"T{tgt:g}_g{gamma:g}"] = m
            if m.get("n"):
                print(f"  T{tgt:g} g={gamma:<4g} unitEV {m['unit_ev']:+6.2f} expo/d {m['exposure_pd']:5.2f} "
                      f"hitW {100*m['hit_w']:4.1f}% ROI/mo {100*m['roi_monthly']:+6.1f}% DD {-100*m['maxdd']:5.1f}% "
                      f"Sh {m['sharpe']:4.2f} So {min(m['sortino'],99):5.2f} | LOFOmin {m['lofo_min']:+6.2f} "
                      f"negF {m['neg_folds']} floor {m['boot_ev_p5']:+6.2f} bpd_p5 {m['boot_bpd_p5']:+5.2f} "
                      f"{'PASS' if m['gate_pass'] else 'fail'}", flush=True)

bk.blob(f"research_runs/HBV1_SIZEDU_{SYM}.json").upload_from_string(json.dumps(allout, default=float))
print("\n[saved HBV1_SIZEDU]", flush=True)
