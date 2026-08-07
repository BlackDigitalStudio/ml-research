#!/usr/bin/env python3
"""HBV1 analysis: normalize gate-PASS forms to a COMMON maxDD by scaling per-trade
capital fraction F (user request 2026-08-07). DD compounds nonlinearly in F, so F*
is found by bisection on the trade-level compounded curve. Fee-honest nets (4bp).
Env: TARGET_DD (default 0.50). Forms hard-listed = the gate-PASS set of the
extended audit."""
import io
import json
import os
import sys

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = "DOGE"
KDAYS = 30
FEE_BP = 4.0
TARGET_DD = float(os.environ.get("TARGET_DD", "0.50"))
B = "maker_labels_tb3s_h150anch"

FORMS = [
    ("v2-nooi4  union T1.25", [(B + "_v2_nooi", s) for s in range(4)], 1.25),
    ("v2-nooi4  union T0.625", [(B + "_v2_nooi", s) for s in range(4)], 0.625),
    ("xproto16  union T0.3125", [(B + "_v1_nooi", s) for s in range(8)] + [(B + "_v2_nooi", s) for s in range(8)], 0.3125),
    ("v1-nooi8  union T0.3125", [(B + "_v1_nooi", s) for s in range(8)], 0.3125),
    ("v2-nooi8  union T0.3125", [(B + "_v2_nooi", s) for s in range(8)], 0.3125),
    ("v2-nooi4  union T0.3125", [(B + "_v2_nooi", s) for s in range(4)], 0.3125),
]

_cache = {}


def load(sub, seed, f):
    k = (sub, seed, f)
    if k not in _cache:
        _cache[k] = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/PERFOLD_S{seed}_{SYM}_qm0_f{f}.npz")
                                       .download_as_bytes()))
    return _cache[k]


def causal_sel(sc_tr, sc_te, day_tr, day_te, tgt):
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return np.array(sel, dtype=int)


def union_nets(members, tgt):
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"research_runs/{members[0][0]}/PERFOLD_S{members[0][1]}_{SYM}_qm0_f")
             if b.name.endswith(".npz"))
    Z = {m: [load(m[0], m[1], f) for f in range(nf)] for m in members}
    nets, days = [], []
    days_all = set()
    for f in range(nf):
        sets = {m: set(causal_sel(Z[m][f]["axb_tr"].astype(np.float64), Z[m][f]["axb_te"].astype(np.float64),
                                  Z[m][f]["day_tr"], Z[m][f]["day_te"], tgt).tolist()) for m in members}
        z0 = Z[members[0]][f]
        days_all |= set(int(d) for d in np.unique(z0["day_te"]))
        for i in sorted(set().union(*sets.values())):
            ks = [m for m in members if i in sets[m]]
            sides = [bool(Z[m][f]["side"][i]) for m in ks]
            nl_ = sum(sides)
            if nl_ * 2 == len(sides):
                continue
            s_ = nl_ * 2 > len(sides)
            net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
            fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
            if fill and np.isfinite(net):
                nets.append(net - FEE_BP); days.append(int(z0["day_te"][i]))
    return np.array(nets), np.array(days), sorted(days_all)


def metrics_at(nets, days, days_sorted, F):
    eq = np.cumprod(1.0 + F * nets * 1e-4)
    curve = np.concatenate([[1.0], eq])
    dd = float((1.0 - curve / np.maximum.accumulate(curve)).max())
    span = max(len(days_sorted), 1)
    roi_m = float(eq[-1] ** (30.0 / span) - 1.0)
    by_day = {}
    for n_, d_ in zip(nets, days):
        by_day.setdefault(d_, []).append(n_)
    dret = np.array([float(np.prod([1.0 + F * x * 1e-4 for x in by_day.get(int(d), [])]) - 1.0)
                     for d in days_sorted])
    sh = float(dret.mean() / dret.std() * np.sqrt(365)) if dret.std() > 0 else 0.0
    dn = dret[dret < 0]
    so = float(dret.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365)) if len(dn) else float("inf")
    roi_a = float((1 + roi_m) ** 12 - 1)
    cal = roi_a / dd if dd > 0 else float("inf")
    return dict(dd=dd, roi_m=roi_m, roi_a=roi_a, sharpe=sh, sortino=so, calmar=cal,
                worst_day=float(dret.min()))


out = {}
print(f"target maxDD = {100*TARGET_DD:.0f}% (fee 4bp, bisection on F)", flush=True)
for name, members, tgt in FORMS:
    nets, days, days_sorted = union_nets(members, tgt)
    lo, hi = 0.1, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if metrics_at(nets, days, days_sorted, mid)["dd"] < TARGET_DD:
            lo = mid
        else:
            hi = mid
    F = 0.5 * (lo + hi)
    m = metrics_at(nets, days, days_sorted, F)
    out[name] = dict(F=F, n=int(len(nets)), **m)
    print(f"{name:26s} F*={F:5.2f} n={len(nets):4d} | ROI/mo {100*m['roi_m']:+7.1f}% ann {100*m['roi_a']:+9.0f}% | "
          f"DD {100*m['dd']:.1f}% worst-d {100*m['worst_day']:+.1f}% | Sh {m['sharpe']:.1f} So {m['sortino']:.1f} "
          f"Ca {m['calmar']:.0f}", flush=True)
bk.blob(f"research_runs/HBV1_LEVNORM_DD{int(100*TARGET_DD)}.json").upload_from_string(json.dumps(out, default=float))
print("[saved]", flush=True)
