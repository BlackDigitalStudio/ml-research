#!/usr/bin/env python3
"""HBV1 rev13b: ROBUST capital sizing — day-block bootstrap version of the DD
normalization (user directive: robustness over point estimates).

For each gate-PASS form: build fee-honest nets; generate 1000 day-block bootstrap
paths (L=7, same construction as the standing BOOT battery, fixed rng); bisect F
so that P(maxDD > TARGET_DD) <= P_EXCEED across paths (DD is monotone in F per
path). Report F_robust for P_EXCEED in {0.10, 0.05}, and at F10: the bootstrap
ROI/month distribution (p10/p50/p90) + realized-path metrics.
Env: TARGET_DD (0.25), LOCAL_GCS_ROOT."""
import io
import json
import os

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = "DOGE"
KDAYS = 30
FEE_BP = 4.0
TARGET_DD = float(os.environ.get("TARGET_DD", "0.25"))
L = 7
REPS = 1000
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


def boot_paths(nets, days, days_sorted):
    by_day = {}
    for n_, d_ in zip(nets, days):
        by_day.setdefault(int(d_), []).append(n_)
    span = len(days_sorted)
    rng = np.random.default_rng(1)
    paths = []
    for _ in range(REPS):
        picked = []
        while len(picked) < span:
            i0 = rng.integers(0, max(span - L, 1))
            picked.extend(days_sorted[i0:i0 + L])
        seq = [x for d in picked[:span] for x in by_day.get(int(d), [])]
        paths.append(np.array(seq))
    return paths, span


def dd_of(path, F):
    if not len(path):
        return 0.0
    eq = np.cumprod(1.0 + F * path * 1e-4)
    curve = np.concatenate([[1.0], eq])
    return float((1.0 - curve / np.maximum.accumulate(curve)).max())


def p_exceed(paths, F):
    return float(np.mean([dd_of(p, F) > TARGET_DD for p in paths]))


def find_F(paths, p_target):
    lo, hi = 0.05, 60.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if p_exceed(paths, mid) <= p_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


out = {}
print(f"target DD {100*TARGET_DD:.0f}% | day-block BOOT L={L} x{REPS} | F for P(DD>tgt)<=10%/5%", flush=True)
for name, members, tgt in FORMS:
    nets, days, days_sorted = union_nets(members, tgt)
    paths, span = boot_paths(nets, days, days_sorted)
    F10 = find_F(paths, 0.10)
    F5 = find_F(paths, 0.05)
    rois = np.array([(np.prod(1.0 + F10 * p * 1e-4) ** (30.0 / span) - 1.0) if len(p) else 0.0 for p in paths])
    dd_real = dd_of(nets, F10)
    eq = np.cumprod(1.0 + F10 * nets * 1e-4)
    roi_real = float(eq[-1] ** (30.0 / span) - 1.0)
    out[name] = dict(F10=F10, F5=F5, roi_real=roi_real, dd_real=dd_real,
                     roi_p10=float(np.quantile(rois, .1)), roi_p50=float(np.quantile(rois, .5)),
                     roi_p90=float(np.quantile(rois, .9)), Ppos=float(100 * np.mean(rois > 0)))
    print(f"{name:26s} F10={F10:5.2f} F5={F5:5.2f} | ROI/mo real {100*roi_real:+7.1f}% "
          f"boot p10/p50/p90 {100*np.quantile(rois,.1):+6.1f}/{100*np.quantile(rois,.5):+6.1f}/"
          f"{100*np.quantile(rois,.9):+6.1f}% P(>0)={100*np.mean(rois>0):3.0f}% | DD(real path)@F10 "
          f"{100*dd_real:.1f}%", flush=True)
bk.blob(f"research_runs/HBV1_BOOTF_DD{int(100*TARGET_DD)}.json").upload_from_string(json.dumps(out, default=float))
print("[saved]", flush=True)
