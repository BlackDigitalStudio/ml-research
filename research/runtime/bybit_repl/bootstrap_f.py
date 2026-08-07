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
FEE_BP = float(os.environ.get("FEE_BP", "4"))
TARGET_DD = float(os.environ.get("TARGET_DD", "0.25"))
L = 7
REPS = 1000
B = "maker_labels_tb3s_h150anch"

V2N4 = [(B + "_v2_nooi", s) for s in range(4)]
FB4 = [(B + f"_v2_nooi_fb{j}", j) for j in range(4)]
RF4 = [(B + f"_v2_nooi_rf{j}", j) for j in range(4)]
# (name, members, T_s, K) — K=1 = union, K>=2 = consensus K-of-N (rev14 form).
# Fee-honest gate-PASS set of rev14-16 + the rev7 champion as reference.
FORMS = [
    ("v1-nooi8  union T0.3125", [(B + "_v1_nooi", s) for s in range(8)], 0.3125, 1),
    ("rf4       union T0.625", RF4, 0.625, 1),
    ("rf4       union T0.3125", RF4, 0.3125, 1),
    ("rf4       cons T5 k>=3", RF4, 5.0, 3),
    ("rf4       cons T10 k>=4", RF4, 10.0, 4),
    ("rf-mix8   union T0.3125", V2N4 + RF4, 0.3125, 1),
    ("rf-mix8   cons T1.25 k>=3", V2N4 + RF4, 1.25, 3),
    ("fb4       cons T2.5 k>=3", FB4, 2.5, 3),
    ("fb4       cons T1.25 k>=2", FB4, 1.25, 2),
    ("fbag-mix8 cons T2.5 k>=7", V2N4 + FB4, 2.5, 7),
    ("v2-nooi8  cons T1.25 k>=3", [(B + "_v2_nooi", s) for s in range(8)], 1.25, 3),
]
OUT_TAG = os.environ.get("OUT_TAG", "")

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


def union_nets(members, tgt, K=1):
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
            if len(ks) < K:
                continue
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
    paths, paths_days = [], []
    for _ in range(REPS):
        picked = []
        while len(picked) < span:
            i0 = rng.integers(0, max(span - L, 1))
            picked.extend(days_sorted[i0:i0 + L])
        dlists = [by_day.get(int(d), []) for d in picked[:span]]
        paths.append(np.array([x for dl in dlists for x in dl]))
        paths_days.append(dlists)
    return paths, paths_days, span


def sh_so(dlists, F):
    """daily-return Sharpe & Sortino (annualized) at capital fraction F."""
    dret = np.array([np.prod([1.0 + F * x * 1e-4 for x in dl]) - 1.0 for dl in dlists])
    if not len(dret) or dret.std() == 0:
        return 0.0, 0.0
    sh = float(dret.mean() / dret.std() * np.sqrt(365.0))
    dn = dret[dret < 0]
    so = float(dret.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else float("inf")
    return sh, so


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
for name, members, tgt, K in FORMS:
    nets, days, days_sorted = union_nets(members, tgt, K)
    paths, paths_days, span = boot_paths(nets, days, days_sorted)
    F10 = find_F(paths, 0.10)
    F5 = find_F(paths, 0.05)
    rois = np.array([(np.prod(1.0 + F10 * p * 1e-4) ** (30.0 / span) - 1.0) if len(p) else 0.0 for p in paths])
    shso = np.array([sh_so(dl, F10) for dl in paths_days])
    b_sh, b_so = shso[:, 0], np.minimum(shso[:, 1], 99.0)
    dd_real = dd_of(nets, F10)
    eq = np.cumprod(1.0 + F10 * nets * 1e-4)
    roi_real = float(eq[-1] ** (30.0 / span) - 1.0)
    by_day = {}
    for n_, d_ in zip(nets, days):
        by_day.setdefault(int(d_), []).append(n_)
    sh_real, so_real = sh_so([by_day.get(int(d), []) for d in days_sorted], F10)
    out[name] = dict(F10=F10, F5=F5, n=int(len(nets)), tpd=len(nets) / max(span, 1),
                     roi_real=roi_real, dd_real=dd_real, sharpe_real=sh_real, sortino_real=so_real,
                     roi_p10=float(np.quantile(rois, .1)), roi_p50=float(np.quantile(rois, .5)),
                     roi_p90=float(np.quantile(rois, .9)), Ppos=float(100 * np.mean(rois > 0)),
                     sh_p10=float(np.quantile(b_sh, .1)), sh_p50=float(np.quantile(b_sh, .5)),
                     sh_p90=float(np.quantile(b_sh, .9)),
                     so_p10=float(np.quantile(b_so, .1)), so_p50=float(np.quantile(b_so, .5)),
                     so_p90=float(np.quantile(b_so, .9)))
    print(f"{name:28s} F10={F10:5.2f} F5={F5:5.2f} | ROI/mo real {100*roi_real:+7.1f}% "
          f"boot p10/p50/p90 {100*np.quantile(rois,.1):+6.1f}/{100*np.quantile(rois,.5):+6.1f}/"
          f"{100*np.quantile(rois,.9):+6.1f}% P(>0)={100*np.mean(rois>0):3.0f}% | DD@F10 {100*dd_real:.1f}% | "
          f"Sh {sh_real:.2f} [{np.quantile(b_sh,.1):.2f},{np.quantile(b_sh,.5):.2f},{np.quantile(b_sh,.9):.2f}] "
          f"So {so_real:.2f} [{np.quantile(b_so,.1):.2f},{np.quantile(b_so,.5):.2f},{np.quantile(b_so,.9):.2f}]", flush=True)
bk.blob(f"research_runs/HBV1_BOOTF_DD{int(100*TARGET_DD)}_fee{int(FEE_BP)}{OUT_TAG}.json").upload_from_string(json.dumps(out, default=float))
print("[saved]", flush=True)
