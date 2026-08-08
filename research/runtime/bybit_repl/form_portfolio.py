#!/usr/bin/env python3
"""HBV1 rev18B (non-frozen analysis): PORTFOLIO OF FORMS.

Builds the daily-return series (F=1, fee-honest) of the rev17 representative
gate-PASS forms on the common test days, reports their pairwise daily-return
correlations, then sizes equal-weight and inverse-vol portfolios to the DD-25
budget: in-sample F* (bisection) + robust F10 (day-block BOOT L=7 x1000,
P(maxDD>25%)<=10%) with ROI p10/p50/p90 and bootstrap Sharpe/Sortino.
Env: FEE_BP (4), TARGET_DD (0.25).
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
TARGET_DD = float(os.environ.get("TARGET_DD", "0.25"))
L, REPS = 7, 1000
B = "maker_labels_tb3s_h150anch"
V2N4 = [(B + "_v2_nooi", s) for s in range(4)]
FB4 = [(B + f"_v2_nooi_fb{j}", j) for j in range(4)]
RF4 = [(B + f"_v2_nooi_rf{j}", j) for j in range(4)]

FORMS = [
    ("champ", [(B + "_v1_nooi", s) for s in range(8)], 0.3125, 1),
    ("rf4_U0625", RF4, 0.625, 1),
    ("rf4_cons_T5k3", RF4, 5.0, 3),
    ("rfmix8_U03125", V2N4 + RF4, 0.3125, 1),
    ("fb4_cons_T25k3", FB4, 2.5, 3),
    ("fbagmix8_cons_T25k7", V2N4 + FB4, 2.5, 7),
    ("v2nooi8_cons_T125k3", [(B + "_v2_nooi", s) for s in range(8)], 1.25, 3),
]

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


def form_day_nets(members, tgt, K):
    nf = sum(1 for b in bk.client.list_blobs(
        bk, prefix=f"research_runs/{members[0][0]}/PERFOLD_S{members[0][1]}_{SYM}_qm0_f")
        if b.name.endswith(".npz"))
    by_day = {}
    days_all = set()
    for f in range(nf):
        Z = {m: load(m[0], m[1], f) for m in members}
        sets = {m: causal_sel(Z[m], tgt) for m in members}
        z0 = Z[members[0]]
        days_all |= set(int(d) for d in np.unique(z0["day_te"]))
        for i in sorted(set().union(*sets.values())):
            ks = [m for m in members if i in sets[m]]
            if len(ks) < K:
                continue
            sides = [bool(Z[m]["side"][i]) for m in ks]
            nl = sum(sides)
            if nl * 2 == len(sides):
                continue
            s_ = nl * 2 > len(sides)
            net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
            fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
            if fill and np.isfinite(net):
                by_day.setdefault(int(z0["day_te"][i]), []).append(net - FEE_BP)
    return by_day, sorted(days_all)


def dd_of(dret, F):
    eq = np.cumprod(1.0 + F * dret)
    curve = np.concatenate([[1.0], eq])
    return float((1.0 - curve / np.maximum.accumulate(curve)).max())


def metrics(dret, F, span):
    eq = np.cumprod(1.0 + F * dret)
    roi_m = float(eq[-1] ** (30.0 / span) - 1.0)
    sh = float(dret.mean() / dret.std() * np.sqrt(365.0)) if dret.std() > 0 else 0.0
    dn = dret[dret < 0]
    so = float(dret.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else float("inf")
    return roi_m, sh, so, dd_of(dret, F), float((F * dret).min())


# 1. per-form daily return series at F=1
series = {}
days_common = None
for name, members, tgt, K in FORMS:
    by_day, days_sorted = form_day_nets(members, tgt, K)
    days_common = days_sorted if days_common is None else days_common
    dret = np.array([float(np.prod([1.0 + x * 1e-4 for x in by_day.get(int(d), [])]) - 1.0)
                     for d in days_common])
    series[name] = dret
    roi_m, sh, so, dd, wd = metrics(dret, 1.0, len(days_common))
    print(f"{name:22s} ROI/mo {100*roi_m:+6.1f}% Sh {sh:4.2f} So {min(so,99):5.2f} DD {100*dd:4.1f}% "
          f"active days {int((dret!=0).sum())}/{len(days_common)}", flush=True)

names = list(series)
M = np.array([series[n] for n in names])
C = np.corrcoef(M)
print("\ndaily-return correlations:", flush=True)
print("            " + " ".join(f"{n[:10]:>11s}" for n in names), flush=True)
for i, n in enumerate(names):
    print(f"{n[:11]:11s} " + " ".join(f"{C[i,j]:+11.2f}" for j in range(len(names))), flush=True)

# 2. portfolios
span = len(days_common)
vols = M.std(axis=1)
w_eq = np.ones(len(names)) / len(names)
w_iv = (1.0 / np.maximum(vols, 1e-9)); w_iv /= w_iv.sum()
out = {"forms": {n: dict(roi_m=metrics(series[n], 1.0, span)[0], sharpe=metrics(series[n], 1.0, span)[1])
                 for n in names},
       "corr": {f"{a}|{b}": float(C[i, j]) for i, a in enumerate(names) for j, b in enumerate(names) if i < j}}
rng0 = np.random.default_rng(1)
for pname, w in (("equal", w_eq), ("invvol", w_iv)):
    dret = (w[:, None] * M).sum(axis=0)
    # in-sample F* to TARGET_DD
    lo, hi = 0.1, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dd_of(dret, mid) < TARGET_DD:
            lo = mid
        else:
            hi = mid
    Fstar = 0.5 * (lo + hi)
    # bootstrap paths of day indices
    rng = np.random.default_rng(1)
    paths = []
    for _ in range(REPS):
        picked = []
        while len(picked) < span:
            i0 = rng.integers(0, max(span - L, 1))
            picked.extend(range(i0, min(i0 + L, span)))
        paths.append(np.array(picked[:span]))
    def p_exceed(F):
        return float(np.mean([dd_of(dret[p], F) > TARGET_DD for p in paths]))
    lo, hi = 0.05, 300.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if p_exceed(mid) <= 0.10:
            lo = mid
        else:
            hi = mid
    F10 = 0.5 * (lo + hi)
    rois, shs, sos = [], [], []
    for p in paths:
        dr = dret[p]
        roi_m, sh, so, _, _ = metrics(dr, F10, span)
        eqF = np.cumprod(1.0 + F10 * dr)
        rois.append(float(eqF[-1] ** (30.0 / span) - 1.0))
        drF = F10 * dr
        shs.append(float(drF.mean() / drF.std() * np.sqrt(365.0)) if drF.std() > 0 else 0.0)
        dn = drF[drF < 0]
        sos.append(min(float(drF.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else 99.0, 99.0))
    drF = F10 * dret
    roi_real = float(np.cumprod(1.0 + drF)[-1] ** (30.0 / span) - 1.0)
    sh_real = float(drF.mean() / drF.std() * np.sqrt(365.0))
    dn = drF[drF < 0]
    so_real = min(float(drF.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else 99.0, 99.0)
    out[pname] = dict(weights={n: float(x) for n, x in zip(names, w)}, Fstar=Fstar, F10=F10,
                      roi_real=roi_real, dd_real=dd_of(dret, F10), sharpe_real=sh_real, sortino_real=so_real,
                      roi_p10=float(np.quantile(rois, .1)), roi_p50=float(np.quantile(rois, .5)),
                      roi_p90=float(np.quantile(rois, .9)), Ppos=float(100 * np.mean(np.array(rois) > 0)),
                      sh_p10=float(np.quantile(shs, .1)), sh_p50=float(np.quantile(shs, .5)),
                      sh_p90=float(np.quantile(shs, .9)),
                      so_p10=float(np.quantile(sos, .1)), so_p50=float(np.quantile(sos, .5)),
                      so_p90=float(np.quantile(sos, .9)))
    print(f"\nportfolio {pname}: F*={Fstar:.2f} F10={F10:.2f} | ROI/mo real {100*roi_real:+.1f}% "
          f"boot p10/p50/p90 {100*np.quantile(rois,.1):+.1f}/{100*np.quantile(rois,.5):+.1f}/"
          f"{100*np.quantile(rois,.9):+.1f}% P(>0)={100*np.mean(np.array(rois)>0):.0f}% | "
          f"DD@F10 {100*dd_of(dret,F10):.1f}% | Sh {sh_real:.2f} [{np.quantile(shs,.1):.2f},{np.quantile(shs,.5):.2f},"
          f"{np.quantile(shs,.9):.2f}] So {so_real:.2f} [{np.quantile(sos,.1):.2f},{np.quantile(sos,.5):.2f},"
          f"{np.quantile(sos,.9):.2f}]", flush=True)

bk.blob(f"research_runs/HBV1_PORTFOLIO_{SYM}.json").upload_from_string(json.dumps(out, default=float))
print("\n[saved HBV1_PORTFOLIO]", flush=True)
