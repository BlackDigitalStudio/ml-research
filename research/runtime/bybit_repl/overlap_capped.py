#!/usr/bin/env python3
"""HBV1 rev24 (non-frozen analysis): EXPOSURE-CAPPED re-audit of clustered forms.

The rev21 DD-25 battery compounds trades sequentially at fraction F of equity —
implicitly assuming near-zero overlap. The rev24 overlap probe measured burst
concurrency up to 51 (rf32 cons) / 205 (7-form portfolio) simultaneous
positions, i.e. peak gross exposure ~30x / ~86x equity at the battery's F10 —
not implementable (margin/liquidation) and outside the DD model. This script
re-simulates the SAME trade streams under an explicit execution policy:

  per-trade nominal fraction f_nom = F * w_form; a new entry is downsized to
  min(f_nom, CAP - current_gross) (skipped at 0); position active for SPAN_S.
  Day return = sum(f_used * net) * 1e-4 (arithmetic within day — sequential
  product is undefined under overlap), compounded daily. F10 re-bisected per
  CAP: P(maxDD>25%) <= 10% over 1000 day-block bootstrap paths (L=7).
  CAP="burst1" means CAP=F (first trade of a burst takes the whole budget).

Reports per CAP: F10, ROI25/mo real + boot p10/p50/p90, Sharpe/Sortino,
realized-vs-nominal notional share, exposure stats at F10.
Env: SYM, FORMS "name~sub:seed,...~T~K~weight;..." (weights=portfolio invvol,
1 for single form), FEE_BP (4), SPAN_S (210), CAPS ("0.5,1,2,3,burst1"),
OUT_TAG.
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
SPAN_S = float(os.environ.get("SPAN_S", "210"))
CAPS = os.environ.get("CAPS", "0.5,1,2,3,burst1").split(",")
OUT_TAG = os.environ.get("OUT_TAG", "capped")
TARGET_DD, L, REPS = 0.25, 7, 1000
B = "maker_labels_tb3s_h150anch"

FORMS = []
for spec in os.environ["FORMS"].split(";"):
    name, members, t, k, w = spec.split("~")
    FORMS.append((name, [(("research_runs/" + p.split(":")[0]), int(p.split(":")[1]))
                         for p in members.split(",")], float(t), int(k), float(w)))

_cache = {}


def load(sub, seed, f):
    key = (sub, seed, f)
    if key not in _cache:
        _cache[key] = np.load(io.BytesIO(bk.blob(f"{sub}/PERFOLD_S{seed}_{SYM}_qm0_f{f}.npz")
                                         .download_as_bytes()))
    return _cache[key]


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


print("loading dataset ts/day...", flush=True)
d = np.load(io.BytesIO(bk.blob(f"research_runs/{B}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
ts_all = d["ts"].astype(np.float64) / 1e9
day_all = d["day"].astype(int)
del d
ndays = int(day_all.max()) + 1
W, T, EMB = 200, 30, 2
fold_masks = []
t0 = W + EMB
while t0 < ndays:
    te = min(t0 + T, ndays)
    tst = (day_all >= t0) & (day_all < te)
    if tst.sum() >= 50:
        fold_masks.append(tst)
    t0 += T

events = []  # (ts, w_nom, net_after_fee, day)
for name, members, tgt, K, w in FORMS:
    nf = sum(1 for b in bk.client.list_blobs(
        bk, prefix=f"{members[0][0]}/PERFOLD_S{members[0][1]}_{SYM}_qm0_f") if b.name.endswith(".npz"))
    cnt = 0
    for f in range(nf):
        Z = {m: load(m[0], m[1], f) for m in members}
        sets = {m: causal_sel(Z[m], tgt) for m in members}
        z0 = Z[members[0]]
        ts_te = ts_all[fold_masks[f]]
        assert len(ts_te) == len(z0["axb_te"]), f"fold {f}: mask {len(ts_te)} != perfold {len(z0['axb_te'])}"
        for i in sorted(set().union(*sets.values())):
            ks = [m for m in members if i in sets[m]]
            if len(ks) < K:
                continue
            sides = [bool(Z[m]["side"][i]) for m in ks]
            nl = sum(sides)
            if nl * 2 == len(sides):
                continue
            s_ = nl * 2 > len(sides)
            fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
            net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
            if fill and np.isfinite(net):
                events.append((float(ts_te[i]), w, net - FEE_BP, int(z0["day_te"][i])))
                cnt += 1
    print(f"form {name}: {cnt} trades", flush=True)

events.sort()
ev_ts = np.array([e[0] for e in events]); ev_w = np.array([e[1] for e in events])
ev_net = np.array([e[2] for e in events]); ev_day = np.array([e[3] for e in events])
days_sorted = sorted(set(ev_day.tolist()) | {int(dd) for m in fold_masks for dd in np.unique(day_all[m])})
day_idx = {dd: i for i, dd in enumerate(days_sorted)}
span = len(days_sorted)
print(f"\ntotal trades {len(events)}, {span} test days", flush=True)


def sim(F, cap_mode):
    """Returns (dret vector over days_sorted, used_notional_share, max_gross)."""
    C = F if cap_mode == "burst1" else float(cap_mode)
    active = []  # (end_ts, f_used)
    dpnl = np.zeros(span)
    used, nom = 0.0, 0.0
    max_g = 0.0
    for t, w, net, dd in zip(ev_ts, ev_w, ev_net, ev_day):
        active = [(e, fu) for e, fu in active if e > t]
        g = sum(fu for _, fu in active)
        f_nom = F * w
        f_used = min(f_nom, max(C - g, 0.0))
        nom += f_nom
        if f_used > 0:
            active.append((t + SPAN_S, f_used))
            dpnl[day_idx[dd]] += f_used * net * 1e-4
            used += f_used
            max_g = max(max_g, g + f_used)
    return dpnl, used / max(nom, 1e-12), max_g


def dd_of(r):
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    return float((1.0 - curve / np.maximum.accumulate(curve)).max())


rng = np.random.default_rng(1)
paths = []
for _ in range(REPS):
    picked = []
    while len(picked) < span:
        i0 = rng.integers(0, max(span - L, 1))
        picked.extend(range(i0, min(i0 + L, span)))
    paths.append(np.array(picked[:span]))

out = {}
for cap in CAPS:
    def p_exceed(F):
        dret, _, _ = sim(F, cap)
        return float(np.mean([dd_of(dret[p]) > TARGET_DD for p in paths]))
    lo, hi = 0.02, 200.0
    capped_hi = False
    if p_exceed(hi) <= 0.10:
        capped_hi = True
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if p_exceed(mid) <= 0.10:
                lo = mid
            else:
                hi = mid
    F10 = hi if capped_hi else 0.5 * (lo + hi)
    dret, used_share, max_g = sim(F10, cap)
    eq = np.cumprod(1.0 + dret)
    roi_real = float(eq[-1] ** (30.0 / span) - 1.0)
    rois = []
    shs, sos = [], []
    for p in paths:
        r = dret[p]
        rois.append(float(np.cumprod(1.0 + r)[-1] ** (30.0 / span) - 1.0))
        shs.append(float(r.mean() / r.std() * np.sqrt(365.0)) if r.std() > 0 else 0.0)
        dn = r[r < 0]
        sos.append(min(float(r.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else 99.0, 99.0))
    sh = float(dret.mean() / dret.std() * np.sqrt(365.0)) if dret.std() > 0 else 0.0
    dn = dret[dret < 0]
    so = min(float(dret.mean() / np.sqrt(np.mean(dn ** 2)) * np.sqrt(365.0)) if len(dn) else 99.0, 99.0)
    out[str(cap)] = dict(F10=F10, f10_capped=capped_hi, roi25_monthly=roi_real,
                         roi25_p10=float(np.quantile(rois, .1)), roi25_p50=float(np.quantile(rois, .5)),
                         roi25_p90=float(np.quantile(rois, .9)), Ppos=float(100 * np.mean(np.array(rois) > 0)),
                         sharpe=sh, sortino=so, sh_p10=float(np.quantile(shs, .1)),
                         used_notional_share=used_share, max_gross=max_g,
                         dd_real=dd_of(dret))
    print(f"CAP {cap:>6s}: F10 {F10:6.2f}{'^' if capped_hi else ' '} ROI25/mo {100*roi_real:+7.1f}% "
          f"p10 {100*out[str(cap)]['roi25_p10']:+6.1f}% p50 {100*out[str(cap)]['roi25_p50']:+6.1f}% "
          f"Sh {sh:5.2f} So {so:5.2f} | used-notional {100*used_share:5.1f}% max-gross {max_g:5.2f} "
          f"DDreal {100*out[str(cap)]['dd_real']:4.1f}%", flush=True)

bk.blob(f"research_runs/HBV1_OVERLAP_CAPPED_{OUT_TAG}_{SYM}.json").upload_from_string(
    json.dumps(out, default=float))
print(f"\n[saved HBV1_OVERLAP_CAPPED_{OUT_TAG}]", flush=True)
