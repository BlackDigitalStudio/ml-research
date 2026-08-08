#!/usr/bin/env python3
"""HBV1 rev24d (non-frozen analysis): CONSENSUS-SIZED BURST BUDGET.

rev24c measured that honest (overlap-capped) execution degenerates to ~one
position per burst and that deeper-consensus forms lose less. This script
implements the continuous version (user-proposed 2026-08-08): the burst
position's budget is allocated PROPORTIONALLY to consensus depth — by member
count k, or refined by the summed score-depth of the agreeing members.

Execution = ratchet top-up: one growing position per burst window. At each
signal, target gross = F * g(signal); if target exceeds current gross, add a
slice (target - gross) at that signal's exec net; slices expire SPAN_S after
their own entry; no partial exits mid-burst. g by mode:
  flat          g = 1                     (== rev24b/c burst1, consistency check)
  propk_g{y}    g = (k/N)^y               k = number of selecting members
  propscore_g{y} g = (S/N)^y              S = sum over selecting members of
                 (pct - q)/(1 - q), pct = member's causal percentile of the
                 signal score in its own rolling threshold buffer (same buffer
                 and timing as the frozen causal_rolling tau; depth ONLY
                 re-weights size, selection is unchanged).
F10 re-bisected per mode (day-block boot L=7 x1000, P(maxDD>25%)<=10%);
day return = arithmetic within day, compounded daily.
Env: SYM, FORMS "name~sub:seed,...~T~K~1;...", FEE_BP (4), SPAN_S (210),
GAMMAS ("1,2"), OUT_TAG.
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
GAMMAS = [float(x) for x in os.environ.get("GAMMAS", "1,2").split(",")]
OUT_TAG = os.environ.get("OUT_TAG", "sized")
TARGET_DD, L, REPS = 0.25, 7, 1000
B = "maker_labels_tb3s_h150anch"

FORMS = []
for spec in os.environ["FORMS"].split(";"):
    name, members, t, k, w = spec.split("~")
    FORMS.append((name, [(("research_runs/" + p.split(":")[0]), int(p.split(":")[1]))
                         for p in members.split(",")], float(t), int(k)))

_cache = {}


def load(sub, seed, f):
    key = (sub, seed, f)
    if key not in _cache:
        _cache[key] = np.load(io.BytesIO(bk.blob(f"{sub}/PERFOLD_S{seed}_{SYM}_qm0_f{f}.npz")
                                         .download_as_bytes()))
    return _cache[key]


def causal_sel_pct(z, tgt):
    """Frozen causal_rolling selection + per-signal percentile depth.
    Returns dict {test_idx: pctnorm in [0,1]} for selected indices only."""
    sc_tr = z["axb_tr"].astype(np.float64); sc_te = z["axb_te"].astype(np.float64)
    day_tr = z["day_tr"]; day_te = z["day_te"]
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1)
    out = {}
    for d in days:
        idx = np.where(day_te == d)[0]
        if buf:
            sbuf = np.sort(buf)
            tau = float(np.quantile(buf, q))
        else:
            sbuf = np.array([0.0]); tau = 0.0
        for i in idx:
            if sc_te[i] >= tau:
                pct = np.searchsorted(sbuf, sc_te[i], side="right") / len(sbuf)
                out[int(i)] = float(np.clip((pct - q) / max(1.0 - q, 1e-9), 0.0, 1.0))
        buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return out


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

days_sorted = sorted({int(dd) for m in fold_masks for dd in np.unique(day_all[m])})
day_idx = {dd: i for i, dd in enumerate(days_sorted)}
span = len(days_sorted)

rng = np.random.default_rng(1)
paths = []
for _ in range(REPS):
    picked = []
    while len(picked) < span:
        i0 = rng.integers(0, max(span - L, 1))
        picked.extend(range(i0, min(i0 + L, span)))
    paths.append(np.array(picked[:span]))


def dd_of(r):
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    return float((1.0 - curve / np.maximum.accumulate(curve)).max())


def build_events(members, tgt, K):
    """Returns sorted list of (ts, day, net_after_fee, k, S)."""
    nf = sum(1 for b in bk.client.list_blobs(
        bk, prefix=f"{members[0][0]}/PERFOLD_S{members[0][1]}_{SYM}_qm0_f") if b.name.endswith(".npz"))
    ev = []
    for f in range(nf):
        Z = {m: load(m[0], m[1], f) for m in members}
        pcts = {m: causal_sel_pct(Z[m], tgt) for m in members}
        z0 = Z[members[0]]
        ts_te = ts_all[fold_masks[f]]
        assert len(ts_te) == len(z0["axb_te"]), f"fold {f}: mask {len(ts_te)} != perfold {len(z0['axb_te'])}"
        allsel = sorted(set().union(*[set(p) for p in pcts.values()]))
        for i in allsel:
            ks = [m for m in members if i in pcts[m]]
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
                S = float(sum(pcts[m][i] for m in ks))
                ev.append((float(ts_te[i]), int(z0["day_te"][i]), net - FEE_BP, len(ks), S))
    ev.sort()
    return ev


def sim(events, N, F, mode, gamma):
    active = []  # (end_ts, size)
    dpnl = np.zeros(span)
    max_g, used, n_slices = 0.0, 0.0, 0
    for t, dd, net, k, S in events:
        active = [(e, sz) for e, sz in active if e > t]
        g_cur = sum(sz for _, sz in active)
        if mode == "flat":
            g = 1.0
        elif mode == "propk":
            g = (k / N) ** gamma
        else:
            g = (min(S, N) / N) ** gamma
        target = F * g
        if target > g_cur:
            sl = target - g_cur
            active.append((t + SPAN_S, sl))
            dpnl[day_idx[dd]] += sl * net * 1e-4
            used += sl; n_slices += 1
            max_g = max(max_g, target)
    return dpnl, max_g, used, n_slices


def f10_metrics(events, N, mode, gamma):
    def p_exceed(F):
        dret = sim(events, N, F, mode, gamma)[0]
        return float(np.mean([dd_of(dret[p]) > TARGET_DD for p in paths]))
    lo, hi = 0.02, 200.0
    capped_hi = p_exceed(hi) <= 0.10
    if not capped_hi:
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if p_exceed(mid) <= 0.10:
                lo = mid
            else:
                hi = mid
    F10 = hi if capped_hi else 0.5 * (lo + hi)
    dret, max_g, used, n_slices = sim(events, N, F10, mode, gamma)
    eq = np.cumprod(1.0 + dret)
    roi = float(eq[-1] ** (30.0 / span) - 1.0)
    rois = [float(np.cumprod(1.0 + dret[p])[-1] ** (30.0 / span) - 1.0) for p in paths]
    sh = float(dret.mean() / dret.std() * np.sqrt(365.0)) if dret.std() > 0 else 0.0
    return dict(F10=F10, f10_capped=capped_hi, roi25_monthly=roi,
                roi25_p10=float(np.quantile(rois, .1)), roi25_p50=float(np.quantile(rois, .5)),
                Ppos=float(100 * np.mean(np.array(rois) > 0)), sharpe=sh,
                max_gross=max_g, used_notional=used, n_slices=n_slices, dd_real=dd_of(dret))


allout = {}
for name, members, tgt, K in FORMS:
    try:
        events = build_events(members, tgt, K)
        if not events:
            print(f"form {name}: 0 trades, skipped", flush=True)
            continue
        N = len(members)
        res = {"_n_trades": len(events)}
        modes = [("flat", 0.0)] + [("propk", g) for g in GAMMAS] + [("propscore", g) for g in GAMMAS]
        for mode, gamma in modes:
            tag = mode if mode == "flat" else f"{mode}_g{gamma:g}"
            res[tag] = f10_metrics(events, N, mode, gamma)
            m = res[tag]
            print(f"form {name} {tag:14s}: F10 {m['F10']:6.2f}{'^' if m['f10_capped'] else ' '} "
                  f"ROI25 {100*m['roi25_monthly']:+7.1f}% p10 {100*m['roi25_p10']:+6.1f}% "
                  f"Sh {m['sharpe']:5.2f} max-gross {m['max_gross']:6.2f} slices {m['n_slices']} "
                  f"DDreal {100*m['dd_real']:4.1f}%", flush=True)
        allout[name] = res
    except Exception as ex:
        print(f"form {name}: FAILED ({type(ex).__name__}: {ex})", flush=True)

bk.blob(f"research_runs/HBV1_SIZED_BURST_{OUT_TAG}_{SYM}.json").upload_from_string(
    json.dumps(allout, default=float))
print(f"\n[saved HBV1_SIZED_BURST_{OUT_TAG}]", flush=True)
