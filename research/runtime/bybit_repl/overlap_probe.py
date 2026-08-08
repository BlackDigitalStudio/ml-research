#!/usr/bin/env python3
"""HBV1 rev24 probe: SIMULTANEOUS-POSITION / PEAK-EXPOSURE measurement for a
form (or weighted form list). The battery compounds sequentially (fair only at
near-zero overlap); consensus trades cluster in bursts — this measures the
actual concurrency from decision timestamps.

Position active window per trade: [ts, ts + ENTRY_S + HOLD_S (+ CHASE_S for
the pessimistic bound)]. Reports concurrency distribution (weighted by form
weight x F), peak gross exposure, and time-in-market.
Env: SYM, FORMS "name~sub:seed,...~T~K~weight;..." , F ("1"), HOLD_S (150),
ENTRY_S (60), CHASE_S (300).
"""
import io
import json
import os

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = os.environ.get("SYM", "DOGE")
KDAYS = 30
F = float(os.environ.get("F", "1"))
HOLD_S = float(os.environ.get("HOLD_S", "150"))
ENTRY_S = float(os.environ.get("ENTRY_S", "60"))
CHASE_S = float(os.environ.get("CHASE_S", "300"))
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


# dataset ts/day for PERFOLD-index -> timestamp mapping (trainer fold definition:
# W,T,EMB = 200,30,2; test mask = (day>=ts0)&(day<te), arrays in dataset order)
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

events = []  # (ts, weight)
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
                events.append((float(ts_te[i]), w))
                cnt += 1
    print(f"form {name}: {cnt} trades", flush=True)

events.sort()
ts_arr = np.array([e[0] for e in events]); w_arr = np.array([e[1] for e in events])
print(f"\ntotal trades {len(events)}", flush=True)

for span, tag in ((ENTRY_S + HOLD_S, "typical (entry+hold)"),
                  (ENTRY_S + HOLD_S + CHASE_S, "pessimistic (+exit chase)")):
    # sweep events: concurrency at each entry = number of active earlier trades + 1
    conc_n, conc_w = [], []
    for i in range(len(ts_arr)):
        active = (ts_arr[i] - ts_arr[:i + 1]) < span
        # only same-window trades count; use searchsorted for speed
        j0 = np.searchsorted(ts_arr, ts_arr[i] - span, side="right")
        nn = i - j0 + 1
        conc_n.append(nn)
        conc_w.append(float(w_arr[j0:i + 1].sum()))
    conc_n = np.array(conc_n); conc_w = np.array(conc_w) * F
    # time in market (union of intervals)
    covered = 0.0
    cur_s, cur_e = ts_arr[0], ts_arr[0] + span
    for t in ts_arr[1:]:
        if t <= cur_e:
            cur_e = max(cur_e, t + span)
        else:
            covered += cur_e - cur_s; cur_s, cur_e = t, t + span
    covered += cur_e - cur_s
    total_span = ts_arr[-1] - ts_arr[0]
    print(f"\n[{tag}, window {span:.0f}s] concurrency N: mean {conc_n.mean():.2f} p50 {np.percentile(conc_n,50):.0f} "
          f"p95 {np.percentile(conc_n,95):.0f} p99 {np.percentile(conc_n,99):.0f} max {conc_n.max()}", flush=True)
    print(f"  gross exposure (xF={F:g}): mean-at-entry {conc_w.mean():.2f} p95 {np.percentile(conc_w,95):.2f} "
          f"p99 {np.percentile(conc_w,99):.2f} max {conc_w.max():.2f}", flush=True)
    print(f"  time-in-market {100*covered/max(total_span,1):.1f}% of test span", flush=True)
