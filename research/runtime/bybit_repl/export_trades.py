#!/usr/bin/env python3
"""HBV1 rev25 stage 1: export a form's reconstructed trade stream (decision ts,
side, day, k, net) to research_runs/HBV1_TRADES_{name}_{SYM}.json so the
tick-tail day stage (which runs where the raw L2 lives) can consume it without
the PERFOLD artifacts. Selection/side/net logic identical to overlap_capped.
Env: SYM, FORMS "name~sub:seed,...~T~K~1;...", FEE_BP (4).
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

for name, members, tgt, K in FORMS:
    nf = sum(1 for b in bk.client.list_blobs(
        bk, prefix=f"{members[0][0]}/PERFOLD_S{members[0][1]}_{SYM}_qm0_f") if b.name.endswith(".npz"))
    ev = []
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
                ev.append([float(ts_te[i]), 1 if s_ else -1, int(z0["day_te"][i]),
                           len(ks), round(net - FEE_BP, 2)])
    ev.sort()
    bk.blob(f"research_runs/HBV1_TRADES_{name}_{SYM}.json").upload_from_string(json.dumps(ev))
    print(f"form {name}: {len(ev)} trades exported", flush=True)
print("[export done]", flush=True)
