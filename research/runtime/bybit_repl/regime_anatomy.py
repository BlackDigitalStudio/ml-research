#!/usr/bin/env python3
"""HBV1 rev18C (non-frozen analysis): DOGE day-level regime anatomy of the new
forms (rev22-BNB-style). Buckets day EV of selected forms by: day-vol quintile
(std of rH30 within the day), funding sign/|magnitude| tercile (bybit_aux
settled rates), day-of-week, calendar month. Day-level only — the 31%-gain ToD
family cannot be tested at day granularity; flagged, not claimed.
Env: FEE_BP (4)."""
import io
import json
import os

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = "DOGE"
KDAYS = 30
FEE_BP = float(os.environ.get("FEE_BP", "4"))
B = "maker_labels_tb3s_h150anch"
V2N4 = [(B + "_v2_nooi", s) for s in range(4)]
FB4 = [(B + f"_v2_nooi_fb{j}", j) for j in range(4)]
RF4 = [(B + f"_v2_nooi_rf{j}", j) for j in range(4)]
FORMS = [
    ("champ_U03125", [(B + "_v1_nooi", s) for s in range(8)], 0.3125, 1),
    ("rf4_U0625", RF4, 0.625, 1),
    ("fbagmix8_cons_T25k7", V2N4 + FB4, 2.5, 7),
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
    for f in range(nf):
        Z = {m: load(m[0], m[1], f) for m in members}
        sets = {m: causal_sel(Z[m], tgt) for m in members}
        z0 = Z[members[0]]
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
    return by_day


# ---- day-level conditioning variables from the dataset (ts, day, rH30)
print("loading dataset cols (ts, day, rH30)...", flush=True)
raw = bk.blob(f"research_runs/{B}/DOGE.npz").download_as_bytes()
d = np.load(io.BytesIO(raw), allow_pickle=True)
ts = d["ts"].astype(np.float64) / 1e9  # dataset ts is ns -> seconds
day = d["day"].astype(int); rH = d["rH30"].astype(np.float64)
del raw
ndays = int(day.max()) + 1
day_vol = np.full(ndays, np.nan)
day_date = {}
for dd in range(ndays):
    m = day == dd
    if m.sum() > 1:
        day_vol[dd] = np.nanstd(rH[m])
        t0 = float(np.nanmin(ts[m]))
        day_date[dd] = np.datetime64(int(t0), "s").astype("datetime64[D]")
# funding per calendar date (funding_rates.json = [[ms_ts, rate], ...])
rows = json.loads(bk.blob("bybit_aux/funding_rates.json").download_as_bytes())
fund_by_date = {}
for t, v in rows:
    dte = np.datetime64(int(t) // 1000, "s").astype("datetime64[D]")
    fund_by_date.setdefault(dte, []).append(float(v))
fund_day = {dd: float(np.sum(fund_by_date.get(dt, [0.0]))) for dd, dt in day_date.items()}

out = {}
for name, members, tgt, K in FORMS:
    by_day = form_day_nets(members, tgt, K)
    days = sorted(by_day)
    ev_day = {dd: float(np.mean(by_day[dd])) for dd in days}
    sum_day = {dd: float(np.sum(by_day[dd])) for dd in days}
    res = {}

    def bucket(tag, key_fn, dds):
        groups = {}
        for dd in dds:
            groups.setdefault(key_fn(dd), []).append(dd)
        rows = {}
        for g in sorted(groups, key=str):
            ds_ = groups[g]
            tr = [x for dd in ds_ for x in by_day[dd]]
            rows[str(g)] = dict(days=len(ds_), n=len(tr), ev=float(np.mean(tr)) if tr else None,
                                sum_bp=float(np.sum(tr)) if tr else 0.0)
        res[tag] = rows
        line = " | ".join(f"{g}: n={v['n']} EV {v['ev']:+.1f}" if v["ev"] is not None else f"{g}: n=0"
                          for g, v in rows.items())
        print(f"  {tag}: {line}", flush=True)

    print(f"\n### {name} ({len(days)} active days)", flush=True)
    vq = np.nanquantile([day_vol[dd] for dd in days], [0.2, 0.4, 0.6, 0.8])
    bucket("vol_quintile", lambda dd: int(np.searchsorted(vq, day_vol[dd])), days)
    fq = np.nanquantile([abs(fund_day.get(dd, 0.0)) for dd in days], [1 / 3, 2 / 3])
    bucket("fund_mag_tercile", lambda dd: int(np.searchsorted(fq, abs(fund_day.get(dd, 0.0)))), days)
    bucket("fund_sign", lambda dd: "pos" if fund_day.get(dd, 0.0) >= 0 else "neg", days)
    bucket("dow", lambda dd: int((day_date[dd].astype(int) + 3) % 7), days)  # 0=Mon (epoch day 0 = Thu)
    bucket("month", lambda dd: str(day_date[dd])[:7], days)
    out[name] = res

bk.blob(f"research_runs/HBV1_ANATOMY_{SYM}.json").upload_from_string(json.dumps(out, default=float))
print("\n[saved HBV1_ANATOMY]", flush=True)
