#!/usr/bin/env python3
"""HBV1 analysis (non-frozen): per-seed t10 + ensemble t10 from the PERFOLD artifacts.
Same causal_rolling/metrics math as perseed_from_pf.py / ens_sym.py, TGT parameterized.
Usage: bybit_t10.py SYM   (XSYM_SUB overridable; TGT env, default 10)."""
import io
import json
import os
import sys

import numpy as np
from google.cloud import storage

SYM = sys.argv[1]
SUB = "research_runs/" + os.environ.get("XSYM_SUB", "maker_labels_tb3s_h150anch")
KDAYS = 30
TGT = float(os.environ.get("TGT", "10"))
SEEDS = [0, 1, 2, 3]
bk = storage.Client(project="x").bucket("market-data-0998ac51")


def causal(sc_tr, sc_te, day_tr, day_te, tgt, sideB, fl, fs, nl, ns):
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl[sel], ns[sel]); fc = np.where(side, fl[sel], fs[sel])
    ex = fc & np.isfinite(net)
    return net[ex]


def metrics(pf, tot_days):
    a = np.concatenate(pf) if pf else np.array([])
    if not len(a):
        return dict(n=0)
    ev = float(a.mean()); tpd = len(a) / max(tot_days, 1)
    return dict(n=int(len(a)), ev=ev, tpd=tpd, hit=float((a > 0).mean()),
                perfold=[round(float(p.sum() * 0.01), 1) for p in pf])


nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SUB}/PERFOLD_S0_{SYM}_qm0_f") if b.name.endswith(".npz"))
Z = {s: [np.load(io.BytesIO(bk.blob(f"{SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
         for f in range(nf)] for s in SEEDS}
tot_days = sum(len(set(Z[0][f]["day_te"].tolist())) for f in range(nf))

out = {"tgt": TGT, "nf": nf}
evs = []
for s in SEEDS:
    pf = [causal(z["axb_tr"].astype(np.float64), z["axb_te"].astype(np.float64), z["day_tr"], z["day_te"],
                 TGT, z["side"], z["fl"], z["fs"], z["netl"].astype(np.float64), z["nets"].astype(np.float64))
          for z in Z[s]]
    m = metrics(pf, tot_days)
    evs.append(m.get("ev", np.nan))
    out[f"seed{s}"] = m
    print(f"  seed{s} AxB_t{TGT:g}: EV {m.get('ev', float('nan')):+.2f}bp n={m.get('n', 0)} "
          f"tpd {m.get('tpd', 0):.1f} hit {100*m.get('hit', float('nan')):.1f}%", flush=True)
evs = np.array(evs)
print(f"=== PER-SEED t{TGT:g}: {np.nanmean(evs):+.2f} +- {np.nanstd(evs, ddof=1):.2f} ===", flush=True)

folds = []
for f in range(nf):
    zs = [Z[s][f] for s in SEEDS]
    tr = np.mean([z["axb_tr"].astype(np.float64) for z in zs], 0)
    te = np.mean([z["axb_te"].astype(np.float64) for z in zs], 0)
    votes = np.sum([z["side"].astype(int) for z in zs], 0)
    z0 = zs[0]
    folds.append((tr, te, z0["day_tr"], z0["day_te"], votes >= len(SEEDS) / 2.0,
                  z0["fl"], z0["fs"], z0["netl"].astype(np.float64), z0["nets"].astype(np.float64)))
pf = [causal(f_[0], f_[1], f_[2], f_[3], TGT, f_[4], f_[5], f_[6], f_[7], f_[8]) for f_ in folds]
m = metrics(pf, tot_days)
out["ensemble"] = m
print(f"=== ENSEMBLE t{TGT:g}: EV {m.get('ev', float('nan')):+.2f}bp n={m.get('n', 0)} "
      f"tpd {m.get('tpd', 0):.2f} hit {100*m.get('hit', float('nan')):.1f}% perfold {m.get('perfold')}", flush=True)
bk.blob(f"{SUB}/HBV1_t{int(TGT)}_{SYM}.json").upload_from_string(json.dumps(out, default=float))
print("[saved]", flush=True)
