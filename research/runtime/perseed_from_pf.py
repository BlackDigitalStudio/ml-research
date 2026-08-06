#!/usr/bin/env python3
"""HD3 rev8: deterministic recompute of the per-seed OPTUNA_IC SEED json from the
seed-tagged PERFOLD artifacts — removes the shared-OPTUNA-json write race and thereby
the seeds-sequential constraint. Math is IDENTICAL to subs60_xgb_optuna_ic.py's
causal_rolling + metrics (KDAYS=30, t5); inputs (axb_tr/axb_te/day/side/fills/nets)
are exactly what the frozen script saved per fold. Validated bit-equal against the
sequential-era BTC/ETH/LTC SEED jsons before use.
Usage: perseed_from_pf.py SYM SEED"""
import io
import json
import sys
import numpy as np
from google.cloud import storage

SYM, SEED = sys.argv[1], int(sys.argv[2])
SUB = "research_runs/maker_labels_tb3s_h150anch"
KDAYS = 30
TGT = 5.0
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")

names = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{SUB}/PERFOLD_S{SEED}_{SYM}_qm0_f")
               if b.name.endswith(".npz"))
assert names, f"no PERFOLD_S{SEED}_{SYM} artifacts"
folds = [np.load(io.BytesIO(bk.blob(n).download_as_bytes())) for n in names]
print(f"{SYM} seed{SEED}: {len(folds)} folds", flush=True)


def causal_rolling(sc_tr, sc_te, day_tr, day_te, target_tpd, sideB, fl, fs, nl, ns):
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - target_tpd / max(wpd, 1.0))
    tr_days = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, tr_days[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    side = sideB[sel]; net = np.where(side, nl[sel], ns[sel]); fc = np.where(side, fl[sel], fs[sel])
    ex = fc & np.isfinite(net); return net[ex]


tot_days = sum(len(set(z["day_te"].tolist())) for z in folds)
pf = [causal_rolling(z["axb_tr"].astype(np.float64), z["axb_te"].astype(np.float64),
                     z["day_tr"], z["day_te"], TGT, z["side"], z["fl"], z["fs"],
                     z["netl"].astype(np.float64), z["nets"].astype(np.float64)) for z in folds]
a = np.concatenate(pf) if pf else np.array([])
n = len(a)
if not n:
    x = dict(tpd=0, ev=float("nan"), ann=float("nan"), hit=float("nan"), tot=float("nan"), perfold=[])
else:
    ev = float(a.mean()); std = float(a.std()); tpd = n / max(tot_days, 1)
    sh = ev / std if std > 0 else 0.0
    x = dict(tpd=tpd, ev=ev, ann=sh * np.sqrt(tpd * 365.0), hit=float((a > 0).mean()),
             tot=ev * tpd, perfold=[round(float(p.sum() * 0.01), 1) for p in pf])
print(f"  AxB t5: EV {x['ev']:+.2f}bp tpd {x['tpd']:.1f} ann {x['ann']:+.2f} "
      f"hit {100*x['hit']:.1f}% perfold {x['perfold']}", flush=True)
out = {"AxB_t5": x, "recomputed_from_perfold": True}
bk.blob(f"{SUB}/OPTUNA_IC_{SYM}_qm0_SEED{SEED}.json").upload_from_string(json.dumps(out, default=float))
print(f"[saved] OPTUNA_IC_{SYM}_qm0_SEED{SEED}.json", flush=True)
