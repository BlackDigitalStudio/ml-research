#!/usr/bin/env python3
"""f2 investigation, step 1: study the per-fold TRAIN (and TEST) DATA. For each walk-forward
fold characterize both its train window and its test window: calendar dates, n, volatility
(p95/median |rH| at 15/30/60s), drift (mean rH30 = trend), up-fraction (directional bias),
non-flat rate, and the better-side oracle payoff (gross, zero fee). Highlights whether f2's
TRAIN is anomalous or its dominance is purely a TEST-regime (vol spike) effect. Checks the
train->test embargo gap (no leakage). Reads research_runs/maker_labels_h/DOGE.npz.
"""
import io, json
import numpy as np
from google.cloud import storage
from datetime import datetime, timezone

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
W, T, EMB = 200, 30, 2; NF = 0.05
bk = storage.Client(project=PROJ).bucket(BUCKET)

d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_h/DOGE.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
ts = d["ts"].astype(np.int64); day = d["day"]
rH = {15: d["rH15"].astype(np.float64), 30: d["rH30"].astype(np.float64), 60: d["rH60"].astype(np.float64)}
PL = d["pnl_long"].astype(np.float64); PS = d["pnl_short"].astype(np.float64)
fl = d["fill_long"].astype(bool)[0]; fs = d["fill_short"].astype(bool)[0]
netl30 = PL[1, 0, :] * 100.0; nets30 = PS[1, 0, :] * 100.0   # 30s hold gross (zero fee)


def ds(x): return datetime.fromtimestamp(x / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


def stats(mask):
    r = rH[30][mask]; cal = ts[mask]
    p95_30 = float(np.quantile(np.abs(r), 0.95))
    keep = mask & (fl | fs)
    best = np.where(fl[keep] & fs[keep], np.maximum(netl30[keep], nets30[keep]),
                    np.where(fl[keep], netl30[keep], np.where(fs[keep], nets30[keep], np.nan)))
    nf = mask & (np.abs(rH[30]) >= p95_30)
    return dict(d0=ds(cal.min()), d1=ds(cal.max()), nd=len(set(day[mask].tolist())), n=int(mask.sum()),
                v15=float(np.quantile(np.abs(rH[15][mask]), 0.95)), v30=p95_30,
                v60=float(np.quantile(np.abs(rH[60][mask]), 0.95)),
                med=float(np.median(np.abs(r))), drift=float(r.mean()), upf=float((r > 0).mean()),
                nf_rate=float(nf.sum() / max(mask.sum(), 1)), oracle=float(np.nanmean(best)) if keep.any() else float("nan"))


print(f"{'fold':>4} {'win':>5} {'calendar':>23} {'nd':>3} {'v15':>5} {'v30':>5} {'v60':>5} "
      f"{'med':>5} {'drift':>6} {'up%':>5} {'nf%':>5} {'oracle':>7}")
ts0 = W + EMB; fi = 0
while ts0 < ndays:
    te = min(ts0 + T, ndays)
    trn = (day >= ts0 - EMB - W) & (day < ts0 - EMB); tst = (day >= ts0) & (day < te)
    if tst.sum() < 50 or trn.sum() < 5000:
        ts0 += T; continue
    for lbl, mask in (("TRAIN", trn), ("TEST", tst)):
        s = stats(mask)
        print(f"{fi:>4} {lbl:>5} {s['d0'] + '..' + s['d1']:>23} {s['nd']:>3} {s['v15']:>5.1f} {s['v30']:>5.1f} "
              f"{s['v60']:>5.1f} {s['med']:>5.2f} {s['drift']:>+6.2f} {100*s['upf']:>4.1f} {100*s['nf_rate']:>4.1f} "
              f"{s['oracle']:>+6.2f}", flush=True)
    # embargo gap check
    tr_max = ts[trn].max(); te_min = ts[tst].min()
    print(f"       embargo gap train->test = {(te_min - tr_max) / 1e9 / 86400:.1f} days", flush=True)
    ts0 += T; fi += 1
print("\nv = p95|rH| bp at 15/30/60s; drift = mean rH30 bp/window (trend); up% = P(rH30>0); "
      "nf% = top-5% vol rate; oracle = best-side 30s gross EV bp on filled (zero fee).")
