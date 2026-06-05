#!/usr/bin/env python3
"""Characterize each walk-forward fold's TEST window as a market regime: calendar dates,
volatility (p95/median |rH60|), drift (mean rH60 = trend), non-flat rate, fill rate, and the
ORACLE maker-EV ceiling (best-side net on apred-non-flat fillable windows = available alpha).
Explains WHY one fold dominates and why the edge fades away from it."""
import io, json
import numpy as np
from google.cloud import storage
from datetime import datetime, timezone

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
W, T, EMB = 200, 30, 2
bk = storage.Client(project=PROJ).bucket(BUCKET)

d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_rr/DOGE.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"]))
rH = d["rH60"].astype(np.float64); day = d["day"]; ts = d["ts"].astype(np.int64)
fee = m["maker_rt_fee_pct"] * 100.0
netl = d["pnl_long"][:, 0, :].astype(np.float64) * 100.0 - fee   # (NC,N) qm0=touch; [0]=hold-60s
nets = d["pnl_short"][:, 0, :].astype(np.float64) * 100.0 - fee
fl = d["fill_long"][0].astype(bool); fs = d["fill_short"][0].astype(bool)
ndays = m["n_days"]


def ds(x): return datetime.fromtimestamp(x / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


print(f"{'fold':>4} {'test calendar':>25} {'d':>3} {'volp95':>7} {'volmed':>7} {'drift':>7} "
      f"{'nf%':>5} {'fill%':>6} {'oracle':>7} {'thr':>6}")
ts0 = W + EMB; fi = 0
while ts0 < ndays:
    te = min(ts0 + T, ndays)
    trn = (day >= ts0 - EMB - W) & (day < ts0 - EMB); tst = (day >= ts0) & (day < te)
    if tst.sum() < 50 or trn.sum() < 5000:
        ts0 += T; continue
    thr = float(np.quantile(np.abs(rH[trn]), 0.95))
    cal = ts[tst]
    vp95 = float(np.quantile(np.abs(rH[tst]), 0.95)); vmed = float(np.median(np.abs(rH[tst])))
    drift = float(np.mean(rH[tst]))
    nf = (np.abs(rH) >= thr) & tst
    nf_rate = float(nf.sum() / max(tst.sum(), 1))
    keep = np.where(nf & (fl | fs))[0]
    fill_rate = float((fl[tst] | fs[tst]).mean())
    best = np.where(fl[keep] & fs[keep], np.maximum(netl[0][keep], nets[0][keep]),
                    np.where(fl[keep], netl[0][keep], nets[0][keep]))
    oracle = float(np.nanmean(best)) if len(best) else float("nan")
    print(f"{fi:>4} {ds(cal.min()) + '..' + ds(cal.max()):>25} {len(set(day[tst].tolist())):>3} "
          f"{vp95:>6.1f} {vmed:>6.2f} {drift:>+6.2f} {100*nf_rate:>4.1f} {100*fill_rate:>5.1f} "
          f"{oracle:>+6.2f} {thr:>5.1f}")
    ts0 += T; fi += 1
print("\nvol = |rH60| bp; drift = mean rH60 bp/window (signed trend); oracle = best-side net maker EV bp "
      "on apred-non-flat fillable windows (available alpha ceiling).")
