#!/usr/bin/env python3
"""HX1 rev5 — CV-block builder: cross-venue features on the 3s decision grid.

Consumes the HX1 rev4 1s grid cache (research_runs/hx1_oos/grid/{COIN}/{day}.npz
— binance mid/obi5 + per-venue last px / signed qty / total qty per second) and
emits the frozen CV11 block sampled at the anchored 3s decision grid:

  cols 0-3   basis_dm per venue (bybit, okx, bitget, gateio; 120s demean)
  cols 4-6   flow_imb pooled (mean of 4 venues), trailing 15/30/60s
  cols 7-10  flow_imb_15 per venue

Causality: the decision tick at grid ts T uses the 1s-grid value of second
T-1 (strictly pre-T data; the venue feed is not part of the book-tick clock).
NaN (first 120s of day / dead venue) stays NaN — the consumer decides the
fill policy (frozen in the rev5 prereg: NaN -> 0.0, features_v1 convention).

Output: gs://market-data-0998ac51/research_runs/hx1_stack/cv/{COIN}/{day}.npy
float32 [28800 x 11] (one row per 3s grid point of the UTC day) + META.json.

Env: COINS(DOGE,XRP,BTC,ETH) DAY0(20260628) DAYN(20260714)
"""
import json
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("DAY0", "20260628")
os.environ.setdefault("DAYN", "20260714")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx1_oos import (BUCKET_OUT as HX1_GRID_ROOT, VENUES, day_list,  # noqa: E402
                     day_signals, SEC_DAY)

OUT = os.environ.get("OUT", "gs://market-data-0998ac51/research_runs/hx1_stack/cv")
COINS = os.environ.get("COINS", "DOGE,XRP,BTC,ETH").split(",")
STEP_S = 3
NGRID = SEC_DAY // STEP_S  # 28800

CV_COLS = ([f"basis_{v}" for v in VENUES]
           + [f"flow{w}_pool" for w in (15, 30, 60)]
           + [f"flow15_{v}" for v in VENUES])


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1000:]}")


def build(coin, day, workdir="hx1_stack_local"):
    os.makedirs(workdir, exist_ok=True)
    lf = f"{workdir}/{coin}_{day}.npz"
    if not os.path.exists(lf):
        sh(f"gcloud storage cp {HX1_GRID_ROOT}/grid/{coin}/{day}.npz {lf} -q")
    sig, _ = day_signals(np.load(lf))
    # decision grid ts = 0,3,6,...s of the UTC day; use second T-1 (pre-T data)
    sec_idx = np.arange(NGRID) * STEP_S - 1
    sec_idx[0] = 0  # midnight point has no pre-T second; stays NaN anyway (warmup)
    cv = np.column_stack([sig[c][sec_idx] for c in CV_COLS]).astype(np.float32)
    out_local = f"{workdir}/{coin}_{day}_cv.npy"
    np.save(out_local, cv)
    sh(f"gcloud storage cp {out_local} {OUT}/{coin}/{day}.npy -q")
    return cv


def main():
    meta = dict(cols=CV_COLS, step_s=STEP_S, ngrid=NGRID,
                causality="value of second T-1 for grid ts T",
                source=f"{HX1_GRID_ROOT}/grid (HX1 rev4 cache)",
                nan_policy="NaN preserved; consumer fills 0.0 per rev5 prereg")
    with open("cv_META.json", "w") as f:
        json.dump(meta, f, indent=1)
    sh(f"gcloud storage cp cv_META.json {OUT}/META.json -q")
    for coin in COINS:
        for day in day_list():
            cv = build(coin, day)
            nanpct = 100 * np.isnan(cv).mean()
            print(f"{coin} {day} cv[{cv.shape[0]}x{cv.shape[1]}] nan%={nanpct:.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
