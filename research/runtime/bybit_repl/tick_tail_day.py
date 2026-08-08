#!/usr/bin/env python3
"""HBV1 rev25 stage 2 (day worker): tick-granularity intra-position adverse
excursion (MAE) from the raw L2 book replay.

Reads the day's book parquet (per-message top-of-book), builds the mid-price
path, and for every exported trade of every target form that decides within
this day computes, over the position windows 210s (entry+hold) and 510s
(+exit chase):
  MAE_bp  — worst adverse mid move vs the decision-time mid, signed by side
  MFE_bp  — best favorable move (210s)
  END_bp  — signed move at window end (210s)
  spread_bp at decision time; trunc flag if the window crosses the day's
  last message (position spills past midnight — path truncated, MAE is a
  LOWER bound for those).
Entry-price approximation: mid at decision ts (maker peg fills within 60s at
touch; ±half-spread ~0.5-1bp on DOGE — noted, not modeled).
Env: DAY (YYYY-MM-DD), SYM, FORMS_LIST (comma-separated form names whose
HBV1_TRADES_{name}_{SYM}.json exist), LOCAL_GCS_ROOT.
Output: research_runs/HBV1_TICKTAIL_{SYM}/{DAY}.json
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
DAY = os.environ["DAY"]
SYM = os.environ.get("SYM", "DOGE")
FORMS_LIST = os.environ["FORMS_LIST"].split(",")
ROOT = os.environ.get("LOCAL_GCS_ROOT", "/vol/gcs")
BOOK = (f"{ROOT}/market-data-0998ac51/raw/book/exchange=BINANCE_FUTURES/"
        f"symbol={SYM}-USDT-PERP/dt={DAY}/1.parquet")
NS = 1_000_000_000
W_HOLD, W_CHASE = 210.0, 510.0

day_t0 = np.datetime64(DAY).astype("datetime64[s]").astype(np.int64)
day_t1 = day_t0 + 86400

trades = {}
for name in FORMS_LIST:
    ev = json.loads(bk.blob(f"research_runs/HBV1_TRADES_{name}_{SYM}.json").download_as_bytes())
    mine = [e for e in ev if day_t0 <= e[0] < day_t1]
    if mine:
        trades[name] = mine
if not trades:
    print(f"{DAY}: no trades", flush=True)
    raise SystemExit

t = pq.read_table(BOOK, columns=["timestamp", "bid_0_price", "ask_0_price"])
ts = np.asarray(t["timestamp"], np.int64).astype(np.float64) / NS
bid = np.asarray(t["bid_0_price"], np.float64)
ask = np.asarray(t["ask_0_price"], np.float64)
mid = 0.5 * (bid + ask)
o = np.argsort(ts, kind="stable")
ts, bid, ask, mid = ts[o], bid[o], ask[o], mid[o]

out = {}
for name, evs in trades.items():
    rows = []
    for ts0, side, dd, k, net in evs:
        i0 = np.searchsorted(ts, ts0, side="right") - 1
        if i0 < 0:
            continue
        entry = mid[i0]
        spread_bp = (ask[i0] - bid[i0]) / entry * 1e4
        rec = [round(ts0, 1), side, dd, k]
        for w in (W_HOLD, W_CHASE):
            i1 = np.searchsorted(ts, ts0 + w, side="right")
            seg = mid[i0:i1]
            trunc = ts0 + w > ts[-1]
            if side > 0:
                mae = (entry - seg.min()) / entry * 1e4
                mfe = (seg.max() - entry) / entry * 1e4
            else:
                mae = (seg.max() - entry) / entry * 1e4
                mfe = (entry - seg.min()) / entry * 1e4
            end = (seg[-1] - entry) / entry * 1e4 * (1 if side > 0 else -1)
            if w == W_HOLD:
                rec += [round(float(mae), 2), round(float(mfe), 2), round(float(end), 2)]
            else:
                rec += [round(float(mae), 2), int(trunc)]
        rec.append(round(float(spread_bp), 3))
        rows.append(rec)
    out[name] = rows

bk.blob(f"research_runs/HBV1_TICKTAIL_{SYM}/{DAY}.json").upload_from_string(json.dumps(out))
print(f"{DAY}: " + " ".join(f"{n}:{len(r)}" for n, r in out.items()), flush=True)
