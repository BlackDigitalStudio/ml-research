#!/usr/bin/env python3
"""Reference check of the strict-entry-fill semantics on the six real live events.

Ports both branches (library = current frozen behaviour, strict = the fix) and runs
them on the recorder book+flow for each event, on BOTH venues. Verifies:
  - strict never fills where library misses (strict is a subset = more conservative)
  - the phantom fills (sim filled, live did not) disappear where the cause was the
    unconditional gap-through with no flow.
Flow is aggregated per book snapshot interval, as build_samples does.
"""
import os
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime, timezone

SCR = os.path.dirname(os.path.abspath(__file__))
US = 1_000_000

EVENTS = [
    ("miss#1", "20260716_14", "2026-07-16T14:56:12.743", 0.07323, "MISS"),
    ("miss#2", "20260720_05", "2026-07-20T05:13:13.365", 0.07146, "MISS"),
    ("miss#3", "20260723_12", "2026-07-23T12:36:57.804", 0.07138, "MISS"),
    ("fill#A", "20260720_19", "2026-07-20T19:38:03.713", 0.07150, "FILL"),
    ("fill#B", "20260723_12", "2026-07-23T12:46:13.243", 0.07076, "FILL"),
    ("fill#C", "20260723_13", "2026-07-23T13:26:36.426", 0.07031, "FILL"),
]


def entry_lib(level, q0, bid, sell_vol):
    """live_sim::simulate_maker_entry, Long branch (frozen behaviour)."""
    eps = level * 1e-7
    q = max(q0, 0.0)
    for k in range(len(bid)):
        if bid[k] <= 0:
            continue
        if bid[k] < level - eps:
            return k, "gap-through"
        if bid[k] <= level + eps:
            q -= sell_vol[k]
            if q <= 0:
                return k, "queue-cleared"
    return None, "miss"


def entry_strict(level, q0, bid, sell_vol):
    """--strict-entry-fill: gap zeroes the queue but flow is still required."""
    eps = level * 1e-7
    q = max(q0, 0.0)
    for k in range(len(bid)):
        if bid[k] <= 0:
            continue
        if bid[k] < level - eps:
            q = 0.0
            if sell_vol[k] > 0:
                return k, "gap+flow"
        elif bid[k] <= level + eps:
            q -= sell_vol[k]
            if q <= 0:
                return k, "queue-cleared"
    return None, "miss"


def build(sym, dh, t0, t1, level):
    bk = pq.read_table(f"{SCR}/usdc/{sym}/depth_snapshot/{dh}.parquet",
                       columns=["local_ts_us", "bid_prices", "bid_qtys"]).to_pandas()
    bk = bk[bk["local_ts_us"].notna()].sort_values("local_ts_us")
    w = bk[(bk["local_ts_us"] >= t0) & (bk["local_ts_us"] <= t1)].reset_index(drop=True)
    bid = np.array([float(r[0]) for r in w["bid_prices"]])
    tsx = w["local_ts_us"].to_numpy()
    tr = pq.read_table(f"{SCR}/usdc/{sym}/agg_trade/{dh}.parquet",
                       columns=["local_ts_us", "qty", "is_buyer_maker"]).to_pandas()
    tw = tr[(tr["local_ts_us"] >= t0) & (tr["local_ts_us"] <= t1)]
    sells = tw[tw["is_buyer_maker"].astype(bool)]
    # aggregate aggressive sell volume into each book-snapshot interval
    idx = np.searchsorted(tsx, sells["local_ts_us"].to_numpy(), "right") - 1
    sv = np.zeros(len(bid))
    ok = idx >= 0
    np.add.at(sv, idx[ok], sells["qty"].astype(float).to_numpy()[ok])
    # queue ahead at our level when we join
    q0 = 0.0
    if len(w):
        p0, qq0 = w.iloc[0]["bid_prices"], w.iloc[0]["bid_qtys"]
        for i in range(len(p0)):
            if abs(float(p0[i]) - level) < 5e-7:
                q0 = float(qq0[i]); break
    return bid, sv, q0


print(f"{'event':>7} {'venue':>9} {'live':>5} | {'LIBRARY (frozen)':>22} | {'STRICT (fix)':>22}")
for tag, dh, ts, px_usdc, outcome in EVENTS:
    t0 = int(datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp() * US)
    t1 = t0 + 60 * US
    for sym in ("DOGEUSDC", "DOGEUSDT"):
        bid, sv, q0 = build(sym, dh, t0, t1, px_usdc if sym == "DOGEUSDC" else 0.0)
        level = px_usdc if sym == "DOGEUSDC" else float(bid[0])
        if sym == "DOGEUSDT":
            bid, sv, q0 = build(sym, dh, t0, t1, level)
        kl, wl = entry_lib(level, q0, bid, sv)
        ks, ws = entry_strict(level, q0, bid, sv)
        f_l = "FILL" if kl is not None else "miss"
        f_s = "FILL" if ks is not None else "miss"
        flag = "  <-- phantom removed" if (kl is not None and ks is None) else ""
        assert not (ks is not None and kl is None), "strict must be a subset of library"
        print(f"{tag:>7} {sym:>9} {outcome:>5} | {f_l:>5} ({wl:>14}) | {f_s:>5} ({ws:>14}){flag}")
print("\ninvariant held: strict never fills where the library misses (subset property)")
