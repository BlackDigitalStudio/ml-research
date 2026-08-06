#!/usr/bin/env python3
"""Did the price actually reach our resting BUY order in the 3 missed entries?

Distinguishes three physically different answers, per event, on the DOGEUSDC book:
  A) price NEVER came down to our level        -> only a re-quoting entry could help
  B) price reached our level but NOBODY sold   -> no flow; queue position irrelevant
  C) price reached, sells happened, but the queue ahead was never cleared -> queue position

Measures: seconds spent with best_bid <= our price, min best_bid, min traded price,
trade count/volume exactly at our level, and the queue ahead when we first became touch.
"""
import os
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime, timezone

SCR = os.path.dirname(os.path.abspath(__file__))
US = 1_000_000
TOL = 5e-7

EVENTS = [
    ("miss#1", "20260716_14", "2026-07-16T14:56:12.743", 0.07323, 137),
    ("miss#2", "20260720_05", "2026-07-20T05:13:13.365", 0.07146, 140),
    ("miss#3", "20260723_12", "2026-07-23T12:36:57.804", 0.07138, 141),
]

def lvl_qty(row, px):
    p, q = row["bid_prices"], row["bid_qtys"]
    if p is None:
        return 0.0
    for i in range(len(p)):
        if abs(float(p[i]) - px) < TOL:
            return float(q[i])
    return 0.0

for tag, dh, ts, px, myqty in EVENTS:
    t0 = int(datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp() * US)
    t1 = t0 + 60 * US
    bk = pq.read_table(f"{SCR}/usdc/DOGEUSDC/depth_snapshot/{dh}.parquet",
                       columns=["local_ts_us", "bid_prices", "bid_qtys", "ask_prices"]).to_pandas()
    bk = bk[bk["local_ts_us"].notna()].sort_values("local_ts_us")
    w = bk[(bk["local_ts_us"] >= t0) & (bk["local_ts_us"] <= t1)].reset_index(drop=True)
    tr = pq.read_table(f"{SCR}/usdc/DOGEUSDC/agg_trade/{dh}.parquet",
                       columns=["local_ts_us", "price", "qty", "is_buyer_maker"]).to_pandas()
    tw = tr[(tr["local_ts_us"] >= t0) & (tr["local_ts_us"] <= t1)]
    tw_px = tw["price"].astype(float).to_numpy()
    tw_q = tw["qty"].astype(float).to_numpy()
    tw_bm = tw["is_buyer_maker"].astype(bool).to_numpy()
    tw_ts = tw["local_ts_us"].to_numpy()

    bid = np.array([float(r[0]) for r in w["bid_prices"]])
    ask = np.array([float(r[0]) for r in w["ask_prices"]])
    tsx = w["local_ts_us"].to_numpy()
    at = bid <= px + TOL                     # our order is at (or better than) the touch
    # time spent at touch, from snapshot spacing
    dt = np.diff(np.append(tsx, t1)) / US
    secs_at = float(dt[at].sum())
    # first moment we became touch, and the queue ahead of us then
    qa = None
    if at.any():
        i0 = int(np.argmax(at))
        qa = lvl_qty(w.iloc[i0], px)
    # sells at exactly our level, and anything at or below
    at_lvl = tw_bm & (np.abs(tw_px - px) < TOL)
    below = tw_bm & (tw_px <= px + TOL)
    print(f"\n===== {tag}  our BUY {myqty} @ {px:.5f}   (60s window)")
    print(f"  best bid: min {bid.min():.5f}  max {bid.max():.5f}   |  our price is "
          f"{1e4*(bid.min()-px)/px:+.1f}bp below the window's LOWEST bid" if bid.min() > px + TOL
          else f"  best bid: min {bid.min():.5f}  max {bid.max():.5f}")
    if bid.min() > px + TOL:
        print(f"  -> (A) PRICE NEVER REACHED US. Closest approach {1e4*(bid.min()-px)/px:.1f}bp above our order.")
    else:
        print(f"  -> price DID reach our level: best_bid <= ours in {at.sum()}/{len(at)} snapshots "
              f"= {secs_at:.1f}s of 60s; queue ahead when we first became touch: {qa:,.0f} DOGE")
    print(f"  trades in window: {len(tw_px)} (min traded px {tw_px.min():.5f})")
    print(f"  aggressive SELLS exactly at our level: {int(at_lvl.sum())} trades, "
          f"{tw_q[at_lvl].sum():,.0f} DOGE | at or below our level: {int(below.sum())} trades, "
          f"{tw_q[below].sum():,.0f} DOGE")
    if bid.min() <= px + TOL:
        if tw_q[below].sum() == 0:
            print("  -> (B) reached, but ZERO aggressive sell flow at our price while we were the touch")
        elif qa is not None and tw_q[below].sum() < qa + myqty:
            print(f"  -> (C) reached with flow {tw_q[below].sum():,.0f} but queue ahead {qa:,.0f} never cleared")
        else:
            print("  -> ANOMALY: flow exceeded the queue — we should have filled")
