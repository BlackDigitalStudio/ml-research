#!/usr/bin/env python3
"""Why did 3 live entries miss while the USDT-book sim filled all of them?

Tests, per event, on the RECORDER's own DOGEUSDC book (the venue we actually trade):
  H1 "engine/venue price mismatch": was our limit price BELOW the USDC touch at
      placement (i.e. we quoted a stale/wrong level and rested behind the book)?
  H2 "no flow": was our price AT the touch but the aggressive sell volume at
      price <= ours never cleared the queue ahead?
  H3 "price ran away": did the bid leave our level upward and never return?
and the USDT counterfactual (what the sim saw) + the USDC-vs-USDT mid gap (HUSDC).
"""
import os
import numpy as np
import pyarrow.parquet as pq

SCR = os.path.dirname(os.path.abspath(__file__))
US = 1_000_000

EVENTS = [   # (tag, day_hour, iso_ts_local, side, price, qty, outcome)
    ("miss#1", "20260716_14", "2026-07-16T14:56:12.743", "BUY", 0.07323, 137, "MISS"),
    ("miss#2", "20260720_05", "2026-07-20T05:13:13.365", "BUY", 0.07146, 140, "MISS"),
    ("miss#3", "20260723_12", "2026-07-23T12:36:57.804", "BUY", 0.07138, 141, "MISS"),
    ("fill#A", "20260720_19", "2026-07-20T19:38:03.713", "BUY", 0.07150, 140, "FILL"),
    ("fill#B", "20260723_12", "2026-07-23T12:46:13.243", "BUY", 0.07076, 142, "FILL"),
    ("fill#C", "20260723_13", "2026-07-23T13:26:36.426", "BUY", 0.07031, 144, "FILL"),
]
WIN_S = 60.0

def load_book(sym, dh):
    t = pq.read_table(f"{SCR}/usdc/{sym}/depth_snapshot/{dh}.parquet",
                      columns=["exchange_event_ts_us", "local_ts_us", "bid_prices", "bid_qtys",
                               "ask_prices", "ask_qtys"]).to_pandas()
    t = t[t["local_ts_us"].notna()].sort_values("local_ts_us")
    return t

def load_trades(sym, dh):
    t = pq.read_table(f"{SCR}/usdc/{sym}/agg_trade/{dh}.parquet",
                      columns=["exchange_event_ts_us", "local_ts_us", "price", "qty",
                               "is_buyer_maker"]).to_pandas()
    t = t[t["local_ts_us"].notna()].sort_values("local_ts_us")
    return t

def iso_us(s):
    from datetime import datetime, timezone
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * US)

def lvl_qty(row, px, side="bid"):
    """resting qty at exactly px (queue ahead of a joining order)."""
    p = row[f"{side}_prices"]; q = row[f"{side}_qtys"]
    if p is None:
        return 0.0
    for i in range(len(p)):
        if abs(float(p[i]) - px) < 5e-7:
            return float(q[i])
    return 0.0

for tag, dh, ts, side, px, qty, outcome in EVENTS:
    t0 = iso_us(ts); t1 = t0 + int(WIN_S * US)
    print(f"\n===== {tag} [{outcome}] {ts} {side} {qty} @ {px}")
    for sym in ("DOGEUSDC", "DOGEUSDT"):
        bk = load_book(sym, dh); tr = load_trades(sym, dh)
        w = bk[(bk["local_ts_us"] >= t0 - 2 * US) & (bk["local_ts_us"] <= t1)]
        if not len(w):
            print(f"  {sym}: no book rows in window"); continue
        b0row = w.iloc[(w["local_ts_us"] - t0).abs().argmin()]
        b0 = float(b0row["bid_prices"][0]); a0 = float(b0row["ask_prices"][0])
        # our effective price on this venue: USDC = the real order; USDT = the sim's touch
        my = px if sym == "DOGEUSDC" else b0
        wt = w[w["local_ts_us"] >= t0]
        bb = np.array([float(r[0]) for r in wt["bid_prices"]])
        aa = np.array([float(r[0]) for r in wt["ask_prices"]])
        at_touch = float(np.mean(bb <= my + 5e-7))          # our order is at/above the touch
        behind = float(np.mean(bb > my + 5e-7))             # bid above us -> we rest deeper
        q_ahead = lvl_qty(b0row, my, "bid")
        tw = tr[(tr["local_ts_us"] >= t0) & (tr["local_ts_us"] <= t1)]
        sells = tw[(tw["is_buyer_maker"].astype(bool)) & (tw["price"].astype(float) <= my + 5e-7)]
        vol = float(sells["qty"].astype(float).sum())
        print(f"  {sym}: touch at t0 {b0:.5f}/{a0:.5f} | our px {my:.5f} "
              f"({'AT TOUCH' if abs(my-b0)<5e-7 else ('ABOVE bid' if my>b0 else f'BELOW bid by {1e4*(b0-my)/b0:.1f}bp')})")
        print(f"      window: bid<=ours {100*at_touch:.0f}% of snapshots | bid>ours {100*behind:.0f}% | "
              f"bid range {bb.min():.5f}-{bb.max():.5f}")
        print(f"      queue ahead at t0 {q_ahead:.0f} | aggressive SELL vol at <=ours {vol:.0f} "
              f"-> {'CLEARS queue+our size' if vol > q_ahead + qty else 'insufficient'}")
    # HUSDC: venue mid gap over the window
    bc = load_book("DOGEUSDC", dh); bt = load_book("DOGEUSDT", dh)
    wc = bc[(bc["local_ts_us"] >= t0) & (bc["local_ts_us"] <= t1)]
    wtt = bt[(bt["local_ts_us"] >= t0) & (bt["local_ts_us"] <= t1)]
    if len(wc) and len(wtt):
        mc = np.array([(float(r["bid_prices"][0]) + float(r["ask_prices"][0])) / 2 for _, r in wc.iterrows()])
        mt = np.array([(float(r["bid_prices"][0]) + float(r["ask_prices"][0])) / 2 for _, r in wtt.iterrows()])
        i = np.searchsorted(wtt["local_ts_us"].to_numpy(), wc["local_ts_us"].to_numpy()) - 1
        i = np.clip(i, 0, len(mt) - 1)
        gap = 1e4 * (mc - mt[i]) / mt[i]
        print(f"  HUSDC mid gap USDC-USDT: mean {gap.mean():+.1f}bp sd {gap.std():.1f} "
              f"range [{gap.min():+.1f},{gap.max():+.1f}]")
