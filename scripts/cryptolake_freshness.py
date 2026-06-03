#!/usr/bin/env python3
"""Probe cryptolake (lakeapi) for the LATEST available dt per symbol/table.

Metadata-only: uses lakeapi.list_data (S3 contents catalog) -> does NOT download
book/trades parquet, so it does not count against the 300GB/mo data quota.
Credentials come from ~/.aws/credentials (profile [default]); none are stored here.
"""
import datetime as dt
import sys
import lakeapi

SYMS = [f"{s}-USDT-PERP" for s in ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]]
EXCH = ["BINANCE_FUTURES"]
TABLES = ["book", "trades"]
OUR_EDGE = "2026-05-08"   # current freshest dt on our GCS bucket (book/trades, most syms)


def latest_per_symbol(table, start, end):
    try:
        objs = lakeapi.list_data(table=table, exchanges=EXCH, symbols=SYMS, start=start, end=end)
    except Exception as e:
        print(f"  [{table}] list_data error: {e}")
        return {}
    mx = {}
    for o in objs:
        s = o["symbol"]; d = o["dt"]
        if s not in mx or d > mx[s]:
            mx[s] = d
    return mx


def main():
    today = dt.datetime.now()
    start = dt.datetime(2026, 4, 1)
    end = today + dt.timedelta(days=2)   # include today/tomorrow if present
    print(f"today={today.date()}  our_current_edge={OUR_EDGE}  query=[{start.date()} .. {end.date()})")
    res = {t: latest_per_symbol(t, start, end) for t in TABLES}
    print(f"\n{'SYM':16s} {'book latest':>12s} {'trades latest':>14s} {'vs our edge':>12s}")
    for s in SYMS:
        b = res["book"].get(s, "—"); t = res["trades"].get(s, "—")
        eff = min([x for x in (b, t) if x != "—"], default="—")
        gain = ""
        if eff != "—":
            d_eff = dt.date.fromisoformat(eff); d_edge = dt.date.fromisoformat(OUR_EDGE)
            gain = f"+{(d_eff - d_edge).days}d" if d_eff > d_edge else f"{(d_eff - d_edge).days}d"
        print(f"{s:16s} {b:>12s} {t:>14s} {gain:>12s}")
    # global freshest
    alld = [d for t in TABLES for d in res[t].values()]
    if alld:
        print(f"\nFRESHEST available anywhere: {max(alld)}   (we have up to {OUR_EDGE})")
    try:
        u = lakeapi.used_data()
        print(f"quota: downloaded_gb={u.get('downloaded_gb')} over {u.get('timeframe_days')}d  user={u.get('user')}")
    except Exception as e:
        print(f"[used_data unavailable: {e}]")


if __name__ == "__main__":
    sys.exit(main())
