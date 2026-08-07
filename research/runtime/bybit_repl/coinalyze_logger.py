#!/usr/bin/env python3
"""HBV1 rev5: forward accrual of Coinalyze 1-min per-exchange series (free-tier
sub-daily history is only ~2-3 months deep — measured 2026-08-06 — so the only way
to have liq/OI for future year-cells is to log them daily from now on).

Deployed Modal app (schedule: daily 03:10 UTC). Pulls YESTERDAY (UTC) of
  /liquidation-history  (l/s long-short liq volumes, 1min buckets)
  /open-interest-history (1min OHLC of OI)
for the 8 legacy symbols on Bybit (.6) + the .A Binance codes, appends parquet to
the bybit-cl volume under coinalyze_fwd/{kind}/{symbol}/{YYYY-MM-DD}.parquet.
Key from Modal secret `coinalyze-api-key` (never in git). Idempotent per day.

Deploy:  modal deploy coinalyze_logger.py
Stop:    modal app stop coinalyze-liq-logger
"""
import datetime as dt
import json
import os
import time
import urllib.request

import modal

app = modal.App("coinalyze-liq-logger")
vol = modal.Volume.from_name("bybit-cl", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("pyarrow", "numpy")

SYMS = [f"{s}USDT.6" for s in ("BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP")] + \
       [f"{s}USDT_PERP.A" for s in ("BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP")]
KINDS = {"liquidation": "liquidation-history", "open_interest": "open-interest-history"}


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("coinalyze-api-key")],
              timeout=1800, schedule=modal.Cron("10 3 * * *"))
def pull_daily():
    day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date()
    return _pull_for(day)


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("coinalyze-api-key")],
              timeout=1800)
def pull_backfill_day(days_ago: int):
    day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)).date()
    return _pull_for(day)


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("coinalyze-api-key")],
              timeout=3 * 3600)
def backfill_seq(days: int = 75):
    """Sequential backfill in ONE container — the parallel map hit the 40 req/min
    rate limit and dropped most days (2026-08-06). Idempotent: existing files skip."""
    done = 0
    for i in range(days, 0, -1):
        day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=i)).date()
        done += _pull_for(day)
        time.sleep(3)
    return f"backfill_seq: {done} series"


@app.local_entrypoint()
def backfill(days: int = 75):
    print(backfill_seq.remote(days))


def _pull_for(day):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    key = os.environ["COINALYZE_API_KEY"]
    t0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())
    t1 = t0 + 86400
    root = "/vol/gcs/market-data-0998ac51/coinalyze_fwd"
    n_ok = 0
    for kind, ep in KINDS.items():
        for chunk in range(0, len(SYMS), 8):
            syms = SYMS[chunk:chunk + 8]
            url = (f"https://api.coinalyze.net/v1/{ep}?symbols={','.join(syms)}"
                   f"&interval=1min&from={t0}&to={t1}")
            req = urllib.request.Request(url, headers={"api_key": key})
            for att in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = json.loads(r.read())
                    break
                except Exception:
                    time.sleep(3 * (att + 1))
            else:
                continue
            for entry in data:
                sym = entry["symbol"]; h = entry.get("history", [])
                if not h:
                    continue
                d = os.path.join(root, kind, sym)
                os.makedirs(d, exist_ok=True)
                dst = os.path.join(d, f"{day}.parquet")
                if os.path.exists(dst):
                    continue
                keys = sorted({k for row in h for k in row})
                cols = {k: pa.array(np.array([row.get(k) for row in h],
                                             dtype=np.int64 if k == "t" else np.float64)) for k in keys}
                pq.write_table(pa.table(cols), dst + ".tmp")
                os.replace(dst + ".tmp", dst)
                n_ok += 1
            time.sleep(2)
    vol.commit()
    print(f"[coinalyze_fwd backfill] {day}: {n_ok} series", flush=True)
    return n_ok
