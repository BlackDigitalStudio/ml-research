#!/usr/bin/env python3
"""One-shot aux inputs for the Bybit replication (HBV1 AMEND1).

Bybit REST is reachable from Modal egress (probed 2026-08-06, HTTP 200) though
geo-fenced from the dev container — so funding and OI come from the REAL venue:

  bybit_aux/funding_rates.json  <- /v5/market/funding/history (settled rates, ms ts)
  bybit_aux/oi_5m.parquet       <- /v5/market/open-interest intervalTime=5min

Fallback (if REST fails mid-run): Binance Vision archives as in the original
prereg (fundingRate monthly zips + metrics daily zips), venue-proxy ledger-noted.

Env: START, END, SYM (default DOGEUSDT), LOCAL_GCS_ROOT.
"""
import csv
import io
import json
import os
import time
import urllib.request
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NS = 1_000_000_000
ROOT = os.environ.get("LOCAL_GCS_ROOT", "/vol/gcs")
AUX = os.path.join(ROOT, "market-data-0998ac51", "bybit_aux")
os.makedirs(AUX, exist_ok=True)
SYM = os.environ.get("SYM", "DOGEUSDT")
START = os.environ.get("START", "2025-05-01")
END = os.environ.get("END", "2026-06-05")
BV = "https://data.binance.vision/data/futures/um"
BYBIT = "https://api.bybit.com"


def get(url, retries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    for att in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if "404" in str(e):
                return None
            time.sleep(1.5 * (att + 1))
    return None


def ms(day):
    return int(np.datetime64(day).astype("datetime64[s]").astype(np.int64)) * 1000


def bybit_funding():
    rows, end_t = [], ms(END) + 86_400_000
    t0 = ms(START)
    while end_t > t0:
        raw = get(f"{BYBIT}/v5/market/funding/history?category=linear&symbol={SYM}"
                  f"&startTime={t0}&endTime={end_t}&limit=200")
        if raw is None:
            return None
        lst = json.loads(raw)["result"]["list"]
        if not lst:
            break
        for x in lst:
            rows.append((int(x["fundingRateTimestamp"]), float(x["fundingRate"])))
        oldest = min(int(x["fundingRateTimestamp"]) for x in lst)
        if oldest <= t0 or len(lst) < 200:
            break
        end_t = oldest - 1
        time.sleep(0.15)
    rows = sorted(set(rows))
    return rows if rows else None


def bybit_oi():
    rows, end_t = [], ms(END) + 86_400_000
    t0 = ms(START)
    cursor = ""
    while True:
        url = (f"{BYBIT}/v5/market/open-interest?category=linear&symbol={SYM}"
               f"&intervalTime=5min&startTime={t0}&endTime={end_t}&limit=200")
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        raw = get(url)
        if raw is None:
            return None
        res = json.loads(raw)["result"]
        lst = res.get("list", [])
        if not lst:
            break
        for x in lst:
            rows.append((int(x["timestamp"]), float(x["openInterest"])))
        cursor = res.get("nextPageCursor", "")
        if not cursor:
            # windowed fallback: move end to oldest seen - 1
            oldest = min(int(x["timestamp"]) for x in lst)
            if oldest <= t0 or len(lst) < 200:
                break
            end_t = oldest - 1
        time.sleep(0.12)
    rows = sorted(set(rows))
    return rows if rows else None


# ---------------- Binance Vision fallback (original prereg path) ----------------

def months(a, b):
    y, m = int(a[:4]), int(a[5:7])
    while f"{y:04d}-{m:02d}" <= b[:7]:
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def days(a, b):
    d = np.datetime64(a)
    while str(d) <= b:
        yield str(d)
        d += 1


def binance_funding():
    rows = []
    for mo in months(START, END):
        raw = get(f"{BV}/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{mo}.zip")
        if raw is None:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z, z.open(z.namelist()[0]) as f:
            rd = csv.reader(io.TextIOWrapper(f))
            hdr = next(rd)
            ci = hdr.index("calc_time"); ri = hdr.index("last_funding_rate")
            for r in rd:
                rows.append((int(r[ci]), float(r[ri])))
    return sorted(rows) or None


def binance_oi():
    rows = []
    for d in days(START, END):
        raw = get(f"{BV}/daily/metrics/{SYM}/{SYM}-metrics-{d}.zip")
        if raw is None:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z, z.open(z.namelist()[0]) as f:
            rd = csv.reader(io.TextIOWrapper(f))
            hdr = next(rd)
            ti = hdr.index("create_time"); si = hdr.index("sum_open_interest")
            for r in rd:
                t = np.datetime64(r[ti].replace(" ", "T")).astype("datetime64[s]").astype(np.int64)
                rows.append((int(t) * 1000, float(r[si])))
    return sorted(rows) or None


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  (used in bybit_oi)

    fr = bybit_funding()
    src_f = "bybit_rest"
    if fr is None:
        fr = binance_funding(); src_f = "binance_vision_proxy"
    assert fr, "no funding data from either source"
    json.dump(fr, open(os.path.join(AUX, "funding_rates.json"), "w"))
    print(f"funding[{src_f}]: {len(fr)} settlements {fr[0][0]}..{fr[-1][0]}", flush=True)

    oi = bybit_oi()
    src_o = "bybit_rest"
    if oi is None:
        oi = binance_oi(); src_o = "binance_vision_proxy"
    assert oi, "no OI data from either source"
    ts = np.array([x[0] for x in oi], np.int64) * 1_000_000
    ov = np.array([x[1] for x in oi], np.float64)
    pq.write_table(pa.table({"timestamp": pa.array(ts, pa.int64()),
                             "open_interest": pa.array(ov, pa.float64())}),
                   os.path.join(AUX, "oi_5m.parquet"))
    json.dump({"funding_source": src_f, "oi_source": src_o},
              open(os.path.join(AUX, "sources.json"), "w"))
    print(f"oi[{src_o}]: {len(ts)} rows {str(ts[0])[:13]}..{str(ts[-1])[:13]}", flush=True)
