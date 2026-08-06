#!/usr/bin/env python3
"""One-shot aux inputs for the Bybit replication (Bybit REST is geo-fenced from US
egress; the Bybit file archive has no funding/OI streams). Venue-proxy from the
official Binance Vision archive, ledger-noted:

  bybit_aux/funding_rates.json  <- monthly fundingRate zips (settled rates, ms ts)
  bybit_aux/oi_5m.parquet       <- daily metrics zips: sum_open_interest @5min

Env: START, END, SYM (default DOGEUSDT), LOCAL_GCS_ROOT.
"""
import csv
import io
import json
import os
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


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except Exception:
        return None


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


def funding():
    rows = []
    for mo in months(START, END):
        raw = get(f"{BV}/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{mo}.zip")
        if raw is None:
            print(f"  funding {mo}: missing", flush=True)
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(z.namelist()[0]) as f:
                rd = csv.reader(io.TextIOWrapper(f))
                hdr = next(rd)
                ci = hdr.index("calc_time"); ri = hdr.index("last_funding_rate")
                for r in rd:
                    rows.append((int(r[ci]), float(r[ri])))
    rows.sort()
    json.dump(rows, open(os.path.join(AUX, "funding_rates.json"), "w"))
    print(f"funding: {len(rows)} settlements {rows[0][0]}..{rows[-1][0]}", flush=True)


def oi():
    ts_l, oi_l = [], []
    for d in days(START, END):
        raw = get(f"{BV}/daily/metrics/{SYM}/{SYM}-metrics-{d}.zip")
        if raw is None:
            print(f"  metrics {d}: missing", flush=True)
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(z.namelist()[0]) as f:
                rd = csv.reader(io.TextIOWrapper(f))
                hdr = next(rd)
                ti = hdr.index("create_time"); si = hdr.index("sum_open_interest")
                for r in rd:
                    t = np.datetime64(r[ti].replace(" ", "T")).astype("datetime64[s]").astype(np.int64)
                    ts_l.append(t * NS); oi_l.append(float(r[si]))
    o = np.argsort(np.asarray(ts_l, np.int64), kind="stable")
    ts = np.asarray(ts_l, np.int64)[o]; ov = np.asarray(oi_l, np.float64)[o]
    pq.write_table(pa.table({"timestamp": pa.array(ts, pa.int64()),
                             "open_interest": pa.array(ov, pa.float64())}),
                   os.path.join(AUX, "oi_5m.parquet"))
    print(f"oi: {len(ts)} rows", flush=True)


if __name__ == "__main__":
    funding()
    oi()
