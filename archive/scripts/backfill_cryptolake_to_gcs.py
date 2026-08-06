#!/usr/bin/env python3
"""Backfill cryptolake raw (S3) -> GCS raw/ for a date window, preserving the native
parquet (schema is a verified direct mirror: book timestamp+bid_0_price..ask_19_size,
trades side/amount/price/id/timestamp).

Creds: boto3 reads cryptolake S3 from AWS_* env vars; GCS write uses the host's
default (VM = virgin.ship03 service account). NO secrets stored in this file.
Idempotent: skips objects already present on GCS.

Run on VM:
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=eu-west-1 \
    python3 backfill_cryptolake_to_gcs.py --start 2026-05-09 --end 2026-06-02
"""
import argparse, datetime as dt, sys, time
import boto3
from google.cloud import storage

S3_BUCKET = "qnt.data"; S3_PREFIX = "market-data/cryptofeed"
GCS_PROJ = "project-0998ac51-36ba-445c-bc7"; GCS_BUCKET = "market-data-0998ac51"
PLAN = {
    "DOGE-USDT-PERP": ["book", "trades", "funding", "open_interest"],
    "BTC-USDT-PERP":  ["book", "trades", "funding", "open_interest"],
    "ETH-USDT-PERP":  ["trades"],
}


def daterange(a, b):
    d0 = dt.date.fromisoformat(a); d1 = dt.date.fromisoformat(b)
    return [(d0 + dt.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name="eu-west-1")
    gcs = storage.Client(project=GCS_PROJ).bucket(GCS_BUCKET)
    days = daterange(a.start, a.end)
    print(f"window {days[0]}..{days[-1]} ({len(days)}d) plan={ {k:len(v) for k,v in PLAN.items()} }", flush=True)
    tot_files = 0; tot_bytes = 0; t0 = time.time()
    for sym, tables in PLAN.items():
        for tbl in tables:
            n_ok = n_skip = n_miss = 0; b_sum = 0
            for day in days:
                pref = f"{S3_PREFIX}/{tbl}/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
                objs = [o for o in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=pref).get("Contents", [])
                        if o["Key"].endswith(".parquet")]
                if not objs:
                    n_miss += 1; continue
                for o in objs:
                    fn = o["Key"].split("/")[-1]
                    blob = gcs.blob(f"raw/{tbl}/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/{fn}")
                    if blob.exists():
                        n_skip += 1; continue
                    body = s3.get_object(Bucket=S3_BUCKET, Key=o["Key"])["Body"].read()
                    blob.upload_from_string(body, content_type="application/octet-stream")
                    n_ok += 1; b_sum += len(body); tot_files += 1; tot_bytes += len(body)
            print(f"  {sym:16s} {tbl:14s} copied={n_ok} skip={n_skip} miss_days={n_miss} {b_sum/1e6:7.0f}MB", flush=True)
    print(f"DONE files={tot_files} {tot_bytes/1e9:.2f}GB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
