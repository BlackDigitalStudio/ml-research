#!/usr/bin/env bash
# Hourly mirror of Chronos parquet + raw archive -> GCS (additive). Excludes
# the transient .parts/ working dir and half-written *.tmp. The raw
# .jsonl.gz archive IS synced (loss-less re-normalization safety net).
# Auth via the VM's attached service account (metadata) — no keys.
set -euo pipefail
BUCKET="@BUCKET@"
HOST="$(hostname)"
exec gcloud storage rsync -r \
  -x '.*/\.parts/.*|.*\.tmp$' \
  /home/scalper/crypto-market-recorder/data \
  "gs://${BUCKET}/chronos/${HOST}"
