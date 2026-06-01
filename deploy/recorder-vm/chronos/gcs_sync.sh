#!/usr/bin/env bash
# Hourly mirror of Chronos parquet + raw archive -> GCS (additive). Excludes
# the transient .parts/ working dir and half-written *.tmp. The raw
# .jsonl.gz archive IS synced (loss-less re-normalization safety net).
# Auth via the VM's attached service account (metadata) — no keys.
set -euo pipefail
BUCKET="@BUCKET@"
HOST="$(hostname)"
gcloud storage rsync -r \
  -x '.*/\.parts/.*|.*\.tmp$' \
  /home/scalper/crypto-market-recorder/data \
  "gs://${BUCKET}/chronos/${HOST}"
# Mark a successful sync; the watchdog alerts if this goes stale (a silent
# sync failure would otherwise let 3-day retention delete un-uploaded data).
date +%s > /home/scalper/.last_gcs_sync
