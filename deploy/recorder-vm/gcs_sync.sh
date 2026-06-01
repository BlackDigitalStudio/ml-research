#!/usr/bin/env bash
# Hourly mirror of recorded parquet -> GCS (additive; never deletes remote,
# so data outlives the recorder's 7-day local retention). Excludes the
# transient .parts/ working dir and half-written *.tmp files. Auth comes from
# the VM's attached service account via the metadata server (no keys).
set -euo pipefail
BUCKET="@BUCKET@"
HOST="$(hostname)"
exec gcloud storage rsync -r \
  -x '.*/\.parts/.*|.*\.tmp$' \
  /home/scalper/scalper-bot/data \
  "gs://${BUCKET}/recorder/${HOST}"
