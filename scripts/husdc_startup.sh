#!/bin/bash
# HUSDC run1 VM startup: install deps, pull analyzer, smoke then full, upload
# results to GCS, then SELF-DELETE (leaves no running/stopped VM or disk).
set -x
set -o pipefail
exec > >(tee /var/log/husdc.log) 2>&1
export HOME=/root
export PATH=/usr/local/bin:/usr/bin:/bin:/sbin:/snap/bin
GS="gs://market-data-0998ac51/research_runs/husdc"

self_destruct(){
  echo "[startup] uploading log + self-destruct"
  gsutil -q cp /var/log/husdc.log "$GS/husdc_startup.log" 2>/dev/null || true
  NAME=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
  ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone \
    | awk -F/ '{print $NF}')
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet \
    || shutdown -h now
}
trap 'rc=$?; echo "[startup] EXIT rc=$rc"; self_destruct' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y python3-pip || true
pip3 install --break-system-packages -q numpy pandas requests \
  google-cloud-storage \
  || pip3 install -q numpy pandas requests google-cloud-storage || true

cd /root
gsutil -q cp "$GS/husdc_run.py" /root/husdc_run.py

echo "[startup] ===== SMOKE ====="
timeout 600 python3 /root/husdc_run.py --smoke
echo "[startup] smoke rc=$?"

echo "[startup] ===== FULL (run2: matched-freshness label) ====="
# lightest-first (DOGE,SOL,ETH,BTC) + incremental GCS saves; 14d/4 workers/16GB.
timeout 3600 python3 /root/husdc_run.py \
  --symbols DOGE SOL ETH BTC --start 2026-05-01 --end 2026-05-14 \
  --workers 4 --tag run2 --fresh-tol-s 1.0
echo "[startup] full rc=$?"
echo "[startup] complete"
# trap (EXIT) uploads the log and self-deletes the VM.
