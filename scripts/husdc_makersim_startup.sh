#!/bin/bash
# HUSDC maker-sim VM startup: deps, pull husdc_run.py + husdc_makersim.py,
# smoke then full (USDT, adverse-selection prototype), upload, SELF-DELETE.
set -x
set -o pipefail
exec > >(tee /var/log/husdc_maker.log) 2>&1
export HOME=/root
export PATH=/usr/local/bin:/usr/bin:/bin:/sbin:/snap/bin
GS="gs://market-data-0998ac51/research_runs/husdc"

self_destruct(){
  echo "[startup] upload log + self-destruct"
  gsutil -q cp /var/log/husdc_maker.log "$GS/husdc_maker_startup.log" 2>/dev/null || true
  NAME=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
  ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || shutdown -h now
}
trap 'rc=$?; echo "[startup] EXIT rc=$rc"; self_destruct' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y python3-pip || true
pip3 install --break-system-packages -q numpy pandas requests google-cloud-storage \
  || pip3 install -q numpy pandas requests google-cloud-storage || true

cd /root
gsutil -q cp "$GS/husdc_run.py" /root/husdc_run.py
gsutil -q cp "$GS/husdc_makersim.py" /root/husdc_makersim.py

echo "[startup] ===== SMOKE ====="
timeout 900 python3 /root/husdc_makersim.py --smoke --stride 5
echo "[startup] smoke rc=$?"

echo "[startup] ===== FULL (maker-fill / adverse-selection, USDT) ====="
timeout 3600 python3 /root/husdc_makersim.py \
  --symbols DOGE SOL ETH BTC --start 2026-05-01 --end 2026-05-14 \
  --workers 4 --stride 5 --tag maker1
echo "[startup] full rc=$?"
echo "[startup] complete"
