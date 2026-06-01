#!/usr/bin/env bash
# Deploy / redeploy Chronos (recorder-v2) onto the existing recorder VM,
# replacing the legacy scalper-bot recorder. Run from the scalper-bot repo
# root. Requires the crypto-market-recorder repo checked out at $CHRONOS_SRC.
set -euo pipefail

PROJECT="project-0998ac51-36ba-445c-bc7"
BUCKET="recorder-data-asia-0998ac51"
ZONE="asia-northeast1-b"
VM="scalper-recorder"
CHRONOS_SRC="${CHRONOS_SRC:-/c/Dev/crypto-market-recorder}"
HERE="deploy/recorder-vm/chronos"

echo "1/4 build + upload app tarball (crypto-market-recorder @ HEAD)"
git -C "$CHRONOS_SRC" archive --format=tar.gz -o /tmp/chronos.tar.gz HEAD
gcloud storage cp /tmp/chronos.tar.gz "gs://$BUCKET/_bootstrap/chronos.tar.gz"

echo "2/4 build + upload project overlay tarball"
tar -czf /tmp/chronos-deploy.tar.gz -C "$HERE" \
  chronos_run.py chronos.service \
  chronos-gcs-sync.service chronos-gcs-sync.timer \
  chronos-watchdog.service chronos-watchdog.timer \
  chronos-retention.service chronos-retention.timer \
  gcs_sync.sh watchdog.sh retention.sh chronos.logrotate
gcloud storage cp /tmp/chronos-deploy.tar.gz "gs://$BUCKET/_bootstrap/chronos-deploy.tar.gz"

echo "3/4 point VM startup-script at the chronos variant"
gcloud compute instances add-metadata "$VM" --project="$PROJECT" --zone="$ZONE" \
  --metadata-from-file=startup-script="$HERE/startup.sh"

echo "4/4 run startup now (installs + swaps recorder)"
gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --quiet \
  --command 'sudo google_metadata_script_runner startup'

echo "done. verify:"
echo "  gcloud compute ssh $VM --zone $ZONE --command 'systemctl status chronos --no-pager | head; tail -n 30 /home/scalper/crypto-market-recorder/logs/chronos.log'"
