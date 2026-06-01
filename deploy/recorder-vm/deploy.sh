#!/usr/bin/env bash
# One-shot provision of the 24/7 recorder VM (Tokyo) + GCS bucket.
# Run from the repo root with gcloud authed to the target project. Idempotent
# enough to re-run: bucket create + IAM are no-ops if already present, and
# re-running rebuilds + re-uploads the bootstrap tarball.
#
# To redeploy code to an EXISTING VM: re-run steps 3 (tarball) then
#   gcloud compute ssh $VM --zone $ZONE --command 'sudo google_metadata_script_runner startup'
set -euo pipefail

PROJECT="project-0998ac51-36ba-445c-bc7"
BUCKET="recorder-data-asia-0998ac51"
REGION="asia-northeast1"
ZONE="asia-northeast1-b"
VM="scalper-recorder"
SA="916020991759-compute@developer.gserviceaccount.com"

echo "1/4 bucket gs://$BUCKET ($REGION)"
gcloud storage buckets create "gs://$BUCKET" \
  --project="$PROJECT" --location="$REGION" \
  --uniform-bucket-level-access 2>/dev/null || echo "  (exists)"

echo "2/4 grant $SA objectAdmin on bucket"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SA" --role="roles/storage.objectAdmin" >/dev/null

echo "3/4 build + upload repo tarball"
git archive --format=tar.gz -o /tmp/scalper-bot.tar.gz HEAD
gcloud storage cp /tmp/scalper-bot.tar.gz "gs://$BUCKET/_bootstrap/scalper-bot.tar.gz"

echo "4/4 create VM $VM ($ZONE, e2-small)"
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=e2-small \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard \
  --service-account="$SA" \
  --scopes=storage-rw \
  --metadata-from-file=startup-script=deploy/recorder-vm/startup.sh

echo "done. tail startup with:"
echo "  gcloud compute ssh $VM --zone $ZONE --command 'sudo tail -f /var/log/scalper-startup.log'"
