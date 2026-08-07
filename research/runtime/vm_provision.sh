#!/bin/sh
# Create a research VM CORRECTLY (see KNOWN_PITFALLS: scopes, /tmp, RAM sizing).
# Usage: vm_provision.sh <name> <project> <machine-type> [zone=europe-west1-b]
# Sizing: 14GB RAM per concurrent training job. Bucket market-data-0998ac51 is in
# EUROPE-WEST1 — same-region VM = zero egress. If the project is not the bucket's,
# grant its compute SA objectAdmin on the bucket (one-time, from the bucket owner):
#   gcloud storage buckets add-iam-policy-binding gs://market-data-0998ac51 \
#     --member=serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com \
#     --role=roles/storage.objectAdmin
set -e
NAME=${1:?name}; PROJ=${2:?project}; MT=${3:?machine-type}; ZONE=${4:-europe-west1-b}

gcloud compute instances create "$NAME" --project="$PROJ" --zone="$ZONE" \
  --machine-type="$MT" --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --scopes=cloud-platform          # <- REQUIRED: default scopes are storage read-only

sleep 20
gcloud compute ssh "$NAME" --project="$PROJ" --zone="$ZONE" --command='
  sudo apt-get -qq update && sudo apt-get -qq install -y python3-pip git &&
  pip3 install -q --break-system-packages numpy==2.4.6 xgboost==3.2.0 optuna==4.8.0 \
      google-cloud-storage==3.10.1 scipy==1.17.1 pyarrow &&
  if ! sudo /usr/sbin/swapon --show 2>/dev/null | grep -q swapfile; then
    sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile &&
    sudo /usr/sbin/swapon /swapfile &&
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi &&
  python3 -c "import xgboost,optuna,numpy; print(\"deps:\", xgboost.__version__, optuna.__version__, numpy.__version__)" &&
  echo test | gsutil cp - gs://market-data-0998ac51/research_runs/_provision_write_test.txt &&
  gsutil rm gs://market-data-0998ac51/research_runs/_provision_write_test.txt && echo BUCKET-WRITE-OK'
echo "[vm_provision DONE] $NAME ($MT, $ZONE). Next: bins.sh if building datasets; launch runners via systemd-run -p OOMPolicy=continue."
