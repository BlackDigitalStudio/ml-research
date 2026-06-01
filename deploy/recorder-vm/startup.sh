#!/usr/bin/env bash
# GCE startup-script for the 24/7 market-data recorder VM (Tokyo).
#
# Passed to the instance via `--metadata-from-file startup-script=`.
# Runs as root on EVERY boot — must be idempotent. It:
#   1. installs OS deps (python venv, jemalloc),
#   2. creates the `scalper` user,
#   3. pulls the repo tarball from GCS (built by deploy.sh),
#   4. builds the venv + installs requirements,
#   5. writes config.env (public-data-only Binance creds — recorder needs
#      no signed calls; the user-data stream self-disables on PermissionError),
#   6. installs + starts the recorder, the hourly GCS-sync timer, and the
#      health-watchdog timer.
#
# The VM's attached service account (metadata) provides GCS auth — no keys
# on disk. That SA must hold roles/storage.objectAdmin on $BUCKET.
set -euo pipefail
exec > >(tee -a /var/log/scalper-startup.log) 2>&1
echo "=== scalper startup $(date -u) ==="

BUCKET="recorder-data-asia-0998ac51"
SVCUSER="scalper"
REPO_DIR="/home/${SVCUSER}/scalper-bot"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip libjemalloc2

id -u "$SVCUSER" &>/dev/null || useradd -m -s /bin/bash "$SVCUSER"

# --- fetch repo (idempotent: always refresh to latest bootstrap tarball) ---
mkdir -p "$REPO_DIR"
gcloud storage cp "gs://${BUCKET}/_bootstrap/scalper-bot.tar.gz" /tmp/sb.tar.gz
tar -xzf /tmp/sb.tar.gz -C "$REPO_DIR"
mkdir -p "$REPO_DIR"/{data,models,logs}
chown -R "$SVCUSER:$SVCUSER" "/home/${SVCUSER}"

# --- venv + deps (skip rebuild if already present) ---
if [[ ! -x "$REPO_DIR/venv/bin/python" ]]; then
  sudo -u "$SVCUSER" bash -lc "cd '$REPO_DIR' && python3 -m venv venv && \
    ./venv/bin/pip install --upgrade pip wheel && \
    ./venv/bin/pip install -r requirements.txt"
fi

# --- config.env (recorder uses only PUBLIC market streams + public REST) ---
cat > "$REPO_DIR/config.env" <<EOF
BINANCE_API_KEY=public_data_only
BINANCE_API_SECRET=public_data_only
SYMBOL=BTCUSDT
SECONDARY_SYMBOL=ETHUSDT
DATA_DIR=${REPO_DIR}/data
MODEL_DIR=${REPO_DIR}/models
LOG_DIR=${REPO_DIR}/logs
EOF
chown "$SVCUSER:$SVCUSER" "$REPO_DIR/config.env"

# --- install helper scripts ---
install -m 0755 "$REPO_DIR/deploy/recorder-vm/gcs_sync.sh"  /usr/local/bin/scalper-gcs-sync
install -m 0755 "$REPO_DIR/deploy/recorder-vm/watchdog.sh"  /usr/local/bin/scalper-watchdog
sed -i "s|@BUCKET@|${BUCKET}|g" /usr/local/bin/scalper-gcs-sync

# --- install systemd units ---
cp "$REPO_DIR/systemd/scalper-recorder.service"              /etc/systemd/system/
cp "$REPO_DIR/deploy/recorder-vm/scalper-gcs-sync.service"   /etc/systemd/system/
cp "$REPO_DIR/deploy/recorder-vm/scalper-gcs-sync.timer"     /etc/systemd/system/
cp "$REPO_DIR/deploy/recorder-vm/scalper-watchdog.service"   /etc/systemd/system/
cp "$REPO_DIR/deploy/recorder-vm/scalper-watchdog.timer"     /etc/systemd/system/

systemctl daemon-reload
systemctl enable --now scalper-recorder.service
systemctl enable --now scalper-gcs-sync.timer
systemctl enable --now scalper-watchdog.timer

echo "=== scalper startup complete $(date -u) ==="
