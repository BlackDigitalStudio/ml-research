#!/usr/bin/env bash
# GCE startup-script for the Chronos (recorder-v2) deployment. Replaces the
# legacy scalper-bot recorder on the same VM. Idempotent (runs on every boot
# and on-demand via `google_metadata_script_runner startup`).
#
# Pulls two tarballs from GCS (built by deploy.sh):
#   _bootstrap/chronos.tar.gz        — crypto-market-recorder repo (the app)
#   _bootstrap/chronos-deploy.tar.gz — project overlay (entrypoint, units, scripts)
set -euo pipefail
exec > >(tee -a /var/log/chronos-startup.log) 2>&1
echo "=== chronos startup $(date -u) ==="

BUCKET="recorder-data-asia-0998ac51"
SVCUSER="scalper"
REPO="/home/${SVCUSER}/crypto-market-recorder"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# libjemalloc2: heap-fragmentation control (LD_PRELOAD in the unit).
# logrotate: ships in base image but ensure present.
apt-get install -y python3-venv python3-pip cloud-guest-utils libjemalloc2 logrotate

# Grow the root filesystem if the boot disk was resized (idempotent no-op
# when already at full size). Root is /dev/sda1 on these pd-standard images.
growpart /dev/sda 1 2>/dev/null || true
resize2fs /dev/sda1 2>/dev/null || true

id -u "$SVCUSER" &>/dev/null || useradd -m -s /bin/bash "$SVCUSER"

# --- stop the legacy recorder (if present) ---
systemctl disable --now scalper-recorder.service scalper-gcs-sync.timer scalper-watchdog.timer 2>/dev/null || true

# --- fetch app + overlay ---
mkdir -p "$REPO"
gcloud storage cp "gs://${BUCKET}/_bootstrap/chronos.tar.gz" /tmp/chronos.tar.gz
gcloud storage cp "gs://${BUCKET}/_bootstrap/chronos-deploy.tar.gz" /tmp/chronos-deploy.tar.gz
tar -xzf /tmp/chronos.tar.gz -C "$REPO"
rm -rf /tmp/chronos-deploy && mkdir -p /tmp/chronos-deploy
tar -xzf /tmp/chronos-deploy.tar.gz -C /tmp/chronos-deploy
# Defensive: strip any CRLF that survived a Windows-built tarball — a CRLF
# shebang/path would break scripts and the logrotate config on Linux.
sed -i 's/\r$//' /tmp/chronos-deploy/* 2>/dev/null || true

# overlay project entrypoint at repo root
install -m 0644 /tmp/chronos-deploy/chronos_run.py "$REPO/chronos_run.py"
mkdir -p "$REPO"/{data,logs}
chown -R "$SVCUSER:$SVCUSER" "/home/${SVCUSER}"

# --- venv + deps (chronos needs aiohttp orjson pyarrow pandas) ---
if [[ ! -x "$REPO/venv/bin/python" ]]; then
  sudo -u "$SVCUSER" bash -lc "cd '$REPO' && python3 -m venv venv && \
    ./venv/bin/pip install --upgrade pip wheel && \
    ./venv/bin/pip install aiohttp orjson pyarrow pandas"
fi

# --- config.env ---
cat > "$REPO/config.env" <<EOF
CHRONOS_ROOT=${REPO}/data
RECORDER_HOST_ID=chronos-tokyo
CHRONOS_HEALTH_FILE=/home/${SVCUSER}/chronos.health
EOF
chown "$SVCUSER:$SVCUSER" "$REPO/config.env"

# --- on-call agent runtime: Node + Claude Code CLI (idempotent) ---
# Subscription creds live at /home/scalper/.claude/.credentials.json (placed
# out-of-band, NOT in this tarball). If absent, the on-call agent simply can't
# authenticate — harmless; recording is unaffected.
if ! command -v claude >/dev/null 2>&1; then
  apt-get install -y nodejs npm
  npm install -g @anthropic-ai/claude-code || echo "WARN: claude CLI install failed"
fi

# --- helper scripts ---
install -m 0755 /tmp/chronos-deploy/gcs_sync.sh  /usr/local/bin/chronos-gcs-sync
install -m 0755 /tmp/chronos-deploy/watchdog.sh  /usr/local/bin/chronos-watchdog
install -m 0755 /tmp/chronos-deploy/retention.sh /usr/local/bin/chronos-retention
install -m 0755 /tmp/chronos-deploy/oncall.sh    /usr/local/bin/chronos-oncall
install -m 0644 /tmp/chronos-deploy/oncall-charter.md /usr/local/share/chronos-oncall-charter.md
sed -i "s|@BUCKET@|${BUCKET}|g" /usr/local/bin/chronos-gcs-sync

# on-call sudoers (scoped to chronos.service mgmt) — validate before installing
if visudo -c -f /tmp/chronos-deploy/chronos-oncall.sudoers >/dev/null 2>&1; then
  install -m 0440 /tmp/chronos-deploy/chronos-oncall.sudoers /etc/sudoers.d/chronos-oncall
else
  echo "WARN: oncall sudoers failed visudo check — not installed"
fi

# --- log rotation (chronos.log + startup log grow unbounded otherwise) ---
install -m 0644 /tmp/chronos-deploy/chronos.logrotate /etc/logrotate.d/chronos

# --- systemd units ---
install -m 0644 /tmp/chronos-deploy/chronos.service             /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-gcs-sync.service    /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-gcs-sync.timer      /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-watchdog.service    /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-watchdog.timer      /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-retention.service   /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-retention.timer     /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-oncall.service      /etc/systemd/system/
install -m 0644 /tmp/chronos-deploy/chronos-oncall.timer        /etc/systemd/system/

systemctl daemon-reload
systemctl enable --now chronos.service
systemctl enable --now chronos-gcs-sync.timer
systemctl enable --now chronos-watchdog.timer
systemctl enable --now chronos-retention.timer
systemctl enable --now chronos-oncall.timer
# Pick up an edited entrypoint / symbol set on re-run.
systemctl restart chronos.service

echo "=== chronos startup complete $(date -u) ==="
