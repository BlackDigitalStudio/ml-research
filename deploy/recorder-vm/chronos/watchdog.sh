#!/usr/bin/env bash
# Liveness + disk + sync watchdog for Chronos. Runs every 2 min via timer.
# Three independent guards (each isolated so one failing never skips the rest).
set -uo pipefail
DATA="/home/scalper/crypto-market-recorder/data"
HEALTH="/home/scalper/chronos.health"
SYNC="/home/scalper/.last_gcs_sync"
HEALTH_MAX=120          # 8 missed 15s flushes -> process wedged
SYNC_MAX=21600          # 6h with no successful GCS sync -> data-loss risk
DISK_PCT_MAX=90         # emergency prune threshold

# 1) Disk guard — backstop against any fill cause (retention lag, sync backlog).
use=$(df --output=pcent / 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$use" ]] && (( use > DISK_PCT_MAX )); then
  logger -t chronos-watchdog "CRITICAL disk ${use}% > ${DISK_PCT_MAX}% — emergency prune (>1 day)"
  find "$DATA" -type f \( -name '*.parquet' -o -name '*.jsonl.gz' \) -mtime +1 -delete 2>/dev/null || true
fi

# 2) GCS sync staleness — surfaces a silent sync failure before retention
#    (3 days) would delete un-uploaded data.
if [[ -f "$SYNC" ]]; then
  sage=$(( $(date +%s) - $(cat "$SYNC" 2>/dev/null || echo 0) ))
  if (( sage > SYNC_MAX )); then
    logger -t chronos-watchdog "CRITICAL gcs sync stale ${sage}s (>${SYNC_MAX}s) — data-loss risk vs retention"
  fi
fi

# 3) Recorder liveness — restart if health file is stale while unit is active.
if systemctl is-active --quiet chronos.service && [[ -f "$HEALTH" ]]; then
  age=$(( $(date +%s) - $(stat -c %Y "$HEALTH") ))
  if (( age > HEALTH_MAX )); then
    logger -t chronos-watchdog "health stale ${age}s > ${HEALTH_MAX}s — restarting chronos"
    systemctl restart chronos.service
  fi
fi
exit 0
