#!/usr/bin/env bash
# External liveness watchdog for Chronos. The recorder rewrites its health
# file mtime on every 15s flush. If it goes stale while the unit is still
# "active", the process is wedged — force a restart.
set -euo pipefail
HEALTH="/home/scalper/chronos.health"
MAX_AGE=120  # 8 missed 15s flushes

systemctl is-active --quiet chronos.service || exit 0
[[ -f "$HEALTH" ]] || exit 0

age=$(( $(date +%s) - $(stat -c %Y "$HEALTH") ))
if (( age > MAX_AGE )); then
  logger -t chronos-watchdog "health stale ${age}s > ${MAX_AGE}s — restarting chronos"
  systemctl restart chronos.service
fi
