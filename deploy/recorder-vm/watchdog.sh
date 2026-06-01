#!/usr/bin/env bash
# External liveness watchdog. The recorder updates the mtime of its health
# file on every successful 15s flush; main.py's in-process watchdog already
# exits (-> systemd Restart=always) if both BTC streams die. This catches the
# rarer case where the process is alive but wedged (no flush) — if the health
# file is stale while the unit is still "active", force a restart.
set -euo pipefail
HEALTH="/tmp/scalper_recorder_health"
MAX_AGE=120  # seconds; flush interval is 15s, so 120s = 8 missed flushes

systemctl is-active --quiet scalper-recorder.service || exit 0
[[ -f "$HEALTH" ]] || exit 0

age=$(( $(date +%s) - $(stat -c %Y "$HEALTH") ))
if (( age > MAX_AGE )); then
  logger -t scalper-watchdog "health stale ${age}s > ${MAX_AGE}s — restarting recorder"
  systemctl restart scalper-recorder.service
fi
