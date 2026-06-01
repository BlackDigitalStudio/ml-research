#!/usr/bin/env bash
# Chronos on-call medic. Runs as `scalper` every few minutes via timer.
# Detects an incident; if one is present (and debounce/cap/lock allow) it wakes
# a headless Claude (subscription auth in ~/.claude) to diagnose + fix, then
# records the run. NO external alerts by design.
set -uo pipefail
REPO=/home/scalper/crypto-market-recorder
DATA="$REPO/data"
HEALTH=/home/scalper/chronos.health
OC=/home/scalper/oncall
LOCK="$OC/run.lock"
STATE="$OC/state"
CHARTER=/usr/local/share/chronos-oncall-charter.md
BUCKET=recorder-data-asia-0998ac51
DEBOUNCE=900          # >=15 min between agent runs
DAILY_CAP=10          # max agent runs per UTC day
RUN_TIMEOUT=600       # 10 min hard cap per run
mkdir -p "$OC"
now=$(date +%s)
active(){ systemctl is-active --quiet chronos.service; }

# --- detect incident ---
reasons=""
active || reasons="$reasons chronos_inactive"
if active && [[ -f "$HEALTH" ]]; then
  (( now - $(stat -c %Y "$HEALTH") > 180 )) && reasons="$reasons health_stale"
fi
use=$(df --output=pcent / 2>/dev/null | tail -1 | tr -dc '0-9')
[[ -n "${use:-}" ]] && (( use > 90 )) && reasons="$reasons disk_${use}pct"
if active; then
  for v in binance_futures bybit okx bitget gateio; do
    newest=$(find "$DATA/$v" -name '*.parquet' -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
    [[ -n "$newest" ]] && (( now - newest > 600 )) && reasons="$reasons stale:$v"
  done
fi
reasons="${reasons# }"
[[ -z "$reasons" ]] && exit 0     # healthy — nothing to do

# --- guards: single-flight, debounce, daily cap ---
[[ -f "$LOCK" ]] && { logger -t chronos-oncall "incident [$reasons]; a run is in progress"; exit 0; }
day=$(date -u +%Y%m%d); sday=$day; scount=0; last=0
[[ -f "$STATE" ]] && read -r sday scount last < "$STATE" 2>/dev/null || true
[[ "$sday" != "$day" ]] && { sday=$day; scount=0; }
(( now - last < DEBOUNCE )) && { logger -t chronos-oncall "incident [$reasons]; debounced (<${DEBOUNCE}s)"; exit 0; }
(( scount >= DAILY_CAP )) && { logger -t chronos-oncall "incident [$reasons]; daily cap ${DAILY_CAP} reached"; exit 0; }

touch "$LOCK"; trap 'rm -f "$LOCK"' EXIT
ts=$(date -u +%Y%m%dT%H%M%SZ)
report="$OC/report-$ts.md"; runlog="$OC/run-$ts.log"; prompt="$OC/prompt-$ts.txt"
logger -t chronos-oncall "INCIDENT [$reasons] — waking on-call Claude ($ts)"
{ cat "$CHARTER"; printf '\n\n## Incident (detected %s UTC)\nReasons: %s\nWrite your summary to: %s\n' "$ts" "$reasons" "$report"; } > "$prompt"

timeout "$RUN_TIMEOUT" claude -p --permission-mode bypassPermissions --max-turns 40 < "$prompt" > "$runlog" 2>&1 || true

echo "$day $((scount+1)) $now" > "$STATE"
# Surface artifacts to GCS (passive — no alert).
gcloud storage cp "$runlog" "gs://$BUCKET/oncall/" >/dev/null 2>&1 || true
[[ -f "$report" ]] && gcloud storage cp "$report" "gs://$BUCKET/oncall/" >/dev/null 2>&1 || true
post=$(active && echo active || echo INACTIVE)
logger -t chronos-oncall "on-call run $ts finished; chronos=$post"
