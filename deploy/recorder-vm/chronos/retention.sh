#!/usr/bin/env bash
# Local retention for Chronos (which has no built-in rotation). Data is
# mirrored to GCS hourly and kept there indefinitely; locally we keep only a
# short buffer. Deletes compacted parquet + raw archive older than RETENTION_DAYS.
# The transient .parts/ (always recent) and current-hour files are never old
# enough to match, so they are untouched.
set -euo pipefail
DATA="/home/scalper/crypto-market-recorder/data"
RETENTION_DAYS=3

[[ -d "$DATA" ]] || exit 0
n=$(find "$DATA" -type f \( -name '*.parquet' -o -name '*.jsonl.gz' \) -mtime +${RETENTION_DAYS} -print -delete 2>/dev/null | wc -l)
find "$DATA" -type d -empty -delete 2>/dev/null || true
[[ "$n" -gt 0 ]] && logger -t chronos-retention "pruned ${n} files older than ${RETENTION_DAYS}d"
exit 0
