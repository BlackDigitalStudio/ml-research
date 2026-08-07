---
name: verify-no-duplicate-vm-runs
description: "Before/after launching a long VM run, check for a duplicate of the same process — ScheduleWakeup/chained launchers can double-fire"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6287b103-b1f8-4152-bb04-df2d7ec2e6cf
---

When launching a long compute run on the GCP VM (e.g. `hd2-feats-003`), always
`ps aux | grep <script>` to confirm exactly ONE instance is running. A launcher
fired from a ScheduleWakeup (or a manual launch overlapping a wakeup) can start
the **same command twice** — both grind the same grid, oversubscribe the 8-vCPU
box (load ≫ cores → each ~2x slower), append to the same `>`-truncated log, and
clobber the same output JSON.

**Why:** happened 2026-06-01 — `subs60_xgb_b2_grid.py` (94966-config grid) was
launched twice (16:06 + 16:09), wasting ~50 min of oversubscribed compute. Fix
was to kill one duplicate (the survivor instantly ~2x'd and advanced a symbol).

**How to apply:** check duplicates at launch AND when inheriting an in-flight
run from a handoff. To resolve: kill the later/redundant PID, keep the most-
progressed one (don't restart — lose progress). Confirm no cron/at re-fire risk
(`crontab -l`, `atq`). Note many of these grid scripts upload the result JSON
**only at the end** but `log()` per-symbol — so per-symbol numbers survive a
crash via the log even if the JSON never gets written. See [[audit-before-long-runs]].
