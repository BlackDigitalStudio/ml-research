---
name: research-runtime-infra
description: "research/runtime/ in-repo is the canonical run machinery (orchestrators, bins.sh, vm_provision.sh, KNOWN_PITFALLS) — extend it, never write ad-hoc runners; research throughput ranks above any single strategy"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e405d5fd-5d12-437d-a431-d396bf143b9a
---

User directive (2026-07-11): research velocity+accuracy is a level ABOVE any single
algorithm — "деньги у того кто исследует, проверяет и разрабатывает постоянно".
Agents kept rewriting run code each session, re-hitting the same bugs.

**Why:** every rewrite re-pays the debugging tax (H_TICKS forensics, /tmp binary
wipes, GCE scopes, OOM sizing…) and degrades instead of compounding.

**How to apply:** before ANY run/backtest orchestration work, read
`research/runtime/README.md` + `KNOWN_PITFALLS.md` (in-repo, committed 2026-07-11,
also bound in CLAUDE.md). Improve those runners in place; parity ritual (one-cell
byte-compare) for anything touching the frozen measurement layer; record every
production invocation (env+command) in the ledger. Related: [[xsym-cross-symbol-run]],
[[audit-before-long-runs]].
