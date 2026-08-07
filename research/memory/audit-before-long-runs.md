---
name: audit-before-long-runs
description: "Before launching any long/expensive run, statically audit the code for bugs first — don't discover bugs mid-run and relaunch in a loop"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0fb2797-9e8b-421d-8f45-0ea10e0801e7
---

Before kicking off any long-running or expensive process (multi-hour build, full
training, large sweep), do a thorough **static audit for bugs and inefficiencies
FIRST** — read the code, trace shapes/indices/timing/memory, run a tiny smoke if
cheap. Do NOT fall into the reactive cycle of "launch → wait hours → discover a
bug/slowness → kill → fix → relaunch → repeat." That cycle burns money and time
(the project's actual cost).

**Why:** the user called this out after I (a) launched the 8-symbol maker-label
build, hit disk-IO thrash mid-run, killed + relaunched; then (b) launched full
XGBoost training, found it was ~8x too slow (nthread=4 + full-data Optuna search)
~40min in, and started another kill/patch/relaunch. Each restart wasted the
already-spent compute.

**How to apply:** treat a launch as a commitment. Pre-launch checklist: correctness
(shapes, indices, alignment, leakage, timestamps), resource scope (per-unit timing
probe → total wall-clock + RAM + disk), parallelism/IO contention, and that ALL
needed outputs are captured (see [[capture-all-information]]). Only launch once the
audit passes. Scope cost and get sign-off for real cost-gate tiers (CLAUDE.md).
