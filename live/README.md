# live/ — deployed trading units (DOGEUSDC maker, anchored h150 ensemble)

Deployed state (since 2026-07-09, RESEARCH_LOG s22): Rust `axb_engine` decides,
Python sidecar executes. Runs on the recorder VM (`scalper-recorder`, Tokyo).

| File | Role |
|---|---|
| `axb-engine-doge.service` | LIVE unit: `axb_engine` (Rust, ~70µs decision path) — book mirror, day-anchored features, 4-seed bit-exact XGBoost ensemble, causal tau, decision JSONL. |
| `axb-exec-doge.service` + `axb_exec.py` | LIVE unit: maker trade lifecycle (GTX entry at touch, 60s window, hold 150s from fill, pegged reduce-only exit chase, taker only at catastrophic guards), own DOGEUSDC bookTicker WS, hourly GCS decision upload. |
| `axb_boot.py` | ExecStartPre: GCS model bundle → npys, empirically-solved xgboost base-margin bits, tau seed from anchored recorder scores, funding day-anchor (local file → recorder GCS → REST fallback). |
| `axb_live.py`, `axb-live*.service`, `axb-shadow*.service`, `axb_shadow_eval.py` | The Python engine generation (pre-Rust) and shadow-mode units — superseded by the Rust engine but kept: the shadow harness measured the 74/74 take-agreement that qualified the Rust engine. |

Policy definition (the ANCHORED cell, validated at year scale): funding col13
frozen at the day's first mark-price value, col44 = 0; budget t5; ensemble = mean
of 4 per-seed rank scores, side = mean pBg ≥ 0.5. Do NOT "fix" the anchored
funding semantics to true funding — the true-funding variant measured negative
on the same days (ledger 2026-07-08 forensics; it is a policy definition, not a bug).

Any change to models/features here must re-pass BOTH parity harnesses
(`rust_ingest`: `fb_incr_harness`, `score_harness`) before deploy. Keys/config
live on the VM (`config.env`), never in the repo.
