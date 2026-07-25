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

## exec v3 — slot-pool concurrency (measured-policy parity, 2026-07-26)

Every validation cell (year walk-forward PERFOLD, recorder-EV deploy gates)
scores EVERY above-tau 3s-grid decision as an independent trade with its own
150s hold, overlaps included. The v2 exec's single-position `busy` flag
silently dropped each signal arriving mid-trade: measured live (DOGE FIXQ t10,
0716–0725) 89/104 budget signals dropped (85.6%); lock-sim on the gate window
keeps ~4% of the measured bpd and takes the systematically worst trade of each
signal cluster (cluster-first EV +2.9bp vs cluster-rest +10.3bp).

`axb_exec.py` v3 replaces `busy` with a slot pool: same-side signals stack up
to `MAX_CONC` independent trades (each exits its OWN quantity reduce-only);
opposite-side signals are skipped while any trade is open (one-way position
mode; measured opposite overlap DOGE 4.3% / XRP 0%). `MAX_CONC=1` (unit
default) preserves the legacy behavior and legacy avail-based sizing;
`MAX_CONC>1` sizes each trade to wallet×SIZE_FRAC×NOTIONAL_MULT/MAX_CONC —
keep that ≥ ~5.5 USDC (minNotional 5 + haircut/floor) or trades skip_small.
Fidelity to the measured policy (recev lock-sim, busy=155s empirical): DOGE
kept bpd 4% at M=1 → 85% at M=8; XRP 23% → 98% at M=8. Offline test:
`python live/test_axb_exec_slots.py`. Unifies the per-VM forks: ETH's
LEVERAGE/NOTIONAL_MULT envs + ETHUSDC/XRPUSDC precision maps + per-symbol
minNotional (ETH 20) now live in the one canonical file.
