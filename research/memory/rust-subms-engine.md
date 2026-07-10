---
name: rust-subms-engine
description: "axb_engine (Rust) LIVE since 2026-07-09 09:13 UTC — bit-exact decision path vs anchored validation (harness-proven), ~70µs compute; units axb-engine-doge + axb-exec-doge (Python order sidecar via Unix socket); EV(latency) flat 0–3000ms so the engine's value is parity-by-construction, not EV"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2fba63ea-658b-4410-928d-a8c01aff03c4
---

**Deployed** 2026-07-09 09:13 UTC on scalper-recorder VM, replacing the Python
perfect-parity engine. Units: `axb-engine-doge` (Rust, `/home/delmi/axb_engine`,
WORKDIR /home/delmi/axb_h150, boot via `axb_boot.py`) + `axb-exec-doge`
(`axb_exec.py`: verbatim maker trade lifecycle + own DOGEUSDC bookTicker WS +
hourly GCS decision upload; Unix socket WORKDIR/exec.sock). Old `axb-live-doge`
disabled. Binary also at gs://market-data-0998ac51/research_runs/deploy_h150/bin/.

**Bit-exactness (harness-proven on day 20260707, commits 461bcbd/3604b9b/b080bea):**
- `features_incr::FeatState`: day-anchored append-only prefixes reproduce the batch
  builders' float summation order → 0/2.03M cells vs frozen feature_builder;
  compute64 p50 1.9µs. Event-commit contract: trades/eth/liq/oi with ts ≤ tick
  pushed before the tick; tick_intensity deferred-drained; rolling windows extend
  when their last tick is final.
- `gbt`: 0/228k predictions vs xgboost. Keys: f32 margin accumulates base-margin
  FIRST then leaves in tree order; xgboost's base can be 1 ulp off the float
  ProbToMargin formula → boot solves exact bits from a one-tree prediction;
  sigmoid = glibc expf (== Rust f32::exp; numpy's SIMD exp does NOT match);
  leaf value lives in split_conditions.
- ensemble score 0/28546; tau = np.quantile-linear port, matched Python to 6dp.

**Semantics:** midnight-anchored 3s grid, FeatState reset at UTC midnight (== per-day
sim files); mid-day restart = 400s span gate for that partial day only (state lacks
the day's earlier history until next midnight). Funding day-anchor via boot
(recorder hour-00 file, REST fallback). Live-vs-live shadow parity vs Python engine:
take5 74/74, score |Δ| p50 0.007 (independent WS jitter — expected; byte parity is
only definable vs the validation pipeline, which the harnesses cover).

**How to apply:** future validations of this policy = FUNDING_MODE=anchor +
midnight grid (subs60_recorder_ev_h150). Any change to features/models must re-pass
fb_incr_harness + score_harness byte-equality before deploy. EV(latency) measured
flat 0–3000ms (ledger 2026-07-09) — don't sell latency work as EV.
Related: [[h150-sim-live-parity]], [[live-trading-deploy]], [[capture-all-information]].
