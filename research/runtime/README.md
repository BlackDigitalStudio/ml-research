# research/runtime — the run machinery (READ ME BEFORE WRITING ANY RUN CODE)

Purpose: **stop re-deriving the backtest/run infrastructure.** Everything needed to
execute a year-scale walk-forward measurement of the h150 family (and future tiers)
lives here, battle-tested. If you are an agent about to "quickly write a runner
script" — the runner already exists in this directory. Improve it in place, with a
parity check, instead of writing a new one. Velocity and accuracy of RUNS is a
first-class asset of this project, ranked above any single strategy
(user directive 2026-07-11; see CLAUDE.md pointer).

## The two layers — never confuse them

1. **FROZEN PROTOCOL (measurement)** — byte-identified by git, never copy-edited:
   - `scripts/subs60_xgb_optuna_ic.py` — walk-forward W200/T30/EMB2, per-fold Optuna
     25 trials (A-AUC, B-IC), causal rolling tau. Parameterized ONLY via
     env/argv: `SYM LABELSUB QMIDX NTHREAD` + `SEED CFGIDX BUDGETS SAVE_PF PFTAG N_TRIALS`.
   - `scripts/subs60_build_tb3s_labels.py` — dataset builder (raw CL -> daily npz ->
     combined). Env-parameterized (see invocation table below).
   - Rust bins: `feature_builder` (master `rust_ingest`), `build_samples` +
     `grid_sim_exitdbg` (lib from branch **`claude/husdc-rev1`** + bin sources
     `scripts/build_samples_husdc.rs`, `scripts/grid_sim_exitdbg.rs` overlaid).
     Built by `bins.sh`. Proven bit-exact vs July-2026 artifacts (ledger
     tb3s-20260710_h150anch_year_xsym_PREREG_AMEND1).
     **The bin sources carry an OPT-IN entry-fill correction — see §"Entry-fill
     model" below BEFORE running any maker-EV measurement.**
   Changing ANY of these requires: preregistered amendment + one-cell byte-parity
   run old-vs-new BEFORE production (this ritual caught the H_TICKS=1500-vs-1800
   error and validated the binary rebuild — keep it).

2. **ORCHESTRATION (this dir)** — free to improve, no measurement semantics:
   - `orchestrate2.py` — seed-parallel job runner: independent (symbol, seed) jobs,
     `XSYM_JOBS="BNB:1,BTC:3,..."`, `XSYM_NTHREAD`, slot pool via `SLOTS` file
     (live-adjustable). Skip/done marker = `OPTUNA_IC_{SYM}_qm0_SEED{s}.json` in GCS.
   - `orchestrate.py` (v1) — build->combine->anch->train chains per symbol; still the
     entry point when datasets must be BUILT first. Seeds sequential (superseded by
     v2 for training; use v1 for builds, v2 for training).
   - `perseed_from_pf.py` — recomputes the per-seed json from PERFOLD artifacts
     (deterministic, <1e-7bp vs direct; makes seeds parallelizable).
   - `ens_sym.py` — the DEPLOYED-scoring ensemble cell (mean 4-seed rank score,
     majority-vote side) + LOFO + jitter sd .02/.05 (the REQUIRED perturbation gate).
   - `prep_anch_sym.py` — anchored-semantics dataset intervention (col13:=day-first,
     col44:=0).
   - `bins.sh` — builds all 3 rust binaries into a PERSISTENT dir + optional parity check.
   - `vm_provision.sh` — creates a GCP VM CORRECTLY (scopes, deps, swap).
   - `KNOWN_PITFALLS.md` — **read it; every entry cost hours once.**

## Canonical invocation (h150 cross-symbol year cell, HD3 rev8)

Dataset build (per symbol; v1 orchestrator does this + combine + anch + chain):
```
SYMF={SYM}-USDT-PERP FULLFEAT=1 H_TICKS=1800 ENTRY_MS=60000 HOLDS_S=90,150,240 \
CHASE_MS=300000 STEP_S=3 START=2025-05-09 END=2026-06-02 \
OUTSUB=research_runs/maker_labels_tb3s_h150 \
FB_BIN=$BINS/fb_target/release/feature_builder \
BS_BIN=$BINS/husdc_target/release/build_samples \
GRID_BIN=$BINS/husdc_target/release/grid_sim_exitdbg \
python3 subs60_build_tb3s_labels.py        # then again with COMBINE=1
```
**H_TICKS=1800 is the cross-symbol protocol** (not 1500 — that is the DOGE-dedicated
script's constant). Deploy-scope: `BUDGETS=5` (t5; +10 if asked), noA unreported.

Training job (one (symbol,seed)):
```
SEED={s} CFGIDX=1 BUDGETS=5 SAVE_PF=1 PFTAG=_S{s} \
python3 subs60_xgb_optuna_ic.py {SYM} maker_labels_tb3s_h150anch 0 {NTHREAD}
# then: python3 perseed_from_pf.py {SYM} {s}
```

## Entry-fill model — READ BEFORE ANY MAKER-EV NUMBER (added 2026-07-26, OPS-EXEC rev15-16)

**Every maker cell in this repo dated before 2026-07-26 was produced by a fill model
that OVER-FILLS entries. Those numbers are upper bounds, not estimates.**

The library entry (`live_sim::simulate_maker_entry`, husdc-rev1) fills a resting order
unconditionally as soon as the touch gaps past our level:
`if b.bid < level_px - eps { return FILLED }` — no flow required, queue ignored. That
collapses "the level was traded through" (fill only if flow exceeded the queue ahead)
with "the level was cancelled" (we are alone at the top of book, unfilled). The queue
model itself is honest — `--queue-mult 1.0` / `--exit-queue-mult 1.0` = always last —
the gap branch simply bypasses it.

Why it could not be fixed in place: `flow_paths.npy` is **price-agnostic** (total taker
volume per tick), so the model cannot ask "did volume trade THROUGH our level". An
intermediate patch that only demanded *some* flow in the gap tick was measured and
REJECTED (rev15: removed nothing on the USDT book, produced a false negative on a real
USDC fill).

**The correction (opt-in, both flags OFF by default — every historical cell stays
byte-reproducible):**

```
build_samples ... --emit-level-flow        # writes flow_lvl_paths.npy [ns,h,2] =
                                           # [sell vol at px <= entry_long,
                                           #  buy  vol at px >= entry_short] per tick
                                           # (prices read in a SEPARATE pass; the frozen
                                           #  aggregation path is untouched)
grid_sim_exitdbg ... --strict-entry-fill --level-flow-paths <dir>/flow_lvl_paths.npy
                                           # textbook queue rule: queue -= volume traded
                                           # through our level; fill at <=0; else MISS.
                                           # errors out if --level-flow-paths is missing
```

**Validation status (do not re-derive this):**
- reference model vs the six real live DOGEUSDC events: **6/6** (all 3 live entry misses
  miss, all 3 live fills fill); the frozen model scores 3 phantom fills there;
- compiles, binaries rebuilt, run end-to-end on a full day (DOGEUSDC 2026-07-23, 2833
  samples, deploy protocol): frozen long/short fill **0.720 / 0.742** vs strict
  **0.419 / 0.435**, and the surviving fills are worse (netl −1.78 → −3.82bp) —
  the invented fills were the favourable ones (adverse-selection signature);
- **NOT measured:** effect on τ-selected cells, on the USDT book that produced every
  historical cell, and at year scale. No cell may be re-priced before that run.

**Venue note:** the fill layer and the signal layer are different questions. Features and
scores are computed on the USDT book in both sim and live — that is correct and must not
change. Only FILLS happen on USDC. The recorder has DOGEUSDC `depth_snapshot` + `agg_trade`
(both streams needed), so the fill-realism correction can be measured on data we already
own — it does NOT require buying history.

## Sizing (measured 2026-07-10, 371d/9-10M-row dataset)
- Training job: **13-14 GB RSS**, ~2.2h at nthread=4, ~2.7h at 3, ~3.5h at 2.
  xgboost-hist scales weakly past ~4 threads. Budget RAM = 14GB x concurrent jobs + 10%.
- Day build: ~5-10s CPU + download; ~1-1.5h per 365d symbol on idle 8 vCPU.
- Combine: ~5GB RAM, minutes. Anch prep: ~4GB, minutes.
- Full 7-symbol x 4-seed campaign ~ 200 core-hours.

## Artifact layout (capture-everything; GCS bucket market-data-0998ac51)
`research_runs/maker_labels_tb3s_h150/` — dailies + combined npz (true-funding).
`research_runs/maker_labels_tb3s_h150anch/` — anchored npz + per-seed
`OPTUNA_IC_{SYM}_qm0_SEED{s}.json` + `PERFOLD_S{s}_{SYM}_qm0_f{k}.npz` +
`ENS_{SYM}_t5.json` + run logs. The shared `OPTUNA_IC_{SYM}_qm0.json` (no SEED
suffix) is a DEAD artifact under seed-parallel runs — never consume it.

## Rules that keep this from degrading
1. Record every production invocation (env + command) in the ledger record — the
   2026-07-10 H_TICKS forensics-from-journald must never be needed again.
2. Parity ritual before trusting rebuilt/refactored components (one cell, byte-compare).
3. Binaries and venvs live in HOME or a baked image — never /tmp.
4. New VM? Use `vm_provision.sh`. New run? Extend the orchestrators here.
5. Smoke-test one day / one fold before a long run (audit-before-long-runs).
