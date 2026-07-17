# research/runtime — the run machinery (READ ME BEFORE WRITING ANY RUN CODE)

Purpose: **stop re-deriving the backtest/run infrastructure.** Everything needed to
execute a year-scale walk-forward measurement of the h150 family (and future tiers)
lives here, battle-tested. If you are an agent about to "quickly write a runner
script" — the runner already exists in this directory. Improve it in place, with a
parity check, instead of writing a new one. Run velocity/accuracy is an instrument
of the real objective — speed of alpha discovery and strategy development, which
ranks above any single static strategy (user directive 2026-07-11, clarified
2026-07-15; see CLAUDE.md).

## The two layers — never confuse them

1. **FROZEN PROTOCOL (measurement)** — byte-identified by git, never copy-edited:
   - **`scripts/subs60_xgb_sobol_v2.py` — PROTOCOL v2, THE DEFAULT for new tiers**
     (adopted 2026-07-16; ledger xgb-sobol-v2-BRIDGE + AMEND2; **v2.1 md5
     2a435f32e2517479e0232260c85aad1a** — adds MODEL_DUMP=1 default: per-fold
     boosters+HP saved always, non-perturbation bit-proven in rev16). Same measurement semantics as v1; search =
     deterministic seeded Sobol (25 pts, same ranges), QuantileDMatrix finals +
     inplace_predict (isolation-proven RESULT-IDENTICAL to v1's matrix path),
     ~7.5GB RSS/job, x2.4-2.7 faster/job; FOLD_PAR=6 bit-equal to sequential.
     Extra env: `SOBOL_PAR` (trial threads, dflt 6), `OUT_SUB` (artifact subdir),
     `DATA_CACHE` (local npz cache), `SEARCH_MODE=tpe` (isolation mode only).
     **RULES: (i) v2 cells are ledger-tagged v2 and NEVER compared to v1 cells on
     seed-sd — v2 removes the TPE-trajectory noise that dominated v1 seed variance
     (measured: BTC h150d sd 6.32 -> 1.80, ens unchanged +12.27 <-> +12.59); mean-sd
     seed-gates need v2 recalibration; cross-protocol comparisons at ensemble/
     conclusion level only. (ii) v1 stays frozen for v1-comparable reruns.**
   - `scripts/subs60_xgb_optuna_ic.py` — PROTOCOL v1 (walk-forward W200/T30/EMB2,
     per-fold sequential Optuna TPE 25 trials (A-AUC, B-IC), causal rolling tau).
     Parameterized ONLY via env/argv: `SYM LABELSUB QMIDX NTHREAD` + `SEED CFGIDX
     BUDGETS SAVE_PF PFTAG N_TRIALS`.
   - `scripts/subs60_build_tb3s_labels.py` — dataset builder (raw CL -> daily npz ->
     combined). Env-parameterized (see invocation table below).
   - Rust bins: `feature_builder` (master `rust_ingest`), `build_samples` +
     `grid_sim_exitdbg` (lib from branch **`claude/husdc-rev1`** + bin sources
     `scripts/build_samples_husdc.rs`, `scripts/grid_sim_exitdbg.rs` overlaid).
     Built by `bins.sh`. Proven bit-exact vs July-2026 artifacts (ledger
     tb3s-20260710_h150anch_year_xsym_PREREG_AMEND1).
   Changing ANY of these requires: preregistered amendment + one-cell byte-parity
   run old-vs-new BEFORE production (this ritual caught the H_TICKS=1500-vs-1800
   error and validated the binary rebuild — keep it).

2. **ORCHESTRATION (this dir)** — free to improve, no measurement semantics:
   - `orchestrate2.py` — seed-parallel job runner: independent (symbol, seed) jobs,
     `XSYM_JOBS="BNB:1,BTC:3,..."`, `XSYM_NTHREAD`, slot pool via `SLOTS` file
     (live-adjustable). Skip/done marker = `OPTUNA_IC_{SYM}_qm0_SEED{s}.json` in GCS.
   - `orchestrate.py` (v1) — build->combine->anch->train chains per symbol; still the
     entry point when datasets must be BUILT first. Seeds sequential (superseded by
     v2 for training; use v1 for builds, v2 for training). `XSYM_BUILD_SHARDS=K`
     runs K sharded instances of the frozen builder per symbol (PARITY/NSHARD; day
     sets disjoint) — near-linear build speedup, ~6GB disk churn per shard workdir.
     Subdir/param envs: XSYM_SUB_H, XSYM_SUB_A, XSYM_H_TICKS, XSYM_TRAIN=0, XSYM_XD.
   - `perseed_from_pf.py` — recomputes the per-seed json from PERFOLD artifacts
     (deterministic, <1e-7bp vs direct; makes seeds parallelizable).
   - `ens_sym.py` — the DEPLOYED-scoring ensemble cell (mean 4-seed rank score,
     majority-vote side) + LOFO + jitter sd .02/.05. The BINDING perturbation gate
     is **sd=0.02 only** (covers measured real live jitter 0.017; user decision
     2026-07-16, ledger perturbation-gate-redefinition-20260716); sd=0.05 is a
     reported diagnostic, not a gate.
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

## Sizing (measured 2026-07-10, 371d/9-10M-row dataset)
- Training job: **13-14 GB RSS**, ~2.2h at nthread=4, ~2.7h at 3, ~3.5h at 2.
  xgboost-hist scales weakly past ~4 threads. Budget RAM = 14GB x concurrent jobs
  **<= 75% of physical RAM** (orchestrate2 clamps this itself via XSYM_JOB_GB; the
  2026-07-15 96%-RAM packing killed the guest network — see KNOWN_PITFALLS) + swap on.
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
