# Phase A — cheap-hypothesis plan

## CURRENT DIRECTION (2026-05-17): alpha-first, execution deferred to RL

Decomposition (user, 2026-05-17): **prediction (alpha)** and **execution**
are separate problems and were getting mixed. We search ONLY for alpha now;
execution (entry placement, TP/SL, timeout, partial/trailing, selectivity)
is **deferred to a future RL agent** — it derives optimal execution given
alpha and cannot create alpha. This skips the entire combinatorial
execution-variation space for now.

Why: the historical frontier already had ultra-selective models with real
classification signal (lift 2.5–4.7× over base rate) and still lost
(−0.027…−0.077 %/tr; the one honest MAKER-first reval = −0.047 %). So
selectivity/execution is **not** the missing piece — the open question is
whether harvestable *alpha* exists at all.

Goal is **not** net 0/+15 % now — it is a **sensitivity map**: which
prediction axis (if any) yields OOS signal that clears the cost floor.
Hypotheses are ~∞ (combinatorial); the job is direction-finding.

**Alpha testbed (fixed control):** Cryptolake LINK + SOL, 90 d,
`features_v1` X (59 cols), XGB, MAKER_FIRST labels (H5 done), honest
val→test, per symbol. Cache already built → screens are ~$0, small/no VM.

**Metric — NOT EV/tr** (that conflates signal with execution+cost).
Execution-neutral, on honest OOS, per symbol:
- target = forward mid log-return `r_h` and `sign(r_h)`, h swept;
- **statistical:** OOS rank-IC (Spearman pred↔r_h), R²/AUC, decile
  monotonicity;
- **economic (decisive), two floors:** mean |r_h| in the top predicted
  decile must exceed the round-trip cost floor — `economic_pass_loose`
  (~0.08 % maker, idealised fills) and the **binding**
  `economic_pass_strict` (~0.13 % = taker 0.10 % + slippage/latency
  haircut, since unfilled makers become takers). RL converts existing
  alpha, never manufactures it, and cannot beat the floor — so
  statistically-significant-but-sub-cost signal is worthless. The ledger
  refuses to `confirm` an alpha with `economic_pass_strict≠1`
  (`v_alpha_audit`).

**Alpha axes (cheapest-first), in the ledger as `kind='alpha'`:**

| # | Hypothesis | Axis | Compute |
|---|---|---|---|
| 1 | **HA1** | forward horizon h ∈ {30,60,120,180}s | $0 eval-only |
| 2 | **HA2** | target form: logret / sign / vol-norm / path-stat | $0 |
| 3 | **HA3** | feature subset & normalization (of the 59) | $0 |
| 4 | **HA4** | decision cadence (48/72 s subsample = $0; 12 s = recompute) | $0 / med |

**Deferred to the RL execution phase (ledger `blocked`):** H2 (PT/TS),
H6 (wider TP/SL+timeout), H10 (regime TP/SL), H12 (selectivity/top-k),
H11 (VIP fees). H5 stays the confirmed foundation. H1/H3/H7/H9 remain
alpha-relevant signal/feature hypotheses (lower priority / costlier).

The section below is the prior (2026-05-16) framing — superseded for the
execution items; kept for the H5 history and methodology notes.

---

# Phase A — cheap-hypothesis plan (prior framing, 2026-05-16)

Goal: net **+15% / 30 days**. Current honest state: **no setup has positive
EV/tr under realistic MAKER-first labels**; best honest TAKER result is
LINK TCN −0.040%/tr. We get to +15% by *search*, and search is only as
good as its bookkeeping — hence the ledger ships first (done) and every run
below writes a row to `research/experiments.jsonl`.

Principles:
- **Recorded-only first.** Test what is already written down (RESEARCH_LOG
  §7 / `v_current_hypotheses`) before inventing new directions. Hypotheses
  are ~∞; compute and attention are not.
- **Cheapest compute first**, ordered by `priority_rank`.
- **Trust before lift.** A bigger number under the wrong methodology is
  negative value (it cost us 3× already). H5 is a *gate*, not a lever.

## Execution order (locked, user decision 2026-05-16)

```
0. Infra bring-up (GCP)         ~no research compute, prerequisite
1. H5  MAKER-first label gate   $0 code + cache rebuild     <- TRUST GATE
2. H2  Inner PT/TS sweep        $0 eval-only (~1h)          <- first lever
3. H6  Wider TP/SL + timeout    $0 + cache rebuild
4. H3  Cross-symbol BTC-lead    $0 + cache rebuild
5..    H1/H4/H10/H8/H7/H9/H11   $50+ / GPU / downstream (see hypotheses.jsonl)
```

Nothing in steps 2+ is trustworthy until step 1 lands. Until then every
new result is recorded with `status:"suspect"` (the ledger enforces this
for +EV TAKER rows).

---

### Step 0 — Infra bring-up (GCP `blackdigital.kz`)

Contabo is lost; this is not optional setup, it is recon + provisioning.

1. Provision the cheap 96 vCPU VM (europe-west1, co-located with
   `gs://blackdigital-scalper-data` to avoid egress).
2. Clone repo at the pinned commit; `cd rust_ingest && cargo build
   --release` (≈18 min cold) → produces `feature_builder`, `sim_labels`,
   `build_samples`, `grid_sim`.
3. `gcloud storage ls gs://blackdigital-scalper-data` — confirm the 8
   Cryptolake symbols + raw size; `gs://scalper-bot-research-data` for
   volaware checkpoints/oof.
4. Recon, then **record infra state** as a note in RESEARCH_LOG §infra
   (what survived, exact bucket inventory). The asset is the record.

Exit criterion: Rust binaries built, GCS readable, repo at known commit.

### Step 1 — H5: MAKER-first label gate (TRUST GATE)

**Reality correction (2026-05-16):** `scripts/build_cryptolake_cache.py`
is **NOT in the repo** — it died with Contabo. RESEARCH_LOG's "~30 min
edit" was stale info. The builder must be **reconstructed** against the
now-decoded GCS schema (`research/CRYPTOLAKE_SCHEMA.md`). Good news: raw
LOB survived and the path is fully specified there. MAKER-first is built
in **by construction**, not bolted on.

Work (cost-safe split — Python here at $0, Rust+rebuild on the VM):
1. Write `scripts/build_cryptolake_cache.py` per `CRYPTOLAKE_SCHEMA.md`:
   read `raw/book` parquet, decimate by `indices.npy`, build
   `entry_long/entry_short` (maker-first: long=bid_0, short=ask_0) +
   `mid_path`/`book_path` over a wall-clock horizon (gaps! use
   `timestamp`), align X = `features.npy`. `fee_regime` is an explicit
   arg: MAKER_FIRST `0.04/0.07` (gate) vs TAKER `0.07/0.10` (A/B).
2. **Validate the data-prep here on 1 BTC symbol-day at $0** (shapes,
   index alignment, entry/mid sanity vs the parquet). The rust
   `sim_labels` call is the only part that can't run in the planning
   container (crates.io 403).
3. Parity assertion: same sample's `pl/ps` must differ MAKER_FIRST vs
   TAKER (structurally proves the regime is wired, not defaulted).
4. **Only after 1-2 above pass**, provision the VM, `cargo build`, run
   `sim_labels` MAKER-first on a date-range subset (NOT 8×546 by
   default — that is a later deliberate spend).
4. Re-evaluate the **current best** (LINK TCN −0.040% TAKER) under
   MAKER_FIRST. This is the new honest baseline.

Record: one experiment row, `hypothesis_id:"H5"`,
`fee_regime:"MAKER_FIRST"`, the 7 owner metrics populated (the cache
rebuild is the chance to start capturing them). Then append a hypotheses
revision: H5 `status:"confirmed"` (gate lands) and **unblock H2**
(`status:"active"`).

Decision: H5 has no "lift" success criterion — success = labels match
live execution and the LINK baseline is re-measured honestly. The number
will likely get *worse*; that is the point (we stop hiding losses).

### Step 2 — H2: inner PT/TS params via fused grid_sim (first lever)

Infra already exists: `rust_ingest/.../grid_sim` (130× speedup, ~30s per
100K configs), `src/rust_bridge.simulate_labels_grid`,
`scripts/test_pt_ts_sweep.py` (smoke), `scripts/grid_ensemble_b300_100k.py`
(the 100K driver). **Eval-only — the ensemble is already trained.**

Rationale (RESEARCH_LOG §7 #2, docs/SESSION_2026_04_16): the dominant
loss mechanism is full-SL losses (−0.14% net) swamping tiny timeout-wins
(+0.005–0.06%). Partial TP locks the winning side; trailing SL closes the
losing side earlier → the asymmetric tail compresses. Swept knobs:
`partial_tp_progress`, `trailing_step1_progress`,
`trailing_step1_sl_floor_pct`, `trailing_step1_sl_ratio`,
`trailing_step2_progress`, `trailing_step2_sl_ratio` (+ `par`/`tr`
toggles), on the MAKER-first cache from Step 1.

Work (~1h GCP):
1. Smoke: `python scripts/test_pt_ts_sweep.py` — confirm PT/TS params
   actually reach `simulate_trade` (it asserts distinct `pnl_long` means).
2. Sweep via `simulate_labels_grid` over the 6-param inner grid on the
   MAKER-first cache, honest val→test split, `n_trades ≥ 30` floor.
3. Rank by `ev_per_trade_pct` at matched `trades_per_day`; pull the
   12-reason `exit_hist` for the top configs to confirm the mechanism
   (full-SL share down, partial/trailing share up).

Record: one experiment row per distinct operating point worth keeping
(top-by-EV, top-by-Sharpe, the baseline), `hypothesis_id:"H2"`,
`fee_regime:"MAKER_FIRST"`, full owner-7 + `exit_hist_json`,
`artifact_path` = `gs://…/h2_grid.json` + sha256, exact `repro_cmd`.
Then a hypotheses revision: H2 → `confirmed` / `refuted` with
`result_experiment_id`.

Success criterion: `ev_per_trade_pct > 0` MAKER-first at `n_trades ≥ 30`
on the honest tail, with the exit histogram showing the predicted shift
(else `refuted` even if EV nudges up — mechanism must match).

### Steps 3+ — queue

Per `v_current_hypotheses` ordering. H6 (wider TP/SL, horizon-capped
relabel), H3 (BTC-lead features, cache rebuild), then the GPU-cost tier
(H1/H4/H8/H7) and the structural/downstream ones (H10/H9/H11). Each is
re-costed against the H5-corrected baseline before spending — a cheap
hypothesis that needs a $200 GPU run is not cheap.

## Definition of done for Phase A

- [x] Ledger contract shipped (`schema.sql` + `ledger.py` + CI test).
- [x] History backfilled, artifact chains queryable, gate enforced.
- [x] This plan: cheap-first, recorded-only, H5-gate → H2, executable.
- [ ] Step 0–2 executed on GCP, results recorded MAKER-first. *(Phase B —
      needs the GCP compute node; not runnable from the planning container.)*

When Step 2 completes, regenerate the frontier
(`ledger.py frontier` → RESEARCH_LOG §3) and re-plan from the new baseline.
