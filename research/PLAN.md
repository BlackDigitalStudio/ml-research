# Phase A — cheap-hypothesis plan

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

Problem (RESEARCH_LOG §4, code state §9): `scripts/build_cryptolake_cache.py`
writes **TAKER** labels by default; the maker-first relabel exists but is
never copied back into the cache `pl/ps`. `sim_labels.rs` already has
`--entry-taker-long/short` for the maker-first hybrid — the gap is
pipeline wiring, not new math.

Work (~30 min code + cache rebuild):
1. Wire the maker-first relabel into `build_cryptolake_cache.py` so cache
   `pl/ps` are maker-first by construction (entry maker, exit hybrid;
   fees `commission_win_pct=0.04` / `commission_loss_pct=0.07` are the
   maker-first round-trip, vs TAKER no-VIP `0.07 / 0.10`).
2. Add a parity assertion: a known sample's `pl/ps` differs between
   TAKER and MAKER_FIRST builds (catches the "relabel not copied" bug
   structurally).
3. Rebuild the relevant cache(s) from `gs://blackdigital-scalper-data`
   (30–45 min/symbol, workers=32).
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
