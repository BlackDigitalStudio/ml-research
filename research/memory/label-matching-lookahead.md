---
name: label-matching-lookahead
description: "CORRECTED (2026-07-05): suspected lookahead REFUTED; real finding = year walk-forward EV of THIS protocol (DOGE CL-year, xgb-71f AxB tail-selection) has ±3-5bp structural variance — its old-semantics edge never statistically identified. Scope-bound: this cell only, see scope-bound-claims"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6287b103-b1f8-4152-bb04-df2d7ec2e6cf
---

**The lookahead suspicion was WRONG — refuted by a model-free test** (user correctly challenged it: a learnable bug would have been learned everywhere). Facts: makerlabel_build's build grid is dense (40k/day) so the matched offset is |dt|~0.3s median, overlap move p90 0.91bp; matched labels do NOT pay the overlap (corr(r_ov,netl) −0.031 matched vs −0.042 exact). Matched-vs-exact labels across the year: corr 0.99, mean diff −0.001bp, yB agreement 95%.

**The real finding (HD3 rev6 + §20a):** a ~5% zero-mean perturbation of B's training targets swings the year walk-forward total by ~5.5bp (+3.77 → −1.74). The protocol's year-EV has **structural variance of several bp** (top-5/day tail selection × deep trees × per-fold Optuna). All old-semantics draws (+4.17 qm1, +3.77 robust2, +0.91 my-grid, −1.74 exact) ≈ one distribution, mean ~+1 ± 3 → **no statistically identified edge at the old 2-3min semantics**. The honest-30s cells are consistently negative across every draw (−3.8…−2.4, all budgets, both training horizons) — that part is stable **within its cell: DOGE-USDT CL-year 2025-05→2026-06, xgb-71-feat AxB family, always-last both legs, 12.8s entry / 30s-hold pegged maker exit**. It is NOT a law about "30s trading", other signals/features/models/markets — re-measure there ([[scope-bound-claims]]). B's direction IC stays positive (+0.03..0.09) throughout — signal exists; THIS selection does not monetize it through THIS maker cycle.

**Why:** tail-selection protocols amplify tiny input perturbations into bp-scale total swings; single-run year EVs from this family are not point estimates. **Seed-proof (2026-07-06):** byte-identical robust2 data, only RNG seed varied → AxB t5 = +3.77 / −1.87 / +2.81 / +2.07 (seeds 0-3), 5.6bp from RNG alone; full old-semantics ensemble +1.4 ± 2.4. Deployed live weights = literally the seed-0 draw. Regime pattern (winter folds +, last fold −) reproduces across seeds; amplitudes don't.

**DEAD-COLUMNS FINDING (2026-07-06, measured in the npz):** qm1 (the +4.17 baseline) trained on REAL funding (col 13, 100% nonzero), ETH-lead (14-16, 55), liquidations (56-58) and OI-deltas (59-60); in robust2 AND tb3s these columns are ALL ZERO — the robust rebuild invoked feature_builder without --funding/--liquidations/--open-interest/--eth although the raw CL streams exist in the bucket since 2024-11. So the deployed model and the honest-30s −3.77 were measured with ~11 informative columns dead. These features have NO tick-clock transfer pathology (funding/OI = timestamped states, liq = events; recorder writes all of them). Full-feature honest-30s cell = maker_labels_tb3s_full (built 2026-07-06). Do NOT attribute qm1's fold stability to the old tick-window vol features without this test.

**How to apply:** before trusting ANY year-EV from tail-selection walk-forwards, run a **perturbation gate**: retrain on label-jittered/target-flipped copies; the spread across copies is the error bar. Related: [[live-trading-deploy]], [[cl-recorder-sampling-mismatch]].
