---
name: h150-anchored-year
description: "Year cell of the TRADED anchored policy (371d CL, 6 folds × 4 seeds, frozen protocol, dataset-only col13/col44 intervention) — ENSEMBLE scoring +13.35bp t5 with ALL folds positive and jitter floor ~+3bp; per-seed +8.14±2.55 is fold2-concentrated and jitter-fragile; ensemble averaging is load-bearing"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2fba63ea-658b-4410-928d-a8c01aff03c4
---

Cell: DOGE × 371d CL (tb3s h150 labels) × anchored funding (col13=day-first, col44=0)
× walk-forward W200/T30/EMB2 (6 folds) × causal t5. Preregistered
tb3s-20260709_h150anch_year_4seeds; artifacts under maker_labels_tb3s_h150anch/.

**Per-seed t5**: +8.14±2.55bp [7.0/12.5/6.0/7.1], 4/4 positive, budget surface monotone
(t10 +4.64±0.93 → t20 → t40 ≈ 0). Fold structure: fold2 carries ~60% (+26..+35 all
seeds), fold4 negative in ALL seeds (−2..−5), LOFO−fold2 = **+3.50** [2.4/8.3/0.5/2.7].
Per-seed selection jitter-fragile at year scale: sd=0.02 → p50 +2.94 (P>0=100%);
sd=0.05 → +0.24 (P>0=74%).

**ENSEMBLE (= deployed 4-seed mean rank-score)**: t5 **+13.35bp** (563tr, 3.1/day,
hit 65.2%), **all 6 folds positive** (+2.1..+31.0 %/fold-month — even fold4 +2.5bp/tr),
LOFO +10.85..+15.19, jitter sd=0.02 p50 +7.49 / sd=0.05 p50 **+3.13, P>0=100% both**.
Side = majority-vote approximation (15.9% raw 2-2 ties → long; deployed uses mean pBg).

**Consistency triangle** (same alpha class, two venues, two horizons):
year-ensemble +13.4 / year-per-seed +8.1 / 10d-recorder-ensemble +8.6.

**Why it matters:** ensemble averaging removes BOTH the fold concentration and the
selection-noise fragility — it is load-bearing for the deployed edge, not a nicety.
Conservative stressed floor for the traded config ≈ **+3bp/tr** (jitter sd0.05 median).

**How to apply:** capacity caveat — sim numbers at $10–1k notional; treat as edge
density, not scalable ROI. ROI translation at current size: ~40bp/day base,
~10bp/day stressed floor. Any model/feature change → re-pass harnesses + rerun this
cell (BUDGETS=5,10 only per [[deploy-scope-budgets]]). Related: [[h150-sim-live-parity]],
[[rust-subms-engine]], [[scope-bound-claims]].
