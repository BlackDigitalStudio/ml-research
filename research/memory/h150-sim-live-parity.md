---
name: h150-sim-live-parity
description: "h150 parity saga (2026-07-08): funding ns/ms bug → 3 semantics (train/val/live); ANCHORED policy (col13 day-frozen, col44=0) is the robust cell (+8.61bp t5, LOO 0/10 neg, jitter P>0=100%) vs true-funding −2.14 (noise); perfect-parity engine deployed 07-08 23:32 UTC"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2fba63ea-658b-4410-928d-a8c01aff03c4
---

Cell: DOGE × 20260628–0707 recorder × h150 4-seed ensemble × causal t5.

**Funding ts-unit bug** (fix 079fa29): writers wrote ns, FB's `funding_rate` schema
reads ms → col13 = file's first row, col44 = 0. Training (CL raw `rate` schema) was
correct → three semantics: train (true rate/basis) ≠ validation (day-start const/0)
≠ live (rate t−20min/0).

**The response surface inverted my presumption**: TRUE-funding semantics = −2.14bp t5
(LOO sign-unstable, jitter P(EV>0) 21–56% — noise). DAY-ANCHOR semantics (col13 frozen
at day-first mark_price rate, col44=0) = **+8.61bp t5** (66tr, hit 56%, LOO
+0.94..+18.95 with 0/10 negative, score-jitter P(EV>0)=100/100/98% at sd .02/.05/.10).
Freezing the hypersensitive funding input (±0.17 score per 1e-4 raw) is variance
reduction that the tail selection rewards. Old broken-run +6.59 = same cell on bt[0]
grid/partial days. Prefix: `_recev_h150anch_DOGE` (also the live tau seed).

**Perfect-parity engine** (deployed axb-live-doge 2026-07-08 23:32 UTC, commits
3c4b61a/09fb5ba/e4cd2e4): MirrorBook = port of chronos OrderBookV2 (@depth@100ms
diffs, REST seed limit=100, cap 100 lvls, skip u<=last, no pu-chain, reconcile 900s
reseed>=2 findings) — old @depth20@100ms partial stream shared only ~18% of tick ts
with the recorder's diff-synthesized view; btc_lead from BTCUSDT MirrorBook L1;
funding day-anchor single-row fund.parquet; OI 15s local-ts; decisions on 3s
exchange-ts grid anchored at CALENDAR UTC MIDNIGHT (bt[0] is a flush-phase artifact —
recorder hour-00 files carry a pre-midnight tail). recorder-EV script has
FUNDING_MODE=anchor|true; validations of the deployed policy must use anchor+midnight.

**How to apply:** deployed-policy EV numbers come from `_recev_h150anch_DOGE`-style
runs only; `recorder_ev` numbers before 2026-07-08 measured other cells. Live-vs-sim
decision layer now matches by construction; remaining gap = execution reality
(acceptance: parity check on live days; first-trade execution check passed −27.4bp
inside sim −24.6/−35.5 envelope, ROI@2x −0.55%). Related: [[h150-deploy-candidate]],
[[live-trading-deploy]], [[label-matching-lookahead]], [[scope-bound-claims]].
