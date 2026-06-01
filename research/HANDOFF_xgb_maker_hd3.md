# HANDOFF — XGBoost A/B on sub-60s MAKER-REALISTIC labels (HD3)

> For the next agent on branch `claude/goofy-jackson-8a2e08`. Read this first, then
> `RESEARCH_LOG.md` §15 and the ledger rows (`research/experiments.jsonl`, grep `HD3`).
>
> **⚠️ READ THIS DISCLAIMER FIRST — it is the whole point of this file.**
> Below, **MEASURED** (what a run actually produced, with its caveats) is kept strictly
> separate from **HYPOTHESIS / OPEN** (plausible but NOT established). Do **not** turn a
> hypothesis into dogma. In particular none of these are proven, do not repeat them as
> settled fact: "more data won't help", "direction is the wall", "only signal + execution
> are left", "per-symbol R:R can't beat hold", "no levers remain". Each is a *current
> reading of limited evidence on one dataset/horizon/model*, often a SINGLE train/val/test
> split, frequently NOT walk-forward-confirmed. Treat them as things to TEST, not assert.
> (Per CLAUDE.md: the deliverable is the conditional alpha **surface**, not a verdict.)

Date: 2026-06-01. VM: GCP `hd2-feats-003` (europe-west1-b, account `virgin.ship03`,
project `project-0998ac51-36ba-445c-bc7`, bucket `gs://market-data-0998ac51`).

---

## 0. One-paragraph context
We trained an XGBoost 2-model cascade on sub-60s data with **maker-realistic
(adverse-selection) labels**: **Model A** = per-symbol volatility gate, **Model B** =
direction / profitable-maker-side, deployed with an **A∧B confidence filter + a per-day
trade budget**, scored at **maker-maker fees** (0.02%/side = 0.04% round-trip). Goal
(per CLAUDE.md, exploratory): map *under what conditions* XGBoost yields harvestable
maker alpha — a response surface, not a pass/fail verdict.

---

## 1. SETUP (measured facts about the pipeline)
- **Data**: `feats_sub60/{SYM}/{DATE}.npz`, 8 symbols (BNB BTC DOGE ETH LINK LTC SOL XRP),
  ~2826 symbol-days, 2025-05-09 → 2026-05-08 (LINK only 244 days, 119-day outage; rest 361-362).
  X(64) engineered feats on a 1s grid + rH_{15,30,45,60} (signed fwd book-mid logret, bp) + valid.
- **Maker labels** (`subs60_makerlabel_build.py` → `research_runs/maker_labels/{SYM}.npz`):
  per feats decision point (stride-8), a realistic maker P&L from the HUSDC maker-sim
  (`build_samples` + `grid_sim` maker mode, binaries at `/tmp/husdc/rust_ingest/target/release/`):
  resting limit fills on realized taker-flow **touch/queue**, **MISSES on runaway** → adverse
  selection emerges from the path. Saved 3 cfgs (hold-60s / RR6 / RR2) × 2 queue-mults (touch qm0 / queue qm1),
  pnl_long/short (%, NaN=miss) + fill masks. 71 feats = X64 + signed BTC-lead{5,30,60}s + ToD4.
- **A (per-symbol vol-gate)**: target = `|rH60| ≥ per-symbol TRAIN p95(|rH60|)` (vol-ADAPTIVE
  top-5% non-flat; user's call — a fixed 13bp is 2.4σ for BTC vs 1.25σ for LINK). XGBClassifier, Optuna→val-AUC.
- **B (direction)**: target = `1{net maker pnl_long > pnl_short}` (the profitable MAKER side; net = gross_bp − 4bp fee).
- **Split**: purged day-split, train<0.65 ndays, embargo [0.65,0.68), test≥0.68, val=last 15% of train days. (`honest_val_test`)
- **Fees**: maker-maker 0.04% RT (0.02 entry + 0.02 exit). **Caveat (measured-limitation):** maker ENTRY
  fill is realistic (touch/queue/MISS); maker EXIT is **touch-based** for RR configs and a timeout
  close for hold-60s — a fully-realistic resting maker exit is NOT built (the MAKER_SIM.md "later" item).

---

## 2. MEASURED results (each with its caveat — these are real run outputs)
1. **A vol-gate transfers well**: test AUC **0.818–0.859** all 8 symbols; prec@0.2% argmax BTC 0.62.
   (exp `xgb-20260531_makerlabels_AB`, `xgb_maker/SURFACE.json`)
2. **B raw-direction is weak / inverted**: the profitable-maker-side **anti-correlates with the price
   move** (dir-vs-raw-sign <0.5, rank-IC<0) — you fill on the adverse side. B's accuracy *at its own
   target* (which side is better-maker) is ~0.66–0.77 @conviction. (surface/recency2)
3. **Best DEPLOYABLE point found this session**: apred-gate B (B trained on A-PREDICTED non-flat
   windows, see §3) + **hold-60s** + **A∧B daily-budget at 1 trade/symbol/day**, on the **TEST** set,
   maker-maker fees: **pooled ≈ +2.9 bp/trade, 7/8 symbols net-positive** (LINK +8.5, DOGE +5.7,
   ETH +4.5, BTC +2.8, XRP +2.3, LTC +1.5, SOL +1.1; BNB −2.5).
   **CAVEATS (do not drop these):** ONE train/val/test split (NOT walk-forward-confirmed);
   ~116 trades/symbol (modest n); the edge is at **1/day extreme selectivity** — at **10/day it goes
   NEGATIVE** (edge concentrated in top conviction). (`maker_labels_rr/B2_RESULT_apred.json`)
4. **Volume saturates / A is time-stationary** (so far): a model trained on the **last 90 days ≈ full
   ~200-day history** on the same clean test (mean ΔAUC +0.003); equal-size recent vs old 45% windows
   ≈ tie on A; B mildly recency-sensitive (dir +0.009 recent>old, 6/8). (exps `_recency`, `_recency2`, `_3mo`)
   → on THIS data/horizon, adding more history did not improve OOS. **Not a universal law.**
5. **A-only top-q% cascade single-split argmax did NOT survive walk-forward** (4-fold, op-point chosen
   on val, measured on disjoint test): per-symbol R:R argmax that looked +2bp on one split went net-negative
   OOS; hold-ish robust. (exp `_walkforward`) — this is *why* we now do val→test honestly.
6. **Confidence-filter surface** (`_dailybudget`, `_dailygrid`): on the single test split, tighter A AND B
   both raise EV (A-tightness > B-tightness; A1%×B10% +20 > A10%×B1% +13 bp); corner +23 bp at ~0.11 trd/day.
   **CAVEAT: that grid/argmax is computed ON TEST → selection-over-conditions optimism, NOT a confirmed edge.**

---

## 3. What "apred gate" means (measured mechanic, since it's the best result)
B is trained on the windows **Model A predicts non-flat** (top-`gate_pct`%=5% by **out-of-fold** pA on
train; K-fold-by-day, so A's false positives are included) — NOT on oracle realized non-flat. This
removed a train/deploy mismatch and is what lifted the EV in §2.3. Script: `subs60_xgb_b2.py --gate apred`.

---

## 4. HYPOTHESES / OPEN QUESTIONS (NOT established — test, don't assert)
- **"Per-symbol optimal R:R (TP/SL) doesn't beat hold-60s OOS."** Strongly *suggested* by 32-cfg @1/day,
  32-cfg @10/day, and 6011-cfg @10/day runs (grid picks RR configs on val that lose to hold on test). BUT
  the proper §14-faithful test is STILL RUNNING (see §5): a ~95k-config grid (baseline reused-HP B) + an
  **Optuna-B variant** (per-symbol-tuned B, because reused-pooled-HP may handicap the c* model — a fair
  concern the user raised). **Do not conclude until both finish.** GRU §14 *did* find per-symbol R:R helped,
  so there is a real precedent that it can.
- **"More data won't help / regime-adaptation isn't the bottleneck."** Supported by §2.4 on THIS
  data/horizon/model only. Untested: other horizons, more symbols, regime-conditioned models.
- **"Direction is the wall; need a NEW signal (cross-asset lead-lag, flow toxicity, OI/funding/liq,
  longer context / sequence model)."** A recurring hypothesis across RESEARCH_LOG; plausible given §2.2,
  but the new-signal axis is NOT tested in this XGB-maker context.
- **"Taker (or smarter maker) execution would turn it positive."** From §14 (GRU era: taker entry +5.6bp);
  NOT re-tested here. The maker exit is also only touch-modeled (§1 caveat).
- **"The apred-gate +2.9bp is real alpha."** Only one test split. **Needs walk-forward** before believing it.
- Whether a **deployable throughput** exists (edge at 1/day but gone by 10/day) is OPEN.

---

## 5. WHAT IS RUNNING RIGHT NOW (in-flight on the VM — check before relaunching anything)
- **Baseline 95k-config B2-v2** (reused-HP B, the §14 sequence Stage-1 B → grid_sim 94966 TP/SL configs
  on-demand on val daily-budget(10/day) windows → c* → Stage-2 B → eval test hold vs c*):
  `/tmp/b2grid_full.log`, `subs60_xgb_b2_grid.py --symbols ... --n-tp 317 --n-sl 317 --budget 10`.
  ~16–25 min/symbol → **~2.5–3 h total** (95k grid + OOF-A + per-day path downloads dominate; grid_sim itself is fast).
  Saves `maker_labels_rr/B2GRID_RESULT.json`. As of handoff: BNB done (c*=RR2.7; test hold −5.59 vs c* −5.42, both negative).
- **Optuna-B variant** is QUEUED (a `ScheduleWakeup` will fire to: collect baseline → SMOKE `--optuna-b` on BTC
  → launch full `--optuna-b --b-trials 25` → save `B2GRID_RESULT_optb.json` → compare). It is NOT yet started.
- **A scheduled wakeup (~22:25 UTC) carries the next-step instructions** — it may fire and continue
  autonomously. If you are a fresh context, just check `ps`/logs first; do NOT relaunch a run that's alive.
- The B2-v2 grid uses **on-demand Rust `grid_sim`** from saved paths (`research_runs/maker_paths/{SYM}/{DATE}.npz`,
  5.16 GB total). It is NOT a Python grid.

---

## 6. ARTIFACTS
- **Datasets (GCS)**: `research_runs/maker_labels/{SYM}.npz` (3-cfg × 2-qm); `research_runs/maker_labels_rr/{SYM}.npz`
  (32-cfg R:R, touch qm0); `research_runs/maker_paths/{SYM}/{DATE}.npz` (raw maker arrays for on-demand grid_sim, 5.16 GB).
- **Trained/results (GCS `research_runs/xgb_maker/`)**: `A_{SYM}.json/.xgb.json`, `B_pool.json/.xgb.json`,
  `preds_{SYM}.npz` (per-test-sample pA+pB+all-cfg×qm payoffs → recompute any surface offline, NO retrain),
  `MANIFEST.json`, `SURFACE.json`, `DAILYBUDGET.json`, `DAILYGRID.json`, `WALKFORWARD.json`, `RECENCY.json`, `RECENCY2.json`.
  `research_runs/maker_labels_rr/`: `B2_RESULT_oracle.json`, `B2_RESULT_apred.json`, `B2GRID_RESULT.json` (in progress), `B2GRID_RESULT_optb.json` (pending).
- **Scripts (`scripts/`)**: `subs60_makerlabel_build.py` (build maker labels; `_run_makerlabels*.sh` launchers),
  `subs60_save_maker_paths.py` (save paths), `subs60_xgb_makerlabel.py` (train A/B), `subs60_xgb_surface.py`
  (offline surface), `subs60_xgb_walkforward.py`, `subs60_xgb_recency{,2}.py`, `subs60_xgb_dailybudget.py`,
  `subs60_xgb_dailygrid.py`, `subs60_xgb_b2.py` (per-symbol R:R B2, --gate oracle|apred, --budget),
  `subs60_xgb_b2_grid.py` (on-demand 100k-grid B2-v2, --optuna-b), `subs60_vol_inventory.py`.
- **Ledger** (`research/experiments.jsonl`, kind=strategy, status=exploratory, hypothesis HD3):
  `xgb-20260531_makerlabels_{AB, walkforward, recency, recency2, 3mo, dailybudget, dailygrid}`.
  **NOT yet logged** (pending the in-flight runs): the b2 / b2grid results — log them when done.
- **RESEARCH_LOG.md §15** narrates all of the above.
- **Git**: committed `32660f6` (AB+walkforward), `d25b1c4` (recency), `21f1ebc` (3mo), `592f486`
  (dailybudget+dailygrid+R:R-grid build + b2 scaffolding). **UNCOMMITTED**: later edits to
  `subs60_xgb_b2.py` (--gate apred/OOF/--budget/grid=daily-budget), `subs60_save_maker_paths.py`,
  `scripts/_run_save_paths.sh`, `subs60_xgb_b2_grid.py`. Commit when convenient (user-gated).

---

## 7. SUGGESTED NEXT STEPS (options, not mandates — pick per evidence)
1. **Finish & read the in-flight runs** (baseline 95k + Optuna-B). Question: does a per-symbol R:R, with a
   properly-tuned B, beat hold-60s OOS for any symbol? (If yes → per-symbol R:R is a lever; if no → hold robust.)
2. **Walk-forward the apred-gate 1/day result** (§2.3) — the single most important confirmation: is the
   ≈+2.9 bp pooled real across folds, or single-split optimism (as the earlier argmax was)?
   Reuse `subs60_xgb_walkforward.py` machinery + the apred gate.
3. **Throughput question**: edge at 1/day, gone at 10/day — characterize the EV-vs-budget curve OOS.
4. **Hypotheses worth TESTING (not assuming)**: new direction signal (cross-asset lead-lag / flow toxicity /
   OI-funding-liq / sequence model); taker or resting-maker-exit execution (§14 taker +5.6 bp, untested here);
   breadth across more symbols/markets for aggregate significance.
5. Commit the uncommitted scripts.

---

## 8. WORKING DISCIPLINE (this session's hard-won rules — in user memory too)
- **Audit before long runs** (trace shapes/indices/leakage/memory; smoke 1-2 units; scope cost) — do not
  launch→discover-bug→relaunch in a loop. (This session hit: f32-vs-f64 for grid_sim, OOM in path-loading,
  pgrep self-match in chained waits, an oracle-biased c* discovery, a grid↔deploy metric mismatch — all
  caught by smoking first.)
- **Capture all information** — save predictions/payoffs/trials/weights so any surface is recomputable
  offline without retraining (that's why `preds_{SYM}.npz` + `maker_paths/` exist).
- **State surfaces, not verdicts** (CLAUDE.md). And — per the user — **never present a hypothesis as a fact.**
