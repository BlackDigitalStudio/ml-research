# HANDOFF — execution-lever session (B-universe / abstention / fill-C / taker-direction), HD3

> Next agent on branch `claude/goofy-jackson-8a2e08`. Read this, then `RESEARCH_LOG.md §15`
> (last paragraph "EXECUTION-LEVER session"), the ledger rows below, and the prior
> `HANDOFF_xgb_maker_hd3.md` (still valid for the AB/grid context).
>
> **⚠️ FRAMING DISCIPLINE (CLAUDE.md, the whole point).** The deliverable is the conditional
> alpha **surface**: under what conditions a tradeable edge appears. **Do NOT turn a session
> result into a universal verdict.** "Taker is fee-bound / asymmetry overfits / direction is the
> wall / dead end" are BANNED — they read as true for all models/data/architectures/objectives/
> symbols, which is false. The honest form is **"on conditions [X,Y,Z], this session did not
> achieve …; untested: […]"**. Every negative below is conditional, not closed.

Date: 2026-06-03. VM: GCP `hd2-feats-003` (europe-west1-b, account `virgin.ship03`,
bucket `gs://market-data-0998ac51`). Recompiled `grid_sim` binary: `/tmp/gridbuild/release/grid_sim`.

---

## 0. The asset / best result (unchanged this session)
The tier's best result remains the **maker apred cascade: +3.00 bp pooled, 7/8 symbols net-positive**
(LINK +8.5, DOGE +5.7, ETH +4.5, BTC +2.8, XRP +2.3, LTC +1.5, SOL +1.1; BNB −2.5), at apred-gate B +
hold-60s + A∧B 1-trade/symbol/day, maker-maker 4 bp. Durably re-saved this session (the original 8-sym
`B2_RESULT_apred.json` had been clobbered): `…/maker_labels_rr/B2_RESULT_apred_8sym.{log,json}` + ledger
`xgb-20260601_makerlabels_b2_apred`. **It is ONE split, NOT walk-forward-confirmed.** This is the single
most valuable unclosed question (see §4).

## 1. What this session measured (conditional surfaces — all single honest_val_test split unless noted)
All on **XGB-snapshot 71-feat, `maker_labels_rr`, apred-5% gate** unless stated. Ledger ids in `()`.

1. **B training-universe** (`xgb-20260602_makerlabels_b_universe`, BTC+LINK). Pure-maker EV on a FIXED
   A-top-5% test pool is maximised at the NARROWEST B-training gate (apred 5%) at every budget, both
   symbols; 25%/100% lower it monotonically. B GLOBAL dir-AUC ≈ 0.51 for all → the gap is **train/deploy
   distribution match**, not model quality. Weights `…/b_universe/{A,B_*g5/25/100}.xgb.json` + preds.

2. **Toxicity-gated abstention** (`…_abstain`, BTC+LINK). Ranking the apred-pool trades by an adverse-fill
   predictor did not beat B-confidence ranking; `1{filled trade loses}` ≈ random here (abstAUC 0.51–0.54).
   Insight: the predictable adverse part (MISS-on-runaway, tox-AUC ~0.75) is **non-actionable** for a maker
   (a MISS already pays 0); the actionable part ≈ the unpredictable 60s sign.

3. **Model C = maker-fill predictor** (`…_fillmodel_C`, BTC+LINK). **Fill IS strongly predictable** (test
   fill-AUC BTC 0.73/0.72, LINK 0.64/0.62 ≫ the ~0.52 of 60s direction). BUT fill-asymmetry carries ~no 60s
   direction (rank-IC −0.017), and as a **3rd A∧B selection factor** it did not raise +3.00 on these
   conditions (both signs hurt; fill-rate on B's side already ~0.96 → 'will it fill' near-constant; adverse
   selection lives in the post-fill outcome). **Untested & promising: C as input to maker PLACEMENT**
   (offset/aggressiveness/queue), not as a selection factor. Weights `…/fill_model/`, `…/abc/`.

4. **Adverse selection quantified** (the prize). On A-non-flat windows: fwd-60s | a long limit **FILLED** =
   −0.6 bp (BTC) vs | **MISSED** (price ran away) = **+13.4 (BTC) / +33 (LINK) / +29 (SOL)**. You fill the
   small-adverse moves and miss the big favorable runaways. Computed offline from `maker_labels_rr`.

5. **Taker-entry direction pipeline** (`xgb-20260603_makerlabels_b_taker`, fee_regime=TAKER, BTC+LINK).
   Built **TAKER labels** for all 8 symbols (`…/taker_labels/{SYM}.npz`, gross hold pnl, cross @t0). Pipeline
   (GRU-style, in XGBoost): Stage-1 predict-the-MOVE (XGB **regressor on rH60**) → grid_sim-TAKER c* per
   symbol (captured-alpha on top-3000 VAL conviction) → Stage-2 executed-payoff fine-tune (custom XGB obj
   `grad=-σ(1-σ)(PL-PS)`) on c* taker payoffs → deploy A-5% pool, top-N/day by |pB2|, taker entry, net 7 bp.
   **Taker entry captures POSITIVE GROSS** (+2.6…+4.6 bp at conviction — the runaways ARE catchable gross),
   but under standard 7 bp taker RT and one split, **taker NET did not exceed +3.00 this session** (net
   −3…−8 bp pooled). Weights `…/b_taker/{B1,B2}_{SYM}.xgb.json` + preds + `B_TAKER_RESULT.json`.

## 2. Mechanism findings worth carrying forward (the interesting part)
- **Stage-1 OBJECTIVE drives the config.** Predicting the MOVE (regressor on rH) makes grid_sim pick a
  HIGH-TP/SL c* (BTC RR2.9 → RR6.8 with captured-alpha selection); a binary better-side target picks hold.
  *Signal value must be judged ECONOMICALLY with its matched TP/SL — raw dir-acc/WR is the wrong lens*
  (user: WR 20% @ TP/SL 20:1 is far from a coin-flip). A true global-correlation IC obj **underfits**
  XGBoost greedy trees (dir-AUC 0.50) — the regressor is the XGB-native "predict the move".
- **The high-TP/SL c* chosen on VAL did not transfer to TEST** on these conditions (val-optimistic,
  test-worse) — same val→test gap the maker-R:R argmax showed. → **walk-forward c* is the honest test.**
- **"Direction predictable at selectivity" — which direction?** The strongly-predictable conviction signal
  (§15 dir-acc 0.66–0.77 at top-10%) is the **maker-better-SIDE** (fill microstructure), which the maker
  +3.00 exploits at 4 bp. Raw-60s direction (what a taker needs) rose with conviction here only to ~0.54
  (BTC 0.511→0.542). These are different targets; do not conflate.
- **`grid_sim --absolute-timeout`** added: exit at fixed t0+60 (vs fill+60). Absolute ≈ fill-relative
  (corr 0.998–0.999) — horizon "drift" is small because fills are fast (~1-5 s).

## 3. Artifacts (all on GCS `gs://market-data-0998ac51/research_runs/`)
- `taker_labels/{8 SYM}.npz` (NEW), `b_taker/`, `fill_model/`, `abc/`, `abstain/`, `b_universe/`,
  `maker_labels_rr/B2_RESULT_apred_8sym.{log,json}`.
- **Rust:** recompiled binary `/tmp/gridbuild/release/grid_sim` (VM, --absolute-timeout). Full husdc Rust
  SOURCE (maker-entry machinery + flag) captured to `research_runs/husdc_src/husdc_rust_src_20260603.tgz`
  (sha256 5e8b56cd…). ⚠️ **The repo `rust_ingest/src/bin/grid_sim.rs` is STALE (pre-maker-entry); the
  canonical maker-capable source is VM-only / the GCS tarball.** Syncing the husdc tree into the repo is an
  open capture-info TODO (pre-existing; not introduced this session).
- Scripts committed: `scripts/subs60_{xgb_b_universe,xgb_toxgate,xgb_abstain,xgb_fill,xgb_abc,
  build_taker_labels,xgb_b_taker}.py`. (`xgb_b2_grid`, `xgb_b2` from prior session.)
- Ledger ids this session: `xgb-2026060{1_b2grid_optb, 1_b2_apred, 2_b_universe, 2_abstain,
  2_fillmodel_C, 3_b_taker}`.

## 4. Suggested next steps (options, not mandates — pick per evidence)
1. **Walk-forward the apred +3.00** (the most important confirmation: real edge or single-split optimism?
   the earlier single-split argmax died in walk-forward). Reuse `subs60_xgb_walkforward.py` machinery.
2. **Walk-forward the taker high-TP/SL c\*** — close honestly whether the asymmetric config transfers OOS
   (this session's val→test single split was insufficient), and try **RR/SL-floor-constrained** configs
   (SL floored well above 3 bp) for robustness.
3. **C → maker PLACEMENT** (not selection): use the strong fill signal (AUC ~0.7) to choose offset/queue/
   aggressiveness, staying maker at 4 bp — the untested use of C.
4. **Stronger raw-direction signal** for the taker capture: sequence/longer-context (GRU §14 got dir-acc
   0.612 — but those economics carried **test=val inflation + mid-entry optimism**, treat as upper bound),
   cross-asset lead-lag, flow toxicity, OI/funding/liq. Measure the signal ECONOMICALLY with its TP/SL.
5. **Realistic resting-maker EXIT** (the touch-exit "later" item) — but note (§15) the exit horizon is NOT
   the profit-cutter here; adverse ENTRY selection is.

## 5. Working discipline (hard-won, in user memory)
- **No verdicts — conditional surfaces only** (CLAUDE.md; the user re-flagged this hard this session).
- **Capture everything, never discard** (user escalated, "third agent in a row"): every training script
  MUST save weights+preds+trials; never kill an in-flight run to "save compute".
- **Verify no duplicate VM runs** before/after launch (`ps -C python3`, NOT `pgrep -f` which self-matches
  the launcher); the bg `nohup` pattern survives flaky ssh.
- **Audit before long runs**; smoke 1-2 symbols first.
- Selectivity note: A∧B 1-trade/day is ALREADY tighter (~0.01% of windows) than the GRU's top-0.2% gate —
  cross-day "pick best days" is look-ahead and is BANNED on test.
