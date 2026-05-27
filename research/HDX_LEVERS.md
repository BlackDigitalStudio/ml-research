# HDX — XGBoost lever space (what we can regulate, and how)

Companion to `hypotheses.jsonl` HDX rev1 (XGBoost conditional alpha-surface tier,
matched 1:1 to HD2 Mamba-2). This enumerates EVERY axis we can regulate for XGBoost
in the alpha-research context, how to regulate it, prior evidence, and expected
impact — so the HDX surface sweep is planned, not ad-hoc.

Success metric throughout = **`cap_edge_gross`** (mean over top-confidence decile of
`sign(pred)·r_H`, %/trade), RELATIVE across configs; cost(0.13%)/auc/rank_ic SECONDARY
(HD2 rev10/rev19). Every cell reports seed-sd (CLAUDE.md).

## Priority framing (the HD2 lesson, by analogy)
HD2 found: **objective/loss (IC vs R1) = +0.023 vs +0.008 (big)**; but capacity/reg
sweeps (reg, timescales, d_conv, d_state, n_layers) were **within-noise FLAT** — the
binding constraint was **DATA / direction-skill on large moves**, not model capacity.
By analogy for XGBoost, expected alpha-impact ranking:

1. **HIGH:** data/feature substrate · target form · horizon H · objective/loss · pooling/scope
2. **MED:** normalization · feature subset/selection · decision cadence · sample weighting
3. **LOW (numerous but likely within-noise for alpha):** the tree hyperparameters
   (depth/eta/reg/sampling). Cheap to verify, but do NOT over-invest before the HIGH axes.

→ Sweep HIGH axes first; treat hyperparams as a late, cheap confirm round.

---

## A. DATA / SAMPLE COMPOSITION  (HIGH — biggest alpha lever in our record)
- **A1 feature substrate** — L2-microstructure (`features_v1`, 59-col) vs free-panel
  bars+metrics+funding vs raw lagged bars vs orthogonal(macro/F&G/on-chain) vs Coinalyze
  cross-exch vs event-conditioners. Evidence: L2-micro **+0.054@180s** ≫ engineered
  rollups+xs+orth **+0.028@8h** ≫ raw bars **~0** (RAWBARS). *Dominant axis.*
- **A2 symbol scope / pooling** — solo vs pooled-all-8 vs pooled-universe(120/500) vs
  cross-sectional groups. Evidence: pooling unlocked starved syms (BNB/LTC 0→.053/.083),
  pooled-ALL +0.040@H180; but high crypto correlation → marginal n_eff per extra symbol →0.
- **A3 history depth / window / regime** — full vs recent-only; regime-conditioned train
  (high-vol vs calm). HD2 used 500d; split 2025-12-10.
- **A4 decision cadence / STRIDE** — every bar vs subsample (24s/48s/1m/15m). Density
  lever, not statistical power (block-bootstrap absorbs autocorr). STRIDE=1 best (recent).
- **A5 sample weighting** — by |move| (R1), recency decay, regime, per-symbol balance,
  vol-normalized. (R1 weighting is itself a form of magnitude emphasis.)
- **A6 split / validation** — honest temporal 70/30 + embargo size; walk-forward vs single;
  purging. (Matched to HD2: global temporal, eval-LTC ≥ split.)
- **A7 normalization** (MED) — per-symbol train-fit z-score · cross-sectional rank/demean ·
  robust/winsorize/clip · none. xs-rank adds cross-sectional signal (EXP-3 +0.003–0.005).

## B. TARGET / LABEL  (HIGH — defines what "success" even is)
- **B1 form** — continuous `r_H` (regression; HD2 rev8 correction) · `sign(r_H)` (classif)
  · first-passage barrier ±f (discards magnitude, bakes exit — AVOID per HD2 rev8) ·
  vol-normalized return · path-stat (max-favorable-excursion / Sharpe-like).
- **B2 horizon H** — the holding time. **The HDX first-cell sweep** {180,600,1800,(+3600)}s.
  Magnitude grows with H → cap_edge can rise with H even as AUC falls (concentrates @H1800).
- **B3 barrier/threshold** — none (continuous, matched) vs ±0.13% vs triple-barrier (TP/SL/time).
- **B4 magnitude vs direction emphasis** — pure direction (AUC) vs magnitude×direction
  (cap_edge, the chosen success). 

## C. OBJECTIVE / LOSS  (HIGH — HD2: 3× swing)
- **C1 regression** — `reg:squarederror` · `reg:pseudohubererror` (robust to fat tails) ·
  `reg:absoluteerror` · `reg:quantileerror` (asymmetric/tail).
- **C2 classification** — `binary:logistic` ± |move| weight (R1).
- **C3 custom IC objective** — maximize `corr(pred, r_H)` (grad of −Pearson) = the
  matched-to-Mamba-2 objective; **fixed = IC for the HDX first cell.**
- **C4 learning-to-rank** — `rank:pairwise` / `rank:ndcg` (natural for cross-sectional ranking).
- **C5 eval_metric** (selection/early-stop) — rmse · logloss · auc · **custom cap_edge_gross**.

## D. HYPERPARAMETERS  (LOW expected alpha-impact — but numerous; late cheap confirm)
- **D1 tree structure** — `max_depth`, `min_child_weight`, `max_leaves`, `grow_policy`
  (depthwise/lossguide), `max_bin`.
- **D2 learning** — `eta`/`learning_rate`, `num_boost_round`/`n_estimators`,
  `early_stopping_rounds`.
- **D3 regularization** — `lambda` (L2), `alpha` (L1), `gamma`/`min_split_loss`,
  `min_child_weight`.
- **D4 sampling** — `subsample` (row), `colsample_bytree`/`bylevel`/`bynode`, `sampling_method`.
- **D5 imbalance** — `scale_pos_weight` (classification only).
- **D6 method / constraints** — `tree_method` (hist/approx/exact), `monotone_constraints`,
  `interaction_constraints` (domain priors as constraints).
- **D7 booster** — `gbtree` vs `dart` (tree dropout) vs `gblinear`.

## E. FEATURE ENGINEERING / SELECTION  (MED, within a substrate)
- **E1 subset** — all 59 vs pruned (features_v1 has ~13 dead/constant cols: cross-asset/ETH
  all-zero, large_order≡1, etc.) vs importance-/SHAP-selected.
- **E2 transforms** — raw vs multi-window rollups (momentum/vol over 1h–7d) vs interactions
  vs lags. (Rollups carried the +0.0275@24h long-horizon momentum; raw lags did not.)
- **E3 cross-sectional** — rank/demean across the universe (xs_rank_r24h, breadth, …).
- **E4 constraints** — monotone/interaction priors (e.g., funding, OFI sign priors).

---

## HDX cell map (what's fixed / swept / deferred)
- **First cell (agreed):** FIX objective=IC (C3), target=continuous r_H (B1), data=features_v1
  pooled-8 eval-LTC split-2025-12-10 (A1/A2/A6), metric=cap_edge_gross. **SWEEP H (B2)**.
- **Next candidates:** feature substrate (A1) · pooling/scope (A2) · objective family (C) · normalization (A7).
- **Late/cheap confirm:** hyperparameters (D) — expected within-noise per HD2; verify, don't lead.

STATUS: planning doc; HDX sweep NOT started (awaiting user go). Uncommitted (sits with the
parallel agent's uncommitted HD2 WIP in this checkout).
