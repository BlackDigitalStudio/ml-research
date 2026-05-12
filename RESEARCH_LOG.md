# Research Log — scalper-bot

**Single source of truth for experiment results.** Memory files in `/root/.claude/projects/-root/memory/` remain as append-only audit trail; this file is the canonical reference. Updated after every research session.

**Last updated:** 2026-05-12 (synthesis of all prior sessions, no new experiments).

---

## 1. Glossary — fixed definitions (do not redefine ad-hoc)

| Term | Definition |
|---|---|
| **base rate** | `P(pl_long > 0)` under correct TAKER fees, per-symbol. BTC canonical ≈ 16% (UP+DN); alts (SOL/LINK/etc.) 10-13%. |
| **WR (win rate)** | Fraction of TAKEN trades with **direction-aware realized net PnL > 0**, after TAKER commissions. Not label-WR, not `prec_NF`. |
| **prec_NF** | Classification precision on non-FL labels. **On canonical TB labels** (`y=UP iff pl>0 AND pl>ps AND not fill_miss`) `prec_NF ≡ WR` by construction (Bug B, 2026-05-09). Both metrics are valid; "lift" must be cited with the base it's measured against. |
| **EV/tr%** | Mean realized net PnL per trade, after commissions, % of notional. **Primary frontier metric.** |
| **tr/day** | Trades per calendar day on holdout. Cited alongside EV/tr to anchor the operating point. |
| **net%** | `EV/tr% × n_trades × kelly_fraction`, % of capital over holdout window. Sensitive to Kelly; **never compare nets at different `k`**. |
| **lift** | `WR / base_rate`. Specify the base (canonical-label vs `P(pl_long>0)` — different numbers). |
| **honest val→test** | Threshold picked on val, applied on test. Anything else is post-hoc bias. |
| **CPCV** | Combinatorial Purged Cross-Validation (López de Prado), N=6, k=2 → 15 combos, embargo=0.5%, purge=label_horizon. Yields PBO. |

## 2. Canonical constants

| Constant | Value | Source |
|---|---|---|
| TP_PCT | 0.20% | `CLAUDE.md` strategy spec |
| SL_PCT | 0.10% | `CLAUDE.md` strategy spec |
| SIM_HORIZON | 1300 ticks (130 s) | `src/trainer.py:56` (env-overridable) |
| R:R | 2:1 | TP/SL ratio |
| TAKER commission, win-side | 0.07% round-trip | `rust_ingest/src/live_sim.rs:66` |
| TAKER commission, loss-side | 0.10% round-trip | `rust_ingest/src/live_sim.rs:67` |
| Break-even WR (no commissions) | 33.3% | `1/(1+R)` |
| **Break-even WR (TAKER, full TP/SL outcomes)** | **~40%** | TP+1.0bp − SL−0.85bp commission drag |
| **Break-even WR (TAKER + timeout asymmetry)** | **~42-44%** | timeouts skew loss-heavy in practice |
| FEATURE_KEYS | 49 (old) / 55 (cryptolake) | `src/features.py::FEATURE_KEYS` |
| Holding zone | 60-180 s, **hard floor 60s** | `strategy_timeframe_constraint.md` |

## 3. Frontier — EV/tr at fixed tr/day, by epoch

**The single comparison table.** Each cell = best honest `EV/tr%` at that operating point.

| Date | Setup | Symbols | EV/tr @ best | EV/tr @ ~2 tr/d | EV/tr @ ~10 tr/d | EV/tr @ ~30 tr/d |
|---|---|---|---:|---:|---:|---:|
| 2026-04-29 | xgb solo (49 feat) | BTC | −0.054% | n/a | n/a | −0.039% |
| 2026-05-02 | 8-model vol-scaled + hybrid maker/taker | BTC | **−0.080%** (n=102, ~5/d) | n/a | n/a | n/a |
| 2026-05-09 | cascade_180s canonical 952K | BTC | −0.061% (n=21) | −0.22% | −0.30% | −0.30% |
| 2026-05-09 | per-symbol cascade XGB (Cryptolake, **TAKER labels**) | 8 syms | −0.027% (DOGE) | varies | varies | varies |
| 2026-05-10 | per-symbol XGB regression grid (Cryptolake, **MAKER-first labels**) | 8 syms | **+0.036%** (ETH n=27) | **+0.030%** (DOGE n=8) | n/a | n/a |
| 2026-05-10 | per-symbol XGB grid (Cryptolake, **MAKER-first** revalidation, DOGE step=5.5s) | DOGE | **−0.047%** (best thr +0.06) | n/a | n/a | n/a |
| 2026-05-10 | LINK TCN lookback=1000 (Cryptolake, **TAKER labels**) | LINK | **−0.040%** | −0.040% | n/a | n/a |
| 2026-05-10 | SOL TCN lookback=1000 (Cryptolake, **TAKER labels**) | SOL | −0.077% | n/a | **−0.077%** | n/a |
| 2026-05-10 | SOL Mamba lookback=3000 (Cryptolake, **TAKER labels**) | SOL | −0.065% | −0.065% | n/a | n/a |

**Reading the frontier:**

- BTC-only era → best EV/tr ~ −0.06% to −0.08% on operating points with n_trades > 100.
- Cryptolake (alts, sequence models, **TAKER labels**) → best EV/tr **−0.040%** (LINK TCN). At matched coverage, ~3-4× improvement vs old.
- Cryptolake (alts, **MAKER-first labels**) → best EV/tr +0.036% ETH was found in one session, **but revalidation with maker-first labels integrated into pipeline showed DOGE = −0.047%/tr** (the +0.036% was likely TAKER-label artifact baked into the build script default).
- **No setup has confirmed positive EV/tr under realistic MAKER-first labels** as of 2026-05-12.

## 4. Resolved confusions (high-cost-to-relearn)

| Confusion | Resolution |
|---|---|
| "WR was 76-85% in old runs" | **Label-artifact** (2026-04-14): `target_pnl > 0 ⟺ y != FLAT`. Was measuring "fraction of taken samples whose TB label is non-FLAT", not realized direction-aware PnL. After fix, honest WR ≈ 20%. |
| "WR ≡ prec_NF on canonical labels" | **By construction** (2026-05-09 Bug B): `y=UP iff pl>0 AND pl>ps AND not fill_miss` → `pred==y ⟺ realized>0` for non-FL. Both are valid metrics but it's the same number on canonical labels. |
| "DOGE +3.9%/month, +19.6%/month TAKER" | **Wrong fees** (COMM_WIN=0.04, COMM_LOSS=0.07 are MAKER round-trip). Correct TAKER no-VIP = 0.07/0.10. Adjusted: +3.9% → −1.5%/month under correct fees. |
| "CPCV best_total = sum across 15 combos" | **5× overlap inflation**: each unique trade appears in 5 of 15 combos at N=6/k=2. Correct: `sum_per_30d = (best_total / 5) × 30 / days_total`. |
| "phase56 +1.30%/30d aggregate positive" | **Labels were TAKER** despite intending MAKER-first. Build script default `entry_long=ask, entry_short=bid` = taker entry; maker-first relabel was applied separately but never copied back to cache `pl/ps`. Real maker-first revalidation: DOGE −1.4%/month. |
| "Vol-scaled grid WR = 0.6-3%" | **Kelly multiplier bug** (2026-05-02): `cfg.tp/cfg.sl` were multipliers but Kelly formula treated them as percentages → `kelly_size = 0` for all → false WR. Fixed via per-sample Kelly in `compute_metrics`. |
| "Maker fill check missing" | **Fixed 2026-05-02**: added `entry_fill_window_ticks` to `LiveSimConfig`. At fill_window=10 (1 s @ 100 ms), **77.6% of samples don't get maker fill** — adverse selection is brutal. Real edge was 7× worse than optimistic backtest showed. |
| "n_folds=1/2 in v8 skips folds" | **Documented, not fixed**. For `n_folds=K`, last fold has `te_end=n=va_end` → skip. Workaround: use `n_folds≥3`. |
| "v3-v8 sequence training used wrong early stop" | **Critical bug, not fixed in v8**: unweighted BCE for early stopping on class-imbalanced binary. Old methodology used `f1_up+f1_dn` (NN) or `prec_NF × sqrt(coverage)` (Optuna). Must fix before next training run. |

## 5. What works structurally

- **CPCV proper** (N=6, k=2, 15 combos, embargo, purge) — reliable validation; PBO calc works.
- **Direct PnL regression** (XGBRegressor on `pnl_long`, `pnl_short`) — best ML baseline. Beats cascade variants, MLP, pooled.
- **Liquidation features** — confirmed `10.7% combined gain` on s2 (UP/DN). Rank 11/15/17 of feat importance.
- **Per-symbol training** — beats pooled XGB (delta ~0) and pooled MLP (loss plateaus epoch 1).
- **Cascade XGB** (FL/NON-FL + UP/DN) — `+3.5/+4.4/+4.1/+3.4 pp` prec_NF vs single 3-class per horizon, but pairwise correlation single↔cascade = 0.977-0.980 → diversity benefit marginal.

## 6. What does NOT work (don't re-try without new reason)

- Pooled XGB / MLP cross-symbol — washes out symbol-specific patterns.
- Isotonic calibration on OOF subset — adds variance more than fixes bias.
- L2 stacker xgb-on-softmax over 4 correlated archs — stacker can't beat AVG when correlation > 0.97.
- Cost-aware loss variants (B: CE×|pnl_diff|; A: y_net relabel) — −4 to −7 pp prec_NF. Re-labels boost non-FL coverage at the cost of precision.
- LdP abstention meta on 23 regime features — OOF lift 5 pp; zero holdout transfer.
- Derivable directional features (oi_velocity, mark_basis) — zero lift, in-noise.
- Winsorize @ p99.9 — 0 effect on prec_NF; XGB hist-binning robust to outliers.
- Binary `P(profit > 0)` classifier — too coarse; ≈ base rate WR.

## 7. Active hypotheses (ordered by expected lift)

| # | Hypothesis | Expected lift on EV/tr | Cost |
|---|---|---|---|
| 1 | **Mamba/sequence models on lookback=10K-100K** | unknown; SSM strength emerges at long-context, untested | $50-100 |
| 2 | **Cross-symbol BTC-lead features for alts** (BTC depth/aggTrade as feature for SOL/LINK/etc. models) | OLD: eth_features 6.68% combined gain on s2 → similar order for BTC-lead | cache rebuild |
| 3 | **Multi-axis ensemble**: Mamba + TCN + Transformer + XGB → L2 stacker | low ensemble diversity historically, but archs are different families | model training |
| 4 | **MAKER-first labels integrated in build pipeline** (currently a P0 blocker: cache `pl/ps` are TAKER by default) | aligns backtest to live execution — likely reveals losses we currently hide | code change ~30 min + cache rebuild |
| 5 | **Wider TP/SL with longer timeout** (e.g. 0.30/0.15/600s) — reduces timeout-asymmetry bias | unknown; tested only narrowly | cache rebuild |
| 6 | **SSL pretraining on raw LOB** → fine-tune triple-barrier | unknown, novel for this dataset | $150-250 (8× H100) |
| 7 | **Cross-pair attention** (Mamba on alt + BTC LOB simultaneously) | unknown | $50+ |
| 8 | **Liquidation data — higher-fidelity** (current is frequency-only, not maker/taker-side breakdown) | likely 1-2 pp WR | data procurement |
| 9 | **Dynamic TP/SL per regime** (wide TP when liq-imbalance high) | structural fix for asymmetric loss | medium |
| 10 | **VIP fee tier** (0.04 maker / 0.07 taker) | +0.03% per trade (mechanically) — moves break-even WR down 2-3 pp | requires $1B+/month volume = downstream |

## 8. Cache inventory

| Cache | Status | Notes |
|---|---|---|
| `samples_v3_BTCUSDT_canon_60000h_1778274003` | **LIVE on Contabo** (`/home/scalper/scalper-bot/data/_cache/`) | 952K samples × 1800-tick mid_paths × 49 feat. Canonical era. All sidecars present. |
| `samples_v3_60000h_1777593610` | **DELETED** | 1.85M × 52 feat (49 + 3 derivable). Lost in OOM rebuild. |
| `samples_v3_BTCUSDT_swing30m_unfilt_60000h_1777734022` | LIVE on Contabo | 192K × 18000-tick. 30min swing experiment cache. |
| Cryptolake 8-sym caches | **LOST** (Vast.ai + Runpod terminated) | Rebuildable from `gs://blackdigital-scalper-data` in 30-45 min/symbol with workers=32. |
| Cryptolake source data on GCS | **PERSISTENT** | `gs://blackdigital-scalper-data` (europe-west1). 287.9 GB raw, 1.3 GB features cache. 8 symbols, BINANCE_FUTURES. |

## 9. Code state (as of 2026-05-12)

| Component | State |
|---|---|
| `rust_ingest/src/live_sim.rs` | TAKER fees 0.07/0.10 default. `NotFilled` exit reason. `simulate_trade_hybrid` with taker fallback. Per-sample Kelly fix in `grid_sim.rs`. |
| `rust_ingest/src/bin/sim_labels.rs` | Has `--entry-taker-long/short` for maker-first hybrid. 150ms fill latency, 1s entry fill window canonical. |
| `rust_ingest/src/features.rs` | NUM_FEATURES = 67 (after Cryptolake +8 liquidation cols). |
| `src/features.py` | `_RAW_NUM_FEATURES = 67`, `NUM_FEATURES = 55` after extended `DROP_RAW_INDICES`. |
| `scripts/build_cryptolake_cache.py` | TAKER labels by default. Maker-first relabel **not integrated** (P0 blocker). `--save-mid-paths`, `SCALPER_SAVE_DAY_TICKS`, vol-scaled TP/SL, `--eth-leading` flag. |
| `scripts/train_seq_v8.py` | TCN + Mamba sequence trainer. **Wrong early-stop metric** (unweighted BCE). Must fix before next run. |
| `scripts/cpcv_proper.py` | N=6, k=2, embargo, purge, PBO. Working. |
| Live bot models in `models/` | **EMPTY**. Bot runs in data-collection-only mode. Last weights drained 2026-04-14 during methodology overhaul. |
| Train↔live gap | Research pipeline writes `.pt` + XGB `.json` that **don't match** `HybridModel`'s load format. New inference module needed before paper-trade. |

## 10. Memory pointers (for deep-dive only)

These contain raw session notes. The frontier table above subsumes their decision-relevant content.

- `methodology_bugs_2026_04_14.md` — original 2 bugs (label-WR artifact, full-val stacker fit).
- `experiments_2026_05_01_signal_exhaustion.md` — 5 levers session, BTC era exhaustion.
- `experiments_2026_05_02_methodology_overhaul.md` — maker fill check + Kelly fix.
- `experiments_2026_05_02_swing_attempt.md` — 30 min swing attempt, holdout net=-0.004% at n=9.
- `experiments_2026_05_09_cascade_canonical.md` — cascade vs single, TP/SL grid on cascade_180s.
- `cryptolake_phase0_2026_05_09.md` — 8-symbol cache build, vol-scaled TB, liq features.
- `cryptolake_phase23_2026_05_09.md` — cascade XGB on 8 symbols, pooled training.
- `cryptolake_phase56_2026_05_10.md` — maker-first first POSITIVE EV (**but later found to be TAKER-label artifact**).
- `cryptolake_state_2026_05_10_v2.md` — TAKER vs MAKER reset, DOGE step=5.5s = −1.4%/month real.
- `cryptolake_experiments_2026_05_10_final.md` — 2-day sequence model summary, best −0.040% LINK TCN.

## 11. Update protocol (for Claude)

**Every research session that produces a number:**

1. Append/update row to **Frontier table** (§3) — keep one row per experimentally-distinct setup.
2. If a methodology bug is found/fixed: add row to **Resolved confusions** (§4).
3. If a hypothesis is tested: move it from **Active hypotheses** (§7) to **Frontier** (§3) or **Doesn't work** (§6), with result.
4. If new cache built / old cache deleted: update **Cache inventory** (§8).
5. If code constants change: update **Canonical constants** (§2) and **Code state** (§9).
6. Memory files in `/root/.claude/projects/-root/memory/` still get written per usual auto-memory rules. **Do not duplicate** their full content here — only the decision-relevant rolling state.

**Never:**

- Cite an EV/tr% without `tr/day` and fee regime (TAKER/MAKER).
- Compare nets at different Kelly fractions without renormalizing.
- Report "WR > X%" without specifying base rate AND that it is direction-aware realized.
- Quote a result from this file without verifying against the cited memory or live code (per global Law #2 — facts > theories).
