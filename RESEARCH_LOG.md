# Research Log — scalper-bot

**Structured source of truth is now `research/` (JSONL ledger + SQLite).**
This file is the human narrative; the queryable asset is
`research/experiments.jsonl` + `research/hypotheses.jsonl`, contract in
`research/schema.sql`, plan in `research/PLAN.md`. §3 below is regenerable
via `python3 research/ledger.py frontier` — do not hand-maintain it once
new results land. The ledger *refuses* a result without its fee regime /
cache / split provenance (the chaos that cost us 3 false positives).

> **Infra state 2026-05-16 (critical — do not lose):** Contabo
> `root@84.247.154.229` is **LOST**. Every "LIVE on Contabo" cache (§8) and
> the entire `/root/.claude/projects/-root/memory/*.md` archive are **gone**
> — the §10 memory pointers and STRATEGY.md host references are historical.
> New topology: this repo's container = planning node (stdlib only); GCP
> `blackdigital.kz` 96 vCPU VM = compute node; `gs://blackdigital-scalper-data`
> (Cryptolake, 287.9 GB, persistent) + `gs://scalper-bot-research-data` =
> the only durable data. See `research/README.md` → *Infra reality*.

> **GCP recon verified 2026-05-16** (ADC as `blackdigital.kz@gmail.com`,
> project `project-26a24ad0-1059-4f73-93b` "My First Project"):
> - `gs://blackdigital-scalper-data` — **ALIVE**, `EUROPE-WEST1`, owned by
>   this project. Layout `features_v1/symbol=<SYM>/dt=<YYYY-MM-DD>/{features.npy,indices.npy}`;
>   symbols are Tardis-style (`BNB-USDT-PERP`, not `BNBUSDT`). The Cryptolake
>   feature asset **survived the Contabo loss**.
> - `gs://scalper-bot-research-data` — **403 / no access** for this account
>   (volaware checkpoints/oof; volaware was refuted — not blocking).
> - Compute europe-west1: `CPUS=200` (usage 0), `N2_CPUS=200`,
>   `DISKS_TOTAL_GB=2458`, **`PREEMPTIBLE_CPUS=0` → no spot** (on-demand
>   only), `C2_CPUS=8` (use **N2** for the 96 vCPU box).
> - Connection: ADC user creds in the ephemeral container's
>   `/root/.config/gcloud` — works this session, **not durable** across
>   container death (re-auth or move to SA-secret to persist).

> **Phase B first end-to-end run 2026-05-17 (`phaseb-20260517-003320`):**
> Lost Cryptolake pipeline **reconstructed and run on GCP** (cargo build /
> GCS / rust sim / XGB / grid / ledger all working). **H5 trust gate
> LANDED**: MAKER_FIRST entry integrated, `parity_ok=True` for LINK & SOL
> on 90 d of real data — every number is now MAKER-first honest. First
> numbers are **not a strategy**: the XGB gate is degenerate (~100 %
> take-rate) → LINK EV/tr −0.001 %, 1267 tr/day, net −23.6 %; SOL −0.001 %,
> 645 tr/day, net −15.8 % (`exploratory` in the ledger). H2 inconclusive
> (all PT/TS configs identical → never engaged on a trade-everything
> baseline). **Next bottleneck = model selectivity / trade selection
> (logged as H12, $0 eval-only).** Over-trading, not PT/TS, is the wall.

> **HA1 alpha screen 2026-05-17 (`phaseb-20260517-123148`, 8 alpha rows
> in `v_alpha`):** First execution-neutral signal map. `features_v1` is
> **leak-free** (placebo rank-IC ≈ 0 everywhere). Signal is **real but
> ultra-short-lived**: OOS rank-IC ≈ **0.087/0.073 @30 s** (LINK/SOL),
> decaying monotonically to ≈0.02 by 120-180 s; CI excludes 0 for 7/8.
> **Economically dead as a 60-180 s point prediction:** top-decile
> |move| = 3-10 bp, below even the loose 8 bp maker floor (7/8) and far
> below the 13 bp strict floor (8/8); `decile_monotonic = 0` everywhere.
> RL cannot manufacture 13 bp from a 3-4 bp edge → HA1 **refuted as
> posed**. Decisive redirect: the edge lives **faster than the 24 s
> sampling** — promote **HA4 (sub-24 s cadence)** + HA2 (target form);
> NOT execution/RL. (Run salvaged from an empty-id harness bug, fixed.)

> **The symmetry wall 2026-05-17 (MFE/MAE study, $0, LINK+SOL 5d).**
> Decisive structural result. Median max-favorable excursion: 60 s = 3 bp,
> 180 s = 6 bp, **600 s = 13 bp** (≈ strict floor only at 10 min). At
> 60 s only ~5-6 % of windows ever reach ±13 bp. AND it is ~symmetric:
> `P(MFE≥+13bp) ≈ P(MAE≤−13bp)` at every H (180 s: .235/.220; 600 s:
> .506/.462). → Wider TP/SL + longer timeout (why the old grid always
> "won" wider, and why the feature set is volatility-heavy) **scales the
> win and loss tails equally — it creates no edge**; that is why every
> wide-grid config netted ≈0 − costs. The bind: where moves clear cost
> (≥300-600 s) **direction is unpredictable** (HA1 IC→~0 by 180 s);
> where direction is weakly predictable (≤60 s) **moves are 2-4× below
> cost**. The two never overlap. No TP/SL/timeout/execution/RL fixes a
> symmetric-diffusion-vs-fixed-cost gap. Reconciles HA1 (short-horizon
> IC) with old research (TB-barrier favoured long/wide): different
> targets, both true. **Only escape consistent with the data:
> conditional asymmetry — a rare event/regime that breaks the MFE/MAE
> symmetry on the ≥cost subset (→ HA5).** HA4 (faster cadence) CLOSED:
> √t-trap (shorter window ⇒ smaller move ⇒ worse vs fixed cost).
> Cryptolake event data confirmed available at full fidelity for HA5:
> liquidations (side+qty+price, ~246/d), open_interest (~15k/d), funding
> (rate+mark+index, 1/s), trades (~242k/d) — see
> `research/CRYPTOLAKE_SCHEMA.md`.

> **HA5/HA6 — decisive negative 2026-05-17 (`phaseb-20260517-132629`,
> 6 alpha rows).** On the ≥cost subset (first-passage to ±0.13% within
> H∈{180,300,600}s), LINK+SOL: base `P(up|≥cost) ≈ 0.51-0.52` (symmetry
> confirmed). **head2 directional AUC 0.496-0.522 ≈ placebo 0.48-0.52 —
> indistinguishable from chance** for every conditioner (raw liquidation
> side/qty/count, ΔOI, funding rate/basis, all 59 microstructure feats)
> at every H. (`economic_pass_strict=1` on some rows is a NOISE ARTIFACT
> — cap-sign at AUC≈placebo is meaningless; status forced `refuted`.)
> **head1 ≥cost-feasibility AUC ≈ 0.68-0.71 — strong: volatility/regime
> (WHEN a big move comes) IS predictable, but DIRECTIONLESS (WHICH WAY
> is not).** → HA5 refuted; HA6 refuted (cascade head-2 has nothing to
> predict). **Triple-confirmed (HA1 sub-cost direction · MFE symmetry ·
> HA5 no conditional asymmetry): LINK/SOL LOB+event data contains
> predictable volatility but NO predictable direction at any
> horizon/conditioner.** A directional scalp here is structurally
> non-viable — not fixable by model/features/RL/execution. The cheap
> LOB-directional search space on these alts is **mapped and empty**.
> Open decision **HZ1** (strategy-class pivot, priority 1): non-
> directional vol-harvest (needs options / both-sided MM = different
> instrument), different signal source (cross-asset lead-lag / longer
> timeframe / higher-fidelity events), different asset class, or accept
> no directional alpha here. **Needs a human decision before more compute.**

> **SCOPE CORRECTION 2026-05-17 (over-claim retracted, user challenge).**
> The HA5/HA6 block above is correct about *what was measured* but the
> phrase "directional scalp non-viable / no directional alpha / pivot
> strategy class" **outran the evidence**. Everything run for direction
> (HA1, HA5/HA6, phaseb-003320) shares ONE unvaried slice: **GBT (XGB)
> on a single-tick flattened `features_v1` snapshot** (lost-provenance,
> unverified) + a few hand event aggs; no sequence/temporal model, no
> target-form/feature/cross-asset/ensemble variation, no HP search. The
> negatives bound **only that slice**, not achievable directional
> predictability. Critically, the only prior *honest* best result was a
> **sequence model (LINK TCN −0.040), never reproduced MAKER-first** —
> snapshot-GBT discards the order-flow *dynamics* where short-horizon
> direction lives. **HZ1 (pivot) RETRACTED → refuted.** Real open
> surface = **HD1** (priority 1): direction-improvement cluster —
> HA2 target-form + HA3 feature work ($0 on built cache), then a
> **temporal/sequence model screen** (the conspicuous untested gap).
> Not a strategy-class pivot; a model/representation pivot, still cheap.

**Last updated:** 2026-05-17 (SCOPE CORRECTION: HA5/HA6 negative bounds only the XGB-snapshot slice; "no directional alpha/pivot" RETRACTED; sequence/target-form/feature levers UNTESTED → HD1 priority 1; prior: HA5/HA6, symmetry wall, HA1, Phase B, H5).

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
| 2 | **Inner PT/TS params via fused grid_sim** (partial_tp_progress, trailing_step{1,2}_progress/_sl_ratio, trailing_step1_sl_floor_pct). Structurally addresses the main 2026-05-09 bottleneck — full-SL losses (-0.14% net) dominate timeout-wins (+0.005-0.06%). Partial TP locks gain on winning side, trailing SL closes earlier on losing side → asymmetric tail compresses. **Not tested on Cryptolake-era data or under MAKER-first labels.** Fused `grid_sim` binary already supports the 11-param sweep (~30s per 100K configs on Contabo). Wrapper: `src/rust_bridge.py::simulate_labels_grid`. | likely 0.02-0.05% per trade if winning-side avg moves from timeout-drift (~0.04%) toward TP-hit (~0.16%) on subset of trades | model already trained → eval only, ~1 hr Contabo |
| 3 | **Cross-symbol BTC-lead features for alts** (BTC depth/aggTrade as feature for SOL/LINK/etc. models) | OLD: eth_features 6.68% combined gain on s2 → similar order for BTC-lead | cache rebuild |
| 4 | **Multi-axis ensemble**: Mamba + TCN + Transformer + XGB → L2 stacker | low ensemble diversity historically, but archs are different families | model training |
| 5 | **MAKER-first labels integrated in build pipeline** (currently a P0 blocker: cache `pl/ps` are TAKER by default) | aligns backtest to live execution — likely reveals losses we currently hide | code change ~30 min + cache rebuild |
| 6 | **Wider TP/SL with longer timeout** (e.g. 0.30/0.15/600s) — reduces timeout-asymmetry bias | unknown; tested only narrowly | cache rebuild |
| 7 | **SSL pretraining on raw LOB** → fine-tune triple-barrier | unknown, novel for this dataset | $150-250 (8× H100) |
| 8 | **Cross-pair attention** (Mamba on alt + BTC LOB simultaneously) | unknown | $50+ |
| 9 | **Liquidation data — higher-fidelity** (current is frequency-only, not maker/taker-side breakdown) | likely 1-2 pp WR | data procurement |
| 10 | **Dynamic TP/SL per regime** (wide TP when liq-imbalance high) | structural fix for asymmetric loss | medium |
| 11 | **VIP fee tier** (0.04 maker / 0.07 taker) | +0.03% per trade (mechanically) — moves break-even WR down 2-3 pp | requires $1B+/month volume = downstream |

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
