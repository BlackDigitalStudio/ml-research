# scalper-bot — strategy & architecture

Automated BTCUSDT Perpetual scalping on Binance Futures.
`x20 leverage`, `$50` starting deposit, holding zone `60-180 s`.

Host: Contabo `root@84.247.154.229` — 16 vCPU AMD EPYC, 62 GB RAM, 581 GB
SSD. Single primary host for recorder, research and (future) live
execution. Vultr Tokyo is retired.

This document is the repo-side source of truth for architecture and the
current state of the project. The deep-dive reference memos live in
`/root/.claude/projects/-root/memory/` on the host — see the **Further
reading** section at the bottom.

---

## 1. Two parallel systems (important)

There are two code paths in this repo and they are **not wired to each
other** yet. Any plan that mixes them has to cross the train→live gap
explicitly.

### 1.1 Live bot (production path)

Entrypoint `main.py` → `src/model.py::HybridModel`. Legacy CNN encoder +
classical ensemble. Hot-loads these files from `models/`:

    encoder_latest.pt         CNN LOBEncoder, input (3, 20, 50)
    xgb_{0,1,2}_latest.json   3× XGBoost (seed 42/123/456, bagging)
    lgb_latest.txt            LightGBM
    logreg_latest.pkl         LogisticRegression on top-5 feature idxs
    logreg_features.json      indices of those top-5 features
    calibrators_latest.pkl    3× isotonic calibrators (one per class)

Voting rule: ≥ 3/5 agreement on non-FLAT. Uncertainty gate: per-model
max-prob spread > 0.20 → FLAT. Calibrated mean-prob drives the
confidence threshold. Training side for this bundle lives in
`src/trainer.py::train_ensemble` (called manually / via cron).

### 1.2 Research pipeline (not yet deployable)

Everything under `src/models/` plus `scripts/bakeoff_v3.py` +
`scripts/infer_primaries_v3.py` + `scripts/grid_live.py`.

    build_samples_cached          v3 cache (49 features, 1M-sample target)
        │                          via Rust feature_builder + sim_labels.
        ▼
    bakeoff_v3.py                 22-arch factory, per-arch recipe,
        │                          per-epoch checkpoint, OOM-retry.
        ▼
    {tag}_best.pt × N             neural primaries
        │
        ▼
    infer_primaries_v3.py         softmax inference on the full cache
        │                          → primary_softs_v4.npz
        ▼
    train_stacker (XGBoost L2)    walk-forward 75/25 split on train only.
        │
        ▼
    train_meta (XGBoost L3)       López-de-Prado meta-labeller: precision
        │                          gate conditional on primary prediction.
        ▼
    grid_live.py                  direction-aware grid: TP × SL × timeout
                                    × partial × trailing × kelly × meta_thr
                                    × min_prob × spread_bps × fill_prob
                                    → rank by dd_adj / sharpe / net_return.

**Train→live gap.** The research pipeline writes neural `.pt` weights and
XGBoost `.json` models that DO NOT match the `HybridModel` load format.
Before any research winner can be paper-traded, a separate inference
module has to be written that (a) loads the top-K neural primaries, (b)
runs the L2 stacker and L3 meta, (c) exposes the same `(class, conf)`
contract `signal.py` consumes. This is tracked as a P0 blocker for
paper-trade, and is NOT part of the one-shot training itself.

---

## 2. Architectures in the bake-off (22 total)

Gated behind `SCALPER_ENABLE_HEAVY_ARCHS=1` for the foundation + LLM
tiers. Per-arch recipe lives in `scripts/bakeoff_v3.py::ARCH_RECIPES`.

| Tier | Archs | Recipe summary |
|---|---|---|
| From-scratch | `transformer`, `tcn`, `patchtst`, `mamba`, `hybrid_mamba_attn` | 25 epochs, lr 3e-4, warmup 1k, patience 5 |
| SSL warm-start | `patchtst_pretrained:<path>` | 10 epochs, lr 1e-4, head lr ×10 |
| Frozen foundation | `chronos_bolt_{tiny,mini,small,base}`, `chronos_base_multi`, `timesfm_2p5_{200m,multi}`, `moment_large{,_multi}` | 12 epochs, head lr ×3, bs 512 |
| Unfrozen foundation (LoRA) | `chronos_base_unfrozen`, `timesfm_2p5_unfrozen`, `moment_large_unfrozen` | 8 epochs, LoRA r=16 on q/k/v/o, layerwise LR decay 0.9 |
| Time-LLM | `time_llm_0p5b`, `time_llm_1p5b`, `time_llm_7b_4bit` | 4-5 epochs, LoRA r=8 on q/v, bf16 (4-bit NF4 for 7B) |

Trainer: `src/models/train_efficient.py`. Early-stop metric `f1_up +
f1_dn` (never f1_macro — see pitfall #8 in the ROADMAP).

---

## 3. Features (49)

`src/features.py::FEATURE_KEYS` is the source of truth. Stage A-E of the
2026-04-15 feature overhaul landed the horizon-tier additions and pruned
7 low-information legacy slots.

- Original (34) → horizon momentum/vol (+6, Stage A) → horizon OFI + funding (+5, Stage B) → structural microstructure (+5, Stage C) → cross-exchange + ETH (+6, Stage D) → prune 7 (Stage E) = **49**.
- Stream-side implementation: `src/features.py::FeatureEngine` + `src/features_ext.py::FeatureExtEngine`.
- Batch-side (training): `src/trainer.py::_calc_features_batch` + Rust `rust_ingest/src/features.rs`.
- Byte-exact Rust↔Python parity verified by `scripts/parity_rust_features.py` (max|Δ| = 0 on all 49 cols).
- Rust emits 56 raw cols; Python applies `features.KEPT_RAW_INDICES` slice to get 49. Any caller that wants the raw 56-col path must pass `_return_raw=True`.

Normalisation: z-score over the trailing 300-tick (30 s) window.

---

## 4. Rust pipeline

`rust_ingest/` crate provides byte-parity ports of the hot paths and is
the **default** (`SCALPER_USE_RUST=1` unset → use Rust; set
`SCALPER_USE_RUST=0` only for parity debugging). Missing binaries raise
`RustBinariesMissing` — no silent Python fallback.

| Binary | Role | Perf |
|---|---|---:|
| `feature_builder` | 49-col feature computation (clap CLI over parquet) | ~20× over Python |
| `sim_labels` | Forward-sim triple-barrier labeller (rayon-parallel) | ~91k samples/s, 100-300× over Python loop |
| `build_samples` | Row-group-streaming training cache builder (100 MB peak on 65 M rows) | flat-RAM on 1M-sample runs |
| `depth_parser` | Tardis CSV → binary snapshot parser (legacy) | 517k rows/s |

**Default path** (`SCALPER_USE_RUST=1`): Python loader reads compacted
parquets into memory, Rust computes features + labels.
**Direct-Rust path** (`SCALPER_USE_RUST_DIRECT=1`): requires prebuilt
`data/_merged/*.parquet` via `scripts/merge_streams.py`. Peaks at ~24 GB
RSS on a 65 M-depth / 1 M-sample build — this is the path to use for the
bake-off-scale cache.

Build: `cd rust_ingest && cargo build --release` (~18 min first time,
~5 min incremental on Contabo).

---

## 5. Strategy constants (triple-barrier labels)

`src/trainer.py`:

    WINDOW_SIZE = 50     (5 s of 100 ms LOB snapshots fed to CNN)
    SIM_HORIZON = 1300   (forward ticks handed to live_sim.simulate_trade)
    TP_PCT = 0.20        (upper barrier, 2:1 R:R vs SL base)
    SL_PCT = 0.10        (lower barrier)

Labels use **direction-aware forward simulation**: each sample is
labelled UP if a LONG would win before losing, DOWN if a SHORT would
win, otherwise FLAT. This is honest (López de Prado AFML Ch.3) and
matches live trading economics.

**`target_pnl` is NOT realised PnL** — it is the TB-winner PnL (the
label's own outcome). For honest direction-aware evaluation use
`rust_bridge.simulate_labels` which returns `pnl_long` + `pnl_short`;
`grid_live.py` already does this. Full context in
`/root/.claude/projects/-root/memory/methodology_bugs_2026_04_14.md`.

---

## 6. Timing zone (deploy commitment)

From the 14,400-config direction-aware grid of 2026-04-14 on the 63 k
v3 cache (old 34-feature set): **0 profitable OOS configs**, smallest
BE gap at R:R ≈ 2:1. The conclusion is a zone, not a point.

| parameter | zone | rationale |
|---|---|---|
| holding period (timeout) | **60-180 s** | shorter → maker-fill + commission floor; longer → SIM_HORIZON cap |
| TP (take-profit) | **0.20-0.30 %** | TP < 0.10 → BE WR > 73 %; TP > 0.40 → trade frequency dies |
| SL (stop-loss) | **0.10-0.15 %** | keeps commission-adjusted BE WR in 45-52 % range |
| R:R (TP/SL) | **1.5 : 1 — 2.5 : 1** | commission-favoured sweet spot |
| Kelly fraction | ≤ 0.25 | aggressive sizing always loses OOS; no reason to expect different on 1 M |

Sub-second latency is **not** a blocker at this horizon. Live-execution
Rust port stays deprioritised until the strategy shifts to sub-second
scalping.

---

## 7. Business-facing metrics (source of truth)

Owner-defined metric set (repo file `Самые важные метрики`). Every
backtest must surface these:

1. % закрытых на полный TP (`tp_hit`)
2. % закрытых на полный SL (`sl_hit`)
3. % закрытых по таймауту (`timeout_limit` + `timeout_market`)
4. % закрытых по трейлинг-стопу (`trailing_sl_1/2`, `partial_plus_trailing_sl_1/2`)
5. % закрытых на частичный TP только (`partial_plus_tp`)
6. Прибыль/расход до комиссий и спрэда ($ + %)
7. Прибыль/расход после комиссий и спрэда ($ + %)

Invariants (enforced by `tests/test_business_metrics.py`):
- Metrics 1-5 are mutually exclusive → sum = 100 %.
- Metric 6 ≥ metric 7 in absolute terms.
- Exit-reason taxonomy lives in `src/live_sim.py::TradeOutcome.REASONS`.

---

## 8. Current state (honest, 2026-04-16)

- **14 primary architectures trained** on Modal (~$6 of $30 monthly credit). All weights on Modal Volume + mirrored to `/tmp/metrics/bakeoff_v3/` on Contabo.
- **L2 XGBoost stacker + L3 meta + timing-zone grid** ran end-to-end for the first time with direction-aware `pnl_long`/`pnl_short` via Rust `simulate_labels`.
- **Meta AUC = 0.883** (precision 0.873, recall 0.633) on 4,513-sample val. Strong classification-level signal.
- **0 / 46,080 profitable configs** in the zone grid (TP 0.15-0.30, SL 0.10-0.15).
- **0 / 36,864 profitable configs** in the wide grid (TP 0.25-0.50). But the break-even gap at R:R 2:1 is now **+7.5 pp** (was +14.6 pp on 63k / 34-feature baseline — halved) and at **R:R ≥ 2.67 it is negative** (WR > nominal BE).
- **Net is still ≈ −0.86 % on the best configs** because fixed `(TP, SL, timeout)` let timeout-exits dominate. The live executor's `_adaptive_tp_sl` + `dynamic_timeout_ticks_from_vol_ratio` were NEVER invoked in the grid — the research and execution strategies have diverged at the policy level, not just the weights level.
- **1 M cache** — only `X_feat.npy` partially built; we trained on the 93 k-sample 50 h cache. Rebuilding the 1 M cache is deprioritised against structural upgrades (see §11).
- **No live bot** — `main.py` has never executed trades. `src/model.py::HybridModel` has no loadable weights. No paper-trade ever happened. The "train↔live gap" is **not** a blocker until a profitable policy exists; once it does, we swap `HybridModel` for a new inference module.

Full numbers: `docs/SESSION_2026_04_16_RESULTS.md`.

See the 2026-04-16 memo in `/root/.claude/projects/-root/memory/session_2026_04_16_iql_pivot.md` for the pivot plan.

---

## 9. Script inventory

### Active — part of the current pipeline

| Script | Role |
|---|---|
| `merge_streams.py` | Streaming Python ingest of hourly parquets → `data/_merged/`. One-time, 7 min, 1.9 GB peak RSS. |
| `build_cache.py` | Wrapper around `Trainer.build_samples_cached` that prints peak RSS + wall time + cache file sizes. |
| `bakeoff_v3.py` | **Canonical training runner.** Hardware-aware, 22-arch factory, per-arch recipe, per-epoch checkpoint, OOM retry. |
| `bakeoff_v1.py` | Provides `build_factory(arch)` — imported (not executed) by `bakeoff_v3.py` and `infer_primaries_v3.py`. |
| `infer_primaries_v3.py` | Re-infer primary softmaxes on the v3 cache → `primary_softs_v4.npz`. |
| `grid_live.py` | **Authoritative direction-aware grid**, consumes primary softmaxes + v3 cache sidecars + Rust `simulate_labels`. |
| `fix_stacker_classweight.py` | Balanced-stacker retrain (walk-forward 75/25). |
| `honest_eval.py` | Reference impl of direction-aware realised PnL (debug helper). |
| `recover_stacker.py` | Rebuild ensemble artefacts from saved `.pt` checkpoints when bake-off crashes. |
| `parity_rust_features.py`, `parity_rust_live_sim.py` | Regression harnesses for Rust↔Python parity — run before any Rust change. |
| `pretrain_ssl.py` | SSL pretraining for `patchtst_pretrained:<path>`. |
| `migrate_legacy_depth.py` | One-shot migration of legacy `list<list<f64>>` depth parquets → flat `FixedSizeList<f64,20>`. Kept as runbook reference. |
| `download_tardis_free.py`, `tardis_csv_to_parquet.py`, `ingest_tardis.py` | Tardis historical data pipeline. |
| `record_data.py` | Standalone recorder entry point. Primary recorder runs as the `scalper-recorder.service` systemd unit. |
| `check_api.py`, `check_data.py`, `check_quality.py`, `check_sweep.py`, `clean_cache.py` | Ops / diagnostics. |

### Legacy — do not extend, keep for reference

These scripts are superseded by the pipeline above but not deleted
(either referenced transitively or useful as historical baselines). Do
not add features to them; flag as stale if you read them.

| Script | Superseded by | Reason |
|---|---|---|
| `bakeoff_v2.py` | `bakeoff_v3.py` | `v2` uses `train_generic` (no LoRA, no per-arch recipe, f1_macro early stop). |
| `bakeoff_3gpu.sh`, `bakeoff_parallel.sh` | `bakeoff_v3.py` (built-in HW scaling) | Predate per-arch recipes and per-epoch ckpt. |
| `grid_search.py`, `grid_test_ensemble.py` | `grid_live.py` | `grid_test_ensemble.py` uses `target_pnl` as realised PnL (methodology bug). |
| `backtest.py`, `backtest_ensemble.py`, `backtest_teacher.py` | `grid_live.py` + `live_sim.py` | Pre-direction-aware metric. |
| `run_full_pipeline.sh` | Invoke `build_cache → bakeoff_v3 → infer_primaries_v3 → grid_live` manually | Still points at `bakeoff_v2.py` + `backtest_ensemble.py`. |
| `audit_f1_bug.py` | — | One-shot diagnostic for the F1 convergence pitfall (fixed in `bakeoff_v3`). |
| `b3_regime_grid.py`, `c5_eval_balanced_stacker.py` | `grid_live.py` | Research one-offs. |
| `step2_regime_gate.py`, `step3_live_sim_grid.py`, `step4_fqi_sizer.py` | Separate modules under `src/models/` | Research stages — kept for reproducibility. |
| `d2_offline_rl_sizing.py` | — | Contextual-bandit sizing; artifact-inherited false OOS positive, needs redo on honest PnL. |
| `train_initial.py` | `build_cache.py` + `bakeoff_v3.py` | Pre-cache trainer entry point. |
| `grid_test_ensemble.py` | `grid_live.py` | Old grid driver explicitly replaced by grid_live. |

---

## 11. Next architecture — offline RL (IQL) for per-trade adaptive policy

**Why** — the grid's fixed `(TP, SL, timeout, kelly)` leave profit on the table. At R:R 3:1 with `TP = 0.45 %, timeout = 120 s`, the dominant exit is `timeout_*` which captures tiny mid-moves, not the full TP. WR can exceed nominal BE and still net negative. Adaptive per-trade parameters are a structural upgrade, not a hyper-parameter tweak.

**Architecture** (committed 2026-04-16, carte blanche):

```
(lob, feat)
    ↓
[regime_moe] — hard-gate into 1 of 4 expert ensembles (high-vol / low-vol / trending / mean-rev)
    ↓
[14 primaries × 3 softmaxes per ensemble] → (B, 42)
    ↓
[L2 CatBoost stacker] → P(UP|DOWN|FLAT) + uncertainty  (replaces XGBoost on L2)
    ↓
[L3 neural meta, retarget net_pnl > 0] — target is realised profit not TB-label match
    ↓
[IQL policy head] — state = [softs 42 + feat 49 + regime_id + uncertainty],
                   action = (TP_bucket × SL_bucket × timeout_bucket × kelly_bucket × direction),
                   reward = realised net PnL via Rust simulate_labels
    ↓
[simulate_trade()] — single parameterised simulator (grid + live use the same one)
```

**Day 1 (2026-04-16):** meta retarget + (state, action, reward) dataset generator + CatBoost L2.
**Day 2-3:** IQL implementation + training, walk-forward eval.
**Day 4:** `regime_moe` rewire onto Stage A-D features (Stage E pruned `vol_ratio` / `hurst` which the old regime_moe used), retrain regime-specialists, retrain regime_classifier.
**Day 5:** final end-to-end grid with the IQL policy head driving parameters.

## 12. Further reading

The authoritative prioritised-work memos live in
`/root/.claude/projects/-root/memory/` on Contabo:

- `READ_FIRST_NEXT_STEPS.md` — current P0 / P1 / P2 with do/don't list.
- `ROADMAP_2026_04_15.md` — research tracks post-refactor.
- `feature_overhaul_2026_04_15.md` — Stage A-E details.
- `methodology_bugs_2026_04_14.md` — `target_pnl` label-artefact + stacker leak. Mandatory before writing any new evaluator.
- `business_metrics_canonical.md` — contract for the 7 metrics + test.
- `grid_60_180s_2026_04_14.md` — 14,400-config sweep that identified the timing zone.
- `contabo_primary_host.md` — host state + retired-Vultr notes.
- `rust_pipeline.md` — Rust build + parity notes.
- `user_preferences.md`, `feedback_*.md` — terminology, writing style, anti-perfectionism, anti-downgrade.
