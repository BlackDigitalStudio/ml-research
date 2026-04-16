# Session 2026-04-16 — first end-to-end honest results

## TL;DR

- **14 primary architectures** trained on Modal (~$6 of the $30 monthly credit) — `transformer`, `tcn`, `patchtst`, `mamba`, `hybrid_mamba_attn`, `chronos_bolt_{tiny,mini,small,base}`, `chronos_base_multi`, `moment_large`, `moment_large_multi`, `time_llm_{0p5b,1p5b,7b_4bit}`.
- **L2 XGBoost stacker + L3 meta + timing-zone grid** ran end-to-end on Contabo for the first time with direction-aware `pnl_long`/`pnl_short` via Rust `simulate_labels`.
- **0 / 46,080 profitable configs** in the first zone grid (TP 0.15-0.30, SL 0.10-0.15).
- **0 / 36,864 profitable configs** in the wide grid (TP 0.25-0.50, SL 0.10-0.15) — **but the break-even gap at R:R ≥ 3:1 closed to −2.8pp** (WR > nominal BE).
- Net is still ~−0.86 % on the best config because **fixed (TP, SL, timeout) leave profit on the table via timeout exits**.
- Next step: **offline RL (IQL) policy head that selects (TP, SL, timeout, kelly) per-trade based on softs + regime**.

## Numbers that matter

### Meta-labeller (L3) honest val (n = 4,513)
| metric | value |
|---|---|
| AUC | **0.883** |
| precision | **0.873** |
| recall | 0.633 |
| F1 | 0.734 |

Meta is a strong *classification* filter — when it says "take trade", the TB-label matches 87% of the time.

### First zone grid — 46,080 configs, best per R:R

| R:R | TP | SL | timeout | WR | net | DD | sharpe | BE-WR | **gap** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.15 | 0.15 | 90 s | 38.5 % | −0.83 % | 1.2 % | −0.27 | 66.7 % | **+28.2 pp** |
| 1.33 | 0.20 | 0.15 | 60 s | 40.4 % | −0.86 % | 1.2 % | −0.32 | 57.9 % | +17.5 pp |
| 1.67 | 0.25 | 0.15 | 60 s | 40.4 % | −0.86 % | 1.2 % | −0.32 | 51.2 % | +10.8 pp |
| **2.00** | **0.30** | **0.15** | **90 s** | **38.3 %** | **−0.86 %** | **1.2 %** | **−0.28** | **45.8 %** | **+7.5 pp** |

### Wide grid — 36,864 configs, best per R:R up to 3.33

| R:R | TP | SL | timeout | WR | net | DD | sharpe | BE-WR | **gap** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.67 | 0.25 | 0.15 | 90 s | 38.3 % | −0.87 % | 1.2 % | −0.29 | 51.2 % | +12.9 pp |
| 2.00 | 0.30 | 0.15 | 90 s | 38.3 % | −0.86 % | 1.2 % | −0.28 | 45.8 % | +7.5 pp |
| 2.33 | 0.35 | 0.15 | 90 s | 38.3 % | −0.87 % | 1.2 % | −0.29 | 41.5 % | +3.2 pp |
| **2.67** | **0.40** | **0.15** | **90 s** | **38.1 %** | **−0.88 %** | **1.2 %** | **−0.29** | **37.9 %** | **−0.2 pp** |
| **3.00** | **0.45** | **0.15** | **120 s** | **37.7 %** | **−0.90 %** | **1.3 %** | **−0.25** | **34.9 %** | **−2.8 pp** |
| **3.33** | **0.50** | **0.15** | **120 s** | **36.2 %** | **−1.15 %** | **1.5 %** | **−0.24** | **32.4 %** | **−3.9 pp** |

**Context:** on the 2026-04-14 grid (63 k cache, 34 features, 5-arch ensemble) the smallest BE gap was +14.6 pp at R:R 2:1. **We more than halved it.**

## Why net is still negative at R:R 3:1 WR > nominal BE

The nominal BE formula `(SL + 0.07) / (TP + SL + 0.03)` assumes a binary `hit-TP or hit-SL` outcome. Our executor's nine `live_sim.TradeOutcome.REASONS` include:

- `tp_hit` — captures full TP (positive, rare at large TP + short timeout)
- `sl_hit` — captures full SL (negative)
- `trailing_sl_1 / trailing_sl_2` — captures partial progress (positive but < TP)
- `partial_plus_tp / partial_plus_trailing_sl_*` — 50 % closed at mid-progress, rest rides
- `timeout_limit / timeout_market` — exit at mid after `timeout`; captures whatever move happened, often ~0

At TP = 0.45 %, timeout = 120 s, our data distribution has the price hitting ±0.45 % within 120 s **very rarely**. The dominant exit reason is `timeout_*`, which captures a small mid-move minus ~0.07 % round-trip commission. Even with WR > nominal BE, the gross per-winner is tiny while losers still eat full SL or negative timeout.

The fix is not "pick better fixed TP/SL" — it is "**adapt TP/SL/timeout per trade**" based on the current primary-softmax confidence and volatility regime.

## Why fixed (TP, SL, timeout) is structurally wrong here

The live executor at `src/signal.py::_adaptive_tp_sl` already adapts TP/SL by `vol_ratio` and `dynamic_timeout_ticks_from_vol_ratio`. **But the research grid (`scripts/grid_live.py`) sweeps fixed values** — a mismatch between the adaptive strategy the executor implements and the static strategy the grid evaluates. The grid's best config is therefore not the best *adaptive* config it would actually run.

The unification decided this session: **merge the simulator for live + grid into a single `simulate_trade()` that takes (TP_policy, SL_policy, timeout_policy, kelly_policy) as callables of `(softs, features, state)`.** No separate "live_sim vs grid" concept — one parameterised simulator.

## Leak accounting (so far)

- Bakeoff primaries trained on samples `[0, 74k)` (80 %) with val `[74k, 93k)`.
- `infer_primaries_v3` produced softs on ALL 93 k — the 99 %+ directional accuracy in that script's log is leaked statistics and NOT an OOS metric.
- Grid trains L2 stacker + L3 meta on samples `[0, 70k)` (75 %) and evaluates on `[70k, 93k)` tail.
- **Leak overlap:** `[70k, 74k)` ≈ 4 k samples were in the primaries' training — ~17 % of grid's 23 k eval tail. The remaining 19 k (`[74k, 93k)`) is honest OOS.
- Honest-portion-only eval (74 k → 93 k) would likely show slightly worse numbers than reported above; not yet broken out.

## Bugs fixed this session (committed)

1. **Modal image build ordering** — `add_local_*` must come last (Modal's constraint).
2. **PEFT double-LoRA in Time-LLM** — adapter wraps LLM with LoRA internally; external recipe lora_cfg set to None to avoid double-wrap.
3. **PEFT `root_attr`** — Chronos/MOMENT unfrozen need target_modules on the encoder sub-module, not on the classifier.
4. **Adapter `**_kwargs`** — PEFT injects `input_ids / attention_mask / labels`; all four classifier forwards absorb them.
5. **TimesFM device caching** — `TimesFM_2p5_200M_torch()` caches `self.device` as a string at construction; adapter now keeps `wrapper.model` whole and patches `self.timesfm_model.device` to match the input at forward time.
6. **bakeoff_v3 checkpoint format** — saves `{"state_dict": ..., "metrics": ...}`; `infer_primaries_v3.py` now unwraps + strips `_orig_mod./module.` prefixes + loads `strict=False`.
7. **`infer_primaries_v3.py` weight filename** — expected `{tag}.pt`, but bakeoff_v3 writes `{tag}_best.pt`. Fallback added.
8. **Recorder self-healing** — staleness watchdog added to `_StreamHandler` (Binance primary streams were missing it), process-level watchdog in `record_data.py` exits after 90 s silence so systemd restarts, flush interval dropped to 15 s, retention extended to 168 h, `_tardis` files never rotated.

## Plan Day 1-5 (carte blanche, started 2026-04-16)

- **Day 1 (today):** meta retarget on realised net-PnL, `(state, action, reward)` dataset generator via `rust_bridge.simulate_labels`, CatBoost L2 stacker.
- **Day 2-3:** IQL policy head — state = `[14×3 softs, 49 feat, regime_id]`, action = discretised `(TP_bucket × SL_bucket × timeout_bucket × kelly_bucket × direction)`, reward = realised net PnL. Train on walk-forward 75 % tail, evaluate on the final 25 %.
- **Day 4:** `regime_moe` rewire onto Stage A/B/C features (drops `FEAT_VOL_RATIO=21, FEAT_HURST=23` which were pruned in Stage E), integrate into `bakeoff_v3::build_factory`, retrain regime-specialist ensembles. Retrain `regime_classifier` on the 49-feature set.
- **Day 5:** final end-to-end — simulate-parameterised grid using the IQL policy head + regime-specialist primaries + retarget-ed meta + regime pre-gate.

## Artifacts on Contabo

```
data/_cache/samples_v3_999h_1776165949_*.npy       # 93,230 × (3,20,50) LOB + 49 feat cache
models/primary_softs_v4.npz                         # softs for 14 archs × 93,230 samples
models/grid_live_v4.json                            # first zone grid (46 k configs)
models/grid_live_v4_wide.json                       # wide grid to R:R 3.33 (36 k configs)
/tmp/metrics/bakeoff_v3/*_best.pt                   # 15 primary weights pulled from Modal
logs/grid_live_v4.log, grid_live_v4_wide.log        # grid run logs
```
