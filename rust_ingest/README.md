# scalper_ingest — Rust training pipeline

Byte-parity Rust ports of `Trainer._calc_features_batch` and
`live_sim.simulate_trade`, plus the streaming cache builder.

Introduced `c9b0123` (2026-04-13). Extended through 2026-04-15 with the
direct-Rust flat-RAM path (commits `5699aba`, `ec72e8b`, `96b4f27`,
`b19c2af`).

## Build

```bash
# One-time if rustup is missing:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

cd rust_ingest
cargo build --release   # ~18 min first time (LTO + arrow dep tree), ~5 min incremental
```

Artifacts appear under `target/release/`:

| Binary | Role | Throughput |
|---|---|---:|
| `feature_builder` | 49-col feature computation over parquet → `.npy`. Rust emits 56 raw columns; Python applies `features.KEPT_RAW_INDICES` to get 49 after the Stage-E prune. | ~20× Python on the same cols |
| `sim_labels` | Direction-aware forward-sim triple-barrier labeller (rayon-parallel). Emits `y`, `target_pnl`, `pnl_long`, `pnl_short`, `mid_paths`, `entry_long`, `entry_short`. | ~91k samples/s on 28 cores |
| `build_samples` | Row-group-streaming cache builder. Uses `ParquetRecordBatchReader` + sliding `VecDeque<SlimBatch>` to keep RAM flat. | ~100 MB peak on 65 M rows |
| `depth_parser` | Pre-existing Tardis CSV → binary snapshot parser. | 517k rows/s |

## Run modes

### Default path — `SCALPER_USE_RUST=1` (set by default)

Python loads compacted parquets into memory (pyarrow), Rust computes
features + labels. This is the mode used by `Trainer._calc_features_batch`
and `rust_bridge.simulate_labels`. `SCALPER_USE_RUST=0` opts out for
parity debugging only; missing Rust binaries raise `RustBinariesMissing`
— no silent Python fallback.

### Direct-Rust path — `SCALPER_USE_RUST_DIRECT=1`

Streams raw parquets directly through the Rust `build_samples` binary,
then feeds slices into `compute_features_chunked`. This is the
flat-RAM path for bake-off-scale caches (1 M samples, 65 M depth rows,
24 GB peak RSS on Contabo). Prerequisite: run
`scripts/merge_streams.py` to produce `data/_merged/*.parquet` first
(one-time ingest, 7 min, 1.9 GB peak RSS).

`scripts/build_cache.py` with both env vars set is the canonical way
to build the 1 M cache today.

## Verify parity before any Rust change

```bash
cd /home/scalper/scalper-bot
venv/bin/python scripts/parity_rust_features.py \
    /tmp/tardis_test/depth/20240601_tardis.parquet \
    --trades ... --funding ... --derivs ... --eth ... \
    --bybit ... --okx ... --bitget ... --gateio ... --n 5000

venv/bin/python scripts/parity_rust_live_sim.py --n 500
```

Both must pass with `max|Δ| = 0.000e+00` on the in-scope columns. Any
regression here is a blocker.

## Integration status (2026-04-16)

All three Rust paths are wired into Python:

1. `rust_bridge.compute_features` — called by `Trainer._calc_features_batch` when `SCALPER_USE_RUST=1`. Default.
2. `rust_bridge.simulate_labels` — called by `Trainer.build_samples` (see `trainer.py:794, 1160`); emits `pnl_long` / `pnl_short` / `target_pnl` directly.
3. `_build_samples_rust_direct` — opt-in via `SCALPER_USE_RUST_DIRECT=1`; streams the full pipeline through the Rust binaries.

## Known details

- **Feature [29] sign-boundary tolerance.** When upstream [26] or [28] is near zero, f32 precision differs between numpy's pairwise-sum and Rust's running-window sum, flipping the sign of the gate in the product that drives feature 29. Parity harness excludes samples where `|[26]| < 0.1` or `|[28]| < 0.1`. Model impact is nil (feature is sign-based, tiny signals are noise).
- **Recorder schema (2026-04-14).** `src/recorder.py` now writes flat `FixedSizeList<f64, 20>` depth — matches Tardis and the Rust readers. Legacy 309 files were migrated in place via `scripts/migrate_legacy_depth.py`. Any non-flat parquet now raises an explicit error.
- **rust_bridge round-trips through /tmp parquet** on each `compute_features()` call. Acceptable for one-shot training cache builds (fires once); NOT acceptable inside grid-search hot loops — grid-search caches features already, but confirm before wiring Rust into new grid callers.
- **Warmup.** Rust `feature_builder` emits 0.0 for samples with `idx < window_size` (same as Python). EMA cold-start on features [31] and [33] is identical on both sides.

See `/root/.claude/projects/-root/memory/rust_pipeline.md` for the
full running notes (some claims there are older than this README; this
file is authoritative for build + integration status).
