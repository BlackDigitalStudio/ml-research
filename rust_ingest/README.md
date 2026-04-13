# scalper_ingest — Rust training pipeline

Byte-parity Rust port of `Trainer._calc_features_batch` and
`live_sim.simulate_trade`. Commit introduced: `c9b0123` (2026-04-13).

## Build

```bash
# One-time if rustup is missing:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

cargo build --release   # ~18 min first time (LTO + arrow dep tree), ~5 min incremental
```

Artifacts appear under `target/release/`:
- `feature_builder` — computes 34 features from parquet streams → `.npy`
- `sim_labels` — batch forward-sim (rayon-parallel) → `y.npy` + `target_pnl.npy`
- `depth_parser` — pre-existing Tardis CSV → binary snapshot parser

## Enable in Python

```bash
export SCALPER_USE_RUST=1
# `Trainer._calc_features_batch` now dispatches to src/rust_bridge.py
```

Without the env var, the Python path runs unchanged.

## Verify parity before any change

```bash
cd /home/scalper/scalper-bot
venv/bin/python scripts/parity_rust_features.py \
    /tmp/tardis_test/depth/20240601_tardis.parquet \
    --trades .../binance_trades_synth.parquet \
    --funding ... --derivs ... --eth ... \
    --bybit ... --okx ... --bitget ... --gateio ... --n 5000

venv/bin/python scripts/parity_rust_live_sim.py --n 500
```

Both MUST print `PASS`. Any Rust change that breaks this is a regression.

## Known gaps / things to know

- `trainer.build_samples` per-sample LONG/SHORT forward-sim loop is NOT yet
  wired to `rust_bridge.simulate_labels`. The function exists and is parity-
  tested; wiring it in is the next integration step (~50× on label building).
- Feature [29] has a sign-boundary tolerance in the parity harness — see
  `rust_pipeline.md` memory for why.
- `src/recorder.py` still writes the OLD `list<list<f64>>` depth schema; only
  Tardis ingested parquets use the flat `FixedSizeList<f64, 20>` schema that
  Rust readers understand.
- `rust_bridge.compute_features` round-trips data through /tmp parquet on
  every call — acceptable for one-shot training, NOT for hot loops.

See `/root/.claude/projects/-root/memory/rust_pipeline.md` for the full
handoff doc.
