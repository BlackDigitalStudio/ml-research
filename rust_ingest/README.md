# scalper_ingest — Rust engines (research pipeline + LIVE decision engine)

Two roles in one crate:

1. **Research pipeline binaries** — feature/label/sim engines the frozen
   measurement protocol shells out to (see `research/runtime/README.md`).
2. **`axb_engine` — the LIVE sub-ms decision engine** (deployed 2026-07-09,
   RESEARCH_LOG s22): MirrorBook full-book reconstruction from `@depth@100ms`
   diffs, day-anchored incremental features (`features_incr.rs`), bit-exact
   XGBoost predictor (`gbt.rs`), causal tau, orders via Unix socket to the
   Python exec sidecar (`live/axb_exec.py`).

## Binaries (src/bin)

| Binary | Role |
|---|---|
| `feature_builder` | X64 feature computation over raw parquet at given tick indices. FULL inputs (`--funding --liquidations --open-interest --eth`) or the funding/OI/liq/ETH cols are silently zero — the historical qm1-rebuild bug. |
| `build_samples` | Streaming sample/path builder (entry prices, mid/book/flow forward paths). **The production tb3s variant lives in `scripts/build_samples_husdc.rs`** and is compiled against the lib of branch **`claude/husdc-rev1`** — see below. |
| `grid_sim` / `scripts/grid_sim_exitdbg.rs` | Fused maker-cycle simulator; the exitdbg variant adds time-based windows (`--ts-paths`, `to_ms`, `--entry-window-ms`, `--chase-ms`) and pegged-exit chase — the honest h150 execution model. |
| `sim_labels` | Legacy triple-barrier labeller (pre-tb3s era). |
| `axb_engine` | Live decision engine (deployed unit `axb-engine-doge`). |
| `fb_incr_harness`, `score_harness` | **Golden parity harnesses** — features 0/2.03M cells, predictions 0/228k, ensemble score 0/28546 vs the Python pipeline (day 20260707). Any change to features/gbt/engine must re-pass BOTH before deploy. |
| `depth_parser`, `hd1_seq_build` | Historical (Tardis CSV parser; HD1 sequence cache). |

## Building for research runs

Use `research/runtime/bins.sh <repo> [out_dir]` — it builds `feature_builder`
from this crate at HEAD **and** the husdc pair (`build_samples` + `grid_sim_exitdbg`)
from the **`claude/husdc-rev1` branch lib** with the frozen bin sources from
`scripts/` overlaid. That branch is LOAD-BEARING (has `simulate_maker_entry`,
`FlowL1` in `live_sim.rs` that master lacks) — never delete it. Binaries go to a
persistent dir, never `/tmp` (wiped on VM restart — see
`research/runtime/KNOWN_PITFALLS.md`).

Reproducibility: a fresh 2026-07-10 rebuild (rustc 1.96) reproduced the July
production daily artifact BYTE-EXACT (all keys) — ledger
`tb3s-20260710_h150anch_year_xsym_PREREG_AMEND1`.

## Bit-exactness keys (measured, not assumed — s22.3)

- Day-anchored prefix state reproduces batch float summation order.
- XGBoost f32 margin accumulates base-margin FIRST; empirically-solved base bits.
- Sigmoid must be glibc `expf` (numpy SIMD exp does NOT match).
- Leaf values live in `split_conditions`.
- Results depend on xgboost `nthread` — one (symbol,seed) job = one nthread.

## Historical note

The April-2026 era of this crate (byte-parity ports of the old Python
`Trainer._calc_features_batch` / `live_sim.simulate_trade`, `SCALPER_USE_RUST`
bridge, Contabo paths) is superseded; that Python stack is in `archive/src/`.
This README describes the current (s22) reality.
