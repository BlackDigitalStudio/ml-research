# scalper-bot

Automated BTCUSDT perpetual scalping on Binance Futures (x20 leverage, $50 deposit).
5-arch NN ensemble (transformer + tcn + chronos-bolt tiny/mini/small) → L2 stacker
(XGBoost) → L3 meta-label (López de Prado) → Kelly sizing → live_sim executor.

**This file is a pointer, not the spec.** The authoritative sources of truth live
in the Claude memory at `~/.claude/projects/-root/memory/`.

## Where to start

- **`READ_FIRST_NEXT_STEPS.md`** — current honest state + prioritised next steps + do/don't checklist.
- **`methodology_bugs_2026_04_14.md`** — the two bugs that inflated every pre-2026-04-14 OOS number. Mandatory reading before writing a new evaluator.
- **`project_trading_bot.md`** — canonical project summary. Architecture, 34 features, 7 entry filters, adaptive TP/SL, partial TP, stepped trailing SL, fast-fill tightening, commissions table, data streams, training schedule, roadmap phases.
- **`ROADMAP_2026_04_15.md`** — research tracks A (data), B (extended grid), C (more models), D (RL).
- **`contabo_primary_host.md`** — `root@84.247.154.229`, 16 vCPU / 62 GB. Vultr Tokyo retired 2026-04-14.

## Key scripts

| Script | Purpose |
|---|---|
| `scripts/build_cache.py` | Build v3 sample cache (+ mid_paths, entry_long/short sidecars). |
| `scripts/migrate_legacy_depth.py` | One-shot conversion of old nested `list<list<f64>>` depth parquets to flat `FixedSizeList<f64, 20>`. |
| `scripts/infer_primaries_v3.py` | Run saved `.pt` checkpoints against a v3 cache to produce `primary_softs_v3.npz`. |
| `scripts/grid_live.py` | **Authoritative strategy grid.** Walk-forward stacker + meta; direction-aware PnL via Rust `simulate_labels`; sweeps thousands of (TP, SL, timeout, partial, trailing, kelly, meta_thr, min_prob, spread_bps, fill_prob) configs. Replaces `grid_test_ensemble.py`. |
| `scripts/honest_eval.py` | Minimal reference for direction-aware PnL (debug helper). |
| `scripts/tardis_csv_to_parquet.py` | Tardis free-tier CSV → recorder flat parquet schema. |
| `scripts/download_tardis_free.py` | First-of-month Tardis downloader (2020-01 onward). |
| `scripts/bakeoff_v2.py` / `recover_stacker.py` | Full bake-off + recover-from-crash. Pod-side. |
| `rust_ingest/` | Rust `feature_builder` (34-col vectorised) + `sim_labels` (forward trade sim). Path is default; opt out with `SCALPER_USE_RUST=0`. |

## Do / don't (see `READ_FIRST_NEXT_STEPS.md` for the full checklist)

- **Do** use `grid_live.py` as the authoritative grid. Any new metric must consume `pnl_long` / `pnl_short` from `rust_bridge.simulate_labels`, not `target_pnl`.
- **Do** train stacker and meta on the train split only, then verify `wrong_dir_nfnf > 0` on tail before trusting any number.
- **Don't** use `target_pnl` as realised PnL — it is a TB-winner label, not a trading outcome.
- **Don't** retrain stacker on the full val set — that caused the 2026-04-14 leak.
- **Don't** ship anything live until the grid produces at least one config with `n ≥ 100, net > 0, DD < 15%` on an independent slice and then ≥ 14 days of paper-trading at net > 0.
