# Maker-fill / adverse-selection simulation — operator guide (HUSDC)

**What it is.** Realistic *maker* (resting limit) entry fills with adverse
selection in the forward sim. Previously nothing modelled this: `live_sim`
assumed the entry filled, `grid_sim`'s inner sweep used an IID `fill_prob`
(adverse-blind), and the rev28 sub-60s surface entered at *mid*. Now a resting
limit fills **only when realized taker flow reaches our level** (touch / queue)
and **MISSES** when price runs away (the favorable runaways) — so adverse
selection emerges from the realized path, not a parameter.

Opt-in everywhere: with no new flags, behavior is **byte-identical** to before
(the parallel alpha work is unaffected).

## Code (branch `claude/husdc-rev1`)
- `rust_ingest/src/live_sim.rs`
  - `simulate_maker_entry(dir, level_px, q0, book_path, flow_path, window) -> Option<(fill_tick, fill_px)>`
    — touch (q0=0) / queue (q0>0) / MISS.
  - `simulate_trade_maker(...)` — on fill, reuses `simulate_trade_book` for the
    exit *from the fill tick*. `FlowL1{buy_vol, sell_vol}`.
  - 4 opt-in `LiveSimConfig` fields (default OFF): `maker_entry_enabled`,
    `maker_offset_frac`, `entry_window_ticks`, `queue_mult`.
  - Unit tests in `#[cfg(test)]` → `cargo test --lib` (9/9).
- `rust_ingest/src/bin/build_samples.rs`
  - Emits `flow_paths.npy` (N,H,2 = per-fwd-tick [buy_vol, sell_vol]) and
    `entry_q.npy` (N,2 = top-1 [bid_qty, ask_qty] = queue-ahead).
  - Reads **both** nested (FixedSizeList) **and flat raw cryptolake L2**
    (auto-detect; `bid_k_price/size`, ns→ms; flat trades `amount`+`side`).
- `rust_ingest/src/bin/grid_sim.rs`
  - Maker mode via `--flow-paths` + `--entry-q` (+ tuning flags below).

## Pipeline: raw L2 → maker pnl
1. Build the cache straight from `raw/` (per day; flat L2 read natively — no shim):
   ```
   build_samples --depth <book.parquet> --trades <trades.parquet> \
     --out-dir OUT --window W --horizon H --step S --max-samples N
   # -> OUT/{entry_long,entry_short,mid_paths,book_paths,flow_paths,entry_q,sample_ts,...}.npy
   ```
   (raw layout: `gs://market-data-0998ac51/raw/{book,trades}/exchange=BINANCE_FUTURES/symbol=<SYM>/dt=<DAY>/*.parquet`)
2. Run grid_sim in maker mode:
   ```
   grid_sim --entry-long el.npy --entry-short es.npy --mid-paths mid.npy \
     --book-paths book.npy --entry-book eb.npy \
     --flow-paths flow.npy --entry-q eq.npy --configs cfg.json --out-prefix P \
     --queue-mult 0 --entry-window-ticks 60 --maker-offset-frac 0 \
     --commission-win-pct 0 --commission-loss-pct 0
   # -> P_pnl_long.npy / P_pnl_short.npy  (NaN where the maker order MISSED)
   #    P_filled_long.npy / P_filled_short.npy (u8 fill mask); prints fill-rate.
   # Omit --flow-paths  => legacy assumed-entry behavior (byte-identical).
   ```
3. EV: pick pnl by your signal's predicted side over **filled** samples; mean = EV/trade.

## Tuning knobs (grid_sim)
- `--queue-mult` : `0` = touch (fill on first taker hit at the level); `>0` =
  require `queue_mult × entry_q` cumulative adverse volume to clear (more
  realistic, more adverse, lower fill-rate).
- `--entry-window-ticks` : max forward ticks to wait before MISS.
- `--maker-offset-frac` : rest below bid / above ask (deeper = better fill
  price, lower fill-rate). `0` = join the near side.

## Validation / provenance (all green)
- `cargo test --lib` 9/9 (maker physics + parity).
- `scripts/husdc_e2e_test.py` — synthetic 16/16 (build_samples flow/entry_q +
  grid_sim fill/miss/adverse + legacy parity).
- `scripts/husdc_flat_native_test.py` — native flat-read **bit-identical** to the
  nested path on real SOL L2 (all 7 arrays, atol=0).
- Python research prototype of the same physics: `scripts/husdc_makersim.py`.
- Ledger: **HUSDC rev6** (integration), **rev7** (native flat read); thesis +
  findings in **rev1–5** (USDC↔USDT transfer + 0%-maker) and the maker1 exp
  (`2026-05-28T1547Z_husdc_maker1_usdt_adverse`: adverse-selection haircut
  ~0.2bp at 45–60s; touch fill ~0.99, queue ~0.56).

## Status & next
Tooling is **validated**, but has **not** been run as a deploy-gate with a real
signal. Next: `build_samples(raw L2) → grid_sim maker` **with** the rev28 vol/dir
signal (or a momentum proxy) → EV per trade, on USDT then USDC@0% maker.
This is a deploy/EV question, separate from the exploratory transfer surface.
