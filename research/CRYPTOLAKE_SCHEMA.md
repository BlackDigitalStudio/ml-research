# Cryptolake GCS data contract (reverse-engineered 2026-05-16)

`scripts/build_cryptolake_cache.py` was lost with Contabo. This is the
schema it must be reconstructed against, decoded directly from
`gs://blackdigital-scalper-data` (project `project-26a24ad0-1059-4f73-93b`,
`EUROPE-WEST1`). Hard-won; do not re-discover.

## Bucket layout

```
gs://blackdigital-scalper-data/
  features_v1/symbol=<SYM>/dt=<YYYY-MM-DD>/features.npy   (3602, 59) float32
                                          /indices.npy    (3602,)    int64
  raw/book/exchange=BINANCE_FUTURES/symbol=<SYM>/dt=<YYYY-MM-DD>/1.snappy.parquet
  raw/trades/        ... same partitioning ...
  raw/funding/ raw/liquidations/ raw/open_interest/   (same partitioning)
```

- **8 symbols** (Tardis naming, `-USDT-PERP`): BNB, BTC, DOGE, ETH, LINK,
  LTC, SOL, XRP.
- **546 date partitions / symbol: 2024-11-09 → 2026-05-08** (~18 months).
- `scalper-bot-research-data` (volaware ckpts) → 403, no access. Volaware
  is refuted; non-blocking.

## raw/book parquet (the price/label source)

20-level LOB. ~793,719 rows/BTC-day (~108 ms/update). Columns:

- `timestamp` int64 **nanoseconds**, `receipt_timestamp` int64 ns,
  `sequence_number` int64
- `bid_0_price..bid_19_price` + `bid_*_size`, `ask_0_price..ask_19_price`
  + `ask_*_size` (all float64)
- schema metadata: `date, symbol, exchange, event_type, contains_gaps`.
  **`contains_gaps: 'Yes'`** — the builder MUST handle gaps (do not assume
  a uniform tick grid; use timestamps).

`raw/trades`: `side, amount, price, id, timestamp, receipt_timestamp`
(~598k/BTC-day) — aggressor side for order-flow / maker-taker features.

## features_v1 (the model input X)

- `features.npy` `(3602, 59) float32` — the per-decision-point feature
  matrix. **59 cols** (≠ repo `src/features.py` 49/55 — a different/newer
  set; treat as opaque X for modelling, do not assume FEATURE_KEYS order).
  Row 0 ≈ zeros (warmup).
- `indices.npy` `(3602,) int64`, **monotonic**, step ≈ 220
  (`[0,220,440,...,792220]`). **`features.npy[k]` ↔ `book_parquet.row(indices[k])`**.
  → decision points = every ~220 book updates ≈ ~24 s. ~3602 decisions/day.

## H5 reconstruction path (MAKER-first labels by construction)

For each symbol/day, for each decision point `k` (row `i = indices[k]`):

1. **Entry** at book row `i`:
   - LONG  maker-first: `entry_long  = bid_0_price[i]` (post at bid)
   - SHORT maker-first: `entry_short = ask_0_price[i]` (post at ask)
   - TAKER variant (for A/B vs the old artifacts): long=ask, short=bid.
2. **Forward paths** from `i` for `SIM_HORIZON` book rows:
   - `mid_path  = (bid_0_price + ask_0_price)/2`
   - `book_path = [bid_0_price, ask_0_price]` (book-aware 0-gap sim)
   - Respect gaps via `timestamp`: horizon is a wall-clock window
     (e.g. 60–180 s), not a fixed row count, because `contains_gaps=Yes`.
3. `src/rust_bridge.simulate_labels(entry_long, entry_short, mid_paths,
   …, commission_win_pct=0.04, commission_loss_pct=0.07, book_paths=…)`
   → `pl/ps/y`. **MAKER_FIRST fee regime = 0.04/0.07 round-trip; TAKER
   no-VIP = 0.07/0.10.** This is the H5 gate.
4. `X = features.npy` (the 3602×59 matrix), aligned 1:1 with the labels by
   the same `k`. Per-symbol model + H2 PT/TS grid via
   `simulate_labels_grid`.

## Compute reality

- rust `sim_labels` / `grid_sim` **cannot build in the planning container**
  (crates.io 403, no vendor, 187 deps). Python data-prep validates here
  at $0; the Rust step + full 8-symbol rebuild + H2 run on the GCP VM
  (normal egress → `cargo build` works).
- Per-day book parquet ≈ 40–75 MB. Full 8×546 ≈ ~250 GB raw I/O — use a
  **date-range subset** first (the old experiments did); 546 days × 8 is a
  later, deliberate spend, not the default.
