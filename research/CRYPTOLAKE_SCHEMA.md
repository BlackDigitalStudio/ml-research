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

## raw event streams (the HA5 conditional-asymmetry inputs)

Verified 2026-05-17 on LINK (full fidelity, all symbols, ns timestamps):

- **`raw/liquidations`** — `side, quantity, price, id, status, timestamp`.
  ~246 rows/day (sparse — these ARE the rare events). `side` present:
  `buy` = short liquidation (forced buy → up pressure), `sell` = long
  liquidation (down pressure). **Note: H9's premise ("frequency-only, no
  side") is partly outdated — raw side IS available**; open question is
  whether `features_v1` used it.
- **`raw/open_interest`** — `open_interest, timestamp`. ~15k/day (~4 s).
  OI Δ = positioning regime.
- **`raw/funding`** — `mark_price, rate, next_funding_time, index_price,
  timestamp`. 86 400/day (1 s). `rate` = funding; `mark−index` = basis;
  funding flips = regime events.
- **`raw/trades`** — see above; ~242k/day on LINK.

These are the candidate symmetry-breaking conditioners for HA5
(does a liquidation cascade / OI shock / funding flip make the ≥cost
forward excursion directionally skewed, vs the ~symmetric baseline).

## features_v1 (the model input X)

**DECODED 2026-05-17 (earlier "unrecoverable/opaque" claim RETRACTED —
it was unverified laziness).** `features_v1` is the raw-56 layout of
THIS repo's feature engine (NO `DROP_RAW_INDICES` applied — `hurst`,
`spoof`, `large_order` are present) **+ 3 Cryptolake extension cols =
59**. Cols 0-55 are exactly, in order (from `src/features.py
FEATURE_KEYS` + `DROP_RAW_INDICES=[5,17,18,19,21,22,23]` +
`rust_ingest/src/features.rs` index comments), empirically verified by
value signatures (spread≡1 tick 0.001, funding≡1e-4, cvd ±3e5,
hurst∈[0.35,0.66], vpin∈[0,1], kyle_lambda~1e-8, ofi_* large signed):

```
0 ofi 1 imbalance_ratio 2 imbalance_velocity 3 spread 4 depth_ratio_l5
5 large_order 6 trade_flow_imbalance 7 trade_intensity 8 large_trade
9 cvd 10 volatility_1s 11 vwap_deviation 12 momentum_5s 13 funding_rate
14 eth_momentum_1s 15 eth_ofi 16 eth_leading_signal 17 open_interest_delta
18 long_short_ratio 19 liquidation_proximity 20 spoof_score
21 volatility_ratio 22 trade_intensity_ratio 23 hurst 24 sweep_intensity
25 cancel_rate_diff 26 ofi_1s 27 ofi_5s 28 ofi_30s 29 ofi_divergence
30 cross_exch_mom_500ms 31 queue_pressure 32 top3_asymmetry
33 effective_spread_ratio 34 momentum_30s 35 momentum_60s 36 momentum_120s
37 realized_vol_60s 38 realized_vol_120s 39 bipower_var_120s 40 ofi_60s
41 ofi_120s 42 trade_flow_imbalance_60s 43 funding_time_to_next_min
44 funding_basis_bps 45 microprice_deviation 46 ofi_top5_weighted
47 kyle_lambda_60s 48 vpin_60s 49 cancel_to_trade_ratio_30s
50 bybit_lead_lag_corr_30s 51 okx_net_flow_30s 52 bitget_net_flow_30s
53 gateio_net_flow_30s 54 eth_momentum_60s 55 eth_btc_corr_30s
56-58 CRYPTOLAKE_EXT (liq/OI; 56/57 small signed ±0.015 ≈ ret/delta,
       58 count-like [-26,107] ≈ liquidation magnitude) — light-ID TODO
```

**DEAD COLUMNS in this build (empirical, LINK 2026-05-06) — material:**
all cross-asset/ETH are 100% zero (14,15,16,30,50,51,52,53,54,55) and
several are constant (5 large_order≡1, 18 long_short_ratio≡0, 19
liquidation_proximity≡0.015, 20 spoof≡1, 24 sweep≡0). → models in
HA1/HA5 effectively saw **~46 live features; the ENTIRE cross-asset/ETH
dimension was literally zeros.** This narrows every prior negative and
makes **H3 (BTC-lead) concretely actionable**: the cross-asset slots
exist and are empty; BTC raw is in the same bucket to fill them.

- `features.npy` `(3602, 59) float32` — per-decision-point matrix,
  **decoded above** (raw-56 + 3 ext; ~46 live, cross-asset/ETH all
  zero). Row 0 ≈ zeros (warmup).
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
