# BTCUSDT Scalping Strategy v3

## Overview

Automated scalping algorithm for **BTCUSDT Perpetual** on Binance Futures.

| Parameter | Value |
|---|---|
| Instrument | BTCUSDT Perpetual |
| Leverage | x20 |
| Starting deposit | $50 USDT |
| Server | Vultr Tokyo (108.61.187.191), ~3ms to Binance |
| Stack | Python 3.12, asyncio, PyTorch (CPU), XGBoost, LightGBM, aiohttp/websockets |

---

## 1. Model Architecture

### CNN Encoder
- Input: 3-channel LOB tensor `(3, 20, 50)` — 3 channels x 20 depth levels x 50 time snapshots (5 seconds @ 100ms)
- Architecture: Conv2d(3→32, 3x3) → BN → ReLU → Conv2d(32→64, 3x3) → BN → ReLU → Dropout(0.2) → AdaptiveMaxPool2d(1) → FC(64→64)
- Output: 64-dimensional embedding

### Channels
| Channel | Content |
|---|---|
| 0 | Bid volumes (20 levels x 50 snapshots) |
| 1 | Ask volumes (20 levels x 50 snapshots) |
| 2 | Trade flow (buy/sell volume per timestep) |

### Ensemble (5 models)
Instead of a single XGBoost, the model uses an ensemble of 5 diverse models:

1. **3x XGBoost** — different random seeds (42, 123, 456), each trained on a random 80% of training rows (bagging)
2. **1x LightGBM** — full training data, same hyperparameters
3. **1x Logistic Regression** — trained on top-5 most important hand-crafted features (selected by average feature importance across the 3 XGBoost models)

**Input to ensemble:** concatenated `[64-dim CNN embedding + 30 hand-crafted features]` = 94 features.

**Voting rules:**
- Entry requires **minimum 3/5 agreement** on direction (UP or DOWN)
- If spread of confidences across models > 20% → skip (high uncertainty)
- Return average confidence of agreeing models

### Labels
- **Horizon:** 600 snapshots = **60 seconds** lookahead
- **Threshold:** ±0.2% price move within the horizon
- UP: max upward move ≥ 0.2% and ≥ max downward move
- DOWN: max downward move ≥ 0.2% and > max upward move
- FLAT: neither threshold reached

### Training Schedule
- CNN encoder: retrain every 24 hours
- XGBoost ensemble: retrain every 4 hours
- Training data window: 72 hours rolling
- All training runs with `nice -n 19` to prevent CPU starvation of trading/recorder

---

## 2. Features (30 total)

### LOB Features (0-5) — Primary BTC Order Book
| # | Name | Description |
|---|---|---|
| 0 | `ofi` | Order Flow Imbalance — EMA of (Δbid_vol - Δask_vol) at best level |
| 1 | `imbalance_ratio` | (bid_vol_L5 - ask_vol_L5) / total_vol_L5 |
| 2 | `imbalance_velocity` | Change in imbalance_ratio over 5 ticks (500ms) |
| 3 | `spread` | Best ask - best bid |
| 4 | `depth_ratio_l5` | bid_vol_L5 / ask_vol_L5 |
| 5 | `large_order` | Binary: any order > 1 BTC in top 5 levels |

### Trade Flow Features (6-9) — Primary BTC Trades
| # | Name | Description |
|---|---|---|
| 6 | `trade_flow_imbalance` | (buys - sells) / total over 5-second window |
| 7 | `trade_intensity` | Number of trades in last 1 second |
| 8 | `large_trade` | Binary: any trade > 10 BTC in last 5 seconds |
| 9 | `cvd` | Cumulative Volume Delta over 30 seconds |

### Derived Features (10-13)
| # | Name | Description |
|---|---|---|
| 10 | `volatility_1s` | Std of returns over last 10 ticks (1 second) |
| 11 | `vwap_deviation` | (mid_price - VWAP_60s) / VWAP_60s |
| 12 | `momentum_5s` | Price change over 50 ticks (5 seconds) |
| 13 | `funding_rate` | Current funding rate from markPrice stream |

### ETH Leading Signal (14-16)
| # | Name | Description |
|---|---|---|
| 14 | `eth_momentum_1s` | ETH price change over 1 second (leads BTC by 100-500ms) |
| 15 | `eth_ofi` | ETH Order Flow Imbalance (EMA) |
| 16 | `eth_leading_signal` | BTC/ETH ratio deviation from 30-second rolling mean |

### Liquidation Clusters (17-19)
| # | Name | Description |
|---|---|---|
| 17 | `open_interest_delta` | OI change vs previous poll (~15s interval) |
| 18 | `long_short_ratio` | Top trader long/short ratio (Binance API) |
| 19 | `liquidation_proximity` | Estimated distance to liquidation cluster (±1.5% for x50-x100) |

### Spoofing Detection (20)
| # | Name | Description |
|---|---|---|
| 20 | `spoof_score` | Ratio of large orders that persisted > 2.5s without price reaction |

### Volatility Regime (21-22)
| # | Name | Description |
|---|---|---|
| 21 | `volatility_ratio` | Current vol / 30-second average vol |
| 22 | `trade_intensity_ratio` | Current intensity / 30-second average intensity |

### Market Regime (23)
| # | Name | Description |
|---|---|---|
| 23 | `hurst_exponent` | R/S analysis over 100 ticks. H > 0.55 = trending, H < 0.45 = mean-reverting |

### Sweep Detection (24)
| # | Name | Description |
|---|---|---|
| 24 | `sweep_intensity` | Total bid/ask levels swept in last 1 second |

### Cancellation Rate (25)
| # | Name | Description |
|---|---|---|
| 25 | `cancel_rate_diff` | ask_cancel_rate - bid_cancel_rate over 1-second window. Positive = sellers retreating = bullish |

Cancellation = volume drop at a price level without a corresponding trade. Computed by comparing consecutive depth snapshots.

### Multi-Timeframe OFI (26-29)
| # | Name | Description |
|---|---|---|
| 26 | `ofi_1s` | Sum of raw OFI over 1-second window |
| 27 | `ofi_5s` | Sum of raw OFI over 5-second window |
| 28 | `ofi_30s` | Sum of raw OFI over 30-second window |
| 29 | `ofi_divergence` | ofi_1s - ofi_30s when they have opposite signs (reversal signal) |

### Normalization
All features are z-score normalized using a 30-second rolling window (300 ticks at 100ms).

---

## 3. Data Sources

### Binance WebSocket
| Stream | URL | Rate |
|---|---|---|
| Depth (L20) | `btcusdt@depth@100ms` | 10/sec |
| Agg Trades | `btcusdt@aggTrade` | variable |
| Mark Price | `btcusdt@markPrice@1s` | 1/sec |
| ETH Depth | `ethusdt@depth@100ms` | 10/sec |
| ETH Trades | `ethusdt@aggTrade` | variable |
| User Data | via listenKey | events |

### Bybit WebSocket (Cross-Exchange Signal)
| Stream | URL | Rate |
|---|---|---|
| BTC Trades | `publicTrade.BTCUSDT` via `wss://stream.bybit.com/v5/public/linear` | variable |

Bybit trades lead Binance by 100-500ms. No account needed — public stream.

### Binance REST (Polled)
| Endpoint | Interval |
|---|---|
| `/fapi/v1/openInterest` | 15s |
| `/futures/data/topLongShortAccountRatio` | 15s |

### Data Storage
All data is saved to hourly Parquet files with Snappy compression, 72-hour retention.

| Directory | Content | Source | Used in Training |
|---|---|---|---|
| `data/depth/` | BTC L20 order book snapshots | WS depth@100ms | Features 0-5, 20-25, LOB tensor |
| `data/trades/` | BTC aggregated trades | WS aggTrade | Features 6-9, 11-12 |
| `data/eth_depth/` | ETH L20 order book snapshots | WS ethusdt@depth@100ms | Feature 15 (eth_ofi) |
| `data/eth_trades/` | ETH aggregated trades | WS ethusdt@aggTrade | Features 14, 16 (eth_momentum, leading_signal) |
| `data/bybit_trades/` | Bybit BTC trades | WS publicTrade.BTCUSDT | Cross-exchange signal (future) |
| `data/funding/` | Funding rate + mark price | WS markPrice@1s | Feature 13 |
| `data/derivatives/` | Open interest + L/S ratio | REST poll 15s | Features 17-19 |
| `data/bot_trades/` | Executed bot trades | Trading engine | Performance tracking |

**Training-runtime consistency:** All 30 features are computed from recorded data during training. No placeholders or zeroed features — the model sees identical distributions in training and live inference.

---

## 4. Entry Logic

### Signal Generation
1. CNN encoder produces 64-dim embedding from LOB tensor
2. FeatureEngine computes 30 hand-crafted features
3. Ensemble predicts: UP / DOWN / FLAT with confidence and vote count
4. Entry requires ALL of:
   - Prediction = UP or DOWN (not FLAT)
   - **≥ 3/5 ensemble agreement**
   - Ensemble uncertainty < 20%
   - Confidence ≥ dynamic threshold (base 0.58, self-tuning)
   - Spread ≤ $0.03 (3 ticks)
   - No volatility spike (vol < 3σ)
   - Spoof score < 0.5
   - Fill rate > 25% (last 20 attempts)
   - Imbalance confirms direction (> 0.15 for LONG, < -0.15 for SHORT)
   - Hurst regime compatible (mean-reverting regime needs +0.05 confidence)

### Entry Execution
- **Order type:** LIMIT Post-Only (GTX) at best bid (LONG) / best ask (SHORT)
- **Order timeout:** 2 seconds → cancel if unfilled
- **Backup SL:** STOP_MARKET on exchange immediately after entry order placed (MARK_PRICE trigger)

### Position Sizing
Dynamic based on ensemble confidence and market regime:

| Condition | Position Size |
|---|---|
| 5/5 votes, confidence > 0.65 | 95% of notional |
| 4/5 votes | 75% of notional |
| 3/5 votes | 50% of notional |
| Hurst 0.45-0.55 (uncertain regime) | ×0.5 additional multiplier |

Notional = balance × leverage (x20). Min notional: $100 (Binance BTC minimum).

---

## 5. Exit Logic

### Take-Profit
- **Adaptive TP:** base 0.2% of price, scaled by volatility ratio, clamped [0.10%, 0.60%]
- **Execution:** LIMIT Post-Only (GTX) → maker fee

### Stop-Loss
- **Adaptive SL:** base 0.1% of price, scaled by volatility ratio, clamped [0.05%, 0.30%]
- **Execution:** STOP_MARKET (taker fee) — always on exchange as backup
- **TP:SL ratio:** 2:1 base

### Stepped Trailing Stop-Loss
Tied to adaptive TP target, activates as price moves in favor:

| Price Progress | SL Moves To | Rationale |
|---|---|---|
| ≥ 50% of TP | max(entry + 0.08%, entry + 30% of TP distance) | Lock in profit above taker commission |
| ≥ 75% of TP | entry + 50% of TP distance | Secure larger portion |
| 100% of TP | Close via limit order | Full TP hit |

The +0.08% minimum on step 1 ensures the taker commission (~0.07% for stop-market) is covered, netting +0.01% minimum.
Trailing SL executes as STOP_MARKET (taker fee).

### Partial Take-Profit
At 50% of TP distance:
- Close **50% of position** via LIMIT (GTX) at current price
- Remaining 50% continues with trailing SL
- First half locks profit, second half rides the full move

### Timeout Exit
- **Dynamic timeout:** `base_timeout × (avg_volatility / current_volatility)`, clamped [15s, 120s]
- Base timeout: 60 seconds
- **Execution:** LIMIT (GTX) at mid price first, fallback to MARKET after 2 seconds
- **Counted as loss** in statistics
- If timeouts > 30% of trades in last hour → confidence threshold increases by 0.02

### Adverse Selection Detection
Track `fill_speed` — time from order submission to fill:
| Fill Speed | Action |
|---|---|
| < 50ms | Close immediately — someone aggressively going against you |
| 50-100ms | Tighten SL to 50% of normal distance |
| > 100ms | Normal execution |

This is a **runtime-only filter** (not a model feature) — it detects quality of individual entries.

---

## 6. Commission Structure

| Scenario | Fee | Calculation |
|---|---|---|
| Win (TP hit) | 0.04% round-trip | Entry maker (0.02%) + Exit maker (0.02%) |
| Loss (SL hit) | 0.07% round-trip | Entry maker (0.02%) + Exit taker (0.05%) |
| Timeout (limit) | 0.04% | Entry maker + Exit maker (GTX) |
| Timeout (market fallback) | 0.07% | Entry maker + Exit taker |

---

## 7. Risk Management

### Daily Limits
| Limit | Value |
|---|---|
| Max daily loss | 10% of deposit |
| Max consecutive losses | 10 → 30-minute pause |

### Circuit Breakers
| Condition | Action |
|---|---|
| Volatility spike (vol > 3σ) | Pause entries |
| Spread > $1.00 | Pause entries |
| WebSocket stale > 3 seconds | Emergency close all positions |
| Near funding settlement (±2 min) | Pause entries |
| Liquidity < 50 BTC in top 5 bid levels | Pause entries |
| Fill rate < 25% | Pause entries |
| Rate limit weight > 1100/1200 | Pause API calls |

### Self-Tuning Threshold
Confidence threshold adjusts based on recent performance:
- PF > 1.5 and WR > 55% → threshold -0.02 (more trades)
- PF < 1.0 or WR < 48% → threshold +0.02 (fewer trades)
- Timeouts > 30% → threshold +0.02
- Range: [0.50, 0.70]

### Position Recovery on Startup
On bot start: `GET /fapi/v2/positionRisk` — if open position exists, take over management (set direction, size, entry price, place SL/TP) instead of opening a duplicate.

### Rate Limit Monitoring
Track `X-MBX-USED-WEIGHT-1M` header from all Binance API responses:
- Warning at > 1000 (of 1200 limit)
- Pause non-critical API calls at > 1100

---

## 8. Infrastructure

### Server
- Vultr Tokyo VPS, 2 vCPU, 3.8 GB RAM
- systemd services: `scalper-recorder.service` (data), `scalper-bot.service` (trading)
- Auto-restart on failure

### Data Collection
- Phase 0: 72 hours continuous recording before first training
- Depth + Binance trades + Bybit trades
- Hourly parquet rotation, Snappy compression, 72-hour retention

### Training Pipeline
1. `build_samples()` — vectorized numpy, X_lob written to disk via mmap (never held in RAM)
2. CNN training with PyTorch (CPU)
3. Ensemble training (3× XGBoost + LightGBM + LogReg)
4. Models saved with timestamped files + `_latest` symlinks
5. Hot-swap: trading bot detects new model files and reloads without restart

### Walk-Forward Backtest
`scripts/backtest.py --mode walk-forward`:
- Train on first 80% of data, test on last 20%
- Models exist only in memory (not saved to disk)
- Simulates: 10ms execution delay, 10% Post-Only rejection, realistic commissions

---

## 9. Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 0 | **Active** | Data collection (72 hours) |
| Phase 1 | Pending | Initial model training + walk-forward validation |
| Phase 2 | Pending | Paper trading (simulated on live data) |
| Phase 3 | Pending | Live trading with $50 deposit |
| Phase 4 | Pending | Scale up if profitable |
