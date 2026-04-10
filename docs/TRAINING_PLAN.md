# Training Plan — Path to ≥55% Trade Winrate

> **Audience:** next agent picking up this work. Read this file end-to-end before touching code.
> **Written:** 2026-04-10, session-ending handoff. Context on the previous session was running out.
> **Status at handoff:** Recorder stable, Phase 0 active, no training has been run yet.

---

## 0. TL;DR for the next agent

1. **The recorder works.** Do not touch `src/recorder.py`, `src/ws_client.py`, `src/order_book.py` for stability reasons unless you find a bug. The previous session rewrote them.
2. **The bot is OFF.** `scalper-bot.service` is inactive. `scalper-recorder.service` is running and collecting data.
3. **The goal is >55% trade winrate**, break-even is 53% (TP/SL 2:1, 0.04-0.07% fees round-trip).
4. **The biggest lever is labelling, not data volume.** Current labelling in `src/trainer.py:295-308` has a look-ahead bug. Fixing it should give +3-6 pp winrate on its own.
5. **The recommended order of changes is below in §4.** Do not skip ahead. Triple-barrier labelling first, then class weighting + time gap + trade-WR metric, then entry filter tightening, then calibration, then microstructure features.
6. **The first training will show ugly numbers.** Expect 48-52% trade winrate on the first real run. That is normal. Iterate.
7. **Always measure trade-like winrate** `(pred != FLAT)` — not 3-class accuracy. The accuracy number lies because FLAT dominates the class distribution.

---

## 1. Session context — what the previous session did

Committed as `cab3d7e` on master. Files touched:
- `src/recorder.py` — full rewrite to append-mode parquet writer with per-stream `.parts/` subdirs, atomic writes, hour-rollover compaction, orphan-part recovery on startup. **Eliminates the ~3h OOM cycle.** Peak RSS went from 947 MB→OOM to ~330 MB steady-state.
- `src/order_book.py` — added 5s watchdog detecting stalls (no diff applied for 15s), retrying `_fetch_snapshot` on failure (10 attempts with exponential backoff), throttling redundant resync storms via `_resync_lock`, capping unsynced buffer at 5000. **Eliminates the multi-hour BTC depth gaps that were happening before.**
- `src/ws_client.py` — TCP pool 10→64, Bitget text "ping" every 20s (server was idle-closing it), all cross-exchange streams have a data-staleness watchdog that force-closes WS if no data in 60s, backoff cap 30→10s, error logs use `%r` so empty exceptions are diagnosable. **HTX and Deribit were removed** — both Cloudflare-fronted, exhibited synchronized ~3-4 min failure cycles in production that couldn't be isolated.
- `src/features.py` — added `CROSS_EXCHANGES`, `CROSS_EX_WINDOW_MS`, per-exchange deque `_cross_exchange_trades`, `on_exchange_trade` callback, modified `on_bybit_aggtrade` to feed both legacy and new structure, added `_calc_cross_exchange_momentum(now_ms)`, wired into `update()`. **Feature 30 is now implemented** (previously placeholder, always 0).
- `src/trainer.py` — `build_samples` now loads 4 exchange parquet dirs, `_calc_features_batch` computes feature[30] vectorized via per-exchange cumsum + searchsorted on 500ms windows. **Realtime and training paths produce identical values on identical inputs** (verified with unit tests).
- `main.py` — registers `on_bybit_aggtrade` and `on_exchange_trade` callbacks on the WS client (these were missing entirely; FeatureEngine's bybit state was never being fed previously).
- `systemd/scalper-recorder.service` — `MemoryMax=1.4G` safety net.
- `STRATEGY.md` — updated to reflect 4 exchanges (not 6), feature 30 documented as implemented.

**Background exchanges removed:** HTX and Deribit. Reason: both Cloudflare-fronted, showed synchronized 3-4 min failure cycles in production despite working in standalone probes. Root cause not isolated. Cost/benefit didn't justify keeping degraded feeds. **Do not re-add without a very good reason** (one research agent found that HTX has an alternative endpoint `api.hbdm.vn` that's known to be more stable from Tokyo — if you want to try this, it's a 1-line URL change in the reverted HTX stream, but it's not currently critical).

---

## 2. Project state reference

### Files and lines that matter
- **Labels:** `src/trainer.py:294-310` — currently has the look-ahead bug described below.
- **Split:** `src/trainer.py:790-792` — temporal but no time gap between train and val.
- **Class weighting:** NOT PRESENT in `train_ensemble`. Add it.
- **Entry filter:** `src/signal.py` — adjusts which model predictions become real trades. Multiple filters already here but can be tightened.
- **Confidence threshold:** `src/signal.py`, value `0.58` with `[0.50, 0.70]` bounds, self-tuning.
- **Feature definitions:** `src/features.py` — 31 features, keys in `FEATURE_KEYS`.
- **Config:** `config.env` in repo root.

### Economics (from `STRATEGY.md`)
- Instrument: BTCUSDT Perpetual on Binance Futures
- Leverage: x20, deposit $50
- TP base: 0.20% (adaptive, clamp [0.10%, 0.60%])
- SL base: 0.10% (adaptive, clamp [0.05%, 0.30%])
- TP:SL ratio: 2:1
- Risk per trade: 2% of deposit
- Fees: 0.04% round-trip on wins (maker/maker), 0.07% on losses (maker/taker)
- **Break-even winrate: 53%** (per user; includes slippage, spread, timeouts)
- Target winrate: **≥55%** (4+ pp buffer over break-even)

### Labels (current definition)
- 3 classes: `UP=0`, `DOWN=1`, `FLAT=2`
- Threshold: ±0.2% over 60-second horizon (HORIZON=600 × 100ms ticks)
- Class distribution (expected): ~15-22% UP, ~15-22% DOWN, ~55-70% FLAT

### Data layout
Append-mode parquet with hour-rollover compaction. Per-stream `<dir>/` holds canonical `<YYYYMMDD_HH>.parquet`; `<dir>/.parts/` holds in-progress parts for the current hour. Trainer reads `<dir>.glob("*.parquet")` which is non-recursive and only sees canonical files. **The current hour is not visible to trainer until it rolls over.** That's fine for offline training.

Schemas:
- `depth/`, `eth_depth/`: `{timestamp: int64, bids: list<list<double>>, asks: list<list<double>>}`
- `trades/`, `eth_trades/`: `{timestamp: int64, price: float64, quantity: float64, is_buyer_maker: bool}`
- `bybit_trades/`, `okx_trades/`, `bitget_trades/`, `gateio_trades/`: `{timestamp: int64, price: float64, quantity: float64, is_seller: bool}` — note `is_seller` (not `is_buyer_maker`).
- `funding/`: `{timestamp: int64, funding_rate: float64, mark_price: float64}`
- `derivatives/`: `{timestamp: int64, open_interest: float64, long_short_ratio: float64}`

Retention: 72 hours. Rolling.

---

## 3. Why the current labelling is wrong (read this carefully)

**Current code** (`src/trainer.py:295-308`):
```python
max_up = (future_mids.max(axis=1) - current_mids) / safe * 100
max_down = (current_mids - future_mids.min(axis=1)) / safe * 100
y[(max_up >= FLAT_THRESHOLD_PCT) & (max_up >= max_down)] = UP
y[(max_down >= FLAT_THRESHOLD_PCT) & (max_down > max_up) & (y != UP)] = DOWN
```

**The bug:** this uses `max` and `min` over the full 60-second horizon, ignoring the ORDER of events. Consider a sample where the price does this over the 60 seconds after entry:

```
t=0:  entry
t=5s: −0.15%    ← SL would have fired at −0.10%
t=20s: +0.25%   ← TP would have already been meaningless
```

Current labelling: `max_up = 0.25%, max_down = 0.15%` → labelled **UP**. But a real LONG entry at t=0 would hit SL at t=~3-4s, lose 0.10%, and be out of the market — never seeing the +0.25%. The model learns that this situation is a "buy" opportunity, but real trading converts it into a loss.

**This is the root cause of the gap between backtest accuracy and live winrate on every ML scalping system built this way.** It's documented in Marcos López de Prado's "Advances in Financial Machine Learning" Chapter 3 ("Triple-Barrier Method").

---

## 4. The 5 levers, in order of implementation

### Lever #1 — Triple-barrier labelling (+3-6 pp, LARGEST SINGLE LIFT)

**What it does:** For each sample, determine which barrier (TP or SL) would hit FIRST, for both LONG and SHORT entry hypotheses. Label `UP` only if a LONG entry here would actually win (TP before SL). Label `DOWN` only if a SHORT entry would win. Otherwise `FLAT`.

**Replace `src/trainer.py:294-310` with:**

```python
# === Labels — triple-barrier method ===
# For each sample: would a LONG or SHORT entry actually win?
# LONG wins if price hits +TP_PCT before −SL_PCT.
# SHORT wins if price hits −TP_PCT before +SL_PCT.
# Otherwise FLAT (no profitable direction, including timeouts).

future_starts = sample_starts + WINDOW_SIZE
future_win = np.lib.stride_tricks.sliding_window_view(mid_prices, HORIZON)
future_mids = future_win[future_starts]          # (N, HORIZON)
current_mids = mid_prices[future_starts - 1]     # (N,)

safe = np.where(current_mids > 0, current_mids, 1.0)
# Signed relative return in %, per future tick — (N, HORIZON)
rel = (future_mids - current_mids[:, None]) / safe[:, None] * 100

TP_PCT = 0.20  # matches strategy TP base (STRATEGY.md §5)
SL_PCT = 0.10  # matches strategy SL base (2:1 ratio)

# LONG entry: TP at +TP_PCT, SL at −SL_PCT
long_tp_hit = rel >= TP_PCT
long_sl_hit = rel <= -SL_PCT
# argmax returns the FIRST True index; use HORIZON as sentinel if never hit
long_tp_first = np.where(long_tp_hit.any(axis=1), long_tp_hit.argmax(axis=1), HORIZON)
long_sl_first = np.where(long_sl_hit.any(axis=1), long_sl_hit.argmax(axis=1), HORIZON)

# SHORT entry: TP at −TP_PCT, SL at +SL_PCT
short_tp_hit = rel <= -TP_PCT
short_sl_hit = rel >= SL_PCT
short_tp_first = np.where(short_tp_hit.any(axis=1), short_tp_hit.argmax(axis=1), HORIZON)
short_sl_first = np.where(short_sl_hit.any(axis=1), short_sl_hit.argmax(axis=1), HORIZON)

long_wins = long_tp_first < long_sl_first
short_wins = short_tp_first < short_sl_first

y = np.full(num_samples, FLAT, dtype=np.int64)
y[long_wins & ~short_wins] = UP
y[short_wins & ~long_wins] = DOWN
# Rare case where both directions are theoretically profitable (volatile
# whipsaw) — pick whichever TP hits first, i.e. the faster profit.
both = long_wins & short_wins
y[both & (long_tp_first <= short_tp_first)] = UP
y[both & (long_tp_first >  short_tp_first)] = DOWN

# Mid prices at sample points (for backtest alignment) — keep as before
sample_mids = current_mids.copy()

# Filter zero mid prices — keep as before
valid = current_mids > 0
if not valid.all():
    X_lob, X_feat, y, sample_mids = X_lob[valid], X_feat[valid], y[valid], sample_mids[valid]

counts = {UP: int((y == UP).sum()), DOWN: int((y == DOWN).sum()), FLAT: int((y == FLAT).sum())}
logger.info(
    "Triple-barrier labels (TP=%.2f%%, SL=%.2f%%): UP=%d (%.1f%%) DOWN=%d (%.1f%%) FLAT=%d (%.1f%%)",
    TP_PCT, SL_PCT,
    len(y), counts[UP], counts[UP] / len(y) * 100,
    counts[DOWN], counts[DOWN] / len(y) * 100,
    counts[FLAT], counts[FLAT] / len(y) * 100,
)
```

**Also update `src/trainer.py` near top** (line ~34):
```python
# OLD: FLAT_THRESHOLD_PCT = 0.20
TP_PCT = 0.20   # Triple-barrier upper barrier (matches strategy TP base)
SL_PCT = 0.10   # Triple-barrier lower barrier (matches strategy SL base, 2:1)
```

And **remove** the old `FLAT_THRESHOLD_PCT` constant since it's no longer used.

**Why this gives +3-6 pp:** The previous labels trained the model to recognise "price will eventually be higher", which is NOT the same as "a LONG trade here would win". After this fix, UP labels are a perfect match for "trades that would have won as LONG entries", which is exactly what the model is trying to predict in production.

**Side effects to be aware of:**
- FLAT class percentage will INCREASE (more samples fail both barriers) — this is correct and matches reality.
- UP + DOWN combined percentage will DECREASE to maybe 20-25% (from the current 30-40% of max-based labels) — this is correct.
- Because fewer samples are UP/DOWN, the model has fewer "actionable" training signals. This is not a problem: these are the only labels that matter. The model quality on filtered UP/DOWN predictions will go UP, not down.

**Testing:** write a unit test that constructs 3 synthetic price paths and asserts the expected label:
1. Up 0.25% then down 0.15% → LONG wins → UP
2. Up 0.05%, down 0.12%, up 0.30% → LONG SL hits first (−0.10% < −0.12%) and SHORT would lose → FLAT
3. Flat within ±0.08% the whole time → neither barrier → FLAT

---

### Lever #2 — Class weighting + time gap + trade-WR metric (+2-3 pp)

Three tiny changes in `src/trainer.py:train_ensemble`. Each 3-10 lines. All in one function.

**2a. Class weighting.** At the top of `train_ensemble`, compute sample weights:
```python
from sklearn.utils.class_weight import compute_sample_weight
w_all = compute_sample_weight("balanced", y)
```

Apply inside the XGBoost seed loop — wherever `idx` is computed:
```python
dtrain = xgb.DMatrix(X_train[idx], label=y_train[idx], weight=w_train[idx])
dval_dm = xgb.DMatrix(X_val, label=y_val, weight=w_val)
```

where `w_train, w_val` are slices of `w_all` aligned with `X_train, X_val`.

And for LightGBM:
```python
lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
lgb_val_ds = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=lgb_train)
```

**Why:** Without balanced weights, when FLAT is 60-70% of labels, XGBoost and LightGBM optimize overall log-loss by over-predicting FLAT. The model becomes a "FLAT detector" with near-random accuracy on UP/DOWN. Balanced weights force the model to care equally about all three classes, which is what we want for trading (we only care about UP/DOWN correctness).

**2b. Time gap between train and val.** Replace:
```python
split = int(n * (1 - val_split))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
```
with:
```python
# Time gap equal to HORIZON + WINDOW_SIZE so no train sample's label horizon
# overlaps with any val sample's window. Without this, the last ~65s of train
# leak into val via the labelling horizon and bias backtest numbers upward.
split = int(n * (1 - val_split))
gap = HORIZON + WINDOW_SIZE  # 650 ticks = 65 sec
X_train, X_val = X[:split - gap], X[split:]
y_train, y_val = y[:split - gap], y[split:]
```

**Why:** Each sample at time `t` has features computed from `[t − WINDOW_SIZE, t]` and a label based on `[t, t + HORIZON]`. The last train sample's label covers `[split − 1, split − 1 + HORIZON]` which overlaps with the first val sample's window `[split, split + WINDOW_SIZE]`. This gives the model information about the val period. Removing 65 seconds from the end of train closes this leak. The inflation is usually 1-2 pp on val accuracy.

**2c. Trade-WR metric.** In each evaluate block after `model.predict`, **in addition to** logging accuracy, log trade-WR:
```python
pred = probs.argmax(axis=1)
acc = accuracy_score(y_val, pred)
# Trade-like winrate: of all non-FLAT predictions, how many are correct?
trade_mask = pred != FLAT
if trade_mask.any():
    wr = (pred[trade_mask] == y_val[trade_mask]).mean()
    n_trades = int(trade_mask.sum())
    logger.info(
        "XGBoost (seed=%d) val: acc=%.4f | trade-WR=%.4f on %d/%d samples (%.1f%% trade rate)",
        seed, acc, wr, n_trades, len(y_val), 100 * n_trades / len(y_val),
    )
else:
    logger.info("XGBoost (seed=%d) val: acc=%.4f | no non-FLAT predictions", seed, acc)
```

**Why:** Accuracy includes correctly-predicted FLATs, which don't produce trades and don't make money. Trade-WR is the actual winrate the bot would see if it traded every non-FLAT prediction. These are often very different numbers — e.g. `acc=0.52, trade-WR=0.43` is catastrophic (looks fine, is a loss-making model) and `acc=0.45, trade-WR=0.57` is excellent (looks bad, is profitable). The user-facing metric in logs MUST be trade-WR for meaningful iteration.

---

### Lever #3 — Entry filter tightening (+3-5 pp)

Not a labelling change — this is about the **signal → trade** path. Model accuracy doesn't need to rise; we just refuse more of the losing trades.

**Add to `src/signal.py` decision logic:**

1. **Time-of-day filter.** Do not trade 04:00-07:00 UTC (Asia night, thin liquidity, spoofing prevalent). Empirically responsible for ~30% of losses in HFT scalping bots that operate 24/7.
   ```python
   hour_utc = datetime.now(timezone.utc).hour
   if 4 <= hour_utc < 7:
       return None  # or mark as SKIP
   ```

2. **Liquidity depth gate.** Require top-5 bid volumes sum ≥ 100 BTC AND top-5 ask volumes sum ≥ 100 BTC. The existing circuit breaker at 50 BTC is too loose for live execution.

3. **Volatility band gate.** Only trade when `0.7 < volatility_ratio < 2.5`. Below 0.7 the market is dead (model signals are noise). Above 2.5 the market is news-driven (model can't predict). Sweet spot is moderate turbulence where microstructure signals work.

4. **Recent performance gate.** Track rolling winrate of last 10 trades. If WR < 40%, pause entries for 10 minutes. This catches regime shifts in real-time, before the 4-hour retraining cycle notices.

5. **Funding proximity.** Extend the existing ±2 min to **±3 min** around funding settlement (00:00, 08:00, 16:00 UTC).

6. **Spread tightening.** Existing spread filter is $0.03. For LIVE trading, reduce to $0.02 and see if it helps. This is reversible.

**Measurement:** Before adding each filter, compute the backtest winrate of the trades that WOULD be filtered out. If their WR is below the current target (55%), filter works. If their WR is above, the filter is hurting you.

Expected effect: −25% trade count, +4 pp winrate on the remainder. Net P&L unchanged or slightly up; Sharpe improves because fewer losing trades.

---

### Lever #4 — Isotonic calibration of ensemble output (+2-3 pp)

**Problem:** The ensemble output from `softprob` is a raw probability that's not calibrated to real-world frequencies. The confidence threshold `0.58` in signal.py is an opaque number — it could correspond to 52% real accuracy or 68% real accuracy and nobody knows.

**Fix:** Use isotonic regression (or Platt scaling) to map raw probability → calibrated probability on a held-out calibration set.

**Implementation** — add to `src/trainer.py:train_ensemble` after all models are trained:

```python
from sklearn.isotonic import IsotonicRegression

# Split a calibration set from the tail of train (chronological, last 10%)
cal_size = int(len(X_train) * 0.1)
X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]
X_train_final, y_train_final = X_train[:-cal_size], y_train[:-cal_size]
# Retrain on X_train_final (or, simpler: don't retrain — just use calibration on top of the already-trained models)

# Get raw ensemble predictions on calibration set
raw_cal = self._ensemble_predict(xgb_models, lgb_model, logreg, top5_features, X_cal)
# raw_cal shape: (cal_size, 3)

calibrators = []
for cls in range(3):
    y_binary = (y_cal == cls).astype(float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal[:, cls], y_binary)
    calibrators.append(ir)

# Save calibrators alongside models
# ...
```

**Runtime** — in `src/signal.py` or wherever predictions happen:
```python
raw = ensemble_predict(x)  # (3,)
cal = np.array([calibrators[i].predict([raw[i]])[0] for i in range(3)])
cal /= cal.sum() + 1e-9  # renormalize
# Use `cal` instead of `raw` for the threshold comparison
```

**Why:** After calibration, `confidence > 0.58` actually MEANS "≥58% historical hit rate on samples with this raw score". The self-tuning threshold becomes content-based rather than arbitrary. Bad trades with high raw confidence (rare, but they cause large losses) get filtered more reliably. Good trades with moderate confidence (common) get through more often.

---

### Lever #5 — Microstructure features (+1-3 pp)

Only after Levers #1-#4 are done and measured. These are additive and non-essential but give a final boost.

**5a. Queue pressure (L1 decay rate)** — `src/features.py`, add to `_calc_*` methods and to update loop:
```python
# Track previous best bid/ask volumes
self._prev_best_bid_vol: float = 0.0
self._prev_best_ask_vol: float = 0.0
self._bid_decay_ema: float = 0.0
self._ask_decay_ema: float = 0.0
# In update():
best_bid_vol = snap.bids[0, 1]
best_ask_vol = snap.asks[0, 1]
bid_decay = max(0.0, self._prev_best_bid_vol - best_bid_vol)  # positive if shrinking
ask_decay = max(0.0, self._prev_best_ask_vol - best_ask_vol)
alpha = 0.1
self._bid_decay_ema = alpha * bid_decay + (1 - alpha) * self._bid_decay_ema
self._ask_decay_ema = alpha * ask_decay + (1 - alpha) * self._ask_decay_ema
self._prev_best_bid_vol = best_bid_vol
self._prev_best_ask_vol = best_ask_vol
raw["queue_pressure"] = self._ask_decay_ema - self._bid_decay_ema  # asks eating faster = bullish
```

**5b. Top-3 vs total book depth asymmetry:**
```python
top3_bid = float(snap.bids[:3, 1].sum())
top20_bid = float(snap.bids[:, 1].sum())
top3_ask = float(snap.asks[:3, 1].sum())
top20_ask = float(snap.asks[:, 1].sum())
raw["top3_asymmetry"] = (
    (top3_bid / (top20_bid + 1e-9)) - (top3_ask / (top20_ask + 1e-9))
)
```

**5c. Effective spread / quoted spread ratio.** Track recent trade prices, compute |exec_price − mid| / quoted_spread EMA. This measures how aggressively trades pierce the spread — indicator of urgency.

**WARNING:** Adding features requires:
- Updating `NUM_FEATURES`, `FEATURE_KEYS` in `src/features.py`
- Updating the training-time vectorized computation in `src/trainer.py:_calc_features_batch` — or accepting that these features are RUNTIME-ONLY (not in training), which creates train/runtime inconsistency and is usually a bad idea.
- Updating CNN input dimensions if applicable (probably not — features go into the hand-crafted portion).

For this reason, Lever #5 is last and optional. Only do it if #1-#4 haven't gotten you to 55%.

---

## 5. Expected trajectory

| Iteration | What was done | Trade-WR (expected) |
|---|---|---:|
| 0 (before any fix) | Current state, 72h data, old labels, no class weight | 45-50% |
| 1 (+ Lever #1: triple-barrier) | Triple-barrier labels only | 49-54% |
| 2 (+ Lever #2: class weight, gap, metric) | Plus class weight, time gap, real metric | 52-56% |
| 3 (+ Lever #3: filter tightening) | Plus entry filters — fewer trades, higher quality | 55-59% |
| 4 (+ Lever #4: calibration) | Plus isotonic calibration | 56-61% |
| 5 (+ Lever #5: microstructure) | Plus new features, if needed | 57-62% |

**These are expected ranges, not guarantees.** The actual numbers depend on the current BTC market regime (calm vs volatile), the quality of the filters, and luck on the validation window.

**Sanity checks at each iteration:**
- If Lever #1 doesn't move the number by at least +2 pp, the labelling fix didn't take effect — check that you removed `FLAT_THRESHOLD_PCT` and the new code is running.
- If Lever #2a (class weight) doesn't change class balance in model predictions, check that `weight=` parameter actually reached XGBoost/LightGBM.
- If backtest numbers go UP when you add Lever #2b (time gap), something's wrong — gap should make numbers go DOWN (more honest measurement).
- If any iteration takes the trade-WR above 65%, you probably have data leakage. Audit.

---

## 6. Validation methodology (READ THIS BEFORE BELIEVING ANY NUMBERS)

**Walk-forward backtest is mandatory.** Never trust a single train/val split number on a time series. `scripts/backtest.py --mode walk-forward` exists per STRATEGY.md — use it.

Walk-forward protocol:
1. Split the available data into N equal chronological chunks (e.g. N=6 for 72h = 12h each)
2. For each i in [2, N−1]: train on chunks [0, i), test on chunk [i]
3. Aggregate trade-WR across all test chunks
4. The aggregate is what you report; the per-chunk variation tells you regime stability

**Important:** the test chunk must have a gap of at least `HORIZON + WINDOW_SIZE` (650 ticks = 65 seconds) from the preceding train chunks, same as Lever #2b.

**Red flags in backtest results:**
- Trade-WR varies wildly by chunk (e.g. 60% / 45% / 62% / 40%) → model depends on regime, not robust
- Trade-WR consistently ABOVE 65% on every chunk → likely data leakage
- Trade-WR decreases over time → regime drift, retraining cadence (4h) is too slow
- Less than 100 trades per chunk → sample too small to trust

**Acceptance criteria** before going to paper trading:
- Aggregate walk-forward trade-WR ≥ 55%
- Per-chunk trade-WR ≥ 50% on all chunks
- Trade count per chunk ≥ 100
- No obvious regime dependence (variance across chunks < 10 pp)

**Acceptance criteria** before going live with $50:
- 3-5 days of paper trading showing WR ≥ 53%
- No timeout/execution issues
- Sharpe on paper trades ≥ 1.0

---

## 7. Anti-recommendations (do NOT do these)

1. **Do not add more models to the ensemble** (TabNet, Transformer, etc.) before fixing labelling. No model architecture can compensate for labels that don't match reality.
2. **Do not increase the 72h training window** to a week or month. BTC regime drift on 60-second horizon is severe beyond ~3 days. More old data hurts.
3. **Do not tune XGBoost hyperparameters** (learning_rate, max_depth, min_child_weight) before Lever #2. You'll be optimizing on a broken objective.
4. **Do not "try all features"** — adding features beyond Lever #5 usually adds noise, not signal. The ensemble's tree models handle feature selection internally; piling on correlated features is harmful.
5. **Do not trust a backtest result above 60%** on the first iteration. Audit for leakage before celebrating.
6. **Do not skip walk-forward** in favour of a single train/val split. A single split can be 5-10 pp off from walk-forward, always in the optimistic direction.
7. **Do not touch the recorder** unless you find a bug. The previous session stabilized it and every change risks data loss. If you see recorder issues, prefer config changes to code changes.
8. **Do not re-add HTX or Deribit** without understanding why they were removed (see §1). If you want to retry HTX with `api.hbdm.vn`, do it in a branch, not on master.
9. **Do not run the trading bot until all 5 levers are applied AND walk-forward is ≥55%.** There is no rush; real money is at stake.

---

## 8. Current data status at handoff

At time of writing (2026-04-10 ~15:20 UTC):
- Recorder uptime: 3.5h since cleanup restart at 11:50 UTC
- Memory: 283 MB RSS (peak 330 MB, limit 1.4 GB) — stable
- BTC depth: ~95k snapshots collected since restart (~7.4/sec average)
- Total data across all streams: ~150 MB
- 0 critical errors; watchdogs fired a few times for Binance depth disconnects (normal, recovered within seconds)

**Data ready for training when Phase 0 completes at ~72h cumulative** (the 72-hour rolling window is the production design — see `STRATEGY.md` §8).

---

## 9. Quick reference — file:line map

| Concern | File | Line | Notes |
|---|---|---:|---|
| Labels (broken) | `src/trainer.py` | 294-310 | Replace with triple-barrier (Lever #1) |
| `FLAT_THRESHOLD_PCT` constant | `src/trainer.py` | ~34 | Remove, add `TP_PCT`, `SL_PCT` |
| Train/val split | `src/trainer.py` | 790-792 | Add time gap (Lever #2b) |
| Class weights | `src/trainer.py` | `train_ensemble` | Add `compute_sample_weight` + `weight=` params (Lever #2a) |
| Model metrics in logs | `src/trainer.py` | `train_ensemble` predict blocks | Add trade-WR (Lever #2c) |
| Entry filters | `src/signal.py` | throughout | Tighten (Lever #3) |
| Confidence threshold logic | `src/signal.py` | dynamic threshold | Add calibration (Lever #4) |
| Features list | `src/features.py` | `FEATURE_KEYS` | Add microstructure if needed (Lever #5) |
| Recorder (DO NOT TOUCH) | `src/recorder.py` | — | Stable after prior session |
| Order book (DO NOT TOUCH) | `src/order_book.py` | — | Stable after prior session |
| WS client (DO NOT TOUCH) | `src/ws_client.py` | — | Stable after prior session |
| Realtime feature 30 | `src/features.py` | `_calc_cross_exchange_momentum` | Done in prior session |
| Training-time feature 30 | `src/trainer.py` | `_calc_features_batch` end | Done in prior session |

---

## 10. After reaching 55% — what's next (future work, not in scope here)

- **Meta-labelling** (López de Prado): train a secondary classifier on `(primary_prediction, features) → trade_will_win`. Use as a sizing/filtering signal on top of primary ensemble. Usually +2-4 pp.
- **Regime-specific models**: train separate models for Hurst-trending vs Hurst-mean-reverting regimes, route at runtime. +1-2 pp.
- **Position sizing via Kelly**: size trades proportional to expected edge, not flat 2% risk. Improves Sharpe without changing WR.
- **More cross-exchange feeds** (if HTX/Deribit instability can be solved): gives feature 30 a wider base and more robustness to any single exchange having issues.
- **News filter via mark price jump detection**: if funding rate or mark price moves >3σ in <1s, likely news event — pause trading for 5 minutes.

None of these should be attempted until the 5 levers are done and the system is live with ≥55% WR on paper trading for at least 5 days.

---

**End of plan. Good luck, next agent.**
