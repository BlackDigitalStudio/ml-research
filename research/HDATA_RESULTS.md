# HDATA — experiment results record (session 2026-05-23)

Hypothesis **HDATA rev1** (registered `research/hypotheses.jsonl`, branch
`claude/hdata-rev1`, commit `8ef1b60`, NOT pushed): open/unbounded "under what
training-data composition do models best extract directional signal?".
Metric: `rank_IC = AUC − 0.5` (OOS directional skill). Cost reference: 0.13% round-trip.
Facts/numbers only — no interpretation.

GCS bucket: `gs://blackdigital-scalper-data/`.

---

## 1. Data acquired (free)

**`free_v1/`** — Binance USD-M perp dumps (data.binance.vision), 521 symbols, 2yr, **23.77 GiB**:
- `klines_5m` 521 syms, 73.8M rows · `klines_1m` 521, 369M · `metrics`(OI/top+global LSR/taker-vol) 524, 75.7M · `funding` 521, 1.29M
- history: 206 syms ≥2yr, 150 1–2yr, 171 <1yr.

**`free_v1/orthogonal/`**:
- `fear_greed/fng.parquet` — alternative.me, 3030 daily
- `macro/macro_daily.parquet` — Yahoo chart API: spx, vix, dxy(DX-Y.NYB), gold(GC=F), ust10y(^TNX), ust5y(^FVX); 3181 rows
- `onchain/`: `stablecoin_supply`(DeFiLlama, 3097), `defi_tvl`(3160), `dex_volume`(3671)
- `coinalyze/` — 530 aggregate `.A` cross-exchange perps: `open-interest`131k, `funding-rate`145k, `predicted-funding-rate`135k, `long-short-ratio`121k, `liquidation`130k, `ohlcv`145k rows
- `news/gkg_crypto_15min.parquet` — GDELT GKG via BigQuery, 15-min crypto tone+doc_count, 30,255 bins, 2025-05..2026-05
- `news/gkg_crypto_events.parquet` — GDELT GKG per-event (dt, tone, url, themes), 149,907 events, 12mo
- `news/gdelt_news.parquet` — GDELT DOC API daily tone+vol, 3 topics, 714 rows

**Pre-existing**: `raw/`(L2 book/trades/funding/liq/OI, 8 Cryptolake symbols, ns-timestamped), `features_v1/`(59-col, 8 symbols).

**Not acquired**: Deribit DVOL (deribit.com unreachable from local network, SSLError/http=000); token unlocks (DeFiLlama `/emissions` = HTTP 402 paid).

---

## 2. EXP-1 — L2 pooled vs solo XGB
features_v1 (59 feats + 7 event-conds), 7 symbols (ex-LINK), 120d, first-passage ±0.13% labels, R1 |move|-weighted XGB, honest 70/30 + embargo, STRIDE=4. `research_runs/hdata_pool_poc/results.json`. rank_IC solo/pooled:

| sym | H180 | H300 | H600 |
|---|---|---|---|
| BTC | .0543/.0509 | .0324/.0248 | .0149/.0144 |
| ETH | .0213/.0319 | .0086/.0168 | .0021/.0050 |
| BNB | .0000/.0530 | .0276/.0236 | .0124/.0205 |
| DOGE| .0490/.0456 | .0276/.0353 | .0260/.0306 |
| LTC | .0000/.0832 | .0046/.0100 | −.0032/.0015 |
| SOL | .0233/.0354 | .0044/.0165 | .0191/.0223 |
| XRP | .0227/.0306 | .0132/.0275 | .0038/.0093 |

Pooled-ALL OOS rank_IC: H180 **+0.0399** (n_oos 24,711), H300 +0.0214 (38,405), H600 +0.0126 (59,881).

---

## 3. EXP-2 — Free-panel pooled time-series
120 longest-history symbols, free 5m bars+metrics+funding, hourly decisions, sign(fwd logret) R1-XGB, honest split, per-symbol train-fit z-score. `research_runs/hdata_freepanel/results.json`. pooled_rows 1,918,466.

| H | pooled rank_IC | per-sym median | pos_frac | top-dec \|move\| | n_oos |
|---|---|---|---|---|---|
|1h|+0.0005|−0.0009|0.433|0.826%|575,423|
|4h|+0.0034|+0.0032|0.600|1.570%|575,069|
|8h|+0.0073|+0.0069|0.717|2.279%|574,597|

Top feature importances (4h/8h): vol_24h, r_24h, r_8h, range_1h, r_7d, vol_7d.

---

## 4. EXP-3 — Free-panel horizons + cross-section
Same panel, H∈{4,8,12,24}h, feature sets bars vs bars+cross-sectional. `research_runs/hdata_freepanel2/results.json`. pooled_rows 1,918,466.

| H | bars | bars+xs | xs lift | med\|move\| |
|---|---|---|---|---|
|4h|+0.0034|+0.0084|+0.0050|0.892%|
|8h|+0.0073|+0.0111|+0.0038|1.323%|
|12h|+0.0194|+0.0159|−0.0034|1.660%|
|24h|**+0.0275**|+0.0248|−0.0027|2.394%|

Cross-sectional feats: xs_rank_r24h, xs_demean_r24h, xs_rank_r8h, xs_rank_oichg24h, xs_rank_funding, breadth_r24h.

---

## 5. EXP-4 — Data-axis attribution @ H=8h
120 syms, cumulative feature sets, daily orth/Coinalyze lagged 1 day (leak fix). `research_runs/hdata_freepanel3/results.json`. n_oos 575,317, med|move| 1.32%.

| feature set | rank_IC (lag-fixed) | rank_IC (same-day, leaky) |
|---|---|---|
| bars | +0.0087 | +0.0087 |
| +cross-section | +0.0119 | +0.0119 |
| +global_orth (F&G/macro/on-chain) | +0.0283 | +0.0516 |
| +coinalyze (cross-exch OI/funding/LS/liq) | +0.0238 | +0.1046 |

Leak: daily orth/Coinalyze joined same-date contained end-of-day values overlapping the forward window; fixed by joining day D−1 features to day-D decisions. global_orth feats are cross-sectionally constant across symbols at time t (pooled rank_IC carries cross-sectional redundancy, n_eff ≈ #timestamps). GORTH join coverage 100%, Coinalyze ~50%.

---

## 6. EXP-5 — Liquidation-cascade event-study
8 Cryptolake symbols, 90d, raw/liquidations (event-level) + raw/book L1 mid, 5s cascade bins, per-symbol USD percentile tiers. `research_runs/hdata_liqcascade/results.json`. 326,293 bins. dir_ret = sign(net liq pressure)×fwd_ret.

top-0.1% (n=330, med event $1.48M):
| H | dir_ret | win-rate | rank_IC(pressure,ret) | med\|ret\| | clear_cost |
|---|---|---|---|---|---|
|5s|+2.42bp|0.442|+0.040|0.034%|0.145|
|15s|+4.19bp|0.494|+0.074|0.061%|0.245|
|30s|+8.26bp|0.491|+0.069|0.073%|0.364|
|60s|+13.22bp|0.406|−0.123|0.109%|0.427|
|300s|+13.53bp|0.442|−0.071|0.242%|0.709|
|600s|+1.95bp|0.400|−0.159|0.314%|0.767|

top-1% (n=3266): dir_ret +1.1..+2.5bp, win 0.400–0.448. top-10% (n=32633): dir_ret +0.5..+3.3bp, win 0.419–0.442.

---

## 7. EXP-6 — News aggregate-tone backtest
GDELT GKG 15-min crypto tone vs BTC/ETH 1m klines, entry=bin-end (T+15m). Local. spike-events = doc_count top-10% (n=3047).

| sym | H | rank_IC(tone,ret) | spike win-rate | spike dir_ret | med\|ret\| | clear_cost |
|---|---|---|---|---|---|---|
|BTC|15m|+0.0031|0.515|+0.34bp|0.130%|0.498|
|BTC|60m|+0.0005|0.499|+1.45bp|0.246%|0.714|
|BTC|240m|−0.0052|0.494|−0.41bp|0.512%|0.843|
|ETH|60m|+0.0062|0.511|+2.67bp|0.391%|0.802|
|ETH|240m|+0.0026|0.511|+5.28bp|0.830%|0.898|

(BTC 30m rank_IC −0.0065; ETH 15m +0.0046, 30m +0.0007.)

---

## 8. EXP-7 — News classified-direction backtest
149,907 GKG events, keyword-classified bull/bear (url+themes), vs klines, entry=event dt. Local. dir counts: bull 22,351 / bear 25,770 / neutral 101,786.

| set | n | H15m | H30m | H60m | H240m |
|---|---|---|---|---|---|
| ALL-dir→BTC | 48,121 | dir −0.06bp / win 0.503 | −0.09/0.499 | −0.15/0.501 | −1.51/0.492 |
| BTC-headline→BTC | 21,017 | −0.66/0.497 | −1.15/0.485 | −1.64/0.487 | −5.42/0.474 |
| ETH-headline→ETH | 3,321 | +1.66/0.508 | +0.20/0.503 | +0.03/0.499 | +1.48/0.502 |

BTC-headline bull_mean/bear_mean by H: 15m −0.0/+1.4bp, 30m −0.2/+2.2, 60m −0.5/+2.9, 240m −1.6/+9.6.

---

## 9. EV-A — perp-onboarding event-study (HDATA cell EV-A)
524 Binance USD-M perps, `free_v1/klines_1m` (open_time+close only, GCS col-projection). Event = a symbol's first 1m candle (perp onboarding). entry = first-candle close; no new data. true-onboarding set = t0 > global_data_start (2024-05-01 00:00 UTC) + 7d → **n=320**; censored (anchor = window start) n=204. Local: `ev_onboarding.py` → `ev_onboarding.json`.

Move magnitude + drift from first-candle close, TRUE onboardings (n=320):
| h | med\|r\| | clear_cost@0.13% | mean drift | med drift | frac>0 |
|---|---|---|---|---|---|
|1m|1.758%|0.916|+56.27bp|+18.33bp|0.525|
|5m|2.886%|0.947|+110.03bp|+20.47bp|0.525|
|15m|4.076%|0.950|+162.83bp|+83.02bp|0.550|
|60m|5.785%|0.947|+51.09bp|+0.00bp|0.494|

Early momentum/reversal: rank_IC(early[0,k], next[k,2k]); dir_ret = sign(early)·next; n=320; ± = block-bootstrap SE:
| k | rank_IC | dir_ret | win | med\|next\| |
|---|---|---|---|---|
|5m|+0.0207 ±0.0326|+3.67bp ±31.88|0.500|2.116%|
|15m|−0.0315 ±0.0327|−13.06bp ±41.29|0.453|3.063%|
|30m|−0.0572 ±0.0327|−53.27bp ±44.08|0.444|3.176%|

Placebo (censored, n=204): all share one anchor (2024-05-01 00:00 UTC) → its drift/IC are a single-hour cross-sectional artifact, not a valid null; med|r| 0.063–0.992% (vs TRUE 1.76–5.79%).

Notes: n=320 (rank_IC SE ≈ 0.033 → |rank_IC| < ~0.066 within 2σ of 0); survivorship (universe-present syms only); the first-1m close already contains the onboarding auction pop, so any sub-minute directional move is invisible at 1m granularity.

---

## 10. EV-B — exchange-listing announcement event-study (HDATA cell EV-B)
Upbit (`api-manager /announcements`, category=trade, `listed_at` sec-precision KST) + Binance (`bapi cms article/list` catalogId=48 "New Listing", `releaseDate` ms). Ticker parsed from title parens; sign from EN+KR keywords (list+ / delist+caution− / add+ / airdrop+). Matched ticker→Binance USDT perp (524 universe); entry = first 1m close ≥ announcement ts; events where the perp did not exist at ts dropped. Local: `ev_b_listings.py`, `ev_b_probe.py` → `ev_b_listings.json`; announcements → `gs://…/free_v1/orthogonal/events/listings/announcements.parquet`. Binance FULL 2158 (single clean retry-enabled run; an earlier accidental 4× concurrent launch had triggered Binance rate-limit truncation — fixed by serial run + per-page retry).

Pulled: Upbit 701 notices / Binance 2158 articles → 1249 ticker-events → **292 tradeable** (perp existed at ts). dir_ret = sign·r_h (bp) / win-rate:

| cohort | n | h1m | h5m | h15m | h60m | med\|r\|@60m |
|---|---|---|---|---|---|---|
| UPBIT list(+) | 130 | −8.8/0.462 | −86.1/0.423 | +6.2/0.485 | −100.8/0.408 | 4.66% |
| UPBIT delist+caution(−) | 18 | +3.2/0.389 | +126.0/0.667 | +92.5/0.667 | +125.4/0.611 | 2.11% |
| BINANCE will-list(+) | 31 | −23.2/0.387 | −168.0/0.484 | +64.5/0.548 | +269.1/0.613 | 8.13% |
| BINANCE will-add(+) | 56 | +1.4/0.536 | −11.5/0.554 | −11.7/0.500 | −264.8/0.321 | 7.83% |
| BINANCE airdrop(+) | 33 | −133.7/0.424 | −78.8/0.394 | −25.6/0.455 | +16.5/0.394 | 5.43% |
| ALL combined | 292 | −22.9/0.459 | −64.8/0.470 | +11.1/0.504 | −62.7/0.425 | 5.42% |

clear_cost@0.13% = 0.88–1.00 across cohorts (med|r| 0.85–8.10%).

Surface read (not verdict): listings/events = max-magnitude regime (move clears cost ~always, as EV-A §9). At 1m granularity from the PUBLISHED ts, the "obvious-direction" long-listing is net negative (UPBIT list −86bp@5m / −101bp@60m, win<0.5); most cohorts drift DOWN post-publication. The only directionally-clean cohort is bearish forced-selling — UPBIT delist+caution: short wins 0.61–0.67, dir_ret +92…+126bp (n=18). Caveats: (a) RESOLVED by the drift control below — the random-timestamp null is ~0 at 1–60m and BTC-relative ≈ raw, so the cohort effects are EVENT-SPECIFIC, not the slow 2024–26 alt downtrend (negligible over minutes) or market beta. (b) Upbit `listed_at` semantics (notice-post vs trading-open) unverified — if the impulse precedes `listed_at`, the 1m entry captures reversion, not impulse. (c) the bullish impulse is plausibly sub-minute (entering at next-1m-close misses it) → the live-feed + tick regime where the 1–3 ms edge applies, not backtestable on free 1m klines. (d) small n (delist 18, will-list 31).

**Drift control** (`ev_b_control.py` → `ev_b_control.json`): random-timestamp null (same symbols+signs, random times, B=1000) + BTC-relative. Null mean ≈ 0 at every horizon (e.g. Upbit-list 5m null −0.16±4.55bp); BTC-relative ≈ raw. dir_ret bp / z-vs-null / pctile:

| cohort | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| UPBIT list(+) | −8.8 / z−4.6 / .001 | −86.1 / z−18.9 / .000 | +6.2 / z+0.8 / .830 | −100.8 / z−6.3 / .002 |
| UPBIT delist+caution(−) | +3.2 / z+0.6 | +126.0 / z+11.8 / 1.00 | +92.5 / z+3.4 / .999 | +125.4 / z+3.1 / .998 |
| BINANCE will-list(+) | −23.2 / z−4.9 / .002 | −168.0 / z−18.1 / .000 | +64.5 / z+3.8 / .999 | +269.1 / z+5.8 / 1.00 |
| BINANCE airdrop(+) | −133.7 / z−32.5 / .000 | −78.8 / z−8.3 / .000 | −25.6 / z−1.6 / .045 | +16.5 / z+0.5 / .740 |
| ALL signed | −22.9 / z−14.9 | −64.8 / z−19.0 | +11.1 / z+2.1 | −62.7 / z−5.8 |

Interpretation flip: the sign(announcement)→r effect is REAL in DIRECTION (null≈0, BTC-rel≈raw → not alt-beta/market) — though the |z| up to 32 here OVERSTATE significance (they compare the event mean to a low-variance random-window null, conflating the mean-shift with the events' huge variance; honest cluster-robust t = 0.7–2.2, see Hardening below). The tradeable direction is **SHORT-the-announcement (fade) at 1–5m**, not long — every listing/airdrop cohort's perp drops sharply in the first 1–5 min (Upbit-list −86bp z−18.9@5m; Binance will-list −168bp z−18.1@5m; airdrop −134bp z−32.5@1m). Longer-horizon structure: Binance will-list is a V (−168bp@5m → +269bp@60m, z+5.8); Upbit-list stays down (−101bp@60m, z−6.3); Upbit delist/caution → perp drops (short, z+3…+12). Magnitudes 5–13× the 0.13% cost floor. NEW caveats: (e) z assumes event independence — Upbit/Binance post in BATCHES (multiple tickers at one ts) → clustering inflates z; the dir_ret point estimates are robust (consistent across 3 distinct cohorts), the exact significance is not (cluster/day-block bootstrap pending). (f) entry = first 1m close after ts (~30–80s late): the 1m leg (airdrop −134bp@1m) is partly co-incident with entry; the 5–60m legs are post-entry and capturable with the 1–3 ms edge; sub-minute fills/slippage at these volatile prints not modeled on free 1m klines.

**Hardening** (`ev_b_harden.py` → `ev_b_harden.json`): cluster-robust t (collapse co-ts batch events → t on unique-ts means) + net-of-cost (13bp round-trip) fade economics (strategy = SHORT perp at entry, hold H) + KRW split. SHORT net bp / cluster-robust t / win-rate:

| cohort | n / n_ts | SHORT 5m: net / t / win | SHORT 60m: net / t / win | note |
|---|---|---|---|---|
| UPBIT list ALL | 130/120 | +73 / 1.61 / 0.57 | +88 / 0.75 / 0.59 | med_net +49 / +141 |
| UPBIT list KRW | 104/103 | +58 / 1.37 / 0.52 | +74 / 0.71 / 0.56 | KRW "Upbit-effect" = weaker/noisier |
| UPBIT list non-KRW | 26/19 | +132 / 0.97 / 0.77 | +145 / 0.40 / 0.73 | higher win, small n |
| UPBIT delist+caution | 18/18 | +113 / **2.24** / 0.67 | +112 / 1.06 / 0.61 | |
| BINANCE will-list | 31/23 | +155 / 0.39 / 0.52 | −282 / −1.09 / 0.39 | short ONLY ≤5m (V) |
| BINANCE airdrop | 33/33 | +66 / 0.69 / 0.61 | −30 / − / 0.61 | 1m: +121 / **2.20** / 0.58 |
| BINANCE will-list V long (buy+5m→sell+60m) | 31/23 | net **+424** / t **1.81** / win 0.68 | | recovery leg |

Hardened read: the short-the-announcement (≤5m) and the Binance will-list V-recovery long-leg are POSITIVE-EV in BOTH mean and median, win 0.52–0.77, net magnitudes +58…+424bp (large vs 13bp cost) — BUT cluster-robust significance is MODEST (t mostly 0.7–1.6; only delist-5m t2.24, airdrop-1m t2.20, will-list-V t1.81 approach/cross 2σ) at small n (n_ts 18–120). Economically promising, high-variance, NOT yet statistically nailed at current n — needs more events (forward accrual / more event classes) or live validation. Counter-intuitive: KRW "Upbit-effect" listings fade WEAKER (win 0.52) than non-KRW (0.77, n=26).

---

## 11. EV-C — scheduled macro event-window reaction (HDATA cell EV-C)
Free historical macro consensus + release datetimes UNAVAILABLE from this network: FRED times out; ForexFactory FairEconomy JSON = current-week only (has datetime+forecast+actual); DeFiLlama emissions = 402 paid; token.unlocks = no open API; CryptoRank = keyed. So release DATETIMES hardcoded best-effort (FOMC 14:00 ET ×17; CPI & NFP 08:30 ET ×24 each; America/New_York→UTC via zoneinfo, DST-correct), reaction on BTC/ETH 1m klines. Direction tested WITHOUT consensus via continuation rank_IC(early[0,k]→next[k,2k]) + vol amplification vs same-symbol random-time baseline. Token unlocks + consensus-surprise DEFERRED (paywalled). Local: `ev_c_macro.py` → `ev_c_macro.json`. CAVEATS: hardcoded dates may have minor errors (esp 2025-26) → dilutes toward null (conservative); small n (FOMC 17, CPI/NFP 24).

Vol amplification = event med|r| / random-baseline med|r| (event med|r| in parens), at 5m / 60m:
| class | n | BTC 5m | BTC 60m | ETH 5m | ETH 60m |
|---|---|---|---|---|---|
| FOMC | 17 | 4.58× (0.289%) | 3.44× (0.725%) | 5.13× (0.466%) | 2.01× (0.614%) |
| CPI | 24 | 2.06× (0.130%) | 1.24× (0.262%) | 2.64× (0.240%) | 2.84× (0.867%) |
| NFP | 24 | 3.75× (0.237%) | 1.73× (0.364%) | 3.14× (0.285%) | 1.47× (0.449%) |
| ALL | 65 | 3.80× (0.239%) | 2.06× (0.434%) | 3.15× (0.286%) | 1.79× (0.546%) |

clear_cost@0.13%: BTC 0.50–0.82 (CPI weakest, 0.50@5m), ETH 0.71–1.00. Continuation rank_IC(early→next) / momentum-win:
| class | BTC k5m | BTC k15m | ETH k5m | ETH k15m |
|---|---|---|---|---|
| FOMC (17) | −0.167 / .35 | −0.257 / .29 | −0.139 / .29 | −0.139 / .29 |
| CPI (24) | −0.097 / .42 | +0.102 / .54 | −0.063 / .46 | +0.046 / .42 |
| NFP (24) | +0.018 / .58 | −0.048 / .50 | +0.028 / .46 | −0.107 / .42 |
| ALL (65) | −0.057 / .46 | −0.053 / .46 | −0.033 / .42 | −0.064 / .39 |

Surface read: macro releases are robust VOL amplifiers on crypto (FOMC/NFP 3–5× @5m, CPI 2–2.6×) but absolute moves are MODEST on BTC/ETH (med|r| 0.13–0.87%, ~an order smaller than EV-B alt-listings' 1.2–8%; BTC CPI clears cost only 0.50@5m) — large-cap efficiency. Direction (no consensus, continuation proxy): FOMC leans REVERSAL (BTC+ETH rank_IC −0.14…−0.26, momentum-win 0.29–0.35 → fading the initial spike wins ~0.65–0.71) — echoes EV-B "fade the event" — but n=17, noisy; CPI/NFP ~0. Underpowered without the consensus surprise. Net: macro = real vol catalyst, thin/uncertain direction on liquid majors; far weaker than EV-B listings.

### Event-driven synthesis (EV-A..C)
- **WHERE (vol):** events reliably mark volatility — EV-A listings med|r| 1.8–5.8% (clear cost ~0.95), EV-C macro vol-amp 3–5×. Consistent with the program-wide pattern (free/public data marks vol well).
- **DIRECTION:** EV-A (perp price autocorrelation at listing) → none robust. EV-B (announcement = known type/sign) → a REAL, de-confounded (null≈0, BTC-rel≈raw) **fade** signal: short-the-announcement ≤5m + Binance will-list V-recovery long; net +58…+424bp/event, win 0.52–0.77, but cluster-robust t modest (0.7–2.2), n-limited. EV-C macro → thin reversal-leaning direction on majors, underpowered (no consensus).
- **Unifying theme:** "fade the event" (sell-the-news / initial-spike reversal) recurs in BOTH EV-B (listings) and EV-C (FOMC reversal) — post-publication moves revert, opposite to the naive "trade the obvious direction."
- **Binding constraints:** (1) free HISTORICAL data paywall (consensus, unlocks, on-chain) caps EV-C/D; (2) small n caps significance of the real EV-B signal; (3) the sub-minute impulse is not resolvable on free 1m klines (executability unproven).
- **Standing vs program:** EV-B listings-fade is the largest event-conditional directional edge surfaced in HDATA (net +58…+424bp/event) but LOW-frequency (~hundreds of events/2yr) + statistically modest — complements, not replaces, free-panel momentum (pooled rank_IC +0.0275@24h, EXP-3) and L2 (BTC-H180 +0.054, EXP-1). Path to deploy = forward accrual (EV-E: grows n + macro consensus via FF + tick executability), not more paywalled backtest cells (EV-D on-chain expected to hit the same wall).

---

## 12. Methodology facts
- rank_IC = AUC−0.5; AUC computed via Mann-Whitney (numpy).
- Splits: honest temporal 70/30 + embargo (64 dp / 24h); per-symbol train-fit z-score on bars feats; cross-sectional feats not z-scored.
- Objective: R1 = logloss, sample_weight = |fwd move| clipped p99.
- EXP-1 labels = first-passage ±0.13% (frozen hr1_screen/ha5_screen logic); EXP-2/3/4 = sign(fwd logret @H); EXP-5 = fwd mid logret; EXP-6/7 = fwd logret.
- pandas 2.x dt-resolution fix: all merge keys cast `datetime64[ns,UTC]` (ms vs ns merge_asof error).
- Compute: GCP n2-standard-8 europe-west1, startup-script + freecode-via-metadata, results→GCS, all VMs deleted. BQ free-tier usage this session ≈ 420 GB of 1000 GB/mo.

## 13. Code (C:\Dev\sb-data-poc\)
pulls: pull_full.py, pull_orthogonal.py, pull_news.py, pull_gkg.py, pull_gkg_events.py
experiments: hdata_freepanel.py, hdata_freepanel2.py, hdata_freepanel3.py, hdata_liqcascade.py, news_backtest.py, news_events_backtest.py, ev_onboarding.py (EV-A), ev_b_listings.py + ev_b_probe.py + ev_b_control.py + ev_b_harden.py (EV-B), ev_c_macro.py (EV-C)
result jsons (authoritative on GCS): research_runs/{hdata_pool_poc,hdata_freepanel,hdata_freepanel2,hdata_freepanel3,hdata_liqcascade}/results.json — note hdata_freepanel3/results.json = lag-fixed (EXP-4); EXP-6/7 (news) + EV-A are local-only (not on GCS).
local json copies: free_results.json (EXP-2), free2.json (EXP-3), free3.json (EXP-4 leaky), free3fix.json (EXP-4 lag-fixed), liq.json (EXP-5), ev_onboarding.json (EV-A), ev_b_listings.json + ev_b_control.json + ev_b_harden.json (EV-B), ev_c_macro.json (EV-C); EV-B announcements archive on GCS at free_v1/orthogonal/events/listings/announcements.parquet.
Version control: EV-A..C scripts + this results record committed to the git repo at C:\Dev\sb-VVEBA (branch claude/hdata-rev1): research/ev/*.py + research/HDATA_RESULTS.md. Prior EXP-1..7 code stays local scratch (sb-data-poc, not git); its results are authoritative on GCS + in this record. sb-data-poc itself is NOT a git repo.
event-driven plan (design doc): C:\Dev\sb-VVEBA\research\HDATA_EVENTS_PLAN.md (branch claude/hdata-rev1, commit 4ec8bc7).
