from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

from src.order_book import OrderBook, Snapshot, BOOK_DEPTH

logger = logging.getLogger(__name__)

NUM_FEATURES = 34
NORM_WINDOW = 300   # 30 sec at 100ms
EMA_SPAN = 5
TRADE_WINDOW_SEC = 5
CVD_WINDOW_SEC = 30
VWAP_WINDOW_SEC = 60
SPOOF_LIFETIME_SEC = 2.5
SPOOF_SIZE_THRESHOLD = 1.0  # 1 BTC (was 100 ETH)
HURST_WINDOW = 3000  # ~5 min of 100ms ticks
QUEUE_DECAY_ALPHA = 0.1  # EMA smoothing for queue-pressure feature (Lever 5a)

# Cross-exchange momentum: count exchanges (Bybit + 3 cross-exchange)
# whose net signed volume in the last CROSS_EX_WINDOW_MS is positive (net-buy).
CROSS_EXCHANGES = ("bybit", "okx", "bitget", "gateio")
CROSS_EX_WINDOW_MS = 500

# Feature vector keys (ordered, used for normalization and model input)
# Now primary instrument = BTCUSDT, secondary = ETHUSDT
FEATURE_KEYS = [
    # LOB — primary BTC (0-5)
    "ofi", "imbalance_ratio", "imbalance_velocity", "spread",
    "depth_ratio_l5", "large_order",
    # Trade flow — primary BTC (6-9)
    "trade_flow_imbalance", "trade_intensity", "large_trade", "cvd",
    # Derived — primary BTC (10-13)
    "volatility_1s", "vwap_deviation", "momentum_5s", "funding_rate",
    # ETH leading signal (14-16) — ETH sometimes moves before BTC
    "eth_momentum_1s", "eth_ofi", "eth_leading_signal",
    # Liquidation clusters (17-19)
    "open_interest_delta", "long_short_ratio", "liquidation_proximity",
    # Spoofing (20)
    "spoof_score",
    # Volatility regime (21-22)
    "volatility_ratio", "trade_intensity_ratio",
    # Market regime (23)
    "hurst_exponent",
    # Sweep detection (24)
    "sweep_intensity",
    # Cancellation rate (25) — ask cancel - bid cancel, positive = bullish
    "cancel_rate_diff",
    # Multi-timeframe OFI (26-29) — divergence signals reversal
    "ofi_1s", "ofi_5s", "ofi_30s", "ofi_divergence",
    # Cross-exchange momentum (30) — net-buy count across 4 exchanges
    # (Bybit, OKX, Bitget, Gate.io) in last 500ms. Range 0..4.
    # See _calc_cross_exchange_momentum; equivalent training-time
    # computation lives in trainer._calc_features_batch.
    "cross_exchange_momentum_500ms",
    # Microstructure (31-33) — Lever 5. Realtime in this file, training-time
    # vectorised computation in trainer._calc_features_batch.
    # 31: queue_pressure — EMA of (ask L1 decay - bid L1 decay). Asks
    #     evaporating faster than bids = bullish pressure.
    # 32: top3_asymmetry — (top3_bid/top20_bid) - (top3_ask/top20_ask). High
    #     positive value = depth concentrated near best bid (front-run risk).
    # 33: effective_spread_ratio — |last_trade_price - mid| / spread, EMA.
    #     >0.5 = trades piercing the spread aggressively (urgency).
    "queue_pressure", "top3_asymmetry", "effective_spread_ratio",
]

assert len(FEATURE_KEYS) == NUM_FEATURES


class FeatureEngine:
    def __init__(self, order_book: OrderBook) -> None:
        self._ob = order_book

        # Trade accumulators (ETH)
        self._trades: deque[dict] = deque(maxlen=5000)

        # BTC state
        self._eth_trades: deque[dict] = deque(maxlen=3000)
        self._eth_bids: dict[float, float] = {}
        self._eth_asks: dict[float, float] = {}
        self._eth_mid_history: deque[tuple[float, float]] = deque(maxlen=100)  # (timestamp, mid)
        self._eth_prev_best_bid_vol: float = 0.0
        self._eth_prev_best_ask_vol: float = 0.0
        self._eth_ofi_ema: float = 0.0

        # ETH-BTC spread tracking
        self._eth_btc_ratio_history: deque[float] = deque(maxlen=NORM_WINDOW)

        # Rolling stats for z-score normalization
        self._feature_history: deque[np.ndarray] = deque(maxlen=NORM_WINDOW)

        # OFI EMA state
        self._ofi_ema: float = 0.0
        self._ofi_alpha: float = 2.0 / (EMA_SPAN + 1)

        # Previous snapshot for OFI delta
        self._prev_snap: Snapshot | None = None

        # Mark price / funding
        self.funding_rate: float = 0.0
        self.mark_price: float = 0.0

        # Volatility tracking
        self._vol_history: deque[float] = deque(maxlen=NORM_WINDOW)
        self._intensity_history: deque[float] = deque(maxlen=NORM_WINDOW)

        # Spoofing detection: track large orders
        # {price_level: (first_seen_time, volume)}
        self._large_order_tracker: dict[float, tuple[float, float]] = {}

        # Hurst exponent: rolling mid-price history
        self._mid_price_history: deque[float] = deque(maxlen=HURST_WINDOW)

        # Sweep detection: previous best bid/ask levels count
        self._prev_bid_levels: int = 0
        self._prev_ask_levels: int = 0
        self._sweep_events: deque[tuple[float, int]] = deque(maxlen=100)  # (timestamp, levels_swept)

        # Cancellation rate tracking (item 8)
        self._prev_bid_depth: dict[float, float] = {}  # {price: vol} from previous tick
        self._prev_ask_depth: dict[float, float] = {}
        self._bid_cancel_vol: deque[tuple[float, float]] = deque(maxlen=100)  # (ts, vol_cancelled)
        self._ask_cancel_vol: deque[tuple[float, float]] = deque(maxlen=100)

        # Multi-timeframe OFI (item 9) — OFI at 1s, 5s, 30s windows
        self._ofi_raw_history: deque[tuple[float, float]] = deque(maxlen=3000)  # (ts, raw_ofi)

        # Bybit cross-exchange signal (item 7)
        self._bybit_trades: deque[dict] = deque(maxlen=3000)
        self.bybit_momentum: float = 0.0  # last 1s price change on Bybit

        # Cross-exchange momentum (feature 30): per-exchange ring of recent
        # signed-volume samples. Tuple = (timestamp_ms, signed_qty) where
        # signed_qty > 0 = buyer-initiated, < 0 = seller-initiated.
        self._cross_exchange_trades: dict[str, deque[tuple[int, float]]] = {
            ex: deque(maxlen=2000) for ex in CROSS_EXCHANGES
        }

        # Microstructure features (Lever 5).
        # 31 — queue pressure: EMA of L1 decay rates. Track previous best
        # bid/ask volumes and accumulate the *positive* drop (cancel/fill)
        # from one tick to the next. Asks decaying faster than bids → bullish.
        self._prev_best_bid_vol: float = 0.0
        self._prev_best_ask_vol: float = 0.0
        self._bid_decay_ema: float = 0.0
        self._ask_decay_ema: float = 0.0
        # 33 — effective-spread ratio: rolling EMA of |last_price - mid| / spread.
        self._eff_spread_ema: float = 0.0

        # Derivatives data (set by ws_client polling)
        self._ws = None  # set via set_ws_client()

        # Current feature vector
        self.features: np.ndarray = np.zeros(NUM_FEATURES, dtype=np.float64)
        self.features_raw: dict[str, float] = {}
        self.last_update: float = 0.0

    def set_ws_client(self, ws: Any) -> None:
        """Set reference to ws_client for OI/LS data access."""
        self._ws = ws

    # --- Data ingestion callbacks ---

    def on_aggtrade(self, data: dict) -> None:
        self._trades.append({
            "T": data.get("T", int(time.time() * 1000)),
            "p": float(data.get("p", 0)),
            "q": float(data.get("q", 0)),
            "m": data.get("m", False),
        })

    def on_markprice(self, data: dict) -> None:
        self.funding_rate = float(data.get("r", 0))
        self.mark_price = float(data.get("p", 0))

    def on_bybit_aggtrade(self, data: dict) -> None:
        """Bybit BTCUSDT trade — leads Binance by 100-500ms.

        Feeds two structures:
          1. `_bybit_trades` — used by `bybit_momentum` (price-change signal)
          2. `_cross_exchange_trades["bybit"]` — used by feature 30
             (cross-exchange net-buy momentum)
        """
        ts = int(data.get("T", time.time() * 1000))
        # Defensive: some cross-exchange WS paths send a signed qty that
        # double-codes the side (Gate.io is the known case — it ships the
        # raw Gate.io `size` which is negative for sells while `is_seller`
        # is also populated). `abs()` here means the side lives exclusively
        # in the `is_seller` flag, preserving the invariant that feature 30
        # sign matches the actual net-buy direction.
        qty = abs(float(data.get("q", 0)))
        is_seller = bool(data.get("m", False))
        self._bybit_trades.append({
            "T": ts,
            "p": float(data.get("p", 0)),
            "q": qty,
            "m": is_seller,
        })
        signed = -qty if is_seller else qty
        self._cross_exchange_trades["bybit"].append((ts, signed))

    def on_exchange_trade(self, data: dict) -> None:
        """Cross-exchange trade from OKX/Bitget/Gate.io for feature 30.

        Data shape (from ws_client._dispatch_exchange):
            {"T": ts_ms, "p": price, "q": qty, "m": is_sell, "exchange": name}
        """
        ex = data.get("exchange", "")
        buf = self._cross_exchange_trades.get(ex)
        if buf is None:
            return
        ts = int(data.get("T", time.time() * 1000))
        # See on_bybit_aggtrade: take abs() so the signed-qty invariant
        # holds uniformly across all 4 cross-exchange feeds, shielding us
        # from the Gate.io signed-size quirk and any future exchange with
        # similar semantics.
        qty = abs(float(data.get("q", 0)))
        is_seller = bool(data.get("m", False))
        signed = -qty if is_seller else qty
        buf.append((ts, signed))

    def on_secondary_aggtrade(self, data: dict) -> None:
        ts = data.get("T", int(time.time() * 1000))
        price = float(data.get("p", 0))
        self._eth_trades.append({
            "T": ts,
            "p": price,
            "q": float(data.get("q", 0)),
            "m": data.get("m", False),
        })
        self._eth_mid_history.append((ts / 1000.0, price))

    def on_secondary_depth(self, data: dict) -> None:
        for price_s, qty_s in data.get("b", []):
            p, q = float(price_s), float(qty_s)
            if q == 0:
                self._eth_bids.pop(p, None)
            else:
                self._eth_bids[p] = q
        for price_s, qty_s in data.get("a", []):
            p, q = float(price_s), float(qty_s)
            if q == 0:
                self._eth_asks.pop(p, None)
            else:
                self._eth_asks[p] = q

    # --- Main update ---

    def update(self) -> np.ndarray:
        snap = self._ob.current
        if snap is None or not self._ob.is_synced:
            return self.features

        now_ms = int(time.time() * 1000)
        now_s = time.monotonic()
        raw = {}

        # === LOB features ===
        raw["ofi"] = self._calc_ofi(snap)
        raw["imbalance_ratio"] = self._calc_imbalance_ratio(snap)
        raw["imbalance_velocity"] = self._calc_imbalance_velocity()
        raw["spread"] = snap.spread
        raw["depth_ratio_l5"] = self._calc_depth_ratio(snap)
        raw["large_order"] = self._calc_large_order(snap)

        # === Trade flow features ===
        raw["trade_flow_imbalance"] = self._calc_trade_flow_imbalance(now_ms)
        raw["trade_intensity"] = self._calc_trade_intensity(now_ms)
        raw["large_trade"] = self._calc_large_trade()
        raw["cvd"] = self._calc_cvd(now_ms)

        # === Derived features ===
        raw["volatility_1s"] = self._calc_volatility()
        raw["vwap_deviation"] = self._calc_vwap_deviation(snap, now_ms)
        raw["momentum_5s"] = self._calc_momentum()
        # === Features 13-19: now computed in both training and runtime ===
        raw["funding_rate"] = self.funding_rate
        raw["eth_momentum_1s"] = self._calc_eth_momentum()
        raw["eth_ofi"] = self._calc_eth_ofi()
        raw["eth_leading_signal"] = self._calc_eth_leading_signal(snap)
        raw["open_interest_delta"] = self._calc_oi_delta()
        raw["long_short_ratio"] = self._get_long_short_ratio()
        raw["liquidation_proximity"] = self._calc_liquidation_proximity(snap)

        # === Spoofing detection ===
        raw["spoof_score"] = self._calc_spoof_score(snap, now_s)

        # === Market regime ===
        self._mid_price_history.append(snap.mid_price)
        raw["hurst_exponent"] = self._calc_hurst()

        # === Sweep detection ===
        raw["sweep_intensity"] = self._calc_sweep(snap, now_s)

        # === Cancellation rate (item 8) ===
        raw["cancel_rate_diff"] = self._calc_cancel_rate(snap, now_s)

        # === Multi-timeframe OFI (item 9) ===
        raw_ofi = raw["ofi"]  # already computed above (100ms OFI)
        self._ofi_raw_history.append((now_s, raw_ofi))
        raw["ofi_1s"] = self._calc_ofi_window(now_s, 1.0)
        raw["ofi_5s"] = self._calc_ofi_window(now_s, 5.0)
        raw["ofi_30s"] = self._calc_ofi_window(now_s, 30.0)
        # Divergence: sign mismatch between short and long OFI
        ofi_short = raw["ofi_1s"]
        ofi_long = raw["ofi_30s"]
        raw["ofi_divergence"] = ofi_short - ofi_long if (ofi_short * ofi_long < 0) else 0.0

        # === Cross-exchange momentum (feature 30) ===
        raw["cross_exchange_momentum_500ms"] = self._calc_cross_exchange_momentum(now_ms)

        # === Microstructure features (Lever 5) ===
        raw["queue_pressure"] = self._calc_queue_pressure(snap)
        raw["top3_asymmetry"] = self._calc_top3_asymmetry(snap)
        raw["effective_spread_ratio"] = self._calc_effective_spread_ratio(snap)

        # === Volatility regime ===
        self._vol_history.append(raw["volatility_1s"])
        self._intensity_history.append(raw["trade_intensity"])

        if len(self._vol_history) >= 30:
            vol_mean = float(np.mean(list(self._vol_history)))
            vol_std = float(np.std(list(self._vol_history)))
            raw["volatility_3sigma"] = vol_mean + 3 * vol_std
            raw["volatility_ratio"] = raw["volatility_1s"] / (vol_mean + 1e-10)
        else:
            raw["volatility_3sigma"] = raw["volatility_1s"] * 3
            raw["volatility_ratio"] = 1.0

        if len(self._intensity_history) >= 30:
            int_mean = float(np.mean(list(self._intensity_history)))
            raw["trade_intensity_ratio"] = raw["trade_intensity"] / (int_mean + 1e-10)
        else:
            raw["trade_intensity_ratio"] = 1.0

        self.features_raw = raw
        self._prev_snap = snap

        # Build normalized feature vector
        vec = np.array([raw.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float64)

        # Z-score normalization
        self._feature_history.append(vec.copy())
        if len(self._feature_history) >= 10:
            hist = np.array(list(self._feature_history))
            mean = hist.mean(axis=0)
            std = hist.std(axis=0) + 1e-8
            vec = (vec - mean) / std

        self.features = vec
        self.last_update = time.monotonic()
        return vec

    # === LOB features ===

    def _calc_ofi(self, snap: Snapshot) -> float:
        if self._prev_snap is None:
            return 0.0
        prev = self._prev_snap
        delta_bid = float(snap.bids[0, 1]) - float(prev.bids[0, 1])
        if snap.bids[0, 0] != prev.bids[0, 0]:
            delta_bid = float(snap.bids[0, 1])
        delta_ask = float(snap.asks[0, 1]) - float(prev.asks[0, 1])
        if snap.asks[0, 0] != prev.asks[0, 0]:
            delta_ask = float(snap.asks[0, 1])
        ofi = delta_bid - delta_ask
        self._ofi_ema = self._ofi_alpha * ofi + (1 - self._ofi_alpha) * self._ofi_ema
        return self._ofi_ema

    def _calc_imbalance_ratio(self, snap: Snapshot) -> float:
        bid_vol = float(snap.bids[:5, 1].sum())
        ask_vol = float(snap.asks[:5, 1].sum())
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def _calc_imbalance_velocity(self) -> float:
        ring = self._ob.ring
        if len(ring) < 6:
            return 0.0

        def _imb(s: Snapshot) -> float:
            bv = float(s.bids[:5, 1].sum())
            av = float(s.asks[:5, 1].sum())
            t = bv + av
            return (bv - av) / t if t > 0 else 0.0

        return _imb(ring[-1]) - _imb(ring[-6])

    def _calc_depth_ratio(self, snap: Snapshot) -> float:
        bid_vol = float(snap.bids[:5, 1].sum())
        ask_vol = float(snap.asks[:5, 1].sum())
        return bid_vol / ask_vol if ask_vol > 0 else 10.0

    def _calc_large_order(self, snap: Snapshot) -> float:
        if np.any(snap.bids[:5, 1] > SPOOF_SIZE_THRESHOLD) or \
           np.any(snap.asks[:5, 1] > SPOOF_SIZE_THRESHOLD):
            return 1.0
        return 0.0

    # === Trade flow ===

    def _calc_trade_flow_imbalance(self, now_ms: int) -> float:
        cutoff = now_ms - TRADE_WINDOW_SEC * 1000
        buy_vol = sell_vol = 0.0
        for t in reversed(self._trades):
            if t["T"] < cutoff:
                break
            if t["m"]:
                sell_vol += t["q"]
            else:
                buy_vol += t["q"]
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0.0

    def _calc_trade_intensity(self, now_ms: int) -> float:
        cutoff = now_ms - 1000
        return float(sum(1 for t in reversed(self._trades) if t["T"] >= cutoff))

    def _calc_large_trade(self) -> float:
        if not self._trades:
            return 0.0
        return 1.0 if self._trades[-1]["q"] > 10.0 else 0.0

    def _calc_cvd(self, now_ms: int) -> float:
        cutoff = now_ms - CVD_WINDOW_SEC * 1000
        cvd = 0.0
        for t in reversed(self._trades):
            if t["T"] < cutoff:
                break
            cvd += -t["q"] if t["m"] else t["q"]
        return cvd

    # === Derived ===

    def _calc_volatility(self) -> float:
        ring = self._ob.ring
        if len(ring) < 11:
            return 0.0
        prices = [s.mid_price for s in list(ring)[-11:]]
        returns = np.diff(prices) / np.array(prices[:-1])
        return float(np.std(returns))

    def _calc_vwap_deviation(self, snap: Snapshot, now_ms: int) -> float:
        cutoff = now_ms - VWAP_WINDOW_SEC * 1000
        pv_sum = v_sum = 0.0
        for t in reversed(self._trades):
            if t["T"] < cutoff:
                break
            pv_sum += t["p"] * t["q"]
            v_sum += t["q"]
        if v_sum == 0:
            return 0.0
        vwap = pv_sum / v_sum
        return (snap.mid_price - vwap) / vwap

    def _calc_momentum(self) -> float:
        ring = self._ob.ring
        n = min(50, len(ring))
        if n < 2:
            return 0.0
        curr = ring[-1].mid_price
        prev = ring[-n].mid_price
        return (curr - prev) / prev if prev > 0 else 0.0

    # === ETH leading signal (secondary instrument) ===

    def _calc_eth_momentum(self) -> float:
        """ETH price change over last 1 second (may lead BTC)."""
        if len(self._eth_mid_history) < 2:
            return 0.0
        now = self._eth_mid_history[-1]
        cutoff = now[0] - 1.0
        oldest = now
        for ts, price in reversed(self._eth_mid_history):
            if ts < cutoff:
                oldest = (ts, price)
                break
        if oldest[1] == 0 or oldest[0] == now[0]:
            return 0.0
        return (now[1] - oldest[1]) / oldest[1]

    def _calc_eth_ofi(self) -> float:
        """ETH order flow imbalance from depth updates."""
        if not self._eth_bids:
            return 0.0
        best_bid_price = max(self._eth_bids.keys()) if self._eth_bids else 0
        best_bid_vol = self._eth_bids.get(best_bid_price, 0)
        best_ask_price = min(self._eth_asks.keys()) if self._eth_asks else 0
        best_ask_vol = self._eth_asks.get(best_ask_price, 0)

        delta_bid = best_bid_vol - self._eth_prev_best_bid_vol
        delta_ask = best_ask_vol - self._eth_prev_best_ask_vol
        self._eth_prev_best_bid_vol = best_bid_vol
        self._eth_prev_best_ask_vol = best_ask_vol

        ofi = delta_bid - delta_ask
        self._eth_ofi_ema = self._ofi_alpha * ofi + (1 - self._ofi_alpha) * self._eth_ofi_ema
        return self._eth_ofi_ema

    def _calc_eth_leading_signal(self, snap: Snapshot) -> float:
        """ETH/BTC ratio deviation from rolling average.

        If ETH moved but BTC hasn't yet → leading signal for BTC direction.
        """
        if not self._eth_mid_history:
            return 0.0
        btc_price = self._eth_mid_history[-1][1]
        if btc_price == 0:
            return 0.0
        eth_btc = snap.mid_price / btc_price

        self._eth_btc_ratio_history.append(eth_btc)
        if len(self._eth_btc_ratio_history) < 30:
            return 0.0

        mean = float(np.mean(list(self._eth_btc_ratio_history)))
        return (eth_btc - mean) / mean if mean > 0 else 0.0

    # === Liquidation clusters ===

    def _calc_oi_delta(self) -> float:
        """Open interest change (current vs previous poll, ~15s apart)."""
        if self._ws is None:
            return 0.0
        if self._ws.open_interest_prev == 0:
            return 0.0
        return (self._ws.open_interest - self._ws.open_interest_prev) / self._ws.open_interest_prev

    def _get_long_short_ratio(self) -> float:
        if self._ws is None:
            return 1.0
        return self._ws.long_short_ratio

    def _calc_liquidation_proximity(self, snap: Snapshot) -> float:
        """Estimate proximity to liquidation cluster.

        Heuristic: with x50-x100 leverage, liquidation happens at ~1-2% from entry.
        If L/S ratio is skewed (lots of longs), liquidation cluster is ~1-2% BELOW price.
        If lots of shorts, cluster is ~1-2% ABOVE.

        Returns signed value: positive = cluster above (shorts at risk),
        negative = cluster below (longs at risk).
        """
        if self._ws is None:
            return 0.0
        ls = self._ws.long_short_ratio
        mid = snap.mid_price
        if mid == 0 or ls == 0:
            return 0.0

        # Estimate cluster distance (1.5% for x50-x100 leverage average)
        cluster_pct = 0.015

        if ls > 1.2:
            # More longs → cluster below (longs get liquidated on dip)
            cluster_price = mid * (1 - cluster_pct)
            return -(mid - cluster_price) / mid  # negative: danger below
        elif ls < 0.8:
            # More shorts → cluster above (shorts get liquidated on pump)
            cluster_price = mid * (1 + cluster_pct)
            return (cluster_price - mid) / mid  # positive: danger above
        else:
            return 0.0  # balanced, no dominant cluster

    # === Spoofing detection ===

    def _calc_spoof_score(self, snap: Snapshot, now_s: float) -> float:
        """Detect spoofed orders: large orders that persist without price reaction.

        Real large orders cause price to move toward them.
        Spoof orders sit at a level, create false imbalance, then disappear.
        """
        current_large = set()

        # Track large orders on bid side (top 5 levels)
        for i in range(min(5, len(snap.bids))):
            price = float(snap.bids[i, 0])
            vol = float(snap.bids[i, 1])
            if vol > SPOOF_SIZE_THRESHOLD:
                current_large.add(price)
                if price not in self._large_order_tracker:
                    self._large_order_tracker[price] = (now_s, vol)

        # Track large orders on ask side
        for i in range(min(5, len(snap.asks))):
            price = float(snap.asks[i, 0])
            vol = float(snap.asks[i, 1])
            if vol > SPOOF_SIZE_THRESHOLD:
                current_large.add(price)
                if price not in self._large_order_tracker:
                    self._large_order_tracker[price] = (now_s, vol)

        # Remove orders that disappeared
        gone = [p for p in self._large_order_tracker if p not in current_large]
        for p in gone:
            del self._large_order_tracker[p]

        if not self._large_order_tracker:
            return 0.0

        # Score: how many large orders have persisted > SPOOF_LIFETIME without price moving to them
        spoof_count = 0
        real_count = 0
        mid = snap.mid_price

        for price, (first_seen, vol) in self._large_order_tracker.items():
            age = now_s - first_seen
            if age < SPOOF_LIFETIME_SEC:
                continue  # too early to judge

            # Did price move toward this order?
            distance_pct = abs(price - mid) / mid
            if distance_pct < 0.0001:
                # Price reached the order — real
                real_count += 1
            else:
                # Price didn't move toward it — likely spoof
                spoof_count += 1

        total = spoof_count + real_count
        if total == 0:
            return 0.0
        return spoof_count / total  # 0 = all real, 1 = all spoof

    # === Market regime: Hurst exponent ===

    def _calc_hurst(self) -> float:
        """Simplified Hurst exponent via rescaled range (R/S) method.

        H > 0.55: trending (momentum works)
        H < 0.45: mean-reverting (reversal works)
        0.45-0.55: random walk
        """
        prices = list(self._mid_price_history)
        n = len(prices)
        if n < 100:
            return 0.5  # default: random walk

        # Use last 600-3000 points (1-5 minutes)
        series = np.array(prices[-min(n, HURST_WINDOW):])
        returns = np.diff(np.log(series + 1e-10))

        if len(returns) < 20:
            return 0.5

        # R/S analysis over multiple sub-intervals
        rs_list = []
        for chunk_size in [20, 50, 100, 200]:
            if chunk_size > len(returns):
                break
            n_chunks = len(returns) // chunk_size
            for i in range(n_chunks):
                chunk = returns[i * chunk_size:(i + 1) * chunk_size]
                mean = chunk.mean()
                deviate = np.cumsum(chunk - mean)
                r = deviate.max() - deviate.min()
                s = chunk.std()
                if s > 0:
                    rs_list.append((chunk_size, r / s))

        if len(rs_list) < 4:
            return 0.5

        # Log-log regression: log(R/S) = H * log(n) + c
        sizes = np.array([np.log(x[0]) for x in rs_list])
        rs_vals = np.array([np.log(x[1]) for x in rs_list])

        # Simple linear regression
        n_pts = len(sizes)
        sx = sizes.sum()
        sy = rs_vals.sum()
        sxx = (sizes * sizes).sum()
        sxy = (sizes * rs_vals).sum()
        denom = n_pts * sxx - sx * sx
        if denom == 0:
            return 0.5

        h = (n_pts * sxy - sx * sy) / denom
        return float(np.clip(h, 0.0, 1.0))

    # === Sweep detection ===

    def _calc_sweep(self, snap: Snapshot, now_s: float) -> float:
        """Detect sweeps: market orders that consume 3+ levels in one tick.

        Tracks how many bid/ask levels were removed since last snapshot.
        """
        if self._prev_snap is None:
            self._prev_bid_levels = int((snap.bids[:, 1] > 0).sum())
            self._prev_ask_levels = int((snap.asks[:, 1] > 0).sum())
            return 0.0

        prev = self._prev_snap
        # Count non-zero levels
        curr_bid_levels = int((snap.bids[:, 1] > 0).sum())
        curr_ask_levels = int((snap.asks[:, 1] > 0).sum())

        # Levels disappeared = swept by market orders
        bid_swept = max(0, self._prev_bid_levels - curr_bid_levels)
        ask_swept = max(0, self._prev_ask_levels - curr_ask_levels)

        # Also detect price jumping multiple ticks
        # BTC tick size = $0.10
        tick_size = 0.10
        bid_price_jump = abs(snap.best_bid - prev.best_bid) / tick_size if prev.best_bid > 0 else 0
        ask_price_jump = abs(snap.best_ask - prev.best_ask) / tick_size if prev.best_ask > 0 else 0

        levels_swept = max(bid_swept, ask_swept, int(bid_price_jump) - 1, int(ask_price_jump) - 1)
        levels_swept = max(0, levels_swept)

        self._prev_bid_levels = curr_bid_levels
        self._prev_ask_levels = curr_ask_levels

        if levels_swept >= 3:
            self._sweep_events.append((now_s, levels_swept))

        # Sweep intensity: total levels swept in last 1 second
        cutoff = now_s - 1.0
        intensity = sum(lvl for ts, lvl in self._sweep_events if ts >= cutoff)
        return float(intensity)

    # === Cancellation rate (item 8) ===

    def _calc_cancel_rate(self, snap: Snapshot, now_s: float) -> float:
        """Cancel rate = volume drop without corresponding trade at that level.

        Sliding 1-second window, separate for bid and ask.
        Feature = ask_cancel_rate - bid_cancel_rate.
        Positive = asks cancelled more = sellers retreating = bullish.
        """
        # Build current depth map from snapshot
        curr_bids = {float(snap.bids[i, 0]): float(snap.bids[i, 1])
                     for i in range(min(5, len(snap.bids))) if snap.bids[i, 1] > 0}
        curr_asks = {float(snap.asks[i, 0]): float(snap.asks[i, 1])
                     for i in range(min(5, len(snap.asks))) if snap.asks[i, 1] > 0}

        # Compare with previous tick — volume drops without trade = cancellation
        bid_cancel = 0.0
        for price, prev_vol in self._prev_bid_depth.items():
            curr_vol = curr_bids.get(price, 0.0)
            if curr_vol < prev_vol:
                bid_cancel += prev_vol - curr_vol

        ask_cancel = 0.0
        for price, prev_vol in self._prev_ask_depth.items():
            curr_vol = curr_asks.get(price, 0.0)
            if curr_vol < prev_vol:
                ask_cancel += prev_vol - curr_vol

        self._prev_bid_depth = curr_bids
        self._prev_ask_depth = curr_asks

        # Accumulate in 1-second window
        self._bid_cancel_vol.append((now_s, bid_cancel))
        self._ask_cancel_vol.append((now_s, ask_cancel))

        cutoff = now_s - 1.0
        bid_rate = sum(v for ts, v in self._bid_cancel_vol if ts >= cutoff)
        ask_rate = sum(v for ts, v in self._ask_cancel_vol if ts >= cutoff)

        return ask_rate - bid_rate  # positive = bullish

    # === Multi-timeframe OFI (item 9) ===

    def _calc_ofi_window(self, now_s: float, window_sec: float) -> float:
        """Sum of raw OFI values over a time window."""
        cutoff = now_s - window_sec
        total = 0.0
        for ts, ofi_val in reversed(self._ofi_raw_history):
            if ts < cutoff:
                break
            total += ofi_val
        return total

    def _calc_cross_exchange_momentum(self, now_ms: int) -> float:
        """Count of cross-exchange feeds whose net signed volume in the last
        CROSS_EX_WINDOW_MS is strictly positive (more buys than sells).

        Range: 0 (all 4 exchanges net-selling/flat) to 4 (all net-buying).
        Drops stale entries from each exchange's deque as a side effect.
        """
        cutoff = now_ms - CROSS_EX_WINDOW_MS
        net_buy_count = 0
        for buf in self._cross_exchange_trades.values():
            # Drop entries older than the window
            while buf and buf[0][0] < cutoff:
                buf.popleft()
            net = 0.0
            for _, signed_qty in buf:
                net += signed_qty
            if net > 0:
                net_buy_count += 1
        return float(net_buy_count)

    # === Microstructure (Lever 5) ===

    def _calc_queue_pressure(self, snap: Snapshot) -> float:
        """EMA of (ask L1 decay rate - bid L1 decay rate).

        Decay = the *positive* tick-to-tick drop in best-level volume; that is
        cancellations or fills. Asks evaporating faster than bids = sellers
        retreating or buyers consuming = bullish pressure.
        """
        best_bid_vol = float(snap.bids[0, 1]) if len(snap.bids) > 0 else 0.0
        best_ask_vol = float(snap.asks[0, 1]) if len(snap.asks) > 0 else 0.0
        bid_decay = max(0.0, self._prev_best_bid_vol - best_bid_vol)
        ask_decay = max(0.0, self._prev_best_ask_vol - best_ask_vol)
        a = QUEUE_DECAY_ALPHA
        self._bid_decay_ema = a * bid_decay + (1 - a) * self._bid_decay_ema
        self._ask_decay_ema = a * ask_decay + (1 - a) * self._ask_decay_ema
        self._prev_best_bid_vol = best_bid_vol
        self._prev_best_ask_vol = best_ask_vol
        return self._ask_decay_ema - self._bid_decay_ema

    @staticmethod
    def _calc_top3_asymmetry(snap: Snapshot) -> float:
        """(top3_bid / top20_bid) - (top3_ask / top20_ask).

        Measures concentration of depth at the best 3 levels relative to the
        full visible book. Large positive = depth piled near best bid (front-
        run / iceberg risk on the bid side); negative = ask side concentrated.
        """
        top3_bid = float(snap.bids[:3, 1].sum())
        top20_bid = float(snap.bids[:, 1].sum())
        top3_ask = float(snap.asks[:3, 1].sum())
        top20_ask = float(snap.asks[:, 1].sum())
        bid_share = top3_bid / (top20_bid + 1e-9)
        ask_share = top3_ask / (top20_ask + 1e-9)
        return bid_share - ask_share

    def _calc_effective_spread_ratio(self, snap: Snapshot) -> float:
        """EMA of |last_trade_price - mid| / spread.

        Captures how aggressively recent prints pierce the spread. Stable
        market maker flow keeps the ratio near 0 (trades hit the inside);
        urgency / news prints push it >0.5.
        """
        if not self._trades:
            return self._eff_spread_ema
        last_price = float(self._trades[-1]["p"])
        mid = snap.mid_price
        spread = max(snap.spread, 1e-9)
        ratio = abs(last_price - mid) / spread
        a = QUEUE_DECAY_ALPHA
        self._eff_spread_ema = a * ratio + (1 - a) * self._eff_spread_ema
        return self._eff_spread_ema

    # === LOB tensor for CNN ===

    def build_lob_tensor(self) -> np.ndarray | None:
        ring = self._ob.ring
        if len(ring) < 50:
            return None

        snapshots = list(ring)[-50:]
        tensor = np.zeros((3, BOOK_DEPTH, 50), dtype=np.float32)
        for t, snap in enumerate(snapshots):
            tensor[0, :, t] = snap.bids[:, 1]
            tensor[1, :, t] = snap.asks[:, 1]

        for t, snap in enumerate(snapshots):
            ts = snap.timestamp
            ts_next = snapshots[t + 1].timestamp if t < 49 else ts + 100
            buy_vol = sell_vol = 0.0
            for trade in self._trades:
                if ts <= trade["T"] < ts_next:
                    if trade["m"]:
                        sell_vol += trade["q"]
                    else:
                        buy_vol += trade["q"]
            tensor[2, 0, t] = buy_vol
            tensor[2, 1, t] = sell_vol

        return tensor
