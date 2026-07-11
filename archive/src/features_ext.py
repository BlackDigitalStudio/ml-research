"""Extended feature set — horizon-tier (60-180 s) additions, Stage A + B + C + D.

Sidecar module kept separate from `src/features.py` until Stage E, so the
main feature set and its train-time batch twin share one source of truth
for stream↔batch parity.

Feature order (must mirror Rust `fill_horizon_features{,_b,_c}`):

    Stage A (2026-04-15):
      0: momentum_30s           — (mid - mid_30s_ago) / mid_30s_ago
      1: momentum_60s           — (mid - mid_60s_ago) / mid_60s_ago
      2: momentum_120s          — (mid - mid_120s_ago) / mid_120s_ago
      3: realized_vol_60s       — sqrt(Σ squared 100 ms log-returns, 60 s)
      4: realized_vol_120s      — sqrt(Σ squared 100 ms log-returns, 120 s)
      5: bipower_var_120s       — (π/2) · Σ|r_i|·|r_{i-1}| over 120 s (jump-robust)

    Stage B (2026-04-15):
      6: ofi_60s                — Σ raw OFI over last 600 ticks (60 s)
      7: ofi_120s               — Σ raw OFI over last 1200 ticks (120 s)
      8: trade_flow_imbalance_60s
                                — (Σ buy_qty − Σ sell_qty) / Σ |qty| over last 60 s
                                  of Binance BTCUSDT trades
      9: funding_time_to_next_min
                                — minutes until next 00/08/16 UTC funding boundary
     10: funding_basis_bps      — (mark − mid) / mid × 10_000
                                  (Binance perp premium proxy; we lack the real
                                  index feed, so mid from depth acts as spot)

    Stage C (2026-04-15):
     11: microprice_deviation   — (microprice − mid) / max(spread, eps)
                                  where microprice = (aq0·bp0 + bq0·ap0) / (bq0+aq0)
                                  (Stoikov opposite-side weighted; sign = pressure dir)
     12: ofi_top5_weighted      — Σ_{k=0..4} w_k·(Δbid_qty_k − Δask_qty_k) summed
                                  over last 30 ticks (3 s); weights w_k = 1/(k+1)
     13: kyle_lambda_60s        — rolling OLS slope β of Δmid on signed-volume over
                                  last 600 ticks (60 s): β = Σxy / (Σxx + eps)
                                  where x_t = signed_vol in (ts_{t-1}, ts_t],
                                        y_t = mid_t − mid_{t-1}
                                  Units: price units per BTC of signed flow.
     14: vpin_60s               — time-bucketed Volume-Synchronized PIN over 6
                                  consecutive 10 s sub-buckets: Σ|net_k| / Σ total_k.
                                  Bounded in [0, 1]; 0 = balanced flow, 1 = all one side.
     15: cancel_to_trade_ratio_30s
                                — (Σ bid_cancel + ask_cancel over last 300 ticks) /
                                  (Σ |trade_qty| over last 30 s of trades + eps).
                                  cancel_tick = Σ_{k=0..4} max(0, qty_prev_k − qty_k).

    Stage D (2026-04-15):
     16: bybit_lead_lag_corr_30s   — Pearson corr of BTC 100 ms returns vs Bybit
                                     100 ms returns lagged by 1 tick, over 300
                                     ticks (30 s). Bybit price = last trade price.
     17: okx_net_flow_30s          — Σ signed_qty from OKX trades over last 30 s.
     18: bitget_net_flow_30s       — Σ signed_qty from Bitget trades over last 30 s.
     19: gateio_net_flow_30s       — Σ signed_qty from Gate.io trades over last 30 s.
     20: eth_momentum_60s          — (eth_last[T] − eth_last[T-600]) / eth_last[T-600]
                                     where eth_last is the latest Binance ETH
                                     trade price at or before the BTC depth tick.
     21: eth_btc_corr_30s          — Pearson corr of BTC 100 ms mid-log-returns and
                                     ETH 100 ms price-log-returns over 300 ticks.
                                     ETH price at tick = latest trade price; ETH
                                     depth stream not recorded, so we fall back
                                     to trade-price returns.

All windows are in depth-tick units assuming 100 ms cadence (same
convention as existing features 10/12). `get()` returns a
(NUM_EXT_FEATURES,) float32 vector — 22 as of Stage D (A=6 + B=5 + C=5 +
D=6). Features emit 0 until their window is saturated, matching the
stage-A convention.

Semantic note on basis: Binance perp "basis" is (mark − index) / index but
the real index (weighted cross-exchange mid) is not in the recorded data.
Using the depth mid as the spot proxy is a documented shortcut — it captures
the perp-vs-spot premium the same way for any signal regressed against
prices denominated in USDT.
"""
from __future__ import annotations

from collections import deque
from math import log, sqrt

import numpy as np

NUM_EXT_FEATURES = 22

EXT_FEATURE_KEYS = [
    # Stage A
    "momentum_30s",
    "momentum_60s",
    "momentum_120s",
    "realized_vol_60s",
    "realized_vol_120s",
    "bipower_var_120s",
    # Stage B
    "ofi_60s",
    "ofi_120s",
    "trade_flow_imbalance_60s",
    "funding_time_to_next_min",
    "funding_basis_bps",
    # Stage C
    "microprice_deviation",
    "ofi_top5_weighted",
    "kyle_lambda_60s",
    "vpin_60s",
    "cancel_to_trade_ratio_30s",
    # Stage D
    "bybit_lead_lag_corr_30s",
    "okx_net_flow_30s",
    "bitget_net_flow_30s",
    "gateio_net_flow_30s",
    "eth_momentum_60s",
    "eth_btc_corr_30s",
]
assert len(EXT_FEATURE_KEYS) == NUM_EXT_FEATURES

# Window sizes in 100 ms ticks.
_W_3S = 30
_W_30S = 300
_W_60S = 600
_W_120S = 1200

# Trade-flow window is time-driven (60 s), not tick-driven.
_TFI_WINDOW_MS = 60_000
_CTR_TRADE_WINDOW_MS = 30_000

# VPIN: 6 consecutive sub-buckets × 10 s each → 60 s total.
_VPIN_NUM_BUCKETS = 6
_VPIN_BUCKET_MS = 10_000

# Binance funding schedule: every 8 h at 00:00, 08:00, 16:00 UTC.
_FUNDING_PERIOD_MS = 8 * 3600 * 1000

# π/2 scaling for bipower variation.
_BV_SCALE = np.pi / 2.0

# Per-level weights for ofi_top5_weighted. 1/(k+1) decays with depth.
_OFI5_WEIGHTS = np.array([1.0 / (k + 1) for k in range(5)], dtype=np.float64)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length vectors. Zero when either
    variance is < 1e-24; clipped to [-1, 1]."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    sx = float(x.sum())
    sy = float(y.sum())
    sxx = float((x * x).sum())
    syy = float((y * y).sum())
    sxy = float((x * y).sum())
    den = (n * sxx - sx * sx) * (n * syy - sy * sy)
    if den <= 1e-24:
        return 0.0
    return float((n * sxy - sx * sy) / np.sqrt(den))


def _minutes_to_next_funding(ts_ms: int) -> float:
    """Minutes until the next 00/08/16 UTC funding boundary.

    Returns 0 when ts_ms falls exactly on a boundary.
    """
    if ts_ms <= 0:
        return 0.0
    rem_ms = ts_ms % _FUNDING_PERIOD_MS
    if rem_ms == 0:
        return 0.0
    return (_FUNDING_PERIOD_MS - rem_ms) / 60_000.0


class FeatureExtEngine:
    """Streaming computation of the 11 horizon-tier features (Stage A + B).

    Call in order every 100 ms tick:
        eng.on_mid(ts_ms, mid, bid_qty0, ask_qty0)
    Between ticks, as trades arrive:
        eng.on_trade(ts_ms, signed_qty)        # qty > 0 for buyer-initiated
    When funding/mark updates arrive (poll-rate):
        eng.set_funding(mark_price)
    Then at any point:
        eng.get()                              # (11,) float32

    Monotonic timestamps are expected; the trade buffer evicts based on
    the most recent on_mid ts.
    """

    __slots__ = (
        # mid-price
        "_mid_buf",
        "_ret_buf",
        "_abs_ret_buf",
        "_ret_sq_sum_60",
        "_ret_sq_sum_120",
        "_bv_sum_120",
        "_last_mid",
        "_last_log_mid",
        # L1 quantities for OFI.
        "_last_bid_qty0",
        "_last_ask_qty0",
        "_pending_raw",
        "_has_pending_raw",
        "_ofi_buf_60",
        "_ofi_buf_120",
        "_ofi_sum_60",
        "_ofi_sum_120",
        # current tick timestamp
        "_last_ts_ms",
        # trade flow
        "_trade_buf",
        "_trade_signed_sum",
        "_trade_abs_sum",
        # funding
        "_mark_price",
        # Stage C — structural microstructure.
        # Microprice: computed in get() from the latest top-1 state cached below.
        "_last_bp0",
        "_last_ap0",
        # Top-5 OFI weighted + cancel tick: lagged window [T-W..T-1] means we
        # hold the current-tick raw/cancel back in pending slots and age them
        # into the sums on the NEXT on_depth_l5 call (same trick as Stage B OFI).
        "_last_bq5",
        "_last_aq5",
        "_ofi5_buf",
        "_ofi5_sum",
        "_pending_ofi5",
        "_pending_cancel",
        "_has_pending_c",
        # Kyle's lambda: maintain deques of (x=signed_vol, y=log-return) for
        # 600 ticks, lagged so the window covers ticks [T-W..T-1] (matches
        # batch cumsum + Rust). Current tick's pair is held in _pending_kyle.
        "_kyle_buf",
        "_kyle_xy_sum",
        "_kyle_xx_sum",
        "_pending_kyle_x",
        "_pending_kyle_y",
        "_has_pending_kyle",
        "_signed_vol_current_tick",
        # VPIN: computed on-the-fly in get() from _trade_buf (60 s window)
        # by partitioning trades into 6 sliding sub-buckets ending at now.
        # That matches the searchsorted-based batch path bit-for-bit; a
        # fixed-ring approximation would drift up to one bucket width.
        # Cancel-to-trade: per-tick cancel sums over last 300 ticks + trade
        # volume buffer over last 30 s.
        "_cancel_buf",     # deque of (bid_cancel + ask_cancel) per tick
        "_cancel_sum",
        "_trade_buf_30s",  # deque of (ts_ms, |qty|) — 30 s window
        "_trade_abs_sum_30s",
        # Stage D — cross-exchange + ETH.
        "_btc_ret_buf",
        "_bybit_ret_buf",
        "_eth_ret_buf",
        "_bybit_last_price",
        "_eth_last_price",
        "_bybit_prev_price",
        "_eth_prev_price",
        "_eth_price_buf",
        "_okx_buf",
        "_okx_sum",
        "_bitget_buf",
        "_bitget_sum",
        "_gateio_buf",
        "_gateio_sum",
        # output
        "features",
    )

    def __init__(self) -> None:
        self._mid_buf: deque[float] = deque(maxlen=_W_120S + 1)
        self._ret_buf: deque[float] = deque(maxlen=_W_120S)
        self._abs_ret_buf: deque[float] = deque(maxlen=_W_120S)
        self._ret_sq_sum_60: float = 0.0
        self._ret_sq_sum_120: float = 0.0
        self._bv_sum_120: float = 0.0
        self._last_mid: float = 0.0
        self._last_log_mid: float = 0.0

        self._last_bid_qty0: float = 0.0
        self._last_ask_qty0: float = 0.0
        self._pending_raw: float = 0.0
        self._has_pending_raw: bool = False
        # Rolling windows for OFI sums. Each holds the W most recent raw
        # values that ARE currently in the `[T-W..T-1]` sum; the newest raw
        # (for current tick T) is held back in `_pending_raw` and added on
        # the next on_mid call.
        self._ofi_buf_60: deque[float] = deque(maxlen=_W_60S)
        self._ofi_buf_120: deque[float] = deque(maxlen=_W_120S)
        self._ofi_sum_60: float = 0.0
        self._ofi_sum_120: float = 0.0

        self._last_ts_ms: int = 0

        # Trade buffer: (ts_ms, signed_qty) pairs within the last 60 s.
        self._trade_buf: deque[tuple[int, float]] = deque()
        self._trade_signed_sum: float = 0.0
        self._trade_abs_sum: float = 0.0

        self._mark_price: float = 0.0

        # --- Stage C state ---
        self._last_bp0: float = 0.0
        self._last_ap0: float = 0.0
        self._last_bq5: np.ndarray = np.zeros(5, dtype=np.float64)
        self._last_aq5: np.ndarray = np.zeros(5, dtype=np.float64)
        self._ofi5_buf: deque[float] = deque(maxlen=_W_3S)
        self._ofi5_sum: float = 0.0
        self._pending_ofi5: float = 0.0
        self._pending_cancel: float = 0.0
        self._has_pending_c: bool = False
        # Kyle's lambda.
        self._kyle_buf: deque[tuple[float, float]] = deque(maxlen=_W_60S)
        self._kyle_xy_sum: float = 0.0
        self._kyle_xx_sum: float = 0.0
        self._pending_kyle_x: float = 0.0
        self._pending_kyle_y: float = 0.0
        self._has_pending_kyle: bool = False
        self._signed_vol_current_tick: float = 0.0
        # Cancel-to-trade.
        self._cancel_buf: deque[float] = deque(maxlen=_W_30S)
        self._cancel_sum: float = 0.0
        self._trade_buf_30s: deque[tuple[int, float]] = deque()
        self._trade_abs_sum_30s: float = 0.0

        # Stage D — cross-exchange + ETH.
        self._btc_ret_buf: deque[float] = deque(maxlen=_W_30S)
        self._bybit_ret_buf: deque[float] = deque(maxlen=_W_30S)
        self._eth_ret_buf: deque[float] = deque(maxlen=_W_30S)
        self._bybit_last_price: float = 0.0
        self._eth_last_price: float = 0.0
        self._bybit_prev_price: float = 0.0
        self._eth_prev_price: float = 0.0
        self._eth_price_buf: deque[float] = deque(maxlen=_W_120S + 1)
        self._okx_buf: deque[tuple[int, float]] = deque()
        self._okx_sum: float = 0.0
        self._bitget_buf: deque[tuple[int, float]] = deque()
        self._bitget_sum: float = 0.0
        self._gateio_buf: deque[tuple[int, float]] = deque()
        self._gateio_sum: float = 0.0

        self.features: np.ndarray = np.zeros(NUM_EXT_FEATURES, dtype=np.float32)

    def set_funding(self, mark_price: float) -> None:
        """Record the latest mark price (used for `funding_basis_bps`)."""
        if mark_price > 0:
            self._mark_price = float(mark_price)

    def on_bybit_trade(self, ts_ms: int, price: float) -> None:  # noqa: ARG002 — ts reserved
        """Record latest Bybit BTC trade price for lead-lag correlation."""
        if price > 0:
            self._bybit_last_price = float(price)

    def on_eth_trade(self, ts_ms: int, price: float) -> None:  # noqa: ARG002
        """Record latest Binance ETH trade price for ETH-momentum + ETH/BTC corr."""
        if price > 0:
            self._eth_last_price = float(price)

    def on_cross_flow(self, exchange: str, ts_ms: int, signed_qty: float) -> None:
        """Record a cross-exchange trade for the 30 s net-flow windows.

        Valid `exchange` values: "okx", "bitget", "gateio". Bybit is consumed
        via on_bybit_trade (price, not flow — we already have feature 30 for
        the cross-exchange momentum count).
        """
        if signed_qty == 0.0:
            return
        q = float(signed_qty)
        ts = int(ts_ms)
        if exchange == "okx":
            self._okx_buf.append((ts, q))
            self._okx_sum += q
        elif exchange == "bitget":
            self._bitget_buf.append((ts, q))
            self._bitget_sum += q
        elif exchange == "gateio":
            self._gateio_buf.append((ts, q))
            self._gateio_sum += q

    def _evict_cross(self, now_ms: int) -> None:
        cutoff = now_ms - _CTR_TRADE_WINDOW_MS
        for buf, sum_name in (
            (self._okx_buf, "_okx_sum"),
            (self._bitget_buf, "_bitget_sum"),
            (self._gateio_buf, "_gateio_sum"),
        ):
            while buf and buf[0][0] < cutoff:
                _, q = buf.popleft()
                setattr(self, sum_name, getattr(self, sum_name) - q)

    def on_trade(self, ts_ms: int, signed_qty: float) -> None:
        """Ingest one trade; `signed_qty > 0` for buyer-initiated trades."""
        if signed_qty == 0.0:
            return
        ts = int(ts_ms)
        q = float(signed_qty)
        absq = abs(q)
        # Stage B: 60 s trade-flow window.
        self._trade_buf.append((ts, q))
        self._trade_signed_sum += q
        self._trade_abs_sum += absq
        # Stage C: 30 s trade-volume window for cancel-to-trade.
        self._trade_buf_30s.append((ts, absq))
        self._trade_abs_sum_30s += absq
        # Stage C: accumulate signed volume for Kyle's lambda current tick bin.
        self._signed_vol_current_tick += q
        self._evict_trades(ts)
        self._evict_trades_30s(ts)

    def _evict_trades_30s(self, now_ms: int) -> None:
        cutoff = now_ms - _CTR_TRADE_WINDOW_MS
        buf = self._trade_buf_30s
        while buf and buf[0][0] < cutoff:
            _, absq = buf.popleft()
            self._trade_abs_sum_30s -= absq

    def _vpin_from_buf(self) -> float:
        """Compute VPIN from `_trade_buf` (60 s window) by sliding-sub-buckets.

        Matches the searchsorted-based batch implementation: 6 sub-buckets
        of 10 s each ending at `_last_ts_ms`. O(len(_trade_buf)) per call —
        called only on get() (sampling cadence), not per tick.
        """
        if not self._trade_buf or self._last_ts_ms <= 0:
            return 0.0
        now = self._last_ts_ms
        signed = [0.0] * _VPIN_NUM_BUCKETS
        absq = [0.0] * _VPIN_NUM_BUCKETS
        for ts, q in self._trade_buf:
            # Bucket k (0 = newest) covers (now - (k+1)*10_000, now - k*10_000].
            delta = now - ts
            if delta < 0:
                continue
            k = int(delta // _VPIN_BUCKET_MS)
            if k >= _VPIN_NUM_BUCKETS:
                continue
            signed[k] += q
            absq[k] += abs(q)
        total = sum(absq)
        if total <= 0:
            return 0.0
        net = sum(abs(s) for s in signed)
        return net / total

    def _evict_trades(self, now_ms: int) -> None:
        cutoff = now_ms - _TFI_WINDOW_MS
        buf = self._trade_buf
        while buf and buf[0][0] < cutoff:
            _, q = buf.popleft()
            self._trade_signed_sum -= q
            self._trade_abs_sum -= abs(q)

    def on_mid(
        self,
        ts_ms: int,
        mid: float,
        bid_qty0: float = 0.0,
        ask_qty0: float = 0.0,
    ) -> None:
        """Ingest one depth tick's top-of-book state in O(1)."""
        if mid <= 0:
            return

        self._last_ts_ms = int(ts_ms)

        # --- mid-price side (Stage A state) ---
        log_mid = log(mid)
        if self._last_mid > 0:
            r = log_mid - self._last_log_mid
            abs_r = abs(r)
            r_sq = r * r

            # RV-60 running sum (evict the element that leaves the window).
            if len(self._ret_buf) >= _W_60S:
                old = self._ret_buf[len(self._ret_buf) - _W_60S]
                self._ret_sq_sum_60 -= old * old
            self._ret_sq_sum_60 += r_sq

            # RV-120 running sum.
            if len(self._ret_buf) == _W_120S:
                evicted = self._ret_buf[0]
                self._ret_sq_sum_120 -= evicted * evicted
            self._ret_sq_sum_120 += r_sq

            # Bipower variation: new pair is |r_t| · |r_{t-1}|.
            if self._abs_ret_buf:
                self._bv_sum_120 += abs_r * self._abs_ret_buf[-1]
            if len(self._abs_ret_buf) == _W_120S:
                old_bv_term = self._abs_ret_buf[0] * self._abs_ret_buf[1]
                self._bv_sum_120 -= old_bv_term

            self._ret_buf.append(r)
            self._abs_ret_buf.append(abs_r)

            # --- Stage C: Kyle's lambda running sums (lagged window) ---
            # Age the PREVIOUS tick's pair into the sums first; hold the
            # current tick's pair in `_pending_kyle_*` until the next call.
            if self._has_pending_kyle:
                prev_x = self._pending_kyle_x
                prev_y = self._pending_kyle_y
                if len(self._kyle_buf) == _W_60S:
                    old_x, old_y = self._kyle_buf[0]
                    self._kyle_xy_sum -= old_x * old_y
                    self._kyle_xx_sum -= old_x * old_x
                self._kyle_buf.append((prev_x, prev_y))
                self._kyle_xy_sum += prev_x * prev_y
                self._kyle_xx_sum += prev_x * prev_x
            self._pending_kyle_x = self._signed_vol_current_tick
            self._pending_kyle_y = r
            self._has_pending_kyle = True
        # Reset the per-tick signed-vol accumulator regardless of whether
        # we could emit a Kyle pair (first tick has no previous mid).
        self._signed_vol_current_tick = 0.0

        self._mid_buf.append(mid)
        self._last_mid = mid
        self._last_log_mid = log_mid

        # --- OFI side (Stage B state) ---
        # Window semantics at tick T: sum over raw[T-W..T-1] (EXCLUDES T).
        # We achieve this by holding the most recently computed raw back in
        # `_pending_raw` and aging it into the sums on the NEXT on_mid call.
        #
        # Step 1: age the previous-tick's raw into the 60/120 s sums.
        if self._has_pending_raw:
            prev_raw = self._pending_raw
            # Evict oldest from each window if it's already full before append.
            if len(self._ofi_buf_60) == _W_60S:
                self._ofi_sum_60 -= self._ofi_buf_60[0]
            self._ofi_buf_60.append(prev_raw)
            self._ofi_sum_60 += prev_raw
            if len(self._ofi_buf_120) == _W_120S:
                self._ofi_sum_120 -= self._ofi_buf_120[0]
            self._ofi_buf_120.append(prev_raw)
            self._ofi_sum_120 += prev_raw

        # Step 2: compute current tick's raw (first tick: no previous L1 → 0).
        if self._has_pending_raw or self._last_bid_qty0 > 0 or self._last_ask_qty0 > 0:
            raw_t = (bid_qty0 - self._last_bid_qty0) - (ask_qty0 - self._last_ask_qty0)
        else:
            raw_t = 0.0
        self._pending_raw = raw_t
        self._has_pending_raw = True
        self._last_bid_qty0 = bid_qty0
        self._last_ask_qty0 = ask_qty0

        # Evict stale trades to keep TFI honest.
        self._evict_trades(int(ts_ms))
        self._evict_trades_30s(int(ts_ms))
        self._evict_cross(int(ts_ms))

        # --- Stage D: per-tick BTC/Bybit/ETH log-returns ---
        # BTC log-return r is already computed above when _last_mid > 0; mirror
        # Stage A's deque into the Stage D buffer so correlations see the same
        # series.
        if len(self._ret_buf) > 0:
            self._btc_ret_buf.append(self._ret_buf[-1])
        # Bybit / ETH returns: we use the most recent trade price as proxy.
        # Compute log-return vs the last stored price snapshot.
        if self._bybit_last_price > 0:
            if self._bybit_prev_price > 0:
                self._bybit_ret_buf.append(
                    log(self._bybit_last_price / self._bybit_prev_price))
            self._bybit_prev_price = self._bybit_last_price
        if self._eth_last_price > 0:
            if self._eth_prev_price > 0:
                self._eth_ret_buf.append(
                    log(self._eth_last_price / self._eth_prev_price))
            self._eth_prev_price = self._eth_last_price
        # ETH price buffer for eth_momentum_60s (stores eth_last at each tick).
        self._eth_price_buf.append(self._eth_last_price)

    def on_depth_l5(
        self,
        ts_ms: int,
        mid: float,
        bid_prices: np.ndarray,
        bid_qtys: np.ndarray,
        ask_prices: np.ndarray,
        ask_qtys: np.ndarray,
    ) -> None:
        """Full-depth ingestion for Stage C features.

        Expects top-5 arrays (shape ≥ (5,)); only bid/ask qty at level 0 is
        used for Stage B OFI. The engine also consumes top-5 qtys for the
        weighted OFI and cancel features.
        """
        bq5 = np.ascontiguousarray(bid_qtys[:5], dtype=np.float64)
        aq5 = np.ascontiguousarray(ask_qtys[:5], dtype=np.float64)

        # Step 1: age the previous tick's pending raw5/cancel into the sums.
        # This keeps both windows lagged (batch uses [T-W..T-1]).
        if self._has_pending_c:
            prev_ofi5 = self._pending_ofi5
            if len(self._ofi5_buf) == _W_3S:
                self._ofi5_sum -= self._ofi5_buf[0]
            self._ofi5_buf.append(prev_ofi5)
            self._ofi5_sum += prev_ofi5

            prev_cancel = self._pending_cancel
            if len(self._cancel_buf) == _W_30S:
                self._cancel_sum -= self._cancel_buf[0]
            self._cancel_buf.append(prev_cancel)
            self._cancel_sum += prev_cancel

        # Step 2: compute current-tick raw/cancel (first tick: no prior state → 0).
        if self._last_bp0 > 0 or self._last_ap0 > 0:
            db = self._last_bq5 - bq5
            da = self._last_aq5 - aq5
            cancel_tick = float(np.maximum(db, 0.0).sum() + np.maximum(da, 0.0).sum())
            raw5 = float(((bq5 - self._last_bq5) - (aq5 - self._last_aq5)) @ _OFI5_WEIGHTS)
        else:
            cancel_tick = 0.0
            raw5 = 0.0
        self._pending_ofi5 = raw5
        self._pending_cancel = cancel_tick
        self._has_pending_c = True

        # Cache top-5 for next tick's diffs.
        self._last_bq5 = bq5
        self._last_aq5 = aq5
        self._last_bp0 = float(bid_prices[0])
        self._last_ap0 = float(ask_prices[0])

        # Delegate the Stage A+B path (mid + L1 qty + Kyle + OFI windows).
        self.on_mid(ts_ms, mid, float(bq5[0]), float(aq5[0]))

    def get(self) -> np.ndarray:
        """Return the current (11,) float32 feature vector."""
        f = self.features
        f.fill(0.0)

        n = len(self._mid_buf)
        if n == 0 or self._last_mid <= 0:
            return f

        cur = self._last_mid

        # [0,1,2] momentum 30/60/120 s
        if n > _W_30S:
            past = self._mid_buf[n - 1 - _W_30S]
            if past > 0:
                f[0] = (cur - past) / past
        if n > _W_60S:
            past = self._mid_buf[n - 1 - _W_60S]
            if past > 0:
                f[1] = (cur - past) / past
        if n > _W_120S:
            past = self._mid_buf[n - 1 - _W_120S]
            if past > 0:
                f[2] = (cur - past) / past

        # [3,4] realised vol 60/120 s
        m = len(self._ret_buf)
        if m >= _W_60S:
            f[3] = sqrt(max(self._ret_sq_sum_60, 0.0))
        if m >= _W_120S:
            f[4] = sqrt(max(self._ret_sq_sum_120, 0.0))
            f[5] = _BV_SCALE * self._bv_sum_120

        # [6,7] OFI windows — emit only when window is fully populated.
        # After processing tick T, _ofi_buf_60 holds raw[T-W..T-1]. That's
        # exactly W elements once T >= W.
        if len(self._ofi_buf_60) >= _W_60S:
            f[6] = self._ofi_sum_60
        if len(self._ofi_buf_120) >= _W_120S:
            f[7] = self._ofi_sum_120

        # [8] trade-flow imbalance 60 s
        if self._trade_abs_sum > 0.0:
            f[8] = self._trade_signed_sum / self._trade_abs_sum

        # [9] funding time-to-next (min)
        if self._last_ts_ms > 0:
            f[9] = _minutes_to_next_funding(self._last_ts_ms)

        # [10] funding basis bps (mark - mid) / mid · 10_000
        if self._mark_price > 0 and cur > 0:
            f[10] = (self._mark_price - cur) / cur * 10_000.0

        # [11] microprice deviation
        if self._last_bp0 > 0 and self._last_ap0 > 0:
            b0 = self._last_bp0
            a0 = self._last_ap0
            bq0 = self._last_bq5[0]
            aq0 = self._last_aq5[0]
            tot = bq0 + aq0
            spread = a0 - b0
            if tot > 0 and spread > 1e-12:
                microprice = (aq0 * b0 + bq0 * a0) / tot
                f[11] = (microprice - cur) / spread

        # [12] ofi_top5_weighted over last 30 ticks
        if len(self._ofi5_buf) >= _W_3S:
            f[12] = self._ofi5_sum

        # [13] kyle_lambda_60s
        if len(self._kyle_buf) >= _W_60S and self._kyle_xx_sum > 1e-18:
            f[13] = self._kyle_xy_sum / self._kyle_xx_sum

        # [14] vpin_60s — partition _trade_buf into 6 sliding sub-buckets.
        f[14] = self._vpin_from_buf()

        # [15] cancel_to_trade_ratio_30s
        if len(self._cancel_buf) >= _W_30S and self._trade_abs_sum_30s > 0:
            f[15] = self._cancel_sum / self._trade_abs_sum_30s

        # === Stage D ===
        # [16] bybit_lead_lag_corr_30s — corr(btc_ret[t], bybit_ret[t-1]) over
        # 300-tick window. Requires ≥ 301 aligned pairs (so we can lag by 1).
        btc_n = len(self._btc_ret_buf)
        bybit_n = len(self._bybit_ret_buf)
        if btc_n >= _W_30S and bybit_n >= _W_30S:
            # Take the last 300 btc rets and 300 bybit rets shifted by one
            # from the deques. deques index fast (O(1)) for ends, but loops
            # over middle are acceptable at sample cadence.
            btc_arr = np.fromiter(
                (self._btc_ret_buf[i] for i in range(btc_n - _W_30S, btc_n)),
                dtype=np.float64, count=_W_30S)
            bybit_arr = np.fromiter(
                (self._bybit_ret_buf[i] for i in range(bybit_n - _W_30S, bybit_n)),
                dtype=np.float64, count=_W_30S)
            # Shift bybit by 1 step to realise the lead-lag alignment:
            # pair (btc[t], bybit[t-1]) for t ∈ [1..W-1]; drop t=0.
            x = btc_arr[1:]
            y = bybit_arr[:-1]
            f[16] = _pearson_corr(x, y)

        # [17-19] Cross-exchange net flows.
        f[17] = self._okx_sum
        f[18] = self._bitget_sum
        f[19] = self._gateio_sum

        # [20] eth_momentum_60s
        npb = len(self._eth_price_buf)
        if npb > _W_60S and self._eth_last_price > 0:
            past = self._eth_price_buf[npb - 1 - _W_60S]
            if past > 0:
                f[20] = (self._eth_last_price - past) / past

        # [21] eth_btc_corr_30s
        eth_n = len(self._eth_ret_buf)
        if btc_n >= _W_30S and eth_n >= _W_30S:
            btc_arr = np.fromiter(
                (self._btc_ret_buf[i] for i in range(btc_n - _W_30S, btc_n)),
                dtype=np.float64, count=_W_30S)
            eth_arr = np.fromiter(
                (self._eth_ret_buf[i] for i in range(eth_n - _W_30S, eth_n)),
                dtype=np.float64, count=_W_30S)
            f[21] = _pearson_corr(btc_arr, eth_arr)

        return f


def compute_ext_features_batch(
    mid_prices: np.ndarray,
    indices: np.ndarray,
    *,
    bid_qty0: np.ndarray | None = None,
    ask_qty0: np.ndarray | None = None,
    depth_ts_ms: np.ndarray | None = None,
    trade_ts_ms: np.ndarray | None = None,
    trade_signed_qty: np.ndarray | None = None,
    funding_ts_ms: np.ndarray | None = None,
    funding_mark: np.ndarray | None = None,
    bid_prices_top5: np.ndarray | None = None,
    bid_qtys_top5: np.ndarray | None = None,
    ask_prices_top5: np.ndarray | None = None,
    ask_qtys_top5: np.ndarray | None = None,
    bybit_ts_ms: np.ndarray | None = None,
    bybit_price: np.ndarray | None = None,
    eth_ts_ms: np.ndarray | None = None,
    eth_price: np.ndarray | None = None,
    okx_ts_ms: np.ndarray | None = None,
    okx_signed_qty: np.ndarray | None = None,
    bitget_ts_ms: np.ndarray | None = None,
    bitget_signed_qty: np.ndarray | None = None,
    gateio_ts_ms: np.ndarray | None = None,
    gateio_signed_qty: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised train-time computation of the 11 ext features.

    Parity contract: byte-identical (up to f32 rounding) with
    `FeatureExtEngine` streamed tick-by-tick over the same inputs and
    sampled at `indices`.

    Parameters
    ----------
    mid_prices : (n,) float64
        Per-tick mid price at 100 ms cadence.
    indices : (ns,) int64
        Sample indices into `mid_prices`.
    bid_qty0, ask_qty0 : (n,) float64, optional
        Top-of-book quantities used for OFI windows [6,7]. Omit to skip.
    depth_ts_ms : (n,) int64, optional
        Depth timestamps in ms. Needed for funding time-to-next [9] and
        for trade/funding joins.
    trade_ts_ms, trade_signed_qty : (ntr,) arrays, optional
        Trade stream for trade_flow_imbalance_60s [8]. `signed_qty > 0`
        for buyer-initiated trades.
    funding_ts_ms, funding_mark : (nf,) arrays, optional
        Funding/mark stream for basis [10].

    Returns
    -------
    (ns, 11) float32
    """
    n = len(mid_prices)
    ns = len(indices)
    out = np.zeros((ns, NUM_EXT_FEATURES), dtype=np.float32)
    if n < 2 or ns == 0:
        return out

    idx = np.asarray(indices, dtype=np.int64)

    # === Stage A — momentum / RV / BV ===
    cur_mid = mid_prices[idx]
    for out_col, w in ((0, _W_30S), (1, _W_60S), (2, _W_120S)):
        mask = idx >= w
        if not mask.any():
            continue
        past = mid_prices[idx[mask] - w]
        safe = past > 0
        rel = np.zeros(mask.sum(), dtype=np.float64)
        rel[safe] = (cur_mid[mask][safe] - past[safe]) / past[safe]
        out[mask, out_col] = rel.astype(np.float32)

    safe_prev = mid_prices[:-1] > 0
    safe_cur = mid_prices[1:] > 0
    safe_ret = safe_prev & safe_cur
    log_mid = np.where(mid_prices > 0, np.log(np.where(mid_prices > 0, mid_prices, 1.0)), 0.0)
    r = np.zeros(n - 1, dtype=np.float64)
    r[safe_ret] = log_mid[1:][safe_ret] - log_mid[:-1][safe_ret]

    r_sq = r * r
    abs_r = np.abs(r)

    cum_sq = np.zeros(n, dtype=np.float64)
    cum_sq[1:] = np.cumsum(r_sq)

    for out_col, w in ((3, _W_60S), (4, _W_120S)):
        mask = idx >= w
        if not mask.any():
            continue
        hi = idx[mask]
        lo = hi - w
        rv = cum_sq[hi] - cum_sq[lo]
        rv = np.sqrt(np.maximum(rv, 0.0))
        out[mask, out_col] = rv.astype(np.float32)

    if n >= 3:
        pair = abs_r[1:] * abs_r[:-1]
        cum_pair = np.zeros(n, dtype=np.float64)
        cum_pair[2:] = np.cumsum(pair)
        mask = idx >= _W_120S
        if mask.any():
            hi = idx[mask]
            bv = cum_pair[hi] - cum_pair[hi - _W_120S + 1]
            out[mask, 5] = (_BV_SCALE * bv).astype(np.float32)

    # === Stage B — OFI windows 60 s / 120 s ===
    # Matches streaming semantics: ofi_raw[0] := 0 (no previous tick), then
    # ofi_raw[t] = (bq0[t] - bq0[t-1]) - (aq0[t] - aq0[t-1]). The window at
    # sample index T covers ofi_raw[T-W .. T-1] (W raw values, T-1 inclusive).
    if bid_qty0 is not None and ask_qty0 is not None and n >= 2:
        bq = np.asarray(bid_qty0, dtype=np.float64)
        aq = np.asarray(ask_qty0, dtype=np.float64)
        ofi_raw = np.zeros(n, dtype=np.float64)
        ofi_raw[1:] = (bq[1:] - bq[:-1]) - (aq[1:] - aq[:-1])
        # cum_ofi[k] = Σ_{i=0}^{k-1} ofi_raw[i]; sum over [T-W, T-1] = cum[T] - cum[T-W].
        cum_ofi = np.zeros(n + 1, dtype=np.float64)
        cum_ofi[1:] = np.cumsum(ofi_raw)
        for out_col, w in ((6, _W_60S), (7, _W_120S)):
            mask = idx >= w
            if not mask.any():
                continue
            hi = idx[mask]
            val = cum_ofi[hi] - cum_ofi[hi - w]
            out[mask, out_col] = val.astype(np.float32)

    # === Stage B — trade_flow_imbalance_60s ===
    if (
        trade_ts_ms is not None
        and trade_signed_qty is not None
        and depth_ts_ms is not None
        and len(trade_ts_ms) > 0
    ):
        t_ts = np.asarray(trade_ts_ms, dtype=np.int64)
        t_q = np.asarray(trade_signed_qty, dtype=np.float64)
        abs_q = np.abs(t_q)
        cum_signed = np.zeros(len(t_ts) + 1, dtype=np.float64)
        cum_abs = np.zeros(len(t_ts) + 1, dtype=np.float64)
        cum_signed[1:] = np.cumsum(t_q)
        cum_abs[1:] = np.cumsum(abs_q)

        sample_ts = np.asarray(depth_ts_ms, dtype=np.int64)[idx]
        # Streaming `on_trade` inserts and then `on_mid` evicts the ones
        # with ts < now - window. So the window at sample ts_now covers
        # trades with ts in [ts_now - W, ts_now] — matching trade_flow
        # features elsewhere (e.g. fill_trade_features uses similar bounds).
        lo = np.searchsorted(t_ts, sample_ts - _TFI_WINDOW_MS, side="left")
        hi = np.searchsorted(t_ts, sample_ts, side="right")
        signed = cum_signed[hi] - cum_signed[lo]
        total = cum_abs[hi] - cum_abs[lo]
        safe = total > 0
        tfi = np.zeros(ns, dtype=np.float64)
        tfi[safe] = signed[safe] / total[safe]
        out[:, 8] = tfi.astype(np.float32)

    # === Stage B — funding_time_to_next_min ===
    if depth_ts_ms is not None:
        sample_ts = np.asarray(depth_ts_ms, dtype=np.int64)[idx]
        rem = np.where(sample_ts > 0, sample_ts % _FUNDING_PERIOD_MS, 0)
        mins = np.where(rem == 0, 0.0, (_FUNDING_PERIOD_MS - rem) / 60_000.0)
        out[:, 9] = mins.astype(np.float32)

    # === Stage C — microprice_deviation ===
    # Uses top-1 price/qty only; computed per sample (no window).
    if (
        bid_prices_top5 is not None
        and bid_qtys_top5 is not None
        and ask_prices_top5 is not None
        and ask_qtys_top5 is not None
    ):
        bp = np.asarray(bid_prices_top5, dtype=np.float64)
        bq5 = np.asarray(bid_qtys_top5, dtype=np.float64)
        ap = np.asarray(ask_prices_top5, dtype=np.float64)
        aq5 = np.asarray(ask_qtys_top5, dtype=np.float64)
        b0 = bp[idx, 0]
        a0 = ap[idx, 0]
        bq0 = bq5[idx, 0]
        aq0 = aq5[idx, 0]
        tot = bq0 + aq0
        spread = a0 - b0
        safe = (tot > 0) & (spread > 1e-12)
        mid_cur = mid_prices[idx]
        dev = np.zeros(ns, dtype=np.float64)
        micro = np.where(safe, (aq0 * b0 + bq0 * a0) / np.where(tot > 0, tot, 1.0), 0.0)
        dev_safe = (micro - mid_cur) / np.where(safe, spread, 1.0)
        dev[safe] = dev_safe[safe]
        out[:, 11] = dev.astype(np.float32)

        # === Stage C — ofi_top5_weighted over 30 ticks ===
        # Per-tick raw: Σ_k w_k · ((bq_k[t] - bq_k[t-1]) - (aq_k[t] - aq_k[t-1])).
        # Lagged window [T-30..T-1] to mirror Stage B OFI semantics.
        if n >= 2:
            d_bq = bq5[1:] - bq5[:-1]
            d_aq = aq5[1:] - aq5[:-1]
            raw_per_tick = np.zeros(n, dtype=np.float64)
            raw_per_tick[1:] = (d_bq - d_aq) @ _OFI5_WEIGHTS
            cum_ofi5 = np.zeros(n + 1, dtype=np.float64)
            cum_ofi5[1:] = np.cumsum(raw_per_tick)
            mask = idx >= _W_3S
            if mask.any():
                hi = idx[mask]
                val = cum_ofi5[hi] - cum_ofi5[hi - _W_3S]
                out[mask, 12] = val.astype(np.float32)

        # === Stage C — cancel_to_trade_ratio_30s (numerator only here) ===
        # cancel_tick = Σ_k max(0, bq[t-1,k] - bq[t,k]) + ditto for ask.
        if n >= 2:
            cancel_tick = np.zeros(n, dtype=np.float64)
            cancel_tick[1:] = (
                np.maximum(bq5[:-1] - bq5[1:], 0.0).sum(axis=1)
                + np.maximum(aq5[:-1] - aq5[1:], 0.0).sum(axis=1)
            )
            cum_cancel = np.zeros(n + 1, dtype=np.float64)
            cum_cancel[1:] = np.cumsum(cancel_tick)
            mask = idx >= _W_30S
            cancel_window = np.zeros(ns, dtype=np.float64)
            if mask.any():
                hi = idx[mask]
                cancel_window[mask] = cum_cancel[hi] - cum_cancel[hi - _W_30S]
            # Denominator comes from trades (below); store numerator temporarily.
            out[:, 15] = cancel_window.astype(np.float32)

    # === Stage C — kyle_lambda_60s ===
    # Needs per-tick signed volume (trades aligned to depth ticks) and Δlog_mid.
    # We lag by 1 tick to match the streaming convention: window is (T-W..T-1].
    if (
        trade_ts_ms is not None
        and trade_signed_qty is not None
        and depth_ts_ms is not None
        and len(trade_ts_ms) > 0
    ):
        dts = np.asarray(depth_ts_ms, dtype=np.int64)
        t_ts = np.asarray(trade_ts_ms, dtype=np.int64)
        t_q = np.asarray(trade_signed_qty, dtype=np.float64)
        # x[t] = Σ signed_qty for trades with ts in (dts[t-1], dts[t]].
        cum_q = np.zeros(len(t_ts) + 1, dtype=np.float64)
        cum_q[1:] = np.cumsum(t_q)
        right_cur = np.searchsorted(t_ts, dts, side="right")
        x_per_tick = np.zeros(n, dtype=np.float64)
        x_per_tick[1:] = cum_q[right_cur[1:]] - cum_q[right_cur[:-1]]
        # y[t] = r[t] = log_mid[t] - log_mid[t-1]; reuse `r` computed above.
        y_per_tick = np.zeros(n, dtype=np.float64)
        y_per_tick[1:] = r  # r is len n-1; y[0] stays 0.
        xy = x_per_tick * y_per_tick
        xx = x_per_tick * x_per_tick
        cum_xy = np.zeros(n + 1, dtype=np.float64)
        cum_xx = np.zeros(n + 1, dtype=np.float64)
        cum_xy[1:] = np.cumsum(xy)
        cum_xx[1:] = np.cumsum(xx)
        mask = idx >= _W_60S
        if mask.any():
            hi = idx[mask]
            num = cum_xy[hi] - cum_xy[hi - _W_60S]
            den = cum_xx[hi] - cum_xx[hi - _W_60S]
            beta = np.zeros(mask.sum(), dtype=np.float64)
            safe = den > 1e-18
            beta[safe] = num[safe] / den[safe]
            out[mask, 13] = beta.astype(np.float32)

        # === Stage C — vpin_60s ===
        # 6 sub-buckets × 10 s each = 60 s window ending at sample_ts.
        sample_ts = dts[idx]
        abs_q = np.abs(t_q)
        cum_signed = np.zeros(len(t_ts) + 1, dtype=np.float64)
        cum_abs = np.zeros(len(t_ts) + 1, dtype=np.float64)
        cum_signed[1:] = np.cumsum(t_q)
        cum_abs[1:] = np.cumsum(abs_q)
        sum_abs_net = np.zeros(ns, dtype=np.float64)
        sum_total = np.zeros(ns, dtype=np.float64)
        for k in range(_VPIN_NUM_BUCKETS):
            # Bucket k covers (sample_ts - (k+1)*_VPIN_BUCKET_MS,
            #                  sample_ts - k*_VPIN_BUCKET_MS].
            hi_ts = sample_ts - k * _VPIN_BUCKET_MS
            lo_ts = sample_ts - (k + 1) * _VPIN_BUCKET_MS
            hi_idx = np.searchsorted(t_ts, hi_ts, side="right")
            lo_idx = np.searchsorted(t_ts, lo_ts, side="right")
            net_k = cum_signed[hi_idx] - cum_signed[lo_idx]
            total_k = cum_abs[hi_idx] - cum_abs[lo_idx]
            sum_abs_net += np.abs(net_k)
            sum_total += total_k
        vpin = np.zeros(ns, dtype=np.float64)
        safe = sum_total > 0
        vpin[safe] = sum_abs_net[safe] / sum_total[safe]
        out[:, 14] = vpin.astype(np.float32)

        # === Stage C — cancel_to_trade_ratio_30s denominator ===
        # out[:, 15] currently holds the cancel numerator (set above if depth
        # arrays provided). Divide by trade volume over last 30 s.
        if bid_qtys_top5 is not None:
            lo = np.searchsorted(t_ts, sample_ts - _CTR_TRADE_WINDOW_MS, side="left")
            hi = np.searchsorted(t_ts, sample_ts, side="right")
            trade_vol_30s = cum_abs[hi] - cum_abs[lo]
            num = out[:, 15].astype(np.float64)
            ratio = np.zeros(ns, dtype=np.float64)
            safe = trade_vol_30s > 0
            ratio[safe] = num[safe] / trade_vol_30s[safe]
            # Also require the cancel window to be saturated (matches streaming
            # gate `len(cancel_buf) >= _W_30S`).
            ready = np.asarray(indices, dtype=np.int64) >= _W_30S
            out[:, 15] = np.where(ready & safe, ratio, 0.0).astype(np.float32)

    # === Stage D — cross-exchange net flows ===
    if depth_ts_ms is not None:
        dts = np.asarray(depth_ts_ms, dtype=np.int64)
        sample_ts_d = dts[idx]
        for out_col, ex_ts, ex_q in (
            (17, okx_ts_ms, okx_signed_qty),
            (18, bitget_ts_ms, bitget_signed_qty),
            (19, gateio_ts_ms, gateio_signed_qty),
        ):
            if ex_ts is None or ex_q is None or len(ex_ts) == 0:
                continue
            e_ts = np.asarray(ex_ts, dtype=np.int64)
            e_q = np.asarray(ex_q, dtype=np.float64)
            cum_q = np.zeros(len(e_ts) + 1, dtype=np.float64)
            cum_q[1:] = np.cumsum(e_q)
            lo = np.searchsorted(e_ts, sample_ts_d - _CTR_TRADE_WINDOW_MS, side="left")
            hi = np.searchsorted(e_ts, sample_ts_d, side="right")
            out[:, out_col] = (cum_q[hi] - cum_q[lo]).astype(np.float32)

    # === Stage D — ETH momentum + correlations, Bybit lead-lag ===
    # Pre-compute per-BTC-tick last trade price for bybit + eth by searchsorted.
    def _last_price_per_tick(src_ts, src_price, dts_):
        src_ts = np.asarray(src_ts, dtype=np.int64)
        src_p = np.asarray(src_price, dtype=np.float64)
        if len(src_ts) == 0:
            return np.zeros(len(dts_), dtype=np.float64)
        r = np.searchsorted(src_ts, dts_, side="right")
        fi = np.clip(r - 1, 0, len(src_ts) - 1)
        valid = r > 0
        out_p = np.where(valid, src_p[fi], 0.0)
        return out_p

    bybit_per_tick = None
    eth_per_tick = None
    if (
        depth_ts_ms is not None
        and bybit_ts_ms is not None and bybit_price is not None
        and len(bybit_ts_ms) > 0
    ):
        bybit_per_tick = _last_price_per_tick(bybit_ts_ms, bybit_price, dts)
    if (
        depth_ts_ms is not None
        and eth_ts_ms is not None and eth_price is not None
        and len(eth_ts_ms) > 0
    ):
        eth_per_tick = _last_price_per_tick(eth_ts_ms, eth_price, dts)

    # Per-tick log returns for BTC, Bybit, ETH. First tick has no prev → 0.
    btc_ret = r  # shape (n-1,) with r[t-1] = log(mid[t]/mid[t-1])
    # Align to index convention: ret_per_tick[t] for t in [1..n-1].
    ret_per_tick_btc = np.concatenate([[0.0], btc_ret])  # shape (n,)

    def _log_ret_series(prices):
        out_r = np.zeros(n, dtype=np.float64)
        if prices is None:
            return out_r
        safe_prev = prices[:-1] > 0
        safe_cur = prices[1:] > 0
        m = safe_prev & safe_cur
        vals = np.where(m, np.log(np.where(m, prices[1:] / np.where(prices[:-1] > 0, prices[:-1], 1.0), 1.0)), 0.0)
        out_r[1:] = vals
        return out_r

    ret_per_tick_bybit = _log_ret_series(bybit_per_tick)
    ret_per_tick_eth = _log_ret_series(eth_per_tick)

    # [16] bybit_lead_lag_corr_30s: corr(btc_ret[t], bybit_ret[t-1])
    # over lagged window [T-W..T-1]. Shifted series: x = btc_ret, y = bybit_ret[..-1].
    def _rolling_corr(xs, ys, window):
        """Pearson corr for each sample in `indices` over xs[idx-W..idx-1], ys[idx-W..idx-1]."""
        cx = np.zeros(n + 1)
        cy = np.zeros(n + 1)
        cxx = np.zeros(n + 1)
        cyy = np.zeros(n + 1)
        cxy = np.zeros(n + 1)
        cx[1:] = np.cumsum(xs)
        cy[1:] = np.cumsum(ys)
        cxx[1:] = np.cumsum(xs * xs)
        cyy[1:] = np.cumsum(ys * ys)
        cxy[1:] = np.cumsum(xs * ys)
        corr = np.zeros(ns, dtype=np.float64)
        mask = idx >= window
        if not mask.any():
            return corr
        hi = idx[mask]
        lo = hi - window
        sx = cx[hi] - cx[lo]
        sy = cy[hi] - cy[lo]
        sxx = cxx[hi] - cxx[lo]
        syy = cyy[hi] - cyy[lo]
        sxy = cxy[hi] - cxy[lo]
        num = window * sxy - sx * sy
        den = (window * sxx - sx * sx) * (window * syy - sy * sy)
        out_c = np.zeros(mask.sum(), dtype=np.float64)
        safe = den > 1e-24
        out_c[safe] = num[safe] / np.sqrt(den[safe])
        corr[mask] = out_c
        return corr

    if bybit_per_tick is not None:
        # Lead-lag: y aligned to btc_ret[t], x aligned to bybit_ret[t-1].
        # We form `bybit_lag = shift(ret_per_tick_bybit, 1)`.
        bybit_lag = np.zeros(n, dtype=np.float64)
        bybit_lag[1:] = ret_per_tick_bybit[:-1]
        out[:, 16] = _rolling_corr(ret_per_tick_btc, bybit_lag, _W_30S).astype(np.float32)

    # [20] eth_momentum_60s: (eth[T] - eth[T-W60]) / eth[T-W60]
    if eth_per_tick is not None:
        mask = idx >= _W_60S
        if mask.any():
            hi = idx[mask]
            cur = eth_per_tick[hi]
            past = eth_per_tick[hi - _W_60S]
            safe = (past > 0) & (cur > 0)
            mom = np.zeros(mask.sum(), dtype=np.float64)
            mom[safe] = (cur[safe] - past[safe]) / past[safe]
            out[mask, 20] = mom.astype(np.float32)
        # [21] eth_btc_corr_30s
        out[:, 21] = _rolling_corr(ret_per_tick_btc, ret_per_tick_eth, _W_30S).astype(np.float32)

    # === Stage B — funding_basis_bps ===
    if (
        funding_ts_ms is not None
        and funding_mark is not None
        and depth_ts_ms is not None
        and len(funding_ts_ms) > 0
    ):
        f_ts = np.asarray(funding_ts_ms, dtype=np.int64)
        f_mark = np.asarray(funding_mark, dtype=np.float64)
        sample_ts = np.asarray(depth_ts_ms, dtype=np.int64)[idx]
        r_idx = np.searchsorted(f_ts, sample_ts, side="right")
        fi = np.clip(r_idx - 1, 0, len(f_ts) - 1)
        mark_at = f_mark[fi]
        spot = mid_prices[idx]
        # Streaming mirror: only emit when both mark and mid are positive;
        # matches the `_mark_price > 0` gate in FeatureExtEngine.get().
        safe = (mark_at > 0) & (spot > 0) & (r_idx > 0)
        basis = np.zeros(ns, dtype=np.float64)
        basis[safe] = (mark_at[safe] - spot[safe]) / spot[safe] * 10_000.0
        out[:, 10] = basis.astype(np.float32)

    return out
