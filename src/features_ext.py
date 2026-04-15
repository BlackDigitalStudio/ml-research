"""Extended feature set — horizon-tier (60-180 s) additions, Stage A + B.

Sidecar module kept separate from `src/features.py` until Stage E, so the
main feature set and its train-time batch twin share one source of truth
for stream↔batch parity.

Feature order (must mirror Rust `fill_horizon_features` / `fill_horizon_features_b`):

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

All windows are in depth-tick units assuming 100 ms cadence (same
convention as existing features 10/12). `get()` returns a (11,) float32
vector. Features emit 0 until their window is saturated, matching the
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

NUM_EXT_FEATURES = 11

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
]
assert len(EXT_FEATURE_KEYS) == NUM_EXT_FEATURES

# Window sizes in 100 ms ticks.
_W_30S = 300
_W_60S = 600
_W_120S = 1200

# Trade-flow window is time-driven (60 s), not tick-driven.
_TFI_WINDOW_MS = 60_000

# Binance funding schedule: every 8 h at 00:00, 08:00, 16:00 UTC.
_FUNDING_PERIOD_MS = 8 * 3600 * 1000

# π/2 scaling for bipower variation.
_BV_SCALE = np.pi / 2.0


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
        # We maintain _ofi_sum_60/120 over the lagged window [T-W..T-1],
        # i.e. excluding the current tick's raw. To do that we hold back
        # the most recently computed raw value in `_pending_raw` and only
        # age it into the running sum on the NEXT on_mid call.
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

        self.features: np.ndarray = np.zeros(NUM_EXT_FEATURES, dtype=np.float32)

    def set_funding(self, mark_price: float) -> None:
        """Record the latest mark price (used for `funding_basis_bps`)."""
        if mark_price > 0:
            self._mark_price = float(mark_price)

    def on_trade(self, ts_ms: int, signed_qty: float) -> None:
        """Ingest one trade; `signed_qty > 0` for buyer-initiated trades."""
        if signed_qty == 0.0:
            return
        self._trade_buf.append((int(ts_ms), float(signed_qty)))
        self._trade_signed_sum += float(signed_qty)
        self._trade_abs_sum += abs(float(signed_qty))
        # Eviction against the current tick's time is done in `on_mid`. We
        # still evict opportunistically here in case `get()` is called with
        # no further tick ingress.
        self._evict_trades(int(ts_ms))

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
