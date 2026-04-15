"""Extended feature set — horizon-tier (60-180 s) additions, Stage A.

Sidecar module kept separate from `src/features.py` until Stage E, so the
main 34-feature path (live + Rust parity + cache v3) stays untouched while
we wire up, test and parity-check the new columns independently.

Stage A features (order fixed, must mirror Rust `compute_ext_features`):

    0: momentum_30s        — (mid - mid_30s_ago) / mid_30s_ago
    1: momentum_60s        — (mid - mid_60s_ago) / mid_60s_ago
    2: momentum_120s       — (mid - mid_120s_ago) / mid_120s_ago
    3: realized_vol_60s    — sqrt(sum of squared 100 ms log-returns, 60 s)
    4: realized_vol_120s   — sqrt(sum of squared 100 ms log-returns, 120 s)
    5: bipower_var_120s    — (π/2) · Σ|r_i|·|r_{i-1}| over 120 s (jump-robust)

All windows are in depth-tick units assuming 100 ms cadence (same
convention as existing features 10/12). `get()` returns a (6,) float32
vector — append callers materialize with the main 34-vector at stacker
input time.
"""
from __future__ import annotations

from collections import deque
from math import log, sqrt

import numpy as np

NUM_EXT_FEATURES = 6

EXT_FEATURE_KEYS = [
    "momentum_30s",
    "momentum_60s",
    "momentum_120s",
    "realized_vol_60s",
    "realized_vol_120s",
    "bipower_var_120s",
]
assert len(EXT_FEATURE_KEYS) == NUM_EXT_FEATURES

# Window sizes in 100 ms ticks.
_W_30S = 300
_W_60S = 600
_W_120S = 1200

# π/2 scaling for bipower variation.
_BV_SCALE = np.pi / 2.0


class FeatureExtEngine:
    """Streaming computation of the 6 Stage-A horizon features.

    Call `on_mid(ts_ms, mid)` on every depth tick in order (monotonic ts
    is not required for correctness, only that ticks arrive at ~100 ms
    cadence — same precondition as `FeatureEngine`). Then call `get()` to
    read the current feature vector.
    """

    __slots__ = (
        "_mid_buf",
        "_ret_buf",
        "_abs_ret_buf",
        "_ret_sq_sum_60",
        "_ret_sq_sum_120",
        "_bv_sum_120",
        "_last_mid",
        "_last_log_mid",
        "features",
    )

    def __init__(self) -> None:
        self._mid_buf: deque[float] = deque(maxlen=_W_120S + 1)
        # Per-tick log-returns, needed for RV and bipower windows.
        self._ret_buf: deque[float] = deque(maxlen=_W_120S)
        self._abs_ret_buf: deque[float] = deque(maxlen=_W_120S)
        # Running sums maintained incrementally for O(1) get().
        self._ret_sq_sum_60: float = 0.0
        self._ret_sq_sum_120: float = 0.0
        self._bv_sum_120: float = 0.0
        self._last_mid: float = 0.0
        self._last_log_mid: float = 0.0
        self.features: np.ndarray = np.zeros(NUM_EXT_FEATURES, dtype=np.float32)

    def on_mid(self, mid: float) -> None:
        """Ingest one depth tick's mid price, update running sums in O(1)."""
        if mid <= 0:
            return

        log_mid = log(mid)

        # Compute log-return vs previous tick.
        if self._last_mid > 0:
            r = log_mid - self._last_log_mid
            abs_r = abs(r)
            r_sq = r * r

            # Running sum for RV-60s window.
            if len(self._ret_buf) >= _W_60S:
                old = self._ret_buf[len(self._ret_buf) - _W_60S]
                self._ret_sq_sum_60 -= old * old
            self._ret_sq_sum_60 += r_sq

            # Running sum for RV-120s window (deque-maxlen evicts oldest).
            # Before append: if buffer full, the leftmost element is about
            # to be evicted — subtract its contribution first.
            if len(self._ret_buf) == _W_120S:
                evicted = self._ret_buf[0]
                self._ret_sq_sum_120 -= evicted * evicted
            self._ret_sq_sum_120 += r_sq

            # Bipower term for position t is |r_t| * |r_{t-1}|.
            # Update incrementally: add new term, drop the oldest that falls
            # out of the 120 s window. Window holds pairs (t-1, t) so the
            # number of bipower terms equals min(len(abs_ret_buf), _W_120S-1).
            if self._abs_ret_buf:
                new_bv_term = abs_r * self._abs_ret_buf[-1]
                self._bv_sum_120 += new_bv_term
            if len(self._abs_ret_buf) == _W_120S:
                # After the push below, two earliest abs-returns will leave
                # the window. The bipower term they form gets dropped.
                old_bv_term = self._abs_ret_buf[0] * self._abs_ret_buf[1]
                self._bv_sum_120 -= old_bv_term

            self._ret_buf.append(r)
            self._abs_ret_buf.append(abs_r)

        self._mid_buf.append(mid)
        self._last_mid = mid
        self._last_log_mid = log_mid

    def get(self) -> np.ndarray:
        """Return the current (6,) float32 feature vector.

        Emits zeros for features whose window is not yet full — same
        convention as existing features 10/12/etc.
        """
        f = self.features
        f.fill(0.0)

        n = len(self._mid_buf)
        if n == 0 or self._last_mid <= 0:
            return f

        cur = self._last_mid

        # [0,1,2] momentum at 30/60/120 s
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

        # [3,4] realized vol on 60/120 s windows
        m = len(self._ret_buf)
        if m >= _W_60S:
            f[3] = sqrt(max(self._ret_sq_sum_60, 0.0))
        if m >= _W_120S:
            f[4] = sqrt(max(self._ret_sq_sum_120, 0.0))
            # [5] bipower variation
            f[5] = _BV_SCALE * self._bv_sum_120

        return f


def compute_ext_features_batch(
    mid_prices: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Vectorised train-time computation of the 6 Stage-A ext features.

    Parity contract: byte-identical (up to f32 rounding) with
    `FeatureExtEngine` streamed tick-by-tick over the same `mid_prices`
    and sampled at `indices`.

    Parameters
    ----------
    mid_prices : (n,) float64
        Per-tick mid price at 100 ms cadence.
    indices : (ns,) int64
        Sample indices into `mid_prices`.

    Returns
    -------
    (ns, 6) float32
    """
    n = len(mid_prices)
    ns = len(indices)
    out = np.zeros((ns, NUM_EXT_FEATURES), dtype=np.float32)
    if n < 2 or ns == 0:
        return out

    idx = np.asarray(indices, dtype=np.int64)

    # Momentum: (mid[i] - mid[i-W]) / mid[i-W], zero if i < W or mid[i-W] <= 0.
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

    # Log-returns array aligned so r[i] = log(mid[i+1]/mid[i]) for i in [0, n-2].
    # Then realized_var at tick T = sum(r[T-W .. T-1]^2), i.e. last W returns
    # ending at tick T. We use a cumulative-sum of r^2 for O(1) per query.
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

    # RV on window W: sum of r_sq[T-W .. T-1] = cum_sq[T] - cum_sq[T-W].
    for out_col, w in ((3, _W_60S), (4, _W_120S)):
        mask = idx >= w
        if not mask.any():
            continue
        hi = idx[mask]
        lo = hi - w
        rv = cum_sq[hi] - cum_sq[lo]
        rv = np.sqrt(np.maximum(rv, 0.0))
        out[mask, out_col] = rv.astype(np.float32)

    # Bipower variation (Barndorff-Nielsen):
    #   BV_W(T) = (π/2) · Σ_{k=T-W+1}^{T-1} |r[k]|·|r[k-1]|
    # where r[k] = log(mid[k+1]/mid[k]), W = _W_120S.
    #
    # Let pair[j] = |r[j+1]| · |r[j]|  for j in [0 .. n-3]. This is the term
    # whose "right return index" is j+1. Its contribution to BV_W(T) is
    # included iff (j + 1) in [T-W+1, T-1], i.e. j in [T-W, T-2].
    #
    # Define cum_pair[k] with cum_pair[0]=cum_pair[1]=0, and for k >= 2
    #   cum_pair[k] = Σ_{j=0}^{k-2} pair[j]
    # so that Σ_{j=a}^{b} pair[j] = cum_pair[b+2] - cum_pair[a+1]
    # (for 0 <= a <= b <= n-3).
    #
    # Substituting a = T-W, b = T-2 gives
    #   BV_sum(T) = cum_pair[T] - cum_pair[T-W+1].
    if n >= 3:
        pair = abs_r[1:] * abs_r[:-1]                 # shape (n-2,)
        cum_pair = np.zeros(n, dtype=np.float64)
        cum_pair[2:] = np.cumsum(pair)                # cum_pair[k>=2] valid
        mask = idx >= _W_120S
        if mask.any():
            hi = idx[mask]
            bv = cum_pair[hi] - cum_pair[hi - _W_120S + 1]
            out[mask, 5] = (_BV_SCALE * bv).astype(np.float32)

    return out
