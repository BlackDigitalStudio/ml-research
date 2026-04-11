"""CNN encoder + ensemble trainer.

Reads Parquet data, builds training samples, trains models, saves with symlinks.
Designed to run in a separate process (not in the trading event loop).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

from src.config import Config, load_config
from src.model import LOBEncoder, UP, DOWN, FLAT

logger = logging.getLogger(__name__)

BOOK_DEPTH = 20
WINDOW_SIZE = 50       # 5 seconds of snapshots (50 x 100ms)
HORIZON = 600          # look 60 seconds ahead (600 x 100ms)


def _malloc_trim() -> None:
    """Force glibc to return freed heap pages back to the OS.

    Pandas DataFrames with list<list<float>> columns (our LOB schema)
    allocate hundreds of thousands of tiny Python objects. After `del df`
    those objects are reclaimed inside glibc's arenas but NOT returned
    to the kernel — RSS stays bloated even though Python sees the memory
    as free. `malloc_trim(0)` walks the arenas and madvise-releases any
    fully-free pages. No-op on non-glibc platforms.

    Called explicitly after each large dataframe deletion in build_samples.
    Costs ~100-300ms per call — negligible compared to ~1 GB of avoided
    RSS bloat.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not Linux / not glibc — best-effort


def _xgb_lgb_threads(n_jobs: int) -> int:
    """Map user-facing n_jobs to XGBoost/LightGBM thread count.

    XGBoost `nthread` and LightGBM `num_threads` treat 0 as "all cores".
    We expose -1 (sklearn convention) to mean the same thing, and any
    positive integer as an exact count. n_jobs=1 isolates training to
    a single core — the default on the 2-vCPU production VPS so the
    recorder and bot always have a core free.
    """
    return 0 if n_jobs < 0 else max(1, n_jobs)


def _pytorch_threads(n_jobs: int) -> int:
    """Map user-facing n_jobs to torch.set_num_threads argument.

    PyTorch needs a strict positive integer. -1 → os.cpu_count().
    """
    if n_jobs < 0:
        return os.cpu_count() or 1
    return max(1, n_jobs)

# Triple-barrier labelling (López de Prado, AFML Ch.3): a sample is labelled
# UP only if a LONG entry would actually win (TP hits before SL), DOWN only if
# a SHORT entry would win, otherwise FLAT. This matches the live trading
# economics; the prior max/min approach was a look-ahead bug that ignored the
# *order* of price events.
TP_PCT = 0.20   # upper barrier — matches strategy TP base (STRATEGY.md §5)
SL_PCT = 0.10   # lower barrier — matches strategy SL base (2:1 ratio)


class Trainer:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._data_dir = config.data_dir
        self._model_dir = config.model_dir

    # ---- Data loading ----

    def load_depth_data(self, hours: int = 24) -> pd.DataFrame:
        depth_dir = self._data_dir / "depth"
        files = sorted(depth_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No depth files in {depth_dir}")

        # Take last N hours of files
        files = files[-hours:]
        logger.info("Loading %d depth files...", len(files))

        tables = []
        for f in files:
            tables.append(pq.read_table(f))

        import pyarrow as pa
        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d depth snapshots (%.1f hours)", len(df), len(df) / 36000)
        return df

    def load_trade_data(self, hours: int = 24) -> pd.DataFrame:
        trades_dir = self._data_dir / "trades"
        files = sorted(trades_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No trade files in {trades_dir}")

        files = files[-hours:]
        logger.info("Loading %d trade files...", len(files))

        dfs = []
        for f in files:
            dfs.append(pd.read_parquet(f))

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Deduplicate (Binance can send duplicate aggTrade on reconnect)
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp", "price", "quantity"]).reset_index(drop=True)
        dupes = before - len(df)
        if dupes > 0:
            logger.info("Removed %d duplicate trades (%.2f%%)", dupes, dupes / before * 100)

        logger.info("Loaded %d trades", len(df))
        return df

    def _load_parquet_dir(self, subdir: str, hours: int, dedup_cols: list[str] | None = None) -> pd.DataFrame | None:
        """Load parquet files from a data subdirectory. Returns None if empty."""
        d = self._data_dir / subdir
        if not d.exists():
            return None
        files = sorted(d.glob("*.parquet"))
        if not files:
            return None
        files = files[-hours:]
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if dedup_cols:
            before = len(df)
            df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
            dupes = before - len(df)
            if dupes > 0:
                logger.info("Removed %d duplicates from %s (%.1f%%)", dupes, subdir, dupes / before * 100)
        logger.info("Loaded %d rows from %s (%d files)", len(df), subdir, len(files))
        return df

    # ---- Sample building (vectorized) ----

    def build_samples_cached(
        self, hours: int = 24, force_rebuild: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cached wrapper around build_samples.

        Cache key combines `hours` with the mtime of the newest depth parquet
        file — any new data or a different window size invalidates the cache.
        X_lob is stored as a .npy file loaded via mmap (same semantics as the
        uncached path); X_feat/y/mid are small and fully loaded.

        Set `force_rebuild=True` to ignore the cache (still writes new entry).

        Returns the same (X_lob, X_feat, y, mid_prices) 4-tuple.
        """
        cache_dir = self._data_dir / "_cache"
        cache_dir.mkdir(exist_ok=True)

        # Cheapest valid signal: mtime of the newest compacted depth file.
        # If no compacted files exist yet we fall back to 0 — the cache will
        # still work, it just won't be portable across initial runs.
        depth_files = sorted((self._data_dir / "depth").glob("*.parquet"))
        newest_mtime = int(max((f.stat().st_mtime for f in depth_files), default=0))
        key = f"{hours}h_{newest_mtime}"

        lob_path = cache_dir / f"samples_{key}_X_lob.npy"
        feat_path = cache_dir / f"samples_{key}_X_feat.npy"
        y_path = cache_dir / f"samples_{key}_y.npy"
        mid_path = cache_dir / f"samples_{key}_mid.npy"

        if not force_rebuild and all(
            p.exists() for p in (lob_path, feat_path, y_path, mid_path)
        ):
            logger.info("Sample cache HIT: %s", key)
            X_lob = np.load(str(lob_path), mmap_mode="r")
            X_feat = np.load(str(feat_path))
            y = np.load(str(y_path))
            mid = np.load(str(mid_path))
            return X_lob, X_feat, y, mid

        logger.info("Sample cache MISS (key=%s) — rebuilding", key)
        # Evict stale entries for the same `hours` but different mtimes to
        # keep the cache from growing unbounded across sessions.
        for old in cache_dir.glob(f"samples_{hours}h_*"):
            old.unlink()

        X_lob, X_feat, y, mid = self.build_samples(hours=hours, lob_output_path=lob_path)
        np.save(str(feat_path), X_feat)
        np.save(str(y_path), y)
        np.save(str(mid_path), mid)
        return X_lob, X_feat, y, mid

    def build_samples(
        self, hours: int = 24, lob_output_path: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build (X_lob, X_features, y, mid_prices) from raw data.

        Loads data, parses into numpy, frees DataFrames immediately to save RAM.
        X_lob is written to disk (mmap) — never fully held in memory.

        If `lob_output_path` is given, X_lob is persisted to that path (used
        by `build_samples_cached`); otherwise it goes to the default
        `_tmp_X_lob.npy` location.

        Returns:
            X_lob: (N, 3, 20, 50) — CNN input tensors (mmap'd from disk)
            X_features: (N, 25) — hand-crafted features
            y: (N,) — labels {0=UP, 1=DOWN, 2=FLAT}
            mid_prices: (N,) — mid price at each sample point (for backtest)
        """
        import gc

        # === Load and parse depth (free DataFrame ASAP) ===
        depth_df = self.load_depth_data(hours)
        n = len(depth_df)
        if n < WINDOW_SIZE + HORIZON + 1:
            raise ValueError(f"Not enough data: {n} rows, need {WINDOW_SIZE + HORIZON + 1}")

        logger.info("Parsing order book data...")
        bid_prices = np.zeros((n, BOOK_DEPTH), dtype=np.float64)
        bid_vols = np.zeros((n, BOOK_DEPTH), dtype=np.float32)
        ask_prices = np.zeros((n, BOOK_DEPTH), dtype=np.float64)
        ask_vols = np.zeros((n, BOOK_DEPTH), dtype=np.float32)

        bids_raw = depth_df["bids"].values
        asks_raw = depth_df["asks"].values
        for i in range(n):
            for j, (p, q) in enumerate(bids_raw[i][:BOOK_DEPTH]):
                bid_prices[i, j] = p
                bid_vols[i, j] = q
            for j, (p, q) in enumerate(asks_raw[i][:BOOK_DEPTH]):
                ask_prices[i, j] = p
                ask_vols[i, j] = q

        mid_prices = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0
        depth_ts = depth_df["timestamp"].values.astype(np.int64).copy()
        del depth_df, bids_raw, asks_raw  # free ~1 GB of Python objects
        gc.collect()
        _malloc_trim()  # force glibc to return the list-column pages to OS

        # Filter crossed books (bid >= ask = desync after reconnect)
        valid_book = bid_prices[:, 0] < ask_prices[:, 0]
        n_crossed = (~valid_book).sum()
        if n_crossed > 0:
            logger.info("Filtering %d crossed-book snapshots (%.2f%%)", n_crossed, n_crossed / n * 100)
            bid_prices = bid_prices[valid_book]
            bid_vols = bid_vols[valid_book]
            ask_prices = ask_prices[valid_book]
            ask_vols = ask_vols[valid_book]
            mid_prices = mid_prices[valid_book]
            depth_ts = depth_ts[valid_book]
            n = len(depth_ts)

        logger.info("Depth parsed, %d valid snapshots", n)

        # === Load and parse trades (free DataFrame ASAP) ===
        trade_df = self.load_trade_data(hours)
        trade_ts = trade_df["timestamp"].values.astype(np.int64).copy()
        trade_price = trade_df["price"].values.astype(np.float64).copy()
        trade_qty = trade_df["quantity"].values.astype(np.float64).copy()
        trade_side = trade_df["is_buyer_maker"].values.copy()
        del trade_df
        gc.collect()
        _malloc_trim()

        # === Load auxiliary data (ETH trades, funding, derivatives) ===
        eth_trade_df = self._load_parquet_dir("eth_trades", hours, dedup_cols=["timestamp", "price", "quantity"])
        eth_ts = eth_qty = eth_side = eth_price = None
        if eth_trade_df is not None and len(eth_trade_df) > 0:
            eth_ts = eth_trade_df["timestamp"].values.astype(np.int64).copy()
            eth_price = eth_trade_df["price"].values.astype(np.float64).copy()
            eth_qty = eth_trade_df["quantity"].values.astype(np.float64).copy()
            eth_side = eth_trade_df["is_buyer_maker"].values.copy()
            logger.info("ETH trades loaded: %d", len(eth_ts))
        del eth_trade_df

        funding_df = self._load_parquet_dir("funding", hours)
        funding_ts = funding_rate = None
        if funding_df is not None and len(funding_df) > 0:
            funding_ts = funding_df["timestamp"].values.astype(np.int64).copy()
            funding_rate = funding_df["funding_rate"].values.astype(np.float64).copy()
            logger.info("Funding data loaded: %d", len(funding_ts))
        del funding_df

        deriv_df = self._load_parquet_dir("derivatives", hours)
        deriv_ts = deriv_oi = deriv_ls = None
        if deriv_df is not None and len(deriv_df) > 0:
            deriv_ts = deriv_df["timestamp"].values.astype(np.int64).copy()
            deriv_oi = deriv_df["open_interest"].values.astype(np.float64).copy()
            deriv_ls = deriv_df["long_short_ratio"].values.astype(np.float64).copy()
            logger.info("Derivatives data loaded: %d", len(deriv_ts))
        del deriv_df

        # Cross-exchange trades for feature 30 (cross_exchange_momentum_500ms).
        # Each value: (timestamps_ms_int64, signed_qty_float64) where signed_qty
        # is positive for buyer-initiated and negative for seller-initiated.
        cross_ex_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ex in ("bybit", "okx", "bitget", "gateio"):
            ex_df = self._load_parquet_dir(
                f"{ex}_trades", hours,
                dedup_cols=["timestamp", "price", "quantity"],
            )
            if ex_df is None or len(ex_df) == 0:
                continue
            ex_ts = ex_df["timestamp"].values.astype(np.int64).copy()
            ex_qty = ex_df["quantity"].values.astype(np.float64).copy()
            # Recorder writes "is_seller" for these dirs (True = seller-initiated).
            ex_is_seller = ex_df["is_seller"].values.astype(bool).copy()
            # Gate.io recorder stores `quantity` as the raw Gate.io
            # futures `size` field — a **signed integer in contracts**
            # (positive=buy, negative=sell). `is_seller` is populated
            # correctly from the sign, but the magnitude is negative for
            # sells, so the `where(is_seller, -q, q)` flip below would
            # invert Gate.io's contribution to feature 30. Strip the
            # sign here; dedup happened above on the original signed
            # value, so it still disambiguates buy vs sell at same
            # (ts, price). Feature 30 only cares about the sign of the
            # net sum, so the contracts-vs-BTC unit mismatch is a no-op.
            if ex == "gateio":
                ex_qty = np.abs(ex_qty)
            ex_signed = np.where(ex_is_seller, -ex_qty, ex_qty)
            cross_ex_data[ex] = (ex_ts, ex_signed)
            logger.info("%s trades loaded: %d", ex, len(ex_ts))
            del ex_df, ex_qty, ex_is_seller
        gc.collect()
        _malloc_trim()

        tick_buy_vol = np.zeros(n, dtype=np.float32)
        tick_sell_vol = np.zeros(n, dtype=np.float32)

        t_idx = np.searchsorted(depth_ts, trade_ts, side="right") - 1
        t_idx = np.clip(t_idx, 0, n - 1)
        np.add.at(tick_buy_vol, t_idx[~trade_side], trade_qty[~trade_side].astype(np.float32))
        np.add.at(tick_sell_vol, t_idx[trade_side], trade_qty[trade_side].astype(np.float32))

        # === Sample indices with auto-step for memory ===
        total = n - WINDOW_SIZE - HORIZON
        # X_lob is ~12 KB per sample; cap at ~1.5 GB
        max_samples = 130_000
        step = max(2, 2 * ((total + max_samples - 1) // max_samples)) if total > max_samples * 2 else 2
        sample_starts = np.arange(0, total, step)
        num_samples = len(sample_starts)
        end_indices = sample_starts + WINDOW_SIZE - 1

        if step > 2:
            logger.info("Auto step=%d for memory (%d rows → %d samples)", step, n, num_samples)

        logger.info("Building %d samples (vectorized, window=%d, horizon=%d)...",
                     num_samples, WINDOW_SIZE, HORIZON)

        # === LOB tensors — write to disk via mmap (avoids OOM) ===
        lob_path = lob_output_path if lob_output_path is not None else self._data_dir / "_tmp_X_lob.npy"
        X_lob = np.lib.format.open_memmap(
            str(lob_path), mode="w+", dtype=np.float32,
            shape=(num_samples, 3, BOOK_DEPTH, WINDOW_SIZE),
        )
        offsets = np.arange(WINDOW_SIZE)
        LOB_BATCH = 5_000

        for b in range(0, num_samples, LOB_BATCH):
            e = min(b + LOB_BATCH, num_samples)
            idx = sample_starts[b:e, None] + offsets[None, :]  # (batch, 50)
            X_lob[b:e, 0] = bid_vols[idx].transpose(0, 2, 1)
            X_lob[b:e, 1] = ask_vols[idx].transpose(0, 2, 1)
            X_lob[b:e, 2, 0] = tick_buy_vol[idx]
            X_lob[b:e, 2, 1] = tick_sell_vol[idx]

        X_lob.flush()
        lob_mb = num_samples * 3 * BOOK_DEPTH * WINDOW_SIZE * 4 / 1024 / 1024
        logger.info("LOB tensors written to disk (%.1f MB), RAM free", lob_mb)

        # Re-open as read-only mmap (OS manages page cache)
        del X_lob
        X_lob = np.load(str(lob_path), mmap_mode="r")

        # === Features (vectorized) ===
        X_feat = self._calc_features_batch(
            bid_vols, ask_vols, bid_prices, ask_prices, mid_prices,
            trade_ts, trade_qty, trade_side, depth_ts, end_indices,
            trade_price=trade_price,
            eth_ts=eth_ts, eth_price=eth_price, eth_qty=eth_qty, eth_side=eth_side,
            funding_ts=funding_ts, funding_rate_arr=funding_rate,
            deriv_ts=deriv_ts, deriv_oi=deriv_oi, deriv_ls=deriv_ls,
            cross_ex_data=cross_ex_data,
        )

        # Free large parse arrays (LOB + features are done)
        del bid_vols, ask_vols, bid_prices, ask_prices
        del tick_buy_vol, tick_sell_vol
        del trade_ts, trade_price, trade_qty, trade_side, depth_ts
        del eth_ts, eth_price, eth_qty, eth_side
        del funding_ts, funding_rate, deriv_ts, deriv_oi, deriv_ls
        del cross_ex_data
        import gc; gc.collect()
        _malloc_trim()

        # === Labels — triple-barrier method (vectorized) ===
        # For each sample at time t, look at the future window [t, t+HORIZON):
        #   - LONG would win if price first reaches +TP_PCT before -SL_PCT
        #   - SHORT would win if price first reaches -TP_PCT before +SL_PCT
        # If neither side wins → FLAT.
        future_starts = sample_starts + WINDOW_SIZE
        future_win = np.lib.stride_tricks.sliding_window_view(mid_prices, HORIZON)
        future_mids = future_win[future_starts]              # (N, HORIZON)
        current_mids = mid_prices[future_starts - 1]         # (N,)

        safe = np.where(current_mids > 0, current_mids, 1.0)
        # Signed relative return per future tick, in percent — (N, HORIZON)
        rel = (future_mids - current_mids[:, None]) / safe[:, None] * 100

        # LONG: TP at +TP_PCT, SL at -SL_PCT. argmax returns the FIRST True
        # index (or 0 if all False) — guard with `.any(axis=1)` and use HORIZON
        # as a "never hit" sentinel so a never-hitting barrier loses any race.
        long_tp_hit = rel >= TP_PCT
        long_sl_hit = rel <= -SL_PCT
        long_tp_first = np.where(long_tp_hit.any(axis=1),
                                 long_tp_hit.argmax(axis=1), HORIZON)
        long_sl_first = np.where(long_sl_hit.any(axis=1),
                                 long_sl_hit.argmax(axis=1), HORIZON)

        # SHORT: TP at -TP_PCT, SL at +SL_PCT
        short_tp_hit = rel <= -TP_PCT
        short_sl_hit = rel >= SL_PCT
        short_tp_first = np.where(short_tp_hit.any(axis=1),
                                  short_tp_hit.argmax(axis=1), HORIZON)
        short_sl_first = np.where(short_sl_hit.any(axis=1),
                                  short_sl_hit.argmax(axis=1), HORIZON)

        long_wins = long_tp_first < long_sl_first
        short_wins = short_tp_first < short_sl_first

        y = np.full(num_samples, FLAT, dtype=np.int64)
        y[long_wins & ~short_wins] = UP
        y[short_wins & ~long_wins] = DOWN
        # Both directions theoretically profitable in the same window (volatile
        # whipsaw): pick whichever TP fires first — i.e. the faster profit. Tie
        # goes to LONG which keeps the labels deterministic.
        both = long_wins & short_wins
        y[both & (long_tp_first <= short_tp_first)] = UP
        y[both & (long_tp_first >  short_tp_first)] = DOWN

        # Mid prices at sample points (for backtest alignment)
        sample_mids = current_mids.copy()

        # Filter zero mid prices
        valid = current_mids > 0
        if not valid.all():
            X_lob, X_feat, y, sample_mids = X_lob[valid], X_feat[valid], y[valid], sample_mids[valid]

        counts = {UP: int((y == UP).sum()), DOWN: int((y == DOWN).sum()), FLAT: int((y == FLAT).sum())}
        logger.info(
            "Triple-barrier labels (TP=%.2f%% SL=%.2f%%): UP=%d (%.1f%%) DOWN=%d (%.1f%%) FLAT=%d (%.1f%%)",
            TP_PCT, SL_PCT,
            counts[UP], counts[UP] / len(y) * 100,
            counts[DOWN], counts[DOWN] / len(y) * 100,
            counts[FLAT], counts[FLAT] / len(y) * 100,
        )
        return X_lob, X_feat, y, sample_mids

    def _calc_features_batch(
        self,
        bid_vols: np.ndarray,
        ask_vols: np.ndarray,
        bid_prices: np.ndarray,
        ask_prices: np.ndarray,
        mid_prices: np.ndarray,
        trade_ts: np.ndarray,
        trade_qty: np.ndarray,
        trade_side: np.ndarray,
        depth_ts: np.ndarray,
        indices: np.ndarray,
        *,
        trade_price: np.ndarray | None = None,
        eth_ts=None, eth_price=None, eth_qty=None, eth_side=None,
        funding_ts=None, funding_rate_arr=None,
        deriv_ts=None, deriv_oi=None, deriv_ls=None,
        cross_ex_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Compute all NUM_FEATURES features for all sample indices at once."""
        from src.features import NUM_FEATURES
        ns = len(indices)
        feat = np.zeros((ns, NUM_FEATURES), dtype=np.float32)

        # --- Pre-compute full-array quantities ---
        bv5 = bid_vols[:, :5].sum(axis=1)     # (n,)
        av5 = ask_vols[:, :5].sum(axis=1)
        total_vol = bv5 + av5
        imb_all = np.divide(bv5 - av5, total_vol, out=np.zeros_like(total_vol), where=total_vol > 0)

        # [0] OFI
        d_bid = np.diff(bid_vols[:, 0], prepend=bid_vols[0, 0])
        d_ask = np.diff(ask_vols[:, 0], prepend=ask_vols[0, 0])
        feat[:, 0] = (d_bid - d_ask)[indices]

        # [1] Imbalance ratio
        feat[:, 1] = imb_all[indices]

        # [2] Imbalance velocity
        m5 = indices >= 5
        feat[m5, 2] = imb_all[indices[m5]] - imb_all[indices[m5] - 5]

        # [3] Spread
        feat[:, 3] = (ask_prices[:, 0] - bid_prices[:, 0])[indices]

        # [4] Depth ratio L5
        av5_i = av5[indices]
        feat[:, 4] = np.where(av5_i > 0, bv5[indices] / av5_i, 10.0)

        # [5] Large order presence
        large_bid = np.any(bid_vols[:, :5] > 100, axis=1)
        large_ask = np.any(ask_vols[:, :5] > 100, axis=1)
        feat[:, 5] = (large_bid[indices] | large_ask[indices]).astype(np.float32)

        # --- Trade flow via cumulative sums (O(n) precompute, O(1) per query) ---
        cum_buy = np.zeros(len(trade_ts) + 1, dtype=np.float64)
        cum_sell = np.zeros(len(trade_ts) + 1, dtype=np.float64)
        cum_large = np.zeros(len(trade_ts) + 1, dtype=np.float64)
        cum_buy[1:] = np.cumsum(trade_qty * ~trade_side)
        cum_sell[1:] = np.cumsum(trade_qty * trade_side)
        cum_large[1:] = np.cumsum(trade_qty > 10)

        sample_ts = depth_ts[indices]
        right = np.searchsorted(trade_ts, sample_ts, side="right")

        # [6] Trade flow imbalance (5s)
        left_5s = np.searchsorted(trade_ts, sample_ts - 5000, side="left")
        buys_5s = cum_buy[right] - cum_buy[left_5s]
        sells_5s = cum_sell[right] - cum_sell[left_5s]
        total_5s = buys_5s + sells_5s
        feat[:, 6] = np.divide(buys_5s - sells_5s, total_5s,
                               out=np.zeros(ns, dtype=np.float64), where=total_5s > 0).astype(np.float32)

        # [7] Trade intensity (1s)
        left_1s = np.searchsorted(trade_ts, sample_ts - 1000, side="left")
        feat[:, 7] = (right - left_1s).astype(np.float32)

        # [8] Large trade in 5s window
        feat[:, 8] = np.where(cum_large[right] - cum_large[left_5s] > 0, 1.0, 0.0)

        # [9] CVD (30s)
        left_30s = np.searchsorted(trade_ts, sample_ts - 30000, side="left")
        feat[:, 9] = (cum_buy[right] - cum_buy[left_30s]
                       - cum_sell[right] + cum_sell[left_30s]).astype(np.float32)

        # [10] Volatility 1s (std of 10 returns)
        safe_mid = np.where(mid_prices[:-1] > 0, mid_prices[:-1], 1.0)
        returns_all = np.diff(mid_prices) / safe_mid
        m10 = indices >= 10
        if m10.any():
            ret_win = np.lib.stride_tricks.sliding_window_view(returns_all, 10)
            vol_all = np.asarray(ret_win).std(axis=1)
            adj = np.clip(indices[m10] - 10, 0, len(vol_all) - 1)
            feat[m10, 10] = vol_all[adj].astype(np.float32)

        # [11] VWAP deviation — (mid - VWAP_60s) / VWAP_60s
        # Approximate VWAP as rolling mean of mid_prices over 60s window (vectorized)
        left_60s = np.searchsorted(depth_ts, sample_ts - 60000, side="left")
        cum_mid = np.zeros(len(mid_prices) + 1, dtype=np.float64)
        cum_mid[1:] = np.cumsum(mid_prices)
        hi = indices + 1
        lo = np.clip(left_60s, 0, len(mid_prices))
        counts = hi - lo
        safe_counts = np.where(counts > 0, counts, 1)
        vwap = (cum_mid[hi] - cum_mid[lo]) / safe_counts
        feat[:, 11] = np.where(
            (counts > 0) & (vwap > 0),
            (mid_prices[indices] - vwap) / vwap, 0.0,
        ).astype(np.float32)

        # [12] Momentum 5s
        m50 = indices >= 50
        prev50 = mid_prices[indices[m50] - 50]
        feat[m50, 12] = np.where(prev50 > 0,
                                  (mid_prices[indices[m50]] - prev50) / prev50, 0).astype(np.float32)

        # [13] Funding rate — from funding parquet (1s resolution)
        if funding_ts is not None and len(funding_ts) > 0:
            # For each sample, find the latest funding rate before that timestamp
            fund_idx = np.searchsorted(funding_ts, sample_ts, side="right") - 1
            fund_idx = np.clip(fund_idx, 0, len(funding_rate_arr) - 1)
            feat[:, 13] = funding_rate_arr[fund_idx].astype(np.float32)

        # [14-16] ETH leading signals — from eth_trades parquet
        if eth_ts is not None and len(eth_ts) > 0:
            # ETH cumulative volumes for flow features
            eth_cum_buy = np.zeros(len(eth_ts) + 1, dtype=np.float64)
            eth_cum_sell = np.zeros(len(eth_ts) + 1, dtype=np.float64)
            eth_cum_buy[1:] = np.cumsum(eth_qty * ~eth_side)
            eth_cum_sell[1:] = np.cumsum(eth_qty * eth_side)

            # ETH cumulative price*qty for VWAP / mid-price proxy
            eth_cum_pv = np.zeros(len(eth_ts) + 1, dtype=np.float64)
            eth_cum_pv[1:] = np.cumsum(eth_price * eth_qty)
            eth_cum_qty = np.zeros(len(eth_ts) + 1, dtype=np.float64)
            eth_cum_qty[1:] = np.cumsum(eth_qty)

            eth_right = np.searchsorted(eth_ts, sample_ts, side="right")

            # [14] eth_momentum_1s — ETH price change over 1 second
            eth_left_1s = np.searchsorted(eth_ts, sample_ts - 1000, side="left")
            # VWAP in [left, right) as proxy for price at each boundary
            eth_qty_now = eth_cum_qty[eth_right] - eth_cum_qty[eth_left_1s]
            eth_pv_now = eth_cum_pv[eth_right] - eth_cum_pv[eth_left_1s]
            eth_vwap_1s = np.divide(eth_pv_now, eth_qty_now,
                                     out=np.zeros(ns, dtype=np.float64), where=eth_qty_now > 0)
            # Price 1s ago
            eth_left_2s = np.searchsorted(eth_ts, sample_ts - 2000, side="left")
            eth_qty_prev = eth_cum_qty[eth_left_1s] - eth_cum_qty[eth_left_2s]
            eth_pv_prev = eth_cum_pv[eth_left_1s] - eth_cum_pv[eth_left_2s]
            eth_vwap_prev = np.divide(eth_pv_prev, eth_qty_prev,
                                       out=np.zeros(ns, dtype=np.float64), where=eth_qty_prev > 0)
            feat[:, 14] = np.where(
                (eth_vwap_1s > 0) & (eth_vwap_prev > 0),
                (eth_vwap_1s - eth_vwap_prev) / eth_vwap_prev, 0.0,
            ).astype(np.float32)

            # [15] eth_ofi — approximated from ETH trade flow (buy - sell imbalance)
            eth_left_500ms = np.searchsorted(eth_ts, sample_ts - 500, side="left")
            eth_buys = eth_cum_buy[eth_right] - eth_cum_buy[eth_left_500ms]
            eth_sells = eth_cum_sell[eth_right] - eth_cum_sell[eth_left_500ms]
            eth_total = eth_buys + eth_sells
            feat[:, 15] = np.divide(eth_buys - eth_sells, eth_total,
                                     out=np.zeros(ns, dtype=np.float64), where=eth_total > 0).astype(np.float32)

            # [16] eth_leading_signal — BTC/ETH ratio deviation
            # Current ETH price (VWAP over last 1s)
            eth_mid = eth_vwap_1s
            btc_mid = mid_prices[indices]
            ratio = np.divide(btc_mid, eth_mid,
                              out=np.zeros(ns, dtype=np.float64), where=eth_mid > 0)
            # Rolling mean of ratio over 30s (300 ticks in depth)
            # Use cumsum of ratio for efficient rolling mean
            if ratio.sum() > 0:
                # Simple: compare current ratio to mean of all ratios
                ratio_mean = ratio[ratio > 0].mean() if (ratio > 0).any() else 1.0
                feat[:, 16] = np.where(
                    ratio > 0, (ratio - ratio_mean) / (ratio_mean + 1e-10), 0.0,
                ).astype(np.float32)

        # [17-19] Derivatives — from derivatives parquet (15s resolution)
        if deriv_ts is not None and len(deriv_ts) > 1:
            d_idx = np.searchsorted(deriv_ts, sample_ts, side="right") - 1
            d_idx = np.clip(d_idx, 0, len(deriv_oi) - 1)

            # [17] OI delta — change vs previous poll
            d_idx_prev = np.clip(d_idx - 1, 0, len(deriv_oi) - 1)
            oi_now = deriv_oi[d_idx]
            oi_prev = deriv_oi[d_idx_prev]
            feat[:, 17] = np.where(
                oi_prev > 0, (oi_now - oi_prev) / oi_prev, 0.0,
            ).astype(np.float32)

            # [18] L/S ratio
            feat[:, 18] = deriv_ls[d_idx].astype(np.float32)

            # [19] Liquidation proximity — heuristic from L/S ratio
            ls = deriv_ls[d_idx]
            btc_mid = mid_prices[indices]
            cluster_pct = 0.015  # ~1.5% for x50-x100 leverage
            liq_prox = np.zeros(ns, dtype=np.float32)
            # More longs → cluster below (negative = danger below)
            long_heavy = ls > 1.2
            liq_prox[long_heavy] = -cluster_pct
            # More shorts → cluster above (positive = danger above)
            short_heavy = ls < 0.8
            liq_prox[short_heavy] = cluster_pct
            feat[:, 19] = liq_prox

        # [20] Spoof score approximation
        m25 = indices >= 25
        has_large = feat[:, 5] > 0
        pc = np.abs(mid_prices[indices] - mid_prices[np.maximum(indices - 25, 0)])
        feat[:, 20] = np.where(has_large & m25 & (pc < 0.10), 1.0, 0.0)

        # [21] Volatility ratio (current / 30-tick rolling average)
        if m10.any():
            vol_win_30 = np.lib.stride_tricks.sliding_window_view(vol_all, 30) if len(vol_all) >= 30 else None
            if vol_win_30 is not None:
                vol_mean_all = vol_win_30.mean(axis=1)  # (n - 10 - 29,)
                m_vr = indices >= 40  # need 10 for vol + 30 for rolling mean
                adj_vr = np.clip(indices[m_vr] - 40, 0, len(vol_mean_all) - 1)
                adj_v = np.clip(indices[m_vr] - 10, 0, len(vol_all) - 1)
                feat[m_vr, 21] = np.where(
                    vol_mean_all[adj_vr] > 0,
                    vol_all[adj_v] / (vol_mean_all[adj_vr] + 1e-10),
                    1.0,
                ).astype(np.float32)

        # [22] Trade intensity ratio (current / 30-tick rolling average)
        # Per-tick trade count
        tick_intensity = np.zeros(len(depth_ts), dtype=np.float32)
        t_tick_idx = np.searchsorted(depth_ts, trade_ts, side="right") - 1
        t_tick_idx = np.clip(t_tick_idx, 0, len(depth_ts) - 1)
        np.add.at(tick_intensity, t_tick_idx, 1.0)
        if len(tick_intensity) >= 40:
            int_win = np.lib.stride_tricks.sliding_window_view(tick_intensity, 10)
            curr_int = int_win.sum(axis=1)  # intensity over 10 ticks (1s)
            if len(curr_int) >= 30:
                int_mean_win = np.lib.stride_tricks.sliding_window_view(curr_int, 30)
                int_mean_all = int_mean_win.mean(axis=1)
                m_ir = indices >= 40
                adj_ci = np.clip(indices[m_ir] - 10, 0, len(curr_int) - 1)
                adj_im = np.clip(indices[m_ir] - 40, 0, len(int_mean_all) - 1)
                feat[m_ir, 22] = np.where(
                    int_mean_all[adj_im] > 0,
                    curr_int[adj_ci] / (int_mean_all[adj_im] + 1e-10),
                    1.0,
                ).astype(np.float32)

        # [23] Hurst exponent (R/S, batched for memory)
        log_ret = np.diff(np.log(mid_prices + 1e-10))
        m100 = indices >= 100
        if m100.any() and len(log_ret) >= 100:
            hurst_win = np.lib.stride_tricks.sliding_window_view(log_ret, 100)
            n_hw = len(hurst_win)
            all_hurst = np.full(n_hw, 0.5, dtype=np.float32)
            H_BATCH = 50_000
            for hb in range(0, n_hw, H_BATCH):
                he = min(hb + H_BATCH, n_hw)
                chunk = np.array(hurst_win[hb:he])
                means = chunk.mean(axis=1)
                deviate = np.cumsum(chunk - means[:, None], axis=1)
                r = deviate.max(axis=1) - deviate.min(axis=1)
                s = chunk.std(axis=1)
                all_hurst[hb:he] = np.where(
                    s > 0,
                    np.clip(np.log(r / (s + 1e-10)) / np.log(100), 0, 1),
                    0.5,
                ).astype(np.float32)
            adj = np.clip(indices[m100] - 100, 0, n_hw - 1)
            feat[m100, 23] = all_hurst[adj]
        feat[~m100, 23] = 0.5

        # [24] Sweep intensity
        m1 = indices >= 1
        tick_size = 0.10
        bid_jump = np.abs(bid_prices[indices[m1], 0] - bid_prices[indices[m1] - 1, 0]) / tick_size
        ask_jump = np.abs(ask_prices[indices[m1], 0] - ask_prices[indices[m1] - 1, 0]) / tick_size
        feat[m1, 24] = np.maximum(0, np.maximum(bid_jump, ask_jump) - 1).astype(np.float32)

        # [25] Cancellation rate diff (ask_cancel - bid_cancel over ~10 ticks ≈ 1s)
        # Cancel = volume drop at a level without trade. Approximate: Δvol < 0 at each level.
        bid_vol_diff = np.diff(bid_vols[:, :5], axis=0)  # (n-1, 5)
        ask_vol_diff = np.diff(ask_vols[:, :5], axis=0)  # (n-1, 5)
        # Negative diff = volume removed (cancelled or filled)
        bid_cancel_tick = np.maximum(0, -bid_vol_diff).sum(axis=1)  # (n-1,)
        ask_cancel_tick = np.maximum(0, -ask_vol_diff).sum(axis=1)
        # Pad to length n
        bid_cancel_tick = np.concatenate([[0], bid_cancel_tick])
        ask_cancel_tick = np.concatenate([[0], ask_cancel_tick])
        # Rolling sum over 10 ticks (~1 second)
        if len(bid_cancel_tick) >= 10:
            bc_win = np.lib.stride_tricks.sliding_window_view(bid_cancel_tick, 10).sum(axis=1)
            ac_win = np.lib.stride_tricks.sliding_window_view(ask_cancel_tick, 10).sum(axis=1)
            m_cr = indices >= 10
            adj_cr = np.clip(indices[m_cr] - 10, 0, len(bc_win) - 1)
            feat[m_cr, 25] = (ac_win[adj_cr] - bc_win[adj_cr]).astype(np.float32)

        # [26-29] Multi-timeframe OFI
        # Raw OFI per tick already computed as (d_bid - d_ask)
        ofi_raw = (d_bid - d_ask)  # (n,)
        # [26] OFI 1s (sum over 10 ticks)
        if len(ofi_raw) >= 10:
            ofi_1s_all = np.lib.stride_tricks.sliding_window_view(ofi_raw, 10).sum(axis=1)
            m_o1 = indices >= 10
            feat[m_o1, 26] = ofi_1s_all[np.clip(indices[m_o1] - 10, 0, len(ofi_1s_all) - 1)].astype(np.float32)
        # [27] OFI 5s (sum over 50 ticks)
        if len(ofi_raw) >= 50:
            ofi_5s_all = np.lib.stride_tricks.sliding_window_view(ofi_raw, 50).sum(axis=1)
            m_o5 = indices >= 50
            feat[m_o5, 27] = ofi_5s_all[np.clip(indices[m_o5] - 50, 0, len(ofi_5s_all) - 1)].astype(np.float32)
        # [28] OFI 30s (sum over 300 ticks)
        if len(ofi_raw) >= 300:
            ofi_30s_all = np.lib.stride_tricks.sliding_window_view(ofi_raw, 300).sum(axis=1)
            m_o30 = indices >= 300
            feat[m_o30, 28] = ofi_30s_all[np.clip(indices[m_o30] - 300, 0, len(ofi_30s_all) - 1)].astype(np.float32)
        # [29] OFI divergence: ofi_1s - ofi_30s when signs differ
        if len(ofi_raw) >= 300:
            m_div = indices >= 300
            short = feat[m_div, 26]
            long_ = feat[m_div, 28]
            feat[m_div, 29] = np.where(short * long_ < 0, short - long_, 0.0).astype(np.float32)

        # [30] Cross-exchange momentum (500ms): count of exchanges
        # (Bybit, OKX, Bitget, Gate.io) whose net signed volume in the last
        # 500ms strictly exceeds zero. Range: 0..4. Computed only over
        # exchanges that have data in this window — missing feeds contribute 0.
        if cross_ex_data:
            ex_count = np.zeros(ns, dtype=np.float32)
            for ex_name, (ex_ts, ex_signed) in cross_ex_data.items():
                if len(ex_ts) == 0:
                    continue
                # Cumulative signed volume — O(n) once, O(1) lookup per sample
                cum = np.zeros(len(ex_ts) + 1, dtype=np.float64)
                cum[1:] = np.cumsum(ex_signed)
                right = np.searchsorted(ex_ts, sample_ts, side="right")
                left = np.searchsorted(ex_ts, sample_ts - 500, side="left")
                net = cum[right] - cum[left]  # (ns,)
                ex_count += (net > 0).astype(np.float32)
            feat[:, 30] = ex_count

        # === Microstructure features (Lever 5) — must match features.py ===
        from src.features import QUEUE_DECAY_ALPHA
        # [31] queue_pressure — EMA(ask L1 decay) - EMA(bid L1 decay).
        # `decay` is the positive tick-to-tick drop in best-level volume; the
        # EMA is propagated through the full depth_ts array so each sample
        # reads its lookup-time value, exactly like the realtime path.
        bid_l1 = bid_vols[:, 0].astype(np.float64)
        ask_l1 = ask_vols[:, 0].astype(np.float64)
        bid_decay = np.maximum(0.0, bid_l1[:-1] - bid_l1[1:])  # (n-1,)
        ask_decay = np.maximum(0.0, ask_l1[:-1] - ask_l1[1:])
        bid_decay = np.concatenate([[0.0], bid_decay])         # align to (n,)
        ask_decay = np.concatenate([[0.0], ask_decay])
        a = QUEUE_DECAY_ALPHA
        # Vectorised first-order EMA via Horner-style accumulator. Stay in a
        # python loop because numpy has no native EMA, and (n) is large but
        # the inner op is constant-time.
        bid_ema = np.empty_like(bid_decay)
        ask_ema = np.empty_like(ask_decay)
        b_acc = 0.0
        s_acc = 0.0
        for i in range(len(bid_decay)):
            b_acc = a * bid_decay[i] + (1 - a) * b_acc
            s_acc = a * ask_decay[i] + (1 - a) * s_acc
            bid_ema[i] = b_acc
            ask_ema[i] = s_acc
        feat[:, 31] = (ask_ema[indices] - bid_ema[indices]).astype(np.float32)

        # [32] top3_asymmetry — (top3_bid/top20_bid) - (top3_ask/top20_ask).
        top3_bid = bid_vols[:, :3].sum(axis=1).astype(np.float64)
        top20_bid = bid_vols[:, :].sum(axis=1).astype(np.float64)
        top3_ask = ask_vols[:, :3].sum(axis=1).astype(np.float64)
        top20_ask = ask_vols[:, :].sum(axis=1).astype(np.float64)
        bid_share = top3_bid / (top20_bid + 1e-9)
        ask_share = top3_ask / (top20_ask + 1e-9)
        feat[:, 32] = (bid_share - ask_share)[indices].astype(np.float32)

        # [33] effective_spread_ratio — EMA of |last_trade_price - mid|/spread.
        # For each depth tick we look up the most recent trade (≤ depth_ts);
        # if no trade exists yet (start of session) the per-tick ratio is 0,
        # which matches FeatureEngine returning the EMA's prior value.
        spread_arr = np.maximum(ask_prices[:, 0] - bid_prices[:, 0], 1e-9)
        eff_ratio_per_tick = np.zeros_like(mid_prices)
        if trade_price is not None and len(trade_ts) > 0:
            last_trade_idx = np.searchsorted(trade_ts, depth_ts, side="right") - 1
            valid_lt = last_trade_idx >= 0
            lt_price = np.where(valid_lt, trade_price[np.clip(last_trade_idx, 0, len(trade_price) - 1)], mid_prices)
            eff_ratio_per_tick = np.where(
                valid_lt,
                np.abs(lt_price - mid_prices) / spread_arr,
                0.0,
            )
        # Vectorised EMA over the full depth axis (mirrors realtime semantics)
        eff_ema = np.empty_like(eff_ratio_per_tick)
        e_acc = 0.0
        for i in range(len(eff_ratio_per_tick)):
            e_acc = a * eff_ratio_per_tick[i] + (1 - a) * e_acc
            eff_ema[i] = e_acc
        feat[:, 33] = eff_ema[indices].astype(np.float32)

        return feat

    # ---- Training ----

    def train_cnn(
        self,
        X_lob: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 0.001,
        patience: int = 5,
        n_jobs: int = 1,
        warm_start_path: Path | None = None,
    ) -> LOBEncoder:
        # Constrain PyTorch to the requested thread count BEFORE building the
        # model/DataLoaders so every subsequent tensor op honours the limit.
        # `torch.set_num_threads` is process-global, so a retrain cycle on the
        # live server stays within its budget even when called from main.py.
        torch.set_num_threads(_pytorch_threads(n_jobs))
        # Faster matmul kernels — accuracy impact is negligible for our task,
        # wall time drops ~5-10% on AVX2 CPUs.
        torch.set_float32_matmul_precision("medium")

        n = len(X_lob)
        split = int(n * (1 - val_split))

        # .copy() handles mmap read-only arrays (torch needs writable memory).
        # channels_last NHWC layout is faster on oneDNN conv kernels (Skylake
        # AVX2). Applied to both train and val tensors; model is also placed
        # in channels_last so forward pass runs in NHWC end-to-end.
        X_train = torch.from_numpy(np.array(X_lob[:split])).contiguous(
            memory_format=torch.channels_last
        )
        y_train = torch.from_numpy(y[:split].copy() if isinstance(y, np.memmap) else y[:split])
        X_val = torch.from_numpy(np.array(X_lob[split:])).contiguous(
            memory_format=torch.channels_last
        )
        y_val = torch.from_numpy(y[split:].copy() if isinstance(y, np.memmap) else y[split:])

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        # CNN + classification head for training
        encoder = LOBEncoder().to(memory_format=torch.channels_last)
        head = nn.Linear(64, 3)

        # Warm start from a previous encoder checkpoint if provided. Used by
        # daily retrain cycles in production — 5 fine-tune epochs on warm
        # weights converge to better-or-equal val_loss than 50 cold epochs,
        # and at a fraction of the wall time.
        if warm_start_path is not None and warm_start_path.exists():
            state = torch.load(warm_start_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state)
            logger.info("CNN warm-start loaded from %s — reducing epochs to 5", warm_start_path)
            epochs = 5

        # torch.compile fuses op graph and removes Python dispatch overhead
        # between conv / bn / relu / pool. On CPU with this tiny model the win
        # is ~1.5-2x per epoch after a one-time ~30s graph warmup. "reduce-
        # overhead" mode is tuned for Python-bound workloads (small ops, large
        # batches) — which is exactly our shape after the batch_size bump.
        encoder_compiled = torch.compile(encoder, mode="reduce-overhead")
        head_compiled = torch.compile(head, mode="reduce-overhead")

        params = list(encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        logger.info("Training CNN: %d train, %d val, epochs=%d", split, n - split, epochs)

        for epoch in range(epochs):
            encoder.train()
            head.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for xb, yb in train_dl:
                optimizer.zero_grad()
                emb = encoder_compiled(xb)
                logits = head_compiled(emb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(xb)
                train_correct += (logits.argmax(1) == yb).sum().item()
                train_total += len(xb)

            # Validation
            encoder.eval()
            head.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for xb, yb in val_dl:
                    emb = encoder_compiled(xb)
                    logits = head_compiled(emb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * len(xb)
                    val_correct += (logits.argmax(1) == yb).sum().item()
                    val_total += len(xb)

            train_loss /= train_total
            val_loss /= val_total
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
                epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        encoder.load_state_dict(best_state)
        encoder.eval()
        return encoder

    def extract_embeddings(
        self, encoder: LOBEncoder, X_lob: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        encoder.eval()
        embeddings = []
        for i in range(0, len(X_lob), batch_size):
            xb = torch.from_numpy(np.array(X_lob[i:i + batch_size]))
            with torch.no_grad():
                emb = encoder(xb).numpy()
            embeddings.append(emb)
        return np.vstack(embeddings)

    @staticmethod
    def _ensemble_proba(
        xgb_models: list[xgb.Booster],
        lgb_model: object,
        logreg: object,
        top5_features: list[int],
        X: np.ndarray,
    ) -> np.ndarray:
        """Average predicted probabilities across all 5 ensemble members.

        Returns shape (len(X), 3). Used both for fitting isotonic calibrators
        in `train_ensemble` and at runtime in `HybridModel.predict`. Keeping
        this in one place guarantees the calibration target matches the
        production input distribution.
        """
        n = len(X)
        acc = np.zeros((n, 3), dtype=np.float64)
        n_models = 0
        for m in xgb_models:
            p = m.predict(xgb.DMatrix(X))
            if p.ndim == 1:
                p = p.reshape(-1, 3)
            acc += p
            n_models += 1
        if lgb_model is not None:
            p = lgb_model.predict(X)
            if p.ndim == 1:
                p = p.reshape(-1, 3)
            acc += p
            n_models += 1
        if logreg is not None and top5_features:
            acc += logreg.predict_proba(X[:, top5_features])
            n_models += 1
        return acc / max(n_models, 1)

    def train_ensemble(
        self,
        embeddings: np.ndarray,
        X_feat: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        n_jobs: int = 1,
    ) -> tuple[list[xgb.Booster], object, object, list[int], list]:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.utils.class_weight import compute_sample_weight

        threads = _xgb_lgb_threads(n_jobs)
        logger.info("Ensemble thread budget: n_jobs=%d → xgb/lgb threads=%d",
                    n_jobs, threads if threads > 0 else os.cpu_count() or 0)

        X = np.hstack([embeddings, X_feat])
        n = len(X)
        # Time-aware split with a gap (Lever 2b): each sample at index t has a
        # label horizon of HORIZON ticks and a feature window of WINDOW_SIZE
        # ticks. Without a gap the last train sample's label window overlaps
        # the first val sample's feature window, leaking ~65 sec of future
        # state into training and inflating val metrics by 1-2 pp.
        split = int(n * (1 - val_split))
        gap = HORIZON + WINDOW_SIZE  # 650 ticks ≈ 65 sec
        train_end = split - gap
        if train_end <= 0:
            raise ValueError(
                f"Not enough samples for time-gap split: n={n}, val_split={val_split}, "
                f"split={split}, gap={gap}. Need at least {gap + 100} samples. "
                "Collect more data or reduce val_split."
            )
        X_train, X_val = X[:train_end], X[split:]
        y_train, y_val = y[:train_end], y[split:]

        # Balanced sample weights (Lever 2a) — without these the FLAT class
        # (60-70% of labels) dominates the loss and the model collapses to a
        # FLAT detector with near-random accuracy on the only labels we trade.
        w_train = compute_sample_weight("balanced", y_train)
        w_val = compute_sample_weight("balanced", y_val)

        logger.info(
            "Train/val split: train=%d (0..%d) val=%d (%d..%d) gap=%d",
            len(X_train), train_end, len(X_val), split, n, gap,
        )

        # --- 3 XGBoost with different seeds, each on random 80% of training rows ---
        xgb_models = []
        xgb_importances = []
        for seed in [42, 123, 456]:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_train), size=int(len(X_train) * 0.8), replace=False)
            dtrain = xgb.DMatrix(X_train[idx], label=y_train[idx],
                                 weight=w_train[idx])
            dval_dm = xgb.DMatrix(X_val, label=y_val, weight=w_val)
            params = {
                "objective": "multi:softprob", "num_class": 3, "max_depth": 6,
                "learning_rate": 0.05, "min_child_weight": 50, "subsample": 0.8,
                "colsample_bytree": 0.8, "eval_metric": "mlogloss", "verbosity": 0,
                "seed": seed, "nthread": threads,
            }
            logger.info("Training XGBoost (seed=%d): %d train (80%%), %d val",
                         seed, len(idx), len(X_val))
            # 500 rounds + patience 50: noisier val losses on financial data
            # need more headroom for early-stopping to find a stable optimum.
            model = xgb.train(
                params, dtrain, num_boost_round=500,
                evals=[(dval_dm, "val")], early_stopping_rounds=50, verbose_eval=0,
            )
            xgb_models.append(model)
            scores = model.get_score(importance_type="gain")
            imp = np.zeros(X.shape[1])
            for k, v in scores.items():
                imp[int(k[1:])] = v  # feature names are f0, f1, ...
            xgb_importances.append(imp)

            # Evaluate individual XGBoost (Lever 2c: report trade-WR, not just
            # accuracy — accuracy is dominated by FLATs and hides whether the
            # model is profitable on the labels we'd actually trade).
            preds = model.predict(dval_dm)
            y_pred = preds.argmax(axis=1)
            acc = accuracy_score(y_val, y_pred)
            self._log_trade_wr(f"XGBoost (seed={seed})", y_val, y_pred, acc)

        # Average importance, get top-5 from hand-crafted features (indices 64+)
        avg_imp = np.mean(xgb_importances, axis=0)
        hand_start = embeddings.shape[1]  # 64
        hand_imp = avg_imp[hand_start:]
        top5_local = np.argsort(hand_imp)[-5:][::-1]
        top5_global = (top5_local + hand_start).tolist()
        logger.info("Top-5 hand-crafted features: %s (importance: %s)",
                     top5_local.tolist(), hand_imp[top5_local].tolist())

        # --- 1 LightGBM ---
        import lightgbm as lgb
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
        lgb_val_ds = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=lgb_train)
        lgb_params = {
            "objective": "multiclass", "num_class": 3, "max_depth": 6,
            "learning_rate": 0.05, "min_child_samples": 50, "subsample": 0.8,
            "colsample_bytree": 0.8, "metric": "multi_logloss", "verbosity": -1,
            "num_threads": threads,
        }
        logger.info("Training LightGBM: %d train, %d val", len(X_train), len(X_val))
        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_val_ds],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        # Evaluate LightGBM
        lgb_preds = lgb_model.predict(X_val)
        lgb_y_pred = lgb_preds.argmax(axis=1)
        lgb_acc = accuracy_score(y_val, lgb_y_pred)
        self._log_trade_wr("LightGBM", y_val, lgb_y_pred, lgb_acc)

        # --- 1 LogisticRegression on top-5 features ---
        from sklearn.linear_model import LogisticRegression
        # `multi_class` keyword removed: sklearn ≥1.5 always uses multinomial
        # for multiclass + lbfgs and warns if you pass it explicitly.
        # sklearn uses `n_jobs=None` ↔ 1 thread, `n_jobs=-1` ↔ all cores.
        # Pass through the raw `n_jobs` here so the sklearn convention applies.
        logreg = LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced",
            n_jobs=n_jobs if n_jobs != 0 else 1,
        )
        logger.info("Training LogisticRegression on top-5 features: %s", top5_global)
        logreg.fit(X_train[:, top5_global], y_train)

        # Evaluate LogisticRegression
        lr_y_pred = logreg.predict(X_val[:, top5_global])
        lr_acc = accuracy_score(y_val, lr_y_pred)
        self._log_trade_wr("LogisticRegression", y_val, lr_y_pred, lr_acc)

        # --- Evaluate full ensemble on val ---
        ensemble_votes = np.zeros((len(X_val), 3), dtype=np.int32)
        for xgb_m in xgb_models:
            preds = xgb_m.predict(xgb.DMatrix(X_val))
            ensemble_votes[np.arange(len(X_val)), preds.argmax(axis=1)] += 1
        ensemble_votes[np.arange(len(X_val)), lgb_preds.argmax(axis=1)] += 1
        lr_proba = logreg.predict_proba(X_val[:, top5_global])
        ensemble_votes[np.arange(len(X_val)), lr_proba.argmax(axis=1)] += 1
        ensemble_pred = ensemble_votes.argmax(axis=1)
        ens_acc = accuracy_score(y_val, ensemble_pred)
        self._log_trade_wr("Ensemble (5-model majority)", y_val, ensemble_pred, ens_acc)
        logger.info("\n%s", classification_report(
            y_val, ensemble_pred, target_names=["UP", "DOWN", "FLAT"],
        ))

        # --- Lever 4: isotonic calibration of ensemble probabilities ---
        # Use the chronological tail of train as a calibration set so the val
        # set stays a fully held-out evaluation. Each class gets its own
        # IsotonicRegression mapping (raw_proba → empirical hit rate). At
        # runtime, calibrated[i] = isotonic[i].predict(raw[i]); we then
        # renormalise so the result is a proper probability distribution.
        calibrators = self._fit_calibrators(
            xgb_models, lgb_model, logreg, top5_global, X_train, y_train,
            X_val=X_val, y_val=y_val,
        )

        return xgb_models, lgb_model, logreg, top5_global, calibrators

    def _fit_calibrators(
        self,
        xgb_models: list[xgb.Booster],
        lgb_model: object,
        logreg: object,
        top5_global: list[int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> list:
        """Fit one IsotonicRegression per class on the tail of train.

        Why the tail of train and not val: val is the held-out report card,
        and we don't want calibration to leak its statistics into the metric.
        Why not retrain models on train\\cal: with 130k samples and 3 XGBoost
        + 1 LGB + 1 LR the marginal lift from another 13k rows is tiny, and
        re-training would double the runtime budget.
        """
        from sklearn.isotonic import IsotonicRegression

        cal_size = max(int(len(X_train) * 0.10), 200)
        if cal_size >= len(X_train):
            logger.warning(
                "Calibration skipped: train set %d <= cal_size %d",
                len(X_train), cal_size,
            )
            return [None, None, None]

        X_cal = X_train[-cal_size:]
        y_cal = y_train[-cal_size:]

        raw_cal = self._ensemble_proba(
            xgb_models, lgb_model, logreg, top5_global, X_cal,
        )

        calibrators: list = []
        for cls in range(3):
            y_binary = (y_cal == cls).astype(np.float64)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                # All-zero or all-one columns make IsotonicRegression
                # degenerate; fall through to identity (None).
                logger.warning(
                    "Calibration class %d has no spread (sum=%d/%d) — "
                    "using identity mapping", cls, int(y_binary.sum()), len(y_binary),
                )
                calibrators.append(None)
                continue
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(raw_cal[:, cls], y_binary)
            calibrators.append(ir)

        # Diagnostic: report calibrated trade-WR on val (if provided) so we
        # can see whether calibration changed the picture.
        if X_val is not None and y_val is not None:
            raw_val = self._ensemble_proba(
                xgb_models, lgb_model, logreg, top5_global, X_val,
            )
            cal_val = self._apply_calibrators(calibrators, raw_val)
            cal_pred = cal_val.argmax(axis=1)
            acc = accuracy_score(y_val, cal_pred)
            self._log_trade_wr("Calibrated ensemble (avg-proba)", y_val, cal_pred, acc)

        return calibrators

    @staticmethod
    def _apply_calibrators(calibrators: list, raw: np.ndarray) -> np.ndarray:
        """Apply per-class isotonic calibration to a (N, 3) raw-proba matrix.

        Calibrators may be None for classes that were degenerate at fit time;
        those columns pass through unchanged. Each row is renormalised to a
        proper probability distribution after the per-class transform.
        """
        cal = raw.copy()
        for cls in range(3):
            ir = calibrators[cls] if cls < len(calibrators) else None
            if ir is None:
                continue
            cal[:, cls] = ir.predict(raw[:, cls])
        row_sums = cal.sum(axis=1, keepdims=True) + 1e-9
        return cal / row_sums

    @staticmethod
    def _log_trade_wr(
        label: str, y_true: np.ndarray, y_pred: np.ndarray, acc: float,
    ) -> None:
        """Log accuracy + trade-WR.

        Trade-WR = of all non-FLAT predictions, what fraction match the true
        label. This is the metric the bot will see in production: a model with
        acc=0.52, trade-WR=0.43 is a money-loser; acc=0.45, trade-WR=0.57 is
        excellent. Accuracy alone hides this because FLAT dominates labels.
        """
        trade_mask = y_pred != FLAT
        n_trades = int(trade_mask.sum())
        n_total = int(len(y_pred))
        if n_trades:
            wr = float((y_pred[trade_mask] == y_true[trade_mask]).mean())
            logger.info(
                "%s val: acc=%.4f | trade-WR=%.4f on %d/%d (%.1f%% trade rate)",
                label, acc, wr, n_trades, n_total, 100 * n_trades / n_total,
            )
        else:
            logger.info(
                "%s val: acc=%.4f | trade-WR=N/A (no non-FLAT predictions)",
                label, acc,
            )

    # ---- Save / load ----

    def save_encoder(self, encoder: LOBEncoder) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        path = self._model_dir / f"encoder_{ts}.pt"
        torch.save(encoder.state_dict(), path)

        latest = self._model_dir / "encoder_latest.pt"
        latest.unlink(missing_ok=True)
        latest.symlink_to(path.name)

        logger.info("Saved encoder: %s → %s", latest.name, path.name)
        return path

    def save_ensemble(
        self,
        xgb_models: list[xgb.Booster],
        lgb_model: object,
        logreg: object,
        top5_features: list[int],
        calibrators: list | None = None,
    ) -> dict[str, str]:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        paths: dict[str, str] = {}

        # Save 3 XGBoost models
        for i, model in enumerate(xgb_models):
            path = self._model_dir / f"xgb_{i}_{ts}.json"
            model.save_model(str(path))
            latest = self._model_dir / f"xgb_{i}_latest.json"
            latest.unlink(missing_ok=True)
            latest.symlink_to(path.name)
            logger.info("Saved XGBoost %d: %s -> %s", i, latest.name, path.name)
            paths[f"xgb_{i}"] = str(path)

        # Save LightGBM
        lgb_path = self._model_dir / f"lgb_{ts}.txt"
        lgb_model.save_model(str(lgb_path))
        lgb_latest = self._model_dir / "lgb_latest.txt"
        lgb_latest.unlink(missing_ok=True)
        lgb_latest.symlink_to(lgb_path.name)
        logger.info("Saved LightGBM: %s -> %s", lgb_latest.name, lgb_path.name)
        paths["lgb"] = str(lgb_path)

        # Save LogisticRegression
        logreg_path = self._model_dir / f"logreg_{ts}.pkl"
        joblib.dump(logreg, logreg_path)
        logreg_latest = self._model_dir / "logreg_latest.pkl"
        logreg_latest.unlink(missing_ok=True)
        logreg_latest.symlink_to(logreg_path.name)
        logger.info("Saved LogReg: %s -> %s", logreg_latest.name, logreg_path.name)
        paths["logreg"] = str(logreg_path)

        # Save top-5 feature indices
        feat_path = self._model_dir / f"logreg_features_{ts}.json"
        with open(feat_path, "w") as f:
            json.dump(top5_features, f)
        feat_latest = self._model_dir / "logreg_features.json"
        feat_latest.unlink(missing_ok=True)
        feat_latest.symlink_to(feat_path.name)
        logger.info("Saved LogReg features: %s -> %s", feat_latest.name, feat_path.name)
        paths["logreg_features"] = str(feat_path)

        # Save isotonic calibrators (Lever 4) — joblib pickles a list of
        # IsotonicRegression instances (or None for degenerate classes).
        if calibrators is not None:
            cal_path = self._model_dir / f"calibrators_{ts}.pkl"
            joblib.dump(calibrators, cal_path)
            cal_latest = self._model_dir / "calibrators_latest.pkl"
            cal_latest.unlink(missing_ok=True)
            cal_latest.symlink_to(cal_path.name)
            logger.info("Saved calibrators: %s -> %s", cal_latest.name, cal_path.name)
            paths["calibrators"] = str(cal_path)

        return paths

    # ---- Full pipeline ----

    def train_full(
        self,
        hours: int = 24,
        n_jobs: int = 1,
        warm_start_encoder: Path | None = None,
        force_rebuild: bool = False,
    ) -> dict:
        """Full CNN + ensemble training pipeline.

        `n_jobs` is the user-facing thread budget, applied to PyTorch, XGBoost,
        LightGBM and LogReg. Default is 1 so retrains on the 3-vCPU prod VPS
        never starve the recorder/bot. Pass -1 for development walk-forward.

        `warm_start_encoder`: if set, the CNN loads these weights and runs
        only 5 fine-tune epochs — meant for daily retrain cycles where yesterday's
        encoder is a strong init.

        `force_rebuild`: ignore the sample cache and rebuild X_lob/X_feat/y
        from raw parquet data.
        """
        t0 = time.monotonic()

        X_lob, X_feat, y, _mids = self.build_samples_cached(
            hours=hours, force_rebuild=force_rebuild,
        )

        if len(y) < 100:
            raise ValueError(f"Too few samples ({len(y)}), need at least 100. Collect more data.")

        # Train CNN (optionally warm-started)
        encoder = self.train_cnn(
            X_lob, y, n_jobs=n_jobs, warm_start_path=warm_start_encoder,
        )
        self.save_encoder(encoder)

        # Extract embeddings and train ensemble
        embeddings = self.extract_embeddings(encoder, X_lob)
        xgb_models, lgb_model, logreg, top5_features, calibrators = self.train_ensemble(
            embeddings, X_feat, y, n_jobs=n_jobs,
        )
        ensemble_paths = self.save_ensemble(
            xgb_models, lgb_model, logreg, top5_features, calibrators=calibrators,
        )

        elapsed = time.monotonic() - t0
        logger.info("Full training completed in %.1f minutes", elapsed / 60)

        return {
            "samples": len(y),
            "elapsed_min": elapsed / 60,
            "encoder_path": str(self._model_dir / "encoder_latest.pt"),
            **ensemble_paths,
        }
