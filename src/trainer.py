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
FLAT_THRESHOLD_PCT = 0.20  # ±0.2% for UP/DOWN label


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

    # ---- Sample building (vectorized) ----

    def build_samples(
        self, hours: int = 24,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build (X_lob, X_features, y, mid_prices) from raw data.

        Loads data, parses into numpy, frees DataFrames immediately to save RAM.
        X_lob is written to disk (mmap) — never fully held in memory.

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
        logger.info("Depth parsed, DataFrame freed")

        # === Load and parse trades (free DataFrame ASAP) ===
        trade_df = self.load_trade_data(hours)
        trade_ts = trade_df["timestamp"].values.astype(np.int64).copy()
        trade_qty = trade_df["quantity"].values.astype(np.float64).copy()
        trade_side = trade_df["is_buyer_maker"].values.copy()
        del trade_df
        gc.collect()  # True = sell

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
        lob_path = self._data_dir / "_tmp_X_lob.npy"
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
        )

        # Free large parse arrays (LOB + features are done)
        del bid_vols, ask_vols, bid_prices, ask_prices
        del tick_buy_vol, tick_sell_vol, trade_ts, trade_qty, trade_side, depth_ts
        import gc; gc.collect()

        # === Labels (vectorized) ===
        future_starts = sample_starts + WINDOW_SIZE
        future_win = np.lib.stride_tricks.sliding_window_view(mid_prices, HORIZON)
        future_mids = future_win[future_starts]              # (N, HORIZON)
        current_mids = mid_prices[future_starts - 1]         # (N,)

        safe = np.where(current_mids > 0, current_mids, 1.0)
        max_up = (future_mids.max(axis=1) - current_mids) / safe * 100
        max_down = (current_mids - future_mids.min(axis=1)) / safe * 100

        y = np.full(num_samples, FLAT, dtype=np.int64)
        y[(max_up >= FLAT_THRESHOLD_PCT) & (max_up >= max_down)] = UP
        down_mask = (max_down >= FLAT_THRESHOLD_PCT) & (max_down > max_up)
        y[down_mask & (y != UP)] = DOWN

        # Mid prices at sample points (for backtest alignment)
        sample_mids = current_mids.copy()

        # Filter zero mid prices
        valid = current_mids > 0
        if not valid.all():
            X_lob, X_feat, y, sample_mids = X_lob[valid], X_feat[valid], y[valid], sample_mids[valid]

        counts = {UP: int((y == UP).sum()), DOWN: int((y == DOWN).sum()), FLAT: int((y == FLAT).sum())}
        logger.info(
            "Built %d samples: UP=%d (%.1f%%) DOWN=%d (%.1f%%) FLAT=%d (%.1f%%)",
            len(y), counts[UP], counts[UP] / len(y) * 100,
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
    ) -> np.ndarray:
        """Compute all 25 features for all sample indices at once."""
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

        # [11] VWAP deviation — placeholder in training
        # [13] Funding rate — placeholder
        # [14-16] ETH signals — placeholder
        # [17] OI delta — placeholder
        # [19] Liquidation proximity — placeholder
        feat[:, 18] = 1.0   # neutral L/S ratio

        # [12] Momentum 5s
        m50 = indices >= 50
        prev50 = mid_prices[indices[m50] - 50]
        feat[m50, 12] = np.where(prev50 > 0,
                                  (mid_prices[indices[m50]] - prev50) / prev50, 0).astype(np.float32)

        # [20] Spoof score approximation
        m25 = indices >= 25
        has_large = feat[:, 5] > 0
        pc = np.abs(mid_prices[indices] - mid_prices[np.maximum(indices - 25, 0)])
        feat[:, 20] = np.where(has_large & m25 & (pc < 0.10), 1.0, 0.0)

        # [21-22] Volatility/intensity ratios — placeholder
        feat[:, 21] = 1.0
        feat[:, 22] = 1.0

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

        return feat

    # ---- Training ----

    def train_cnn(
        self,
        X_lob: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 0.001,
        patience: int = 10,
    ) -> LOBEncoder:
        n = len(X_lob)
        split = int(n * (1 - val_split))

        # .copy() handles mmap read-only arrays (torch needs writable memory)
        X_train = torch.from_numpy(np.array(X_lob[:split]))
        y_train = torch.from_numpy(y[:split].copy() if isinstance(y, np.memmap) else y[:split])
        X_val = torch.from_numpy(np.array(X_lob[split:]))
        y_val = torch.from_numpy(y[split:].copy() if isinstance(y, np.memmap) else y[split:])

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        # CNN + classification head for training
        encoder = LOBEncoder()
        head = nn.Linear(64, 3)

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
                emb = encoder(xb)
                logits = head(emb)
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
                    emb = encoder(xb)
                    logits = head(emb)
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

    def train_ensemble(
        self,
        embeddings: np.ndarray,
        X_feat: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
    ) -> tuple[list[xgb.Booster], object, object, list[int]]:
        X = np.hstack([embeddings, X_feat])
        n = len(X)
        split = int(n * (1 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # --- 3 XGBoost with different seeds, each on random 80% of training rows ---
        xgb_models = []
        xgb_importances = []
        for seed in [42, 123, 456]:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_train), size=int(len(X_train) * 0.8), replace=False)
            dtrain = xgb.DMatrix(X_train[idx], label=y_train[idx])
            dval_dm = xgb.DMatrix(X_val, label=y_val)
            params = {
                "objective": "multi:softprob", "num_class": 3, "max_depth": 6,
                "learning_rate": 0.05, "min_child_weight": 50, "subsample": 0.8,
                "colsample_bytree": 0.8, "eval_metric": "mlogloss", "verbosity": 0,
                "seed": seed,
            }
            logger.info("Training XGBoost (seed=%d): %d train (80%%), %d val",
                         seed, len(idx), len(X_val))
            model = xgb.train(
                params, dtrain, num_boost_round=300,
                evals=[(dval_dm, "val")], early_stopping_rounds=20, verbose_eval=0,
            )
            xgb_models.append(model)
            scores = model.get_score(importance_type="gain")
            imp = np.zeros(X.shape[1])
            for k, v in scores.items():
                imp[int(k[1:])] = v  # feature names are f0, f1, ...
            xgb_importances.append(imp)

            # Evaluate individual XGBoost
            preds = model.predict(dval_dm)
            y_pred = preds.argmax(axis=1)
            acc = accuracy_score(y_val, y_pred)
            logger.info("XGBoost (seed=%d) val accuracy: %.4f", seed, acc)

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
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        lgb_params = {
            "objective": "multiclass", "num_class": 3, "max_depth": 6,
            "learning_rate": 0.05, "min_child_samples": 50, "subsample": 0.8,
            "colsample_bytree": 0.8, "metric": "multi_logloss", "verbosity": -1,
        }
        logger.info("Training LightGBM: %d train, %d val", len(X_train), len(X_val))
        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=300,
            valid_sets=[lgb_val_ds],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
        )

        # Evaluate LightGBM
        lgb_preds = lgb_model.predict(X_val)
        lgb_y_pred = lgb_preds.argmax(axis=1)
        lgb_acc = accuracy_score(y_val, lgb_y_pred)
        logger.info("LightGBM val accuracy: %.4f", lgb_acc)

        # --- 1 LogisticRegression on top-5 features ---
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
        logger.info("Training LogisticRegression on top-5 features: %s", top5_global)
        logreg.fit(X_train[:, top5_global], y_train)

        # Evaluate LogisticRegression
        lr_y_pred = logreg.predict(X_val[:, top5_global])
        lr_acc = accuracy_score(y_val, lr_y_pred)
        logger.info("LogisticRegression val accuracy: %.4f", lr_acc)

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
        logger.info("Ensemble (5-model majority) val accuracy: %.4f", ens_acc)
        logger.info("\n%s", classification_report(
            y_val, ensemble_pred, target_names=["UP", "DOWN", "FLAT"],
        ))

        return xgb_models, lgb_model, logreg, top5_global

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

        return paths

    # ---- Full pipeline ----

    def train_full(self, hours: int = 24) -> dict:
        t0 = time.monotonic()

        X_lob, X_feat, y, _mids = self.build_samples(hours=hours)

        if len(y) < 100:
            raise ValueError(f"Too few samples ({len(y)}), need at least 100. Collect more data.")

        # Train CNN
        encoder = self.train_cnn(X_lob, y)
        self.save_encoder(encoder)

        # Extract embeddings and train ensemble
        embeddings = self.extract_embeddings(encoder, X_lob)
        xgb_models, lgb_model, logreg, top5_features = self.train_ensemble(
            embeddings, X_feat, y,
        )
        ensemble_paths = self.save_ensemble(xgb_models, lgb_model, logreg, top5_features)

        elapsed = time.monotonic() - t0
        logger.info("Full training completed in %.1f minutes", elapsed / 60)

        return {
            "samples": len(y),
            "elapsed_min": elapsed / 60,
            "encoder_path": str(self._model_dir / "encoder_latest.pt"),
            **ensemble_paths,
        }
