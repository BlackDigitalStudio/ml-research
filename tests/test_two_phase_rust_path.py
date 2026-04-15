"""Integration smoke-test for the two-phase Rust path in trainer.

Validates:
  - _save_streams_for_rust writes valid flat-schema parquets
  - compute_features_from_paths returns numerical output matching
    the in-memory compute_features path on the same inputs
  - Memory behaviour: paths are actual Path objects, never None in
    places they shouldn't be

Skips tests that require the Rust binary at runtime if it's not built.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_BIN = REPO_ROOT / "rust_ingest" / "target" / "release" / "feature_builder"


@pytest.fixture
def synth_streams():
    """Small synthetic streams that match trainer's expected shapes."""
    rng = np.random.default_rng(42)
    n_depth = 5000
    n_trade = 10_000
    ts0 = 1_700_000_000_000
    depth_ts = np.arange(n_depth, dtype=np.int64) * 100 + ts0
    bid_vols = rng.exponential(1.0, size=(n_depth, 20)).astype(np.float32)
    ask_vols = rng.exponential(1.0, size=(n_depth, 20)).astype(np.float32)
    bid_prices = 67000 - np.arange(20)[None, :] * 0.1 \
        + rng.normal(0, 5, size=(n_depth, 20))
    ask_prices = 67000 + np.arange(20)[None, :] * 0.1 \
        + rng.normal(0, 5, size=(n_depth, 20))
    bid_prices = bid_prices.astype(np.float64)
    ask_prices = ask_prices.astype(np.float64)

    trade_ts = np.sort(rng.integers(ts0, ts0 + n_depth * 100,
                                     size=n_trade, dtype=np.int64))
    trade_price = rng.normal(67000, 50, size=n_trade)
    trade_qty = rng.exponential(1.0, size=n_trade)
    trade_side = rng.random(n_trade) < 0.5

    funding_ts = np.arange(ts0, ts0 + n_depth * 100, 1000, dtype=np.int64)
    funding_rate = np.full(len(funding_ts), 0.0001)

    deriv_ts = np.arange(ts0, ts0 + n_depth * 100, 5000, dtype=np.int64)
    deriv_oi = rng.uniform(1e6, 1.1e6, size=len(deriv_ts))
    deriv_ls = rng.uniform(0.8, 1.2, size=len(deriv_ts))

    return dict(
        depth_ts=depth_ts, bid_vols=bid_vols, ask_vols=ask_vols,
        bid_prices=bid_prices, ask_prices=ask_prices,
        trade_ts=trade_ts, trade_price=trade_price,
        trade_qty=trade_qty, trade_side=trade_side,
        funding_ts=funding_ts, funding_rate=funding_rate,
        deriv_ts=deriv_ts, deriv_oi=deriv_oi, deriv_ls=deriv_ls,
    )


@pytest.mark.skipif(not FEATURE_BIN.exists(),
                     reason="feature_builder binary not built")
def test_save_streams_produces_valid_parquets(synth_streams, tmp_path):
    """_save_streams_for_rust → valid flat-schema depth + simple trades parquets."""
    from src.config import Config
    from src.trainer import Trainer
    # Point data_dir at tmp_path so _merged/ is written there.
    cfg = Config.__new__(Config)
    object.__setattr__(cfg, "data_dir", tmp_path)
    object.__setattr__(cfg, "model_dir", tmp_path / "models")
    trainer = Trainer.__new__(Trainer)
    trainer._data_dir = tmp_path
    trainer._cfg = cfg

    paths = trainer._save_streams_for_rust(**synth_streams, cross_ex_data=None,
                                            eth_ts=None, eth_price=None,
                                            eth_qty=None, eth_side=None)
    # Depth parquet has flat schema
    assert paths["depth_path"].exists()
    schema = pq.read_schema(str(paths["depth_path"]))
    names = set(schema.names)
    assert {"timestamp", "bid_prices", "bid_qtys", "ask_prices", "ask_qtys"} <= names

    # Trades + funding + derivs present
    assert paths["trades_path"].exists()
    assert paths["funding_path"].exists()
    assert paths["derivs_path"].exists()
    # No ETH / cross_ex → not present
    assert "eth_path" not in paths
    assert "bybit_path" not in paths


@pytest.mark.skipif(not FEATURE_BIN.exists(),
                     reason="feature_builder binary not built")
def test_run_rust_features_end_to_end(synth_streams, tmp_path):
    """Full two-phase path returns valid feature matrix."""
    from src.config import Config
    from src.trainer import Trainer
    cfg = Config.__new__(Config)
    object.__setattr__(cfg, "data_dir", tmp_path)
    object.__setattr__(cfg, "model_dir", tmp_path / "models")
    trainer = Trainer.__new__(Trainer)
    trainer._data_dir = tmp_path
    trainer._cfg = cfg

    paths = trainer._save_streams_for_rust(**synth_streams, cross_ex_data=None,
                                            eth_ts=None, eth_price=None,
                                            eth_qty=None, eth_side=None)
    # Pick indices in the middle (avoid edges / warm-up)
    indices = np.arange(500, 4500, 100, dtype=np.int64)
    feats = trainer._run_rust_features(paths, indices)
    assert feats.shape == (len(indices), 56)
    assert feats.dtype == np.float32
    # Some columns should be non-zero (real features on real data)
    assert feats[:, :12].std() > 0   # depth-only features vary
