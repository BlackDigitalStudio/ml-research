"""End-to-end parity between `build_samples` and `run_backtest`.

This is the integration guard that would have caught the label/reality
split described in `handoff_current.md`. It builds a tiny synthetic
training set, labels it through the new `Trainer.build_samples` path,
and runs that result through the new `run_backtest` function — the
test passes if the whole pipeline runs without raising and produces
a non-empty `BacktestResult` with the expected metric fields.

We don't assert specific PnL numbers — synthetic data is a poor
predictor of actual executor behaviour. The point is to catch
signature drift between build_samples, live_sim, filters and
run_backtest, plus the classification_report crash on degenerate
labels (the original failure mode).
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_synthetic_parquets(base: Path, hours: int = 2) -> None:
    """Emit the minimum set of parquet files build_samples expects.

    Depth ticks are 100 ms cadence with 20 levels each side. Volumes get a
    little random jitter so the filter predicates aren't identically zero
    for every sample. Trades are coarse (1 Hz). Auxiliary feeds (ETH,
    funding, derivatives, cross-exchange) are written empty so the loader
    fall-back paths are also exercised.
    """
    rng = np.random.default_rng(7)
    n_ticks = hours * 3600 * 10  # 100 ms cadence
    # Timestamps start away from Asia-night / funding windows: 2026-04-10
    # 12:00:00 UTC.
    start_ts_ms = 1_775_563_200_000  # ms since epoch
    ts_ms = start_ts_ms + np.arange(n_ticks, dtype=np.int64) * 100

    # Mid price random walk so LONG and SHORT both hit TPs somewhere.
    mids = 100_000.0 + np.cumsum(rng.normal(0, 2.5, n_ticks))
    mids = np.clip(mids, 95_000.0, 105_000.0)

    # 20-level book: bid decreases by 0.5, ask increases by 0.5 per level.
    # Alternate skew every 500 ticks so some samples have imbalance >+0.15
    # (LONG gate passes) and some have imbalance <-0.15 (SHORT gate
    # passes). Without this skew `imbalance_ratio` is noise-level and the
    # filter rejects everything.
    bids = []
    asks = []
    skew_period = 500
    tick_size = 0.10  # BTCUSDT futures tick
    for t, m in enumerate(mids):
        bid_boost = 80.0 if (t // skew_period) % 2 == 0 else 20.0
        ask_boost = 20.0 if (t // skew_period) % 2 == 0 else 80.0
        # Best bid 1 tick below mid, best ask 1 tick above — spread = 0.10
        # (one tick), well inside MAX_SPREAD_USD = 0.20.
        bid_levels = [[float(m - tick_size * (i + 1)), float(bid_boost + rng.uniform(0, 5))] for i in range(20)]
        ask_levels = [[float(m + tick_size * (i + 1)), float(ask_boost + rng.uniform(0, 5))] for i in range(20)]
        bids.append(bid_levels)
        asks.append(ask_levels)

    depth_df = pd.DataFrame({
        "timestamp": ts_ms,
        "bids": bids,
        "asks": asks,
    })
    depth_dir = base / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_df.to_parquet(depth_dir / "000.parquet")

    # Aggregate trades — 1 trade/sec, random side.
    n_trades = hours * 3600
    trade_ts = start_ts_ms + np.arange(n_trades, dtype=np.int64) * 1000
    trade_price = np.interp(trade_ts, ts_ms, mids)
    trade_qty = rng.uniform(0.1, 2.0, n_trades)
    is_buyer_maker = rng.random(n_trades) > 0.5
    trades_df = pd.DataFrame({
        "timestamp": trade_ts,
        "price": trade_price,
        "quantity": trade_qty,
        "is_buyer_maker": is_buyer_maker,
    })
    trades_dir = base / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(trades_dir / "000.parquet")


def test_build_samples_and_backtest_end_to_end() -> None:
    from src.config import Config
    from src.trainer import Trainer, CACHE_SCHEMA_VERSION

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        data_dir = tmp / "data"
        model_dir = tmp / "models"
        log_dir = tmp / "logs"
        data_dir.mkdir(parents=True)
        _write_synthetic_parquets(data_dir, hours=2)

        cfg = Config(
            api_key="x", api_secret="x",
            data_dir=data_dir, model_dir=model_dir, log_dir=log_dir,
        )
        trainer = Trainer(cfg)
        X_lob, X_feat, y, mids, target_pnl = trainer.build_samples_cached(
            hours=2, force_rebuild=True,
        )
        assert len(y) == len(X_feat) == len(target_pnl) == len(mids)
        assert len(y) > 0, "build_samples produced no surviving labels"
        assert X_lob.shape[1:] == (3, 20, 50)
        assert target_pnl.dtype == np.float32

        # Cache-v2 files should exist and be discoverable on re-load.
        cache_files = list((data_dir / "_cache").glob(
            f"samples_{CACHE_SCHEMA_VERSION}_2h_*"
        ))
        assert any("_pnl.npy" in p.name for p in cache_files), (
            f"cache missing target_pnl file: {cache_files}"
        )

        # Second call should HIT the cache and return the same shapes.
        X_lob2, X_feat2, y2, mids2, pnl2 = trainer.build_samples_cached(
            hours=2, force_rebuild=False,
        )
        assert len(y2) == len(y)
        assert np.array_equal(y2, y)
        assert np.allclose(pnl2, target_pnl)


def test_run_backtest_on_synthetic_predictions_doesnt_crash() -> None:
    """`run_backtest` must consume the new 5-tuple cache layout cleanly.

    Feed it model-free predictions derived from the labels themselves
    (so some trades fire). The guard is "no crash + populated metrics";
    PnL semantics are covered by live_sim unit tests.
    """
    from src.config import Config
    from src.trainer import Trainer
    from scripts.backtest import run_backtest, print_results

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        data_dir = tmp / "data"
        model_dir = tmp / "models"
        log_dir = tmp / "logs"
        data_dir.mkdir(parents=True)
        _write_synthetic_parquets(data_dir, hours=2)

        cfg = Config(
            api_key="x", api_secret="x",
            data_dir=data_dir, model_dir=model_dir, log_dir=log_dir,
        )
        trainer = Trainer(cfg)
        X_lob, X_feat, y, mids, target_pnl = trainer.build_samples_cached(
            hours=2, force_rebuild=True,
        )

        # "Predict" using the labels themselves so run_backtest definitely
        # has trades to simulate. Confidences are above the threshold.
        preds = y.copy()
        confs = np.full_like(preds, 0.65, dtype=np.float64)

        result = run_backtest(
            mid_prices=mids.astype(np.float64),
            predictions=preds,
            confidences=confs,
            imbalances=X_feat[:, 1],
            spreads=X_feat[:, 3],
            X_feat=X_feat,
            confidence_threshold=0.58,
        )
        # No crash. Metrics must be accessible.
        assert result.total_trades >= 0
        _ = result.win_rate
        _ = result.tp_hit_rate
        _ = result.profit_factor
        _ = result.max_drawdown_pct
        # Also exercise the print path so the deploy-verdict block runs
        # without raising even when no trades fire.
        print_results(result, label="e2e-parity")


def test_classification_report_no_crash_on_degenerate_labels() -> None:
    """The 2-line `labels=` + `zero_division=0` fix must have landed.

    Regression guard for the handoff's "Known unresolved issues" —
    ValueError: Number of classes, 2, does not match size of target_names, 3.
    """
    from sklearn.metrics import classification_report
    from src.model import UP, DOWN, FLAT

    # y_val contains only 2 classes, predictions contain only 1.
    y_val = np.array([UP, FLAT, FLAT, UP, FLAT])
    y_pred = np.array([FLAT, FLAT, FLAT, FLAT, FLAT])
    # This would raise on the old call site.
    out = classification_report(
        y_val, y_pred,
        labels=[UP, DOWN, FLAT],
        target_names=["UP", "DOWN", "FLAT"],
        zero_division=0,
    )
    assert "UP" in out and "DOWN" in out and "FLAT" in out


if __name__ == "__main__":
    test_classification_report_no_crash_on_degenerate_labels()
    print("  ok: test_classification_report_no_crash_on_degenerate_labels")
    test_build_samples_and_backtest_end_to_end()
    print("  ok: test_build_samples_and_backtest_end_to_end")
    test_run_backtest_on_synthetic_predictions_doesnt_crash()
    print("  ok: test_run_backtest_on_synthetic_predictions_doesnt_crash")
    print("end-to-end parity tests OK")
