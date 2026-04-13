"""Bridge module — invokes Rust `feature_builder` / `sim_labels` from Python.

Drop-in replacements for `Trainer._calc_features_batch` and the LONG/SHORT
forward-sim loop in `Trainer.build_samples`. Enable via env var:

    export SCALPER_USE_RUST=1

The binaries must be built (`cargo build --release` in rust_ingest/). This
module is parity-validated against the Python reference; see:
    scripts/parity_rust_features.py
    scripts/parity_rust_live_sim.py

Contract: the Rust path produces byte-identical outputs (to f32 precision)
for all 34 features and byte-identical labels + target_pnl for live_sim.
Any divergence is a bug — do not silently fall through.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_REPO = Path(__file__).resolve().parents[1]
_FEATURE_BIN = _REPO / "rust_ingest" / "target" / "release" / "feature_builder"
_SIM_BIN = _REPO / "rust_ingest" / "target" / "release" / "sim_labels"


def use_rust() -> bool:
    """Return True iff the Rust path is enabled AND binaries exist."""
    if os.environ.get("SCALPER_USE_RUST", "").lower() not in ("1", "true", "yes"):
        return False
    return _FEATURE_BIN.exists() and _SIM_BIN.exists()


def _write_depth_parquet(path: Path, depth_ts, bid_prices, bid_qtys, ask_prices, ask_qtys):
    """Serialize depth arrays to the flat FixedSizeList schema the Rust reader expects."""
    n = len(depth_ts)
    fsl_type = pa.list_(pa.float64(), 20)

    def _fsl(arr):
        flat = arr.astype(np.float64, copy=False).reshape(-1)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float64()), 20)

    table = pa.table({
        "timestamp": pa.array(depth_ts.astype(np.int64), type=pa.int64()),
        "bid_prices": _fsl(bid_prices),
        "bid_qtys":   _fsl(bid_qtys),
        "ask_prices": _fsl(ask_prices),
        "ask_qtys":   _fsl(ask_qtys),
    })
    pq.write_table(table, str(path), compression="snappy")


def _write_trades_parquet(path: Path, ts, price, qty, side_bool, side_col="is_buyer_maker"):
    table = pa.table({
        "timestamp": pa.array(ts.astype(np.int64), type=pa.int64()),
        "price": pa.array(price.astype(np.float64), type=pa.float64()),
        "quantity": pa.array(qty.astype(np.float64), type=pa.float64()),
        side_col: pa.array(side_bool.astype(bool), type=pa.bool_()),
    })
    pq.write_table(table, str(path), compression="snappy")


def compute_features(
    bid_vols, ask_vols, bid_prices, ask_prices, mid_prices,
    trade_ts, trade_qty, trade_side, depth_ts, indices,
    *,
    trade_price=None,
    eth_ts=None, eth_price=None, eth_qty=None, eth_side=None,
    funding_ts=None, funding_rate_arr=None,
    deriv_ts=None, deriv_oi=None, deriv_ls=None,
    cross_ex_data=None,
) -> np.ndarray:
    """Rust drop-in for Trainer._calc_features_batch. Same signature."""
    if not _FEATURE_BIN.exists():
        raise RuntimeError(f"Rust feature_builder not built: {_FEATURE_BIN}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        depth_path = td / "depth.parquet"
        _write_depth_parquet(depth_path, depth_ts, bid_prices, bid_vols, ask_prices, ask_vols)

        idx_path = td / "idx.npy"
        np.save(idx_path, indices.astype(np.int64))
        out_path = td / "feat.npy"

        cmd = [str(_FEATURE_BIN),
               "--depth", str(depth_path),
               "--indices", str(idx_path),
               "--out", str(out_path)]

        if trade_ts is not None and len(trade_ts) > 0:
            tp = td / "trades.parquet"
            px = trade_price if trade_price is not None else np.zeros(len(trade_ts))
            _write_trades_parquet(tp, trade_ts, px, trade_qty, trade_side, "is_buyer_maker")
            cmd += ["--trades", str(tp)]

        if funding_ts is not None and len(funding_ts) > 0:
            fp = td / "funding.parquet"
            pq.write_table(pa.table({
                "timestamp": pa.array(funding_ts.astype(np.int64), type=pa.int64()),
                "funding_rate": pa.array(funding_rate_arr.astype(np.float64), type=pa.float64()),
                "mark_price": pa.array(np.zeros(len(funding_ts)), type=pa.float64()),
            }), str(fp))
            cmd += ["--funding", str(fp)]

        if deriv_ts is not None and len(deriv_ts) > 0:
            dp = td / "derivs.parquet"
            pq.write_table(pa.table({
                "timestamp": pa.array(deriv_ts.astype(np.int64), type=pa.int64()),
                "open_interest": pa.array(deriv_oi.astype(np.float64), type=pa.float64()),
                "long_short_ratio": pa.array(deriv_ls.astype(np.float64), type=pa.float64()),
            }), str(dp))
            cmd += ["--derivs", str(dp)]

        if eth_ts is not None and len(eth_ts) > 0:
            ep = td / "eth.parquet"
            _write_trades_parquet(ep, eth_ts, eth_price, eth_qty, eth_side, "is_buyer_maker")
            cmd += ["--eth", str(ep)]

        if cross_ex_data:
            for ex in ("bybit", "okx", "bitget", "gateio"):
                if ex not in cross_ex_data:
                    continue
                ex_ts, ex_signed = cross_ex_data[ex]
                if len(ex_ts) == 0:
                    continue
                # Write in recorder schema (is_seller, signed qty magnitude).
                cp = td / f"{ex}.parquet"
                # Reconstruct is_seller + qty: Rust re-applies gateio abs() itself.
                qty = np.abs(ex_signed)
                is_seller = ex_signed < 0
                pq.write_table(pa.table({
                    "timestamp": pa.array(ex_ts.astype(np.int64), type=pa.int64()),
                    "price": pa.array(np.zeros(len(ex_ts)), type=pa.float64()),
                    "quantity": pa.array(qty.astype(np.float64), type=pa.float64()),
                    "is_seller": pa.array(is_seller, type=pa.bool_()),
                }), str(cp))
                cmd += [f"--{ex}", str(cp)]

        subprocess.run(cmd, check=True)
        return np.load(out_path)


def simulate_labels(
    entry_long, entry_short, mid_paths, tp_pct, sl_pct, timeout_ticks,
    *,
    commission_win_pct=0.04, commission_loss_pct=0.07,
    partial_enabled=True, trailing_enabled=True,
    fill_latency_ms=150.0,
):
    """Rust drop-in for the LONG/SHORT forward-sim loop. Returns dict of arrays:
       y (u8), target_pnl (f64), reason_long/short (u8), pnl_long/short (f64).
    """
    if not _SIM_BIN.exists():
        raise RuntimeError(f"Rust sim_labels not built: {_SIM_BIN}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.save(td / "el.npy", entry_long.astype(np.float64))
        np.save(td / "es.npy", entry_short.astype(np.float64))
        np.save(td / "mid.npy", mid_paths.astype(np.float64))
        np.save(td / "tp.npy", tp_pct.astype(np.float64))
        np.save(td / "sl.npy", sl_pct.astype(np.float64))
        np.save(td / "to.npy", timeout_ticks.astype(np.int64))
        prefix = td / "out"
        cmd = [str(_SIM_BIN),
               "--entry-long", str(td / "el.npy"),
               "--entry-short", str(td / "es.npy"),
               "--mid-paths", str(td / "mid.npy"),
               "--tp-pct", str(td / "tp.npy"),
               "--sl-pct", str(td / "sl.npy"),
               "--timeout-ticks", str(td / "to.npy"),
               "--commission-win-pct", str(commission_win_pct),
               "--commission-loss-pct", str(commission_loss_pct),
               "--partial-enabled", str(partial_enabled).lower(),
               "--trailing-enabled", str(trailing_enabled).lower(),
               "--fill-latency-ms", str(fill_latency_ms),
               "--out-prefix", str(prefix)]
        subprocess.run(cmd, check=True)
        return {
            "y": np.load(f"{prefix}_y.npy"),
            "target_pnl": np.load(f"{prefix}_target_pnl.npy"),
            "reason_long": np.load(f"{prefix}_reason_long.npy"),
            "reason_short": np.load(f"{prefix}_reason_short.npy"),
            "pnl_long": np.load(f"{prefix}_pnl_long.npy"),
            "pnl_short": np.load(f"{prefix}_pnl_short.npy"),
        }
