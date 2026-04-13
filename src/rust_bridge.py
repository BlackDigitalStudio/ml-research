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


def _write_depth_parquet(path: Path, depth_ts, bid_prices, bid_qtys, ask_prices, ask_qtys,
                          chunk_rows: int = 1_000_000):
    """Serialize depth arrays to flat FixedSizeList schema (Rust reader format).

    Uses ParquetWriter chunks to bound peak RAM at chunk_rows * 320 bytes
    (~320 MB for chunk=1M). Without chunking, pyarrow Table builds the
    entire table in RAM — 40 GB on full 76-day dataset → OOM with the
    Rust binary running concurrently inside cgroup memory limits.
    """
    fsl_type = pa.list_(pa.float64(), 20)
    schema = pa.schema([
        ("timestamp", pa.int64()),
        ("bid_prices", fsl_type),
        ("bid_qtys", fsl_type),
        ("ask_prices", fsl_type),
        ("ask_qtys", fsl_type),
    ])

    def _fsl_chunk(arr):
        flat = np.ascontiguousarray(arr.astype(np.float64, copy=False)).reshape(-1)
        return pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float64()), 20
        )

    n = len(depth_ts)
    with pq.ParquetWriter(str(path), schema, compression="snappy") as writer:
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            tbl = pa.table({
                "timestamp": pa.array(depth_ts[start:end].astype(np.int64, copy=False),
                                       type=pa.int64()),
                "bid_prices": _fsl_chunk(bid_prices[start:end]),
                "bid_qtys": _fsl_chunk(bid_qtys[start:end]),
                "ask_prices": _fsl_chunk(ask_prices[start:end]),
                "ask_qtys": _fsl_chunk(ask_qtys[start:end]),
            })
            writer.write_table(tbl)
            del tbl


def _write_scalar_parquet(path: Path, ts, columns: dict[str, np.ndarray],
                           chunk_rows: int = 5_000_000):
    """Chunked write for funding/derivs (timestamp + N float64 columns)."""
    cols = list(columns.keys())
    schema = pa.schema([("timestamp", pa.int64())] + [(c, pa.float64()) for c in cols])
    n = len(ts)
    with pq.ParquetWriter(str(path), schema, compression="snappy") as writer:
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            data = {"timestamp": pa.array(ts[start:end].astype(np.int64, copy=False),
                                            type=pa.int64())}
            for c in cols:
                data[c] = pa.array(columns[c][start:end].astype(np.float64, copy=False),
                                    type=pa.float64())
            writer.write_table(pa.table(data))


def _write_trades_parquet(path: Path, ts, price, qty, side_bool, side_col="is_buyer_maker",
                           chunk_rows: int = 5_000_000):
    """Chunked write — bounded RAM regardless of input size."""
    schema = pa.schema([
        ("timestamp", pa.int64()),
        ("price", pa.float64()),
        ("quantity", pa.float64()),
        (side_col, pa.bool_()),
    ])
    n = len(ts)
    with pq.ParquetWriter(str(path), schema, compression="snappy") as writer:
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            tbl = pa.table({
                "timestamp": pa.array(ts[start:end].astype(np.int64, copy=False), type=pa.int64()),
                "price": pa.array(price[start:end].astype(np.float64, copy=False), type=pa.float64()),
                "quantity": pa.array(qty[start:end].astype(np.float64, copy=False), type=pa.float64()),
                side_col: pa.array(side_bool[start:end].astype(bool, copy=False), type=pa.bool_()),
            })
            writer.write_table(tbl)
            del tbl


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
            _write_scalar_parquet(fp, funding_ts,
                                   {"funding_rate": funding_rate_arr,
                                    "mark_price": np.zeros(len(funding_ts))})
            cmd += ["--funding", str(fp)]

        if deriv_ts is not None and len(deriv_ts) > 0:
            dp = td / "derivs.parquet"
            _write_scalar_parquet(dp, deriv_ts,
                                   {"open_interest": deriv_oi,
                                    "long_short_ratio": deriv_ls})
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
                cp = td / f"{ex}.parquet"
                qty = np.abs(ex_signed)
                is_seller = ex_signed < 0
                _write_trades_parquet(cp, ex_ts,
                                       np.zeros(len(ex_ts), dtype=np.float64),
                                       qty, is_seller, side_col="is_seller")
                cmd += [f"--{ex}", str(cp)]

        subprocess.run(cmd, check=True)
        return np.load(out_path)


def compute_features_from_paths(
    *,
    indices: np.ndarray,
    depth_path: Path | str,
    trades_path: Path | str | None = None,
    funding_path: Path | str | None = None,
    derivs_path: Path | str | None = None,
    eth_path: Path | str | None = None,
    bybit_path: Path | str | None = None,
    okx_path: Path | str | None = None,
    bitget_path: Path | str | None = None,
    gateio_path: Path | str | None = None,
) -> np.ndarray:
    """Path-based variant of compute_features. Skips array→parquet
    serialization entirely — caller must have already saved each stream
    as a flat-schema parquet that the Rust reader understands.

    Use this when you have streams on disk and don't want the +40 GB
    transient pyarrow allocation for big depth datasets.
    """
    if not _FEATURE_BIN.exists():
        raise RuntimeError(f"Rust feature_builder not built: {_FEATURE_BIN}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx_path = td / "idx.npy"
        np.save(idx_path, indices.astype(np.int64))
        out_path = td / "feat.npy"

        cmd = [str(_FEATURE_BIN),
               "--depth", str(depth_path),
               "--indices", str(idx_path),
               "--out", str(out_path)]
        for flag, p in [("--trades", trades_path), ("--funding", funding_path),
                        ("--derivs", derivs_path), ("--eth", eth_path),
                        ("--bybit", bybit_path), ("--okx", okx_path),
                        ("--bitget", bitget_path), ("--gateio", gateio_path)]:
            if p is not None:
                cmd += [flag, str(p)]
        subprocess.run(cmd, check=True)
        return np.load(out_path)


def save_flat_depth_parquet(
    path: Path | str, ts, bid_prices, bid_qtys, ask_prices, ask_qtys,
    chunk_rows: int = 1_000_000,
) -> None:
    """Public helper for callers that want to pre-stage depth as flat parquet."""
    _write_depth_parquet(Path(path), ts, bid_prices, bid_qtys, ask_prices, ask_qtys,
                          chunk_rows=chunk_rows)


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
