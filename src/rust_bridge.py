"""Bridge module — invokes Rust `feature_builder` / `sim_labels` from Python.

Drop-in replacements for `Trainer._calc_features_batch` and the LONG/SHORT
forward-sim loop in `Trainer.build_samples`. The Rust path is the
**default** — Python is kept only for parity debugging and must be opted
into explicitly:

    export SCALPER_USE_RUST=0   # disable Rust (parity/debug only)

Binaries must be built (`cargo build --release` in rust_ingest/). If they
are missing while Rust is active (the default), use_rust() raises
RustBinariesMissing rather than silently falling back — silent fallback
previously wasted hours of training on the slow path.

Parity is validated by:
    scripts/parity_rust_features.py
    scripts/parity_rust_live_sim.py

Contract: Rust path produces byte-identical outputs (to f32 precision)
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


class RustBinariesMissing(RuntimeError):
    """Raised when Rust is active (default) but binaries are not built."""


def _rust_enabled() -> bool:
    """Default: Rust ON. Opt out with SCALPER_USE_RUST in {0,false,no,off}."""
    v = os.environ.get("SCALPER_USE_RUST", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def use_rust() -> bool:
    """Return True when the Rust path should be used.

    Raises RustBinariesMissing if Rust is active but binaries are absent —
    silent Python fallback is considered a bug (can double training time
    without anyone noticing).
    """
    if not _rust_enabled():
        return False
    missing = [p for p in (_FEATURE_BIN, _SIM_BIN) if not p.exists()]
    if missing:
        raise RustBinariesMissing(
            "Rust pipeline is the default but binaries are missing: "
            f"{[str(p) for p in missing]}. "
            "Build with: cd rust_ingest && cargo build --release. "
            "Or opt out (parity/debug only): SCALPER_USE_RUST=0"
        )
    return True


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
    funding_ts=None, funding_rate_arr=None, funding_mark_arr=None,
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
            _mark = funding_mark_arr if funding_mark_arr is not None else np.zeros(len(funding_ts))
            _write_scalar_parquet(fp, funding_ts,
                                   {"funding_rate": funding_rate_arr,
                                    "mark_price": _mark})
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


def compute_features_chunked(
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
    chunk_samples: int = 200_000,
    lookback_ticks: int = 1500,
    out_width: int = 56,
) -> np.ndarray:
    """Memory-bounded feature computation over arbitrarily-large merged parquets.

    Rather than loading every stream in full and letting feature_builder
    materialise (65M × 640 B ≈ 42 GB depth arrays), we slice each parquet
    by depth-timestamp range per chunk of indices. Each chunk processes
    ≤ `chunk_samples` sample indices spanning a contiguous depth window;
    feature_builder only ever sees the slice that those samples need, so
    Rust peak RSS stays under ~8 GB regardless of total dataset size.

    Parameters
    ----------
    indices : (N,) int64
        Sample indices into the full depth array. Need not be sorted;
        ordering of the returned (N, F) matrix matches input order.
    depth_path, trades_path, ... : Path
        Full merged parquets (output of scripts/merge_streams.py).
    chunk_samples : int
        Max samples per feature_builder invocation. 200k → ~5 chunks on a
        1M-sample run; tune down to 100k if RSS headroom is tight.
    lookback_ticks : int
        Depth-row margin kept before each chunk's earliest index, so the
        longest backward-looking feature window (kyle_lambda_60s uses 600,
        OFI 120s uses 1200) has complete context. Default 1500 > 1200.
    out_width : int
        Raw feature-matrix width emitted by feature_builder (post-Stage-D
        = 56; post-Stage-E prune happens in `KEPT_RAW_INDICES` downstream).

    Memory envelope per chunk:
        depth slice   ≈ 6M rows × 640 B ≈ 4 GB
        trades slice  ≈ proportional (typically ≤ 1 GB)
        eth + cross-ex ≤ 2 GB combined
        feature_builder internals (cumsums, rolling windows) ≈ 2 GB
        Total peak per chunk ≈ 8 GB  (fits 62 GB Contabo with plenty headroom).
    """
    if not _FEATURE_BIN.exists():
        raise RuntimeError(f"Rust feature_builder not built: {_FEATURE_BIN}")

    # Load depth_ts once up-front (small — 8 B/row).
    depth_ts = pq.read_table(str(depth_path), columns=["timestamp"]) \
        ["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
    n_depth = len(depth_ts)

    indices = np.asarray(indices, dtype=np.int64)
    n_samples = len(indices)
    if n_samples == 0:
        return np.zeros((0, out_width), dtype=np.float32)

    # Sort by index so each chunk covers a contiguous depth range.
    sort_order = np.argsort(indices, kind="stable")
    idx_sorted = indices[sort_order]
    inv_order = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(n_samples)

    out_sorted = np.zeros((n_samples, out_width), dtype=np.float32)

    # Stream-slicer: reads row groups of a parquet, writes only rows whose
    # `ts_col` falls in [ts_min, ts_max] to `dst`. Keeps one row group in
    # RAM at a time (~100 MB). Returns True if any rows were written.
    def _slice_by_ts(src: Path, dst: Path, ts_col: str,
                     ts_min: int, ts_max: int) -> bool:
        pf = pq.ParquetFile(str(src))
        schema = pf.schema_arrow
        wrote_any = False
        with pq.ParquetWriter(str(dst), schema, compression="snappy") as w:
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                ts_arr = t[ts_col].to_numpy(zero_copy_only=False)
                if len(ts_arr) == 0:
                    continue
                rg_min = int(ts_arr.min())
                rg_max = int(ts_arr.max())
                if rg_max < ts_min or rg_min > ts_max:
                    continue
                if rg_min >= ts_min and rg_max <= ts_max:
                    w.write_table(t)
                    wrote_any = True
                    continue
                mask = (ts_arr >= ts_min) & (ts_arr <= ts_max)
                if mask.any():
                    import pyarrow as pa
                    w.write_table(t.filter(pa.array(mask)))
                    wrote_any = True
        if not wrote_any:
            dst.unlink(missing_ok=True)
        return wrote_any

    # Depth slicer by row range (more precise than ts filtering).
    def _slice_depth_by_rows(src: Path, dst: Path,
                              row_start: int, row_end: int) -> tuple[int, int]:
        pf = pq.ParquetFile(str(src))
        schema = pf.schema_arrow
        seen = 0
        ts_first = ts_last = 0
        with pq.ParquetWriter(str(dst), schema, compression="snappy") as w:
            for rg in range(pf.num_row_groups):
                rg_meta = pf.metadata.row_group(rg)
                rg_rows = rg_meta.num_rows
                rg_lo = seen
                rg_hi = seen + rg_rows
                seen = rg_hi
                if rg_hi <= row_start or rg_lo >= row_end:
                    continue
                t = pf.read_row_group(rg)
                local_start = max(0, row_start - rg_lo)
                local_end = min(rg_rows, row_end - rg_lo)
                t = t.slice(local_start, local_end - local_start)
                if t.num_rows == 0:
                    continue
                ts_np = t["timestamp"].to_numpy(zero_copy_only=False)
                if ts_first == 0:
                    ts_first = int(ts_np[0])
                ts_last = int(ts_np[-1])
                w.write_table(t)
        return ts_first, ts_last

    import tempfile as _tf
    stream_specs = [
        ("trades", trades_path, "timestamp"),
        ("eth",    eth_path,    "timestamp"),
        ("bybit",  bybit_path,  "timestamp"),
        ("okx",    okx_path,    "timestamp"),
        ("bitget", bitget_path, "timestamp"),
        ("gateio", gateio_path, "timestamp"),
        ("funding", funding_path, "timestamp"),
        ("derivs",  derivs_path,  "timestamp"),
    ]

    n_chunks = (n_samples + chunk_samples - 1) // chunk_samples
    for ci, chunk_start in enumerate(range(0, n_samples, chunk_samples)):
        chunk_end = min(chunk_start + chunk_samples, n_samples)
        chunk_idx = idx_sorted[chunk_start:chunk_end]
        row_start = max(0, int(chunk_idx[0]) - lookback_ticks)
        row_end = min(n_depth, int(chunk_idx[-1]) + 1)  # inclusive last row used

        with _tf.TemporaryDirectory() as td:
            td = Path(td)
            depth_slice = td / "depth.parquet"
            ts_min, ts_max = _slice_depth_by_rows(
                Path(depth_path), depth_slice, row_start, row_end,
            )
            # Slice all dependent streams by ts range derived from the depth slice.
            slice_kwargs: dict = {"depth_path": depth_slice}
            for name, src, ts_col in stream_specs:
                if src is None:
                    continue
                dst = td / f"{name}.parquet"
                if _slice_by_ts(Path(src), dst, ts_col, ts_min, ts_max):
                    slice_kwargs[f"{name}_path"] = dst
            # Rebase indices relative to the depth slice.
            rebased = chunk_idx - row_start
            assert rebased.min() >= 0 and rebased.max() < (row_end - row_start)

            feat_chunk = compute_features_from_paths(
                indices=rebased, **slice_kwargs,
            )
        out_sorted[chunk_start:chunk_end] = feat_chunk
        print(f"[features_chunked] chunk {ci + 1}/{n_chunks}: "
              f"{chunk_end - chunk_start} samples, depth_rows={row_end - row_start}")

    # Undo sort.
    return out_sorted[inv_order]


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
