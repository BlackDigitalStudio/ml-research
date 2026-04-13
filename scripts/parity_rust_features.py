#!/usr/bin/env python3
"""Parity harness: compare Rust `feature_builder` output against Python
`trainer._calc_features_batch` on a flat-schema Tardis depth parquet.

Session-2 scope: LOB-only feature columns [0,1,2,3,4,5,10,11].

Usage:
    python scripts/parity_rust_features.py <depth.parquet> [--n 10000]

Exit 0 on byte-match (tolerances per column defined below), non-zero otherwise.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


LOB_COLS = [0, 1, 2, 3, 4, 5, 10, 11, 12]
TRADE_COLS = [6, 7, 8, 9]
FUNDING_COLS = [13]
ETH_COLS = [14, 15, 16]
DERIV_COLS = [17, 18, 19]
CROSS_COLS = [30]
# Microstructure (always on; [22]/[33] require trades)
MICRO_DEPTH_COLS = [20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32]
MICRO_TRADE_COLS = [22, 33]
# Tolerances chosen generously; in practice parity is usually exact f32.
ATOL = {
    0: 1e-4,    # OFI
    1: 1e-6,
    2: 1e-6,
    3: 1e-4,    # spread in price units
    4: 1e-5,
    5: 0.0,     # binary
    6: 1e-6,    # trade flow imbalance
    7: 0.0,     # count (integer)
    8: 0.0,     # binary
    9: 1e-3,    # CVD in quantity units
    10: 1e-9,
    11: 1e-8,
    12: 1e-7,   # momentum
    13: 1e-9,   # funding rate
    14: 1e-8,   # eth momentum
    15: 1e-6,   # eth ofi
    16: 1e-6,   # btc/eth ratio signal
    17: 1e-7,   # OI delta
    18: 1e-7,   # L/S ratio
    19: 0.0,    # liq proximity (discrete constants)
    20: 0.0,    # spoof binary
    21: 1e-4,   # vol ratio (division tolerance)
    22: 1e-4,   # trade intensity ratio
    23: 1e-7,   # hurst
    24: 1e-5,   # sweep (tick units)
    25: 1e-2,   # cancel rate (quantity sum)
    26: 1e-4,   # OFI 1s
    27: 1e-3,   # OFI 5s
    28: 1e-2,   # OFI 30s (larger magnitudes)
    29: 1e-3,   # OFI divergence
    31: 1e-5,   # queue pressure EMA
    32: 1e-7,   # top3 asymmetry
    33: 1e-6,   # effective spread EMA
    30: 0.0,    # cross-ex count (integer)
}


def load_flat(parquet_path: Path):
    t = pq.read_table(str(parquet_path))
    t = t.combine_chunks()
    ts = t["timestamp"].chunk(0).to_numpy(zero_copy_only=True).astype(np.int64, copy=False)
    bp = t["bid_prices"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, 20)
    bq = t["bid_qtys"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, 20)
    ap = t["ask_prices"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, 20)
    aq = t["ask_qtys"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, 20)
    return ts, bp, bq, ap, aq


def load_scalar(parquet_path: Path, cols: list[str]):
    t = pq.read_table(str(parquet_path)).combine_chunks()
    ts = t["timestamp"].chunk(0).to_numpy(zero_copy_only=True).astype(np.int64, copy=False)
    out = [t[c].chunk(0).to_numpy(zero_copy_only=True).astype(np.float64, copy=False)
           for c in cols]
    return ts, *out


def load_trades(parquet_path: Path, with_price: bool = True):
    t = pq.read_table(str(parquet_path)).combine_chunks()
    ts = t["timestamp"].chunk(0).to_numpy(zero_copy_only=True).astype(np.int64, copy=False)
    qty = t["quantity"].chunk(0).to_numpy(zero_copy_only=True).astype(np.float64, copy=False)
    side = t["is_buyer_maker"].chunk(0).to_numpy(zero_copy_only=False).astype(bool, copy=False)
    price = t["price"].chunk(0).to_numpy(zero_copy_only=True).astype(np.float64, copy=False)
    if with_price:
        return ts, qty, side, price
    return ts, qty, side


def python_features(depth_ts, bid_prices, bid_vols, ask_prices, ask_vols, mid_prices, indices,
                    trade_ts=None, trade_qty=None, trade_side=None, trade_price=None,
                    funding_ts=None, funding_rate=None,
                    deriv_ts=None, deriv_oi=None, deriv_ls=None,
                    eth_ts=None, eth_price=None, eth_qty=None, eth_side=None,
                    cross_ex_data=None):
    """Invoke trainer._calc_features_batch with empty trade/ETH/funding/derivs inputs.
    Only LOB cols are validated — the others will differ/be zero and are ignored."""
    # Lazy import because trainer touches torch/xgboost.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.trainer import Trainer  # noqa: E402

    # Construct a bare trainer instance without running __init__ to avoid needing paths.
    trainer = Trainer.__new__(Trainer)

    if trade_ts is None:
        trade_ts = np.array([], dtype=np.int64)
        trade_qty = np.array([], dtype=np.float64)
        trade_side = np.array([], dtype=bool)

    feat = trainer._calc_features_batch(
        bid_vols=bid_vols,
        ask_vols=ask_vols,
        bid_prices=bid_prices,
        ask_prices=ask_prices,
        mid_prices=mid_prices,
        trade_ts=trade_ts,
        trade_qty=trade_qty,
        trade_side=trade_side,
        trade_price=trade_price,
        depth_ts=depth_ts,
        indices=indices,
        funding_ts=funding_ts,
        funding_rate_arr=funding_rate,
        deriv_ts=deriv_ts,
        deriv_oi=deriv_oi,
        deriv_ls=deriv_ls,
        eth_ts=eth_ts,
        eth_price=eth_price,
        eth_qty=eth_qty,
        eth_side=eth_side,
        cross_ex_data=cross_ex_data,
    )
    return feat


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    p.add_argument("--trades", default=None)
    p.add_argument("--funding", default=None)
    p.add_argument("--derivs", default=None)
    p.add_argument("--eth", default=None)
    p.add_argument("--bybit", default=None)
    p.add_argument("--okx", default=None)
    p.add_argument("--bitget", default=None)
    p.add_argument("--gateio", default=None)
    p.add_argument("--n", type=int, default=10000, help="number of sample indices")
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--offset", type=int, default=30_000,
                   help="skip warmup rows (30k = ~50min at 100ms)")
    p.add_argument("--rust-bin", default=str(Path(__file__).resolve().parents[1]
                                             / "rust_ingest" / "target" / "release" / "feature_builder"))
    args = p.parse_args()

    parquet = Path(args.parquet)
    print(f"[parity] loading {parquet}")
    ts, bp, bq, ap, aq = load_flat(parquet)
    n = len(ts)
    print(f"[parity] {n} depth rows loaded")

    # Build indices: offset + stride * k, staying in bounds.
    start = args.offset
    end = min(n, start + args.n * args.stride)
    indices = np.arange(start, end, args.stride, dtype=np.int64)
    if len(indices) < 2:
        print(f"[parity] not enough rows (n={n})")
        return 2
    print(f"[parity] {len(indices)} sample indices")

    mid = np.zeros(n, dtype=np.float64)
    good = (bp[:, 0] > 0) & (ap[:, 0] > 0)
    mid[good] = 0.5 * (bp[good, 0] + ap[good, 0])

    # --- Optional streams ---
    trade_ts = trade_qty = trade_side = trade_price = None
    if args.trades:
        print(f"[parity] loading trades {args.trades}")
        trade_ts, trade_qty, trade_side, trade_price = load_trades(Path(args.trades))
        print(f"[parity] {len(trade_ts)} trade rows")
    funding_ts = funding_rate = None
    if args.funding:
        print(f"[parity] loading funding {args.funding}")
        funding_ts, funding_rate = load_scalar(Path(args.funding), ["funding_rate"])
        print(f"[parity] {len(funding_ts)} funding rows")
    deriv_ts = deriv_oi = deriv_ls = None
    if args.derivs:
        print(f"[parity] loading derivs {args.derivs}")
        deriv_ts, deriv_oi, deriv_ls = load_scalar(
            Path(args.derivs), ["open_interest", "long_short_ratio"])
        print(f"[parity] {len(deriv_ts)} deriv rows")
    eth_ts = eth_price = eth_qty = eth_side = None
    if args.eth:
        print(f"[parity] loading eth {args.eth}")
        eth_ts, eth_qty, eth_side, eth_price = load_trades(Path(args.eth), with_price=True)
        print(f"[parity] {len(eth_ts)} eth rows")
    cross_ex_data = None
    cross_paths = {"bybit": args.bybit, "okx": args.okx,
                   "bitget": args.bitget, "gateio": args.gateio}
    if any(cross_paths.values()):
        cross_ex_data = {}
        for ex, path in cross_paths.items():
            if not path:
                continue
            ct = pq.read_table(path).combine_chunks()
            cts = ct["timestamp"].chunk(0).to_numpy(zero_copy_only=True).astype(np.int64)
            cqty = ct["quantity"].chunk(0).to_numpy(zero_copy_only=True).astype(np.float64)
            cside = ct["is_seller"].chunk(0).to_numpy(zero_copy_only=False).astype(bool)
            if ex == "gateio":
                cqty = np.abs(cqty)
            csigned = np.where(cside, -cqty, cqty)
            cross_ex_data[ex] = (cts, csigned)
            print(f"[parity] {len(cts)} {ex} rows")

    # --- Python reference ---
    print("[parity] computing Python reference...")
    feat_py = python_features(ts, bp, bq, ap, aq, mid, indices,
                              trade_ts=trade_ts, trade_qty=trade_qty,
                              trade_side=trade_side, trade_price=trade_price,
                              funding_ts=funding_ts, funding_rate=funding_rate,
                              deriv_ts=deriv_ts, deriv_oi=deriv_oi, deriv_ls=deriv_ls,
                              eth_ts=eth_ts, eth_price=eth_price,
                              eth_qty=eth_qty, eth_side=eth_side,
                              cross_ex_data=cross_ex_data)

    # --- Rust ---
    with tempfile.TemporaryDirectory() as td:
        idx_path = Path(td) / "idx.npy"
        out_path = Path(td) / "feat.npy"
        np.save(idx_path, indices)
        print("[parity] invoking Rust feature_builder...")
        cmd = [args.rust_bin, "--depth", str(parquet),
               "--indices", str(idx_path), "--out", str(out_path)]
        if args.trades: cmd += ["--trades", args.trades]
        if args.funding: cmd += ["--funding", args.funding]
        if args.derivs: cmd += ["--derivs", args.derivs]
        if args.eth: cmd += ["--eth", args.eth]
        if args.bybit: cmd += ["--bybit", args.bybit]
        if args.okx: cmd += ["--okx", args.okx]
        if args.bitget: cmd += ["--bitget", args.bitget]
        if args.gateio: cmd += ["--gateio", args.gateio]
        subprocess.run(cmd, check=True)
        feat_rs = np.load(out_path)

    # --- Compare ---
    ok = True
    cols = list(LOB_COLS) + MICRO_DEPTH_COLS
    if args.trades:
        cols += TRADE_COLS + MICRO_TRADE_COLS
    if args.funding:
        cols += FUNDING_COLS
    if args.derivs:
        cols += DERIV_COLS
    if args.eth:
        cols += ETH_COLS
    if cross_ex_data:
        cols += CROSS_COLS
    print(f"\n[parity] shape py={feat_py.shape} rs={feat_rs.shape}")
    for c in cols:
        a = feat_py[:, c].astype(np.float64)
        b = feat_rs[:, c].astype(np.float64)
        if c == 29:
            # [29] OFI divergence is a sign-boundary feature: when upstream
            # [26] or [28] is ~0, f32 precision makes the sign non-deterministic
            # between numpy pairwise-sum and Rust running-sum. Exclude samples
            # where either magnitude is tiny — remaining samples must match.
            s26 = np.abs(feat_py[:, 26])
            s28 = np.abs(feat_py[:, 28])
            mask = (s26 > 0.1) & (s28 > 0.1)
            if mask.any():
                a = a[mask]
                b = b[mask]
            else:
                a = np.zeros(1); b = np.zeros(1)
        diff = np.abs(a - b)
        atol = ATOL[c]
        mx = float(diff.max())
        mean = float(diff.mean())
        status = "OK" if mx <= atol else "FAIL"
        if mx > atol:
            ok = False
        print(f"  col[{c:2d}]  max={mx:.3e}  mean={mean:.3e}  atol={atol:.1e}  {status}")

    if ok:
        print("\n[parity] PASS — Rust matches Python on LOB features")
        return 0
    print("\n[parity] FAIL — cols diverged")
    return 1


if __name__ == "__main__":
    sys.exit(main())
