#!/usr/bin/env python3
"""Grid-search strategy configurations over a trained bake-off.

Designed to run on CPU (Contabo or pod) while GPU is still training other
architectures — grid-search is inference-only + fast Python, and
backtest_ensemble logic runs in seconds per config.

Watches a directory for val_predictions.npz + stacker + meta files. Each
time new files appear OR --once is passed, runs the full grid:

    tp_pct × sl_pct × kelly_fraction × meta_threshold × min_probability

For each combo computes: net_return_pct, win_rate, sharpe_per_trade,
max_drawdown, n_trades. Writes sorted leaderboard JSON.

Usage:
    # Run once against a finished bakeoff dir:
    python scripts/grid_test_ensemble.py \\
        --bakeoff-dir /home/scalper/backups/pod/current/runs/bakeoff_v2 \\
        --once

    # Watch for updates (poll every 30s):
    python scripts/grid_test_ensemble.py \\
        --bakeoff-dir /home/scalper/backups/pod/current/runs/bakeoff_v2 \\
        --watch --interval 30
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sizing import KellyConfig, size_trades_batch  # noqa: E402


# Default grid — 5 × 4 × 4 × 4 × 3 = 960 configs, runs in seconds.
DEFAULT_TP = [0.10, 0.15, 0.20, 0.30, 0.40]
DEFAULT_SL = [0.05, 0.10, 0.15, 0.20]
DEFAULT_KELLY_FRAC = [0.10, 0.25, 0.50, 1.00]
DEFAULT_META_THR = [0.30, 0.50, 0.60, 0.70]
DEFAULT_MIN_PROB = [0.50, 0.55, 0.60]

COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07
INITIAL_CAPITAL = 50.0


def _load_bakeoff(bakeoff_dir: Path) -> dict:
    """Load val_predictions.npz + optional meta.json."""
    vp = bakeoff_dir / "val_predictions.npz"
    if not vp.exists():
        raise FileNotFoundError(f"val_predictions.npz not found at {vp}")
    preds = dict(np.load(vp, allow_pickle=False))

    meta = None
    meta_path = bakeoff_dir / "meta.json"
    if meta_path.exists():
        meta = xgb.XGBClassifier()
        meta.load_model(str(meta_path))

    return {"preds": preds, "meta": meta, "dir": bakeoff_dir}


def _score_config(preds: dict, meta, tp: float, sl: float,
                   kelly_frac: float, meta_thr: float, min_prob: float) -> dict:
    """Score one config."""
    y_val = preds["y_val"]
    pnl_val = preds["pnl_val"]
    stacker_soft = preds["stacker_soft"]
    primary_pred = stacker_soft.argmax(axis=-1)
    primary_max = stacker_soft.max(axis=-1)
    non_flat = primary_pred != 2  # FLAT = class 2

    # Meta filter — if meta model present, compute meta_prob on stacker_soft.
    if meta is not None:
        # Reconstruct meta features same as src/models/meta_label.py
        p_soft = stacker_soft
        p_max = p_soft.max(axis=-1, keepdims=True)
        p_margin = (np.sort(p_soft, axis=-1)[:, -1] - np.sort(p_soft, axis=-1)[:, -2])[:, None]
        p_entropy = -(p_soft * np.log(p_soft.clip(1e-9))).sum(axis=-1, keepdims=True)
        X_meta = np.concatenate([
            p_soft, p_max, p_margin, p_entropy,
            (primary_pred == 0).astype(np.float32)[:, None],  # UP indicator
        ], axis=-1).astype(np.float32)
        meta_prob = np.zeros(len(y_val), dtype=np.float32)
        if non_flat.any():
            meta_prob[non_flat] = meta.predict_proba(X_meta[non_flat])[:, 1]
    else:
        meta_prob = primary_max

    # Combine gates
    take_meta = non_flat & (primary_max >= min_prob) & (meta_prob >= meta_thr)

    # Kelly sizing on candidate trades
    cfg_k = KellyConfig(
        fraction=kelly_frac, max_position_fraction=19.0,
        min_probability=min_prob,
    )
    sizing = size_trades_batch(
        p_win=primary_max, win_pct=tp, loss_pct=sl,
        commission_win_pct=COMMISSION_WIN, commission_loss_pct=COMMISSION_LOSS,
        cfg=cfg_k,
    )
    final_take = take_meta & sizing["take"]
    fractions = sizing["fraction"]

    # Realised PnL per trade = fraction × pnl_val (pnl_val already includes commissions)
    weighted = np.where(final_take, fractions * pnl_val, 0.0)
    returns = weighted / 100.0
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    final_eq = float(equity[-1]) if len(equity) else INITIAL_CAPITAL

    taken_idx = np.where(final_take)[0]
    n_trades = len(taken_idx)
    pnl_taken = pnl_val[taken_idx]
    wr = float((pnl_taken > 0).mean() * 100) if n_trades else 0.0
    sharpe = float(pnl_taken.mean() / (pnl_taken.std() + 1e-9)) if n_trades > 1 else 0.0

    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([INITIAL_CAPITAL])
    max_dd = float(((peaks - equity) / np.maximum(peaks, 1e-12)).max()) if len(equity) else 0.0

    return {
        "tp_pct": tp, "sl_pct": sl, "kelly_frac": kelly_frac,
        "meta_threshold": meta_thr, "min_probability": min_prob,
        "n_trades": n_trades,
        "trade_rate_pct": 100 * n_trades / max(len(y_val), 1),
        "win_rate_pct": wr,
        "sharpe_per_trade": sharpe,
        "max_drawdown_pct": 100 * max_dd,
        "net_return_pct": 100 * (final_eq / INITIAL_CAPITAL - 1),
        "final_equity": final_eq,
    }


def run_grid(bakeoff: dict, tp_grid, sl_grid, kelly_grid, meta_grid, prob_grid,
              top_k: int = 50) -> dict:
    configs = list(itertools.product(tp_grid, sl_grid, kelly_grid, meta_grid, prob_grid))
    print(f"[grid] scoring {len(configs)} configs...")
    rows = []
    t0 = time.time()
    for tp, sl, kf, mt, mp in configs:
        row = _score_config(bakeoff["preds"], bakeoff["meta"], tp, sl, kf, mt, mp)
        rows.append(row)
    dt = time.time() - t0
    # Sort multiple ways, keep top K each
    by_pnl = sorted(rows, key=lambda r: -r["net_return_pct"])[:top_k]
    by_wr = sorted(rows, key=lambda r: -r["win_rate_pct"])[:top_k]
    by_sharpe = sorted(rows, key=lambda r: -r["sharpe_per_trade"])[:top_k]
    return {
        "n_configs": len(configs), "elapsed_seconds": dt,
        "top_by_pnl": by_pnl, "top_by_wr": by_wr, "top_by_sharpe": by_sharpe,
    }


def _signature(bakeoff_dir: Path) -> tuple:
    """Tuple of mtime+size for key files — changes when bake-off updates."""
    files = ["val_predictions.npz", "meta.json", "stacker.json"]
    sig = []
    for f in files:
        p = bakeoff_dir / f
        if p.exists():
            st = p.stat()
            sig.append((f, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bakeoff-dir", required=True)
    p.add_argument("--once", action="store_true",
                   help="run once and exit; else watch for updates")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=30, help="poll interval seconds")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--out", default=None, help="write full results JSON")
    args = p.parse_args()

    bakeoff_dir = Path(args.bakeoff_dir)

    def _run_once():
        try:
            bakeoff = _load_bakeoff(bakeoff_dir)
        except FileNotFoundError as e:
            print(f"[grid] waiting for bakeoff to appear: {e}")
            return None
        n_models = sum(1 for k in bakeoff["preds"].keys() if k.startswith("soft_"))
        print(f"[grid] loaded bakeoff with {n_models} L1 models, "
              f"{len(bakeoff['preds']['y_val'])} val samples")
        res = run_grid(bakeoff, DEFAULT_TP, DEFAULT_SL, DEFAULT_KELLY_FRAC,
                        DEFAULT_META_THR, DEFAULT_MIN_PROB, top_k=args.top_k)
        print(f"[grid] best by net_return: "
              f"{res['top_by_pnl'][0]['net_return_pct']:+.2f}%  "
              f"(tp={res['top_by_pnl'][0]['tp_pct']} sl={res['top_by_pnl'][0]['sl_pct']}, "
              f"WR={res['top_by_pnl'][0]['win_rate_pct']:.1f}%, "
              f"n_trades={res['top_by_pnl'][0]['n_trades']})")
        print(f"[grid] best by WR: "
              f"{res['top_by_wr'][0]['win_rate_pct']:.1f}%  "
              f"(n_trades={res['top_by_wr'][0]['n_trades']})")
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(res, f, indent=2)
            print(f"[grid] saved → {out}")
        return res

    if args.once or not args.watch:
        _run_once()
        return 0

    # Watch mode
    last_sig = None
    while True:
        sig = _signature(bakeoff_dir)
        if sig != last_sig and sig:
            print(f"[grid] update detected, re-running at {time.strftime('%H:%M:%S')}")
            _run_once()
            last_sig = sig
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
