#!/usr/bin/env python3
"""End-to-end backtest of the L1→L2→L3→Kelly→live_sim ensemble.

Consumes outputs from `scripts/bakeoff_v2.py`:
    runs/bakeoff_v2/
        <arch>.pt             # per-arch model weights
        stacker.json          # L2 XGBoost stacker
        meta.json             # L3 XGBoost meta-labeler
        val_predictions.npz   # validation softmaxes + y_val + pnl_val

Plus a training cache (for inference on held-out samples).

Pipeline per sample:
    L1:    primaries → softmax                (B, 5, 3)
    L2:    stacker(concat softmax + feats)    → (B, 3)  primary decision
    L3:    meta-label(stacker softmax + feats) → (B,)   take/skip binary
    Kelly: size_trades_batch(...)             → fraction of capital per trade
    Sim:   live_sim.simulate_trade with Kelly-weighted position

Outputs:
    - 7 canonical business metrics (FullTP%, FullSL%, Timeout%, etc.)
    - PnL distribution (gross + net)
    - WR (win rate), Sharpe, max drawdown
    - Per-direction breakdown (LONG/SHORT/FLAT)
    - Kelly utilization stats (mean/median fraction when traded)

This is a one-shot backtest — no walk-forward retraining. For proper
walk-forward with model refitting use scripts/backtest.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sizing import KellyConfig, size_trades_batch  # noqa: E402
from src.models.stacking import predict_stacked  # noqa: E402
from src.models.meta_label import MetaConfig, combine as meta_combine  # noqa: E402


def _compute_canonical_metrics(reasons: list[str]) -> dict[str, float]:
    """7 canonical buckets from live_sim exit_reason strings."""
    FULL_TP = {"tp_hit"}
    FULL_SL = {"sl_hit", "fast_fill_adverse", "fast_fill_sl"}
    TIMEOUT = {"timeout_limit", "timeout_market", "no_forward_data"}
    TRAILING = {"trailing_sl_1", "trailing_sl_2",
                "partial_plus_trailing_sl_1", "partial_plus_trailing_sl_2"}
    PARTIAL_TP = {"partial_plus_tp"}
    n = len(reasons)
    if n == 0:
        return {k: 0.0 for k in
                ["full_tp_pct", "full_sl_pct", "timeout_pct",
                 "trailing_stop_pct", "partial_tp_only_pct"]}
    def _count(s):
        return 100 * sum(1 for r in reasons if r in s) / n
    return {
        "n_trades": n,
        "full_tp_pct": _count(FULL_TP),
        "full_sl_pct": _count(FULL_SL),
        "timeout_pct": _count(TIMEOUT),
        "trailing_stop_pct": _count(TRAILING),
        "partial_tp_only_pct": _count(PARTIAL_TP),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown as fraction of peak (positive number)."""
    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / np.maximum(peaks, 1e-12)
    return float(dd.max()) if len(dd) else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bakeoff-dir", required=True,
                   help="Output dir of scripts/bakeoff_v2.py with models + val_predictions.npz")
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--kelly-max-fraction", type=float, default=19.0)
    p.add_argument("--kelly-min-prob", type=float, default=0.52)
    p.add_argument("--meta-threshold", type=float, default=0.5)
    p.add_argument("--tp-pct", type=float, default=0.20)
    p.add_argument("--sl-pct", type=float, default=0.10)
    p.add_argument("--commission-win-pct", type=float, default=0.04)
    p.add_argument("--commission-loss-pct", type=float, default=0.07)
    p.add_argument("--initial-capital", type=float, default=50.0)
    p.add_argument("--out", default=None, help="write results JSON here")
    args = p.parse_args()

    bdir = Path(args.bakeoff_dir)
    preds = np.load(bdir / "val_predictions.npz")
    y_val = preds["y_val"]
    pnl_val = preds["pnl_val"]
    stacker_soft = preds["stacker_soft"]
    # L1 softmaxes by tag
    l1_tags = [k[5:] for k in preds.files if k.startswith("soft_")]
    l1_softmaxes = [preds[f"soft_{t}"] for t in l1_tags]
    print(f"[bt] loaded {len(l1_tags)} L1 models: {l1_tags}")
    print(f"[bt] {len(y_val)} samples on val split")

    # Load L3 meta-learner
    meta_path = bdir / "meta.json"
    if meta_path.exists():
        meta = xgb.XGBClassifier()
        meta.load_model(str(meta_path))
        print(f"[bt] loaded meta-labeler from {meta_path}")
    else:
        print(f"[bt] WARNING: {meta_path} not found, skipping meta-label filter")
        meta = None

    # Compose predictions — stacker output is the primary.
    primary_soft = stacker_soft
    primary_pred = primary_soft.argmax(axis=-1)
    primary_max = primary_soft.max(axis=-1)

    # === Apply meta-label filter ===
    if meta is not None:
        take_meta, _, conf = meta_combine(
            primary_soft, meta, X_feat=None,
            cfg=MetaConfig(meta_threshold=args.meta_threshold,
                            min_primary_conf=args.kelly_min_prob),
        )
    else:
        take_meta = (primary_pred != 2) & (primary_max >= args.kelly_min_prob)
        conf = primary_max

    n_after_meta = int(take_meta.sum())
    print(f"[bt] meta-label kept {n_after_meta}/{len(y_val)} "
          f"({100*n_after_meta/len(y_val):.1f}%)")

    # === Kelly sizing on kept trades ===
    # Use primary_max as probability estimate for Kelly.
    cfg_kelly = KellyConfig(
        fraction=args.kelly_fraction,
        max_position_fraction=args.kelly_max_fraction,
        min_probability=args.kelly_min_prob,
    )
    sizing = size_trades_batch(
        p_win=primary_max,
        win_pct=args.tp_pct,
        loss_pct=args.sl_pct,
        commission_win_pct=args.commission_win_pct,
        commission_loss_pct=args.commission_loss_pct,
        cfg=cfg_kelly,
    )
    take_kelly = sizing["take"]
    final_take = take_meta & take_kelly
    n_traded = int(final_take.sum())
    print(f"[bt] after Kelly filter: {n_traded} trades "
          f"({100*n_traded/len(y_val):.1f}% of val)")

    # === Realized PnL (using val_predictions pnl_val as live_sim ground truth) ===
    # pnl_val is the net PnL % of notional the sample would have earned if traded
    # with unit size. Multiply by Kelly fraction for realistic position scaling.
    fractions = sizing["fraction"]
    # For trades we take: capital-weighted PnL = fraction × pnl_val (both in %)
    # PnL dollar contribution per trade on initial_capital:
    #   dollar_pnl = initial_capital × (fraction × pnl_val / 100)
    weighted_pnl_pct = np.where(final_take, fractions * pnl_val, 0.0)
    # Cumulative equity curve (compounded)
    returns_frac = weighted_pnl_pct / 100.0
    equity = args.initial_capital * np.cumprod(1.0 + returns_frac)
    final_equity = float(equity[-1]) if len(equity) > 0 else args.initial_capital

    # Trade outcomes (only on taken trades)
    taken_idx = np.where(final_take)[0]
    pnl_taken = pnl_val[taken_idx]
    wins = int((pnl_taken > 0).sum())
    total_trades = len(taken_idx)
    wr = (100 * wins / total_trades) if total_trades else 0.0

    # Sharpe-like on per-trade returns (annualized rough — caller to interpret)
    if len(pnl_taken) > 1:
        sharpe = float(pnl_taken.mean() / (pnl_taken.std() + 1e-9))
    else:
        sharpe = 0.0
    max_dd = _max_drawdown(equity)

    # Directional breakdown
    dirs = primary_pred[taken_idx]
    long_n = int((dirs == 0).sum()); short_n = int((dirs == 1).sum())

    # Kelly utilization
    util_mean = float(fractions[final_take].mean()) if total_trades else 0.0
    util_max = float(fractions[final_take].max()) if total_trades else 0.0

    result = {
        "n_val_samples": int(len(y_val)),
        "n_after_meta": n_after_meta,
        "n_traded": total_trades,
        "trade_rate_pct": 100 * total_trades / max(len(y_val), 1),
        "win_rate_pct": wr,
        "sharpe_per_trade": sharpe,
        "max_drawdown_pct": 100 * max_dd,
        "initial_capital": args.initial_capital,
        "final_equity": final_equity,
        "net_return_pct": 100 * (final_equity / args.initial_capital - 1),
        "kelly_utilization_mean": util_mean,
        "kelly_utilization_max": util_max,
        "long_count": long_n,
        "short_count": short_n,
        "tp_pct_used": args.tp_pct,
        "sl_pct_used": args.sl_pct,
        "kelly_fraction_cfg": args.kelly_fraction,
        "meta_threshold_cfg": args.meta_threshold,
    }

    print("\n=== Backtest result ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:30s}  {v:+.4f}")
        else:
            print(f"  {k:30s}  {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwritten to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
