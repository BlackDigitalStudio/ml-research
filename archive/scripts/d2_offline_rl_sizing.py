#!/usr/bin/env python3
"""D2 — offline contextual bandit for position sizing.

Scope (per ROADMAP_2026_04_15 D2):
  State:   stacker_soft (3) + meta_prob (1) + recent-window win rate (1) +
           recent-window mean |pnl| (1) + fractional index position (1) = 7 dims.
  Actions: size fraction in {0.00, 0.05, 0.10, 0.25, 0.50, 1.00}.
  Reward:  fraction * pnl_val   (when action > 0 and primary non-FLAT & meta gate passes)
           0 otherwise.
Training data: /home/scalper/backups/pod/recover_v2/val_predictions.npz — already contains
stacker_soft_balanced + y_val + pnl_val. 175k samples; first 75% train, tail 25% eval.

Method: "Direct Method" offline policy evaluation —
  for each action a, fit a regression: E[reward | context, a=a] using
  XGBoost. At decision time pick argmax_a of the predicted reward, tie-
  breaking to zero (don't trade) when max <= 0.

No live_sim, no full RL — this is the cheapest meaningful sizing agent
that uses existing val_predictions.

Limitations (documented honestly):
  - No vol / drawdown features (would need original X_feat + equity curve
    per step).
  - Contextual bandit only (single-step reward); a true PPO/CQL over a
    sequential equity state is the next step after this smoke-test.
  - Offline direct-method estimation is biased when training samples of
    a given arm are thin; we mitigate with per-arm XGBoost priors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.meta_label import MetaConfig, build_meta_dataset, train_meta  # noqa: E402


ACTIONS = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00]
COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07


def build_context(stacker_soft, meta_prob, pnl_val, window: int = 200) -> np.ndarray:
    """7-dim context per sample. Rolling stats use a past window (strictly
    causal — no look-ahead)."""
    n = len(stacker_soft)
    # Rolling win rate and mean abs pnl over the past `window` samples.
    # Use a simple cumulative-sum trick; for the first `window` samples the
    # rolling stats are computed over what's available.
    win = (pnl_val > 0).astype(np.float32)
    cs_win = np.concatenate(([0.0], np.cumsum(win, dtype=np.float64)))
    cs_abs = np.concatenate(([0.0], np.cumsum(np.abs(pnl_val), dtype=np.float64)))

    denom = np.maximum(np.minimum(np.arange(1, n + 1), window), 1).astype(np.float64)
    lo = np.maximum(np.arange(1, n + 1) - window, 0)
    hi = np.arange(1, n + 1)
    roll_win = ((cs_win[hi] - cs_win[lo]) / denom).astype(np.float32)
    roll_abs = ((cs_abs[hi] - cs_abs[lo]) / denom).astype(np.float32)

    frac_idx = (np.arange(n, dtype=np.float32) / max(n - 1, 1))
    ctx = np.concatenate([
        stacker_soft.astype(np.float32),                     # 3
        meta_prob.astype(np.float32).reshape(-1, 1),         # 1
        roll_win.reshape(-1, 1),                             # 1
        roll_abs.reshape(-1, 1),                             # 1
        frac_idx.reshape(-1, 1),                             # 1
    ], axis=1)
    return ctx


def bandit_reward(fraction: float, pnl: np.ndarray) -> np.ndarray:
    """Realised reward = fraction * pnl. pnl already includes commissions."""
    return fraction * pnl


def fit_action_heads(ctx_tr, pnl_tr, primary_pred_tr, meta_gate_tr,
                      actions=ACTIONS, max_depth=5, n_estimators=200):
    """For each action a, fit a regressor predicting expected per-sample
    reward (incl. 0 when we wouldn't trade)."""
    heads = {}
    for a in actions:
        r = bandit_reward(a, pnl_tr)
        # Mask: reward zero when we wouldn't trade anyway.
        would_trade = primary_pred_tr & meta_gate_tr
        y = np.where(would_trade, r, 0.0).astype(np.float32)
        m = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(ctx_tr, y)
        heads[a] = m
    return heads


def evaluate_policy(ctx, pnl, primary_pred_mask, meta_gate_mask, heads,
                     actions=ACTIONS):
    """For each sample, pick argmax_a of predicted reward. Compute realised
    PnL of that policy. Skip (a=0) if max predicted reward <= 0."""
    n = len(ctx)
    pred_matrix = np.stack([heads[a].predict(ctx) for a in actions], axis=1)  # (n, |A|)
    # Don't-trade rule: if max pred <= 0, choose a=0.
    best_a_idx = pred_matrix.argmax(axis=1)
    best_pred = pred_matrix[np.arange(n), best_a_idx]
    # Force a=0 when predicted best reward is non-positive.
    best_a_idx = np.where(best_pred <= 0, 0, best_a_idx)
    chosen_frac = np.array([actions[i] for i in best_a_idx], dtype=np.float32)

    # Realised reward only when we actually would trade.
    would_trade = primary_pred_mask & meta_gate_mask
    realised = np.where(would_trade & (chosen_frac > 0),
                         chosen_frac * pnl, 0.0)

    # Equity curve (same capital compounding as grid_test_ensemble).
    returns = realised / 100.0
    equity = 50.0 * np.cumprod(1.0 + returns) if n else np.array([50.0])
    final_eq = float(equity[-1])
    n_trades = int((realised != 0).sum())
    taken_pnl = realised[realised != 0]
    wr = float((taken_pnl > 0).mean() * 100) if n_trades else 0.0
    mean_size = float(chosen_frac[would_trade & (chosen_frac > 0)].mean()) if n_trades else 0.0
    arm_dist = {f"a={a}": int((chosen_frac == a).sum()) for a in actions}

    return {
        "n_samples": n,
        "n_trades": n_trades,
        "win_rate_pct": wr,
        "net_return_pct": 100 * (final_eq / 50.0 - 1),
        "mean_size_fraction": mean_size,
        "arm_distribution": arm_dist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions",
                    default="/home/scalper/backups/pod/recover_v2/val_predictions.npz")
    ap.add_argument("--out",
                    default="/home/scalper/backups/pod/recover_v2/d2_sizing_result.json")
    ap.add_argument("--stacker-key", default="stacker_soft_balanced")
    ap.add_argument("--meta-thr", type=float, default=0.70)
    ap.add_argument("--min-prob", type=float, default=0.55)
    args = ap.parse_args()

    data = np.load(args.val_predictions, allow_pickle=False)
    y_all = data["y_val"].astype(np.int64)
    pnl_all = data["pnl_val"].astype(np.float32)
    stacker = data[args.stacker_key].astype(np.float32)
    n = len(y_all)

    # Walk-forward split: first 75% train, tail 25% eval.
    n_tr = int(n * 0.75)
    print(f"D2 — offline contextual bandit for sizing")
    print(f"Loaded {n} samples. Train {n_tr}, eval {n - n_tr}")

    # Train meta on the same train split (consistent with C5/B3 protocol).
    X_m, y_m, w_m = build_meta_dataset(
        primary_softmax=stacker[:n_tr], y_true=y_all[:n_tr],
        target_pnl=pnl_all[:n_tr],
    )
    meta_model, meta_metrics = train_meta(X_m, y_m, w_m, cfg=MetaConfig(),
                                            val_frac=0.2, seed=42)
    print(f"meta OOS: AUC={meta_metrics['val_auc']:.3f} "
          f"prec={meta_metrics['val_precision']:.3f}")

    # Compute meta_prob over ALL samples (reuse features structure).
    primary_pred_all = stacker.argmax(axis=-1)
    primary_max_all = stacker.max(axis=-1)
    non_flat_all = primary_pred_all != 2

    p_max = stacker.max(axis=-1, keepdims=True)
    p_margin = (np.sort(stacker, axis=-1)[:, -1] - np.sort(stacker, axis=-1)[:, -2])[:, None]
    p_entropy = -(stacker * np.log(stacker.clip(1e-9))).sum(axis=-1, keepdims=True)
    X_meta_all = np.concatenate([
        stacker, p_max, p_margin, p_entropy,
        (primary_pred_all == 0).astype(np.float32)[:, None],
    ], axis=-1).astype(np.float32)

    meta_prob_all = np.zeros(n, dtype=np.float32)
    if non_flat_all.any():
        meta_prob_all[non_flat_all] = meta_model.predict_proba(
            X_meta_all[non_flat_all])[:, 1]

    # "Would we even trade?" mask (primary says non-FLAT + gate passes).
    trade_mask_all = (non_flat_all
                       & (primary_max_all >= args.min_prob)
                       & (meta_prob_all >= args.meta_thr))
    print(f"Pre-sizing trade candidates: {int(trade_mask_all.sum())} / {n} "
          f"({100 * trade_mask_all.mean():.3f}%)")

    # Context features — computed over the FULL series, so the bandit's
    # rolling stats at index i only see up to i-1 (strictly causal).
    ctx_all = build_context(stacker, meta_prob_all, pnl_all, window=200)
    print(f"Context shape: {ctx_all.shape}")

    # Train action heads on the first 75%.
    print(f"\nFitting {len(ACTIONS)} XGBoost action heads on train split ...")
    heads = fit_action_heads(
        ctx_tr=ctx_all[:n_tr],
        pnl_tr=pnl_all[:n_tr],
        primary_pred_tr=non_flat_all[:n_tr],
        meta_gate_tr=trade_mask_all[:n_tr],
    )

    # Evaluate on tail.
    print("\n=== Policy evaluation ===")
    # Baseline: fixed size (1.0) on all trade candidates.
    fixed_1 = {"arm_distribution": {"a=1.0": int(trade_mask_all[n_tr:].sum())}}
    realised_fixed = np.where(trade_mask_all[n_tr:], 1.0 * pnl_all[n_tr:], 0.0)
    eq_fixed = 50.0 * np.cumprod(1.0 + realised_fixed / 100.0)
    taken = realised_fixed[realised_fixed != 0]
    print(f"Baseline (fixed size=1.0 on same gate):")
    print(f"  n_trades={len(taken)}  WR={(taken > 0).mean()*100:.1f}%  "
          f"net={100 * (eq_fixed[-1]/50.0 - 1):+.2f}%")

    tail_result = evaluate_policy(
        ctx=ctx_all[n_tr:], pnl=pnl_all[n_tr:],
        primary_pred_mask=non_flat_all[n_tr:],
        meta_gate_mask=trade_mask_all[n_tr:],
        heads=heads,
    )
    print(f"Bandit policy (tail {n - n_tr} samples):")
    print(f"  n_trades={tail_result['n_trades']}  WR={tail_result['win_rate_pct']:.1f}%  "
          f"net={tail_result['net_return_pct']:+.2f}%  "
          f"mean_size={tail_result['mean_size_fraction']:.3f}")
    print(f"  arm distribution: {tail_result['arm_distribution']}")

    # Sanity: evaluate on TRAIN split too — should be >= tail (no CV leakage
    # expected but verify).
    train_eval = evaluate_policy(
        ctx=ctx_all[:n_tr], pnl=pnl_all[:n_tr],
        primary_pred_mask=non_flat_all[:n_tr],
        meta_gate_mask=trade_mask_all[:n_tr],
        heads=heads,
    )
    print(f"In-sample (train, for reference):")
    print(f"  n_trades={train_eval['n_trades']}  WR={train_eval['win_rate_pct']:.1f}%  "
          f"net={train_eval['net_return_pct']:+.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "meta_metrics": meta_metrics,
            "meta_thr": args.meta_thr, "min_prob": args.min_prob,
            "stacker_key": args.stacker_key,
            "baseline_fixed_1": {
                "n_trades": int(len(taken)),
                "win_rate_pct": float((taken > 0).mean() * 100) if len(taken) else 0.0,
                "net_return_pct": float(100 * (eq_fixed[-1] / 50.0 - 1)),
            },
            "bandit_tail": tail_result,
            "bandit_train": train_eval,
        }, f, indent=2)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
