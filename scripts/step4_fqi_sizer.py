#!/usr/bin/env python3
"""Step 4 — Fitted Q-Iteration offline sizer with richer sequential state.

D2 v1 was a contextual bandit — one-shot policy learning from
(context, reward) pairs. It failed to beat the grid-optimized gate because
its state didn't include equity / drawdown / streak dynamics.

This Step 4 agent uses Fitted Q-Iteration (FQI), which is offline RL built
on XGBoost regressors. It sequentially iterates the Bellman operator:

    Q_0(s, a) = 0
    Q_{t+1}(s, a) = r + γ · max_{a'} Q_t(s', a')

After ~5-10 iterations Q converges to an approximate action-value. At
inference we pick a = argmax_a Q(s, a), tie-breaking to "don't trade" when
max(Q) ≤ 0.

State (12 dims):
    stacker_soft (3), meta_prob (1), rolling win rate (1), rolling mean pnl (1),
    current drawdown from peak equity (1), equity/initial (1), rolling
    trade frequency (1), avg trade pnl in last 200 samples (1),
    stacker max margin (1), primary argmax == UP one-hot (1)

Actions: fraction ∈ {0.00, 0.05, 0.10, 0.25, 0.50, 1.00}

Reward: `fraction · pnl_val` when primary+meta would trade (non-FLAT gate),
        else 0 regardless of action. This mirrors real execution where
        the primary decides direction + meta gates; sizing only scales
        an already-decided trade.

Eval: apply learned policy on tail 25% + compare net equity vs
  - `fixed1`: always size=1.0 on gate pass (D2 baseline)
  - `bandit`: D2 v1 offline contextual bandit (for delta attribution)
  - `no_trade`: size=0 everywhere (reference lower bound)
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
INITIAL_CAPITAL = 50.0
GAMMA = 0.99


def build_state(stacker_soft, meta_prob, pnl_val, gate_mask,
                 rolling_window: int = 200) -> np.ndarray:
    """Per-sample 12-dim state (strictly causal)."""
    n = len(stacker_soft)
    # Primary / meta (4)
    p_max = stacker_soft.max(axis=-1, keepdims=True)                     # (n,1)
    p_margin = (np.sort(stacker_soft, axis=-1)[:, -1]
                 - np.sort(stacker_soft, axis=-1)[:, -2])[:, None]        # (n,1)
    is_up = (stacker_soft.argmax(axis=-1) == 0).astype(np.float32)[:, None]

    # Causal rolling stats on pnl (pnl_val is the labeled PnL we would earn
    # at fraction 1.0 — training sees this only at past indices).
    win = (pnl_val > 0).astype(np.float32)
    cs_w = np.concatenate(([0.0], np.cumsum(win, dtype=np.float64)))
    cs_p = np.concatenate(([0.0], np.cumsum(pnl_val, dtype=np.float64)))
    hi = np.arange(1, n + 1)
    lo = np.maximum(hi - rolling_window, 0)
    denom = np.maximum((hi - lo).astype(np.float64), 1.0)
    roll_wr = ((cs_w[hi] - cs_w[lo]) / denom).astype(np.float32)
    roll_mean = ((cs_p[hi] - cs_p[lo]) / denom).astype(np.float32)

    # Cumulative equity from "what-if fixed-size gate" trajectory (training
    # fake — gives the agent information about its would-be equity curve).
    fake_ret = gate_mask.astype(np.float32) * pnl_val / 100.0
    fake_eq = INITIAL_CAPITAL * np.cumprod(1.0 + fake_ret, dtype=np.float64)
    fake_eq_shift = np.concatenate(([INITIAL_CAPITAL], fake_eq[:-1]))
    peak = np.maximum.accumulate(fake_eq_shift)
    dd = ((peak - fake_eq_shift) / np.maximum(peak, 1e-12)).astype(np.float32)
    eq_ratio = (fake_eq_shift / INITIAL_CAPITAL).astype(np.float32)

    # Rolling trade frequency + avg-pnl on actual gate trades (past only).
    trades = gate_mask.astype(np.float32) * pnl_val
    cs_t  = np.concatenate(([0.0], np.cumsum(trades, dtype=np.float64)))
    cs_n  = np.concatenate(([0.0], np.cumsum(gate_mask.astype(np.float64))))
    trade_freq = ((cs_n[hi] - cs_n[lo]) / denom).astype(np.float32)
    trade_mean = np.divide(
        (cs_t[hi] - cs_t[lo]),
        np.maximum(cs_n[hi] - cs_n[lo], 1.0)
    ).astype(np.float32)

    state = np.concatenate([
        stacker_soft.astype(np.float32),       # 3
        meta_prob.reshape(-1, 1).astype(np.float32),  # 1
        roll_wr.reshape(-1, 1),                 # 1
        roll_mean.reshape(-1, 1),               # 1
        dd.reshape(-1, 1),                      # 1
        eq_ratio.reshape(-1, 1),                # 1
        trade_freq.reshape(-1, 1),              # 1
        trade_mean.reshape(-1, 1),              # 1
        p_margin.astype(np.float32),            # 1
        is_up,                                  # 1
    ], axis=1)
    return state


def fit_q(state: np.ndarray, action: np.ndarray, q_target: np.ndarray,
           n_estimators: int = 200, max_depth: int = 5):
    """One XGBoost Q-head per action. Returns list indexed by action idx."""
    heads = []
    for ai, a in enumerate(ACTIONS):
        mask = action == ai
        if mask.sum() < 50:
            heads.append(None)
            continue
        m = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(state[mask], q_target[mask])
        heads.append(m)
    return heads


def predict_q(heads, state: np.ndarray) -> np.ndarray:
    """(n, |A|) action-value matrix. Missing-head actions → -inf."""
    n = len(state)
    q = np.full((n, len(ACTIONS)), -np.inf, dtype=np.float64)
    for ai, h in enumerate(heads):
        if h is None:
            continue
        q[:, ai] = h.predict(state)
    return q


def run_policy(heads, state: np.ndarray, pnl: np.ndarray,
                gate_mask: np.ndarray) -> dict:
    q = predict_q(heads, state)
    best_a_idx = np.argmax(q, axis=1)
    best_q = q[np.arange(len(state)), best_a_idx]
    # don't-trade rule: skip if best Q non-positive.
    best_a_idx = np.where(best_q <= 0, 0, best_a_idx)
    chosen_frac = np.array([ACTIONS[i] for i in best_a_idx], dtype=np.float32)
    realised = np.where(gate_mask & (chosen_frac > 0),
                          chosen_frac * pnl, 0.0)
    ret = realised / 100.0
    eq = INITIAL_CAPITAL * np.cumprod(1.0 + ret) if len(ret) else np.array([INITIAL_CAPITAL])
    n_t = int((realised != 0).sum())
    taken = realised[realised != 0]
    return {
        "n_trades": n_t,
        "win_rate_pct": float((taken > 0).mean() * 100) if n_t else 0.0,
        "net_return_pct": 100 * (float(eq[-1]) / INITIAL_CAPITAL - 1),
        "mean_size": float(chosen_frac[gate_mask & (chosen_frac > 0)].mean()) if n_t else 0.0,
        "arm_dist": {f"{a}": int((chosen_frac == a).sum()) for a in ACTIONS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions",
                    default="/home/scalper/backups/pod/recover_v2/val_predictions.npz")
    ap.add_argument("--out",
                    default="/home/scalper/backups/pod/recover_v2/step4_fqi_result.json")
    ap.add_argument("--stacker-key", default="stacker_soft_balanced")
    ap.add_argument("--meta-thr", type=float, default=0.70)
    ap.add_argument("--min-prob", type=float, default=0.55)
    ap.add_argument("--fqi-iters", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    args = ap.parse_args()

    data = np.load(args.val_predictions, allow_pickle=False)
    y_all = data["y_val"].astype(np.int64)
    pnl_all = data["pnl_val"].astype(np.float32)
    stacker = data[args.stacker_key].astype(np.float32)
    n = len(y_all)

    n_tr = int(n * 0.75)
    print(f"[step4] n={n}, train={n_tr}, eval={n - n_tr}")

    # Walk-forward meta on train.
    X_m, y_m, w_m = build_meta_dataset(
        primary_softmax=stacker[:n_tr], y_true=y_all[:n_tr],
        target_pnl=pnl_all[:n_tr],
    )
    meta, meta_metrics = train_meta(X_m, y_m, w_m, cfg=MetaConfig(), val_frac=0.2, seed=42)
    print(f"[step4] meta AUC={meta_metrics['val_auc']:.3f} "
          f"prec={meta_metrics['val_precision']:.3f}")

    # Compute meta_prob + trade gate over FULL series.
    primary_pred = stacker.argmax(axis=-1)
    primary_max = stacker.max(axis=-1)
    non_flat = primary_pred != 2
    p_max = stacker.max(axis=-1, keepdims=True)
    p_marg = (np.sort(stacker, axis=-1)[:, -1] - np.sort(stacker, axis=-1)[:, -2])[:, None]
    p_ent = -(stacker * np.log(stacker.clip(1e-9))).sum(axis=-1, keepdims=True)
    X_meta_all = np.concatenate([
        stacker, p_max, p_marg, p_ent,
        (primary_pred == 0).astype(np.float32)[:, None],
    ], axis=-1).astype(np.float32)
    meta_prob = np.zeros(n, dtype=np.float32)
    if non_flat.any():
        meta_prob[non_flat] = meta.predict_proba(X_meta_all[non_flat])[:, 1]
    gate = non_flat & (primary_max >= args.min_prob) & (meta_prob >= args.meta_thr)
    print(f"[step4] gate candidates: {int(gate.sum())} / {n} "
          f"({100 * gate.mean():.3f}%)")

    # State computation on full series (causal rolling stats).
    state = build_state(stacker, meta_prob, pnl_all, gate)
    print(f"[step4] state shape: {state.shape}")

    # Offline dataset for FQI — we pair (s_i, a_i, r_i, s_{i+1}) where
    # a_i is a random action sampled uniformly (so every action has
    # coverage) and r_i = (a_i * pnl_val) if gate_i else 0.
    rng = np.random.default_rng(0)
    action_idx_train = rng.integers(0, len(ACTIONS), size=n_tr)
    frac_train = np.array([ACTIONS[i] for i in action_idx_train], dtype=np.float32)
    reward_train = np.where(gate[:n_tr] & (frac_train > 0),
                              frac_train * pnl_all[:n_tr], 0.0)

    # Terminal state flag — last train sample has no s' available.
    not_terminal = np.ones(n_tr, dtype=np.float32)
    not_terminal[-1] = 0.0

    # FQI loop
    state_tr = state[:n_tr]
    # s' is state[i+1] for i in [0, n_tr-1); clamp last to self.
    state_next = np.vstack([state_tr[1:], state_tr[-1:]])

    heads = [None] * len(ACTIONS)
    for it in range(args.fqi_iters):
        if it == 0:
            q_target = reward_train
        else:
            # Bellman target: r + γ · max_a' Q(s', a')
            q_next = predict_q(heads, state_next)
            max_q_next = np.nanmax(np.where(np.isfinite(q_next), q_next, -np.inf), axis=1)
            max_q_next = np.where(np.isfinite(max_q_next), max_q_next, 0.0)
            q_target = reward_train + args.gamma * not_terminal * max_q_next
        heads = fit_q(state_tr, action_idx_train, q_target)
        trained = sum(1 for h in heads if h is not None)
        print(f"  FQI iter {it+1}/{args.fqi_iters}: trained {trained}/{len(ACTIONS)} heads, "
              f"q_target mean={q_target.mean():+.4f} std={q_target.std():.4f}")

    # Evaluate on tail.
    print("\n=== Evaluation ===")

    # Baselines.
    tail_pnl = pnl_all[n_tr:]
    tail_gate = gate[n_tr:]
    tail_state = state[n_tr:]

    # fixed1: always take gate, size=1
    realised_f1 = tail_gate.astype(np.float32) * tail_pnl
    eq_f1 = INITIAL_CAPITAL * np.cumprod(1.0 + realised_f1 / 100.0)
    taken_f1 = realised_f1[realised_f1 != 0]
    net_f1 = 100 * (float(eq_f1[-1]) / INITIAL_CAPITAL - 1)
    print(f"fixed_size=1.0 baseline: n={len(taken_f1)} "
          f"WR={(taken_f1 > 0).mean()*100:.1f}% net={net_f1:+.2f}%")

    # no_trade
    print(f"no_trade reference:     n=0 net=+0.00%")

    # fqi
    fqi_result = run_policy(heads, tail_state, tail_pnl, tail_gate)
    print(f"FQI policy:             n={fqi_result['n_trades']} "
          f"WR={fqi_result['win_rate_pct']:.1f}% "
          f"net={fqi_result['net_return_pct']:+.2f}% "
          f"mean_size={fqi_result['mean_size']:.3f}")
    print(f"  arm distribution: {fqi_result['arm_dist']}")

    # In-sample FQI on train — reference only.
    train_result = run_policy(heads, state[:n_tr], pnl_all[:n_tr], gate[:n_tr])
    print(f"FQI in-sample (train):  n={train_result['n_trades']} "
          f"WR={train_result['win_rate_pct']:.1f}% "
          f"net={train_result['net_return_pct']:+.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "meta_metrics": meta_metrics,
            "baseline_fixed1_net": net_f1,
            "baseline_fixed1_n": int(len(taken_f1)),
            "fqi_tail": fqi_result,
            "fqi_train": train_result,
            "config": {
                "meta_thr": args.meta_thr, "min_prob": args.min_prob,
                "fqi_iters": args.fqi_iters, "gamma": args.gamma,
                "stacker_key": args.stacker_key,
            },
        }, f, indent=2)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
