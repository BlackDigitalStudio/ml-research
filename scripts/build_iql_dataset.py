#!/usr/bin/env python3
"""Build (state, action, reward) dataset for offline RL (IQL) training.

Uses Rust `simulate_labels` as the reward generator. Because the simulator
takes per-sample arrays of `(tp, sl, timeout)`, we enumerate the discrete
action space by calling it once per `(tp, sl, timeout)` bucket — one call
yields `pnl_long` + `pnl_short` for all N samples, so 7×4×5 = 140 calls
cover the full (enter-long, enter-short) × TP×SL×timeout space. The SKIP
action always has reward 0. Kelly is post-hoc multiplicative — we fix it
at 0.25 in the dataset and scale at inference.

Action discretisation (420 actions total):
    TP       : {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}  (7)
    SL       : {0.08, 0.10, 0.12, 0.15}                     (4)
    timeout  : {600, 900, 1200, 1500, 1800} ticks  = 60-180s (5)
    direction: {UP=0, DOWN=1, SKIP=2}                        (3)

State (91 dim): 14 × 3 primary softmaxes (42) + 49 handcrafted features.

Output: `models/iql_dataset_v1.npz` with
    states : (N, 91) float32
    actions: (N, n_actions) int8  — not one-hot; rather per-action index
    rewards: (N, n_actions) float32  — realised net PnL × kelly (0.25) per (sample, action)
    meta   : dict with action grid + splits

Cost: runs entirely on Contabo CPU via Rust, ~5-10 min wall.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402


CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
SOFTS_PATH = Path("/home/scalper/scalper-bot/models/primary_softs_v4.npz")
STACKER_META_PATH = Path("/home/scalper/scalper-bot/models/stacker_meta_v1.npz")
OUT = Path("/home/scalper/scalper-bot/models/iql_dataset_v2.npz")

# Action grid
TP_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]        # 7
SL_GRID = [0.08, 0.10, 0.12, 0.15]                          # 4
TIMEOUT_GRID = [600, 900, 1200, 1500, 1800]                  # 5  (60-180 s @ 100 ms)
DIRECTION_GRID = [0, 1, 2]                                   # 3 — UP, DOWN, SKIP

# Commission + fixed execution knobs (mirror grid_live defaults)
COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07
PARTIAL_ENABLED = True
TRAILING_ENABLED = True
FILL_LATENCY_MS = 150.0

# Post-hoc Kelly scaling applied to the raw realised PnL.
KELLY = 0.25

# Walk-forward split — must match grid_live for comparable eval.
TRAIN_FRAC = 0.75


def _load_cache():
    """Load the v3 cache sidecars needed by simulate_labels + state features."""
    cand = sorted(CACHE_DIR.glob("samples_v3_*_mid_paths.npy"))
    if not cand:
        raise FileNotFoundError(f"No samples_v3_* in {CACHE_DIR}")
    prefix = str(cand[-1])[: -len("_mid_paths.npy")]
    print(f"[iql] cache prefix: {prefix}")
    return {
        "prefix": prefix,
        "X_feat": np.load(f"{prefix}_X_feat.npy"),
        "y": np.load(f"{prefix}_y.npy"),
        "mid_paths": np.load(f"{prefix}_mid_paths.npy"),
        "entry_long": np.load(f"{prefix}_entry_long.npy"),
        "entry_short": np.load(f"{prefix}_entry_short.npy"),
    }


def _load_softs() -> tuple[np.ndarray, list[str]]:
    """Load primary_softs_v4.npz, stack into (N, n_archs, 3)."""
    d = np.load(SOFTS_PATH, allow_pickle=False)
    soft_keys = sorted(k for k in d.files if k.startswith("soft_"))
    arch_keys = [k[len("soft_"):] for k in soft_keys]
    softs = np.stack([d[k] for k in soft_keys], axis=1)   # (N, A, 3)
    print(f"[iql] softs: {softs.shape} archs={arch_keys}")
    return softs.astype(np.float32), arch_keys


def build_state(softs: np.ndarray, X_feat: np.ndarray,
                  stacker_soft: np.ndarray | None = None,
                  meta_prob: np.ndarray | None = None) -> np.ndarray:
    """state = [softs_flat (A*3), X_feat (F), stacker_soft (3), meta_prob (1),
    entropy+agreement (2)] — dims vary by whether stacker/meta are provided.

    The stacker + meta features are the single most informative signal in
    the state (meta AUC 0.883). Without them, IQL's Q-net has to recover
    from raw primary softmaxes — possible but leaves signal on the table.
    """
    N, A, _ = softs.shape
    flat = softs.reshape(N, A * 3)
    feat = X_feat.astype(np.float32)
    parts = [flat, feat]
    if stacker_soft is not None:
        parts.append(stacker_soft.astype(np.float32))
    if meta_prob is not None:
        parts.append(meta_prob.astype(np.float32).reshape(-1, 1))
    # Cross-arch agreement features — cheap, hand-computed.
    argmaxes = softs.argmax(axis=-1)     # (N, A)
    # Fraction of archs that agree with the plurality class per sample.
    vals, counts = np.zeros((N,), dtype=np.int64), np.zeros((N,), dtype=np.float32)
    for i in range(N):
        u, c = np.unique(argmaxes[i], return_counts=True)
        counts[i] = c.max() / A
        vals[i] = u[c.argmax()]
    # Entropy of mean softmax (confidence across archs).
    mean_soft = softs.mean(axis=1)        # (N, 3)
    ent = -(mean_soft * np.log(mean_soft.clip(1e-9))).sum(axis=-1)
    parts.append(counts.reshape(-1, 1))
    parts.append(ent.astype(np.float32).reshape(-1, 1))
    return np.concatenate(parts, axis=1)


def build_rewards(c: dict) -> tuple[np.ndarray, list]:
    """For each (TP, SL, timeout) in the grid, call Rust simulate_labels once.
    Returns rewards[N, n_bucket_combos, 2] where last axis = (pnl_long, pnl_short).
    Also returns the list of (tp, sl, timeout) tuples in iteration order.
    """
    N = c["y"].shape[0]
    combos = list(itertools.product(TP_GRID, SL_GRID, TIMEOUT_GRID))
    rewards = np.zeros((N, len(combos), 2), dtype=np.float32)
    t0 = time.monotonic()
    for i, (tp, sl, to) in enumerate(combos):
        tp_arr = np.full(N, tp, dtype=np.float64)
        sl_arr = np.full(N, sl, dtype=np.float64)
        to_arr = np.full(N, to, dtype=np.int64)
        out = rust_bridge.simulate_labels(
            c["entry_long"], c["entry_short"], c["mid_paths"],
            tp_arr, sl_arr, to_arr,
            commission_win_pct=COMMISSION_WIN,
            commission_loss_pct=COMMISSION_LOSS,
            partial_enabled=PARTIAL_ENABLED,
            trailing_enabled=TRAILING_ENABLED,
            fill_latency_ms=FILL_LATENCY_MS,
        )
        rewards[:, i, 0] = out["pnl_long"].astype(np.float32)
        rewards[:, i, 1] = out["pnl_short"].astype(np.float32)
        if (i + 1) % 20 == 0:
            dt = time.monotonic() - t0
            print(f"[iql] sim {i+1}/{len(combos)}  {dt:.1f}s elapsed")
    print(f"[iql] rust simulate_labels x{len(combos)}: {time.monotonic() - t0:.1f}s")
    return rewards, combos


def expand_to_full_action_grid(
    rewards_bucket: np.ndarray, combos: list
) -> tuple[np.ndarray, list]:
    """Expand (N, n_bucket_combos, 2) → (N, n_actions_total) where each action
    is (tp_idx, sl_idx, timeout_idx, direction). SKIP always rewards 0.

    Action ordering is stable and documented — index i corresponds to:
        direction = ACTIONS[i]["direction"]
        tp, sl, timeout = ACTIONS[i]["tp"], ACTIONS[i]["sl"], ACTIONS[i]["timeout"]
    """
    N = rewards_bucket.shape[0]
    actions = []
    reward_cols = []
    for (tp, sl, to), i_combo in zip(combos, range(len(combos))):
        for dir_idx in DIRECTION_GRID:
            if dir_idx == 2:
                # SKIP — reward=0 regardless of tp/sl/timeout. Emit only one
                # SKIP action per (tp,sl,to) so downstream code has a slot,
                # but its reward is flat 0.
                reward = np.zeros(N, dtype=np.float32)
            else:
                reward = rewards_bucket[:, i_combo, dir_idx] * np.float32(KELLY)
            actions.append({
                "tp": tp, "sl": sl, "timeout": to,
                "direction": dir_idx,   # 0=UP, 1=DOWN, 2=SKIP
            })
            reward_cols.append(reward)
    full = np.stack(reward_cols, axis=1)   # (N, n_actions_total)
    return full, actions


def main():
    c = _load_cache()
    softs, arch_keys = _load_softs()
    N = softs.shape[0]
    assert N == c["y"].shape[0], "softs/cache sample-count mismatch"

    # Stacker + meta if available (optional but strongly recommended — they
    # add a meta AUC 0.88 signal on top of raw primary softmaxes).
    stacker_soft = None
    meta_prob = None
    if STACKER_META_PATH.exists():
        sm = np.load(STACKER_META_PATH, allow_pickle=False)
        stacker_soft = sm["stacker_soft"]          # (N, 3)
        meta_prob = sm["meta_prob"]                 # (N,)
        print(f"[iql] stacker_soft {stacker_soft.shape}  meta_prob {meta_prob.shape}  "
              f"meta_prob>0 count={int((meta_prob > 0).sum())}")
    else:
        print(f"[iql] stacker/meta features not found at {STACKER_META_PATH} — "
              f"state will use raw softs + features only")

    state = build_state(softs, c["X_feat"], stacker_soft, meta_prob)
    print(f"[iql] state shape {state.shape}")

    rewards_bucket, combos = build_rewards(c)
    rewards_full, actions = expand_to_full_action_grid(rewards_bucket, combos)
    print(f"[iql] rewards_full shape {rewards_full.shape} "
          f"(n_actions = {len(actions)})")

    split = int(TRAIN_FRAC * N)
    n_train, n_eval = split, N - split

    # Sanity stats on the BEST-action reward per sample
    best_a_reward = rewards_full.max(axis=1)
    worst_a_reward = rewards_full.min(axis=1)
    print(f"[iql] reward stats — best-action mean={best_a_reward.mean():.4f} "
          f"median={np.median(best_a_reward):.4f} p90={np.percentile(best_a_reward, 90):.4f}")
    print(f"[iql] reward stats — worst-action mean={worst_a_reward.mean():.4f}")
    print(f"[iql] split: train {n_train:,}  eval {n_eval:,}")

    meta = {
        "arch_keys": arch_keys,
        "action_grid": {
            "tp": TP_GRID, "sl": SL_GRID, "timeout": TIMEOUT_GRID,
            "direction": DIRECTION_GRID, "kelly_fixed": KELLY,
        },
        "actions": actions,
        "combos_tp_sl_timeout": [list(c) for c in combos],
        "commission": {"win": COMMISSION_WIN, "loss": COMMISSION_LOSS},
        "partial_enabled": PARTIAL_ENABLED,
        "trailing_enabled": TRAILING_ENABLED,
        "fill_latency_ms": FILL_LATENCY_MS,
        "train_frac": TRAIN_FRAC,
        "n_samples": N,
        "n_train": n_train,
        "n_eval": n_eval,
        "state_dim": state.shape[1],
        "n_actions": len(actions),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT,
             state=state,
             rewards=rewards_full,
             y=c["y"],
             X_feat=c["X_feat"],
             meta=json.dumps(meta))
    print(f"[iql] wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
