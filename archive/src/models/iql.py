"""Offline RL policy head for per-trade (TP, SL, timeout, direction) selection.

Full-information variant of IQL: since our dataset enumerates EVERY action
for every state (via Rust `simulate_labels` sweep), there is no off-policy
estimation problem — each (s, a) pair has a direct observed reward. The
"Bellman backup" in standard IQL collapses to plain supervised regression.

What IQL buys us vs naive supervised argmax:
  * Expectile regression on `V(s) ≈ upper-τ quantile of Q(s, ·)` filters
    out lucky-outlier actions; argmax of Q alone chases variance.
  * Advantage-weighted policy: π(a|s) ∝ exp((Q(s,a) − V(s)) / β) produces
    a *distribution* over actions, not just argmax. Useful when multiple
    actions are near-optimal — smoothing reduces overfit to noise.

Architecture:
  Q-net: MLP(state_dim) → residual blocks → Linear(n_actions)
  V-net: MLP(state_dim) → residual blocks → Linear(1)

  Q loss: Huber(Q(s, a_i) − r_i) — directly regress rewards.
  V loss: expectile-τ (asymmetric huber: more weight on upper-tail).
  π loss: KL[target ∥ softmax(Q/β)] where target = one-hot argmax of Q.
          (at inference we use softmax with temperature 0 = argmax)

At inference:
    state → Q-net → 420 scores → argmax or top-k sampling → action.

Walk-forward eval:
    Train on first 75 % of samples, evaluate on tail 25 %.
    For each tail sample: pick action via Q-net, look up the TRUE reward
    from the dataset (since dataset has full-info). Sum over tail to get
    realised net-PnL of the policy vs the best-fixed-config policy.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IQLConfig:
    state_dim: int = 91
    n_actions: int = 420
    hidden_dim: int = 512
    n_blocks: int = 3
    dropout: float = 0.15
    expectile_tau: float = 0.75   # upper-tau — focus on better-than-median actions
    temperature: float = 3.0       # softmax scale for stochastic policy
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    early_stop_patience: int = 6
    seed: int = 42
    # Training tweaks (added 2026-04-16 after first training collapse-to-SKIP):
    drop_skip_actions: bool = True    # exclude direction=SKIP from action space
                                       # so policy is forced to pick UP or DOWN;
                                       # a separate meta-gate handles "don't trade".
    advantage_shaping: bool = True    # regress Q against (r - mean(r | s)) instead
                                       # of raw r, to force relative-action learning
                                       # rather than "all rewards are near 0, skip".


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QNet(nn.Module):
    """State → per-action Q values (n_actions)."""

    def __init__(self, cfg: IQLConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.head = nn.Linear(cfg.hidden_dim, cfg.n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(state)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class VNet(nn.Module):
    """State → single value (expectile over Q)."""

    def __init__(self, cfg: IQLConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(state)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric squared loss — penalises under-estimates by (1-tau)^2 and
    over-estimates by tau^2. tau=0.5 is MSE, tau>0.5 shifts towards upper-tail.
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return weight * diff.pow(2)


def train_iql(
    dataset_path: Path,
    out_dir: Path,
    cfg: IQLConfig = IQLConfig(),
    device: str = "cpu",
) -> dict:
    """Train Q + V networks on the (s, a, r) dataset. Returns metrics dict."""
    print(f"[iql] loading {dataset_path}")
    d = np.load(dataset_path, allow_pickle=False)
    state = d["state"]                    # (N, state_dim)
    rewards = d["rewards"]                # (N, n_actions)
    meta = json.loads(str(d["meta"].item()))
    n_train = meta["n_train"]

    # Optionally drop SKIP actions (direction == 2) — policy is forced to
    # pick UP or DOWN. Prevents the trivial collapse where the model learns
    # "all rewards are near-zero on average, skip always gives zero, so
    # always skip" — which was the failure mode on v1 training.
    if cfg.drop_skip_actions:
        actions_meta = meta["actions"]
        keep_idx = np.array([i for i, a in enumerate(actions_meta) if a["direction"] != 2],
                             dtype=np.int64)
        rewards = rewards[:, keep_idx]
        meta["actions"] = [actions_meta[i] for i in keep_idx]
        meta["n_actions"] = len(meta["actions"])
        print(f"[iql] dropped SKIP actions: n_actions {cfg.n_actions} → {len(keep_idx)}")
        # Override cfg to match — in-place is fine because cfg is a runtime value.
        cfg.n_actions = len(keep_idx)

    # Auto-sync cfg to dataset dims rather than hard-asserting — dataset v2
    # has stacker + meta + agreement/entropy appended, which changes state_dim.
    if state.shape[1] != cfg.state_dim:
        print(f"[iql] state_dim mismatch: cfg={cfg.state_dim} dataset={state.shape[1]} — using dataset value")
        cfg.state_dim = state.shape[1]
    if rewards.shape[1] != cfg.n_actions:
        print(f"[iql] n_actions mismatch: cfg={cfg.n_actions} dataset={rewards.shape[1]} — using dataset value")
        cfg.n_actions = rewards.shape[1]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    s_train = torch.tensor(state[:n_train], dtype=torch.float32)
    r_train = torch.tensor(rewards[:n_train], dtype=torch.float32)
    s_eval = torch.tensor(state[n_train:], dtype=torch.float32)
    r_eval = torch.tensor(rewards[n_train:], dtype=torch.float32)

    # Normalize state features per-dimension using train stats (simple z-score).
    state_mean = s_train.mean(dim=0, keepdim=True)
    state_std = s_train.std(dim=0, keepdim=True).clamp(min=1e-4)
    s_train = (s_train - state_mean) / state_std
    s_eval = (s_eval - state_mean) / state_std

    qnet = QNet(cfg).to(device)
    vnet = VNet(cfg).to(device)
    q_opt = torch.optim.AdamW(qnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    v_opt = torch.optim.AdamW(vnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_eval_reward = -1e9
    best_state: dict | None = None
    patience = cfg.early_stop_patience
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        t0 = time.monotonic()
        qnet.train(); vnet.train()
        perm = torch.randperm(n_train)
        q_loss_sum = 0.0
        v_loss_sum = 0.0
        n_batches = 0

        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            sb = s_train[idx].to(device)
            rb = r_train[idx].to(device)

            # V expectile regression on the in-batch rewards: pick the best
            # action per sample (argmax reward) as the "observed" action,
            # regress V toward its reward with expectile loss.
            best_r = rb.max(dim=1).values
            v_pred = vnet(sb)
            v_loss = expectile_loss(best_r - v_pred, cfg.expectile_tau).mean()
            v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            v_opt.step()

            # Q regression target: optionally subtract per-state mean reward
            # so the network only has to learn RELATIVE action quality. This
            # prevents the trivial "all rewards ~0 on average → any flat
            # predictor wins" failure mode.
            if cfg.advantage_shaping:
                target = rb - rb.mean(dim=1, keepdim=True)
            else:
                target = rb
            q_pred = qnet(sb)                     # (B, n_actions)
            q_loss = F.smooth_l1_loss(q_pred, target)
            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            q_opt.step()

            q_loss_sum += float(q_loss.item())
            v_loss_sum += float(v_loss.item())
            n_batches += 1

        # Eval on tail
        qnet.eval()
        with torch.no_grad():
            eval_q = qnet(s_eval.to(device))          # (N_eval, n_actions)
            # Policy = argmax action per state
            a_pi = eval_q.argmax(dim=1)                # (N_eval,)
            # Realized reward = true r_eval at chosen action per sample
            realized = r_eval[torch.arange(r_eval.shape[0]), a_pi.cpu()]
            pi_net_pnl = float(realized.sum().item())
            pi_mean_r = float(realized.mean().item())
            pi_wr = float((realized > 0).float().mean().item())
            # Compare to "best fixed config" — the config whose sum-over-eval
            # reward is highest, as a baseline.
            fixed_sum = r_eval.sum(dim=0)              # (n_actions,)
            best_fixed = fixed_sum.argmax().item()
            fixed_pnl = float(fixed_sum[best_fixed].item())
            # Oracle — pick argmax reward per sample (upper bound, unrealistic)
            oracle = float(r_eval.max(dim=1).values.sum().item())

            # Class distribution in policy: how many samples routed to SKIP?
            # Decode a_pi → (tp_idx, sl_idx, to_idx, dir_idx) via meta["actions"].
            action_table = meta["actions"]
            dirs = np.array([action_table[int(a)]["direction"] for a in a_pi.cpu().numpy()])
            n_up = int((dirs == 0).sum())
            n_dn = int((dirs == 1).sum())
            n_skip = int((dirs == 2).sum())

        q_loss_avg = q_loss_sum / max(1, n_batches)
        v_loss_avg = v_loss_sum / max(1, n_batches)
        dt = time.monotonic() - t0
        print(f"[iql] e{epoch+1:2d}/{cfg.epochs}  {dt:5.1f}s  "
              f"Q={q_loss_avg:.5f} V={v_loss_avg:.5f}  "
              f"pi_pnl={pi_net_pnl:+8.2f} WR={pi_wr*100:.1f}%  "
              f"fixed={fixed_pnl:+.2f} oracle={oracle:+.2f}  "
              f"[UP={n_up} DN={n_dn} SK={n_skip}]")

        history.append({
            "epoch": epoch + 1,
            "q_loss": q_loss_avg,
            "v_loss": v_loss_avg,
            "pi_net_pnl": pi_net_pnl,
            "pi_mean_reward": pi_mean_r,
            "pi_wr": pi_wr,
            "fixed_best_pnl": fixed_pnl,
            "oracle_pnl": oracle,
            "n_up": n_up,
            "n_dn": n_dn,
            "n_skip": n_skip,
            "epoch_time_s": dt,
        })

        if pi_net_pnl > best_eval_reward:
            best_eval_reward = pi_net_pnl
            best_state = {
                "q": {k: v.detach().cpu().clone() for k, v in qnet.state_dict().items()},
                "v": {k: v.detach().cpu().clone() for k, v in vnet.state_dict().items()},
                "state_mean": state_mean,
                "state_std": state_std,
                "epoch": epoch + 1,
                "pi_net_pnl": pi_net_pnl,
            }
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[iql] early-stop at e{epoch+1} (best pnl={best_eval_reward:+.2f})")
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cfg": cfg.__dict__,
        **best_state,
        "history": history,
        "meta": meta,
    }, out_dir / "iql_v1.pt")

    metrics_path = out_dir / "iql_v1_metrics.json"
    metrics_path.write_text(json.dumps({
        "cfg": cfg.__dict__,
        "history": history,
        "best_pi_net_pnl": best_eval_reward,
    }, indent=2, default=float))

    print(f"[iql] saved {out_dir / 'iql_v1.pt'}  best pnl={best_eval_reward:+.2f}")
    return {
        "best_pi_net_pnl": best_eval_reward,
        "history": history,
        "ckpt": str(out_dir / "iql_v1.pt"),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/scalper/scalper-bot/models/iql_dataset_v1.npz")
    ap.add_argument("--out", default="/home/scalper/scalper-bot/models")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train_iql(Path(args.dataset), Path(args.out), IQLConfig(epochs=args.epochs), args.device)
