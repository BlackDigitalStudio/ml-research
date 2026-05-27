"""In-container env for the HD2 RL side-experiment (imported ONLY on Modal).

Kept in its own module so SubprocVecEnv worker processes can import the env
cleanly (no closure/big-array pickling); each worker loads the 10s series from
the Volume itself. Frictionless trading on the 10-second LTC OOS grid.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

K = 12                 # obs window: last K 10s-returns (~2 min)
TRAIN_FRAC = 0.70      # first 70% of OOS days = RL-train; last 30% = held-out
SERIES = "/cache/results/hd2_pool/POOL_reg_d0.1_wd0.001_s0.rlseries.npz"


def _build(series_path=SERIES):
    d = np.load(series_path)
    ts, mid, logits, day = d["ts"], d["mid"].astype(np.float64), d["logits"], d["day"]
    ret = np.zeros(len(mid), np.float64)
    ret[:-1] = mid[1:] / mid[:-1] - 1.0
    tod = ((ts % (86400 * 10**9)) / (86400.0 * 10**9)).astype(np.float64)
    ranges = []
    i = 0; n = len(day)
    while i < n:
        j = i
        while j < n and day[j] == day[i]:
            j += 1
        if j - i > K + 2:
            ranges.append((i, j))
        i = j
    return dict(ret=ret, logits=logits.astype(np.float64), tod=tod, ranges=ranges)


class TradeEnv(gym.Env):
    """obs = [K scaled 10s-returns, position, episode-equity, time-in-trade,
    (+ 3 Mamba logits if use_logits)]; actions {0 short,1 flat,2 long};
    reward = position * next-step mid return (FRICTIONLESS); episode = one day."""

    def __init__(self, split="train", use_logits=True, series_path=SERIES):
        super().__init__()
        d = _build(series_path)
        r = d["ranges"]; cut = int(len(r) * TRAIN_FRAC)
        self.day_ranges = r[:cut] if split == "train" else r[cut:]
        self.ret = d["ret"]; self.logits = d["logits"]; self.tod = d["tod"]
        self.use_logits = use_logits
        obs_dim = K + 3 + (3 if use_logits else 0)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._ep = 0

    def _obs(self):
        i = self.i
        o = list(self.ret[i - K:i] * 1000.0) + [
            float(self.pos), self.equity * 100.0, min(self.tit / 180.0, 5.0)]
        if self.use_logits:
            o += list(self.logits[i] / 5.0)
        return np.asarray(o, np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        a, b = self.day_ranges[self._ep % len(self.day_ranges)]
        self._ep += 1
        self.a, self.b = a, b; self.i = a + K
        self.pos = 0; self.equity = 0.0; self.tit = 0
        return self._obs(), {}

    def step(self, action):
        new = int(action) - 1                      # -1 short, 0 flat, +1 long
        if new != self.pos:
            self.pos = new; self.tit = 0
        else:
            self.tit += 1
        r = self.pos * self.ret[self.i]            # frictionless PnL
        self.equity += r; self.i += 1
        done = self.i >= self.b - 1
        return self._obs(), float(r * 1000.0), done, False, {}


def evaluate(model, use_logits, split="test"):
    """Deterministic rollout over each day in `split` once -> PnL + trade stats."""
    env = TradeEnv(split=split, use_logits=use_logits)
    pnl = []; n_trades = 0; n_in = 0; prev = 0
    for _ in range(len(env.day_ranges)):
        obs, _ = env.reset(); done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rr, done, _, _ = env.step(int(act))
            pnl.append(rr / 1000.0)
            if env.pos != prev:
                n_trades += 1; prev = env.pos
            n_in += int(env.pos != 0)
    p = np.asarray(pnl)
    sharpe = float(p.mean() / (p.std() + 1e-12) * np.sqrt(8640 * 252))   # 10s grid -> ~yr
    return dict(total_ret_pct=float(p.sum() * 100), sharpe=sharpe,
                n_trades=int(n_trades), frac_in_market=float(n_in / max(1, len(p))),
                n_steps=int(len(p)))
