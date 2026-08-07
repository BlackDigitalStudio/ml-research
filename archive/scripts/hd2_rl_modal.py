#!/usr/bin/env python3
"""HD2 RL side-experiment (frictionless, exploratory) — parallel-env PPO.

A PPO agent trades LTC on a 10s grid observing recent mid returns, position,
equity, time-in-trade and (ablation) the 3 Mamba-2 head logits. Reward =
position x next-step mid return. NO costs/slippage/latency/impact -> UPPER
BOUND ("is there ANY exploitable structure"), not realistic PnL. Ablation:
with-logits vs without -> held-out difference = what the Mamba signal adds over
bare price-reactive trading. The env lives in hd2_rl_env.py so SubprocVecEnv
workers import it cleanly. GPU is pointless here (tiny MLP policy); speed comes
from N parallel envs + CPU cores.

  modal run scripts/hd2_rl_modal.py            # both ablations, detached
"""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent
IMG = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.4.1", "gymnasium==0.29.1",
                 "stable-baselines3==2.3.2")
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
)
VOL = modal.Volume.from_name("hd2-cache")
MNT = "/cache"
N_ENVS = 8
app = modal.App("hd2-rl")


@app.function(image=IMG, cpu=8.0, timeout=5400, volumes={MNT: VOL})
def rl_run(use_logits: bool, timesteps: int = 800000, seed: int = 0):
    import sys, json, os
    sys.path.insert(0, "/root/scripts")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import hd2_rl_env as E
    VOL.reload()
    venv = SubprocVecEnv([lambda: E.TradeEnv("train", use_logits)
                          for _ in range(N_ENVS)], start_method="fork")
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                n_steps=512, batch_size=1024, gamma=0.999, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)
    venv.close()
    te = E.evaluate(model, use_logits, "test")
    tr = E.evaluate(model, use_logits, "train")
    out = {"use_logits": use_logits, "timesteps": timesteps, "train": tr, "heldout": te}
    os.makedirs(f"{MNT}/results/rl", exist_ok=True)
    with open(f"{MNT}/results/rl/rl_logits{use_logits}.json", "w") as f:
        json.dump(out, f, default=float)
    VOL.commit()
    print(f"RL logits={use_logits} HELD-OUT ret={te['total_ret_pct']:.3f}% "
          f"sharpe={te['sharpe']:.2f} trades={te['n_trades']} "
          f"frac_in={te['frac_in_market']:.2f} | train ret={tr['total_ret_pct']:.3f}%")
    return out


@app.local_entrypoint()
def main():
    handles = [(ul, rl_run.spawn(ul)) for ul in (True, False)]
    print(f"HD2 RL ablation SPAWNED ({N_ENVS} parallel envs, cpu=8; held-out=last 30% "
          f"OOS days, frictionless; detached -> /cache/results/rl/rl_logits{{True,False}}.json):")
    for ul, h in handles:
        print(f"  {h.object_id}  use_logits={ul}")
