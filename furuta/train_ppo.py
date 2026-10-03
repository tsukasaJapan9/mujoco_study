"""PPO で回転倒立振子のスイングアップ＋倒立を学習する.

  uv run --extra rl python -m furuta.train_ppo --steps 10000000
  uv run --extra rl tensorboard --logdir runs/   # 学習曲線
"""

import argparse
import functools

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

from .rl_env import FurutaEnv
from .vec import BatchedSubprocVecEnv


def _build_env(rank, kw):
    env = FurutaEnv(**kw)
    env.reset(seed=rank)
    return env


def make_env(rank, **kw):
    return functools.partial(_build_env, rank, kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--resume", default=None, help="続きから学習する zip")
    a = ap.parse_args()
    torch.set_num_threads(1)

    venv = VecMonitor(BatchedSubprocVecEnv([make_env(i) for i in range(a.n_envs)], a.workers))
    eval_env = VecMonitor(BatchedSubprocVecEnv([make_env(1000 + i, p_hanging=1.0) for i in range(16)], a.workers))
    if a.resume:
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=256,
            batch_size=2048,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            # マイコンでも動く小さなネットワーク
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[128, 128]), log_std_init=-0.5),
            tensorboard_log=f"{a.out}/tb",
            device="cpu",
            verbose=0,
        )
    callbacks = [
        EvalCallback(eval_env, n_eval_episodes=16, eval_freq=250_000 // a.n_envs,
                     best_model_save_path=a.out, log_path=a.out, deterministic=True, verbose=1),
        CheckpointCallback(1_000_000 // a.n_envs, f"{a.out}/ckpt"),
    ]
    model.learn(a.steps, callback=callbacks, reset_num_timesteps=a.resume is None, progress_bar=False)
    model.save(f"{a.out}/final")


if __name__ == "__main__":
    main()
