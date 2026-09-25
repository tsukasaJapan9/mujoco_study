# train/eval.py
import numpy as np
from gymnasium.wrappers import TimeLimit

from envs.pendulum_env import PendulumEnv


def evaluate(policy, n_ep=10, steps=500):
  """方策を採点する。すべての比較でこの関数だけを使うこと。"""
  totals, ups = [], []
  for ep in range(n_ep):
    env = TimeLimit(PendulumEnv(), max_episode_steps=steps)
    obs, info = env.reset(seed=ep)
    env.action_space.seed(ep)
    total, up = 0.0, 0
    for t in range(steps):
      obs, reward, terminated, truncated, info = env.step(policy(env, obs))
      total += float(reward)
      if -np.cos(env.unwrapped.data.qpos[0]) > 0.95:
        up += 1
      if terminated or truncated:
        break
    totals.append(total)
    ups.append(up)
  return np.mean(totals), np.mean(ups)


def breakdown(policy, n_ep=10, steps=500):
  """報酬を項ごとに分けて、1 エピソードあたりの平均を返す"""
  keys = ("r_angle", "r_omega", "r_torque")
  acc = {k: [] for k in keys}
  for ep in range(n_ep):
    env = TimeLimit(PendulumEnv(), max_episode_steps=steps)
    obs, info = env.reset(seed=ep)
    env.action_space.seed(ep)
    sums = {k: 0.0 for k in keys}
    for t in range(steps):
      obs, reward, terminated, truncated, info = env.step(policy(env, obs))
      for k in keys:
        sums[k] += info[k]
      if terminated or truncated:
        break
    for k in keys:
      acc[k].append(sums[k])
  return {k: float(np.mean(v)) for k, v in acc.items()}
