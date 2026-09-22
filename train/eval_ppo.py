from stable_baselines3 import PPO

model = PPO.load("train/ppo_pendulum")


import numpy as np
from gymnasium.wrappers import TimeLimit

from envs.pendulum_env import PendulumEnv


def evaluate(policy, n_ep=10, steps=500):
  totals, ups = [], []
  for ep in range(n_ep):
    env = TimeLimit(PendulumEnv(), max_episode_steps=steps)
    obs, info = env.reset(seed=ep)
    env.action_space.seed(ep)  # これが無いと毎回結果が変わる
    total, up = 0.0, 0
    for t in range(steps):
      obs, reward, terminated, truncated, info = env.step(policy(env, obs))
      total += reward
      if -np.cos(env.unwrapped.data.qpos[0]) > 0.95:
        up += 1
      if terminated or truncated:
        break
    totals.append(total)
    ups.append(up)
  return np.mean(totals), np.mean(ups)


def trained_policy(env, obs):
  action, _ = model.predict(obs, deterministic=True)
  return action


total, up = evaluate(trained_policy)  # try_env.py の関数をそのまま使う
print(f"学習した方策: 合計報酬 {total:.2f}   上端にいたステップ {up:.1f}/500")
print("  ランダム方策 : -499.50   (0.0/500)")
print("  古典制御     :   28.31   (216.2/500)")
