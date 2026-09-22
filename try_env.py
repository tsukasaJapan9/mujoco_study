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


def random_policy(env, obs):
  return env.action_space.sample()


# total, up = evaluate(random_policy)
# print(f"ランダム方策: 合計報酬 {total:.2f}   上端にいたステップ {up:.1f}/500")


# ============================================
# 古典制御
# ============================================
I, MGL, TAU = 0.056743, 1.623135, 0.5
E_STAR = 2 * MGL


def wrap(x):
  return (x + np.pi) % (2 * np.pi) - np.pi


def expert(env, obs):
  theta = env.unwrapped.data.qpos[0]
  omega = env.unwrapped.data.qvel[0]
  phi = wrap(theta - np.pi)
  if abs(phi) < 0.3:
    u = np.clip(-8.0 * phi - 1.0 * omega, -TAU, TAU)  # PD
  else:
    E = 0.5 * I * omega**2 + MGL * (1 - np.cos(theta))
    u = np.clip(1.0 * omega * (E_STAR - E), -TAU, TAU)  # エネルギー整形
  return np.array([u / TAU], dtype=np.float32)  # [-1,1] に直す


total, up = evaluate(expert)
print(f"古典制御    : 合計報酬 {total:.2f}   上端にいたステップ {up:.1f}/500")
