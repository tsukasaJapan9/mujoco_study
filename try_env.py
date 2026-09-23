import numpy as np
from stable_baselines3 import PPO

from train.eval import evaluate

model = PPO.load("train/ppo_semi_long_s0")


def trained_policy(env, obs):
  action, _ = model.predict(obs, deterministic=True)
  return action


def random_policy(env, obs):
  return env.action_space.sample()


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


# total, up = evaluate(expert)
# print(f"古典制御    : 合計報酬 {total:.2f}   上端にいたステップ {up:.1f}/500")

total, up = evaluate(trained_policy)
print(f"PPO制御    : 合計報酬 {total:.2f}   上端にいたステップ {up:.1f}/500")
