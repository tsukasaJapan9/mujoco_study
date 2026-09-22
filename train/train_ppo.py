from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.pendulum_env import PendulumEnv


def make_env():
  return TimeLimit(PendulumEnv(), max_episode_steps=500)


venv = make_vec_env(make_env, n_envs=4, seed=0)

model = PPO("MlpPolicy", venv, verbose=1, seed=0, device="cpu")
model.learn(total_timesteps=300_000)  # ← ここだけ考える
model.save("train/ppo_pendulum")
