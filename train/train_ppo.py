import argparse

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.pendulum_env import PendulumEnv

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300_000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tag", default="ppo")
args = parser.parse_args()


def make_env():
  return TimeLimit(PendulumEnv(), max_episode_steps=500)


venv = make_vec_env(make_env, n_envs=4, seed=args.seed)


model = PPO(
  "MlpPolicy",
  venv,
  verbose=1,
  seed=args.seed,
  device="cuda",
  learning_rate=args.lr,
  tensorboard_log="tb_logs",
)
model.learn(total_timesteps=args.steps, tb_log_name=args.tag)
model.save(f"train/ppo_{args.tag}_s{args.seed}")
