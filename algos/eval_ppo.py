import numpy as np
import torch

from algos.ppo_main import ActorCritic
from train.eval import evaluate

net = ActorCritic(3, 1)
net.load_state_dict(torch.load("algos/ppo_min_pendulum.pt"))
net.eval()


def my_policy(env, obs):
  with torch.no_grad():
    mu, _, _ = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
  return np.clip(mu.numpy()[0], -1, 1)  # 評価では平均だけ使う


total, up = evaluate(my_policy)
print(f"自作 PPO    : 合計報酬 {total:.2f}   上端 {up:.1f}/500")
