"""学習済み PPO 方策の重みを numpy 形式（.npz）に書き出す.

推論は 行列積 2 回 + tanh だけなので、PyTorch なしで（将来はマイコン上でも）動かせる。

  uv run --extra rl python -m furuta.export_policy results/ppo_policy.zip results/ppo_policy.npz
"""

import sys

import numpy as np


def export(zip_path, npz_path):
    from stable_baselines3 import PPO

    p = PPO.load(zip_path, device="cpu").policy
    net = p.mlp_extractor.policy_net
    w = {
        "W1": net[0].weight, "b1": net[0].bias,
        "W2": net[2].weight, "b2": net[2].bias,
        "W3": p.action_net.weight, "b3": p.action_net.bias,
    }
    np.savez(npz_path, **{k: v.detach().numpy().astype(np.float32) for k, v in w.items()})

    # SB3 の決定的な出力と一致するか確認
    from .rl_env import NumpyPolicy

    x = np.random.default_rng(0).normal(size=(256, p.observation_space.shape[0])).astype(np.float32)
    ref = p.predict(x, deterministic=True)[0]
    ours = np.stack([NumpyPolicy(npz_path).predict(xi)[0] for xi in x])
    print("max |diff| =", float(np.max(np.abs(ref - ours))))


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
