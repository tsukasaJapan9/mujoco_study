"""PPO 用の Gymnasium 環境（ドメインランダム化つき）.

- 行動: アームの目標角速度（XL330 速度制御モードへの指令）を [-1, 1] に正規化したもの
- 観測: 実機でも得られるセンサー値だけから作る（遅延・量子化・ノイズを含む）
- 報酬: シミュレーションの真の状態から計算する（学習時だけ使える特権情報）
- エピソードごとに、サーボ特性・遅延・ガタ・摩擦を sweep.sample_true_params の範囲でランダムに変える
"""

import math

import gymnasium as gym
import numpy as np

from .params import Config
from .sim import FurutaSim, Obs, wrap
from .sweep import sample_true_params

W_MAX = 9.0  # [rad/s] 指令する角速度の上限
CONTROL_DT = 0.01  # 100 Hz
TIMESTEP = 0.001
PHI_LIMIT = 2 * math.pi  # 配線の都合でアームは ±1 回転まで


def recommended_config() -> Config:
    c = Config().with_(pend_length=0.06, arm_length=0.05, tip_mass=0.004, arm_preload=0.01, control_dt=CONTROL_DT)
    c.timestep = TIMESTEP
    return c


class Features:
    """センサー値 → 方策の入力。実機のマイコンでも同じ計算をする."""

    size = 10

    def __init__(self, dt=CONTROL_DT, vel_tau=0.01):
        self.dt = dt
        self.beta = math.exp(-dt / vel_tau)
        self.reset()

    def reset(self):
        self.prev = None
        self.th_dot = 0.0
        self.a1 = self.a2 = 0.0

    def push_action(self, a: float):
        self.a2, self.a1 = self.a1, a

    def __call__(self, o: Obs) -> np.ndarray:
        prev = self.prev or o
        raw = wrap(o.theta - prev.theta) / self.dt
        self.th_dot = self.beta * self.th_dot + (1 - self.beta) * raw
        self.prev = o
        return np.array(
            [
                math.sin(o.theta), math.cos(o.theta), self.th_dot / 20, o.phi / math.pi, o.phi_vel / 10,
                self.a1, self.a2,
                math.sin(prev.theta), math.cos(prev.theta), prev.phi / math.pi,
            ],
            dtype=np.float32,
        )


class FurutaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, nominal: Config | None = None, randomize=True, episode_s=5.0, p_hanging=0.7, param_sampler=None):
        self.nominal = nominal or recommended_config()
        self.randomize = randomize
        self.max_steps = round(episode_s / CONTROL_DT)
        self.p_hanging = p_hanging
        self.param_sampler = param_sampler or sample_true_params
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (Features.size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.feat = Features()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self.nominal
        if self.randomize:
            p = self.param_sampler(rng)
            p["arm_preload"] = self.nominal.servo.arm_preload * rng.uniform(0.5, 1.5)
            cfg = cfg.with_(**p)
        if rng.random() < self.p_hanging:
            th0 = math.pi + rng.uniform(-0.1, 0.1)
        else:
            th0 = rng.uniform(-math.pi, math.pi)
        self.sim = FurutaSim(cfg, seed=int(rng.integers(2**31)), theta0=th0)
        self.feat.reset()
        self.a_prev = 0.0
        self.t = 0
        return self.feat(self.sim.observe()), {}

    def step(self, action):
        a = float(np.clip(action[0], -1, 1))
        self.sim.advance(W_MAX * a)
        self.feat.push_action(a)
        obs = self.feat(self.sim.observe())
        self.t += 1

        d, s = self.sim.d, self.sim
        th = wrap(d.qpos[s.jp])
        th_dot = d.qvel[s.m.joint("pend").dofadr[0]]
        phi = d.qpos[s.jm]
        up = (1 + math.cos(th)) / 2
        near = abs(th) < 0.25
        reward = (
            up
            + 0.5 * near
            - 0.1 * (phi / math.pi) ** 2
            - 0.05 * (a - self.a_prev) ** 2
            - 0.01 * a**2
            - 0.002 * near * th_dot**2
        )
        self.a_prev = a
        terminated = abs(phi) > PHI_LIMIT
        if terminated:
            reward -= 50.0
        truncated = self.t >= self.max_steps
        return obs, float(reward), terminated, truncated, {}


class PolicyController:
    """学習済み方策を FurutaSim.run から使うためのラッパー（LQR と同じインターフェース）."""

    mode = 0

    def __init__(self, model):
        self.model = model
        self.feat = Features()

    def __call__(self, o: Obs) -> float:
        x = self.feat(o)
        a, _ = self.model.predict(x, deterministic=True)
        a = float(np.clip(a[0], -1, 1))
        self.feat.push_action(a)
        return W_MAX * a
