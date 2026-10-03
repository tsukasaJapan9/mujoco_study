"""XL330 速度制御モード・センサー・遅延を含むシミュレーション."""

import math
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from .model import build_model
from .params import Config


class XL330VelocityMode:
    """サーボ内部の速度 PI ループ + 電源電圧によるトルク-速度制限."""

    def __init__(self, cfg: Config, j_load: float):
        s = cfg.servo
        self.s = s
        self.kp = (s.gear_armature + j_load) / s.tau_servo
        self.ti = 4 * s.tau_servo
        self.integ = 0.0

    def torque(self, w_ref: float, w: float, dt: float) -> float:
        s = self.s
        e = w_ref - w
        tau = self.kp * (e + self.integ / self.ti)
        hi = min(s.stall_torque, s.stall_torque * (1 - w / s.no_load_speed))
        lo = max(-s.stall_torque, -s.stall_torque * (1 + w / s.no_load_speed))
        if lo < tau < hi:
            self.integ += e * dt  # 飽和中は積分しない（anti-windup）
        return min(max(tau, lo), hi)


@dataclass
class Obs:
    t: float
    phi: float  # アーム角（XL330 の Present Position、モーター側）
    theta: float  # 振子角（AS5600、0=倒立、(-pi, pi]）
    phi_vel: float  # アーム角速度（XL330 の Present Velocity、0.229 rpm 単位）


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class FurutaSim:
    def __init__(self, cfg: Config, seed: int = 0, theta0: float = np.pi - 0.03):
        self.cfg = cfg
        self.m = build_model(cfg)
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.jm = self.m.joint("motor").qposadr[0]
        self.jp = self.m.joint("pend").qposadr[0]
        self.vm = self.m.joint("motor").dofadr[0]
        self.d.qpos[self.jp] = theta0
        # ガタ片寄せ用の一定トルク（鉛直軸まわり、アームに作用）
        self.d.xfrc_applied[self.m.body("arm").id, 5] = cfg.servo.arm_preload
        mujoco.mj_forward(self.m, self.d)
        g = cfg.geom
        j_load = g.arm_mass * g.arm_length**2 / 3 + (g.hub_mass + g.pend_mass) * g.arm_length**2
        self.servo = XL330VelocityMode(cfg, j_load)
        dt = cfg.timestep
        n = cfg.sense
        self.steps_per_ctrl = round(n.control_dt / dt)
        self.obs_buf = deque(maxlen=max(1, round(n.obs_delay / dt)) + 1)
        self.cmd_buf = deque([0.0] * (round(n.cmd_delay / dt) + 1), maxlen=round(n.cmd_delay / dt) + 1)
        self.res = 2 * np.pi / n.encoder_counts
        self._k = 0
        self.obs_buf.append(self._raw())

    def _raw(self):
        """物理ステップごとに保存する真値（量子化・ノイズは読み出し時にかける）."""
        d = self.d
        return (d.time, d.qpos[self.jm], d.qpos[self.jp], d.qvel[self.vm])

    def _measure(self, raw) -> Obs:
        t, phi, th, w = raw
        res = self.res
        th += self.rng.normal(0, self.cfg.sense.pend_noise_counts * res)
        vel_unit = 0.229 * 2 * math.pi / 60
        return Obs(t, round(phi / res) * res, wrap(round(th / res) * res), round(w / vel_unit) * vel_unit)

    def observe(self) -> Obs:
        """制御器が受け取るセンサー値（遅延あり）."""
        return self._measure(self.obs_buf[0])

    def advance(self, w_ref: float, log=None, log_every: int = 10, mode: float = 0.0) -> None:
        """目標角速度 w_ref を送って 1 制御周期ぶん進める."""
        for _ in range(self.steps_per_ctrl):
            self.cmd_buf.append(w_ref)
            w_cmd = self.cmd_buf[0]  # 遅延した指令
            w = self.d.qvel[self.vm]
            tau = self.servo.torque(w_cmd, w, self.cfg.timestep)
            self.d.ctrl[0] = tau
            mujoco.mj_step(self.m, self.d)
            self.obs_buf.append(self._raw())
            self._k += 1
            if log is not None and self._k % log_every == 0:
                log.append((self.d.time, wrap(self.d.qpos[self.jp]), self.d.qpos[self.jm], w, w_cmd, tau, mode))

    def run(self, controller, duration: float, log_every: int = 10):
        log = []
        for _ in range(round(duration / self.cfg.sense.control_dt)):
            w_ref = controller(self.observe())
            self.advance(w_ref, log, log_every, getattr(controller, "mode", 0))
        cols = ["t", "theta", "phi", "w", "w_cmd", "tau", "mode"]
        arr = np.array(log, dtype=float).reshape(-1, len(cols))
        return {c: arr[:, i] for i, c in enumerate(cols)}
