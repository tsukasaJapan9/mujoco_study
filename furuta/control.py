"""エネルギー法スイングアップ + LQR 倒立制御.

XL330 は減速比が大きく外力でほぼ回らないため、速度制御モードの「目標角速度」を入力とし、
アーム角速度は一次遅れ（時定数 tau）で追従するとみなす。

振子の運動方程式（アームを加速度入力とみなした場合）:
  J_p * theta_dd = m*g*l*sin(theta) + m*l*r*phi_dd*cos(theta)

LQR はサーボの一次遅れと、センサー＋指令の遅延（1 制御周期未満）を設計モデルに含める。
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from .params import Config
from .sim import Obs, wrap

G = 9.81


def plant_matrices(a, b, tau):
    """状態 x=[theta, theta_d, phi, phi_d]、入力 u=目標角速度 の連続時間モデル."""
    A = np.array(
        [[0, 1, 0, 0], [a, 0, 0, -b / tau], [0, 0, 0, 1], [0, 0, 0, -1 / tau]],
        float,
    )
    B = np.array([0, b / tau, 0, 1 / tau], float)
    return A, B


def discretize_with_delay(A, B, dt, delay):
    """入力に delay (< dt) の遅れがある系の離散化。状態 z=[x, u_{k-1}]."""
    delay = min(delay, dt * 0.999)
    n = A.shape[0]

    def integ(T):  # ∫_0^T e^{As} B ds（拡大行列の指数関数で計算）
        M = np.zeros((n + 1, n + 1))
        M[:n, :n], M[:n, n] = A, B
        return expm(M * T)[:n, n]

    Ad = expm(A * dt)
    B0 = integ(dt - delay)
    B1 = expm(A * (dt - delay)) @ integ(delay)
    Az = np.zeros((n + 1, n + 1))
    Az[:n, :n] = Ad
    Az[:n, n] = B1
    Bz = np.zeros((n + 1, 1))
    Bz[:n, 0] = B0
    Bz[n, 0] = 1
    return Az, Bz


class SwingUpLQR:
    def __init__(
        self,
        cfg: Config,
        q=(30.0, 0.5, 2.0, 0.05, 0.0),
        r=0.3,
        tau_design=None,
        delay_design=None,
        alpha_max=120.0,
        w_max=9.0,
        k_center=(3.0, 1.5),
        energy_margin=0.05,
        catch_angle=0.5,
        lose_angle=0.8,
        vel_tau=0.003,
        use_kalman=True,
        kf_accel_std=(30.0, 30.0),
    ):
        g, s = cfg.geom, cfg.sense
        self.dt = s.control_dt
        self.jp, self.ml, self.r = g.j_pivot, g.m_l, g.arm_length
        self.a = self.ml * G / self.jp
        self.b = self.ml * self.r / self.jp
        tau = tau_design if tau_design is not None else cfg.servo.tau_servo
        delay = delay_design if delay_design is not None else s.obs_delay + s.cmd_delay
        A, B = plant_matrices(self.a, self.b, tau)
        Az, Bz = discretize_with_delay(A, B, self.dt, delay)
        P = solve_discrete_are(Az, Bz, np.diag(q), np.array([[r]]))
        self.K = np.linalg.solve(r + Bz.T @ P @ Bz, Bz.T @ P @ Az)[0]
        # 倒立中の状態推定: 定常カルマンフィルタ（観測 = 振子角, アーム角, アーム角速度）
        self.use_kalman = use_kalman
        self.Az, self.Bz = Az, Bz[:, 0]
        res = 2 * np.pi / s.encoder_counts
        vel_unit = 0.229 * 2 * np.pi / 60
        C = np.zeros((3, 5))
        C[0, 0] = C[1, 2] = C[2, 3] = 1
        Rn = np.diag([(res * max(s.pend_noise_counts, 0.3)) ** 2, res**2 / 12, vel_unit**2 / 12 + 0.05**2])
        sa, sp = kf_accel_std  # 未モデル化の加速度 [rad/s^2]
        Gw = np.zeros((5, 2))
        Gw[0, 0], Gw[1, 0] = self.dt**2 / 2, self.dt
        Gw[2, 1], Gw[3, 1] = self.dt**2 / 2, self.dt
        Qn = Gw @ np.diag([sa**2, sp**2]) @ Gw.T + 1e-12 * np.eye(5)
        Pk = solve_discrete_are(Az.T, C.T, Qn, Rn)
        self.C = C
        self.L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rn)
        self.xhat = None
        self.e_down = 2 * self.ml * G  # 真下から真上までのエネルギー差
        self.e_target = energy_margin * self.e_down  # 摩擦損失を見込んで少し多めに注入
        self.k_energy = 3.0 * alpha_max / self.e_down
        self.alpha_max, self.w_max = alpha_max, w_max
        self.k_center = k_center
        self.catch_angle, self.lose_angle = catch_angle, lose_angle
        self.beta = np.exp(-self.dt / vel_tau)
        self.reset()

    def reset(self):
        self.mode = 0  # 0: スイングアップ, 1: 倒立
        self.w_cmd = 0.0
        self.prev = None
        self.th_dot = 0.0
        self.xhat = None

    def __call__(self, o: Obs) -> float:
        if self.prev is not None:
            raw = wrap(o.theta - self.prev.theta) / self.dt
            self.th_dot = self.beta * self.th_dot + (1 - self.beta) * raw
        self.prev = o
        th = o.theta

        if self.mode == 0 and abs(th) < self.catch_angle:
            self.mode = 1
        elif self.mode == 1 and abs(th) > self.lose_angle:
            self.mode = 0
            self.xhat = None

        if self.mode == 1:
            z = np.array([th, self.th_dot, o.phi, o.phi_vel, self.w_cmd])
            if self.use_kalman:
                if self.xhat is None:
                    self.xhat = z
                else:
                    pred = self.Az @ self.xhat + self.Bz * self.w_cmd
                    y = np.array([th, o.phi, o.phi_vel])
                    self.xhat = pred + self.L @ (y - self.C @ pred)
                    self.xhat[4] = self.w_cmd
                z = self.xhat
            self.w_cmd = float(np.clip(-self.K @ z, -self.w_max, self.w_max))
            return self.w_cmd

        E = 0.5 * self.jp * self.th_dot**2 + self.ml * G * (np.cos(th) - 1)
        s = self.th_dot * np.cos(th)
        alpha = self.k_energy * (self.e_target - E) * np.sign(s) * min(1.0, abs(s) / 0.5)
        if abs(self.th_dot) < 0.05 and abs(wrap(th - np.pi)) < 0.05:
            alpha = self.alpha_max  # 真下で静止していたら一押しする
        kp, kd = self.k_center
        alpha -= kp * o.phi + kd * o.phi_vel
        alpha = float(np.clip(alpha, -self.alpha_max, self.alpha_max))
        self.w_cmd = float(np.clip(self.w_cmd + alpha * self.dt, -self.w_max, self.w_max))
        return self.w_cmd
