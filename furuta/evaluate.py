"""1 エピソードの実行と評価指標."""

import numpy as np

from .control import SwingUpLQR
from .params import Config
from .sim import FurutaSim


def run_episode(cfg: Config, duration=10.0, seed=0, ctrl_kw=None, theta0=None, ctrl_cfg: Config | None = None):
    """cfg が「実機（真値）」、ctrl_cfg が制御器の設計に使う公称値（省略時は cfg と同じ）."""
    ctrl = SwingUpLQR(ctrl_cfg or cfg, **(ctrl_kw or {}))
    rng = np.random.default_rng(seed)
    th0 = theta0 if theta0 is not None else np.pi + rng.uniform(-0.05, 0.05)
    sim = FurutaSim(cfg, seed=seed, theta0=th0)
    log = sim.run(ctrl, duration)
    return log, metrics(log)


def metrics(log, hold=3.0, upright=0.25):
    """最後の hold 秒間ずっと |theta| < upright なら成功."""
    t, th, phi = log["t"], log["theta"], log["phi"]
    up = np.abs(th) < upright
    not_up = np.where(~up)[0]
    if not len(not_up):
        t_up = 0.0
    elif not_up[-1] == len(t) - 1:
        t_up = np.inf
    else:
        t_up = t[not_up[-1] + 1]
    success = bool(t[-1] - t_up >= hold)
    tail = t > t[-1] - hold
    return {
        "success": success,
        "t_up": float(t_up),
        "theta_rms_deg": float(np.degrees(np.sqrt(np.mean(th[tail] ** 2)))),
        "phi_std_hold_deg": float(np.degrees(np.std(phi[tail]))),
        "phi_maxabs_deg": float(np.degrees(np.max(np.abs(phi)))),
        "w_maxabs": float(np.max(np.abs(log["w"]))),
        "tau_maxabs": float(np.max(np.abs(log["tau"]))),
        "tau_rms_hold": float(np.sqrt(np.mean(log["tau"][tail] ** 2))),
    }
