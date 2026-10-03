"""振子のベアリング（摩擦・アーム先端の重さ）を変えたときの成否を確かめる.

608ZZ（内径 8, 外径 22, 幅 7 mm, 約 12 g）を 2 個使う場合を想定し、振子軸の摩擦トルクを振る。
  uv run python -m furuta.bearing_check
"""

import math
from multiprocessing import Pool

import numpy as np

from .control import SwingUpLQR
from .evaluate import metrics
from .rl_env import CONTROL_DT, NumpyPolicy, PolicyController, recommended_config
from .sim import FurutaSim
from .sweep import sample_true_params

FRICTIONS = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4]  # [N m] ベアリング 2 個分の合計
TIPS = [0.004, 0.008, 0.012]
HUB_MASS = 0.035  # 608ZZ x2 (約 24 g) + ハウジング・AS5600
N = 16


def job(a):
    fric, tip, ctrl_name, i = a
    rng = np.random.default_rng(7000 + i)
    nominal = recommended_config().with_(tip_mass=tip, hub_mass=HUB_MASS)
    p = sample_true_params(rng)
    p["pend_friction"] = fric * rng.uniform(0.7, 1.3)
    p["arm_preload"] = nominal.servo.arm_preload * rng.uniform(0.5, 1.5)
    truth = nominal.with_(**p)
    ctrl = SwingUpLQR(nominal) if ctrl_name == "lqr" else PolicyController(NumpyPolicy("results/ppo_policy.npz"))
    sim = FurutaSim(truth, seed=i, theta0=math.pi + rng.uniform(-0.05, 0.05))
    m = metrics(sim.run(ctrl, 10.0))
    return a[:3], m["success"] and m["phi_maxabs_deg"] <= 360, m["theta_rms_deg"]


if __name__ == "__main__":
    grid = [(f, t, c, i) for f in FRICTIONS for t in TIPS for c in ("lqr", "ppo") for i in range(N)]
    with Pool() as pool:
        res = pool.map(job, grid)
    print(f"制御周期 {1 / CONTROL_DT:.0f} Hz、アーム先端 {HUB_MASS * 1e3:.0f} g、各 {N} 台")
    print("摩擦[N·m]  おもり[g]   LQR成功率(θRMS)    PPO成功率(θRMS)")
    for f in FRICTIONS:
        for t in TIPS:
            row = []
            for c in ("lqr", "ppo"):
                rs = [r for r in res if r[0] == (f, t, c)]
                ok = [r for r in rs if r[1]]
                th = np.median([r[2] for r in ok]) if ok else float("nan")
                row.append(f"{len(ok) / len(rs):5.0%} ({th:4.1f}°)")
            print(f"{f:9.0e}  {t * 1e3:7.0f}   {row[0]:>16s}   {row[1]:>16s}")
