"""MuJoCo ビューアで動きを見る（ローカル PC 用）.

  uv run python -m furuta.view            # Linux/Windows
  uv run mjpython -m furuta.view          # macOS
"""

import time

import mujoco
import mujoco.viewer

from .control import SwingUpLQR
from .plots import demo_config
from .sim import FurutaSim


def main():
    nominal, truth = demo_config()
    sim = FurutaSim(truth, seed=0)
    ctrl = SwingUpLQR(nominal)
    with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
        while v.is_running():
            t0 = time.time()
            sim.run(ctrl, 0.02, log_every=10**9)  # 20 ms ずつ進めて描画
            v.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))


if __name__ == "__main__":
    main()
