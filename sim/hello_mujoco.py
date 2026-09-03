import argparse
import contextlib
import time

import mujoco
import mujoco.viewer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--view", action="store_true", help="ビューアで実時間再生する")
args = parser.parse_args()

XML = """
<mujoco model="simple_pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
      <geom name="pole_geom" type="capsule" fromto="0 0 0  0 0 -0.5"
            size="0.02" rgba="0.2 0.4 0.9 1"/>
    </body>
    <site name="axis_x" type="cylinder" fromto="0 0 0  0.3 0 0" size="0.006" rgba="1 0 0 1"/>
    <site name="axis_y" type="cylinder" fromto="0 0 0  0 0.3 0" size="0.006" rgba="0 1 0 1"/>
    <site name="axis_z" type="cylinder" fromto="0 0 0  0 0 0.3" size="0.006" rgba="0 0 1 1"/>

  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
print(f"timestep={model.opt.timestep}")

data.qpos[0] = 1.0

n_steps = 5000  # 0.002 s x 5000 = 10 秒ぶん
log = np.zeros((n_steps, 3))  # 各行 [time, qpos, qvel]

# for i in range(1000):
#     mujoco.mj_step(model, data)
#     if i % 100 == 0:
#         print(f"t={data.time:6.3f}  qpos={data.qpos}  qvel={data.qvel}")

print(f"init qpos: {data.qpos[0]}")

bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pole")
m = model.body_mass[bid]  # 質量 [kg]（geom の形状と密度から自動計算された値）
d = abs(model.body_ipos[bid][2])  # body 原点から重心までの距離 [m]
I_cm = model.body_inertia[bid][1]  # 重心まわりの慣性モーメント（y 軸成分）

I = I_cm + m * d**2  # 平行軸の定理で回転軸まわりに移す
T_theory = 2 * np.pi * np.sqrt(I / (m * 9.81 * d))
print(f"{m=}, {d=}, {I_cm=}")
print(f"theory (small angle) = {T_theory:.4f} s")


i = 0
ctx = (
    mujoco.viewer.launch_passive(model, data) if args.view else contextlib.nullcontext()
)
with ctx as viewer:
    for i in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        log[i] = [data.time, data.qpos[0], data.qvel[0]]

        if viewer is not None:
            viewer.sync()
            wait = model.opt.timestep - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)

        if i >= n_steps:
            break

t, q, v = log[:, 0], log[:, 1], log[:, 2]
idx = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]
period = np.diff(t[idx]).mean()
print(f"measured period = {period:.4f} s")

# idx = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]


# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# # 左: 時系列
# axes[0].plot(t, q, label="angle [rad]")
# axes[0].plot(t, v, label="angular velocity [rad/s]", alpha=0.7)
# axes[0].set_xlabel("time [s]")
# axes[0].set_title("time series")
# axes[0].legend()
# axes[0].grid(alpha=0.3)

# # 右: 位相平面
# axes[1].plot(q, v, lw=0.8)
# axes[1].set_xlabel("angle [rad]")
# axes[1].set_ylabel("angular velocity [rad/s]")
# axes[1].set_title("phase plane")
# axes[1].grid(alpha=0.3)

# fig.tight_layout()
# plt.show()
