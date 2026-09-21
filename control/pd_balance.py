import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("models/pendulum_torque.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M)  # 3.12 では第2引数が MjData

m = model.body_mass[1]
l = abs(model.body_ipos[1][2])
print("I    =", M[0, 0])
print("mgl  =", m * 9.81 * l)


def wrap(x):
  return (x + np.pi) % (2 * np.pi) - np.pi


TAU_MAX = 0.5
kp, kd = 8.0, 1.0
ctrl_hz = 100
decimation = round((1 / model.opt.timestep) / ctrl_hz)

data.qpos[0] = np.pi + 0.1  # 上端から 0.1 rad 傾けて開始
data.qvel[0] = 0.0
u = 0.0

for i in range(6000):  # 6 秒
  if i % decimation == 0:
    phi = wrap(data.qpos[0] - np.pi)
    u = np.clip(-kp * phi - kd * data.qvel[0], -TAU_MAX, TAU_MAX)
  data.ctrl[0] = u
  mujoco.mj_step(model, data)

print("最終誤差 =", wrap(data.qpos[0] - np.pi))
