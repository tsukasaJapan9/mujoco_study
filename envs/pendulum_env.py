import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_PATH = "models/pendulum_torque.xml"
TAU_MAX = 0.5  # ctrlrange と合わせる
CTRL_HZ = 100  # Step 4 と同じ制御周期


class PendulumEnv(gym.Env):
  metadata = {"render_modes": [], "render_fps": CTRL_HZ}

  def __init__(self):
    self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    self.data = mujoco.MjData(self.model)
    self.frame_skip = round((1 / self.model.opt.timestep) / CTRL_HZ)

    self.observation_space = spaces.Box(
      low=np.array([-1.0, -1.0, -20.0], dtype=np.float32),
      high=np.array([1.0, 1.0, 20.0], dtype=np.float32),
      dtype=np.float32,
    )
    self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    self.torque_low = np.array([-TAU_MAX], dtype=np.float32)
    self.torque_high = np.array([TAU_MAX], dtype=np.float32)

  def _obs(self):
    theta = self.data.qpos[0]
    omega = self.data.qvel[0]
    return np.array(
      [np.cos(theta), np.sin(theta), omega], dtype=np.float32
    )  # ← ここだけ考える

  def _scale(self, action):
    low, high = self.torque_low, self.torque_high
    return low + (action + 1.0) * 0.5 * (high - low)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)  # 乱数の種を Gymnasium に渡す
    mujoco.mj_resetData(self.model, self.data)  # MuJoCo を初期状態に戻す
    self.data.qpos[0] = self.np_random.uniform(-0.1, 0.1)
    self.data.qvel[0] = self.np_random.uniform(-0.1, 0.1)
    mujoco.mj_forward(self.model, self.data)  # 派生量を埋める
    return self._obs(), {}

  def step(self, action):
    torque = self._scale(np.clip(action, -1.0, 1.0))[0]
    self.data.ctrl[0] = torque
    for _ in range(self.frame_skip):  # Step 4 の decimation と同じ
      mujoco.mj_step(self.model, self.data)

    theta = self.data.qpos[0]
    omega = self.data.qvel[0]

    r_angle = -np.cos(theta)
    r_omega = -0.01 * omega**2
    r_torque = -0.001 * torque**2
    reward = float(r_angle + r_omega + r_torque)
    upright = float(-np.cos(theta) > 0.95)
    terminated = False  # 6-c で扱う
    truncated = False
    info = {
      "upright": upright,
      "r_angle": r_angle,
      "r_omega": r_omega,
      "r_torque": r_torque,
    }
    return self._obs(), reward, terminated, truncated, info
