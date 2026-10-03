"""MJCF の生成.

関節構成:
  motor    : XL330 出力軸（鉛直軸）。ギア慣性を armature で表す
  backlash : 出力軸とアームの間のガタ（可動範囲だけ持つ自由関節）
  pend     : 振子（アーム方向の軸）。qpos=0 が倒立、pi が真下
"""

import mujoco

from .params import Config

MOUNT_HEIGHT = 0.075  # 机からモーター出力軸までの高さ [m]（振子が真下で机に当たらない高さ）

# 見た目だけの部品（質量 0・接触なし。物理には影響しない）
_VIS = 'contype="0" conaffinity="0" mass="0" group="1"'


def _visual_world(g) -> str:
    h = MOUNT_HEIGHT
    tower_top = h - 0.036
    ticks = "".join(
        f'<geom type="box" size="0.0004 {0.003 if i % 5 == 0 else 0.0018} 0.0003" '
        f'pos="{-0.05 + 0.01 * i} {-0.062 + (0.003 if i % 5 == 0 else 0.0018)} 0.0063" rgba="0.1 0.1 0.1 1" {_VIS}/>'
        for i in range(11)
    )
    return f"""
    <geom type="plane" size="0.3 0.3 0.01" material="table" {_VIS}/>
    <geom name="base" type="box" size="0.09 0.07 0.003" pos="0.02 0 0.003" material="wood" {_VIS}/>
    <geom name="ruler" type="box" size="0.052 0.004 0.0002" pos="0 -0.0585 0.0062" rgba="0.95 0.95 0.92 1" {_VIS}/>
    {ticks}
    <geom name="tower" type="box" size="0.016 0.02 {(tower_top - 0.006) / 2}" pos="0 0 {(tower_top + 0.006) / 2}" material="print" {_VIS}/>
    <geom name="xl330" type="box" size="0.010 0.013 0.017" pos="0 0 {h - 0.019}" material="servo" {_VIS}/>
    <geom name="xl330_label" type="box" size="0.0101 0.008 0.006" pos="0 0 {h - 0.017}" rgba="0.75 0.75 0.78 1" {_VIS}/>
    <geom name="openrb" type="box" size="0.025 0.0125 0.0008" pos="0.07 0.045 0.0072" rgba="0.1 0.45 0.25 1" {_VIS}/>
    <geom type="box" size="0.004 0.004 0.002" pos="0.092 0.045 0.0098" rgba="0.7 0.7 0.7 1" {_VIS}/>
    <geom name="cable" type="capsule" fromto="0.010 0.006 {h - 0.03} 0.05 0.045 0.009" size="0.0012" rgba="0.15 0.15 0.15 1" {_VIS}/>
"""



def build_xml(cfg: Config) -> str:
    h = MOUNT_HEIGHT
    g, s, n = cfg.geom, cfg.servo, cfg.sense
    half = s.backlash / 2
    backlash = (
        f'<joint name="backlash" type="hinge" axis="0 0 1" limited="true" range="{-half} {half}" damping="1e-5" frictionloss="{s.backlash_friction}"/>'
        if s.backlash > 0
        else ""
    )
    tip = (
        f'<geom name="tip" type="cylinder" pos="0 0 {g.pend_length}" size="0.0045 0.003" mass="{g.tip_mass}" material="brass"/>'
        if g.tip_mass > 0
        else ""
    )
    # アームの対地角 = motor + backlash。その角度に対する摩擦を固定テンドンで与える
    tendon = (
        f'<tendon><fixed name="arm_world" frictionloss="{s.arm_drag}"><joint joint="motor" coef="1"/>'
        f'<joint joint="backlash" coef="1"/></fixed></tendon>'
        if s.arm_drag > 0 and s.backlash > 0
        else ""
    )
    return f"""
<mujoco model="furuta_xl330">
  <compiler angle="radian"/>
  <option timestep="{cfg.timestep}" integrator="implicitfast">
    <flag contact="disable"/>
  </option>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.98 0.98 0.97" rgb2="0.86 0.88 0.9" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.91" rgb2="0.88 0.88 0.86" width="512" height="512"/>
    <material name="table" texture="grid" texrepeat="12 12"/>
    <material name="wood" rgba="0.78 0.66 0.5 1"/>
    <material name="print" rgba="0.95 0.95 0.95 1"/>
    <material name="servo" rgba="0.12 0.12 0.13 1" specular="0.3"/>
    <material name="brass" rgba="0.8 0.65 0.3 1" specular="0.6" shininess="0.6"/>
  </asset>
  <worldbody>
    <light pos="0.15 -0.2 0.45" dir="-0.3 0.4 -1" castshadow="true" diffuse="0.6 0.6 0.6"/>
    {_visual_world(g)}
    <geom name="spring" type="cylinder" size="0.011 0.0015" pos="0 0 {h + 0.0095}" material="brass" {_VIS}/>
    <body name="motor" pos="0 0 {h}">
      <joint name="motor" type="hinge" axis="0 0 1" armature="{s.gear_armature}"
             frictionloss="{s.gear_friction}" damping="{s.gear_damping}"/>
      <geom name="horn" type="cylinder" size="0.008 0.001" mass="0.001" rgba="0.7 0.7 0.72 1"/>
      <body name="arm">
        {backlash}
        <geom type="box" size="{g.arm_length / 2} 0.004 0.003" pos="{g.arm_length / 2} 0 0.004" mass="{g.arm_mass}" material="print"/>
        <geom name="hub" type="box" size="0.006 0.01 0.01" pos="{g.arm_length - 0.006} 0 0.004" mass="{g.hub_mass}" rgba="0.25 0.5 0.85 1"/>
        <geom name="as5600" type="box" size="0.0008 0.009 0.009" pos="{g.arm_length - 0.0128} 0 0.004" rgba="0.1 0.45 0.25 1" {_VIS}/>
        <body name="pend" pos="{g.arm_length} 0 0.004">
          <joint name="pend" type="hinge" axis="1 0 0" damping="{n.pend_damping}" frictionloss="{n.pend_friction}"/>
          <geom name="shaft" type="cylinder" fromto="-0.012 0 0 0.003 0 0" size="0.0015" rgba="0.75 0.75 0.75 1" {_VIS}/>
          <geom name="rod" type="capsule" fromto="0 0 0 0 0 {g.pend_length}" size="0.0015" mass="{g.rod_mass}" rgba="0.1 0.1 0.1 1"/>
          {tip}
        </body>
      </body>
    </body>
  </worldbody>
  {tendon}
  <actuator>
    <motor name="motor" joint="motor" ctrllimited="false"/>
  </actuator>
</mujoco>
"""


def build_model(cfg: Config) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(cfg))
