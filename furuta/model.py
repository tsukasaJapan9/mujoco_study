"""MJCF の生成.

関節構成:
  motor    : XL330 出力軸（鉛直軸）。ギア慣性を armature で表す
  backlash : 出力軸とアームの間のガタ（可動範囲だけ持つ自由関節）
  pend     : 振子（アーム方向の軸）。qpos=0 が倒立、pi が真下
"""

import mujoco

from .params import Config


def build_xml(cfg: Config) -> str:
    g, s, n = cfg.geom, cfg.servo, cfg.sense
    half = s.backlash / 2
    backlash = (
        f'<joint name="backlash" type="hinge" axis="0 0 1" limited="true" range="{-half} {half}" damping="1e-5" frictionloss="{s.backlash_friction}"/>'
        if s.backlash > 0
        else ""
    )
    tip = (
        f'<geom type="sphere" pos="0 0 {g.pend_length}" size="0.004" mass="{g.tip_mass}" rgba="0.9 0.3 0.2 1"/>'
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
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="box" size="0.0115 0.017 0.0135" pos="0 0 -0.0135" rgba="0.2 0.2 0.2 1" mass="0"/>
    <body name="motor">
      <joint name="motor" type="hinge" axis="0 0 1" armature="{s.gear_armature}"
             frictionloss="{s.gear_friction}" damping="{s.gear_damping}"/>
      <geom type="cylinder" size="0.008 0.001" mass="0.001"/>
      <body name="arm">
        {backlash}
        <geom type="box" size="{g.arm_length / 2} 0.004 0.003" pos="{g.arm_length / 2} 0 0.004" mass="{g.arm_mass}" rgba="0.8 0.8 0.8 1"/>
        <geom type="box" size="0.006 0.01 0.01" pos="{g.arm_length - 0.006} 0 0.004" mass="{g.hub_mass}" rgba="0.3 0.5 0.8 1"/>
        <body name="pend" pos="{g.arm_length} 0 0.004">
          <joint name="pend" type="hinge" axis="1 0 0" damping="{n.pend_damping}" frictionloss="{n.pend_friction}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {g.pend_length}" size="0.0015" mass="{g.rod_mass}" rgba="0.1 0.1 0.1 1"/>
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
