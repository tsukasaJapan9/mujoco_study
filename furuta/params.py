"""機構・サーボ・センサーのパラメータ.

XL330-M288-T の値はデータシート（5V 時）を基準にした推定値。
ギア慣性・サーボ応答・ガタは不明なので、robustness 評価で範囲を振る。
"""

from dataclasses import dataclass, field, replace
import math


@dataclass
class Geometry:
    arm_length: float = 0.06  # r [m] モーター軸〜振子軸
    pend_length: float = 0.12  # L [m] 振子の棒の長さ
    rod_density: float = 0.0113  # [kg/m] φ3mm カーボンロッド相当
    tip_mass: float = 0.004  # [kg] 振子先端のおもり（M3 ナット数個程度）
    arm_mass: float = 0.004  # [kg] 3D プリントのアーム
    hub_mass: float = 0.008  # [kg] アーム先端（ベアリング・AS5600 基板・ハウジング）

    @property
    def rod_mass(self) -> float:
        return self.rod_density * self.pend_length

    @property
    def pend_mass(self) -> float:
        return self.rod_mass + self.tip_mass

    @property
    def m_l(self) -> float:
        """振子の m * l_c（重心までの距離との積）."""
        return self.rod_mass * self.pend_length / 2 + self.tip_mass * self.pend_length

    @property
    def j_pivot(self) -> float:
        """振子軸まわりの慣性モーメント."""
        return self.rod_mass * self.pend_length**2 / 3 + self.tip_mass * self.pend_length**2


@dataclass
class Servo:
    stall_torque: float = 0.52  # [N m] 5V
    no_load_speed: float = 103 * 2 * math.pi / 60  # [rad/s] 5V
    gear_armature: float = 2.0e-3  # [kg m^2] 回転子慣性 x 減速比^2（未知・要同定）
    gear_friction: float = 0.03  # [N m] クーロン摩擦（未知・要同定）
    gear_damping: float = 0.0  # [N m s]
    tau_servo: float = 0.015  # [s] 速度制御ループの時定数（未知・要同定）
    backlash: float = math.radians(0.5)  # [rad] 全ガタ幅（未知・要同定）
    backlash_friction: float = 0.0  # [N m] ガタの中でアームが受ける摩擦（出力軸ブッシュ等、未知）
    arm_drag: float = 0.0  # [N m] アームと固定フレームの間にわざと付ける摩擦（O リング等）
    arm_preload: float = 0.0  # [N m] アームに常に一方向へかける一定トルク（ゼンマイばね等でガタを片寄せ）


@dataclass
class Sensing:
    control_dt: float = 0.005  # [s] 制御周期 200 Hz
    obs_delay: float = 0.002  # [s] センサー読み出しの遅延
    cmd_delay: float = 0.002  # [s] 指令がサーボに届くまでの遅延
    encoder_counts: int = 4096  # AS5600 / XL330 とも 12bit
    pend_noise_counts: float = 1.0  # 振子エンコーダのノイズ [LSB, 1σ]
    pend_damping: float = 2e-6  # [N m s] 振子ベアリング
    pend_friction: float = 5e-5  # [N m] 振子ベアリング


@dataclass
class Config:
    geom: Geometry = field(default_factory=Geometry)
    servo: Servo = field(default_factory=Servo)
    sense: Sensing = field(default_factory=Sensing)
    timestep: float = 0.0005

    def with_(self, **kw) -> "Config":
        """geom/servo/sense のフィールド名を直接指定してコピーを作る."""
        g, s, n = {}, {}, {}
        for k, v in kw.items():
            if hasattr(self.geom, k):
                g[k] = v
            elif hasattr(self.servo, k):
                s[k] = v
            elif hasattr(self.sense, k):
                n[k] = v
            else:
                raise KeyError(k)
        return Config(replace(self.geom, **g), replace(self.servo, **s), replace(self.sense, **n), self.timestep)
