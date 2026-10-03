"""推奨寸法のハードウェアをレンダリングする（スクリーンショット）.

  MUJOCO_GL=osmesa uv run python -m furuta.render   # 画面のない環境
  uv run python -m furuta.render                    # デスクトップ環境
"""

import math

import matplotlib

matplotlib.use("Agg")
import logging

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .control import SwingUpLQR
from .model import MOUNT_HEIGHT
from .plots import demo_config
from .sim import FurutaSim

W, H = 1200, 900
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["IPAGothic", "Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic", "sans-serif"]


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def project(m, cam, p):
    """ワールド座標の点を画像座標へ（MuJoCo の自由カメラと同じ規約）."""
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.array(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = np.asarray(p) - pos
    x, y, z = d @ right, d @ up, d @ fwd
    f = (H / 2) / math.tan(math.radians(m.vis.global_.fovy) / 2)
    return W / 2 + f * x / z, H / 2 - f * y / z


def snapshots(times):
    """デモ走行から指定時刻の qpos を取り出す."""
    nominal, truth = demo_config()
    sim = FurutaSim(truth, seed=1, theta0=math.pi - 0.03)
    ctrl = SwingUpLQR(nominal)
    out, t = [], 0.0
    for target in times:
        if target > t:
            sim.run(ctrl, target - t, log_every=10**9)
            t = target
        out.append(sim.d.qpos.copy())
    return sim.m, out


def render(m, qpos, cam):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, H, W) as r:
        opt = mujoco.MjvOption()
        r.update_scene(d, cam, opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        return r.render(), d


def annotated_overview(m, qpos):
    cam = camera([0.025, 0.0, 0.07], 0.30, 120, -22)
    img, d = render(m, qpos, cam)
    g = lambda name: d.geom_xpos[m.geom(name).id]
    labels = [
        ("tip", "先端おもり 4 g", (60, -40)),
        ("rod", "振子 φ3 カーボン棒 60 mm", (90, 0)),
        ("hub", "ベアリング 2 個 + 磁石", (110, -20)),
        ("as5600", "AS5600（振子角度）", (40, 110)),
        ("spring", "ゼンマイばね（ガタ片寄せ）", (-260, -40)),
        ("xl330", "XL330-M288-T（速度制御モード）", (-300, 10)),
        ("tower", "3D プリントの台", (-220, 40)),
        ("openrb", "OpenRB-150（200 Hz 制御）", (60, 70)),
        ("ruler", "目盛 100 mm", (-160, 60)),
    ]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img)
    for geom, text, (dx, dy) in labels:
        p = g(geom)
        u, v = project(m, cam, p)
        ax.annotate(
            text, (u, v), (u + dx, v + dy), fontsize=13, color="#0b0b0b",
            arrowprops=dict(arrowstyle="-", color="#52514e", lw=1),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )
    ax.set_axis_off()
    ax.set_title("XL330 回転倒立振子（推奨寸法：振子 60 mm・アーム 50 mm）", fontsize=15, loc="left")
    fig.subplots_adjust(0, 0, 1, 0.95)
    fig.savefig("results/hardware_overview.png", dpi=100)
    plt.close(fig)


def sequence(m, states, titles):
    cam = camera([0.02, 0.0, 0.07], 0.26, 150, -15)
    fig, axes = plt.subplots(1, len(states), figsize=(5 * len(states), 4.2), dpi=100)
    for ax, q, t in zip(axes, states, titles):
        img, _ = render(m, q, cam)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(t, fontsize=14, loc="left")
    fig.tight_layout()
    fig.savefig("results/hardware_sequence.png", dpi=100)
    plt.close(fig)


def main():
    times = [0.0, 0.42, 0.78, 3.0]
    m, qs = snapshots(times)
    annotated_overview(m, qs[-1])
    sequence(m, qs, ["① 真下で静止", "② 振って勢いをつける", "③ 振り上げ", "④ 倒立を保持"])
    print("saved results/hardware_overview.png, results/hardware_sequence.png", "mount height", MOUNT_HEIGHT)


if __name__ == "__main__":
    main()
