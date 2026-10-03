"""学習した PPO 方策と LQR を同じ仮想実機で動かし、並べて動画にする.

  MUJOCO_GL=osmesa uv run python -m furuta.video   # 画面のない環境
  uv run python -m furuta.video                    # デスクトップ環境

方策は numpy で推論する（PyTorch 不要）。

前半: 実時間で 8 秒。後半: スイングアップ部分（最初の 1.6 秒）を 1/4 速度で再生。
"""

import argparse
import math
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .control import SwingUpLQR
from .plots import demo_config
from .render import camera
from .rl_env import CONTROL_DT, TIMESTEP, NumpyPolicy, PolicyController, recommended_config
from .sim import FurutaSim, wrap

FPS = 50
PANEL_W, PANEL_H = 640, 540
FONT_PATHS = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
]


def font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rollout(controller, truth, duration):
    """制御周期ごと（100 Hz）の qpos と表示用の値を記録する."""
    sim = FurutaSim(truth, seed=1, theta0=math.pi - 0.03)
    qs, info = [sim.d.qpos.copy()], [(0.0, sim.d.qpos[sim.jp], 0.0)]
    for _ in range(round(duration / CONTROL_DT)):
        w_ref = controller(sim.observe())
        tau = []
        sim.advance(w_ref, tau, log_every=1)
        qs.append(sim.d.qpos.copy())
        info.append((sim.d.time, sim.d.qpos[sim.jp], np.sqrt(np.mean([x[5] ** 2 for x in tau]))))
    return sim.m, np.array(qs), info


def frame_at(qs, t):
    """時刻 t の qpos（制御周期の間は線形補間。ヒンジ角は連続値なのでそのまま補間できる）."""
    x = t / CONTROL_DT
    i = min(int(x), len(qs) - 2)
    a = x - i
    return (1 - a) * qs[i] + a * qs[i + 1], min(round(x), len(qs) - 1)


def draw_overlay(img, title, sub, t, th, tau, slow):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, PANEL_W, 74], fill=(252, 252, 251))
    d.text((16, 10), title, font=font(26), fill=(11, 11, 11))
    d.text((16, 44), sub, font=font(17), fill=(82, 81, 78))
    up = abs(wrap(th)) < math.radians(15)
    status = "倒立中" if up else "スイングアップ中"
    lines = [f"t = {t:4.2f} s" + ("   1/4 スロー" if slow else ""), f"振子角 {math.degrees(wrap(th)):+6.1f}°   {status}",
             f"サーボトルク {tau:4.2f} N·m"]
    y = PANEL_H - 92
    d.rectangle([0, y - 8, PANEL_W, PANEL_H], fill=(252, 252, 251))
    for i, s in enumerate(lines):
        d.text((16, y + 26 * i), s, font=font(19), fill=(11, 11, 11) if i else (82, 81, 78))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/ppo_policy.npz", help="export_policy.py で書き出した重み")
    ap.add_argument("--out", default="results/ppo_vs_lqr.mp4")
    ap.add_argument("--duration", type=float, default=8.0)
    a = ap.parse_args()

    _, truth = demo_config()
    truth = truth.with_(control_dt=CONTROL_DT)
    truth.timestep = TIMESTEP
    runs = [
        ("PPO（学習した方策）", "ニューラルネット 64×64、ドメインランダム化", PolicyController(NumpyPolicy(a.model))),
        ("LQR（古典制御）", "エネルギー法スイングアップ + LQR", SwingUpLQR(recommended_config())),
    ]
    sims = [(title, sub, *rollout(c, truth, a.duration)) for title, sub, c in runs]
    model = sims[0][2]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, PANEL_H, PANEL_W)
    cam = camera([0.02, 0.0, 0.07], 0.27, 150, -15)

    # 実時間パート + スローモーション（最初の 1.6 秒を 1/4 速度）
    times = [(t, False) for t in np.arange(0, a.duration, 1 / FPS)]
    times += [(t, True) for t in np.arange(0, 1.6, 1 / FPS / 4)]

    w, h = PANEL_W * 2, PANEL_H
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE,
    )
    for t, slow in times:
        panels = []
        for title, sub, m, qs, info in sims:
            q, k = frame_at(qs, t)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            _, th, tau = info[k]
            panels.append(draw_overlay(renderer.render(), title, sub, t, th, tau, slow))
        frame = np.concatenate(panels, axis=1)
        frame[:, PANEL_W - 1 : PANEL_W + 1] = 200  # 区切り線
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    renderer.close()
    print(f"saved {a.out} ({len(times) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
