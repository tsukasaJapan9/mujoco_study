"""スイープ結果とデモ走行の図を作る.

  uv run python -m furuta.plots
"""

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .evaluate import run_episode
from .params import Config
from .sweep import ARM_LENGTHS, PEND_LENGTHS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": False,
        "font.size": 9,
    }
)


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            r[k] = v == "True" if v in ("True", "False") else float(v)
    return rows


def success_grid(rows, tip, cable_deg=360.0):
    g = np.full((len(PEND_LENGTHS), len(ARM_LENGTHS)), np.nan)
    for i, L in enumerate(PEND_LENGTHS):
        for j, r in enumerate(ARM_LENGTHS):
            rs = [x for x in rows if x["pend_length"] == L and x["arm_length"] == r and x["tip_mass"] == tip]
            if rs:
                g[i, j] = np.mean([x["success"] and x["phi_maxabs_deg"] <= cable_deg for x in rs])
    return g


def heat(ax, g, title):
    im = ax.imshow(g, cmap=BLUES, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(ARM_LENGTHS)), [f"{r * 1e3:.0f}" for r in ARM_LENGTHS])
    ax.set_yticks(range(len(PEND_LENGTHS)), [f"{L * 1e3:.0f}" for L in PEND_LENGTHS])
    ax.set_xlabel("arm length r [mm]")
    ax.set_title(title, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def plot_sweeps():
    pre = load("results/sweep_preload.csv")
    nop = load("results/sweep_no_preload.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True, constrained_layout=True)
    heat(axes[0], success_grid(nop, 0.004), "No backlash countermeasure")
    im = heat(axes[1], success_grid(pre, 0.004), "Backlash preload 0.01 N·m")
    axes[0].set_ylabel("pendulum length L [mm]")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    fig.suptitle("Swing-up + 3 s hold success rate (16 randomized servos/size, tip mass 4 g, arm within ±1 rev)",
                 fontsize=10, x=0.02, ha="left")
    fig.savefig("results/success_map.png", dpi=150)

    tips = sorted({r["tip_mass"] for r in pre})
    fig, axes = plt.subplots(1, len(tips), figsize=(3.0 * len(tips), 3.6), sharey=True, constrained_layout=True)
    for ax, tip in zip(axes, tips):
        heat(ax, success_grid(pre, tip), f"tip mass {tip * 1e3:.0f} g")
    axes[0].set_ylabel("pendulum length L [mm]")
    fig.suptitle("Effect of tip mass (with backlash preload)", fontsize=10, x=0.02, ha="left")
    fig.savefig("results/tip_mass_effect.png", dpi=150)


def demo_config():
    """推奨寸法 + 不確かなパラメータは中央付近の値."""
    nominal = Config().with_(pend_length=0.06, arm_length=0.05, tip_mass=0.004, arm_preload=0.01)
    truth = nominal.with_(
        gear_armature=1.7e-3, gear_friction=0.04, tau_servo=0.02, backlash=math.radians(1.0),
        obs_delay=0.0025, cmd_delay=0.0025, pend_friction=4e-5,
    )
    return nominal, truth


def plot_demo():
    nominal, truth = demo_config()
    log, m = run_episode(truth, duration=6.0, seed=1, ctrl_cfg=nominal)
    title = f"LQR demo: L=60 mm, r=50 mm, tip 4 g, backlash 1.0°, delay 5 ms — swing-up in {m['t_up']:.2f} s"
    plot_timeseries(log, title, "results/demo_timeseries.png", "LQR balance mode")
    print(m)


def plot_timeseries(log, title, path, shade_label=None):
    t = log["t"]
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 7), sharex=True, constrained_layout=True)
    th = np.degrees(log["theta"])
    th[1:][np.abs(np.diff(th)) > 180] = np.nan  # ±180° の折り返しで線をつながない
    panels = [
        (th, "pendulum θ [deg]\n(0 = upright)"),
        (np.degrees(log["phi"]), "arm φ [deg]"),
        (None, "arm speed [rad/s]"),
        (log["tau"], "servo torque [N·m]"),
    ]
    for ax, (y, lab) in zip(axes, panels):
        if y is None:
            ax.plot(t, log["w_cmd"], color=SERIES[1], lw=1.5, label="command")
            ax.plot(t, log["w"], color=SERIES[0], lw=1.5, label="actual")
            ax.legend(frameon=False, loc="upper right", ncols=2)
        else:
            ax.plot(t, y, color=SERIES[0], lw=1.5)
        ax.set_ylabel(lab)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        up = log["mode"] > 0.5
        if up.any():
            ax.axvspan(t[np.argmax(up)], t[-1], color="#f0efec", zorder=0)
    axes[0].set_yticks([-180, -90, 0, 90, 180])
    axes[3].axhline(0.52, color=INK2, lw=0.8, ls="--")
    axes[3].axhline(-0.52, color=INK2, lw=0.8, ls="--")
    axes[3].text(t[-1], 0.45, "stall torque ±0.52", ha="right", va="top", fontsize=8, color=INK2)
    axes[-1].set_xlabel("time [s]" + (f"   (shaded: {shade_label})" if shade_label else ""))
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_demo()
    plot_sweeps()


def plot_learning_curve(npz="runs/ppo/evaluations.npz", path="results/ppo_learning_curve.png"):
    """EvalCallback の評価結果（真下から 5 秒、16 エピソードの平均報酬）."""
    d = np.load(npz)
    steps, r = d["timesteps"], d["results"]
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    ax.fill_between(steps / 1e6, r.min(1), r.max(1), color="#cde2fb", lw=0, label="min–max")
    ax.plot(steps / 1e6, r.mean(1), color=SERIES[0], lw=2, label="mean")
    ax.set_xlabel("environment steps [million]  (≈ hours of real-time experience: ×2.8)")
    ax.set_ylabel("episode reward")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("PPO learning curve (eval: 16 randomized servos, start hanging)", fontsize=10, loc="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)
