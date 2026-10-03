"""学習した PPO 方策と LQR を、同じ「ランダムな仮想実機」で比較する.

  uv run --extra rl python -m furuta.rl_eval runs/ppo/best_model.zip
"""

import argparse
import math
from multiprocessing import Pool

import numpy as np

from .control import SwingUpLQR
from .evaluate import metrics
from .rl_env import PolicyController, recommended_config
from .sim import FurutaSim
from .stress import harsh
from .sweep import sample_true_params

MODEL_PATH = None


def make_truth(i, kind):
    rng = np.random.default_rng(5000 + i)
    nominal = recommended_config()
    p = sample_true_params(rng) if kind == "standard" else harsh(rng)
    p["arm_preload"] = nominal.servo.arm_preload * rng.uniform(0.5, 1.5)
    return nominal, nominal.with_(**p), p, rng


def episode(args):
    i, kind, ctrl_name = args
    nominal, truth, p, rng = make_truth(i, kind)
    if ctrl_name == "ppo":
        from stable_baselines3 import PPO

        import torch

        torch.set_num_threads(1)
        ctrl = PolicyController(PPO.load(MODEL_PATH, device="cpu"))
    elif ctrl_name == "lqr":
        ctrl = SwingUpLQR(nominal)
    else:  # lqr_identified: サーボ応答と遅延を ±20% の誤差で同定して設計
        e = lambda: rng.uniform(0.8, 1.2)
        ident = nominal.with_(tau_servo=p["tau_servo"] * e(), obs_delay=p["obs_delay"] * e(), cmd_delay=p["cmd_delay"] * e())
        ctrl = SwingUpLQR(ident)
    sim = FurutaSim(truth, seed=i, theta0=math.pi + np.random.default_rng(i).uniform(-0.05, 0.05))
    log = sim.run(ctrl, 10.0)
    m = metrics(log)
    m["ok"] = m["success"] and m["phi_maxabs_deg"] <= 360
    return (kind, ctrl_name), m, log


def init(path):
    global MODEL_PATH
    MODEL_PATH = path


def plot_ppo_demo(model_path):
    """LQR のデモ（plots.plot_demo）と同じ仮想実機で PPO を動かす."""
    from stable_baselines3 import PPO

    from .plots import demo_config, plot_timeseries
    from .rl_env import CONTROL_DT, TIMESTEP

    _, truth = demo_config()
    truth = truth.with_(control_dt=CONTROL_DT)
    truth.timestep = TIMESTEP
    sim = FurutaSim(truth, seed=1, theta0=math.pi - 0.03)
    log = sim.run(PolicyController(PPO.load(model_path, device="cpu")), 6.0)
    m = metrics(log)
    title = f"PPO demo: same virtual servo as LQR demo — swing-up in {m['t_up']:.2f} s"
    plot_timeseries(log, title, "results/ppo_demo_timeseries.png")
    print("PPO demo", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-n", type=int, default=32)
    a = ap.parse_args()
    plot_ppo_demo(a.model)
    jobs = [(i, kind, c) for kind in ("standard", "harsh") for c in ("ppo", "lqr", "lqr_identified") for i in range(a.n)]
    with Pool(initializer=init, initargs=(a.model,)) as pool:
        res = pool.map(episode, jobs)
    print(f"{'条件':10s} {'制御器':16s} {'成功率':>6s} {'振り上げ時間中央値[s]':>10s} {'倒立中θ RMS[deg]':>8s} {'倒立中トルクRMS中央値[Nm]':>10s}")
    for kind in ("standard", "harsh"):
        for c in ("ppo", "lqr", "lqr_identified"):
            ms = [m for k, m, _ in res if k == (kind, c)]
            ok = [m for m in ms if m["ok"]]
            med = lambda key: np.median([m[key] for m in ok]) if ok else float("nan")
            print(f"{kind:10s} {c:16s} {len(ok) / len(ms):6.0%} {med('t_up'):10.2f} {med('theta_rms_deg'):8.2f} {med('tau_rms_hold'):10.3f}")
    return res


if __name__ == "__main__":
    main()
