"""候補寸法を厳しめの条件（遅いサーボ・長い遅延・大きなガタ）で試す（ガタ片寄せあり）.

公称値のまま設計した制御器と、サーボ応答・遅延を同定して設計し直した制御器を比べる。

  uv run python -m furuta.stress
"""

import math
from multiprocessing import Pool

import numpy as np

from .evaluate import run_episode
from .params import Config
from .sweep import sample_true_params

CANDIDATES = [  # (L, r, tip)
    (0.04, 0.05, 0.008),
    (0.05, 0.05, 0.008),
    (0.06, 0.05, 0.004),
    (0.06, 0.05, 0.008),
    (0.08, 0.05, 0.004),
]


def harsh(rng):
    p = sample_true_params(rng)
    p.update(
        tau_servo=rng.uniform(0.03, 0.05),
        obs_delay=rng.uniform(0.003, 0.006),
        cmd_delay=rng.uniform(0.003, 0.006),
        backlash=math.radians(rng.uniform(1.0, 2.5)),
    )
    return p


def job(a):
    (L, r, tip), identified, i = a
    rng = np.random.default_rng(1000 + i)
    nominal = Config().with_(pend_length=L, arm_length=r, tip_mass=tip, arm_preload=0.01)
    p = harsh(rng)
    truth = nominal.with_(**p, arm_preload=0.01 * rng.uniform(0.5, 1.5))
    if identified:  # サーボ応答と遅延を同定し、その値で制御器を設計した場合（誤差 ±20%）
        e = lambda: rng.uniform(0.8, 1.2)
        nominal = nominal.with_(
            tau_servo=p["tau_servo"] * e(), obs_delay=p["obs_delay"] * e(), cmd_delay=p["cmd_delay"] * e()
        )
    _, m = run_episode(truth, duration=10.0, seed=i, ctrl_cfg=nominal)
    return a, m["success"] and m["phi_maxabs_deg"] <= 360, m["tau_rms_hold"]


if __name__ == "__main__":
    n = 24
    grid = [(c, ident, i) for c in CANDIDATES for ident in (False, True) for i in range(n)]
    with Pool() as pool:
        res = pool.map(job, grid)
    print("L[mm] r[mm] tip[g] identified  success  hold_torque_rms_median[Nm]")
    for c in CANDIDATES:
        for p in (False, True):
            rs = [x for x in res if x[0][0] == c and x[0][1] == p]
            ok = [x for x in rs if x[1]]
            tq = np.median([x[2] for x in ok]) if ok else float("nan")
            print(f"{c[0]*1e3:5.0f} {c[1]*1e3:5.0f} {c[2]*1e3:6.0f} {str(p):>10}  {len(ok)/len(rs):6.0%}  {tq:.3f}")
