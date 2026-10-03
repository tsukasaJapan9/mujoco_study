"""寸法スイープ + 不確かなパラメータのランダム化による実現性評価.

各寸法（振子長 L × アーム長 r）について、ギア慣性・摩擦・サーボ応答・ガタ・遅延などを
範囲内でランダムに振った「仮想実機」を N 台つくり、公称値で設計した同じ制御器で
スイングアップ→倒立保持できた割合を数える。

  uv run python -m furuta.sweep            # 結果は results/ に保存
"""

import argparse
import csv
import itertools
import math
from multiprocessing import Pool

import numpy as np

from .evaluate import run_episode
from .params import Config

PEND_LENGTHS = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
ARM_LENGTHS = [0.03, 0.04, 0.05, 0.06, 0.08]
CABLE_LIMIT_DEG = 360.0  # 配線のねじれ対策でアームを ±1 回転以内に収めたい


def sample_true_params(rng: np.random.Generator) -> dict:
    """実機の不確かさ（未同定のパラメータ）の範囲."""
    s = Config().servo
    return dict(
        gear_armature=float(np.exp(rng.uniform(np.log(0.7e-3), np.log(4e-3)))),
        gear_friction=rng.uniform(0.01, 0.06),
        tau_servo=rng.uniform(0.008, 0.03),
        backlash=math.radians(rng.uniform(0.0, 1.5)),
        backlash_friction=rng.uniform(0.0, 0.003),
        obs_delay=rng.uniform(0.001, 0.004),
        cmd_delay=rng.uniform(0.001, 0.004),
        pend_friction=float(np.exp(rng.uniform(np.log(1e-5), np.log(1.5e-4)))),
        stall_torque=s.stall_torque * rng.uniform(0.8, 1.0),
        no_load_speed=s.no_load_speed * rng.uniform(0.85, 1.0),
    )


def job(args):
    L, r, tip, preload, i = args
    rng = np.random.default_rng(i)
    nominal = Config().with_(pend_length=L, arm_length=r, tip_mass=tip, arm_preload=preload)
    true_params = sample_true_params(rng)
    true_params["arm_preload"] = preload * rng.uniform(0.5, 1.5)
    truth = nominal.with_(**true_params)
    _, m = run_episode(truth, duration=10.0, seed=i, ctrl_cfg=nominal)
    return dict(pend_length=L, arm_length=r, tip_mass=tip, sample=i, **m, **true_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=16, help="寸法ごとのランダム実機の数")
    ap.add_argument("--tip", type=float, nargs="+", default=[0.004])
    ap.add_argument("--preload", type=float, default=0.01, help="ガタ片寄せトルク [N m]（0 で対策なし）")
    ap.add_argument("--out", default="results/sweep.csv")
    a = ap.parse_args()
    grid = list(itertools.product(PEND_LENGTHS, ARM_LENGTHS, a.tip, [a.preload], range(a.n)))
    with Pool() as p:
        rows = p.map(job, grid, chunksize=4)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {a.out} ({len(rows)} episodes)")
    summarize(rows)


def summarize(rows):
    keys = sorted({(r["tip_mass"], r["pend_length"], r["arm_length"]) for r in rows})
    print("tip[g]  L[mm]  r[mm]  success  success&cable  median_t_up[s]  hold_torque_rms_p90[Nm]")
    for k in keys:
        rs = [r for r in rows if (r["tip_mass"], r["pend_length"], r["arm_length"]) == k]
        ok = [r for r in rs if r["success"]]
        okc = [r for r in ok if r["phi_maxabs_deg"] <= CABLE_LIMIT_DEG]
        tu = np.median([r["t_up"] for r in ok]) if ok else float("nan")
        tq = np.percentile([r["tau_rms_hold"] for r in ok], 90) if ok else float("nan")
        print(
            f"{k[0] * 1e3:5.0f} {k[1] * 1e3:6.0f} {k[2] * 1e3:6.0f}"
            f"  {len(ok) / len(rs):6.0%}  {len(okc) / len(rs):12.0%}  {tu:10.1f}  {tq:14.3f}"
        )


if __name__ == "__main__":
    main()
