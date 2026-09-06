# sim/view_model.py — models/ 以下の MJCF を読み込んで表示する
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# mjtJoint の並び順（0=free, 1=ball, 2=slide, 3=hinge）
JOINT_TYPES = ("free", "ball", "slide", "hinge")
SENSOR_PREFIX = len("mjSENS_")


def load(name):
  path = MODELS / name
  if not path.exists():
    available = [p.name for p in sorted(MODELS.glob("*.xml"))]
    raise SystemExit(f"モデルが見つかりません: {path}\n利用可能: {available}")
  try:
    return mujoco.MjModel.from_xml_path(str(path))
  except Exception as e:
    # MJCF のコンパイルエラーはここに来る。行番号が出るのでそのまま読む
    raise SystemExit(f"MJCF の読み込みに失敗しました:\n{e}")


def summary(model):
  """モデルの構造を表示する。書いた MJCF が意図どおりか確認するために毎回見る。"""
  print(
    f"nq={model.nq}  nv={model.nv}  nu={model.nu}  "
    f"nbody={model.nbody}  njnt={model.njnt}  nsensordata={model.nsensordata}"
  )

  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    jtype = JOINT_TYPES[model.jnt_type[j]]
    print(
      f"  joint[{j}] {name:<12} {jtype:<6} "
      f"qposadr={model.jnt_qposadr[j]} dofadr={model.jnt_dofadr[j]}"
    )

  for s in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)
    stype = mujoco.mjtSensor(model.sensor_type[s]).name[SENSOR_PREFIX:].lower()
    adr, dim = model.sensor_adr[s], model.sensor_dim[s]
    print(f"  sensor[{s}] {name:<12} {stype:<12} adr={adr} dim={dim}")

  for a in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    print(f"  actuator[{a}] {name:<12} ctrlrange={model.actuator_ctrlrange[a]}")


def main():
  parser = argparse.ArgumentParser(description="MJCF を読み込んで表示する")
  parser.add_argument("model", help="models/ 以下のファイル名 (例: cartpole.xml)")
  parser.add_argument(
    "--sim", action="store_true", help="物理を進める（既定は静止表示）"
  )
  parser.add_argument("--speed", type=float, default=1.0, help="再生速度の倍率")
  args = parser.parse_args()

  model = load(args.model)
  data = mujoco.MjData(model)

  # 初期値の設定
  data.qpos[1] = 0.25
  # トルクの設定
  # data.ctrl[0] = 2.0

  summary(model)

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  mujoco.mj_forward(model, data)  # 進めずに派生量だけ計算（xpos などを埋める）

  with mujoco.viewer.launch_passive(model, data) as viewer:
    # XMLにあるカメラをシーンカメラに設定
    if model.vis.global_.cameraid >= 0:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = model.vis.global_.cameraid
    else:
      # フリーカメラをmodel(extent)由来の初期値で上書き
      viewer.cam.distance = cam.distance
      viewer.cam.azimuth = cam.azimuth
      viewer.cam.elevation = cam.elevation
      viewer.cam.lookat[:] = cam.lookat

    i = 0
    while viewer.is_running():
      t0 = time.time()

      if args.sim:
        mujoco.mj_step(model, data)
        viewer.sync()

        wait = model.opt.timestep / args.speed - (time.time() - t0)
        if wait > 0:
          time.sleep(wait)
      else:
        viewer.sync()
        time.sleep(1 / 60)  # 静止表示なら 60Hz で十分


if __name__ == "__main__":
  main()
