# Step 1: 環境構築と最初のシミュレーション

**目安時間: 4時間** / 前提: Python・NumPy が書ける

---

## 1-0. このステップのゴール

このステップが終わると、次のことができるようになります。

- MuJoCo が入った venv がある
- XML でモデルを書き、Python から読み込んで動かせる
- **`MjModel` と `MjData` の違い**を自分の言葉で説明できる
- ビューアで 3D 表示しながらシミュレーションを回せる

MuJoCo の学習で最初につまずくのは「物理」ではなく「何が定数で何が状態か」の混乱です。
そこだけは今日中に潰します。

---

## 1-1. 環境構築

### venv を作る

```bash
cd ~/works/mujoco_study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

以降、作業のたびに `source .venv/bin/activate` を忘れないこと。
プロンプトの先頭に `(.venv)` が出ていれば有効です。

### MuJoCo を入れる

```bash
pip install mujoco
```

**今はこれだけ入れます。** Gymnasium も PyTorch もまだ不要です。
一度に全部入れると依存関係の衝突が起きたとき原因が分からなくなります。

### `.gitignore` を作る

リポジトリ直下に以下の内容で `.gitignore` を作ってください。

```gitignore
.venv/
__pycache__/
*.pyc
runs/
logs/
models_out/
*.zip
*.mp4
```

### 動作確認

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

バージョン番号（`3.x.x`）が出れば OK。

### ビューアの動作確認

```bash
python -m mujoco.viewer
```

MuJoCo のビューアウィンドウが開けば成功です。
マウス左ドラッグで回転、右ドラッグで平行移動、ホイールでズームできます。

> **うまくいかないとき（Ubuntu 22.04 でよくある症状）**
>
> | 症状 | 対処 |
> |---|---|
> | `libGL.so.1` が見つからない | `sudo apt install libgl1-mesa-glx libglfw3 libosmesa6` |
> | ウィンドウが真っ黒 | NVIDIA ドライバの GL 設定。`export MUJOCO_GL=glfw` を試す |
> | SSH 越しで X が無い | `export MUJOCO_GL=egl` にしてオフスクリーン描画（ビューアは使えないが計算は動く） |
> | `GLFWError` | `sudo apt install libglfw3-dev` |
>
> 解決しない場合は、エラーメッセージ全文を貼って質問してください。

---

## 1-2. 概念: `MjModel` と `MjData`

MuJoCo の Python API はほぼこの2つだけ理解すれば動かせます。

```
┌─────────────────────────────────────────────────────────┐
│  MJCF (XML)                                             │
│  「このロボットはこういう形で、こういう関節がある」       │
└──────────────────────┬──────────────────────────────────┘
                       │  mujoco.MjModel.from_xml_path()
                       ▼
┌─────────────────────────────────────────────────────────┐
│  MjModel  ── 定数（コンパイル済みの設計図）              │
│                                                         │
│   ・body の質量・慣性・形状                              │
│   ・joint の種類・軸・可動範囲                           │
│   ・actuator の設定、gear 比                             │
│   ・opt.timestep, opt.gravity, 積分器の種類              │
│                                                         │
│   ★ シミュレーション中に変化しない                       │
│   ★ 実機で言えば「ロボットの図面とスペック表」            │
└──────────────────────┬──────────────────────────────────┘
                       │  mujoco.MjData(model)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  MjData  ── 状態（今この瞬間の値）                       │
│                                                         │
│   ・qpos  一般化座標（関節角度・位置）      [nq]         │
│   ・qvel  一般化速度                        [nv]         │
│   ・ctrl  アクチュエータへの指令（入力）    [nu]         │
│   ・time  シミュレーション時刻                           │
│   ・xpos  各 body のワールド座標（計算結果）             │
│   ・sensordata  センサの読み値                           │
│                                                         │
│   ★ mj_step() のたびに書き換わる                         │
│   ★ 実機で言えば「今のセンサ値とモータ指令」              │
└─────────────────────────────────────────────────────────┘
```

**`mj_step(model, data)` が1回でやること（概念）**:

1. 現在の `qpos`, `qvel` から順運動学を解いて各 body のワールド座標 `xpos` を求める
2. 質量行列 `M(q)`、コリオリ・遠心力 `C(q,q̇)`、重力を計算
3. `ctrl` からアクチュエータ力を計算
4. 接触を検出し、拘束力を解く
5. 加速度 `q̈` を求め、`timestep` だけ時間積分して `qpos`, `qvel` を更新
6. `data.time += timestep`

この 1〜6 が「1 physics step」です。デフォルトの `timestep` は 0.002 秒（500Hz）。

### `nq` と `nv` は必ずしも一致しない

- `nq`: 一般化**座標**の次元
- `nv`: 一般化**速度**の次元

hinge や slide 関節だけなら `nq == nv` です。しかし **free joint（6自由度の浮遊body）**では
位置3 + クォータニオン4 = **7** が `nq`、速度3 + 角速度3 = **6** が `nv` になります。
四足歩行ロボットで必ず出てくるので、今のうちに頭の隅に置いてください。

---

## 1-3. 最初のモデル（写経ではなく、読んで理解する）

以下は最小の振り子モデルです。**まずこれを Python の文字列として使います**（ファイル分離は Step 2 で）。

```xml
<mujoco model="simple_pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
      <geom name="pole_geom" type="capsule" fromto="0 0 0  0 0 -0.5"
            size="0.02" rgba="0.2 0.4 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
```

読み方:

| 要素 | 意味 |
|---|---|
| `<option>` | シミュレーション全体の設定。`timestep` は 1 ステップの時間 [秒] |
| `<worldbody>` | 世界の根。ここに置いた `geom` は動かない静的な物体 |
| `<body>` | 剛体。`pos` は**親からの相対位置** |
| `<joint>` | 親 body との接続方法。`type="hinge"` は回転関節、`axis="0 1 0"` は Y 軸まわり |
| `<geom>` | 形状。**質量と慣性はここから自動計算される**（密度デフォルト 1000 kg/m³） |
| `fromto` | カプセルの両端点。ここでは body 原点から Z 方向に -0.5m（下向き 50cm の棒） |

---

## 1-4. 課題 1: シミュレーションを回して状態を観察する

`sim/hello_mujoco.py` を作ってください。ディレクトリも自分で作ります。

**やること**:
1. 上の XML を Python の文字列変数に入れる
2. `mujoco.MjModel.from_xml_string()` でモデルを作る
3. `mujoco.MjData(model)` でデータを作る
4. モデルの構造情報（`nq`, `nv`, `nu`, `njnt`, `nbody`, `opt.timestep`）を print する
5. 初期角度を少しだけ傾ける（`data.qpos[0] = 0.1`）
6. 1000 ステップ回し、100 ステップごとに `time`, `qpos`, `qvel` を print する

**骨格スニペット**（これを埋めて完成させてください）:

```python
import mujoco
import numpy as np

XML = """
... ここに上の XML ...
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
print(f"timestep={model.opt.timestep}")

data.qpos[0] = 0.1

for i in range(1000):
    mujoco.mj_step(model, data)
    if i % 100 == 0:
        print(f"t={data.time:6.3f}  qpos={data.qpos}  qvel={data.qvel}")
```

**観察して答えを出すこと**:
- `nq`, `nv`, `nu` はいくつになったか。なぜその値か
- 1000 ステップ後の `data.time` はいくつか。`timestep` との関係は
- `qpos[0]` の符号は時間とともにどう変化するか。それは物理的に正しいか
- 振動の周期は何秒か。理論値 `T = 2π√(L/g)`（L は重心までの距離 = 0.25m）と比べてどうか

> **ヒント**: 重心は棒の中心なので、単振り子の理論式をそのまま当てるとズレます。
> 剛体棒の物理振子の周期は `T = 2π√(I/(m·g·d))`（I は回転軸まわりの慣性モーメント、d は軸から重心までの距離）。
> ズレた理由を考えるのがこの課題の本題です。

---

## 1-5. 課題 2: ビューアで見る

`sim/view_pendulum.py` を作り、リアルタイム表示してください。

**骨格スニペット**:

```python
import time
import mujoco
import mujoco.viewer

# model, data の作成は課題1と同じ

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        # 実時間に合わせて待つ（これが無いと超高速で回る）
        wait = model.opt.timestep - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)
```

**`launch_passive` とは**: MuJoCo にシミュレーションループを渡さず、
**自分のループで `mj_step` を呼びながら表示だけしてもらう**モードです。
RL では自分がループを持つので、常にこちらを使います。

**観察すること**:
- `time.sleep` の行を消すとどうなるか（**シミュレーション時間と実時間は別物**という感覚を掴む）
- ビューア上でスペースキー、`Ctrl+ドラッグ`で body を掴んで力を加えられる。試してみる
- 左パネルの各種可視化オプション（接触点、慣性、座標軸）を切り替えてみる

---

## 1-6. 課題 3: 状態を配列に貯めてプロットする（任意だが強く推奨）

RL の実験では「ログを取ってプロットする」を毎回やります。今のうちに癖にしてください。

```bash
pip install matplotlib
```

角度 `qpos[0]` と角速度 `qvel[0]` の時系列をプロットしてください。
横軸は `data.time`。位相平面（横軸 `qpos[0]`、縦軸 `qvel[0]`）も描くと、
**閉じた軌道（エネルギー保存）**が見えます。

さらに `<joint>` に `damping="0.1"` を足して、位相平面が渦巻きに変わるのを確認してください。
これが「エネルギーが散逸する」ということの目で見える形です。

---

## 1-7. 完了条件チェックリスト

- [ ] `python -c "import mujoco; print(mujoco.__version__)"` が通る
- [ ] `python -m mujoco.viewer` でウィンドウが開き、マウス操作できる
- [ ] `sim/hello_mujoco.py` が動き、状態が時間発展するのを確認した
- [ ] `nq` / `nv` / `nu` の値と、その理由を説明できる
- [ ] `MjModel` と `MjData` のどちらが `mj_step` で変化するか説明できる
- [ ] 振動周期を実測し、理論値と比較して差の理由を考えた
- [ ] `launch_passive` でリアルタイム表示できた
- [ ] （推奨）位相平面をプロットし、damping ありなしの違いを見た

---

## 1-8. つまずきやすいポイント

| 症状 | 原因と対処 |
|---|---|
| `qpos` が全く変化しない | 初期角度がちょうど 0（安定/不安定平衡点）。`data.qpos[0] = 0.1` を入れる |
| 振り子が一瞬で飛んでいく | 初期値の入れ間違い、または `timestep` が大きすぎる |
| ビューアがすぐ閉じる | `with` ブロックを抜けている。`while viewer.is_running()` のループが必要 |
| 動きが速すぎて見えない | `time.sleep` による実時間同期を入れていない |
| `data.qpos` を代入しても反映されない | `data.qpos = [...]` ではなく `data.qpos[:] = [...]` または要素代入を使う（ビューを壊さない） |

---

## 次のステップ

Step 1 が終わったら「終わりました」と教えてください。
**課題の観察結果（特に周期の理論値との比較）を報告**してもらえると、
理解のズレをその場で直せます。

Step 2 では XML をファイルに分離し、MJCF の文法を体系的に学びます。
カートポールを自力で組めるところまで行きます。
