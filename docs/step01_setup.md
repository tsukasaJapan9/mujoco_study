# Step 1: 環境構築と最初のシミュレーション

**目安時間: 4時間** / 前提: Python・NumPy が書ける

---

## 1-0. このステップのゴール

このステップが終わると、次のことができるようになります。

- MuJoCo が入った uv 環境がある
- XML でモデルを書き、Python から読み込んで動かせる
- **`MjModel` と `MjData` の違い**を自分の言葉で説明できる
- ビューアで 3D 表示しながらシミュレーションを回せる

MuJoCo の学習で最初につまずくのは「物理」ではなく「何が定数で何が状態か」の混乱です。
そこだけは今日中に潰します。

---

## 1-1. 環境構築

### uv プロジェクトを作る

このリポジトリは **uv** で管理する（`venv` + `pip` は使わない）。

```bash
cd ~/works/mujoco_study
uv init                # 未実施なら。pyproject.toml と .venv が作られる
uv add mujoco
```

**今はこれだけ入れる。** Gymnasium も PyTorch もまだ不要。
一度に全部入れると依存衝突が起きたとき原因を切り分けられなくなる。

実行はすべて `uv run` を通す。仮想環境を手で activate する必要はない。

```bash
uv run python sim/hello_mujoco.py
```

| やりたいこと | コマンド |
|---|---|
| パッケージを追加 | `uv add <name>` |
| スクリプトを実行 | `uv run python <path>` |
| 依存を同期（clone 直後など） | `uv sync` |

### `.gitignore` を作る

リポジトリ直下に以下の内容で `.gitignore` を作る。

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
uv run python -c "import mujoco; print(mujoco.__version__)"
```

バージョン番号（`3.x.x`）が出れば OK。

### ビューアの動作確認

```bash
uv run python -m mujoco.viewer
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

## 1-5b. Tips: 座標軸を表示する

### 方法1: ビューアの機能（XML 変更不要・推奨）

ビューア左側の **Rendering** パネルで:

| 項目 | 設定 | 表示されるもの |
|---|---|---|
| **Frame** | `World` | 原点の XYZ 軸 |
| **Frame** | `Body` | 全 body の座標系（親相対座標の理解に効く） |
| **Joint** | ON | **関節軸が矢印で表示される** |

色は世界共通の慣習で **赤=X / 緑=Y / 青=Z**（RGB = XYZ）。

Step 2 以降で最も重要なのは **Joint 表示**。`axis="0 1 0"` と書いた関節が意図した向きに
付いているかは、数字を睨むより見たほうが確実。回転軸を間違えたまま気づかず RL を回す事故は
初期に非常によく起きる。

### 方法2: XML に書き込む（`geom` ではなく `site` を使う）

```xml
<worldbody>
  <site name="axis_x" type="cylinder" fromto="0 0 0  0.3 0 0" size="0.006" rgba="1 0 0 1"/>
  <site name="axis_y" type="cylinder" fromto="0 0 0  0 0.3 0" size="0.006" rgba="0 1 0 1"/>
  <site name="axis_z" type="cylinder" fromto="0 0 0  0 0 0.3" size="0.006" rgba="0 0 1 1"/>
</worldbody>
```

**なぜ `site` なのか**:

| | `geom` | `site` |
|---|---|---|
| 衝突判定 | **する** | しない |
| 質量・慣性への寄与 | **する** | しない |
| 用途 | 実体のある物体 | マーカー、センサ取付点、目標位置 |

`geom` で軸を描くと**見た目のための飾りが物理に混入する**（見えない棒に衝突する、質量が増えて
周期が変わる）。`site` は純粋な目印なので物理は一切変わらない。

`site` はこの先も頻出する。`<sensor>` は site を参照するし、到達目標点や力の作用点も site。
**「物理には関わらないが位置を知りたい点」は全部 site**。

`size` は**半径**で単位はメートル。`0.006` = 半径 6mm。桁を間違えると画面を覆う巨大な円盤になる。

X 軸と Y 軸は床（`z=0` の plane）と同じ高さなので半分埋まって見える。気になるなら始点を
わずかに浮かせる（`fromto="0 0 0.001  0.3 0 0.001"`）。

**`group` 属性の注意**: `group="3"` のように 3 以上を付けると、**デフォルトでは描画されない**。
MuJoCo の初期状態は group 0-2 のみ表示で、3-5 は非表示。

```
sitegroup  [1, 1, 1, 0, 0, 0]   <- group 0,1,2 は表示 / 3,4,5 は非表示
geomgroup  [1, 1, 1, 0, 0, 0]
jointgroup [1, 1, 1, 0, 0, 0]
```

常に見せたいなら `group` を付けない（= group 0）。撮影時だけ消したいなど切り替えたい場合に
`group="3"` を使い、ビューアの Rendering パネルの **Site group** で 3 番にチェックを入れる。

`<body>` の中に site を置けば、その body に貼り付いて一緒に動く。振り子の body 内に置くと、
棒と一緒に回る座標系が見えて、親相対座標がどういうことか体感できる。

### MuJoCo の座標系の約束

- **Z が上**（重力が `0 0 -9.81` なのはこのため）
- **右手系**（X→Y への回転で Z が進む向き）
- 長さの単位は **メートル**

ROS や多くの CAD と同じ Z-up 右手系だが、Unity（Y-up 左手系）とは違う。
実機の CAD から MJCF を起こす Step 12 で取り違えるとモデルが横倒しになる。

---

## 1-6. 課題 3: 状態を記録してプロットする

RL の実験では「ログを取ってプロットする」を毎回やる。今のうちに癖にしておくこと。
**学習が失敗したとき、原因が分かるかどうかはログの質で決まる。**

```bash
uv add matplotlib
```

### ステップ1: ログを配列に貯める

ビューアと違い、ここでは**描画せずに一気に回して**後からまとめてプロットする。
シミュレーションは実時間に縛られないので、10 秒分の物理が一瞬で終わる。

```python
import matplotlib.pyplot as plt
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
data.qpos[0] = 0.1  # 初期角度 [rad]

n_steps = 5000  # 0.002 s x 5000 = 10 秒ぶん
log = np.zeros((n_steps, 3))  # 各行 [time, qpos, qvel]

for i in range(n_steps):
    mujoco.mj_step(model, data)
    log[i] = [data.time, data.qpos[0], data.qvel[0]]

t, q, v = log[:, 0], log[:, 1], log[:, 2]
```

> **なぜ `list.append` ではなく `np.zeros` で先に確保するのか**
> ステップ数が事前に分かっているなら確保しておくほうが速く、型も揃う。
> RL の学習ループでも「バッファを先に確保して埋める」書き方が標準。今から慣れておく。

### ステップ2: 時系列と位相平面を並べて描く

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 左: 時系列
axes[0].plot(t, q, label="angle [rad]")
axes[0].plot(t, v, label="angular velocity [rad/s]", alpha=0.7)
axes[0].set_xlabel("time [s]")
axes[0].set_title("time series")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右: 位相平面
axes[1].plot(q, v, lw=0.8)
axes[1].set_xlabel("angle [rad]")
axes[1].set_ylabel("angular velocity [rad/s]")
axes[1].set_title("phase plane")
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()
```

> **ラベルは英語で書くこと。** matplotlib のデフォルトフォントは日本語グリフを持たないので、
> 日本語を書くと豆腐（□□□）になる。日本語を出すにはフォント設定が要るが、
> 学習用のグラフは英語で十分。

### 見るべきもの

**位相平面（右のグラフ）が今日の主役。** 横軸に角度、縦軸に角速度を取ると、
振り子の状態が平面上の 1 点として表される。時間が進むとその点が軌跡を描く。

| 設定 | 位相平面の形 | 意味 |
|---|---|---|
| `damping` なし | **閉じた楕円**（同じ軌道を回り続ける） | エネルギーが保存されている |
| `damping="0.1"` | **内側に巻き込む渦** | エネルギーが散逸して原点に収束する |

`<joint>` に `damping="0.1"` を足して、渦に変わることを確認すること。
「エネルギーが失われる」という言葉が、目に見える形になる。

この位相平面は Step 4（エネルギー整形によるスイングアップ）で再登場する。
そのとき、**振り子を立てるとは位相平面上で特定の軌道に乗せることだ**という見方をする。
今のうちに図の読み方に慣れておくと、そこが一気に楽になる。

### ステップ3: 周期を実測する（完了条件）

目視で読み取ってもよいが、コードで測ったほうが正確で速い。
角度が**プラスからマイナスへ変わる瞬間**（下降ゼロ交差）を拾い、その間隔を平均する。

```python
idx = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]  # 下降ゼロ交差のインデックス
period = np.diff(t[idx]).mean()
print(f"measured period = {period:.4f} s")
```

これを理論値と比べる。1-4 のヒントにある剛体棒の物理振子の式を使うこと。
**ズレたら、その理由を考えるのがこの課題の本題。**

> ヒント: 理論式は「微小振動」の仮定の上に立っている。初期角度 `0.1 rad` は微小か？
> `data.qpos[0]` を `0.05` や `1.0` に変えて周期を測り直すと、何が起きるか。

### 発展: 条件を変えて重ねて描く

`damping` を変えて複数条件を比較したくなる。XML を毎回書き直すのではなく、
**XML をテンプレートにして値を差し込み、シミュレーションを関数にまとめる**とよい。

```python
XML_TEMPLATE = """..."""  # damping の値を {damping} にしておく


def run(damping: float, n_steps: int = 5000) -> np.ndarray:
    """1 条件ぶん回して [time, qpos, qvel] の配列を返す"""
    ...


for d in [0.0, 0.05, 0.2]:
    log = run(d)
    plt.plot(log[:, 1], log[:, 2], label=f"damping={d}")
```

この「条件を変えて回す関数」は、Step 11 のドメインランダマイゼーションで
**そのままの形で使う**。物理パラメータを引数に取って結果を返す構造は変わらない。

### Tips: 「見る実行」と「測る実行」を分ける

ビューア + `time.sleep` は**実時間**で再生する。5000 ステップ = 10 秒分の物理に現実の 10 秒かかる。
測定のたびに待つのは無駄なので、フラグで切り替えられるようにする。
ビューアと sleep を外せば同じ 10 秒分が **0.05 秒程度**で終わる（約 200 倍速）。

```python
import argparse, contextlib

parser = argparse.ArgumentParser()
parser.add_argument("--view", action="store_true", help="ビューアで実時間再生する")
args = parser.parse_args()

ctx = mujoco.viewer.launch_passive(model, data) if args.view else contextlib.nullcontext()

with ctx as viewer:
    for i in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        log[i] = [data.time, data.qpos[0], data.qvel[0]]

        if viewer is not None:
            viewer.sync()
            wait = model.opt.timestep - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)
```

```bash
uv run python sim/hello_mujoco.py           # 一瞬で終わる（測定用）
uv run python sim/hello_mujoco.py --view    # 実時間で見る
```

- `contextlib.nullcontext()` は「何もしない `with`」。`as` で受けると `None` が入るので、
  viewer の有無でループを二重に書かずに済む
- ループ条件を `while viewer.is_running()` から `for i in range(n_steps)` に変えている点に注意。
  ビューアなしの実行では `is_running()` が存在しない

**見るときも実時間である必要はない**:

```python
SPEED = 4.0                                            # 4 倍速
wait = model.opt.timestep / SPEED - (time.time() - step_start)

if i % 8 == 0:      # 物理 500Hz に対し画面は 60Hz 程度で十分
    viewer.sync()
```

**なぜ重要か**: この分離はそのまま RL のコード構造になる。

| 用途 | 描画 | 速度 |
|---|---|---|
| **学習**（Step 7〜） | なし | 実時間の数千倍。数時間で数百万ステップ |
| **評価・デバッグ** | あり | 実時間。人間が目で確認する |

SB3 でも学習時は `render_mode=None`、評価時だけ `render_mode="human"` で環境を作り直す。
**学習ループの中で描画したら、その時点で学習は成立しない。**

### 図を保存する

```python
fig.savefig("docs/img/step01_phase_plane.png", dpi=150, bbox_inches="tight")
```

`plt.show()` はウィンドウを閉じるまでプログラムが止まる。
保存だけしたい場合や SSH 越しで実行する場合は、import の前に次を入れる。

```python
import matplotlib

matplotlib.use("Agg")  # 画面を使わない描画バックエンド
```

---

### 「何も見えない」ときの点検順序

MuJoCo では今後も繰り返し起きる。手順として持っておくこと。

1. **group が 3 以上になっていないか**（デフォルト非表示）
2. **`rgba` の 4 番目（アルファ）が 0 になっていないか**（透明）
3. **`size` の桁は合っているか**（小さすぎて点 / 大きすぎて画面を覆い気づかない）
4. **他の物体の内部に埋まっていないか**（Rendering パネルの Transparent をオンにする）
5. **そもそもモデルに入っているか** — `print(model.nsite)` / `model.ngeom` で数を確認

5 は地味だが重要。XML の書き間違いでコンパイルエラーにならず要素が無視されることがある。
数が合っていれば「モデルには入っているが見えていない」と切り分けられる。

---

## 1-7. 完了条件チェックリスト

- [ ] `uv run python -c "import mujoco; print(mujoco.__version__)"` が通る
- [ ] `uv run python -m mujoco.viewer` でウィンドウが開き、マウス操作できる
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
