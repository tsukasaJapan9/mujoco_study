import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from torch import nn

from envs.pendulum_env import PendulumEnv


# --------------------------------------------------------------------------------
# 1. ネットワーク定義クラス（Actor-Critic）
# --------------------------------------------------------------------------------
class ActorCritic(nn.Module):
  """Actor（方策：行動を決める）と Critic（価値関数：状態の良さを評価する）を

  一つにまとめたニューラルネットワーククラス。
  """

  def __init__(self, obs_dim, act_dim, hidden=64):
    """クラスの初期化関数

    obs_dim : 観測空間の次元数（Pendulumの場合は3：cos(θ), sin(θ), angular velocity）
    act_dim : 行動空間の次元数（Pendulumの場合は1：トルク）
    hidden  : 隠れ層のノード数（デフォルト64）
    """
    super().__init__()

    # ActorとCriticで共有する特徴抽出用ネットワーク
    self.shared = nn.Sequential(
      nn.Linear(obs_dim, hidden),  # 入力層 -> 隠れ層1
      nn.Tanh(),  # 活性化関数 Tanh
      nn.Linear(hidden, hidden),  # 隠れ層1 -> 隠れ層2
      nn.Tanh(),  # 活性化関数 Tanh
    )

    # Actorの出力層：行動の平均値（mu）を出力
    self.mu = nn.Linear(hidden, act_dim)

    # Criticの出力層：状態価値（v）を1次元のスカラーで出力
    self.v = nn.Linear(hidden, 1)

    # 行動の標準偏差の対数（log_std）：観測に依存しない学習可能なパラメータとして定義
    # 初期値は0（std = exp(0) = 1.0 に対応）
    self.log_std = nn.Parameter(torch.zeros(act_dim))

  def forward(self, obs):
    """順伝播計算（ネットワークに観測を入力して予測結果を得る）

    obs : 状態の観測データ (Tensor)
    ---
    返り値:
      mu      : 行動分布の平均
      v       : 状態価値（最後の次元を圧縮して1次元テンソルにする）
      log_std : 標準偏差の対数
    """
    h = self.shared(obs)
    # v(h) の出力形状は [batch_size, 1] なので squeeze(-1) で [batch_size] に整形
    return self.mu(h), self.v(h).squeeze(-1), self.log_std

  def dist(self, obs):
    """観測から正規分布（確率分布）と状態価値を計算して返す

    obs : 状態の観測データ
    ---
    返り値:
      dist : 行動を出力するための正規分布オブジェクト
      v    : 状態価値
    """
    mu, v, log_std = self(obs)
    std = log_std.exp()  # log_std を指数変換して標準偏差(std > 0)にする
    dist = torch.distributions.Normal(mu, std)  # 平均 mu, 標準偏差 std の正規分布を作成
    return dist, v

  def act(self, obs):
    """環境と対話（データ収集）する際に使用するメソッド

    obs : 状態の観測データ
    ---
    返り値:
      action   : サンプリングされた行動
      log_prob : その行動が選ばれる対数確率密度
      v        : 予測された状態価値
    """
    dist, v = self.dist(obs)
    action = dist.sample()  # 分布から行動を1つサンプリング（確率的行動決定）
    log_prob = dist.log_prob(action).sum(
      -1
    )  # 各次元の対数確率の和を計算（多次元行動用）
    return action, log_prob, v


# --------------------------------------------------------------------------------
# 2. GAE (Generalized Advantage Estimation) の計算関数
# --------------------------------------------------------------------------------
def gae(rewards, values, next_values, ends, gamma=0.99, lam=0.95):
  """Generalized Advantage Estimation (GAE) を計算する関数。

  「予想していた価値よりどれだけ良い行動だったか」を表すアドバンテージ(adv)を算出。

  rewards     : 各ステップで得た報酬の配列 (長さ T)
  values      : 各ステップの状態価値 V(s_t) の配列 (長さ T)
  next_values : 各ステップの「次の状態の価値」 V(s_{t+1}) の配列 (長さ T)
  ends        : 終端または打ち切りフラグの配列（終了なら 1.0, 継続なら 0.0）
  gamma       : 割引率（将来の報酬をどれくらい割り引くか）
  lam (lambda): GAEのバイアスと分散をトレードオフするハイパーパラメータ
  """
  T = len(rewards)
  adv = np.zeros(T, dtype=np.float32)  # アドバンテージ格納用配列を初期化
  running = 0.0  # 後ろから累積計算していくための変数

  # 後ろのステップ（T-1）から過去（0）へ逆順にループ処理
  for t in reversed(range(T)):
    # TD誤差 delta = r_t + gamma * V(s_{t+1}) - V(s_t)
    delta = rewards[t] + gamma * next_values[t] - values[t]

    # エピソードが終了（ends[t] == 1.0）した場合、次ステップからの影響 running をリセット(0にする)
    running = delta + gamma * lam * (1.0 - ends[t]) * running

    adv[t] = running

  return adv


# --------------------------------------------------------------------------------
# 3. PPO Policy Loss (クリップ付き目的関数) 計算関数
# --------------------------------------------------------------------------------
def policy_loss(log_prob_new, log_prob_old, adv, clip=0.2):
  """PPOの特徴である「クリップ付き方策損失」を計算する関数

  log_prob_new : 新しい方策での行動の対数確率
  log_prob_old : データ収集時（古い方策）での行動の対数確率
  adv          : アドバンテージ
  clip         : クリッピングのパラメータ（通常 0.1 ~ 0.2）
  """
  # 確率比 ratio = exp(log_prob_new - log_prob_old) = prob_new / prob_old
  ratio = (log_prob_new - log_prob_old).exp()

  # 1. クリップなしの目的関数項
  a1 = ratio * adv

  # 2. ratioを [1-clip, 1+clip] の範囲に制限したクリップ付き目的関数項
  a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv

  # a1 と a2 の小さい方を取り（PPOの悲観的評価）、全体平均して符号を反転（勾配降下法で最大化するため）
  return -torch.min(a1, a2).mean()


# --------------------------------------------------------------------------------
# 4. データ収集（Rollout）関数
# --------------------------------------------------------------------------------
def collect(env, net, obs, n_steps, obs_dim, act_dim, lo, hi):
  """環境内でエージェントを動かし、学習に必要なn_steps分のデータを収集する関数

  env     : Gym環境インスタンス
  net     : ActorCriticモデル
  obs     : 現在の観測
  n_steps : 収集するステップ数（例: 8192）
  obs_dim : 観測次元
  act_dim : 行動次元
  lo, hi  : 環境の行動空間の最小値・最大値（Pendulumは [-2.0, 2.0]）
  """
  # バッファ（データ保管場所）の確保
  O = np.zeros((n_steps, obs_dim), dtype=np.float32)  # 観測 (Observation)
  A = np.zeros((n_steps, act_dim), dtype=np.float32)  # 行動 (Action)
  LP = np.zeros(n_steps, dtype=np.float32)  # サンプリング時の対数確率 (Log Prob)
  R = np.zeros(n_steps, dtype=np.float32)  # 報酬 (Reward)
  V = np.zeros(n_steps, dtype=np.float32)  # 予測状態価値 (Value)
  NV = np.zeros(n_steps, dtype=np.float32)  # 次の状態の予測価値 (Next Value)
  E = np.zeros(n_steps, dtype=np.float32)  # 終了フラグ (End)

  ep_rets, ep_ret = [], 0.0  # 1エピソードの合計報酬履歴と、現在の累積報酬

  for t in range(n_steps):
    # 観測をPyTorchテンソルに変換してバッチ次元を追加 [1, obs_dim]
    ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    # 勾配計算を行わずにネットワークから行動を取得
    with torch.no_grad():
      a, lp, v = net.act(ot)

    # 現在のデータをバッファに保存
    O[t], A[t], LP[t], V[t] = obs, a.numpy()[0], lp.item(), v.item()

    # ネットワークが出力した行動(-inf~inf)を[-1.0, 1.0]にクリップ
    raw = np.clip(a.numpy()[0], -1.0, 1.0)

    # [-1, 1] の範囲の行動を、本来の環境の行動範囲 [lo, hi] (例: [-2, 2]) にスケーリングして環境に入力
    nxt, r, term, trunc, _ = env.step(lo + (raw + 1.0) * 0.5 * (hi - lo))

    R[t] = r  # 報酬を保存
    ep_ret += r  # エピソード累積報酬を加算

    if term:
      # タスク失敗/成功による終了(Terminal State)の場合、次の状態の価値は厳密に 0.0
      NV[t] = 0.0
    else:
      # 継続またはステップ制限による中断(Truncated)の場合、次の状態の価値をネットワークで予測
      with torch.no_grad():
        _, nv, _ = net(torch.as_tensor(nxt, dtype=torch.float32).unsqueeze(0))
      NV[t] = nv.item()

    # 終端（term）またはステップ数上限（trunc）のどちらかであれば 1.0 を記録
    E[t] = float(term or trunc)

    # エピソードが終了した場合の初期化処理
    if term or trunc:
      ep_rets.append(ep_ret)  # 1エピソード分の合計報酬を記録
      ep_ret = 0.0  # 累積報酬をリセット
      obs, _ = env.reset()  # 環境をリセットして次のエピソードを開始
    else:
      obs = nxt  # 次のステップへ観測を更新

  return obs, (O, A, LP, R, V, NV, E), ep_rets


# --------------------------------------------------------------------------------
# 5. メイン学習ループ関数
# --------------------------------------------------------------------------------
def train(
  env_fn,  # 環境を作成する関数
  obs_dim,  # 観測の次元数
  act_dim,  # 行動の次元数
  total_steps=1_000_000,  # 合計学習ステップ数
  n_steps=8192,  # 1回の更新（データ収集）で集めるステップ数
  epochs=10,  # 1回のデータ収集あたり何周学習するか
  batch_size=64,  # ミニバッチサイズ
  lr=3e-4,  # 学習率
  gamma=0.99,  # 割引率
  lam=0.95,  # GAEのλ
  clip=0.2,  # PPOのクリップ幅
  vf_coef=0.5,  # Value Lossの係数
  ent_coef=0.0,  # エントロピー損失の係数（探索を促す）
  max_grad=0.5,  # 勾配クリッピングの最大ノルム
  seed=0,  # 乱数シード
):
  # 乱数シードの固定（再現性の確保）
  torch.manual_seed(seed)
  np.random.seed(seed)

  env = env_fn()  # 環境の生成
  lo, hi = env.action_space.low, env.action_space.high  # 行動範囲を取得
  net = ActorCritic(obs_dim, act_dim)  # モデル構築
  opt = torch.optim.Adam(net.parameters(), lr=lr)  # 最適化手法（Adam）
  obs, _ = env.reset(seed=seed)  # 初期状態を取得

  # 総ステップ数をn_stepsで割った回数分、イテレーション（更新）を実行
  for update in range(total_steps // n_steps):
    # --- Step 1: データの収集 ---
    obs, (O, A, LP, R, V, NV, E), rets = collect(
      env, net, obs, n_steps, obs_dim, act_dim, lo, hi
    )

    # --- Step 2: アドバンテージとターゲット価値（収益）の計算 ---
    adv = gae(R, V, NV, E, gamma, lam)  # GAEを計算
    ret = adv + V  # ターゲット価値 Target Value = Advantage + V(s)

    # 計算用テンソルを作成
    Ot, At = torch.as_tensor(O), torch.as_tensor(A)
    LPt, RETt = torch.as_tensor(LP), torch.as_tensor(ret)
    ADVt = torch.as_tensor(adv)

    # アドバンテージの標準化（平均0, 標準偏差1にして学習を安定化させる標準的なテクニック）
    ADVt = (ADVt - ADVt.mean()) / (ADVt.std() + 1e-8)

    # --- Step 3: PPOネットワークの最適化（学習） ---
    idx = np.arange(n_steps)  # 0 から n_steps-1 までのインデックス配列を作成
    for _ in range(epochs):  # 同じ集めたデータを epochs 回使い回して学習
      np.random.shuffle(idx)  # インデックスをシャッフル

      # ミニバッチごとに分割して処理
      for s in range(0, n_steps, batch_size):
        b = idx[s : s + batch_size]  # ミニバッチのインデックスを取り出す

        # 現在のネットワークで行動分布と価値を取得
        d, v = net.dist(Ot[b])
        lp = d.log_prob(At[b]).sum(-1)  # 新しい対数確率を計算

        # --- 各種 Loss の計算 ---
        # 1. Policy Loss (クリップ付きSurrogate Loss)
        ratio = (lp - LPt[b]).exp()
        a1 = ratio * ADVt[b]
        a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADVt[b]
        pg = -torch.min(a1, a2).mean()

        # 2. Value Loss (価値関数の二乗誤差損失)
        vl = ((v - RETt[b]) ** 2).mean()

        # 3. Entropy (探索を促進するための確率分布の広がり具合)
        ent = d.entropy().sum(-1).mean()

        # 総合損失 Loss = PolicyLoss + 0.5 * ValueLoss - 0.0 * Entropy
        loss = pg + vf_coef * vl - ent_coef * ent

        # ネットワークのパラメータ更新
        opt.zero_grad()  # 勾配のリセット
        loss.backward()  # 逆伝播（勾配計算）
        nn.utils.clip_grad_norm_(
          net.parameters(), max_grad
        )  # 勾配爆発を防ぐクリッピング
        opt.step()  # 重みの更新

    # ログ表示（直近20エピソードの平均報酬を表示）
    print(
      f"  step={(update + 1) * n_steps:>8}  ep_rew={np.mean(rets[-20:]) if rets else 0:8.2f}",
      flush=True,
    )

  return net


# --------------------------------------------------------------------------------
# 6. エントリーポイント（実行部）
# --------------------------------------------------------------------------------
def make_env():
  """環境生成ヘルパー関数：PendulumEnvに500ステップの制限時間を付与"""
  return TimeLimit(PendulumEnv(), max_episode_steps=500)


if __name__ == "__main__":
  # 1,000,000 ステップ分の学習を開始
  net = train(make_env, obs_dim=3, act_dim=1, total_steps=1_000_000, n_steps=8192)

  # 学習したモデルの重みを保存
  torch.save(net.state_dict(), "algos/ppo_min_pendulum.pt")
