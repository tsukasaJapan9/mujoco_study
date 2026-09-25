import numpy as np

CLIFF = [
  "......",
  "......",
  "S!!!!G",
]
DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]
N_ROW, N_COL = 3, 6
START, GOAL = (2, 0), (2, 5)
STATES = [(r, c) for r in range(N_ROW) for c in range(N_COL)]


def step(s, a):
  dr, dc = DELTA[a]
  r, c = s[0] + dr, s[1] + dc
  if not (0 <= r < N_ROW and 0 <= c < N_COL):
    r, c = s  # 外に出ようとしたら留まる
  nxt = (r, c)
  if CLIFF[nxt[0]][nxt[1]] == "!":
    return START, -100.0, False  # 崖: 大損してスタートに逆戻り
  return nxt, (0.0 if nxt == GOAL else -1.0), nxt == GOAL


def make_Q():
  return {s: [0.0] * 4 for s in STATES}


def choose(Q, s, rng, eps=0.1):
  if rng.random() < eps:
    return int(rng.integers(4))
  return int(np.argmax(Q[s]))


def learn(kind, n_ep=3000, alpha=0.1, eps=0.1, gamma=1.0, seed=0, cap=200):
  rng = np.random.default_rng(seed)
  Q = make_Q()
  returns = []
  for ep in range(n_ep):
    s = START
    a = choose(Q, s, rng, eps)  # 最初の行動を先に決めておく
    total = 0.0
    for t in range(cap):
      s_next, r, done = step(s, a)
      total += r
      a_next = choose(Q, s_next, rng, eps)  # 次の行動も先に決める

      if kind == "q":
        target = r + (0.0 if done else gamma * max(Q[s_next]))
      else:
        target = r + (0.0 if done else gamma * Q[s_next][a_next])

      Q[s][a] += alpha * (target - Q[s][a])
      s, a = s_next, a_next
      if done:
        break
    returns.append(total)
  return Q, returns


def greedy_path(Q, cap=30):
  s, path = START, [START]
  for t in range(cap):
    s, r, done = step(s, int(np.argmax(Q[s])))
    path.append(s)
    if done:
      return path
  return path


for kind, name in (("q", "Q 学習  (off-policy)"), ("sarsa", "SARSA  (on-policy)")):
  Q, returns = learn(kind)
  path = greedy_path(Q)
  print(f"{name}")
  print(f"  経路         : {path}")
  print(f"  歩数         : {len(path) - 1}")
  print(f"  学習中の平均収益: {np.mean(returns[-500:]):.2f}")
