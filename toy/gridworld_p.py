GRID = [
  "...G",
  ".#..",
  "....",
  "S.#.",
]
ACTIONS = ["up", "right", "down", "left"]
DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 行,列 の増分。上は行が減る

N_ROW, N_COL = len(GRID), len(GRID[0])


def find(ch):
  for r, row in enumerate(GRID):
    for c, x in enumerate(row):
      if x == ch:
        return (r, c)


START, GOAL = find("S"), find("G")


def step(state, action, penalty=0.0):
  dr, dc = DELTA[action]
  r, c = state[0] + dr, state[1] + dc

  if (
    r < 0 or N_ROW <= r or c < 0 or N_COL <= c or GRID[r][c] == "#"
  ):  # ← ここだけ考える
    r, c = state  # 動かずにその場に留まる

  next_state = (r, c)
  reward = (1.0 if next_state == GOAL else 0.0) - penalty
  done = next_state == GOAL
  return next_state, reward, done


# GRID, DELTA, N_ROW, N_COL, START, GOAL, step() をここにコピーしてくる


def make_Q():
  # 各マスに 4 つの値（上右下左）。最初は全部 0
  return {
    (r, c): [0.0, 0.0, 0.0, 0.0]
    for r in range(N_ROW)
    for c in range(N_COL)
    if GRID[r][c] != "#"
  }


Q = make_Q()


def update(Q, s, a, r, s_next, done, alpha=0.1, gamma=0.9):
  target = r + (0.0 if done else gamma * max(Q[s_next]))
  Q[s][a] = Q[s][a] + alpha * (target - Q[s][a])


Q = make_Q()
# update(Q, (0, 2), 1, 1.0, GOAL, True)  # (0,2) から右へ。ゴールして報酬 1.0
# print(Q[(0, 2)])  # -> [0.0, 0.1, 0.0, 0.0]

import numpy as np


def choose(Q, s, rng, eps=0.1):
  if rng.random() < eps:
    return int(rng.integers(4))  # 探索: でたらめ
  return int(np.argmax(Q[s]))  # 活用: 表で最大のもの


def q_learning(n_ep=500, alpha=0.1, eps=0.1, gamma=0.9, seed=0, cap=500):
  rng = np.random.default_rng(seed)
  Q = make_Q()
  lengths = []
  for ep in range(n_ep):
    s = START
    for t in range(1, cap + 1):
      a = choose(Q, s, rng, eps)
      s_next, r, done = step(s, a)
      update(Q, s, a, r, s_next, done, alpha, gamma)
      s = s_next  # 次のマスへ移る
      if done:
        break
    lengths.append(t)
  return Q, lengths


Q, lengths = q_learning()
print("最初の 10 エピソードの歩数:", lengths[:10])
print("最後の 10 エピソードの歩数:", lengths[-10:])


def run_greedy(Q, cap=100):
  s, path = START, [START]
  for t in range(1, cap + 1):
    s, r, done = step(s, int(np.argmax(Q[s])))  # eps なし
    path.append(s)
    if done:
      return t, path
  return None, path


steps, path = run_greedy(Q)
print(f"{steps} 歩でゴール")
print(path)
