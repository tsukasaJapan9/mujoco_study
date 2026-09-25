import numpy as np

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
print("START =", START, " GOAL =", GOAL)


def G(rewards, gamma):
  g = 0.0
  for r in reversed(rewards):  # 後ろから前へ
    g = r + gamma * g
  return g


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


def value_of(s0, gamma=0.9, n_ep=20000, seed=2):
  rng = np.random.default_rng(seed)
  if s0 == GOAL:
    return 0.0  # ゴールは着いた時点で終わりなので 0
  totals = []
  for ep in range(n_ep):
    state, rewards = s0, []  # ← ここだけ考える
    for t in range(500):
      state, reward, done = step(state, rng.integers(4))
      rewards.append(reward)
      if done:
        break
    totals.append(G(rewards, gamma))
  return np.mean(totals)


def backup(s, V, gamma=0.9):
  # s の新しい値を、4 方向すべて試して計算する
  candidates = []
  for a in range(4):
    next_state, reward, done = step(s, a)
    candidate = reward + (0.0 if done else gamma * V[next_state])
    candidates.append(candidate)  # ← ここだけ考える
  return max(candidates)


V = {(r, c): 0.0 for r in range(N_ROW) for c in range(N_COL) if GRID[r][c] != "#"}
# print(backup((0, 2), V))  # -> 1.0   ゴールに入れるので報酬がもらえる
# print(backup((3, 0), V))  # -> 0.0   周りが全部 0 なのでまだ何も分からない


def value_iteration(gamma=0.9, tol=1e-10):
  V = {(r, c): 0.0 for r in range(N_ROW) for c in range(N_COL) if GRID[r][c] != "#"}
  for sweep in range(1, 1000):
    delta = 0.0
    for s in V:
      if s == GOAL:
        continue  # ゴールの価値は 0 のまま動かさない
      new = backup(s, V, gamma)
      delta = max(delta, abs(new - V[s]))
      V[s] = new  # その場で書き換える（次のマスにすぐ伝わる）
    # print(f"{sweep} スイープ後  V(START)={V[START]:.5f}  最大変化={delta:.2e}")
    if delta < tol:
      break
  return V


def policy_evaluation(gamma=0.9, tol=1e-12):
  V = {(r, c): 0.0 for r in range(N_ROW) for c in range(N_COL) if GRID[r][c] != "#"}
  for sweep in range(1, 100000):
    delta = 0.0
    for s in V:
      if s == GOAL:
        continue
      total = 0.0
      for a in range(4):
        nxt, reward, done = step(s, a)
        total += 0.25 * (reward + (0.0 if done else gamma * V[nxt]))
      delta = max(delta, abs(total - V[s]))
      V[s] = total
    if delta < tol:
      break
  print(f"{sweep} スイープで収束")
  return V


V_pi = policy_evaluation()
print(f"正確な値       V(START) = {V_pi[START]:.6f}")
print(f"モンテカルロ推定 V(START) = {value_of(START):.6f}")
print(V_pi)


# V_star = value_iteration()


# print(value_of(START))  # -> 0.0523  (理論値 0.0532)
# print(value_of((0, 2)))  # -> 0.4993  ゴールの隣（理論値 0.5000）

# V = {}
# for r in range(N_ROW):
#   for c in range(N_COL):
#     if GRID[r][c] != "#":
#       V[(r, c)] = value_of((r, c))

# for r in range(N_ROW):
#   cells = []
#   for c in range(N_COL):
#     cells.append("  ##  " if GRID[r][c] == "#" else f"{V[(r, c)]:6.3f}")
#   print(" ".join(cells))


# print(step((3, 0), 0))  # 上に動ける   -> ((2, 0), 0.0, False)
# print(step((3, 0), 3))  # 左は外なので留まる -> ((3, 0), 0.0, False)


# for penalty in (0.0, 0.01, 0.05):
#   rng = np.random.default_rng(1)
#   totals = []
#   for ep in range(20000):
#     state, total = START, 0.0
#     for t in range(500):
#       state, reward, done = step(state, rng.integers(4), penalty)
#       total += reward
#       if done:
#         break
#     totals.append(total)
#   print(f"penalty={penalty}  ランダム方策の平均収益 = {np.mean(totals):+.4f}")


# print(G([0, 0, 0, 0, 0, 1.0], 1.0))  # -> 1.0
# print(G([0, 0, 0, 0, 0, 1.0], 0.9))  # -> 0.59049  (= 0.9 の 5 乗)


# OPT = [0, 0, 0, 0, 0, 1.0]  # 最短 6 歩。最後だけ +1

# for gamma in (0.9, 0.95, 0.99, 0.999, 1.0):
#   rng = np.random.default_rng(1)
#   vals = []
#   for ep in range(20000):
#     state, rewards = START, []
#     for t in range(500):
#       state, reward, done = step(state, rng.integers(4))
#       rewards.append(reward)
#       if done:
#         break
#     vals.append(G(rewards, gamma))
#   print(f"gamma={gamma:<6} 最短={G(OPT, gamma):.5f}  ランダム={np.mean(vals):.5f}")
