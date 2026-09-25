GAMMA = 0.9


def run(T, mode, n_ep=20000, alpha=0.05):
  # 状態は 1 つ。何をしても報酬 +1。タスクとしての終わりは無い。
  # T ステップで必ず「打ち切り」になる。
  Q = 0.0
  for ep in range(n_ep):
    for t in range(1, T + 1):
      r = 1.0
      truncated = t == T

      if truncated and mode == "wrong":
        target = r  # 打ち切りを終端扱い（誤り）
      else:
        target = r + GAMMA * Q  # ← ここだけ考える

      Q += alpha * (target - Q)
  return Q


true_value = 1 / (1 - GAMMA)
print(f"理論値 = 1/(1-gamma) = {true_value:.4f}")
for T in (5, 10, 20, 50, 100, 500):
  a, b = run(T, "correct"), run(T, "wrong")
  print(
    f"T={T:<4} 正しい={a:.4f}  誤り={b:.4f}  ずれ={abs(b - true_value) / true_value * 100:.1f}%"
  )
