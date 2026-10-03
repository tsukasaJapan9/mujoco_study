#!/bin/sh
# 結果の再生成（4 コアで約 10 分）
set -e
uv run python -m furuta.sweep -n 16 --tip 0.004 --preload 0 --out results/sweep_no_preload.csv > results/sweep_no_preload.txt
uv run python -m furuta.sweep -n 16 --tip 0 0.002 0.004 0.008 --preload 0.01 --out results/sweep_preload.csv > results/sweep_preload.txt
