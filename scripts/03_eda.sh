#!/usr/bin/env bash
set -euo pipefail #
cd "$(dirname "$0")/.." #
UV="${UV:-uv}" #
"$UV" run python -m src.visualization.integrity #
echo "[EDA] integrity done" #
"$UV" run python -m src.visualization.metadata #
echo "[EDA] metadata done" #
"$UV" run python -m src.visualization.labels #
echo "[EDA] labels done" #
"$UV" run python -m src.visualization.galleries #
echo "[EDA] galleries done" #
"$UV" run python -m src.visualization.duplicates #
echo "[EDA] duplicates done" #
"$UV" run python -m src.visualization.consistency #
echo "[EDA] consistency done" #
