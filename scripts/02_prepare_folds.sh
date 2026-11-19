#!/usr/bin/env bash
set -euo pipefail #
cd "$(dirname "$0")/.." #
UV="${UV:-uv}" #
"$UV" run python -m src.data.prepare_data #
