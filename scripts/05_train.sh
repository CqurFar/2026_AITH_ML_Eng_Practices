#!/usr/bin/env bash
set -euo pipefail #
cd "$(dirname "$0")/.." #
UV="${UV:-uv}" #
"$UV" run --locked python -m src.models.model_building #
