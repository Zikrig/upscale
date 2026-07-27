#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export TORCH_HOME="$(pwd)/models"
export HF_HOME="$(pwd)/models"
export XDG_CACHE_HOME="$(pwd)/models"

cd app
python3 main.py
