#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export TORCH_HOME="$(pwd)/models"
export HF_HOME="$(pwd)/models"
export XDG_CACHE_HOME="$(pwd)/models"

# ensure translate deps
python -c "import faster_whisper, edge_tts" 2>/dev/null || \
  pip install faster-whisper edge-tts

cd app
python3 translate.py
