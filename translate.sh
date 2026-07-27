#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export TORCH_HOME="$(pwd)/models"
export HF_HOME="$(pwd)/models"
export XDG_CACHE_HOME="$(pwd)/models"
export COQUI_TOS_AGREED=1

python -c "import faster_whisper, TTS" 2>/dev/null || \
  pip install faster-whisper "coqui-tts"

cd app
python3 translate.py
