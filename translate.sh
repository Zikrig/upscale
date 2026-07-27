#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export TORCH_HOME="$(pwd)/models"
export HF_HOME="$(pwd)/models"
export XDG_CACHE_HOME="$(pwd)/models"
export COQUI_TOS_AGREED=1

# free leftover temp from previous runs
rm -rf /tmp/dub_* 2>/dev/null || true

pip install -q faster-whisper "coqui-tts" "transformers>=4.43,<5"

cd app
# Examples:
#   python3 translate.py --force
#   python3 translate.py --only bitards --force
python3 translate.py "$@"
