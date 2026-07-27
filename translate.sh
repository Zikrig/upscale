#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export TORCH_HOME="$(pwd)/models"
export HF_HOME="$(pwd)/models"
export XDG_CACHE_HOME="$(pwd)/models"
export COQUI_TOS_AGREED=1

# coqui-tts needs transformers 4.x (5.x removed isin_mps_friendly)
pip install -q faster-whisper "coqui-tts" "transformers>=4.43,<5"

cd app
python3 translate.py
