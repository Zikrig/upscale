#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export TORCH_HOME="$(pwd)/models"
export HF_HOME="$(pwd)/models"
export XDG_CACHE_HOME="$(pwd)/models"

mkdir -p video_input output/subs output/subs_burned output/subs_long output/subs_long_burned models
rm -rf /tmp/subs_* 2>/dev/null || true

df -h / | tail -1

python -c "import faster_whisper" 2>/dev/null || pip install -q faster-whisper

cd app
python3 subtitles.py "$@"
