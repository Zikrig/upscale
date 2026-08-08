#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

only=""
voice="en-US-BrianNeural"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only="${2:?missing value for --only}"; shift 2 ;;
    --voice) voice="${2:?missing value for --voice}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$only" ]] || { echo "Use --only p1, p2, etc." >&2; exit 2; }

python3 app/srt_to_txt.py
text="output/tts_text/to_translate_${only}.txt"
out="output/dub_en/to_translate_${only}_en.wav"
[[ -f "$text" ]] || { echo "Missing $text" >&2; exit 1; }

python3 app/tts_text.py --text "$text" --out "$out" --voice "$voice"
