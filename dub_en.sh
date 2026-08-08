#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export PYTHONUNBUFFERED=1

only=""
voice="en-US-BrianNeural"
subs_dir="output/subs_tts_en"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only="${2:?missing value for --only}"; shift 2 ;;
    --voice) voice="${2:?missing value for --voice}"; shift 2 ;;
    --subs-dir) subs_dir="${2:?missing value for --subs-dir}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$only" ]]; then
  echo "Use --only p1 for the first fragment." >&2
  exit 2
fi

srt="${subs_dir}/to_translate_${only}.srt"
out="output/dub_en/to_translate_${only}_en.wav"
[[ -f "$srt" ]] || { echo "Missing $srt" >&2; exit 1; }

python3 app/dub_from_srt.py --srt "$srt" --out "$out" --voice "$voice"
