from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = str(ROOT / "input")
OUTPUT = str(ROOT / "output")
TRANSLATED = str(ROOT / "output" / "translated")
VIDEO_INPUT = str(ROOT / "video_input")
SUBS_DIR = str(ROOT / "output" / "subs")
SUBS_BURNED_DIR = str(ROOT / "output" / "subs_burned")
MODEL_DIR = str(ROOT / "models")
TMP_DIR = "/tmp"

WHISPER_MODEL = "large-v3"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

REF_SECONDS = 15.0

# Audio translate (legacy)
MERGE_MAX_GAP = 0.55
MERGE_TARGET_SEC = 6.0
MERGE_MAX_SEC = 14.0
MERGE_MAX_CHARS = 320

# Video subtitles: short, roughly equal cues
SUB_TARGET_SEC = 2.3
SUB_MAX_SEC = 3.5
SUB_MIN_SEC = 0.7
SUB_MAX_CHARS = 42
