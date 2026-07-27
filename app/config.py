from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = str(ROOT / "input")
OUTPUT = str(ROOT / "output")
TRANSLATED = str(ROOT / "output" / "translated")
MODEL_DIR = str(ROOT / "models")
TMP_DIR = "/tmp"

WHISPER_MODEL = "large-v3"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

REF_SECONDS = 15.0

# Merge Whisper crumbs into natural phrases
MERGE_MAX_GAP = 0.55
MERGE_TARGET_SEC = 6.0
MERGE_MAX_SEC = 14.0
MERGE_MAX_CHARS = 320
