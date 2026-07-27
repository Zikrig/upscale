from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = str(ROOT / "input")
OUTPUT = str(ROOT / "output")
TRANSLATED = str(ROOT / "output" / "translated")
MODEL_DIR = str(ROOT / "models")
TMP_DIR = "/tmp"

WHISPER_MODEL = "large-v3"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Reference clip for voice cloning (seconds)
REF_SECONDS = 12.0
# Soft limits for time-stretch (beyond this audio quality degrades)
STRETCH_MIN = 0.65
STRETCH_MAX = 1.8
