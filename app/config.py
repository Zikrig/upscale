from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = str(ROOT / "input")
OUTPUT = str(ROOT / "output")
TRANSLATED = str(ROOT / "output" / "translated")
MODEL_DIR = str(ROOT / "models")
TMP_DIR = "/tmp"

# Whisper: large-v3 is best quality; use medium if VRAM is tight
WHISPER_MODEL = "large-v3"
TTS_VOICE = "en-US-AndrewNeural"
