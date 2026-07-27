import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import OUTPUT, TMP_DIR
from resemble_runner import run_enhance

PYTHON = sys.executable


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "track"


def extract_audio(mp3: str, wav: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            mp3,
            "-ar",
            "48000",
            "-ac",
            "2",
            wav,
        ],
        check=True,
    )


def separate_music(audio: str, work_dir: str) -> Path:
    separated_dir = Path(work_dir) / "separated"
    subprocess.run(
        [
            PYTHON,
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "-d",
            "cuda",
            "-o",
            str(separated_dir),
            audio,
        ],
        check=True,
    )

    base = Path(audio).stem
    vocals = separated_dir / "htdemucs" / base / "vocals.wav"
    if not vocals.exists():
        raise FileNotFoundError(f"Vocals not found: {vocals}")

    return vocals


def enhance(vocals: Path, output: Path) -> None:
    run_enhance(vocals, output)


def process(filename: str) -> None:
    original = Path(filename)
    base = _safe_name(original.stem)
    work_dir = Path(TMP_DIR) / base
    wav = work_dir / f"{base}.wav"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    extract_audio(str(original), str(wav))
    vocals = separate_music(str(wav), str(work_dir))
    enhance(vocals, Path(OUTPUT) / f"{base}_clean.wav")

    shutil.rmtree(work_dir, ignore_errors=True)
