import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import MODEL_DIR, OUTPUT, TMP_DIR

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
    enhance_in = Path(TMP_DIR) / "enhance_in"
    enhance_out = Path(TMP_DIR) / "enhance_out"

    if enhance_in.exists():
        shutil.rmtree(enhance_in)
    if enhance_out.exists():
        shutil.rmtree(enhance_out)

    enhance_in.mkdir(parents=True)
    enhance_out.mkdir(parents=True)

    # flat name without spaces — resemble-enhance walks directories
    src = enhance_in / "vocals.wav"
    shutil.copy2(vocals, src)

    env = {**os.environ, "HF_HOME": MODEL_DIR, "TORCH_HOME": MODEL_DIR}
    cmd = shutil.which("resemble_enhance") or shutil.which("resemble-enhance")
    if not cmd:
        raise FileNotFoundError("resemble_enhance CLI not found in PATH")

    subprocess.run([cmd, str(enhance_in), str(enhance_out)], check=True, env=env)

    enhanced_files = list(enhance_out.rglob("*.wav"))
    if not enhanced_files:
        raise FileNotFoundError(f"No enhanced audio in {enhance_out}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(enhanced_files[0]), str(output))


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
