import os
import shutil
import subprocess
from pathlib import Path

from config import INPUT, MODEL_DIR, OUTPUT, TMP_DIR


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

    shutil.copy2(vocals, enhance_in / vocals.name)

    subprocess.run(
        [
            "resemble-enhance",
            str(enhance_in),
            str(enhance_out),
        ],
        check=True,
        env={**os.environ, "HF_HOME": MODEL_DIR, "TORCH_HOME": MODEL_DIR},
    )

    enhanced_files = list(enhance_out.glob("*.wav"))
    if not enhanced_files:
        raise FileNotFoundError(f"No enhanced audio in {enhance_out}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(enhanced_files[0]), str(output))


def process(filename: str) -> None:
    base = Path(filename).stem
    work_dir = Path(TMP_DIR) / base
    wav = work_dir / f"{base}.wav"

    work_dir.mkdir(parents=True, exist_ok=True)

    extract_audio(filename, str(wav))
    vocals = separate_music(str(wav), str(work_dir))
    enhance(vocals, Path(OUTPUT) / f"{base}_clean.wav")

    shutil.rmtree(work_dir, ignore_errors=True)
