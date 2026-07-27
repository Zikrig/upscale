"""Translate cleaned vocal WAVs to English speech.

Pipeline:
  1. Whisper (task=translate) → English transcript
  2. edge-tts → English WAV
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from pathlib import Path

from config import MODEL_DIR, OUTPUT, TMP_DIR, TTS_VOICE, TRANSLATED, WHISPER_MODEL


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "track"


def _format_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_to_english(wav: Path) -> tuple[str, list[dict]]:
    from faster_whisper import WhisperModel

    device = "cuda"
    compute_type = "float16"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "int8"
    except Exception:
        device = "cpu"
        compute_type = "int8"

    model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
        download_root=str(Path(MODEL_DIR) / "whisper"),
    )

    segments_iter, info = model.transcribe(
        str(wav),
        task="translate",
        language=None,
        vad_filter=True,
        beam_size=5,
    )

    segments: list[dict] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        parts.append(text)

    full = " ".join(parts).strip()
    print(f"  language≈{info.language} p={info.language_probability:.2f}, {len(segments)} segments")
    return full, segments


def write_transcript(base: Path, text: str, segments: list[dict]) -> None:
    base.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")

    srt_lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        srt_lines.append(str(i))
        srt_lines.append(f"{_format_ts(seg['start'])} --> {_format_ts(seg['end'])}")
        srt_lines.append(seg["text"])
        srt_lines.append("")
    base.with_suffix(".srt").write_text("\n".join(srt_lines), encoding="utf-8")


async def _tts_edge(text: str, out_mp3: Path, voice: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_mp3))


def synthesize_english(text: str, out_wav: Path, voice: str = TTS_VOICE) -> None:
    if not text.strip():
        raise ValueError("Empty transcript — nothing to synthesize")

    tmp_mp3 = Path(TMP_DIR) / f"{out_wav.stem}_tts.mp3"
    asyncio.run(_tts_edge(text, tmp_mp3, voice))

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_mp3),
            "-ar",
            "44100",
            "-ac",
            "1",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )
    tmp_mp3.unlink(missing_ok=True)


def translate_file(wav: Path, out_dir: Path) -> Path:
    base_name = _safe_name(wav.stem.replace("_clean", ""))
    print(f"Translating {wav.name} → English")

    text, segments = transcribe_to_english(wav)
    stem = out_dir / f"{base_name}_en"
    write_transcript(stem, text, segments)

    out_wav = stem.with_suffix(".wav")
    synthesize_english(text, out_wav)
    print(f"  wrote {out_wav.name}, {stem.name}.txt, {stem.name}.srt")
    return out_wav


def main() -> None:
    out_dir = Path(TRANSLATED)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(Path(OUTPUT).glob("*_clean.wav"))
    if not sources:
        # fallback: any wav in output/
        sources = sorted(p for p in Path(OUTPUT).glob("*.wav") if p.parent == Path(OUTPUT))

    if not sources:
        print(f"No WAV files in {OUTPUT}")
        sys.exit(1)

    for wav in sources:
        translate_file(wav, out_dir)

    print("Done")


if __name__ == "__main__":
    main()
