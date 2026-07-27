"""Dub cleaned vocals into English with voice cloning.

No speed-up / slow-down. Phrases keep natural XTTS tempo.
Placement is sequential so clips never overlap:
  start = max(original_start, end_of_previous_clip)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import (
    MERGE_MAX_CHARS,
    MERGE_MAX_GAP,
    MERGE_MAX_SEC,
    MERGE_TARGET_SEC,
    MODEL_DIR,
    OUTPUT,
    REF_SECONDS,
    TMP_DIR,
    TRANSLATED,
    WHISPER_MODEL,
    XTTS_MODEL,
)

SR = 24000


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "track"


def _format_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, file_sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr is not None and file_sr != sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        return audio, sr
    return audio, file_sr


def write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak > 1.0:
        audio = audio / peak * 0.99
    sf.write(str(path), audio, sr, subtype="PCM_16")


def extract_reference(wav: Path, out_path: Path, seconds: float = REF_SECONDS) -> Path:
    audio, sr = load_audio(wav)
    if len(audio) < int(sr * 3):
        write_audio(out_path, audio, sr)
        return out_path

    win = min(int(sr * seconds), len(audio))
    hop = max(win // 5, sr // 2)
    best_i, best_e = 0, -1.0
    for i in range(0, max(len(audio) - win, 1), hop):
        chunk = audio[i : i + win]
        # Prefer speechy energy, avoid pure silence
        energy = float(np.mean(np.abs(chunk)))
        if energy > best_e:
            best_e, best_i = energy, i

    write_audio(out_path, audio[best_i : best_i + win], sr)
    return out_path


def transcribe_to_english(wav: Path) -> tuple[str, list[dict], float]:
    from faster_whisper import WhisperModel

    device = _device()
    compute_type = "float16" if device == "cuda" else "int8"
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
        condition_on_previous_text=True,
    )

    segments: list[dict] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text or seg.end <= seg.start:
            continue
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        parts.append(text)

    audio, sr = load_audio(wav)
    duration = len(audio) / sr
    print(
        f"  language≈{info.language} p={info.language_probability:.2f}, "
        f"{len(segments)} raw segments, {duration:.1f}s"
    )
    return " ".join(parts).strip(), segments, duration


def merge_phrases(segments: list[dict]) -> list[dict]:
    """Build longer natural phrases — key for XTTS quality."""
    if not segments:
        return []

    merged: list[dict] = []
    cur = dict(segments[0])

    for seg in segments[1:]:
        gap = seg["start"] - cur["end"]
        dur = cur["end"] - cur["start"]
        combined = f"{cur['text']} {seg['text']}".strip()
        ends_sentence = bool(re.search(r"[.!?…][\"')\]]*$", cur["text"].strip()))

        can_merge = (
            gap <= MERGE_MAX_GAP
            and len(combined) <= MERGE_MAX_CHARS
            and (seg["end"] - cur["start"]) <= MERGE_MAX_SEC
            and (dur < MERGE_TARGET_SEC or (not ends_sentence and dur < MERGE_MAX_SEC))
        )
        if can_merge:
            cur["end"] = seg["end"]
            cur["text"] = combined
        else:
            merged.append(cur)
            cur = dict(seg)

    merged.append(cur)
    print(f"  merged into {len(merged)} phrases (target ~{MERGE_TARGET_SEC:.0f}s)")
    return merged


def write_transcript(base: Path, text: str, segments: list[dict], timed: list[dict] | None = None) -> None:
    """Write full text + SRT. If timed is given, use actual placed starts/ends."""
    base.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    rows = timed if timed is not None else segments
    lines: list[str] = []
    for i, seg in enumerate(rows, 1):
        lines += [
            str(i),
            f"{_format_ts(seg['start'])} --> {_format_ts(seg['end'])}",
            seg["text"],
            "",
        ]
    base.with_suffix(".srt").write_text("\n".join(lines), encoding="utf-8")


def _patch_transformers_isin() -> None:
    try:
        import torch
        from transformers import pytorch_utils

        if not hasattr(pytorch_utils, "isin_mps_friendly"):
            pytorch_utils.isin_mps_friendly = torch.isin
    except Exception:
        pass


def load_xtts():
    warnings.filterwarnings("ignore")
    _patch_transformers_isin()
    from TTS.api import TTS

    device = _device()
    print(f"  loading XTTS on {device}…")
    tts = TTS(XTTS_MODEL, progress_bar=False)
    tts.to(device)
    return tts


def xtts_speak_array(tts, text: str, speaker_wav: Path) -> np.ndarray:
    """Natural-speed synthesis in memory (no tempo change)."""
    wav = tts.tts(
        text=text,
        speaker_wav=str(speaker_wav),
        language="en",
        split_sentences=True,
    )
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def dub_segments(
    tts,
    segments: list[dict],
    speaker_wav: Path,
    sr: int = SR,
) -> tuple[np.ndarray, list[dict]]:
    """Place phrases at natural length without overlap.

    Each clip starts at max(original_start, previous_end).
    Timeline grows if English runs longer than the source.
    """
    chunks: list[tuple[int, np.ndarray]] = []
    timed: list[dict] = []
    cursor = 0.0  # seconds, end of last placed clip

    for i, seg in enumerate(tqdm(segments, desc="  dubbing", leave=False)):
        text = seg["text"].strip()
        if not text:
            continue

        try:
            audio = xtts_speak_array(tts, text, speaker_wav)
        except Exception as exc:
            print(f"  ! phrase {i} TTS failed: {exc}")
            continue

        start_sec = max(float(seg["start"]), cursor)
        start = int(round(start_sec * sr))
        dur_sec = len(audio) / sr
        end_sec = start_sec + dur_sec

        fade = min(int(0.015 * sr), max(len(audio) // 10, 1))
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            audio = audio.copy()
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]

        chunks.append((start, audio))
        timed.append({"start": start_sec, "end": end_sec, "text": text})
        cursor = end_sec

    total_n = int(round(cursor * sr)) + sr
    if chunks:
        last_start, last_audio = chunks[-1]
        total_n = max(total_n, last_start + len(last_audio) + sr)

    timeline = np.zeros(total_n, dtype=np.float32)
    for start, audio in chunks:
        end = start + len(audio)
        if end > len(timeline):
            timeline = np.pad(timeline, (0, end - len(timeline) + sr))
        timeline[start:end] += audio

    peak = float(np.max(np.abs(timeline)) + 1e-8)
    if peak > 0.99:
        timeline *= 0.99 / peak

    print(f"  timeline {len(timeline) / sr:.1f}s (no overlap, natural tempo)")
    return timeline, timed


def cleanup_work(base_name: str) -> None:
    work = Path(TMP_DIR) / f"dub_{base_name}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)


def translate_file(wav: Path, out_dir: Path, tts, force: bool = False) -> Path | None:
    base_name = _safe_name(wav.stem.replace("_clean", ""))
    stem = out_dir / f"{base_name}_en"
    out_wav = stem.with_suffix(".wav")

    if out_wav.exists() and out_wav.stat().st_size > 10_000 and not force:
        print(f"Skip {wav.name} (exists: {out_wav.name})")
        return out_wav

    print(f"Translating {wav.name} → English (clone, natural speed, no overlap)")
    cleanup_work(base_name)

    text, segments, _duration = transcribe_to_english(wav)
    phrases = merge_phrases(segments)

    work = Path(TMP_DIR) / f"dub_{base_name}"
    work.mkdir(parents=True, exist_ok=True)
    ref = extract_reference(wav, work / "speaker_ref.wav")

    dubbed, timed = dub_segments(tts, phrases, ref)
    write_transcript(stem, text, phrases, timed=timed)
    write_audio(out_wav, dubbed, SR)
    cleanup_work(base_name)

    print(f"  wrote {out_wav.name} ({len(dubbed) / SR:.1f}s), {stem.name}.txt/.srt")
    return out_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Substring filter, e.g. bitards",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    # Free space from previous failed runs
    for p in Path(TMP_DIR).glob("dub_*"):
        shutil.rmtree(p, ignore_errors=True)

    out_dir = Path(TRANSLATED)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(Path(OUTPUT).glob("*_clean.wav"))
    if args.only:
        sources = [p for p in sources if args.only.lower() in p.name.lower()]

    if not sources:
        print(f"No matching WAV files in {OUTPUT}")
        sys.exit(1)

    tts = load_xtts()
    for wav in sources:
        translate_file(wav, out_dir, tts, force=args.force)

    print("Done")


if __name__ == "__main__":
    main()
