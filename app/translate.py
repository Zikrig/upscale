"""Dub cleaned vocals into English with voice cloning + soft timeline match.

Quality-first approach:
  - merge Whisper fragments into longer phrases (not 1s crumbs)
  - XTTS clones voice; prefer its native `speed` over heavy time-stretch
  - only gentle stretch/pad; keep temp disk usage near zero
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
    STRETCH_MAX,
    STRETCH_MIN,
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


def write_transcript(base: Path, text: str, segments: list[dict]) -> None:
    base.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
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


def xtts_speak_array(tts, text: str, speaker_wav: Path, speed: float) -> np.ndarray:
    """Synthesize in-memory to avoid filling /tmp with thousands of WAVs."""
    speed = float(np.clip(speed, 0.6, 1.6))
    wav = tts.tts(
        text=text,
        speaker_wav=str(speaker_wav),
        language="en",
        split_sentences=True,
        speed=speed,
    )
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def fit_gently(audio: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Only light stretch; if still off — pad silence or trim edges (don't wreck voice)."""
    import librosa

    target_n = max(int(round(target_sec * sr)), 1)
    cur_sec = max(len(audio) / sr, 1e-6)
    rate = cur_sec / max(target_sec, 1e-6)

    if abs(rate - 1.0) >= 0.04:
        rate = float(np.clip(rate, STRETCH_MIN, STRETCH_MAX))
        audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)

    if len(audio) > target_n:
        # trim evenly instead of hard-chopping the end only
        excess = len(audio) - target_n
        left = excess // 2
        audio = audio[left : left + target_n]
    elif len(audio) < target_n:
        pad = target_n - len(audio)
        left = pad // 2
        audio = np.pad(audio, (left, pad - left))

    return audio.astype(np.float32)


def dub_segments(
    tts,
    segments: list[dict],
    speaker_wav: Path,
    total_duration: float,
    sr: int = SR,
) -> np.ndarray:
    timeline = np.zeros(int(round(total_duration * sr)) + sr, dtype=np.float32)

    for i, seg in enumerate(tqdm(segments, desc="  dubbing", leave=False)):
        text = seg["text"].strip()
        if not text:
            continue

        target = max(seg["end"] - seg["start"], 0.25)
        # Rough native speed hint: English often needs slight speed-up vs RU timing
        approx_chars_per_sec = max(len(text) / target, 1e-3)
        # ~14 chars/sec is a calm English speaking rate
        speed = float(np.clip(approx_chars_per_sec / 14.0, 0.85, 1.35))

        try:
            audio = xtts_speak_array(tts, text, speaker_wav, speed=speed)
        except Exception as exc:
            print(f"  ! phrase {i} TTS failed: {exc}")
            continue

        fitted = fit_gently(audio, sr, target)
        start = int(round(seg["start"] * sr))
        end = start + len(fitted)
        if start >= len(timeline):
            continue
        if end > len(timeline):
            fitted = fitted[: len(timeline) - start]
            end = len(timeline)

        fade = min(int(0.02 * sr), max(len(fitted) // 8, 1))
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            fitted[:fade] *= ramp
            fitted[-fade:] *= ramp[::-1]

        timeline[start:end] += fitted

    peak = float(np.max(np.abs(timeline)) + 1e-8)
    if peak > 0.99:
        timeline *= 0.99 / peak
    return timeline[: int(round(total_duration * sr))]


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

    print(f"Translating {wav.name} → English (cloned, quality mode)")
    cleanup_work(base_name)

    text, segments, duration = transcribe_to_english(wav)
    # Prefer cached transcript if re-running only audio and srt already good? always refresh
    phrases = merge_phrases(segments)
    write_transcript(stem, text, phrases)

    work = Path(TMP_DIR) / f"dub_{base_name}"
    work.mkdir(parents=True, exist_ok=True)
    ref = extract_reference(wav, work / "speaker_ref.wav")

    dubbed = dub_segments(tts, phrases, ref, duration)
    write_audio(out_wav, dubbed, SR)
    cleanup_work(base_name)

    print(f"  wrote {out_wav.name} ({duration:.1f}s), {stem.name}.txt/.srt")
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
