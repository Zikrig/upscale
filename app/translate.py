"""Dub cleaned vocals into English with voice cloning + timeline match.

Pipeline:
  1. Whisper (task=translate) → English text with original timestamps
  2. XTTS-v2 clones voice from the source WAV
  3. Each segment is time-stretched into its original [start, end] window
  4. Segments are mixed onto a timeline of the original duration
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import (
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
    peak = np.max(np.abs(audio)) + 1e-8
    if peak > 1.0:
        audio = audio / peak * 0.99
    sf.write(str(path), audio, sr)


def extract_reference(wav: Path, out_path: Path, seconds: float = REF_SECONDS) -> Path:
    """Pick a loud mid-file window as XTTS speaker reference."""
    audio, sr = load_audio(wav)
    if len(audio) < int(sr * 2):
        write_audio(out_path, audio, sr)
        return out_path

    win = int(sr * seconds)
    win = min(win, len(audio))
    hop = max(win // 4, sr // 2)
    best_i, best_e = 0, -1.0
    for i in range(0, max(len(audio) - win, 1), hop):
        chunk = audio[i : i + win]
        energy = float(np.mean(chunk * chunk))
        if energy > best_e:
            best_e, best_i = energy, i

    ref = audio[best_i : best_i + win]
    write_audio(out_path, ref, sr)
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
        word_timestamps=False,
    )

    segments: list[dict] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        if seg.end <= seg.start:
            continue
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        parts.append(text)

    duration = segments[-1]["end"] if segments else 0.0
    # Prefer source file duration for the timeline
    audio, sr = load_audio(wav)
    duration = max(duration, len(audio) / sr)

    print(
        f"  language≈{info.language} p={info.language_probability:.2f}, "
        f"{len(segments)} segments, {duration:.1f}s"
    )
    return " ".join(parts).strip(), segments, duration


def merge_short_segments(segments: list[dict], min_dur: float = 0.8, max_chars: int = 180) -> list[dict]:
    """Merge tiny Whisper fragments so XTTS gets more natural phrases."""
    if not segments:
        return []

    merged: list[dict] = []
    cur = dict(segments[0])
    for seg in segments[1:]:
        cur_dur = cur["end"] - cur["start"]
        gap = seg["start"] - cur["end"]
        combined = f"{cur['text']} {seg['text']}".strip()
        if cur_dur < min_dur and gap < 0.35 and len(combined) <= max_chars:
            cur["end"] = seg["end"]
            cur["text"] = combined
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged


def write_transcript(base: Path, text: str, segments: list[dict]) -> None:
    base.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_ts(seg['start'])} --> {_format_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    base.with_suffix(".srt").write_text("\n".join(lines), encoding="utf-8")


def load_xtts():
    warnings.filterwarnings("ignore")
    _patch_transformers_isin()
    from TTS.api import TTS

    device = _device()
    print(f"  loading XTTS on {device}…")
    tts = TTS(XTTS_MODEL, progress_bar=False)
    tts.to(device)
    return tts


def _patch_transformers_isin() -> None:
    """coqui-tts expects isin_mps_friendly; missing in transformers 5.x."""
    try:
        from transformers import pytorch_utils
        import torch

        if not hasattr(pytorch_utils, "isin_mps_friendly"):
            pytorch_utils.isin_mps_friendly = torch.isin
    except Exception:
        pass


def xtts_speak(tts, text: str, speaker_wav: Path, out_path: Path) -> None:
    tts.tts_to_file(
        text=text,
        file_path=str(out_path),
        speaker_wav=str(speaker_wav),
        language="en",
        split_sentences=True,
    )


def fit_to_duration(audio: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Time-stretch (pitch-preserving) to target length; pad/trim as fallback."""
    import librosa

    if target_sec <= 0.05:
        return np.zeros(int(sr * 0.05), dtype=np.float32)

    target_n = int(round(target_sec * sr))
    cur_sec = max(len(audio) / sr, 1e-6)
    rate = cur_sec / target_sec  # >1 → speed up

    if abs(rate - 1.0) < 0.03:
        out = audio
    else:
        rate = float(np.clip(rate, STRETCH_MIN, STRETCH_MAX))
        out = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)

    if len(out) > target_n:
        out = out[:target_n]
    elif len(out) < target_n:
        out = np.pad(out, (0, target_n - len(out)))
    return out.astype(np.float32)


def dub_segments(
    tts,
    segments: list[dict],
    speaker_wav: Path,
    total_duration: float,
    work_dir: Path,
    sr: int = 24000,
) -> np.ndarray:
    timeline = np.zeros(int(round(total_duration * sr)) + sr, dtype=np.float32)
    work_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(tqdm(segments, desc="  dubbing", leave=False)):
        text = seg["text"].strip()
        if not text:
            continue

        target = max(seg["end"] - seg["start"], 0.1)
        raw_path = work_dir / f"seg_{i:05d}_raw.wav"
        try:
            xtts_speak(tts, text, speaker_wav, raw_path)
        except Exception as exc:
            print(f"  ! segment {i} TTS failed: {exc}")
            continue

        audio, file_sr = load_audio(raw_path, sr=sr)
        fitted = fit_to_duration(audio, sr, target)

        start = int(round(seg["start"] * sr))
        end = start + len(fitted)
        if start >= len(timeline):
            continue
        if end > len(timeline):
            fitted = fitted[: len(timeline) - start]
            end = len(timeline)

        # short fade to avoid clicks
        fade = min(int(0.01 * sr), len(fitted) // 4)
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            fitted[:fade] *= ramp
            fitted[-fade:] *= ramp[::-1]

        timeline[start:end] += fitted

    # soft limiter
    peak = np.max(np.abs(timeline)) + 1e-8
    if peak > 0.99:
        timeline *= 0.99 / peak
    return timeline[: int(round(total_duration * sr))]


def translate_file(wav: Path, out_dir: Path, tts) -> Path:
    base_name = _safe_name(wav.stem.replace("_clean", ""))
    print(f"Translating {wav.name} → English (cloned + timed)")

    text, segments, duration = transcribe_to_english(wav)
    segments = merge_short_segments(segments)
    stem = out_dir / f"{base_name}_en"
    write_transcript(stem, text, segments)

    work = Path(TMP_DIR) / f"dub_{base_name}"
    work.mkdir(parents=True, exist_ok=True)
    ref = extract_reference(wav, work / "speaker_ref.wav")

    dubbed = dub_segments(tts, segments, ref, duration, work / "segs")
    out_wav = stem.with_suffix(".wav")
    write_audio(out_wav, dubbed, 24000)

    print(f"  wrote {out_wav.name} ({duration:.1f}s), {stem.name}.txt, {stem.name}.srt")
    return out_wav


def main() -> None:
    out_dir = Path(TRANSLATED)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(Path(OUTPUT).glob("*_clean.wav"))
    if not sources:
        sources = sorted(p for p in Path(OUTPUT).glob("*.wav") if p.parent == Path(OUTPUT))

    if not sources:
        print(f"No WAV files in {OUTPUT}")
        sys.exit(1)

    tts = load_xtts()
    for wav in sources:
        translate_file(wav, out_dir, tts)

    print("Done")


if __name__ == "__main__":
    main()
