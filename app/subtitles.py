"""Stage 1: short, roughly equal-length subtitles for MP4 videos.

Whisper ASR (word timestamps) → pack cues → .srt + burn-in MP4.
No voice clone. One video at a time (disk/VRAM limits).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import (
    MODEL_DIR,
    SUB_MAX_CHARS,
    SUB_MAX_SEC,
    SUB_MIN_SEC,
    SUB_TARGET_SEC,
    SUBS_BURNED_DIR,
    SUBS_DIR,
    TMP_DIR,
    VIDEO_INPUT,
    WHISPER_MODEL,
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "video"


def _format_ts(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_audio(mp4: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )


def transcribe_words(wav: Path) -> tuple[list[dict], str]:
    from faster_whisper import WhisperModel

    device = _device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"  Whisper {WHISPER_MODEL} on {device} ({compute_type})")
    model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
        download_root=str(Path(MODEL_DIR) / "whisper"),
    )

    segments, info = model.transcribe(
        str(wav),
        language=None,
        task="transcribe",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=True,
    )

    words: list[dict] = []
    parts: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
        if not seg.words:
            if text:
                words.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "word": text,
                    }
                )
            continue
        for w in seg.words:
            token = (w.word or "").strip()
            if not token:
                continue
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": token,
                }
            )

    print(
        f"  language≈{info.language} p={info.language_probability:.2f}, "
        f"{len(words)} words"
    )
    return words, " ".join(parts).strip()


_BREAK_CHARS = set(".!?…,;:")


def pack_cues(
    words: list[dict],
    target_sec: float = SUB_TARGET_SEC,
    max_sec: float = SUB_MAX_SEC,
    min_sec: float = SUB_MIN_SEC,
    max_chars: int = SUB_MAX_CHARS,
) -> list[dict]:
    """Pack words into short, roughly equal subtitle cues."""
    if not words:
        return []

    cues: list[dict] = []
    buf: list[dict] = []

    def buf_text() -> str:
        return re.sub(r"\s+", " ", " ".join(w["word"] for w in buf)).strip()

    def buf_start() -> float:
        return buf[0]["start"]

    def buf_end() -> float:
        return buf[-1]["end"]

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = buf_text()
        if text:
            cues.append({"start": buf_start(), "end": max(buf_end(), buf_start() + 0.05), "text": text})
        buf = []

    for w in words:
        if not buf:
            buf.append(w)
            continue

        trial = buf + [w]
        trial_text = re.sub(r"\s+", " ", " ".join(x["word"] for x in trial)).strip()
        trial_dur = trial[-1]["end"] - trial[0]["start"]
        gap = w["start"] - buf_end()
        last_ch = buf_text()[-1] if buf_text() else ""

        should_break = False
        if trial_dur > max_sec or len(trial_text) > max_chars:
            should_break = True
        elif trial_dur >= target_sec and (last_ch in _BREAK_CHARS or gap >= 0.28):
            should_break = True
        elif gap >= 0.55 and trial_dur >= min_sec:
            should_break = True

        if should_break:
            flush()
            buf.append(w)
        else:
            buf.append(w)

    flush()

    # Merge micro-cues into neighbors
    if not cues:
        return []

    merged: list[dict] = [dict(cues[0])]
    for cue in cues[1:]:
        prev = merged[-1]
        prev_dur = prev["end"] - prev["start"]
        cur_dur = cue["end"] - cue["start"]
        combined = f"{prev['text']} {cue['text']}".strip()
        if (prev_dur < min_sec or cur_dur < min_sec) and len(combined) <= max_chars + 12:
            prev["end"] = cue["end"]
            prev["text"] = combined
        else:
            merged.append(dict(cue))

    # Fix overlapping / zero-length
    for i, cue in enumerate(merged):
        if cue["end"] <= cue["start"]:
            cue["end"] = cue["start"] + 0.4
        if i + 1 < len(merged) and cue["end"] > merged[i + 1]["start"]:
            cue["end"] = max(cue["start"] + 0.2, merged[i + 1]["start"] - 0.02)

    print(f"  packed {len(merged)} cues (target≈{target_sec}s, max_chars={max_chars})")
    return merged


def write_srt(path: Path, cues: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, cue in enumerate(cues, 1):
        lines += [
            str(i),
            f"{_format_ts(cue['start'])} --> {_format_ts(cue['end'])}",
            cue["text"],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def burn_subtitles(mp4: Path, srt: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg subtitles filter: escape \ : '
    srt_escaped = str(srt.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", r"\'")
    style = (
        "FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF&,"
        "OutlineColour=&H00000000&,BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=2,MarginV=28"
    )
    vf = f"subtitles={srt_escaped}:force_style='{style}'"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4),
            "-vf",
            vf,
            "-c:a",
            "copy",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            str(out_mp4),
        ],
        check=True,
        capture_output=True,
    )


def process_video(mp4: Path, force: bool = False) -> None:
    base = _safe_name(mp4.stem)
    srt_path = Path(SUBS_DIR) / f"{base}.srt"
    txt_path = Path(SUBS_DIR) / f"{base}.txt"
    burned = Path(SUBS_BURNED_DIR) / f"{base}_subs.mp4"
    work = Path(TMP_DIR) / f"subs_{base}"
    wav = work / "audio.wav"

    if srt_path.exists() and burned.exists() and burned.stat().st_size > 10_000 and not force:
        print(f"Skip {mp4.name} (exists)")
        return

    print(f"Subtitles for {mp4.name}")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    try:
        extract_audio(mp4, wav)
        words, full_text = transcribe_words(wav)
        cues = pack_cues(words)
        if not cues:
            print("  ! no speech detected")
            return
        write_srt(srt_path, cues)
        txt_path.write_text(full_text + "\n", encoding="utf-8")
        print(f"  wrote {srt_path.name}")
        print("  burning subtitles into video…")
        burn_subtitles(mp4, srt_path, burned)
        print(f"  wrote {burned.name}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate short subtitles for MP4 videos")
    parser.add_argument("--only", default="", help="Substring filter for filenames")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    Path(SUBS_DIR).mkdir(parents=True, exist_ok=True)
    Path(SUBS_BURNED_DIR).mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(VIDEO_INPUT).glob("*.mp4"))
    if args.only:
        videos = [v for v in videos if args.only.lower() in v.name.lower()]

    if not videos:
        print(f"No MP4 in {VIDEO_INPUT}")
        sys.exit(1)

    for mp4 in videos:
        process_video(mp4, force=args.force)

    print("Done")


if __name__ == "__main__":
    main()
