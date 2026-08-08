"""Stage 1: coherent phrase subtitles for MP4 videos.

Whisper ASR (word timestamps) → punctuation-aware phrase packing
→ .srt + burn-in MP4.
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
    SUBS_LONG_BURNED_DIR,
    SUBS_LONG_DIR,
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
_HARD_MAX_CHARS = 350
_PREFERRED_MAX_CHARS = 160


def pack_cues(
    words: list[dict],
    preferred_chars: int = _PREFERRED_MAX_CHARS,
    hard_max_chars: int = _HARD_MAX_CHARS,
) -> list[dict]:
    """Pack words into coherent phrases with punctuation-aware boundaries.

    We prefer a boundary around 160 characters, especially after commas or
    sentence punctuation. A cue may exceed that preference when necessary,
    but never exceeds 350 characters. Word timestamps are retained.
    """
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
        current_text = buf_text()
        current_last = current_text[-1:] if current_text else ""
        word_last = w["word"].rstrip()[-1:] if w["word"].rstrip() else ""

        if len(current_text) >= 80 and current_last in _BREAK_CHARS:
            flush()
            buf.append(w)
            continue

        # The hard limit is absolute: never split inside a word.
        if len(trial_text) > hard_max_chars:
            flush()
            buf.append(w)
            continue

        buf.append(w)
        length = len(trial_text)

        # Finish complete thoughts as soon as punctuation gives us a clean
        # boundary. Commas are preferred, but not before a useful phrase.
        if length >= preferred_chars and word_last in _BREAK_CHARS:
            flush()
        elif length >= preferred_chars and current_last in _BREAK_CHARS:
            flush()
        elif length >= preferred_chars and word_last in ".!?":
            flush()

    flush()

    # Do not leave one or two words stranded after a punctuation split.
    if not cues:
        return []

    merged: list[dict] = []
    for cue in cues:
        if merged and len(cue["text"].split()) <= 2:
            combined = f"{merged[-1]['text']} {cue['text']}".strip()
            if len(combined) <= hard_max_chars:
                merged[-1]["end"] = cue["end"]
                merged[-1]["text"] = combined
                continue
        merged.append(dict(cue))

    # A final micro-cue is joined backward whenever the hard limit permits it.
    if len(merged) > 1 and len(merged[-1]["text"].split()) <= 2:
        combined = f"{merged[-2]['text']} {merged[-1]['text']}".strip()
        if len(combined) <= hard_max_chars:
            merged[-2]["end"] = merged[-1]["end"]
            merged[-2]["text"] = combined
            merged.pop()

    # Fix overlapping / zero-length timestamps.
    for i, cue in enumerate(merged):
        if cue["end"] <= cue["start"]:
            cue["end"] = cue["start"] + 0.4
        if i + 1 < len(merged) and cue["end"] > merged[i + 1]["start"]:
            cue["end"] = max(cue["start"] + 0.2, merged[i + 1]["start"] - 0.02)

    print(
        f"  packed {len(merged)} coherent cues "
        f"(preferred~{preferred_chars} chars, hard_max={hard_max_chars})"
    )
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


def process_video(mp4: Path, force: bool = False, burn: bool = False) -> None:
    base = _safe_name(mp4.stem)
    srt_path = Path(SUBS_LONG_DIR) / f"{base}.srt"
    txt_path = Path(SUBS_LONG_DIR) / f"{base}.txt"
    burned = Path(SUBS_LONG_BURNED_DIR) / f"{base}_subs.mp4"
    work = Path(TMP_DIR) / f"subs_{base}"
    wav = work / "audio.wav"

    output_ready = srt_path.exists() and txt_path.exists()
    if burn:
        output_ready = output_ready and burned.exists() and burned.stat().st_size > 10_000
    if output_ready and not force:
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
        if burn:
            print("  burning subtitles into video…")
            burn_subtitles(mp4, srt_path, burned)
            print(f"  wrote {burned.name}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate short subtitles for MP4 videos")
    parser.add_argument("--only", default="", help="Substring filter for filenames")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--burn",
        action="store_true",
        help="Also create videos with subtitles burned in",
    )
    args = parser.parse_args()

    Path(SUBS_LONG_DIR).mkdir(parents=True, exist_ok=True)
    Path(SUBS_LONG_BURNED_DIR).mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(VIDEO_INPUT).glob("*.mp4"))
    if args.only:
        videos = [v for v in videos if args.only.lower() in v.name.lower()]

    if not videos:
        print(f"No MP4 in {VIDEO_INPUT}")
        sys.exit(1)

    for mp4 in videos:
        process_video(mp4, force=args.force, burn=args.burn)

    print("Done")


if __name__ == "__main__":
    main()
