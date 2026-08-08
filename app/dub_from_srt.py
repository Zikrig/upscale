"""Create English speech from an existing English SRT.

The SRT text is used verbatim: no Whisper transcription or translation is
performed. Each cue is synthesized as one short phrase, fitted to its own
subtitle window, and joined with a short crossfade.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from config import OUTPUT

SR = 24_000
CROSSFADE_MS = 45
SILENCE_RMS = 0.008
TIMESTAMP = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def parse_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    h, m, s, ms = (int(part) for part in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def read_srt(path: Path) -> list[dict]:
    blocks = re.split(r"\r?\n\r?\n+", path.read_text(encoding="utf-8-sig").strip())
    cues: list[dict] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        match = TIMESTAMP.fullmatch(lines[1].strip())
        if not match:
            raise ValueError(f"Invalid timestamp in {path.name}: {lines[1]}")
        cues.append(
            {
                "start": parse_timestamp(lines[1].split("-->")[0].strip()),
                "end": parse_timestamp(lines[1].split("-->")[1].strip()),
                "text": " ".join(line.strip() for line in lines[2:]).strip(),
            }
        )
    if not cues:
        raise ValueError(f"No cues found in {path}")
    return cues


async def synthesize(text: str, output: Path, voice: str) -> None:
    import edge_tts

    # One SRT cue is already a short phrase. Do not split it again.
    await edge_tts.Communicate(
        text,
        voice=voice,
        rate="+0%",
        volume="+0%",
    ).save(str(output))


def to_wav(mp3: Path, wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-ac",
            "1",
            "-ar",
            str(SR),
            str(wav),
        ],
        check=True,
    )


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Remove only quiet padding, never speech-shaped material."""
    if len(audio) == 0:
        return audio
    window = max(round(SR * 0.01), 1)
    energy = np.convolve(np.abs(audio), np.ones(window) / window, mode="same")
    active = np.flatnonzero(energy >= SILENCE_RMS)
    if len(active) == 0:
        return audio
    margin = round(SR * 0.012)
    start = max(int(active[0]) - margin, 0)
    end = min(int(active[-1]) + margin + 1, len(audio))
    return audio[start:end]


def fit_to_window(audio: np.ndarray, duration_sec: float) -> np.ndarray:
    """Fit a clip without time-stretching; trim only its boundaries."""
    limit = max(round(duration_sec * SR), 1)
    audio = trim_silence(audio)
    if len(audio) <= limit:
        return audio
    excess = len(audio) - limit
    left = excess // 2
    return audio[left : left + limit]


def add_crossfade(
    timeline: np.ndarray,
    start: int,
    audio: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Mix a short linear crossfade at a cue boundary."""
    if start >= len(timeline):
        timeline = np.pad(timeline, (0, start - len(timeline) + 1))
    end = start + len(audio)
    if end > len(timeline):
        timeline = np.pad(timeline, (0, end - len(timeline)))
    fade = min(crossfade_samples, len(audio), start, len(timeline))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        timeline[start - fade : start] *= 1.0 - ramp
        timeline[start - fade : start] += audio[:fade] * ramp
        timeline[start + fade : end] += audio[fade:]
    else:
        timeline[start:end] += audio
    return timeline


def create_dub(srt: Path, output: Path, voice: str) -> None:
    cues = read_srt(srt)
    with tempfile.TemporaryDirectory(prefix="srt_dub_") as temp_name:
        temp = Path(temp_name)
        pieces: list[tuple[int, np.ndarray]] = []
        crossfade_samples = round(SR * CROSSFADE_MS / 1000)

        for index, cue in enumerate(cues, 1):
            if not cue["text"]:
                continue
            mp3 = temp / f"{index:04d}.mp3"
            wav = temp / f"{index:04d}.wav"
            print(f"[{index}/{len(cues)}] {cue['text']}")
            asyncio.run(synthesize(cue["text"], mp3, voice))
            to_wav(mp3, wav)
            audio, file_sr = sf.read(str(wav), always_2d=False)
            if file_sr != SR:
                raise RuntimeError(f"Unexpected sample rate in {wav}: {file_sr}")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Keep every clip inside its own SRT window. If it is too long,
            # only its quiet/outer boundaries are removed; tempo is unchanged.
            start_sec = cue["start"]
            audio = fit_to_window(
                audio.astype(np.float32),
                cue["end"] - cue["start"],
            )
            pieces.append((round(start_sec * SR), audio.astype(np.float32)))

        total = max(
            max((start + len(audio) for start, audio in pieces), default=SR) + SR,
            round(cues[-1]["end"] * SR) + SR,
        )
        timeline = np.zeros(total, dtype=np.float32)
        for start, audio in pieces:
            timeline = add_crossfade(timeline, start, audio, crossfade_samples)

    peak = float(np.max(np.abs(timeline), initial=0.0))
    if peak > 0.99:
        timeline *= 0.99 / peak
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), timeline, SR, subtype="PCM_16")
    print(f"Wrote {output} ({len(timeline) / SR:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--voice", default="en-US-BrianNeural")
    args = parser.parse_args()

    if not args.srt:
        parser.error("--srt is required")
    if not args.out:
        args.out = Path(OUTPUT) / "dub_en" / f"{args.srt.stem}.wav"
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    create_dub(args.srt, args.out, args.voice)


if __name__ == "__main__":
    main()
