"""Synthesize one complete text file without subtitle timing."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path


async def synthesize(text: str, mp3: Path, voice: str) -> None:
    import edge_tts

    await edge_tts.Communicate(
        text,
        voice=voice,
        rate="+0%",
        volume="+0%",
    ).save(str(mp3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voice", default="en-US-BrianNeural")
    args = parser.parse_args()

    if not args.text.exists():
        raise SystemExit(f"Missing text file: {args.text}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mp3 = args.out.with_suffix(".mp3")
    text = args.text.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Empty text file: {args.text}")

    asyncio.run(synthesize(text, mp3, args.voice))
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
            "24000",
            str(args.out),
        ],
        check=True,
    )
    mp3.unlink(missing_ok=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
