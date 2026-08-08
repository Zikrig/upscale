"""Extract one continuous text file from each English SRT."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract(path: Path) -> str:
    blocks = re.split(r"\r?\n\r?\n+", path.read_text(encoding="utf-8-sig").strip())
    parts: list[str] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            text = " ".join(line.strip() for line in lines[2:]).strip()
            if text:
                parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("output/subs_long_en"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/tts_text"))
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.srt"))
    if not files:
        raise SystemExit(f"No SRT files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = args.output_dir / f"{source.stem}.txt"
        target.write_text(extract(source) + "\n", encoding="utf-8")
        print(f"{source.name} -> {target}")


if __name__ == "__main__":
    main()
