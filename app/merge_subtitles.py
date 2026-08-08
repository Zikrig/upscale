"""Build long English subtitles from long timings and short English text."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TIMESTAMP = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value)
    if not match:
        raise ValueError(f"Invalid timestamp: {value}")
    h, m, s, ms = (int(part) for part in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def read_srt(path: Path) -> list[dict]:
    blocks = re.split(r"\r?\n\r?\n+", path.read_text(encoding="utf-8-sig").strip())
    cues: list[dict] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        match = TIMESTAMP.fullmatch(lines[1].strip())
        if not match:
            raise ValueError(f"Invalid SRT block in {path}: {lines[1]}")
        start, end = lines[1].split("-->")
        cues.append(
            {
                "start": timestamp(start.strip()),
                "end": timestamp(end.strip()),
                "text": " ".join(line.strip() for line in lines[2:]).strip(),
            }
        )
    return cues


def overlap(left: dict, right: dict) -> float:
    return max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def merge_file(long_path: Path, short_path: Path, output_path: Path) -> None:
    long_cues = read_srt(long_path)
    short_cues = read_srt(short_path)
    if not long_cues or not short_cues:
        raise ValueError(f"Empty subtitle file: {long_path} or {short_path}")

    assigned: list[list[dict]] = [[] for _ in long_cues]
    for short_cue in short_cues:
        ranked = sorted(
            (
                (overlap(long_cue, short_cue), long_index)
                for long_index, long_cue in enumerate(long_cues)
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        best_overlap, best_index = ranked[0]
        if best_overlap <= 0:
            short_center = (short_cue["start"] + short_cue["end"]) / 2
            best_index = min(
                range(len(long_cues)),
                key=lambda index: abs(
                    (long_cues[index]["start"] + long_cues[index]["end"]) / 2
                    - short_center
                ),
            )
        assigned[best_index].append(short_cue)

    merged: list[dict] = []
    for long_index, long_cue in enumerate(long_cues):
        selected = sorted(assigned[long_index], key=lambda cue: cue["start"])
        text = " ".join(cue["text"] for cue in selected).strip()
        merged.append(
            {
                "start": long_cue["start"],
                "end": long_cue["end"],
                "text": re.sub(r"\s+", " ", text),
            }
        )

    # Timing boundaries can fall inside one translated sentence. Move the
    # first complete sentence from the next cue back to the current cue so a
    # long subtitle never ends with an unfinished phrase.
    sentence_prefix = re.compile(r"^(.+?[.!?])(?:\s+|$)(.*)$", re.S)
    for index in range(len(merged) - 1):
        current = merged[index]
        following = merged[index + 1]
        if not current["text"] or not following["text"]:
            continue
        if current["text"].rstrip()[-1:] in ".!?":
            continue
        match = sentence_prefix.match(following["text"])
        if not match:
            continue
        prefix, remainder = match.groups()
        combined = f"{current['text']} {prefix}".strip()
        if len(combined) <= 350:
            current["text"] = combined
            following["text"] = remainder.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, cue in enumerate(merged, 1):
        lines += [
            str(index),
            f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}",
            cue["text"],
            "",
        ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{long_path.name} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-dir", type=Path, default=Path("output/subs_long"))
    parser.add_argument("--en-dir", type=Path, default=Path("output/subs_en"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/subs_long_en"))
    args = parser.parse_args()

    long_files = sorted(args.long_dir.glob("*.srt"))
    if not long_files:
        raise SystemExit(f"No long SRT files found in {args.long_dir}")

    for long_path in long_files:
        short_path = args.en_dir / long_path.name
        if not short_path.exists():
            raise SystemExit(f"Missing English SRT: {short_path}")
        merge_file(long_path, short_path, args.out_dir / long_path.name)


if __name__ == "__main__":
    main()
