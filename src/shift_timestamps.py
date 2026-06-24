#!/usr/bin/env python3
"""Shift SRT timestamps by a given offset starting from a specified entry index."""

import re
from datetime import timedelta


def parse_time(time_str: str) -> timedelta:
    h, m, rest = time_str.split(":")
    s, ms = rest.split(",")
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))


def format_time(td: timedelta) -> str:
    total_ms = int(td.total_seconds() * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
)


def shift_srt(input_path: str, output_path: str, offset_ms: int, from_index: int) -> None:
    offset = timedelta(milliseconds=offset_ms)

    with open(input_path, encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    result = []

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            result.append(block)
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            result.append(block)
            continue

        if index >= from_index:
            def shift_match(m):
                start = parse_time(m.group(1)) + offset
                end = parse_time(m.group(2)) + offset
                return f"{format_time(start)} --> {format_time(end)}"

            new_block = lines[0] + "\n" + TIMESTAMP_RE.sub(shift_match, "\n".join(lines[1:]))
            result.append(new_block)
        else:
            result.append(block)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(result) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shift SRT timestamps from a given entry index.")
    parser.add_argument("input", help="Input .srt file")
    parser.add_argument("output", nargs="?", help="Output .srt file (defaults to overwrite input)")
    parser.add_argument("--offset", type=int, default=5000, help="Offset in milliseconds (default: 5000)")
    parser.add_argument("--from-index", type=int, default=3, help="Start shifting from this entry index (default: 3)")
    args = parser.parse_args()

    output = args.output or args.input
    shift_srt(args.input, output, args.offset, args.from_index)
    print(f"Shifted entries from index {args.from_index} by {args.offset}ms -> {output}")
