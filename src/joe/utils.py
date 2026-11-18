from __future__ import annotations

from typing import Any


def format_srt_time(seconds: float) -> str:
    """Convert seconds to the SRT timestamp format HH:MM:SS,mmm."""
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def build_srt(segments: list[dict[str, Any]]) -> str:
    """Build an SRT document from Parakeet segment timestamps."""
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = format_srt_time(float(segment.get("start", 0.0)))
        end = format_srt_time(float(segment.get("end", 0.0)))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")
