from __future__ import annotations

from math import isfinite
from typing import Any


def normalize_timed_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return valid timed transcript segments, preserving all original metadata."""
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            continue
        if not text or not isfinite(start) or not isfinite(end):
            continue
        if start < 0 or end <= start:
            continue
        normalized.append({**segment, "text": text, "start": start, "end": end})
    return sorted(normalized, key=lambda item: (item["start"], item["end"]))


def active_segment_index(segments: list[dict[str, Any]], position_ms: int) -> int | None:
    """Find the segment containing the playback position, or None during a gap."""
    position = max(0, int(position_ms)) / 1000.0
    for index, segment in enumerate(segments):
        if segment["start"] <= position < segment["end"]:
            return index
        if segment["start"] > position:
            break
    return None
