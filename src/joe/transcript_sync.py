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


def _format_vtt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def build_vtt(segments: list[dict[str, Any]]) -> str:
    """Build a WebVTT subtitle document from timed transcript segments."""
    normalized = normalize_timed_segments(segments)
    cues = ["WEBVTT"]
    for segment in normalized:
        start = _format_vtt_time(segment["start"])
        end = _format_vtt_time(segment["end"])
        cues.extend(["", f"{start} --> {end}", segment["text"]])
    return "\n".join(cues) + "\n"
