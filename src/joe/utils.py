from __future__ import annotations

from datetime import datetime
from pathlib import Path
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


def build_audio_output_path(
    directory: str | Path,
    prefix: str = "joe_audio",
    timestamp: datetime | None = None,
) -> Path:
    """Create the output directory and return a unique, generic WAV filename."""
    output_dir = Path(directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_characters: list[str] = []
    for character in prefix.strip():
        if character.isalnum() or character in "._-":
            safe_characters.append(character)
        elif safe_characters and safe_characters[-1] != "_":
            safe_characters.append("_")
    safe_prefix = "".join(safe_characters).strip("._-") or "joe_audio"

    date_part = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{safe_prefix}_{date_part}.wav"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{safe_prefix}_{date_part}_{suffix:03d}.wav"
        suffix += 1
    return candidate


def format_vtt_time(seconds: float) -> str:
    """Convert seconds to the WebVTT timestamp format HH:MM:SS.mmm."""
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def build_vtt(segments: list[dict[str, Any]]) -> str:
    """Build a WebVTT document from timestamped transcript segments."""
    blocks: list[str] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = format_vtt_time(float(segment.get("start", 0.0)))
        end = format_vtt_time(float(segment.get("end", 0.0)))
        blocks.append(f"{start} --> {end}\n{text}")
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")
