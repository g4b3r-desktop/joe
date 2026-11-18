from __future__ import annotations

from pathlib import Path


def decide_playback_action(
    current_source: str,
    target_path: str | Path,
    *,
    is_playing: bool,
    is_paused: bool,
) -> str:
    """Choose whether to stop, resume, or switch the shared audio player."""
    target = str(Path(target_path).expanduser().resolve(strict=False))
    current = (
        str(Path(current_source).expanduser().resolve(strict=False))
        if current_source
        else ""
    )
    if current == target and is_playing:
        return "stop"
    if current == target and is_paused:
        return "resume"
    return "play"
