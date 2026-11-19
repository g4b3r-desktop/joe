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

def decide_clipboard_shortcut_action(
    *,
    dialog_open: bool,
    generation_active: bool,
) -> str:
    """Decide whether a clipboard shortcut should focus, wait, or start a dialog."""
    if dialog_open:
        return "focus"
    if generation_active:
        return "busy"
    return "start"
