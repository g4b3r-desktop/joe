from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_HISTORY_ENTRIES = 200


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def normalize_audio_history_entry(value: Any) -> dict[str, Any] | None:
    """Validate a single history item and keep only non-sensitive metadata."""
    if not isinstance(value, dict):
        return None
    path = str(value.get("path", "")).strip()
    if not path:
        return None

    return {
        "path": str(Path(path).expanduser()),
        "created_at": str(value.get("created_at", "")).strip(),
        "duration_seconds": _number(value.get("duration_seconds")),
        "elapsed_seconds": _number(value.get("elapsed_seconds")),
        "character_count": max(0, int(_number(value.get("character_count")))),
        "language": str(value.get("language", "")).strip(),
        "voice_path": str(value.get("voice_path", "")).strip(),
        "voice_alias": str(value.get("voice_alias", "")).strip(),
        "backend": str(value.get("backend", "")).strip(),
        "source": str(value.get("source", "Texto → Áudio")).strip(),
    }


def load_audio_history(raw: Any, limit: int = MAX_HISTORY_ENTRIES) -> list[dict[str, Any]]:
    """Load recent audio records from QSettings JSON, tolerating old/corrupt data."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        entry = normalize_audio_history_entry(item)
        if entry is None:
            continue
        key = (entry["path"], entry["created_at"])
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
        if len(entries) >= max(1, limit):
            break
    return entries


def add_audio_history_entry(
    entries: list[dict[str, Any]],
    entry: dict[str, Any],
    limit: int = MAX_HISTORY_ENTRIES,
) -> list[dict[str, Any]]:
    """Prepend a normalized record and avoid duplicate path/timestamp entries."""
    normalized = normalize_audio_history_entry(entry)
    if normalized is None:
        raise ValueError("O registro do histórico precisa conter o caminho do áudio.")
    key = (normalized["path"], normalized["created_at"])
    remaining = [
        item for item in load_audio_history(entries, limit)
        if (item["path"], item["created_at"]) != key
    ]
    return [normalized, *remaining][:max(1, limit)]


def serialize_audio_history(entries: list[dict[str, Any]]) -> str:
    """Serialize validated metadata only; generated text is never stored here."""
    normalized = load_audio_history(entries)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def make_audio_history_entry(
    path: str | Path,
    *,
    duration_seconds: float = 0.0,
    elapsed_seconds: float = 0.0,
    character_count: int = 0,
    language: str = "",
    voice_path: str = "",
    voice_alias: str = "",
    backend: str = "",
    source: str = "Texto → Áudio",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a timestamped history record without retaining the original text."""
    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "path": str(Path(path).expanduser()),
        "created_at": timestamp,
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "character_count": character_count,
        "language": language,
        "voice_path": voice_path,
        "voice_alias": voice_alias,
        "backend": backend,
        "source": source,
    }
