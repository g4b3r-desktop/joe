from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_voice_path(value: str | Path) -> str:
    """Return a stable, expanded path for comparing and storing voice files."""
    path = Path(value).expanduser()
    try:
        return str(path.resolve(strict=False))
    except OSError:
        return str(path)


def load_voice_favorites(raw: Any) -> dict[str, str]:
    """Load favorite voices from QSettings JSON, ignoring malformed entries."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if not isinstance(raw, dict):
        return {}

    favorites: dict[str, str] = {}
    for raw_path, raw_alias in raw.items():
        path_text = str(raw_path).strip()
        alias = str(raw_alias).strip()
        if not path_text:
            continue
        canonical_path = canonical_voice_path(path_text)
        favorites[canonical_path] = alias or Path(canonical_path).stem
    return favorites


def serialize_voice_favorites(favorites: dict[str, str]) -> str:
    """Serialize normalized favorite paths and readable aliases."""
    normalized = {
        canonical_voice_path(path): (str(alias).strip() or Path(path).stem)
        for path, alias in favorites.items()
        if str(path).strip()
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
