from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from joe.voice_favorites import canonical_voice_path

SUPPORTED_LANGUAGES = frozenset({
    "portuguese", "english", "spanish", "french", "german", "italian", "dutch"
})


def _clean_dictionary(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(term).strip(): str(pronunciation).strip()
        for term, pronunciation in value.items()
        if str(term).strip() and str(pronunciation).strip()
    }


def normalize_voice_profile(value: Any) -> dict[str, Any]:
    """Return a validated, portable voice profile payload."""
    if not isinstance(value, dict):
        raise ValueError("O perfil precisa ser um objeto de configurações.")

    raw_language = str(value.get("language", "portuguese")).strip().lower()
    language = raw_language if raw_language in SUPPORTED_LANGUAGES else "portuguese"
    raw_voice_path = str(value.get("voice_path", "") or "").strip()
    voice_path = canonical_voice_path(raw_voice_path) if raw_voice_path else ""

    return {
        "voice_path": voice_path,
        "voice_alias": str(value.get("voice_alias", "") or "").strip(),
        "language": language,
        "normalize_text": bool(value.get("normalize_text", True)),
        "pronunciation_dictionary": _clean_dictionary(
            value.get("pronunciation_dictionary", {})
        ),
    }


def load_voice_profiles(raw: Any) -> dict[str, dict[str, Any]]:
    """Load saved profiles, ignoring malformed individual entries."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if not isinstance(raw, dict):
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for raw_name, raw_profile in raw.items():
        name = str(raw_name).strip()
        if not name:
            continue
        try:
            profiles[name] = normalize_voice_profile(raw_profile)
        except (TypeError, ValueError):
            continue
    return profiles


def serialize_voice_profiles(profiles: dict[str, dict[str, Any]]) -> str:
    """Serialize normalized profile names and settings with readable UTF-8 labels."""
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, profile in profiles.items():
        name = str(raw_name).strip()
        if not name:
            continue
        try:
            normalized[name] = normalize_voice_profile(profile)
        except (TypeError, ValueError):
            continue
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
