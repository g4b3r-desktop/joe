from __future__ import annotations

import json
from typing import Any


def normalize_voice_profile(value: Any) -> dict[str, Any] | None:
    """Validate a profile and discard unknown or malformed fields."""
    if not isinstance(value, dict):
        return None
    dictionary = value.get("pronunciation_dictionary", {})
    if not isinstance(dictionary, dict):
        dictionary = {}
    return {
        "voice_path": str(value.get("voice_path", "")).strip(),
        "language": str(value.get("language", "portuguese")).strip() or "portuguese",
        "tts_normalization_enabled": bool(value.get("tts_normalization_enabled", True)),
        "pronunciation_dictionary": {
            str(term).strip(): str(pronunciation).strip()
            for term, pronunciation in dictionary.items()
            if str(term).strip() and str(pronunciation).strip()
        },
        "voice_alias": str(value.get("voice_alias", "")).strip(),
    }


def load_voice_profiles(raw: Any) -> dict[str, dict[str, Any]]:
    """Load named voice profiles from QSettings JSON, ignoring invalid entries."""
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
        profile = normalize_voice_profile(raw_profile)
        if name and profile is not None:
            profiles[name] = profile
    return profiles


def serialize_voice_profiles(profiles: dict[str, dict[str, Any]]) -> str:
    """Serialize validated profile settings without unrelated app credentials."""
    normalized = {
        str(name).strip(): profile
        for name, value in profiles.items()
        if str(name).strip() and (profile := normalize_voice_profile(value)) is not None
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def upsert_voice_profile(
    profiles: dict[str, dict[str, Any]],
    name: str,
    profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return a new profile mapping with one named profile inserted or replaced."""
    clean_name = str(name).strip()
    normalized = normalize_voice_profile(profile)
    if not clean_name or normalized is None:
        raise ValueError("Informe um nome e uma configuração de perfil válida.")
    result = dict(profiles)
    result[clean_name] = normalized
    return result


def remove_voice_profile(
    profiles: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, dict[str, Any]]:
    """Return profiles without the selected name."""
    return {key: value for key, value in profiles.items() if key != name}
