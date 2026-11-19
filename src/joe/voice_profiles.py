from __future__ import annotations

import json
from typing import Any

SUPPORTED_VOICE_PROFILE_LANGUAGES = {
    "portuguese",
    "english",
    "spanish",
    "french",
    "italian",
    "german",
    "dutch",
}


def _normalize_voice_profile(raw: Any) -> dict[str, str | bool] | None:
    if not isinstance(raw, dict):
        return None
    profile_id = str(raw.get("id", "")).strip()
    name = str(raw.get("name", "")).strip()
    language = str(raw.get("language", "portuguese")).strip()
    if not profile_id or not name or language not in SUPPORTED_VOICE_PROFILE_LANGUAGES:
        return None
    return {
        "id": profile_id,
        "name": name,
        "voice_path": str(raw.get("voice_path", "")).strip(),
        "language": language,
        "normalization_enabled": bool(raw.get("normalization_enabled", True)),
    }


def load_voice_profiles(raw: Any) -> dict[str, dict[str, str | bool]]:
    """Load named TTS profiles from QSettings, ignoring malformed entries."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return {}

    profiles: dict[str, dict[str, str | bool]] = {}
    for item in raw:
        profile = _normalize_voice_profile(item)
        if profile is not None:
            profiles[str(profile["id"])] = profile
    return profiles


def serialize_voice_profiles(
    profiles: dict[str, dict[str, Any]],
) -> str:
    """Serialize valid voice profiles without touching other Joe settings."""
    normalized = []
    for profile_id, raw in profiles.items():
        candidate = dict(raw)
        candidate.setdefault("id", profile_id)
        profile = _normalize_voice_profile(candidate)
        if profile is not None:
            normalized.append(profile)
    normalized.sort(key=lambda item: str(item["name"]).casefold())
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
