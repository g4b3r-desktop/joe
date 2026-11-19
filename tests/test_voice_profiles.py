from __future__ import annotations

import json

from joe.voice_profiles import load_voice_profiles, serialize_voice_profiles


def test_load_voice_profiles_preserves_voice_language_and_normalization(tmp_path):
    voice = tmp_path / "Narradora.safetensors"
    raw = json.dumps(
        [{
            "id": "profile-1",
            "name": "Narradora PT",
            "voice_path": str(voice),
            "language": "portuguese",
            "normalization_enabled": False,
        }],
        ensure_ascii=False,
    )

    assert load_voice_profiles(raw) == {
        "profile-1": {
            "id": "profile-1",
            "name": "Narradora PT",
            "voice_path": str(voice),
            "language": "portuguese",
            "normalization_enabled": False,
        }
    }


def test_load_voice_profiles_accepts_mapping_and_ignores_invalid_entries():
    profiles = load_voice_profiles({
        "profile-1": {
            "id": "profile-1",
            "name": "English",
            "language": "english",
            "normalization_enabled": True,
        },
        "bad-language": {
            "id": "bad-language",
            "name": "Broken",
            "language": "klingon",
        },
        "missing-name": {"id": "missing-name", "language": "english"},
    })

    assert list(profiles) == ["profile-1"]


def test_load_voice_profiles_handles_invalid_storage():
    assert load_voice_profiles("not-json") == {}
    assert load_voice_profiles(None) == {}
    assert load_voice_profiles(["invalid"]) == {}


def test_serialize_and_reload_preserves_unicode_names_and_empty_default_voice():
    profiles = {
        "perfil-1": {
            "name": "Voz do João",
            "voice_path": "",
            "language": "portuguese",
            "normalization_enabled": True,
        }
    }

    loaded = load_voice_profiles(serialize_voice_profiles(profiles))

    assert loaded == {
        "perfil-1": {
            "id": "perfil-1",
            "name": "Voz do João",
            "voice_path": "",
            "language": "portuguese",
            "normalization_enabled": True,
        }
    }


def test_voice_profiles_can_store_safetensors_paths_without_touching_files():
    path = "/models/voices/narradora.safetensors"
    profile = {
        "id": "voice-state",
        "name": "Narradora",
        "voice_path": path,
        "language": "portuguese",
        "normalization_enabled": False,
    }

    loaded = load_voice_profiles(serialize_voice_profiles({"voice-state": profile}))

    assert loaded["voice-state"]["voice_path"] == path
