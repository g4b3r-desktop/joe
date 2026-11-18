from __future__ import annotations

import json

from joe.voice_profiles import (
    load_voice_profiles,
    normalize_voice_profile,
    serialize_voice_profiles,
)


def test_normalize_profile_preserves_voice_language_and_dictionary(tmp_path):
    voice = tmp_path / "voice.wav"
    profile = normalize_voice_profile({
        "voice_path": str(voice),
        "voice_alias": "Narradora",
        "language": "english",
        "normalize_text": False,
        "pronunciation_dictionary": {"Joe": "Djou", "": "ignored"},
    })

    assert profile["voice_path"] == str(voice.resolve())
    assert profile["voice_alias"] == "Narradora"
    assert profile["language"] == "english"
    assert profile["normalize_text"] is False
    assert profile["pronunciation_dictionary"] == {"Joe": "Djou"}


def test_load_profiles_ignores_malformed_storage_and_entries():
    assert load_voice_profiles("not-json") == {}
    assert load_voice_profiles(["invalid"]) == {}

    profiles = load_voice_profiles({
        "Narrador": {"language": "portuguese", "normalize_text": True},
        "Bad": None,
        "   ": {"language": "english"},
    })
    assert list(profiles) == ["Narrador"]


def test_unknown_language_falls_back_to_portuguese():
    profile = normalize_voice_profile({"language": "unknown-language"})
    assert profile["language"] == "portuguese"


def test_serialize_and_reload_keeps_unicode_profile_name(tmp_path):
    voice = tmp_path / "voz.safetensors"
    profile = {
        "voice_path": str(voice),
        "voice_alias": "João",
        "language": "portuguese",
        "normalize_text": True,
        "pronunciation_dictionary": {"OpenAI": "Open êi ai"},
    }
    payload = serialize_voice_profiles({"Narração do João": profile})
    loaded = load_voice_profiles(payload)

    assert loaded["Narração do João"]["voice_alias"] == "João"
    assert loaded["Narração do João"]["pronunciation_dictionary"]["OpenAI"] == "Open êi ai"
