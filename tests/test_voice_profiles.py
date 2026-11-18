from __future__ import annotations

import json

from joe.voice_profiles import (
    load_voice_profiles,
    normalize_voice_profile,
    remove_voice_profile,
    serialize_voice_profiles,
    upsert_voice_profile,
)


def sample_profile():
    return {
        "voice_path": "/voices/narradora.wav",
        "language": "portuguese",
        "tts_normalization_enabled": False,
        "pronunciation_dictionary": {"Joe": "djo"},
        "voice_alias": "Narradora",
    }


def test_profile_round_trip_preserves_language_voice_and_pronunciation():
    profiles = {"Documentário": sample_profile()}
    assert load_voice_profiles(serialize_voice_profiles(profiles)) == profiles


def test_profile_loader_ignores_invalid_json_and_entries():
    assert load_voice_profiles("not-json") == {}
    assert load_voice_profiles(json.dumps({"": {}, "bad": None})) == {}


def test_profile_normalizer_filters_invalid_dictionary_entries():
    profile = normalize_voice_profile({
        **sample_profile(),
        "pronunciation_dictionary": {" GPU ": "gê pê u", "": "x", "Joe": " "},
        "unexpected": "ignored",
    })
    assert profile["pronunciation_dictionary"] == {"GPU": "gê pê u"}
    assert "unexpected" not in profile


def test_upsert_and_remove_do_not_mutate_original_mapping():
    original = {"Narrador": sample_profile()}
    updated = upsert_voice_profile(original, "Infantil", sample_profile())
    removed = remove_voice_profile(updated, "Narrador")
    assert list(original) == ["Narrador"]
    assert set(updated) == {"Narrador", "Infantil"}
    assert set(removed) == {"Infantil"}
