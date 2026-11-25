from __future__ import annotations

import json

from joe.voice_favorites import (
    canonical_voice_path,
    load_voice_favorites,
    serialize_voice_favorites,
)


def test_load_favorites_normalizes_paths_and_aliases(tmp_path):
    voice = tmp_path / "voice.wav"
    raw = json.dumps({str(voice): "  Narradora  "}, ensure_ascii=False)

    favorites = load_voice_favorites(raw)

    assert favorites == {canonical_voice_path(voice): "Narradora"}


def test_load_favorites_uses_filename_when_alias_is_empty(tmp_path):
    voice = tmp_path / "calm_voice.safetensors"

    assert load_voice_favorites({str(voice): " "}) == {
        canonical_voice_path(voice): "calm_voice"
    }


def test_load_favorites_ignores_invalid_storage():
    assert load_voice_favorites("not-json") == {}
    assert load_voice_favorites(["not", "a", "mapping"]) == {}


def test_serialize_and_reload_preserves_unicode_alias(tmp_path):
    voice = tmp_path / "voz.wav"
    serialized = serialize_voice_favorites({str(voice): "Voz do João"})

    assert load_voice_favorites(serialized) == {
        canonical_voice_path(voice): "Voz do João"
    }
