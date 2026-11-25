from __future__ import annotations

from pathlib import Path

import pytest

from joe.temporary_voice_cache import TemporaryVoiceFileCache


def test_cache_copies_voice_once_and_reuses_it_until_selection_changes(tmp_path):
    source = tmp_path / "reference voice.wav"
    source.write_bytes(b"voice-version-one")
    cache = TemporaryVoiceFileCache()
    try:
        first_copy = cache.get(source)
        assert first_copy is not None
        assert first_copy != source
        assert first_copy.suffix == ".wav"
        assert first_copy.read_bytes() == b"voice-version-one"

        # Editing the source file does not change the active temporary snapshot.
        source.write_bytes(b"voice-version-two")
        second_copy = cache.get(source)
        assert second_copy == first_copy
        assert second_copy.read_bytes() == b"voice-version-one"
    finally:
        cache.close()

    assert not first_copy.exists()


def test_cache_replaces_voice_snapshot_after_invalidation(tmp_path):
    first_source = tmp_path / "first.safetensors"
    second_source = tmp_path / "second.flac"
    first_source.write_bytes(b"first voice")
    second_source.write_bytes(b"second voice")
    cache = TemporaryVoiceFileCache()
    try:
        first_copy = cache.get(first_source)
        assert first_copy is not None
        cache.invalidate()

        # The old copy stays available until the caller says the worker is idle.
        assert first_copy.is_file()
        cache.cleanup_retired()
        assert not first_copy.exists()

        second_copy = cache.get(second_source)
        assert second_copy is not None
        assert second_copy != first_copy
        assert second_copy.suffix == ".flac"
        assert second_copy.read_bytes() == b"second voice"
    finally:
        cache.close()


def test_cache_returns_none_for_default_voice(tmp_path):
    cache = TemporaryVoiceFileCache()
    try:
        assert cache.get(None) is None
        assert cache.get("") is None
    finally:
        cache.close()


def test_cache_rejects_missing_voice_file(tmp_path):
    cache = TemporaryVoiceFileCache()
    try:
        with pytest.raises(FileNotFoundError, match="Arquivo de voz não encontrado"):
            cache.get(tmp_path / "missing.wav")
    finally:
        cache.close()
