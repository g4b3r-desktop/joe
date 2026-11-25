from __future__ import annotations

import pytest

from joe.temporary_voice_cache import TemporaryVoiceFileCache


def test_cache_reuses_snapshot_across_shortcut_processes_until_voice_changes(tmp_path):
    source = tmp_path / "reference voice.wav"
    cache_directory = tmp_path / "joe-cache"
    source.write_bytes(b"voice-version-one")

    first_process_cache = TemporaryVoiceFileCache(cache_directory)
    first_copy = first_process_cache.get(source)
    assert first_copy is not None
    assert first_copy != source
    assert first_copy.suffix == ".wav"
    assert first_copy.read_bytes() == b"voice-version-one"

    # The source may change or disappear; the shortcut keeps its selected snapshot.
    source.write_bytes(b"voice-version-two")
    source.unlink()
    first_process_cache.close()

    second_process_cache = TemporaryVoiceFileCache(cache_directory)
    try:
        second_copy = second_process_cache.get(source)
        assert second_copy == first_copy
        assert second_copy.read_bytes() == b"voice-version-one"
    finally:
        second_process_cache.close()

    # Closing Joe does not remove the active cache: the next shortcut uses it too.
    assert first_copy.is_file()


def test_cache_replaces_and_deletes_old_snapshot_when_voice_changes(tmp_path):
    first_source = tmp_path / "first.safetensors"
    second_source = tmp_path / "second.flac"
    cache_directory = tmp_path / "joe-cache"
    first_source.write_bytes(b"first voice")
    second_source.write_bytes(b"second voice")
    cache = TemporaryVoiceFileCache(cache_directory)
    try:
        first_copy = cache.get(first_source)
        assert first_copy is not None
        cache.invalidate()

        # A running worker may still be reading the previous file.
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

    # The newly selected voice remains cached for future shortcut launches.
    assert second_copy.is_file()


def test_cache_clears_active_snapshot_when_default_voice_is_selected(tmp_path):
    source = tmp_path / "reference.wav"
    source.write_bytes(b"voice")
    cache = TemporaryVoiceFileCache(tmp_path / "joe-cache")
    try:
        cached = cache.get(source)
        assert cached is not None and cached.is_file()

        assert cache.get(None) is None
        cache.cleanup_retired()
        assert not cached.exists()
        assert cache.get("") is None
    finally:
        cache.close()


def test_cache_removes_leftovers_from_interrupted_shortcut_runs(tmp_path):
    cache_directory = tmp_path / "joe-cache"
    cache_directory.mkdir()
    orphaned_voice = cache_directory / "voice-interrupted.wav"
    partial_copy = cache_directory / ".voice-interrupted.part"
    unrelated_file = cache_directory / "keep-this.txt"
    orphaned_voice.write_bytes(b"orphan")
    partial_copy.write_bytes(b"partial")
    unrelated_file.write_text("keep", encoding="utf-8")

    cache = TemporaryVoiceFileCache(cache_directory)
    try:
        assert not orphaned_voice.exists()
        assert not partial_copy.exists()
        assert unrelated_file.is_file()
    finally:
        cache.close()


def test_cache_rejects_missing_voice_file_when_no_snapshot_exists(tmp_path):
    cache = TemporaryVoiceFileCache(tmp_path / "joe-cache")
    try:
        with pytest.raises(FileNotFoundError, match="Arquivo de voz não encontrado"):
            cache.get(tmp_path / "missing.wav")
    finally:
        cache.close()
