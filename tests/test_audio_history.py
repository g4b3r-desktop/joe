from __future__ import annotations

import wave
from pathlib import Path

from joe.audio_history import AudioHistoryStore


def make_wave(path: Path, seconds: float = 1.5) -> Path:
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8000)
        audio_file.writeframes(b"\x00\x00" * int(8000 * seconds))
    return path


def test_add_stores_audio_metadata_without_source_text_by_default(tmp_path: Path):
    audio = make_wave(tmp_path / "generated.wav")
    store = AudioHistoryStore(tmp_path / "data" / "history.sqlite3")

    entry_id = store.add(
        audio,
        voice_path=tmp_path / "voice.wav",
        voice_alias="Narradora",
        backend="Local",
        elapsed_seconds=0.5,
        chars_per_second=120,
    )

    entry = store.get(entry_id)
    assert entry is not None
    assert entry.path == audio.resolve()
    assert entry.duration_seconds == 1.5
    assert entry.size_bytes == audio.stat().st_size
    assert entry.voice_alias == "Narradora"
    assert entry.backend == "Local"
    assert entry.realtime_speed == 3.0
    assert entry.chars_per_second == 120
    assert entry.text is None


def test_source_text_is_saved_only_when_provided(tmp_path: Path):
    audio = make_wave(tmp_path / "text.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")

    entry_id = store.add(audio, text="Texto original autorizado")
    entry = store.get(entry_id)

    assert entry is not None
    assert entry.text == "Texto original autorizado"


def test_search_matches_filename_voice_alias_and_backend(tmp_path: Path):
    first = make_wave(tmp_path / "narration.wav")
    second = make_wave(tmp_path / "tutorial.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    store.add(first, voice_alias="Narradora", backend="Local")
    store.add(second, voice_alias="Tutorial", backend="Google Colab")

    assert [item.path.name for item in store.list_entries(search="narration")] == [
        "narration.wav"
    ]
    assert [item.voice_alias for item in store.list_entries(search="narradora")] == [
        "Narradora"
    ]
    assert [item.backend for item in store.list_entries(search="colab")] == [
        "Google Colab"
    ]


def test_removing_history_entry_does_not_delete_wav(tmp_path: Path):
    audio = make_wave(tmp_path / "keep.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry_id = store.add(audio)

    assert store.remove_entry(entry_id)
    assert store.get(entry_id) is None
    assert audio.is_file()


def test_missing_or_non_wav_files_are_rejected(tmp_path: Path):
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    missing = tmp_path / "missing.wav"
    mp3 = tmp_path / "audio.mp3"
    mp3.write_bytes(b"not a wav")

    try:
        store.add(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Esperava FileNotFoundError")

    try:
        store.add(mp3)
    except ValueError:
        pass
    else:
        raise AssertionError("Esperava ValueError")
