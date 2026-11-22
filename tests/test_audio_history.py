from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from joe.audio_history import AudioHistoryStore


def make_audio(path: Path, content: bytes = b"RIFFfake-wave-data") -> Path:
    path.write_bytes(content)
    return path


def test_add_and_read_metadata_without_storing_text_by_default(tmp_path):
    audio = make_audio(tmp_path / "narracao.wav")
    store = AudioHistoryStore(tmp_path / "db" / "history.sqlite3")

    entry = store.add_generated_audio(
        audio,
        duration_seconds=20,
        generation_seconds=5,
        voice_path="/voices/voice.wav",
        voice_alias="Narradora",
        language="portuguese",
        backend="Google Colab",
    )

    assert entry.file_name == "narracao.wav"
    assert entry.size_bytes == len(b"RIFFfake-wave-data")
    assert entry.duration_seconds == 20
    assert entry.generation_seconds == 5
    assert entry.realtime_speed == 4
    assert entry.voice_label == "Narradora"
    assert entry.model_variant == ""
    assert entry.source_label == "Google Colab"
    assert entry.text_content is None
    assert store.get(entry.id) == entry


def test_text_is_stored_only_when_explicitly_supplied(tmp_path):
    audio = make_audio(tmp_path / "texto.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")

    entry = store.add_generated_audio(
        audio,
        duration_seconds=4,
        generation_seconds=2,
        text_content="Texto privado",
    )

    assert entry.text_content == "Texto privado"


def test_readding_same_path_updates_one_history_row(tmp_path):
    audio = make_audio(tmp_path / "same.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    first = store.add_generated_audio(audio, duration_seconds=2, generation_seconds=1)

    second = store.add_generated_audio(
        audio, duration_seconds=8, generation_seconds=4, voice_alias="Nova voz"
    )

    assert first.id == second.id
    assert len(store.list_entries()) == 1
    assert second.duration_seconds == 8
    assert second.voice_alias == "Nova voz"


def test_list_filters_by_search_voice_dates_and_duration(tmp_path):
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    old_audio = make_audio(tmp_path / "old.wav")
    new_audio = make_audio(tmp_path / "new.wav")
    store.add_generated_audio(
        old_audio,
        duration_seconds=3,
        generation_seconds=2,
        voice_alias="Narradora",
        created_at=datetime.fromisoformat("2026-10-01T10:00:00-03:00"),
    )
    store.add_generated_audio(
        new_audio,
        duration_seconds=12,
        generation_seconds=4,
        voice_alias="Documentário",
        created_at=datetime.fromisoformat("2026-11-10T10:00:00-03:00"),
    )

    result = store.list_entries(
        search="new",
        voice="Documentário",
        date_from="2026-11-01",
        date_to="2026-11-30",
        min_duration=10,
        max_duration=15,
    )

    assert len(result) == 1
    assert result[0].file_name == "new.wav"
    assert store.available_voices() == ["Documentário", "Narradora"]


def test_rename_moves_audio_and_updates_history_path(tmp_path):
    audio = make_audio(tmp_path / "old-name.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(audio, duration_seconds=1, generation_seconds=1)

    renamed = store.rename_audio(entry.id, "novo-nome")

    assert renamed.file_name == "novo-nome.wav"
    assert Path(renamed.audio_path).is_file()
    assert not audio.exists()


def test_rename_rejects_existing_destination_without_losing_original(tmp_path):
    audio = make_audio(tmp_path / "voice.wav")
    make_audio(tmp_path / "other.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(audio, duration_seconds=1, generation_seconds=1)

    with pytest.raises(FileExistsError):
        store.rename_audio(entry.id, "other.wav")

    assert audio.is_file()
    assert store.get(entry.id).audio_path == str(audio.resolve())


def test_delete_removes_audio_and_history_record(tmp_path):
    audio = make_audio(tmp_path / "delete.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(audio, duration_seconds=1, generation_seconds=1)

    assert store.delete(entry.id)
    assert not audio.exists()
    assert store.get(entry.id) is None


def test_delete_record_can_keep_original_audio_file(tmp_path):
    audio = make_audio(tmp_path / "keep.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(audio, duration_seconds=1, generation_seconds=1)

    assert store.delete(entry.id, delete_file=False)
    assert audio.is_file()
    assert store.get(entry.id) is None


def test_export_copies_audio_without_moving_original(tmp_path):
    audio = make_audio(tmp_path / "source.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(audio, duration_seconds=1, generation_seconds=1)

    exported = store.export_copy(entry.id, tmp_path / "exports" / "copy.wav")

    assert exported.read_bytes() == audio.read_bytes()
    assert audio.is_file()


def test_missing_audio_is_reported_instead_of_creating_a_broken_record(tmp_path):
    store = AudioHistoryStore(tmp_path / "history.sqlite3")

    with pytest.raises(FileNotFoundError):
        store.add_generated_audio(
            tmp_path / "missing.wav", duration_seconds=1, generation_seconds=1
        )


def test_voice_without_favorite_alias_uses_stable_filename_label(tmp_path):
    audio = make_audio(tmp_path / "character.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(
        audio,
        duration_seconds=3,
        generation_seconds=1,
        voice_path="/voices/character.wav",
    )

    assert entry.voice_label == "character"
    assert store.available_voices() == ["character"]
    assert store.list_entries(voice="character") == [entry]



def test_clear_text_content_preserves_audio_and_metadata(tmp_path):
    audio = make_audio(tmp_path / "keep.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    entry = store.add_generated_audio(
        audio,
        duration_seconds=18,
        generation_seconds=6,
        text_content="Texto privado para apagar",
    )

    assert store.count_text_entries() == 1
    assert store.clear_text_content() == 1

    updated = store.get(entry.id)
    assert updated is not None
    assert updated.text_content is None
    assert updated.duration_seconds == 18
    assert updated.generation_seconds == 6
    assert audio.is_file()
    assert store.count_text_entries() == 0


def test_clear_text_content_is_idempotent(tmp_path):
    store = AudioHistoryStore(tmp_path / "history.sqlite3")

    assert store.count_text_entries() == 0
    assert store.clear_text_content() == 0


def test_audio_history_stores_the_selected_model_variant(tmp_path):
    audio = make_audio(tmp_path / "model-choice.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")

    entry = store.add_generated_audio(
        audio,
        duration_seconds=12,
        generation_seconds=4,
        language="portuguese",
        model_variant="portuguese_24l",
    )

    assert entry.model_variant == "portuguese_24l"
    assert store.get(entry.id).model_variant == "portuguese_24l"


def test_existing_history_database_is_migrated_with_default_model_variant(tmp_path):
    import sqlite3

    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE audio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                voice_path TEXT NOT NULL DEFAULT '',
                voice_alias TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                generation_seconds REAL NOT NULL DEFAULT 0,
                realtime_speed REAL NOT NULL DEFAULT 0,
                backend TEXT NOT NULL DEFAULT 'Local',
                text_content TEXT
            )
            """
        )

    store = AudioHistoryStore(database)
    audio = make_audio(tmp_path / "legacy-new.wav")
    entry = store.add_generated_audio(audio, duration_seconds=2, generation_seconds=1)

    assert entry.model_variant == ""



def test_generation_settings_are_saved_and_survive_history_reads(tmp_path):
    audio = make_audio(tmp_path / "cuda-tts.wav")
    store = AudioHistoryStore(tmp_path / "history.sqlite3")
    settings = {
        "backend": "Pocket TTS local",
        "device": "cuda",
        "cpu_threads": 4,
        "quantization": False,
        "profile": "performance",
        "local_settings_applied": True,
    }

    entry = store.add_generated_audio(
        audio,
        duration_seconds=8,
        generation_seconds=2,
        generation_settings=settings,
    )

    assert '"device": "cuda"' in entry.generation_settings_json
    assert store.get(entry.id) == entry
