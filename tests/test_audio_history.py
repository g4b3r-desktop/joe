from __future__ import annotations

import json

from joe.audio_history import (
    add_audio_history_entry,
    load_audio_history,
    make_audio_history_entry,
    serialize_audio_history,
)


def test_history_round_trip_keeps_metadata_without_text(tmp_path):
    audio = tmp_path / "generated.wav"
    entry = make_audio_history_entry(
        audio,
        duration_seconds=12.5,
        elapsed_seconds=4.0,
        character_count=240,
        language="portuguese",
        voice_alias="Narradora",
        backend="Local",
        created_at="2026-11-18T12:00:00-03:00",
    )

    restored = load_audio_history(serialize_audio_history([entry]))

    assert restored == [entry]
    assert "text" not in restored[0]


def test_history_ignores_malformed_storage_and_entries():
    assert load_audio_history("not-json") == []
    assert load_audio_history(json.dumps([None, {}, {"path": ""}])) == []


def test_history_is_newest_first_and_deduplicates_same_record(tmp_path):
    first = make_audio_history_entry(
        tmp_path / "first.wav", created_at="2026-11-17T10:00:00-03:00"
    )
    second = make_audio_history_entry(
        tmp_path / "second.wav", created_at="2026-11-18T10:00:00-03:00"
    )

    result = add_audio_history_entry([first], second)
    result = add_audio_history_entry(result, second)

    assert result == [second, first]


def test_history_is_bounded(tmp_path):
    entries = [
        make_audio_history_entry(
            tmp_path / f"{index}.wav",
            created_at=f"2026-11-18T10:{index:02d}:00-03:00",
        )
        for index in range(12)
    ]

    assert len(load_audio_history(entries, limit=5)) == 5


def test_history_serialization_does_not_store_original_text(tmp_path):
    entry = make_audio_history_entry(tmp_path / "private.wav")
    payload = serialize_audio_history([{**entry, "text": "conteúdo particular"}])

    assert "conteúdo particular" not in payload
