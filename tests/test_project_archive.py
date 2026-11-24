from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from joe.project_archive import (
    JOE_AUDIO_PROJECT_FORMAT,
    JOE_AUDIO_PROJECT_VERSION,
    save_joe_audio_project,
)


def test_joe_audio_project_contains_original_text_and_all_audio(tmp_path: Path):
    full_audio = tmp_path / "complete.wav"
    first_phrase = tmp_path / "phrase_001.wav"
    second_phrase = tmp_path / "phrase_002.wav"
    full_audio.write_bytes(b"complete-wave-data")
    first_phrase.write_bytes(b"first-wave-data")
    second_phrase.write_bytes(b"second-wave-data")
    destination = tmp_path / "speech.joeaudio"

    saved = save_joe_audio_project(
        destination,
        original_text="Texto original\ncom quebra de linha.",
        chunk_texts_by_index={
            1: "Primeira frase.",
            2: "Segunda frase.",
        },
        full_audio_path=full_audio,
        phrase_audio_paths={1: first_phrase, 2: second_phrase},
        recognized_text="Texto reconhecido.",
        recognized_segments=[
            {"start": 0.0, "end": 1.2, "text": "Texto reconhecido."}
        ],
        performance={
            "elapsed_seconds": 2.5,
            "audio_duration_seconds": 1.2,
            "character_count": 18,
            "remote": False,
        },
    )

    assert saved == destination
    assert zipfile.is_zipfile(destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["format"] == JOE_AUDIO_PROJECT_FORMAT
        assert manifest["format_version"] == JOE_AUDIO_PROJECT_VERSION
        assert manifest["application"] == "Joe Audio Studio"
        assert manifest["audio"]["full"] == "audio/full.wav"
        assert len(manifest["audio"]["phrases"]) == 2
        assert archive.read("texts/original.txt").decode("utf-8") == (
            "Texto original\ncom quebra de linha."
        )
        phrases = json.loads(archive.read("texts/phrases.json").decode("utf-8"))
        assert phrases == [
            {"index": 1, "text": "Primeira frase."},
            {"index": 2, "text": "Segunda frase."},
        ]
        assert archive.read("texts/recognized.txt").decode("utf-8") == (
            "Texto reconhecido."
        )
        assert archive.read("audio/full.wav") == b"complete-wave-data"
        assert archive.read("audio/phrases/001.wav") == b"first-wave-data"
        assert archive.read("audio/phrases/002.wav") == b"second-wave-data"
        assert archive.getinfo("audio/full.wav").compress_type == zipfile.ZIP_DEFLATED


def test_joe_audio_project_can_be_saved_without_phrase_or_recognition_data(
    tmp_path: Path,
):
    full_audio = tmp_path / "complete.wav"
    full_audio.write_bytes(b"wave-data")
    destination = tmp_path / "minimal.joeaudio"

    save_joe_audio_project(
        destination,
        original_text="Somente o texto original.",
        full_audio_path=full_audio,
    )

    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["audio"]["phrases"] == []
        assert "recognized" not in manifest["texts"]
        assert "recognized_segments" not in manifest["texts"]
        assert json.loads(archive.read("texts/phrases.json")) == []


def test_joe_audio_project_does_not_replace_destination_if_audio_is_missing(
    tmp_path: Path,
):
    destination = tmp_path / "existing.joeaudio"
    destination.write_bytes(b"keep this existing file")

    with pytest.raises(FileNotFoundError):
        save_joe_audio_project(
            destination,
            original_text="Texto",
            full_audio_path=tmp_path / "missing.wav",
        )

    assert destination.read_bytes() == b"keep this existing file"
