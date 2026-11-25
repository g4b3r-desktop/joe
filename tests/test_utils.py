from datetime import datetime

from joe.utils import build_audio_output_path, build_srt, format_srt_time


def test_format_srt_time():
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(61.234) == "00:01:01,234"
    assert format_srt_time(3662.0) == "01:01:02,000"


def test_build_srt():
    result = build_srt([
        {"start": 0.2, "end": 1.4, "text": "Olá mundo."},
        {"start": 1.5, "end": 3.0, "text": "Tudo bem?"},
    ])
    assert "1\n00:00:00,200 --> 00:00:01,400\nOlá mundo." in result
    assert "2\n00:00:01,500 --> 00:00:03,000\nTudo bem?" in result


def test_build_srt_skips_empty_segments():
    assert build_srt([{"start": 0, "end": 1, "text": " " }]) == ""



def test_build_audio_output_path_creates_directory_and_uses_generic_name(tmp_path):
    timestamp = datetime(2026, 11, 18, 12, 30, 45)

    result = build_audio_output_path(tmp_path / "Joe", timestamp=timestamp)

    assert result == tmp_path / "Joe" / "joe_audio_20261118_123045.wav"
    assert result.parent.is_dir()
    assert not result.exists()


def test_build_audio_output_path_avoids_overwriting_existing_audio(tmp_path):
    timestamp = datetime(2026, 11, 18, 12, 30, 45)
    first = build_audio_output_path(tmp_path, timestamp=timestamp)
    first.touch()

    second = build_audio_output_path(tmp_path, timestamp=timestamp)

    assert second.name == "joe_audio_20261118_123045_001.wav"
    assert second != first


def test_build_audio_output_path_sanitizes_prefix(tmp_path):
    timestamp = datetime(2026, 11, 18, 12, 30, 45)

    result = build_audio_output_path(
        tmp_path, prefix="  Meu áudio / teste  ", timestamp=timestamp
    )

    assert result.name == "Meu_áudio_teste_20261118_123045.wav"



def test_write_bytes_atomically_replaces_complete_file(tmp_path):
    from joe.utils import write_bytes_atomically

    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"old audio")

    result = write_bytes_atomically(destination, b"RIFF-new-wave")

    assert result == destination
    assert destination.read_bytes() == b"RIFF-new-wave"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio.wav"]


def test_write_bytes_atomically_preserves_old_file_when_replace_fails(tmp_path, monkeypatch):
    import os

    import pytest

    from joe.utils import write_bytes_atomically

    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"old audio")

    def fail_replace(_source, _target):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated disk failure"):
        write_bytes_atomically(destination, b"new audio")

    assert destination.read_bytes() == b"old audio"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio.wav"]
