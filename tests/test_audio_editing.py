from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from joe.audio_editing import (
    export_trimmed_audio, export_without_silence, find_silence_regions,
    load_audio, remove_long_silences, waveform_peaks,
)


def test_waveform_peaks_returns_compact_envelope():
    audio = np.array([[0.1], [-0.5], [0.2], [0.8], [0.1]], dtype=np.float32)
    assert waveform_peaks(audio, point_count=2) == pytest.approx([0.5, 0.8])


def test_find_silence_regions_detects_only_long_quiet_intervals():
    signal = np.concatenate([
        np.full(500, 0.2, dtype=np.float32), np.zeros(800, dtype=np.float32),
        np.full(400, 0.2, dtype=np.float32), np.zeros(100, dtype=np.float32),
    ])
    assert find_silence_regions(
        signal, 1000, threshold_db=-40, minimum_silence_seconds=0.5
    ) == [(500, 1300)]


def test_remove_long_silences_keeps_short_pause_and_reports_removed_time():
    speech = np.full((500, 1), 0.2, dtype=np.float32)
    silence = np.zeros((1500, 1), dtype=np.float32)
    audio = np.concatenate([speech, silence, speech])
    reduced, removed, count = remove_long_silences(
        audio, 1000, threshold_db=-40,
        minimum_silence_seconds=0.5, keep_silence_seconds=0.2,
    )
    assert count == 1
    assert removed == pytest.approx(1.3)
    assert reduced.shape[0] == 1200
    assert np.max(np.abs(reduced)) == pytest.approx(0.2)


def test_remove_long_silences_handles_leading_and_trailing_pauses():
    audio = np.concatenate([
        np.zeros((1000, 1), dtype=np.float32),
        np.full((500, 1), 0.3, dtype=np.float32),
        np.zeros((1000, 1), dtype=np.float32),
    ])
    reduced, removed, count = remove_long_silences(
        audio, 1000, minimum_silence_seconds=0.5, keep_silence_seconds=0.1
    )
    assert count == 2
    assert removed == pytest.approx(1.8)
    assert reduced.shape[0] == 700


def test_trim_exports_new_wav_and_preserves_source(tmp_path):
    source, destination = tmp_path / "source.wav", tmp_path / "trimmed.wav"
    sf.write(source, np.linspace(-0.5, 0.5, 4000, dtype=np.float32), 1000)
    output = export_trimmed_audio(source, destination, 1.0, 2.5)
    trimmed, rate = load_audio(output)
    original, _ = load_audio(source)
    assert rate == 1000
    assert trimmed.shape[0] == 1500
    assert original.shape[0] == 4000
    assert source.is_file()


def test_silence_export_writes_copy_and_reports_durations(tmp_path):
    source, destination = tmp_path / "source.wav", tmp_path / "reduced.wav"
    audio = np.concatenate([
        np.full(500, 0.2, dtype=np.float32), np.zeros(1000, dtype=np.float32),
        np.full(500, 0.2, dtype=np.float32),
    ])
    sf.write(source, audio, 1000)
    output, removed, count, before, after = export_without_silence(
        source, destination, threshold_db=-40,
        minimum_silence_seconds=0.5, keep_silence_seconds=0.2,
    )
    assert output == destination
    assert count == 1
    assert removed == pytest.approx(0.8)
    assert before == pytest.approx(2.0)
    assert after == pytest.approx(1.2)
    assert source.is_file() and destination.is_file()


def test_trim_rejects_overwriting_original(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full(1000, 0.2, dtype=np.float32), 1000)
    with pytest.raises(ValueError, match="não pode ser sobrescrito"):
        export_trimmed_audio(source, source, 0.1, 0.8)


def test_silence_removal_rejects_audio_that_is_only_silence():
    with pytest.raises(ValueError, match="apenas silêncio"):
        remove_long_silences(
            np.zeros((1500, 1), dtype=np.float32), 1000,
            minimum_silence_seconds=0.5,
        )
