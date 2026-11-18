from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import soundfile as sf

from joe.audio_waveform import load_waveform_peaks


def test_waveform_is_normalized_and_detects_local_peak(tmp_path: Path):
    path = tmp_path / "pulse.wav"
    samples = np.zeros(8000, dtype=np.float32)
    samples[3900:4100] = 0.8
    sf.write(path, samples, 8000)

    peaks = load_waveform_peaks(path, max_points=80, block_frames=512)

    assert len(peaks) == 80
    assert max(peaks) == 1.0
    assert max(peaks[35:45]) == 1.0
    assert all(0.0 <= value <= 1.0 for value in peaks)


def test_waveform_reads_multichannel_audio(tmp_path: Path):
    path = tmp_path / "stereo.wav"
    samples = np.zeros((4000, 2), dtype=np.float32)
    samples[1000:1500, 1] = 0.4
    sf.write(path, samples, 16000)

    peaks = load_waveform_peaks(path, max_points=40)

    assert len(peaks) == 40
    assert max(peaks) == 1.0


def test_waveform_rejects_missing_files(tmp_path: Path):
    try:
        load_waveform_peaks(tmp_path / "missing.wav")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Era esperado FileNotFoundError")


def test_waveform_rejects_invalid_settings(tmp_path: Path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"")

    try:
        load_waveform_peaks(path, max_points=4)
    except ValueError as exc:
        assert "pelo menos 8" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError")
