from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from joe.services.pocket_tts import _write_wav_atomically


def test_wav_write_is_atomic_and_produces_a_valid_file(tmp_path):
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"old audio")
    samples = np.zeros(1600, dtype=np.float32)

    result = _write_wav_atomically(sf, samples, 16_000, destination)

    assert result == destination
    assert sf.info(destination).frames == 1600
    assert destination.read_bytes().startswith(b"RIFF")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio.wav"]


def test_failed_wav_write_preserves_previous_audio_and_cleans_temp_file(tmp_path):
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"old audio")

    class FailingSoundFile:
        @staticmethod
        def write(path, *_args, **_kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated write failure"):
        _write_wav_atomically(
            FailingSoundFile(),
            np.zeros(10, dtype=np.float32),
            16_000,
            destination,
        )

    assert destination.read_bytes() == b"old audio"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio.wav"]
