from __future__ import annotations

import numpy as np
import soundfile as sf

from joe.audio_chunks import merge_wav_files, split_text_into_chunks


def test_split_text_prefers_sentence_boundaries_and_respects_limit():
    text = (
        "Esta é a primeira frase. Esta é a segunda frase, um pouco maior. "
        "A terceira termina aqui."
    )
    chunks = split_text_into_chunks(text, max_chars=42)

    assert len(chunks) > 1
    assert all(len(chunk) <= 42 for chunk in chunks)
    assert " ".join(chunks) == text


def test_split_text_does_not_cut_words_when_sentence_is_too_long():
    text = "palavra " * 30
    chunks = split_text_into_chunks(text, max_chars=48)

    assert len(chunks) > 1
    assert all(len(chunk) <= 48 for chunk in chunks)
    assert " ".join(chunks) == text.strip()


def test_split_empty_text_returns_no_chunks():
    assert split_text_into_chunks("   ") == []


def test_split_rejects_unreasonably_small_limit():
    try:
        split_text_into_chunks("texto", max_chars=10)
    except ValueError as exc:
        assert "pelo menos 40" in str(exc)
    else:
        raise AssertionError("Era esperado um ValueError")


def test_merge_wav_files_joins_audio_with_short_silence(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "merged.wav"
    sample_rate = 16000
    sf.write(first, np.ones(1600, dtype=np.float32) * 0.1, sample_rate)
    sf.write(second, np.ones(800, dtype=np.float32) * 0.2, sample_rate)

    result = merge_wav_files([first, second], output, pause_ms=50)
    info = sf.info(result)

    assert result == output
    assert info.samplerate == sample_rate
    assert info.channels == 1
    assert info.frames == 1600 + 800 + 800


def test_merge_resamples_and_normalizes_mono_stereo(tmp_path):
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    output = tmp_path / "merged.wav"
    sf.write(mono, np.ones(800, dtype=np.float32) * 0.1, 8000)
    sf.write(stereo, np.ones((1600, 2), dtype=np.float32) * 0.2, 16000)

    merge_wav_files([mono, stereo], output, pause_ms=0)
    info = sf.info(output)

    assert info.samplerate == 8000
    assert info.channels == 1
    assert info.frames == 800 + 800
