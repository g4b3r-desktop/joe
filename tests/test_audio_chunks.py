from __future__ import annotations

import numpy as np
import soundfile as sf

from joe.audio_chunks import (
    merge_wav_files,
    prepare_clipboard_chunks,
    prepare_tts_chunks,
    split_text_into_chunks,
    split_text_into_phrases,
)


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


def test_split_text_into_phrases_preserves_sentence_boundaries_and_punctuation():
    text = "Primeira frase. Segunda frase! Terceira frase?"
    phrases = split_text_into_phrases(text, max_chars=80)

    assert phrases == ["Primeira frase.", "Segunda frase!", "Terceira frase?"]
    assert " ".join(phrases) == text


def test_split_text_into_phrases_keeps_long_sentences_word_safe():
    text = "uma palavra " * 20 + ". Outra frase."
    phrases = split_text_into_phrases(text, max_chars=48)

    assert len(phrases) > 2
    assert all(len(phrase) <= 48 for phrase in phrases)
    assert " ".join(phrases) == text.strip()


def test_split_text_into_phrases_handles_newlines_and_empty_text():
    assert split_text_into_phrases("  ") == []
    assert split_text_into_phrases("Linha um\nLinha dois.") == [
        "Linha um", "Linha dois."
    ]


def test_split_text_into_phrases_rejects_small_limit():
    try:
        split_text_into_phrases("texto", max_chars=20)
    except ValueError as exc:
        assert "pelo menos 40" in str(exc)
    else:
        raise AssertionError("Era esperado um ValueError")



def test_clipboard_non_interactive_respects_selected_character_limit():
    text = "Primeira frase. Segunda frase. " + ("Texto longo. " * 80)

    chunks = prepare_clipboard_chunks(
        text,
        interactive_phrases=False,
        max_phrase_chars=280,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 280 for chunk in chunks)
    assert " ".join(chunks) == text.strip()


def test_tts_full_text_option_keeps_one_chunk_when_api_limit_allows():
    text = "Primeira frase. " + ("Texto longo. " * 80)

    chunks = prepare_tts_chunks(
        text,
        max_chunk_chars=0,
    )

    assert chunks == [text.strip()]


def test_tts_full_text_option_still_respects_remote_api_character_limit():
    text = "palavra " * 3_000

    chunks = prepare_tts_chunks(
        text,
        max_chunk_chars=0,
        max_text_chars=20_000,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 20_000 for chunk in chunks)
    assert " ".join(chunks) == text.strip()


def test_clipboard_interactive_preserves_individual_sentences():
    text = "Primeira frase. Segunda frase. Terceira frase."

    chunks = prepare_clipboard_chunks(
        text,
        interactive_phrases=True,
        max_phrase_chars=560,
    )

    assert chunks == [
        "Primeira frase.",
        "Segunda frase.",
        "Terceira frase.",
    ]


def test_clipboard_remote_text_over_api_limit_is_split_only_at_limit():
    text = "palavra " * 3_000

    chunks = prepare_clipboard_chunks(
        text,
        interactive_phrases=False,
        max_phrase_chars=0,
        max_text_chars=20_000,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 20_000 for chunk in chunks)
    assert " ".join(chunks) == text.strip()
