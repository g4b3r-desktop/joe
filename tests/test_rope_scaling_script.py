from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.benchmark_cuda_rope_scaling import (
    measured_audio_duration,
    normalize_text,
    reduce_text_to_duration,
)


def test_normalize_text_collapses_whitespace_without_losing_content():
    assert normalize_text("  Olá,\n\n mundo!\tTudo bem?  ") == "Olá, mundo! Tudo bem?"


def test_reduce_text_to_duration_builds_progressive_sentence_safe_prefixes():
    source = " ".join(f"Frase {index} termina." for index in range(1, 81))

    short = reduce_text_to_duration(source, 30, 120)
    medium = reduce_text_to_duration(source, 60, 120)
    full = reduce_text_to_duration(source, 120, 120)

    assert full == normalize_text(source)
    assert len(short.split()) < len(medium.split()) < len(full.split())
    assert short.endswith(".")
    assert medium.endswith(".")
    assert len(short.split()) == 40
    assert len(medium.split()) == 80


def test_reduce_text_to_duration_cuts_long_sentence_at_word_boundary():
    source = " ".join(f"palavra{index}" for index in range(100))

    shortened = reduce_text_to_duration(source, 30, 120)

    assert len(shortened.split()) == 25
    assert shortened.endswith(".")


@pytest.mark.parametrize(
    ("target", "reference"),
    [(0, 120), (-1, 120), (30, 0), (30, -10)],
)
def test_reduce_text_to_duration_rejects_invalid_durations(target, reference):
    with pytest.raises(ValueError):
        reduce_text_to_duration("Texto de teste.", target, reference)


def test_measured_audio_duration_averages_both_paired_modes():
    result = SimpleNamespace(
        baseline_audio_durations_seconds=(29.0, 31.0),
        cached_audio_durations_seconds=(30.0, 32.0),
    )
    assert measured_audio_duration(result) == 30.5


def test_measured_audio_duration_returns_zero_when_no_durations_exist():
    result = SimpleNamespace(
        baseline_audio_durations_seconds=(),
        cached_audio_durations_seconds=(),
    )
    assert measured_audio_duration(result) == 0.0
