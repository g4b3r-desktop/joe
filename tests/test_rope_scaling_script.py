from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_cuda_rope_scaling import (
    DEFAULT_MODEL_CACHE,
    _configure_hf_token,
    measured_audio_duration,
    normalize_text,
    reduce_text_to_duration,
)


def test_normalize_text_collapses_whitespace_without_losing_content():
    assert normalize_text("  Olá,\n\n mundo!\tTudo bem?  ") == "Olá, mundo! Tudo bem?"


def test_reduce_text_to_duration_builds_progressive_sentence_safe_prefixes():
    source = " ".join(f"Frase{index} termina." for index in range(1, 81))

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


def test_default_model_cache_points_to_the_user_models_directory():
    assert DEFAULT_MODEL_CACHE == Path(
        "/mnt/46d6d246-1a5f-4388-b273-2ace8e54f409/"
        ".llms/.AUDIO/joe/.models/huggingface"
    )



def test_hf_token_prefers_existing_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-from-env")
    cache_token = tmp_path / "token"
    cache_token.write_text("token-from-file", encoding="utf-8")

    source = _configure_hf_token(tmp_path)

    assert source == "variável de ambiente"
    assert os.environ["HF_TOKEN"] == "token-from-env"


def test_hf_token_loads_from_custom_model_cache_without_printing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token_file = tmp_path / "token"
    token_file.write_text("hf_example_secret_value", encoding="utf-8")

    source = _configure_hf_token(tmp_path)

    assert source == "cache de modelos"
    assert os.environ["HF_TOKEN"] == "hf_example_secret_value"


def test_hf_token_falls_back_to_default_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    user_token = tmp_path / "user-cache" / "token"
    user_token.parent.mkdir()
    user_token.write_text("hf_user_cache_secret", encoding="utf-8")

    source = _configure_hf_token(model_cache, user_token_file=user_token)

    assert source == "cache padrão do usuário"
    assert os.environ["HF_TOKEN"] == "hf_user_cache_secret"



def test_hf_token_path_environment_variable_is_respected(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    token_file = tmp_path / "external-token"
    token_file.write_text("hf_external_path_secret", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_file))

    # Exercise the explicit-path fallback independently of an installed Hub
    # version or any token configured on the test runner.
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.get_token = lambda: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    source = _configure_hf_token(tmp_path)

    assert source == "HF_TOKEN_PATH"
    assert os.environ["HF_TOKEN"] == "hf_external_path_secret"
