from __future__ import annotations

import pytest

from joe.tts_profiles import build_generation_settings, performance_profile_settings


def test_economy_profile_uses_cpu_int8_and_one_thread():
    assert performance_profile_settings(
        "economy", cpu_count=8, cuda_available=True
    ) == {"device": "cpu", "cpu_threads": 1, "quantization": True}


def test_balanced_profile_uses_half_available_cpu_threads():
    assert performance_profile_settings(
        "balanced", cpu_count=8, cuda_available=True
    ) == {"device": "cpu", "cpu_threads": 4, "quantization": True}


def test_performance_profile_prefers_cuda_when_available():
    assert performance_profile_settings(
        "performance", cpu_count=4, cuda_available=True
    ) == {"device": "cuda", "cpu_threads": 4, "quantization": False}


def test_performance_profile_falls_back_to_unquantized_cpu():
    assert performance_profile_settings(
        "performance", cpu_count=4, cuda_available=False
    ) == {"device": "cpu", "cpu_threads": 4, "quantization": False}


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="desconhecido"):
        performance_profile_settings("ultra", cpu_count=4)


def test_history_settings_record_effective_quantization_only():
    settings = build_generation_settings(
        device="cuda",
        cpu_threads=4,
        quantization=True,
        profile="performance",
    )
    assert settings["device"] == "cuda"
    assert settings["quantization"] is False
    assert settings["local_settings_applied"] is True


def test_remote_settings_are_marked_as_not_applied_locally():
    settings = build_generation_settings(
        device="cuda",
        cpu_threads=4,
        quantization=True,
        profile="performance",
        remote=True,
    )
    assert settings["device"] == "não aplicado (remoto)"
    assert settings["quantization"] is False
    assert settings["local_settings_applied"] is False
