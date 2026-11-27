from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _fake_model_with_shared_rope(rope):
    class FakeModel:
        def __init__(self):
            self.expansions = []

            def expand_kv_cache(model_state, sequence_length):
                self.expansions.append((model_state, sequence_length))
                return "expanded"

            self._expand_kv_cache = expand_kv_cache
            self.flow_lm = SimpleNamespace(
                transformer=SimpleNamespace(
                    rope=rope,
                    layers=[
                        SimpleNamespace(self_attn=SimpleNamespace(rope=rope))
                        for _ in range(24)
                    ],
                )
            )

    return FakeModel()


def test_cached_rope_matches_pocket_tts_reference_and_deduplicates_shared_module():
    from pocket_tts.modules.rope import RotaryEmbedding, apply_rope

    from joe.pocket_tts_rope_cache import install_cuda_rope_cache

    torch.manual_seed(41)
    rope = RotaryEmbedding(max_period=10_000.0)
    original_forward = rope.forward
    model = _fake_model_with_shared_rope(rope)
    cache = install_cuda_rope_cache(model, torch, initial_positions=32)

    q = torch.randn(2, 7, 4, 8)
    k = torch.randn(2, 7, 2, 8)
    for offset in (0, 5, torch.tensor(9, dtype=torch.long)):
        expected_q, expected_k = apply_rope(q, k, offset, max_period=10_000.0)
        actual_q, actual_k = rope(q, k, offset=offset)
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)

    # Pocket TTS 4.0.1 shares one RotaryEmbedding among all 24 layers.
    assert len(cache._caches) == 1
    assert cache.calls == 3
    assert cache.fallbacks == 0
    assert cache.capacity_positions == 32
    assert cache.allocated_bytes > 0

    cache.close()
    assert rope.forward == original_forward
    assert model._expand_kv_cache is not None


def test_cached_rope_grows_with_kv_cache_and_restores_model_method():
    from pocket_tts.modules.rope import RotaryEmbedding, apply_rope

    from joe.pocket_tts_rope_cache import install_cuda_rope_cache

    rope = RotaryEmbedding(max_period=10_000.0)
    original_forward = rope.forward
    model = _fake_model_with_shared_rope(rope)
    original_expand = model._expand_kv_cache
    cache = install_cuda_rope_cache(model, torch, initial_positions=8)

    q = torch.randn(1, 3, 2, 8)
    k = torch.randn(1, 3, 2, 8)
    rope(q, k, offset=0)
    assert cache.capacity_positions == 8

    state = object()
    assert model._expand_kv_cache(state, 40) == "expanded"
    assert model.expansions == [(state, 40)]
    assert cache.capacity_positions == 40

    offset = torch.tensor(37, dtype=torch.long)
    expected_q, expected_k = apply_rope(q, k, offset, max_period=10_000.0)
    actual_q, actual_k = rope(q, k, offset=offset)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    assert cache.calls == 2
    assert cache.fallbacks == 0

    cache.close()
    assert rope.forward == original_forward
    assert model._expand_kv_cache == original_expand


def test_cached_rope_uses_original_implementation_for_out_of_range_python_offset():
    from pocket_tts.modules.rope import RotaryEmbedding, apply_rope

    from joe.pocket_tts_rope_cache import install_cuda_rope_cache

    rope = RotaryEmbedding(max_period=10_000.0)
    model = _fake_model_with_shared_rope(rope)
    cache = install_cuda_rope_cache(model, torch, initial_positions=8)
    q = torch.randn(1, 2, 2, 8)
    k = torch.randn(1, 2, 2, 8)

    expected_q, expected_k = apply_rope(q, k, offset=20, max_period=10_000.0)
    actual_q, actual_k = rope(q, k, offset=20)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    assert cache.calls == 0
    assert cache.fallbacks == 1

    cache.close()


def test_rope_cache_benchmark_report_mentions_speed_audio_and_cache_metrics():
    from joe.audio_performance import (
        TTSCudaRopeCacheBenchmarkResult,
        format_cuda_rope_cache_benchmark_report,
    )

    result = TTSCudaRopeCacheBenchmarkResult(
        model_load_seconds=4.2,
        voice_prepare_seconds=0.1,
        baseline_generation_times=(6.0, 5.9, 6.1),
        cached_generation_times=(5.3, 5.2, 5.4),
        baseline_audio_durations_seconds=(6.8, 6.8, 6.8),
        cached_audio_durations_seconds=(6.8, 6.8, 6.8),
        audio_shapes_match=True,
        audio_max_abs_difference=1e-7,
        audio_mean_abs_difference=1e-8,
        audio_rms_difference=2e-8,
        baseline_peak_vram_mb=1454,
        cached_peak_vram_mb=1458,
        warmup_runs=2,
        cache_positions=8192,
        cache_calls=255,
        cache_fallbacks=0,
        cache_bytes=4 * 1024 * 1024,
    )

    report = format_cuda_rope_cache_benchmark_report(result)

    assert "RoPE original" in report
    assert "RoPE com cache" in report
    assert "Diferença absoluta máxima" in report
    assert "Chamadas atendidas pelo cache: 255" in report
    assert "Fallbacks para implementação original: 0" in report
    assert "4,00 MB" in report
