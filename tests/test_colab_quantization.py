import sys
from types import ModuleType, SimpleNamespace

import pytest

from colab.tts_runtime import apply_tts_int8_quantization


def test_colab_quantization_opt_out_returns_before_importing_backend(monkeypatch):
    monkeypatch.delitem(sys.modules, "pocket_tts.quantization", raising=False)

    result = apply_tts_int8_quantization(
        SimpleNamespace(flow_lm=object()),
        "cuda",
        requested=False,
    )

    assert result is None
    assert "pocket_tts.quantization" not in sys.modules


def test_colab_cuda_int8_uses_torchao_only_after_explicit_request(monkeypatch):
    calls = []
    fake_pocket_tts = ModuleType("pocket_tts")
    fake_quantization = ModuleType("pocket_tts.quantization")
    fake_quantization.RECOMMENDED_CONFIG = {"attention", "ffn"}
    fake_quantization._get_backend = lambda: "torchao"
    fake_quantization.apply_dynamic_int8 = (
        lambda flow_lm, groups: calls.append((flow_lm, groups))
    )
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    monkeypatch.setitem(sys.modules, "pocket_tts.quantization", fake_quantization)

    flow_lm = object()
    backend = apply_tts_int8_quantization(
        SimpleNamespace(flow_lm=flow_lm),
        "cuda",
        requested=True,
    )

    assert backend == "torchao"
    assert calls == [(flow_lm, {"attention", "ffn"})]


def test_colab_cuda_int8_rejects_cpu_only_backend(monkeypatch):
    fake_pocket_tts = ModuleType("pocket_tts")
    fake_quantization = ModuleType("pocket_tts.quantization")
    fake_quantization.RECOMMENDED_CONFIG = {"attention", "ffn"}
    fake_quantization._get_backend = lambda: "torch.ao"
    fake_quantization.apply_dynamic_int8 = lambda *_args: pytest.fail(
        "The CPU-only backend must not be called for CUDA."
    )
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    monkeypatch.setitem(sys.modules, "pocket_tts.quantization", fake_quantization)

    with pytest.raises(RuntimeError, match="CUDA exige TorchAO"):
        apply_tts_int8_quantization(
            SimpleNamespace(flow_lm=object()),
            "cuda",
            requested=True,
        )
