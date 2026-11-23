from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from joe.services import parakeet


class FakePhotonContext:
    def __init__(self, speech):
        self.speech = speech
        self.closed = False

    def __enter__(self):
        return self.speech

    def __exit__(self, *_args):
        self.closed = True


def test_cuda_uses_photon_without_requiring_cpu_avx512(monkeypatch):
    speech = SimpleNamespace(transcribe=lambda **_kwargs: {"text": "olá", "segments": []})
    calls = []
    context = FakePhotonContext(speech)
    module = ModuleType("moondream")
    module.photon = lambda model_id, device: (
        calls.append((model_id, device)) or context
    )
    monkeypatch.setitem(sys.modules, "moondream", module)
    monkeypatch.setattr(parakeet, "photon_cpu_supported", lambda: False)

    service = parakeet.ParakeetReduxService()
    service._ensure_loaded("cuda")

    assert calls == [(parakeet.MODEL_ID, "cuda")]
    assert service.backend_name == "Photon CUDA"
    assert service._speech is speech
    assert service._loaded_preference == "cuda"


def test_cpu_without_avx512_skips_photon(monkeypatch):
    service = parakeet.ParakeetReduxService()
    monkeypatch.setattr(parakeet, "photon_cpu_supported", lambda: False)

    assert service._load_photon("cpu") is False
    assert service._speech is None


def test_transformers_fallback_pins_compatible_model_revision(monkeypatch):
    calls = []

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return object()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return FakeModel()

    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.set_num_threads = lambda _count: None
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    transformers = ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel
    transformers.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    service = parakeet.ParakeetReduxService()
    service._load_transformers(prefer_cuda=False)

    assert [entry[0] for entry in calls] == ["processor", "model"]
    for _kind, model_id, kwargs in calls:
        assert model_id == parakeet.MODEL_ID
        assert kwargs["revision"] == parakeet.TRANSFORMERS_REFERENCE_REVISION
        assert kwargs["trust_remote_code"] is True
    assert service.backend_name == "Transformers CPU (referência)"
