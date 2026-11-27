from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from joe.services.parakeet import (
    TRANSFORMERS_MODEL_REVISION,
    ParakeetReduxService,
)


def test_transformers_backend_pins_processor_and_weights_to_same_revision(monkeypatch):
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.set_num_threads = Mock()
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    load_model = Mock(return_value=model)
    load_processor = Mock(return_value=object())

    transformers = ModuleType("transformers")
    transformers.AutoModel = SimpleNamespace(from_pretrained=load_model)
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=load_processor)

    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)

    service = ParakeetReduxService()
    service._load_transformers(prefer_cuda=False)

    assert load_processor.call_args.kwargs == {
        "revision": TRANSFORMERS_MODEL_REVISION,
        "trust_remote_code": True,
    }
    assert load_model.call_args.kwargs == {
        "revision": TRANSFORMERS_MODEL_REVISION,
        "trust_remote_code": True,
        "dtype": torch.float32,
    }
    assert service.backend_name == "Transformers CPU"



def test_transformers_transcription_runs_inside_inference_mode(monkeypatch, tmp_path):
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    state = {"active": False}

    @contextmanager
    def inference_mode():
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    class FakeModel:
        def transcribe(self, _waveform, _processor):
            assert state["active"]
            return "Transcrição de teste"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=inference_mode),
    )

    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"fake audio")
    service = ParakeetReduxService()
    service._backend = "Transformers CPU"
    service._loaded_preference = "cpu"
    service._model = FakeModel()
    service._processor = object()
    monkeypatch.setattr(service, "_read_audio_16khz", lambda _path: [0.0, 0.1])

    result = service.transcribe(audio_path, device_preference="cpu")

    assert result["text"] == "Transcrição de teste"
