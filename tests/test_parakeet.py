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



def test_cuda_preference_uses_photon_cuda_before_transformers(monkeypatch, tmp_path):
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    from joe.services.parakeet import ParakeetReduxService

    transcribe = Mock(return_value={"text": "Photon acelerado", "segments": []})

    class FakeContext:
        def __enter__(self):
            return SimpleNamespace(transcribe=transcribe)

        def __exit__(self, *_args):
            return False

    photon = Mock(return_value=FakeContext())
    moondream = ModuleType("moondream")
    moondream.photon = photon
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "moondream", moondream)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"fake audio")
    service = ParakeetReduxService()
    monkeypatch.setattr(
        service,
        "_load_transformers",
        Mock(side_effect=AssertionError("Transformers should not be loaded")),
    )

    result = service.transcribe(audio_path, device_preference="cuda")

    photon.assert_called_once_with("moondream/parakeet-redux", device="cuda")
    transcribe.assert_called_once_with(audio=str(audio_path))
    assert result["text"] == "Photon acelerado"
    assert result["backend"] == "Photon CUDA"
