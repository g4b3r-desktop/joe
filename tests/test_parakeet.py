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
