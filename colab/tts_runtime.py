from __future__ import annotations

import math
import os
import re
from typing import Any


def available_cpu_count() -> int:
    """Return the CPUs visible to this process, respecting container quotas."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def resolve_tts_device(mode: str, cuda_available: bool) -> str:
    """Resolve auto/cpu/cuda without silently ignoring an explicit CUDA request."""
    normalized = str(mode or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("JOE_TTS_DEVICE deve ser 'auto', 'cpu' ou 'cuda'.")
    if normalized == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA foi selecionada para o Pocket TTS, mas torch.cuda.is_available() "
            "retornou False. Ative uma GPU no Colab ou selecione CPU/Automático."
        )
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    return normalized


def resolve_cpu_threads(requested: Any, cpu_count: int | None = None) -> int:
    """Clamp the requested intra-op thread count to CPUs available to the runtime."""
    maximum = max(1, int(cpu_count or available_cpu_count()))
    try:
        value = 1 if requested in (None, "") else int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("JOE_TTS_CPU_THREADS precisa ser um número inteiro.") from exc
    if value < 1:
        raise ValueError("JOE_TTS_CPU_THREADS precisa ser pelo menos 1.")
    return min(value, maximum)


def configure_torch_threads(torch_module: Any, requested: Any) -> int:
    """Apply configurable CPU threads; one thread is the upstream Pocket TTS default."""
    count = resolve_cpu_threads(requested)
    torch_module.set_num_threads(count)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before parallel work starts.
        pass
    return count


def ensure_supported_torch(torch_version: str) -> tuple[int, int, int]:
    """Reject PyTorch 2.4, which Pocket TTS documents as producing incorrect audio."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(torch_version).strip())
    if not match:
        raise RuntimeError(
            f"Não foi possível interpretar a versão do PyTorch: {torch_version!r}."
        )
    parsed = tuple(int(part or 0) for part in match.groups())
    if parsed < (2, 5, 0):
        raise RuntimeError(
            f"PyTorch {torch_version} não é compatível com áudio confiável no Pocket TTS. "
            "A documentação do projeto informa que PyTorch 2.4 produz áudio incorreto. "
            "Atualize para PyTorch 2.5 ou superior e reinicie o runtime do Colab."
        )
    return parsed


def inspect_audio_samples(samples: Any, sample_rate: int) -> dict[str, float | int]:
    """Reject empty/corrupt/silent outputs and return objective WAV signal diagnostics."""
    import numpy as np

    array = np.asarray(samples)
    if array.size == 0:
        raise RuntimeError("O Pocket TTS retornou um áudio vazio.")
    if array.ndim > 2:
        raise RuntimeError(
            f"Formato de áudio inesperado retornado pelo modelo: {array.shape!r}."
        )
    if not np.isfinite(array).all():
        raise RuntimeError(
            "O Pocket TTS retornou NaN ou infinito. Confira PyTorch, pesos e tokenizer."
        )
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RuntimeError(f"Taxa de amostragem inválida: {sample_rate!r}.")

    flat = array.astype(np.float64, copy=False).reshape(-1)
    peak = float(np.max(np.abs(flat)))
    rms = float(math.sqrt(float(np.mean(np.square(flat)))))
    duration = float(array.shape[0] / sample_rate)
    if duration < 0.08:
        raise RuntimeError(
            f"O Pocket TTS retornou somente {duration:.3f}s de áudio para o texto. "
            "A síntese parece ter terminado prematuramente."
        )
    if peak < 1e-5 or rms < 1e-6:
        raise RuntimeError(
            "O Pocket TTS retornou áudio praticamente silencioso. Confira a voz, "
            "o checkpoint, o tokenizer e a versão do PyTorch."
        )
    clipping_fraction = float(np.mean(np.abs(flat) >= 0.999))
    return {
        "duration_seconds": duration,
        "peak": peak,
        "rms": rms,
        "clipping_fraction": clipping_fraction,
        "sample_count": int(array.shape[0]),
    }



def voice_state_for_safetensors_export(voice_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Copy voice-state tensors to contiguous CPU memory before Safetensors export."""
    cpu_state: dict[str, dict[str, Any]] = {}
    for module_name, module_state in voice_state.items():
        cpu_state[module_name] = {}
        for tensor_name, tensor in module_state.items():
            if not hasattr(tensor, "detach") or not hasattr(tensor, "to"):
                raise TypeError(
                    f"Estado de voz inválido: {module_name}/{tensor_name} não é tensor PyTorch."
                )
            cpu_state[module_name][tensor_name] = tensor.detach().to("cpu").contiguous()
    return cpu_state
