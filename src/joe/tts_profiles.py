from __future__ import annotations

import os


PROFILE_OPTIONS = (
    ("custom", "Personalizado"),
    ("economy", "Econômico"),
    ("balanced", "Equilibrado"),
    ("performance", "Desempenho"),
)


def performance_profile_settings(
    profile: str,
    *,
    cpu_count: int | None = None,
    cuda_available: bool = False,
) -> dict[str, object]:
    """Return the concrete Pocket TTS settings represented by a preset."""
    cores = max(1, int(cpu_count or os.cpu_count() or 1))
    if profile == "economy":
        return {"device": "cpu", "cpu_threads": 1, "quantization": True}
    if profile == "balanced":
        return {
            "device": "cpu",
            "cpu_threads": max(1, cores // 2),
            "quantization": True,
        }
    if profile == "performance":
        if cuda_available:
            return {"device": "cuda", "cpu_threads": cores, "quantization": False}
        return {"device": "cpu", "cpu_threads": cores, "quantization": False}
    raise ValueError(f"Perfil de desempenho desconhecido: {profile!r}")


def build_generation_settings(
    *,
    device: str,
    cpu_threads: int,
    quantization: bool,
    profile: str = "custom",
    remote: bool = False,
) -> dict[str, object]:
    """Create reproducible, non-sensitive settings metadata for audio history."""
    local_device = device if device in {"cpu", "cuda"} else "cpu"
    return {
        "backend": "Google Colab" if remote else "Pocket TTS local",
        "local_settings_applied": not remote,
        "device": "não aplicado (remoto)" if remote else local_device,
        "cpu_threads": max(1, int(cpu_threads)),
        "quantization": bool(quantization) and local_device == "cpu" and not remote,
        "profile": str(profile or "custom"),
    }
