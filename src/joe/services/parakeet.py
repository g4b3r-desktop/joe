from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any

MODEL_ID = "moondream/parakeet-redux"
# The current Redux repository is Photon-only. Keep the last published
# Transformers-compatible snapshot for CPUs without AVX-512 VNNI.
TRANSFORMERS_REFERENCE_REVISION = "475a51b8c2d3588cd2cad04f1bc646c832d76366"
logger = logging.getLogger(__name__)


def photon_cpu_supported() -> bool:
    """Photon's optimized x86 Redux path requires AVX-512 VNNI."""
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return True
    if machine not in {"x86_64", "amd64"}:
        return False
    try:
        flags = set()
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("flags"):
                    flags.update(line.split(":", 1)[-1].strip().lower().split())
                    break
        return "avx512_vnni" in flags
    except OSError:
        return False


class ParakeetReduxService:
    """Parakeet Redux adapter: Photon on supported hardware, pinned Transformers fallback."""

    def __init__(self) -> None:
        self._backend: str | None = None
        self._photon_context: Any = None
        self._speech: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        self._loaded_preference: str | None = None
        self._backend_note = ""

    @property
    def backend_name(self) -> str:
        return self._backend or "não carregado"

    @property
    def backend_note(self) -> str:
        return self._backend_note

    def _load_photon(self, device: str) -> bool:
        """Load Redux with Photon on CUDA or on CPUs supported by its kernels."""
        if device == "cpu" and not photon_cpu_supported():
            return False
        try:
            import moondream as md

            photon_device = device
            if (
                device == "cpu"
                and sys.platform == "darwin"
                and platform.machine().lower() == "arm64"
            ):
                photon_device = "mps"

            context = md.photon(MODEL_ID, device=photon_device)
            speech = context.__enter__()
            self._photon_context = context
            self._speech = speech
            self._backend = f"Photon {photon_device.upper()}"
            self._device = photon_device
            self._backend_note = (
                "Parakeet Redux executado pelo Photon; pesos ternários carregados diretamente."
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._backend_note = f"Photon indisponível; usando fallback Transformers ({exc})"
            if self._photon_context is not None:
                try:
                    self._photon_context.__exit__(None, None, None)
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Falha ao fechar Photon após erro de carregamento: %s",
                        cleanup_error,
                    )
            self._photon_context = None
            self._speech = None
            return False

    def _load_transformers(self, prefer_cuda: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "O fallback Transformers precisa do extra ASR. Execute: "
                "python -m pip install -e '.[asr-fallback]'"
            ) from exc

        try:
            torch.set_num_threads(3)
        except RuntimeError:
            pass

        # The current model revision is Photon-only and no longer ships the
        # Transformers processor files. Pin the last reference snapshot so
        # AutoProcessor never follows the mutable main branch.
        model_kwargs = {
            "revision": TRANSFORMERS_REFERENCE_REVISION,
            "trust_remote_code": True,
        }
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, **model_kwargs)

        device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        if prefer_cuda and device == "cpu":
            self._backend_note = (
                "CUDA foi solicitada, mas torch.cuda.is_available() é falso; fallback para CPU."
            )
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            model = AutoModel.from_pretrained(
                MODEL_ID,
                **model_kwargs,
                dtype=dtype,
            )
            model.to(device).eval()
        except Exception as exc:
            if device != "cuda":
                raise
            # A CUDA build may be installed with insufficient VRAM or an
            # incompatible runtime. Retry the reference implementation on CPU.
            try:
                del model
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()
            self._backend_note = f"Fallback Transformers CUDA indisponível ({exc}); usando CPU."
            model = AutoModel.from_pretrained(
                MODEL_ID,
                **model_kwargs,
                dtype=torch.float32,
            )
            model.to("cpu").eval()
            device = "cpu"

        self._model = model
        self._device = device
        self._backend = (
            "Transformers CUDA (FP16)" if device == "cuda" else "Transformers CPU (referência)"
        )
        if not self._backend_note:
            self._backend_note = (
                "Fallback de referência fixado em uma revisão compatível do modelo; "
                "pode ser mais lento que Photon."
            )
        else:
            self._backend_note += " — modo de referência ativo."

    def _ensure_loaded(self, device_preference: str) -> None:
        if device_preference not in {"cpu", "cuda"}:
            raise ValueError("Dispositivo inválido. Escolha cpu ou cuda.")
        if self._backend is not None and self._loaded_preference != device_preference:
            self.close()
        if self._backend is not None:
            return

        # Photon CUDA is supported by the current SDK for Parakeet Redux.
        # On CPU, use Photon only when the CPU has the required instruction set.
        if device_preference == "cuda":
            if self._load_photon("cuda"):
                self._loaded_preference = device_preference
                return
        elif self._load_photon("cpu"):
            self._loaded_preference = device_preference
            return

        self._load_transformers(prefer_cuda=(device_preference == "cuda"))
        self._loaded_preference = device_preference

    @staticmethod
    def _read_audio_16khz(path: Path):
        from math import gcd

        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)
        if sample_rate != 16_000:
            divisor = gcd(int(sample_rate), 16_000)
            samples = resample_poly(
                samples,
                16_000 // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32)
        return samples

    def transcribe(
        self,
        audio_path: str | Path,
        timestamps: bool = False,
        device_preference: str = "cpu",
    ) -> dict[str, Any]:
        source = Path(audio_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {source}")

        self._ensure_loaded(device_preference)

        if self._speech is not None:
            options: dict[str, Any] = {}
            if timestamps:
                options["timestamps"] = "segment"
            result = self._speech.transcribe(audio=str(source), **options)
            return {
                "text": str(result.get("text", "")).strip(),
                "segments": result.get("segments", []),
                "backend": self.backend_name,
                "note": self.backend_note,
            }

        waveform = self._read_audio_16khz(source)
        result = self._model.transcribe(waveform, self._processor)
        if isinstance(result, (list, tuple)):
            result = result[0] if result else ""
        if isinstance(result, dict):
            result = result.get("text", "")
        return {
            "text": str(result).strip(),
            "segments": [],
            "backend": self.backend_name,
            "note": self.backend_note,
        }

    def close(self) -> None:
        if self._photon_context is not None:
            try:
                self._photon_context.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao fechar o backend Photon: %s", exc)
        self._photon_context = None
        self._speech = None
        self._model = None
        self._processor = None
        self._backend = None
        self._device = "cpu"
        self._loaded_preference = None
        self._backend_note = ""
