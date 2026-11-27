from __future__ import annotations

import gc
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

from joe.audio_performance import TTSBenchmarkResult
from joe.model_cache import configure_model_cache
from joe.pocket_tts_models import (
    default_model_variant,
    default_voice_for_language,
    is_model_variant_for_language,
    model_loading_options,
)

DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"


def _inference_context(torch):
    """Use PyTorch inference mode when available; keep lightweight test doubles compatible."""
    inference_mode = getattr(torch, "inference_mode", None)
    return inference_mode() if callable(inference_mode) else nullcontext()


class PocketTTSService:
    """Lazy, reusable adapter around Kyutai Pocket TTS."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str, bool, int | None], object] = {}
        self._voice_states: dict[tuple[str, str, bool, str, int | None], object] = {}

    @staticmethod
    def _configure_torch(torch, cpu_threads: int) -> None:
        threads = int(cpu_threads)
        if threads < 1:
            raise ValueError("A quantidade de threads da CPU precisa ser maior que zero.")
        try:
            torch.set_num_threads(threads)
        except RuntimeError:
            # PyTorch can reject changes after parallel work has already started.
            pass

    def _get_model(
        self,
        torch,
        tts_model_class,
        model_variant: str,
        device: str,
        quantize: bool,
        sampler_decode_steps: int | None = None,
    ):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Dispositivo TTS inválido. Escolha CPU ou NVIDIA GPU (CUDA).")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA foi selecionada para o Pocket TTS, mas não está disponível nesta "
                "instalação do PyTorch. Instale/ative a versão CUDA do PyTorch e confirme "
                "que o driver NVIDIA está funcionando."
            )

        # Pocket TTS dynamic INT8 quantization is supported on CPU only.
        effective_quantize = bool(quantize) and device == "cpu"
        if sampler_decode_steps is not None and not 1 <= int(sampler_decode_steps) <= 16:
            raise ValueError("Os passos do sampler precisam ficar entre 1 e 16.")
        sampler_steps = int(sampler_decode_steps) if sampler_decode_steps is not None else None
        cache_key = (model_variant, device, effective_quantize, sampler_steps)
        model = self._models.get(cache_key)
        if model is not None:
            return model

        # Keep a single model/device active to avoid retaining duplicate weights in RAM/VRAM.
        self._voice_states.clear()
        self._models.clear()
        loading_options = model_loading_options(model_variant)
        if sampler_steps is not None:
            loading_options["sampler_decode_steps"] = sampler_steps
        model = tts_model_class.load_model(
            **loading_options,
            quantize=effective_quantize,
        )
        if device == "cuda":
            try:
                # Pocket TTS is moved to CUDA after loading; dynamic INT8 is CPU-only.
                model.to("cuda")
            except Exception as exc:  # noqa: BLE001
                del model
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Não foi possível mover o Pocket TTS para a GPU CUDA: {exc}"
                ) from exc
        self._models[cache_key] = model
        return model

    def generate(
        self,
        text: str,
        language: str,
        output_path: str | Path,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        device_preference: str = "cpu",
        cpu_threads: int = 2,
        quantize: bool = True,
        sampler_decode_steps: int | None = None,
    ) -> Path:
        if not text.strip():
            raise ValueError("Digite algum texto antes de gerar o áudio.")

        configure_model_cache()
        try:
            import soundfile as sf
            import torch
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Dependências do Pocket TTS ausentes. Execute ./scripts/install-linux.sh."
            ) from exc

        model_variant = model_variant or default_model_variant(language)
        if not is_model_variant_for_language(language, model_variant):
            raise ValueError(
                f"A versão {model_variant!r} não está disponível para o idioma {language!r}."
            )

        self._configure_torch(torch, cpu_threads)
        effective_quantize = bool(quantize) and device_preference == "cpu"
        model = self._get_model(
            torch,
            TTSModel,
            model_variant,
            device_preference,
            effective_quantize,
            sampler_decode_steps,
        )

        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        sampler_steps = int(sampler_decode_steps) if sampler_decode_steps is not None else None
        state_key = (
            model_variant,
            device_preference,
            effective_quantize,
            voice_source,
            sampler_steps,
        )
        with _inference_context(torch):
            voice_state = self._voice_states.get(state_key)
            if voice_state is None:
                voice_state = model.get_state_for_audio_prompt(voice_source)
                self._voice_states[state_key] = voice_state

            audio = model.generate_audio(voice_state, text.strip())
        samples = audio.detach().cpu().numpy()
        destination = Path(output_path).expanduser()
        return _write_wav_atomically(sf, samples, int(model.sample_rate), destination)

    def export_voice_state(
        self,
        voice_audio_path: str | Path,
        output_path: str | Path,
        language: str,
        model_variant: str | None = None,
        *,
        device_preference: str = "cpu",
        cpu_threads: int = 2,
        quantize: bool = True,
    ) -> Path:
        """Encode a reference recording once and save its reusable KV state."""
        source = Path(voice_audio_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Áudio de referência não encontrado: {source}")
        destination = Path(output_path).expanduser()
        if destination.suffix.lower() != ".safetensors":
            raise ValueError("O estado de voz precisa ser salvo com a extensão .safetensors.")

        configure_model_cache()
        try:
            from pocket_tts import export_model_state, TTSModel
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "A versão instalada do Pocket TTS não oferece exportação de estado de voz."
            ) from exc

        model_variant = model_variant or default_model_variant(language)
        if not is_model_variant_for_language(language, model_variant):
            raise ValueError(
                f"A versão {model_variant!r} não está disponível para o idioma {language!r}."
            )
        self._configure_torch(torch, cpu_threads)
        effective_quantize = bool(quantize) and device_preference == "cpu"
        model = self._get_model(
            torch, TTSModel, model_variant, device_preference, effective_quantize
        )

        source_key = str(source.resolve())
        state_key = (
            model_variant, device_preference, effective_quantize, source_key, None
        )
        with _inference_context(torch):
            voice_state = self._voice_states.get(state_key)
            if voice_state is None:
                voice_state = model.get_state_for_audio_prompt(source_key)
                self._voice_states[state_key] = voice_state

        destination.parent.mkdir(parents=True, exist_ok=True)
        export_model_state(voice_state, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("O Pocket TTS não criou um arquivo .safetensors válido.")
        return destination

    def benchmark_devices(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        cpu_threads: int = 2,
        repeats: int = 2,
        quantize: bool = True,
        sampler_decode_steps: int | None = None,
    ) -> list[TTSBenchmarkResult]:
        """Compare CPU and CUDA using the same model, voice and text without saving audio."""
        if not text.strip():
            raise ValueError("Informe um texto para comparar o desempenho do Pocket TTS.")

        configure_model_cache()
        try:
            import torch
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Dependências do Pocket TTS ausentes. Execute ./scripts/install-linux.sh."
            ) from exc

        model_variant = model_variant or default_model_variant(language)
        if not is_model_variant_for_language(language, model_variant):
            raise ValueError(
                f"A versão {model_variant!r} não está disponível para o idioma {language!r}."
            )
        self._configure_torch(torch, cpu_threads)
        # Release any model retained by normal generation before the two isolated runs.
        # This prevents a previously selected CUDA model from consuming VRAM during the test.
        self.close()

        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        results: list[TTSBenchmarkResult] = []
        repeat_count = max(1, int(repeats))
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                results.append(
                    TTSBenchmarkResult(
                        device=device,
                        cpu_threads=cpu_threads,
                        error="CUDA não está disponível na instalação atual do PyTorch.",
                    )
                )
                continue

            model = None
            voice_state = None
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                model_started = time.perf_counter()
                loading_options = model_loading_options(model_variant)
                if sampler_decode_steps is not None:
                    if not 1 <= int(sampler_decode_steps) <= 16:
                        raise ValueError("Os passos do sampler precisam ficar entre 1 e 16.")
                    loading_options["sampler_decode_steps"] = int(sampler_decode_steps)
                model = TTSModel.load_model(
                    **loading_options,
                    quantize=(device == "cpu" and bool(quantize)),
                )
                if device == "cuda":
                    model.to("cuda")
                    torch.cuda.synchronize()
                model_load_seconds = time.perf_counter() - model_started

                voice_started = time.perf_counter()
                with _inference_context(torch):
                    voice_state = model.get_state_for_audio_prompt(voice_source)
                if device == "cuda":
                    torch.cuda.synchronize()
                voice_prepare_seconds = time.perf_counter() - voice_started

                # Warm up kernels/caches once; do not include this first call in the average.
                if device == "cuda":
                    torch.cuda.synchronize()
                with _inference_context(torch):
                    warmup_audio = model.generate_audio(voice_state, text.strip())
                warmup_samples = warmup_audio.detach().cpu().numpy()
                if device == "cuda":
                    torch.cuda.synchronize()
                del warmup_audio, warmup_samples

                generation_times: list[float] = []
                audio_duration_seconds = 0.0
                for _ in range(repeat_count):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    generation_started = time.perf_counter()
                    with _inference_context(torch):
                        audio = model.generate_audio(voice_state, text.strip())
                    samples = audio.detach().cpu().numpy()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    generation_times.append(time.perf_counter() - generation_started)
                    sample_count = int(samples.shape[-1]) if samples.ndim else int(samples.size)
                    audio_duration_seconds = sample_count / max(1, int(model.sample_rate))
                    del audio, samples

                peak_vram_mb = 0.0
                if device == "cuda":
                    peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
                results.append(
                    TTSBenchmarkResult(
                        device=device,
                        cpu_threads=cpu_threads,
                        model_load_seconds=model_load_seconds,
                        voice_prepare_seconds=voice_prepare_seconds,
                        generation_seconds=sum(generation_times) / len(generation_times),
                        generation_min_seconds=min(generation_times),
                        generation_median_seconds=statistics.median(generation_times),
                        generation_max_seconds=max(generation_times),
                        audio_duration_seconds=audio_duration_seconds,
                        character_count=len(text.strip()),
                        peak_vram_mb=peak_vram_mb,
                        quantized=(device == "cpu" and bool(quantize)),
                        repeats=repeat_count,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    TTSBenchmarkResult(
                        device=device,
                        cpu_threads=cpu_threads,
                        error=f"{type(exc).__name__}: {exc}",
                        quantized=(device == "cpu" and bool(quantize)),
                    )
                )
            finally:
                voice_state = None
                model = None
                gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def close(self) -> None:
        self._voice_states.clear()
        self._models.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _write_wav_atomically(soundfile_module, samples, sample_rate: int, destination: Path) -> Path:
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=".wav",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        soundfile_module.write(
            temporary_path,
            samples,
            int(sample_rate),
            subtype="PCM_16",
        )
        info = soundfile_module.info(temporary_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError("O Pocket TTS produziu um WAV vazio ou inválido.")
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination
