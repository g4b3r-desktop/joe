from __future__ import annotations

import gc
import tempfile
import time
import threading
from pathlib import Path

from joe.audio_performance import (
    TTSBenchmarkResult,
    TTSCudaQuantizationBenchmarkResult,
    TTSCudaPrecisionBenchmarkResult,
    TTSCudaProfileBenchmarkResult,
    TTSCudaRopeCacheBenchmarkResult,
)
from joe.model_cache import configure_model_cache
from joe.pocket_tts_rope_cache import install_cuda_rope_cache
from joe.pocket_tts_models import (
    default_model_variant,
    default_voice_for_language,
    is_model_variant_for_language,
    model_loading_options,
)

DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"


class PocketTTSService:
    """Lazy, reusable adapter around Kyutai Pocket TTS."""

    def __init__(self) -> None:
        self._models: dict[
            tuple[str, str, bool, int | None, float | None, float | None], object
        ] = {}
        self._voice_states: dict[tuple[str, str, bool, str], object] = {}

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

    @staticmethod
    def _apply_cuda_int8_quantization(model) -> None:
        """Use Pocket TTS' TorchAO INT8 path after the model is on CUDA."""
        try:
            from pocket_tts.quantization import (
                RECOMMENDED_CONFIG,
                _get_backend,
                apply_dynamic_int8,
            )
        except ImportError as exc:
            raise RuntimeError(
                "A quantização INT8 na GPU requer o extra TorchAO do Pocket TTS. "
                "Reinstale as dependências do Joe com pocket-tts[quantize]."
            ) from exc

        backend = _get_backend()
        if backend != "torchao":
            raise RuntimeError(
                "A quantização INT8 na CUDA exige o backend TorchAO com extensões "
                "nativas disponíveis. O backend torch.ao/FBGEMM é exclusivo da CPU."
            )
        flow_lm = getattr(model, "flow_lm", None)
        if flow_lm is None:
            raise RuntimeError(
                "A versão deste Pocket TTS não expõe flow_lm para quantização CUDA."
            )
        try:
            apply_dynamic_int8(flow_lm, RECOMMENDED_CONFIG)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "TorchAO não conseguiu aplicar INT8 às camadas de atenção e FFN "
                f"do Pocket TTS na GPU: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _apply_cuda_fp16_autocast(model, torch) -> None:
        """Apply autocast inside Pocket TTS' own worker-thread entry points.

        Pocket TTS 4.0.1 runs the initial flow prefill, autoregressive decoding,
        and Mimi audio decoding in different thread contexts. Autocast is
        thread-local, so wrapping generate_audio() alone would not cover them.
        """
        method_names = (
            "_generate",
            "_autoregressive_generation",
            "_decode_audio_worker",
        )
        originals = {
            name: getattr(model, name, None)
            for name in method_names
        }
        missing = [name for name, method in originals.items() if not callable(method)]
        if missing:
            raise RuntimeError(
                "A versão instalada do Pocket TTS não expõe os pontos internos "
                "necessários para ativar FP16 com segurança: "
                + ", ".join(missing)
                + ". Atualize o Pocket TTS ou use CUDA FP32."
            )
        if not callable(getattr(torch, "autocast", None)):
            raise RuntimeError(
                "Esta versão do PyTorch não oferece torch.autocast para a CUDA."
            )

        for name, original in originals.items():
            def run_with_fp16_autocast(*args, _original=original, **kwargs):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return _original(*args, **kwargs)

            setattr(model, name, run_with_fp16_autocast)

    def _get_model(
        self,
        torch,
        tts_model_class,
        model_variant: str,
        device: str,
        quantize: bool,
        sampler_decode_steps: int | None = None,
        temperature: float | None = None,
        eos_threshold: float | None = None,
    ):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Dispositivo TTS inválido. Escolha CPU ou NVIDIA GPU (CUDA).")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA foi selecionada para o Pocket TTS, mas não está disponível nesta "
                "instalação do PyTorch. Instale/ative a versão CUDA do PyTorch e confirme "
                "que o driver NVIDIA está funcionando."
            )

        # CPU uses Pocket TTS' standard dynamic INT8 path. CUDA uses TorchAO
        # after the float model has been moved to the GPU.
        effective_quantize = bool(quantize)
        quantize_on_load = effective_quantize and device == "cpu"
        cache_key = (
            model_variant,
            device,
            effective_quantize,
            sampler_decode_steps,
            temperature,
            eos_threshold,
        )
        model = self._models.get(cache_key)
        if model is not None:
            return model

        # Keep a single model/device active to avoid retaining duplicate weights in RAM/VRAM.
        self._voice_states.clear()
        self._models.clear()
        model_load_options = {}
        if sampler_decode_steps is not None:
            model_load_options["sampler_decode_steps"] = int(sampler_decode_steps)
        if temperature is not None:
            model_load_options["temp"] = float(temperature)
        if eos_threshold is not None:
            model_load_options["eos_threshold"] = float(eos_threshold)
        model = tts_model_class.load_model(
            **model_loading_options(model_variant),
            quantize=quantize_on_load,
            **model_load_options,
        )
        if device == "cuda":
            try:
                model.to("cuda")
                if effective_quantize:
                    self._apply_cuda_int8_quantization(model)
            except Exception as exc:  # noqa: BLE001
                del model
                torch.cuda.empty_cache()
                if isinstance(exc, RuntimeError) and "quantização INT8" in str(exc):
                    raise
                raise RuntimeError(
                    f"Não foi possível preparar o Pocket TTS na GPU CUDA: {exc}"
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
        quantize: bool | None = None,
        sampler_decode_steps: int | None = None,
        temperature: float | None = None,
        eos_threshold: float | None = None,
        frames_after_eos: int | None = None,
        max_tokens: int | None = None,
        teacher_force_previous_chunk: bool = False,
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
        # Preserve the existing CPU default; CUDA stays unquantized unless
        # the caller explicitly opts in.
        effective_quantize = (
            device_preference == "cpu" if quantize is None else bool(quantize)
        )
        if sampler_decode_steps is not None and not 1 <= int(sampler_decode_steps) <= 16:
            raise ValueError("sampler_decode_steps deve ficar entre 1 e 16.")
        if temperature is not None and not 0.05 <= float(temperature) <= 2.0:
            raise ValueError("temperature deve ficar entre 0,05 e 2,0.")
        if eos_threshold is not None and not -20.0 <= float(eos_threshold) <= 20.0:
            raise ValueError("eos_threshold deve ficar entre -20 e 20.")
        if frames_after_eos is not None and not 1 <= int(frames_after_eos) <= 20:
            raise ValueError("frames_after_eos deve ficar entre 1 e 20; use None para automático.")
        if max_tokens is not None and not 1 <= int(max_tokens) <= 10_000:
            raise ValueError("max_tokens deve ficar entre 1 e 10.000.")
        model = self._get_model(
            torch,
            TTSModel,
            model_variant,
            device_preference,
            effective_quantize,
            sampler_decode_steps,
            temperature,
            eos_threshold,
        )

        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        state_key = (
            model_variant, device_preference, effective_quantize, voice_source
        )
        voice_state = self._voice_states.get(state_key)
        if voice_state is None:
            voice_state = model.get_state_for_audio_prompt(voice_source)
            self._voice_states[state_key] = voice_state

        generation_options = {}
        if frames_after_eos is not None:
            generation_options["frames_after_eos"] = int(frames_after_eos)
        if max_tokens is not None:
            generation_options["max_tokens"] = int(max_tokens)
        if teacher_force_previous_chunk:
            generation_options["teacher_force_previous_chunk"] = True
        audio = model.generate_audio(voice_state, text.strip(), **generation_options)
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
        quantize: bool | None = None,
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
        effective_quantize = (
            device_preference == "cpu" if quantize is None else bool(quantize)
        )
        model = self._get_model(
            torch, TTSModel, model_variant, device_preference, effective_quantize
        )

        source_key = str(source.resolve())
        state_key = (
            model_variant, device_preference, effective_quantize, source_key
        )
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
        quantize_cuda: bool = False,
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
                model = self._get_model(
                    torch,
                    TTSModel,
                    model_variant,
                    device,
                    bool(quantize) if device == "cpu" else bool(quantize_cuda),
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                model_load_seconds = time.perf_counter() - model_started

                voice_started = time.perf_counter()
                voice_state = model.get_state_for_audio_prompt(voice_source)
                if device == "cuda":
                    torch.cuda.synchronize()
                voice_prepare_seconds = time.perf_counter() - voice_started

                generation_times: list[float] = []
                audio_duration_seconds = 0.0
                for _ in range(repeat_count):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    generation_started = time.perf_counter()
                    # Pocket TTS owns its no-grad context. Do not use inference_mode:
                    # its streaming decoder can use worker threads, while inference_mode
                    # is thread-local and can make tensors unsafe to update in those threads.
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
                        audio_duration_seconds=audio_duration_seconds,
                        character_count=len(text.strip()),
                        peak_vram_mb=peak_vram_mb,
                        quantized=bool(quantize) if device == "cpu" else bool(quantize_cuda),
                        repeats=repeat_count,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    TTSBenchmarkResult(
                        device=device,
                        cpu_threads=cpu_threads,
                        error=f"{type(exc).__name__}: {exc}",
                        quantized=bool(quantize) if device == "cpu" else bool(quantize_cuda),
                    )
                )
            finally:
                voice_state = None
                model = None
                self._models.clear()
                self._voice_states.clear()
                gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def benchmark_cuda_quantization(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        cpu_threads: int = 2,
        warmup_runs: int = 2,
        measured_runs: int = 7,
    ) -> list[TTSCudaQuantizationBenchmarkResult]:
        """Compare CUDA inference with and without INT8 after separate warmups."""
        if not text.strip():
            raise ValueError("Informe um texto para comparar a quantização CUDA.")
        warmup_count = max(1, int(warmup_runs))
        measured_count = max(3, int(measured_runs))

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
        if not torch.cuda.is_available():
            error = "CUDA não está disponível na instalação atual do PyTorch."
            return [
                TTSCudaQuantizationBenchmarkResult(quantized=False, error=error),
                TTSCudaQuantizationBenchmarkResult(quantized=True, error=error),
            ]

        self._configure_torch(torch, cpu_threads)
        # Do not benchmark on top of a model already held by normal generation.
        # The chosen device/INT8 preferences in the UI are not changed.
        self.close()

        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        results: list[TTSCudaQuantizationBenchmarkResult] = []

        for quantized in (False, True):
            model = None
            voice_state = None
            model_load_seconds = 0.0
            quantization_seconds = 0.0
            voice_prepare_seconds = 0.0
            generation_times: list[float] = []
            audio_durations: list[float] = []
            peak_vram_mb = 0.0
            peak_vram_reserved_mb = 0.0
            error = ""
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                model_started = time.perf_counter()
                model = TTSModel.load_model(
                    **model_loading_options(model_variant),
                    quantize=False,
                )
                model.to("cuda")
                torch.cuda.synchronize()
                model_load_seconds = time.perf_counter() - model_started

                if quantized:
                    quantization_started = time.perf_counter()
                    self._apply_cuda_int8_quantization(model)
                    torch.cuda.synchronize()
                    quantization_seconds = time.perf_counter() - quantization_started

                voice_started = time.perf_counter()
                voice_state = model.get_state_for_audio_prompt(voice_source)
                torch.cuda.synchronize()
                voice_prepare_seconds = time.perf_counter() - voice_started

                def generate_once() -> float:
                    started = time.perf_counter()
                    # Keep the same execution context as normal Joe generation.
                    # Pocket TTS decorates its generation API with torch.no_grad();
                    # inference_mode is thread-local and is unsafe with its worker threads.
                    audio = model.generate_audio(
                        voice_state,
                        text.strip(),
                        copy_state=True,
                    )
                    samples = audio.detach().cpu().numpy()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    sample_count = (
                        int(samples.shape[-1])
                        if getattr(samples, "ndim", 0)
                        else int(samples.size)
                    )
                    audio_durations.append(
                        sample_count / max(1, int(model.sample_rate))
                    )
                    del audio, samples
                    return elapsed

                # Discard lazy kernel setup/compilation and GPU clock ramp-up.
                for _ in range(warmup_count):
                    generate_once()
                audio_durations.clear()

                # The measured sample contains only completed generations, not
                # model loading, voice preparation, or the warm-up runs.
                for _ in range(measured_count):
                    generation_times.append(generate_once())

                peak_vram_mb = (
                    float(torch.cuda.max_memory_allocated()) / (1024**2)
                )
                peak_vram_reserved_mb = (
                    float(torch.cuda.max_memory_reserved()) / (1024**2)
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            finally:
                voice_state = None
                model = None
                self._models.clear()
                self._voice_states.clear()
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

            results.append(
                TTSCudaQuantizationBenchmarkResult(
                    quantized=quantized,
                    model_load_seconds=model_load_seconds,
                    quantization_seconds=quantization_seconds,
                    voice_prepare_seconds=voice_prepare_seconds,
                    generation_times=tuple(generation_times),
                    audio_durations_seconds=tuple(audio_durations),
                    warmup_runs=warmup_count,
                    peak_vram_mb=peak_vram_mb,
                    peak_vram_reserved_mb=peak_vram_reserved_mb,
                    error=error,
                )
            )
        return results

    def benchmark_cuda_precision(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        cpu_threads: int = 2,
        warmup_runs: int = 2,
        measured_runs: int = 7,
    ) -> list[TTSCudaPrecisionBenchmarkResult]:
        """Compare CUDA FP32 against experimental FP16 autocast without changing generation defaults."""
        if not text.strip():
            raise ValueError("Informe um texto para comparar FP32 e FP16 na CUDA.")
        warmup_count = max(1, int(warmup_runs))
        measured_count = max(3, int(measured_runs))

        configure_model_cache()
        try:
            import numpy as np
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
        if not torch.cuda.is_available():
            error = "CUDA não está disponível na instalação atual do PyTorch."
            return [
                TTSCudaPrecisionBenchmarkResult(precision="FP32", error=error),
                TTSCudaPrecisionBenchmarkResult(precision="FP16 autocast", error=error),
            ]

        self._configure_torch(torch, cpu_threads)
        self.close()

        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        results: list[TTSCudaPrecisionBenchmarkResult] = []

        for use_fp16 in (False, True):
            model = None
            voice_state = None
            model_load_seconds = 0.0
            autocast_setup_seconds = 0.0
            voice_prepare_seconds = 0.0
            generation_times: list[float] = []
            audio_durations: list[float] = []
            audio_peaks: list[float] = []
            audio_rms_levels: list[float] = []
            peak_vram_mb = 0.0
            peak_vram_reserved_mb = 0.0
            error = ""
            precision = "FP16 autocast" if use_fp16 else "FP32"
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                model_started = time.perf_counter()
                model = TTSModel.load_model(
                    **model_loading_options(model_variant),
                    quantize=False,
                )
                model.to("cuda")
                torch.cuda.synchronize()
                model_load_seconds = time.perf_counter() - model_started

                # Keep model weights in FP32. Autocast selectively uses FP16 for
                # CUDA operations that support it, including both Pocket TTS workers.
                if use_fp16:
                    autocast_started = time.perf_counter()
                    self._apply_cuda_fp16_autocast(model, torch)
                    autocast_setup_seconds = time.perf_counter() - autocast_started

                voice_started = time.perf_counter()
                voice_state = model.get_state_for_audio_prompt(voice_source)
                torch.cuda.synchronize()
                voice_prepare_seconds = time.perf_counter() - voice_started

                def generate_once() -> float:
                    started = time.perf_counter()
                    audio = model.generate_audio(
                        voice_state,
                        text.strip(),
                        copy_state=True,
                    )
                    samples = audio.detach().cpu().numpy()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    if samples.size == 0 or not np.isfinite(samples).all():
                        raise RuntimeError(
                            f"{precision} produziu áudio vazio ou com valores NaN/Inf."
                        )
                    samples64 = np.asarray(samples, dtype=np.float64)
                    sample_count = int(samples.shape[-1]) if samples.ndim else int(samples.size)
                    audio_durations.append(
                        sample_count / max(1, int(model.sample_rate))
                    )
                    audio_peaks.append(float(np.max(np.abs(samples64))))
                    audio_rms_levels.append(
                        float(np.sqrt(np.mean(np.square(samples64))))
                    )
                    del audio, samples, samples64
                    return elapsed

                for _ in range(warmup_count):
                    generate_once()
                audio_durations.clear()
                audio_peaks.clear()
                audio_rms_levels.clear()

                for _ in range(measured_count):
                    generation_times.append(generate_once())

                peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
                peak_vram_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024**2)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            finally:
                voice_state = None
                model = None
                self._models.clear()
                self._voice_states.clear()
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

            results.append(
                TTSCudaPrecisionBenchmarkResult(
                    precision=precision,
                    model_load_seconds=model_load_seconds,
                    autocast_setup_seconds=autocast_setup_seconds,
                    voice_prepare_seconds=voice_prepare_seconds,
                    generation_times=tuple(generation_times),
                    audio_durations_seconds=tuple(audio_durations),
                    audio_peak_amplitudes=tuple(audio_peaks),
                    audio_rms_levels=tuple(audio_rms_levels),
                    warmup_runs=warmup_count,
                    peak_vram_mb=peak_vram_mb,
                    peak_vram_reserved_mb=peak_vram_reserved_mb,
                    error=error,
                )
            )
        return results


    def benchmark_cuda_tf32(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        cpu_threads: int = 2,
        warmup_runs: int = 2,
        measured_runs: int = 7,
    ) -> list[TTSCudaPrecisionBenchmarkResult]:
        """Compare IEEE FP32 and TF32 matmul on CUDA without changing generation defaults."""
        if not text.strip():
            raise ValueError("Informe um texto para comparar FP32 IEEE e TF32 na CUDA.")
        warmup_count = max(1, int(warmup_runs))
        measured_count = max(3, int(measured_runs))

        configure_model_cache()
        try:
            import numpy as np
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
        if not torch.cuda.is_available():
            error = "CUDA não está disponível na instalação atual do PyTorch."
            return [
                TTSCudaPrecisionBenchmarkResult(precision="FP32 IEEE", error=error),
                TTSCudaPrecisionBenchmarkResult(precision="FP32 TF32", error=error),
            ]

        capability = torch.cuda.get_device_capability()
        if int(capability[0]) < 8:
            error = (
                "TF32 requer GPU NVIDIA Ampere (compute capability 8.0) ou superior; "
                f"a GPU atual informa {capability[0]}.{capability[1]}."
            )
            return [
                TTSCudaPrecisionBenchmarkResult(precision="FP32 IEEE", error=error),
                TTSCudaPrecisionBenchmarkResult(precision="FP32 TF32", error=error),
            ]

        matmul_backend = getattr(
            getattr(getattr(torch, "backends", None), "cuda", None),
            "matmul",
            None,
        )
        if matmul_backend is None:
            error = "Esta versão do PyTorch não expõe o controle de precisão CUDA matmul."
            return [
                TTSCudaPrecisionBenchmarkResult(precision="FP32 IEEE", error=error),
                TTSCudaPrecisionBenchmarkResult(precision="FP32 TF32", error=error),
            ]

        # PyTorch 2.9+ prefers fp32_precision; older releases expose allow_tf32.
        precision_attribute = (
            "fp32_precision"
            if hasattr(matmul_backend, "fp32_precision")
            else "allow_tf32"
            if hasattr(matmul_backend, "allow_tf32")
            else ""
        )
        if not precision_attribute:
            error = "PyTorch não oferece controle IEEE/TF32 para CUDA matmul."
            return [
                TTSCudaPrecisionBenchmarkResult(precision="FP32 IEEE", error=error),
                TTSCudaPrecisionBenchmarkResult(precision="FP32 TF32", error=error),
            ]
        original_precision = getattr(matmul_backend, precision_attribute)

        def set_tf32(enabled: bool) -> None:
            value = (
                ("tf32" if enabled else "ieee")
                if precision_attribute == "fp32_precision"
                else bool(enabled)
            )
            setattr(matmul_backend, precision_attribute, value)

        self._configure_torch(torch, cpu_threads)
        self.close()
        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        results: list[TTSCudaPrecisionBenchmarkResult] = []

        try:
            for use_tf32 in (False, True):
                model = None
                voice_state = None
                model_load_seconds = 0.0
                voice_prepare_seconds = 0.0
                generation_times: list[float] = []
                audio_durations: list[float] = []
                audio_peaks: list[float] = []
                audio_rms_levels: list[float] = []
                peak_vram_mb = 0.0
                peak_vram_reserved_mb = 0.0
                error = ""
                precision = "FP32 TF32" if use_tf32 else "FP32 IEEE"
                try:
                    # Process-wide policy also applies to Pocket TTS worker threads.
                    set_tf32(use_tf32)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    model_started = time.perf_counter()
                    model = TTSModel.load_model(
                        **model_loading_options(model_variant),
                        quantize=False,
                    )
                    model.to("cuda")
                    torch.cuda.synchronize()
                    model_load_seconds = time.perf_counter() - model_started

                    voice_started = time.perf_counter()
                    voice_state = model.get_state_for_audio_prompt(voice_source)
                    torch.cuda.synchronize()
                    voice_prepare_seconds = time.perf_counter() - voice_started

                    def generate_once() -> float:
                        started = time.perf_counter()
                        audio = model.generate_audio(
                            voice_state,
                            text.strip(),
                            copy_state=True,
                        )
                        samples = audio.detach().cpu().numpy()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        if samples.size == 0 or not np.isfinite(samples).all():
                            raise RuntimeError(
                                f"{precision} produziu áudio vazio ou com valores NaN/Inf."
                            )
                        samples64 = np.asarray(samples, dtype=np.float64)
                        sample_count = (
                            int(samples.shape[-1]) if samples.ndim else int(samples.size)
                        )
                        audio_durations.append(
                            sample_count / max(1, int(model.sample_rate))
                        )
                        audio_peaks.append(float(np.max(np.abs(samples64))))
                        audio_rms_levels.append(
                            float(np.sqrt(np.mean(np.square(samples64))))
                        )
                        del audio, samples, samples64
                        return elapsed

                    for _ in range(warmup_count):
                        generate_once()
                    audio_durations.clear()
                    audio_peaks.clear()
                    audio_rms_levels.clear()

                    for _ in range(measured_count):
                        generation_times.append(generate_once())

                    peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
                    peak_vram_reserved_mb = (
                        float(torch.cuda.max_memory_reserved()) / (1024**2)
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    voice_state = None
                    model = None
                    self._models.clear()
                    self._voice_states.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        torch.cuda.empty_cache()

                results.append(
                    TTSCudaPrecisionBenchmarkResult(
                        precision=precision,
                        model_load_seconds=model_load_seconds,
                        voice_prepare_seconds=voice_prepare_seconds,
                        generation_times=tuple(generation_times),
                        audio_durations_seconds=tuple(audio_durations),
                        audio_peak_amplitudes=tuple(audio_peaks),
                        audio_rms_levels=tuple(audio_rms_levels),
                        warmup_runs=warmup_count,
                        peak_vram_mb=peak_vram_mb,
                        peak_vram_reserved_mb=peak_vram_reserved_mb,
                        error=error,
                    )
                )
        finally:
            # Never leak this experimental setting into ordinary generation.
            setattr(matmul_backend, precision_attribute, original_precision)
        return results


    def benchmark_cuda_rope_cache(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        cpu_threads: int = 2,
        warmup_runs: int = 2,
        measured_runs: int = 7,
    ) -> TTSCudaRopeCacheBenchmarkResult:
        """Compare the original FP32 RoPE with cached FP32 trigonometric tables."""
        import numpy as np

        if not text.strip():
            return TTSCudaRopeCacheBenchmarkResult(
                error="Informe um texto para comparar o RoPE original com o cache."
            )
        warmup_count = max(1, int(warmup_runs))
        measured_count = max(3, int(measured_runs))
        configure_model_cache()
        try:
            import torch
            from pocket_tts import TTSModel
        except ImportError as exc:
            return TTSCudaRopeCacheBenchmarkResult(
                error=f"Dependências do Pocket TTS ausentes: {exc}"
            )

        model_variant = model_variant or default_model_variant(language)
        if not is_model_variant_for_language(language, model_variant):
            return TTSCudaRopeCacheBenchmarkResult(
                error=f"A versão {model_variant!r} não está disponível para {language!r}."
            )
        if not torch.cuda.is_available():
            return TTSCudaRopeCacheBenchmarkResult(
                error="CUDA não está disponível na instalação atual do PyTorch."
            )

        self._configure_torch(torch, cpu_threads)
        self.close()
        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )
        model = None
        cache_handle = None
        model_load_seconds = 0.0
        voice_prepare_seconds = 0.0
        baseline_times: list[float] = []
        cached_times: list[float] = []
        baseline_durations: list[float] = []
        cached_durations: list[float] = []
        baseline_audio: list[np.ndarray] = []
        cached_audio: list[np.ndarray] = []
        baseline_vram = 0.0
        baseline_reserved = 0.0
        cached_vram = 0.0
        cached_reserved = 0.0
        cache_positions = 0
        cache_calls = 0
        cache_fallbacks = 0
        cache_bytes = 0
        error = ""

        def peak_memory() -> tuple[float, float]:
            torch.cuda.synchronize()
            allocated = float(torch.cuda.max_memory_allocated()) / (1024**2)
            reserved_getter = getattr(torch.cuda, "max_memory_reserved", None)
            reserved = (
                float(reserved_getter()) / (1024**2)
                if callable(reserved_getter)
                else 0.0
            )
            return allocated, reserved

        def run_mode() -> tuple[list[float], list[float], list[np.ndarray]]:
            times: list[float] = []
            durations: list[float] = []
            audio_samples: list[np.ndarray] = []
            total_runs = warmup_count + measured_count
            for run_index in range(total_runs):
                seed = 1729 + run_index
                torch.cuda.synchronize()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    started = time.perf_counter()
                    audio = model.generate_audio(
                        voice_state,
                        text.strip(),
                        copy_state=True,
                    )
                    samples = np.asarray(audio.detach().cpu().numpy()).copy()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                if run_index >= warmup_count:
                    times.append(elapsed)
                    durations.append(
                        float(samples.shape[-1]) / float(model.sample_rate)
                        if samples.ndim
                        else 0.0
                    )
                    audio_samples.append(samples)
            return times, durations, audio_samples

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            model = self._get_model(
                torch,
                TTSModel,
                model_variant,
                "cuda",
                False,
            )
            torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - load_started

            voice_started = time.perf_counter()
            voice_state = model.get_state_for_audio_prompt(voice_source)
            torch.cuda.synchronize()
            voice_prepare_seconds = time.perf_counter() - voice_started

            # Reference mode: original upstream RoPE implementation, unchanged.
            torch.cuda.reset_peak_memory_stats()
            baseline_times, baseline_durations, baseline_audio = run_mode()
            baseline_vram, baseline_reserved = peak_memory()

            # Experimental mode: cache FP32 cos/sin tables on the same shared
            # RotaryEmbedding object. First-use table creation is inside discarded
            # warmup runs. The handle restores both patched methods in finally.
            cache_handle = install_cuda_rope_cache(
                model,
                torch,
                initial_positions=8192,
            )
            torch.cuda.reset_peak_memory_stats()
            cached_times, cached_durations, cached_audio = run_mode()
            cached_vram, cached_reserved = peak_memory()
            cache_positions = cache_handle.capacity_positions
            cache_calls = cache_handle.calls
            cache_fallbacks = cache_handle.fallbacks
            cache_bytes = cache_handle.allocated_bytes

        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if cache_handle is not None:
                cache_handle.close()

        shapes_match = (
            len(baseline_audio) == len(cached_audio)
            and all(
                left.shape == right.shape
                for left, right in zip(baseline_audio, cached_audio)
            )
        )
        differences: list[np.ndarray] = []
        for original, cached in zip(baseline_audio, cached_audio):
            original_flat = original.reshape(-1)
            cached_flat = cached.reshape(-1)
            shared_length = min(original_flat.size, cached_flat.size)
            if shared_length:
                differences.append(
                    np.abs(
                        original_flat[:shared_length].astype(np.float64)
                        - cached_flat[:shared_length].astype(np.float64)
                    )
                )
        if differences:
            all_differences = np.concatenate(differences)
            audio_max_difference = float(all_differences.max(initial=0.0))
            audio_mean_difference = float(all_differences.mean())
            audio_rms_difference = float(
                np.sqrt(np.mean(np.square(all_differences)))
            )
        else:
            audio_max_difference = 0.0
            audio_mean_difference = 0.0
            audio_rms_difference = 0.0

        if not error and (not baseline_times or not cached_times):
            error = "Uma das modalidades não concluiu gerações suficientes."

        return TTSCudaRopeCacheBenchmarkResult(
            model_load_seconds=model_load_seconds,
            voice_prepare_seconds=voice_prepare_seconds,
            baseline_generation_times=tuple(baseline_times),
            cached_generation_times=tuple(cached_times),
            baseline_audio_durations_seconds=tuple(baseline_durations),
            cached_audio_durations_seconds=tuple(cached_durations),
            audio_shapes_match=shapes_match,
            audio_max_abs_difference=audio_max_difference,
            audio_mean_abs_difference=audio_mean_difference,
            audio_rms_difference=audio_rms_difference,
            baseline_peak_vram_mb=baseline_vram,
            baseline_peak_vram_reserved_mb=baseline_reserved,
            cached_peak_vram_mb=cached_vram,
            cached_peak_vram_reserved_mb=cached_reserved,
            warmup_runs=warmup_count,
            cache_positions=cache_positions,
            cache_calls=cache_calls,
            cache_fallbacks=cache_fallbacks,
            cache_bytes=cache_bytes,
            error=error,
        )

    def benchmark_cuda_profile(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        model_variant: str | None = None,
        *,
        cpu_threads: int = 2,
        warmup_runs: int = 2,
        measured_runs: int = 7,
    ) -> list[TTSCudaProfileBenchmarkResult]:
        """Profile CUDA FP32 Flow LM, Mimi decoding, latent queue waits and D2H transfer."""
        if not text.strip():
            raise ValueError("Informe um texto para diagnosticar o desempenho CUDA.")
        warmup_count = max(1, int(warmup_runs))
        measured_count = max(3, int(measured_runs))

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
        if not torch.cuda.is_available():
            return [
                TTSCudaProfileBenchmarkResult(
                    error="CUDA não está disponível na instalação atual do PyTorch."
                )
            ]

        self._configure_torch(torch, cpu_threads)
        self.close()
        voice_source = str(voice_path) if voice_path else (
            default_voice_for_language(language) or DEFAULT_VOICE
        )

        model = None
        voice_state = None
        model_load_seconds = 0.0
        voice_prepare_seconds = 0.0
        generation_times: list[float] = []
        audio_durations: list[float] = []
        audio_transfer_seconds: list[float] = []
        peak_vram_mb = 0.0
        peak_vram_reserved_mb = 0.0
        error = ""

        trace_lock = threading.Lock()
        active_run: dict[str, int | bool | None] = {
            "index": None,
            "autoregressive": False,
            "flow_thread_id": None,
        }
        trace_runs: list[dict[str, object]] = []
        component_keys = (
            "flow_lm_input_proj",
            "attention_norm",
            "attention_qkv_projection",
            "attention_rope",
            "kv_cache_append",
            "kv_cache_write",
            "attention_sdpa",
            "attention_output_projection",
            "ffn_norm",
            "ffn_linear1",
            "ffn_gelu",
            "ffn_linear2",
            "flow_lm_output_norm",
            "flow_lm_eos_projection",
            "flow_net_input_projection",
            "flow_net_condition_projection",
            "flow_net_time_embedding",
            "flow_net_residual_blocks",
            "flow_net_final_layer",
        )

        def new_trace_run() -> dict[str, object]:
            trace = {
                "flow_ar_loop_wall": [],
                "flow_prefill_wall": [],
                "flow_prefill_events": [],
                "flow_auto_wall": [],
                "flow_auto_events": [],
                "flow_attention_wall": [],
                "flow_attention_events": [],
                "flow_ffn_wall": [],
                "flow_ffn_events": [],
                "flow_net_wall": [],
                "flow_net_events": [],
                "mimi_decode_wall": [],
                "mimi_decode_events": [],
                "mimi_worker_wall": [],
                "queue_wait": [],
                "queue_get_calls": [],
            }
            for component_key in component_keys:
                trace[f"{component_key}_wall"] = []
                trace[f"{component_key}_events"] = []
            return trace

        trace_runs.extend(new_trace_run() for _ in range(measured_count))

        def record_value(
            key: str,
            value: object,
            *,
            autoregressive_only: bool = False,
        ) -> None:
            with trace_lock:
                index = active_run["index"]
                if index is None:
                    return
                if autoregressive_only and (
                    not active_run["autoregressive"]
                    or active_run["flow_thread_id"] != threading.get_ident()
                ):
                    return
                trace_runs[index][key].append(value)

        def record_phase(
            key: str,
            started: float,
            start_event,
            end_event,
            *,
            autoregressive_only: bool = False,
        ) -> None:
            record_value(
                f"{key}_wall",
                time.perf_counter() - started,
                autoregressive_only=autoregressive_only,
            )
            record_value(
                f"{key}_events",
                (start_event, end_event),
                autoregressive_only=autoregressive_only,
            )

        def cuda_event():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def install_thread_tracing(target_model):
            import importlib

            original_flow = getattr(
                target_model, "_run_flow_lm_and_increment_step", None
            )
            original_decode_worker = getattr(
                target_model, "_decode_audio_worker", None
            )
            original_autoregressive = getattr(
                target_model, "_autoregressive_generation", None
            )
            mimi = getattr(target_model, "mimi", None)
            original_mimi_decode = getattr(mimi, "decode_from_latent", None)
            flow_lm = getattr(target_model, "flow_lm", None)
            transformer = getattr(flow_lm, "transformer", None)
            transformer_layers = tuple(getattr(transformer, "layers", ()))
            flow_net = getattr(flow_lm, "flow_net", None)
            original_flow_net_forward = getattr(flow_net, "forward", None)

            # Fail with the intended compatibility message before importing internals
            # when a Pocket TTS build does not expose the profiling entry points.
            if not all(
                callable(method)
                for method in (
                    original_flow,
                    original_decode_worker,
                    original_autoregressive,
                    original_mimi_decode,
                    original_flow_net_forward,
                )
            ) or not transformer_layers:
                raise RuntimeError(
                    "A versão instalada do Pocket TTS não expõe os métodos internos "
                    "necessários ao profiling detalhado: Flow LM, Transformer, "
                    "FlowNet e workers de geração/decodificação."
                )

            attention_module = importlib.import_module("pocket_tts.modules.attention")
            transformer_module = importlib.import_module("pocket_tts.modules.transformer")
            attention_functional = getattr(attention_module, "F", None)
            transformer_functional = getattr(transformer_module, "F", None)
            original_sdpa = getattr(
                attention_functional, "scaled_dot_product_attention", None
            )
            original_gelu = getattr(transformer_functional, "gelu", None)
            original_complete_kv = getattr(attention_module, "complete_kv", None)

            layer_methods = all(
                callable(getattr(layer, method, None))
                for layer in transformer_layers
                for method in ("_sa_block", "_ff_block")
            )
            attention_components = all(
                callable(
                    getattr(
                        getattr(getattr(layer, "self_attn", None), name, None),
                        "forward",
                        None,
                    )
                )
                for layer in transformer_layers
                for name in ("in_proj", "rope", "out_proj")
            )
            cache_components = all(
                callable(getattr(getattr(layer.self_attn, "_cache_backend", None), "append_and_get", None))
                for layer in transformer_layers
            )
            layer_components = all(
                callable(getattr(getattr(layer, name, None), "forward", None))
                for layer in transformer_layers
                for name in ("norm1", "norm2", "linear1", "linear2")
            )
            flow_lm_components = all(
                callable(getattr(getattr(flow_lm, name, None), "forward", None))
                for name in ("input_linear", "out_norm", "out_eos")
            )
            flow_net_components = all(
                callable(getattr(getattr(flow_net, name, None), "forward", None))
                for name in ("input_proj", "cond_embed", "final_layer")
            )
            if not all(
                callable(method)
                for method in (
                    original_flow,
                    original_decode_worker,
                    original_autoregressive,
                    original_mimi_decode,
                    original_flow_net_forward,
                    original_sdpa,
                    original_gelu,
                    original_complete_kv,
                )
            ) or not transformer_layers or not layer_methods or not attention_components or not cache_components or not layer_components or not flow_lm_components or not flow_net_components:
                raise RuntimeError(
                    "A versão instalada do Pocket TTS não expõe os métodos internos "
                    "necessários ao microprofiling: Flow LM autoregressivo, "
                    "self-attention (projeções, RoPE, cache KV e SDPA), "
                    "camadas norm/FFN e componentes do FlowNet. "
                    "Use a geração CUDA FP32 normal ou confira a versão do Pocket TTS."
                )

            restore_callbacks = []

            def patch_attribute(owner, attribute: str, replacement) -> None:
                original = getattr(owner, attribute)
                setattr(owner, attribute, replacement)
                restore_callbacks.append(
                    lambda owner=owner, attribute=attribute, original=original:
                        setattr(owner, attribute, original)
                )

            def timed_component(original, key: str):
                """Measure a coarse block with CUDA Events and host wall time."""
                def wrapped(*args, **kwargs):
                    started = time.perf_counter()
                    start_event = cuda_event()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        end_event = cuda_event()
                        record_phase(
                            key,
                            started,
                            start_event,
                            end_event,
                            autoregressive_only=True,
                        )
                return wrapped

            def timed_component_wall(original, key: str):
                """Measure a fine-grained call on the host without event overhead."""
                def wrapped(*args, **kwargs):
                    started = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        record_value(
                            f"{key}_wall",
                            time.perf_counter() - started,
                            autoregressive_only=True,
                        )
                return wrapped

            patched_module_ids: set[int] = set()

            def patch_module_forward(module, key: str) -> None:
                # Pocket TTS shares one RotaryEmbedding instance across all
                # Transformer layers. Do not wrap the same object once per layer:
                # nested wrappers multiply both call counts and reported time.
                module_id = id(module)
                if module_id in patched_module_ids:
                    return
                original = getattr(module, "forward", None)
                if not callable(original):
                    raise RuntimeError(
                        f"Componente {key} não expõe forward() para microprofiling."
                    )
                patch_attribute(module, "forward", timed_component_wall(original, key))
                patched_module_ids.add(module_id)

            try:
                # Split the attention block into its actual operations. Timings are
                # collected only in the autoregressive worker, excluding text prefill.
                for layer in transformer_layers:
                    patch_attribute(
                        layer, "_sa_block",
                        timed_component(layer._sa_block, "flow_attention")
                    )
                    patch_attribute(
                        layer, "_ff_block",
                        timed_component(layer._ff_block, "flow_ffn")
                    )
                    self_attention = layer.self_attn
                    patch_module_forward(layer.norm1, "attention_norm")
                    patch_module_forward(self_attention.in_proj, "attention_qkv_projection")
                    patch_module_forward(self_attention.rope, "attention_rope")
                    patch_attribute(
                        self_attention._cache_backend,
                        "append_and_get",
                        timed_component(
                            self_attention._cache_backend.append_and_get,
                            "kv_cache_append",
                        ),
                    )
                    patch_module_forward(self_attention.out_proj, "attention_output_projection")
                    patch_module_forward(layer.norm2, "ffn_norm")
                    patch_module_forward(layer.linear1, "ffn_linear1")
                    patch_module_forward(layer.linear2, "ffn_linear2")

                # complete_kv performs the actual in-place KV write. It also includes
                # the offset scalar read, which can force a host/device synchronization.
                patch_attribute(
                    attention_module,
                    "complete_kv",
                    timed_component_wall(original_complete_kv, "kv_cache_write"),
                )
                patch_attribute(
                    attention_functional,
                    "scaled_dot_product_attention",
                    timed_component_wall(original_sdpa, "attention_sdpa"),
                )
                patch_attribute(
                    transformer_functional,
                    "gelu",
                    timed_component_wall(original_gelu, "ffn_gelu"),
                )

                patch_module_forward(flow_lm.input_linear, "flow_lm_input_proj")
                patch_module_forward(flow_lm.out_norm, "flow_lm_output_norm")
                patch_module_forward(flow_lm.out_eos, "flow_lm_eos_projection")

                patch_module_forward(flow_net.input_proj, "flow_net_input_projection")
                patch_module_forward(flow_net.cond_embed, "flow_net_condition_projection")
                patch_module_forward(flow_net.final_layer, "flow_net_final_layer")
                for module in tuple(getattr(flow_net, "time_embed", ())):
                    patch_module_forward(module, "flow_net_time_embedding")
                for module in tuple(getattr(flow_net, "res_blocks", ())):
                    patch_module_forward(module, "flow_net_residual_blocks")

                patch_attribute(
                    flow_net,
                    "forward",
                    timed_component(original_flow_net_forward, "flow_net"),
                )

                def timed_flow(*args, **kwargs):
                    text_tokens = kwargs.get("text_tokens")
                    if text_tokens is None and len(args) > 1:
                        text_tokens = args[1]
                    is_prefill = (
                        text_tokens is not None
                        and len(getattr(text_tokens, "shape", ())) > 1
                        and int(text_tokens.shape[1]) > 0
                    )
                    phase = "flow_prefill" if is_prefill else "flow_auto"
                    with trace_lock:
                        if active_run["index"] is not None:
                            active_run["autoregressive"] = not is_prefill
                    started = time.perf_counter()
                    start_event = cuda_event()
                    try:
                        return original_flow(*args, **kwargs)
                    finally:
                        end_event = cuda_event()
                        record_phase(phase, started, start_event, end_event)

                def timed_autoregressive(*args, **kwargs):
                    with trace_lock:
                        if active_run["index"] is not None:
                            active_run["autoregressive"] = True
                            active_run["flow_thread_id"] = threading.get_ident()
                    started = time.perf_counter()
                    try:
                        return original_autoregressive(*args, **kwargs)
                    finally:
                        record_value(
                            "flow_ar_loop_wall",
                            time.perf_counter() - started,
                        )

                def timed_mimi_decode(*args, **kwargs):
                    started = time.perf_counter()
                    start_event = cuda_event()
                    try:
                        return original_mimi_decode(*args, **kwargs)
                    finally:
                        end_event = cuda_event()
                        record_phase("mimi_decode", started, start_event, end_event)

                def timed_decode_worker(latents_queue, *args, **kwargs):
                    original_get = latents_queue.get

                    def timed_queue_get(*get_args, **get_kwargs):
                        started = time.perf_counter()
                        try:
                            return original_get(*get_args, **get_kwargs)
                        finally:
                            record_value(
                                "queue_wait", time.perf_counter() - started
                            )
                            record_value("queue_get_calls", 1)

                    # Queue.get_nowait() delegates to this same get method, so this
                    # captures blocking waits and non-blocking drain attempts.
                    latents_queue.get = timed_queue_get
                    started = time.perf_counter()
                    try:
                        return original_decode_worker(latents_queue, *args, **kwargs)
                    finally:
                        record_value(
                            "mimi_worker_wall", time.perf_counter() - started
                        )

                patch_attribute(target_model, "_run_flow_lm_and_increment_step", timed_flow)
                patch_attribute(target_model, "_autoregressive_generation", timed_autoregressive)
                patch_attribute(target_model, "_decode_audio_worker", timed_decode_worker)
                patch_attribute(mimi, "decode_from_latent", timed_mimi_decode)
            except Exception:
                for restore in reversed(restore_callbacks):
                    try:
                        restore()
                    except Exception:
                        pass
                raise

            def restore_thread_tracing() -> None:
                for restore in reversed(restore_callbacks):
                    try:
                        restore()
                    except Exception:
                        pass

            return restore_thread_tracing

        restore_tracing = lambda: None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model_started = time.perf_counter()
            model = TTSModel.load_model(
                **model_loading_options(model_variant),
                quantize=False,
            )
            model.to("cuda")
            torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - model_started

            restore_tracing = install_thread_tracing(model)

            voice_started = time.perf_counter()
            voice_state = model.get_state_for_audio_prompt(voice_source)
            torch.cuda.synchronize()
            voice_prepare_seconds = time.perf_counter() - voice_started

            def generate_once(run_index: int | None) -> tuple[float, float, float]:
                with trace_lock:
                    active_run["index"] = run_index
                started = time.perf_counter()
                try:
                    audio = model.generate_audio(
                        voice_state,
                        text.strip(),
                        copy_state=True,
                    )
                    transfer_started = time.perf_counter()
                    samples = audio.detach().cpu().numpy()
                    torch.cuda.synchronize()
                    finished = time.perf_counter()
                    transfer_elapsed = finished - transfer_started
                    total_elapsed = finished - started
                    sample_count = (
                        int(samples.shape[-1])
                        if getattr(samples, "ndim", 0)
                        else int(samples.size)
                    )
                    duration = sample_count / max(1, int(model.sample_rate))
                    del audio, samples
                    return total_elapsed, transfer_elapsed, duration
                finally:
                    with trace_lock:
                        active_run["index"] = None

            for _ in range(warmup_count):
                generate_once(None)

            for run_index in range(measured_count):
                total, transfer, duration = generate_once(run_index)
                generation_times.append(total)
                audio_transfer_seconds.append(transfer)
                audio_durations.append(duration)

            # One final device synchronization makes all CUDA events from both
            # worker threads readable without serializing the individual phases.
            torch.cuda.synchronize()
            peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
            peak_vram_reserved_mb = (
                float(torch.cuda.max_memory_reserved()) / (1024**2)
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with trace_lock:
                active_run["index"] = None
                active_run["autoregressive"] = False
                active_run["flow_thread_id"] = None
            try:
                restore_tracing()
            except Exception:
                pass
            voice_state = None
            model = None
            self._models.clear()
            self._voice_states.clear()
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()

        if error:
            return [TTSCudaProfileBenchmarkResult(error=error)]

        def phase_wall(key: str, index: int) -> float:
            values = trace_runs[index][f"{key}_wall"]
            return sum(float(value) for value in values)

        def phase_gpu(key: str, index: int) -> float:
            events = trace_runs[index][f"{key}_events"]
            return sum(
                float(start.elapsed_time(end)) / 1000.0
                for start, end in events
            )

        def phase_count(key: str, index: int) -> int:
            if key == "queue_get_calls":
                return len(trace_runs[index][key])
            return len(trace_runs[index][f"{key}_wall"])

        return [
            TTSCudaProfileBenchmarkResult(
                model_load_seconds=model_load_seconds,
                voice_prepare_seconds=voice_prepare_seconds,
                generation_times=tuple(generation_times),
                audio_durations_seconds=tuple(audio_durations),
                audio_transfer_seconds=tuple(audio_transfer_seconds),
                flow_prefill_wall_seconds=tuple(
                    phase_wall("flow_prefill", i) for i in range(measured_count)
                ),
                flow_prefill_gpu_seconds=tuple(
                    phase_gpu("flow_prefill", i) for i in range(measured_count)
                ),
                flow_autoregressive_wall_seconds=tuple(
                    phase_wall("flow_auto", i) for i in range(measured_count)
                ),
                flow_autoregressive_gpu_seconds=tuple(
                    phase_gpu("flow_auto", i) for i in range(measured_count)
                ),
                flow_step_counts=tuple(
                    phase_count("flow_auto", i) for i in range(measured_count)
                ),
                flow_ar_loop_wall_seconds=tuple(
                    sum(float(value) for value in trace_runs[i]["flow_ar_loop_wall"])
                    for i in range(measured_count)
                ),
                flow_ar_control_wall_seconds=tuple(
                    max(
                        0.0,
                        sum(float(value) for value in trace_runs[i]["flow_ar_loop_wall"])
                        - phase_wall("flow_auto", i),
                    )
                    for i in range(measured_count)
                ),
                flow_component_wall_seconds={
                    key: tuple(phase_wall(key, i) for i in range(measured_count))
                    for key in component_keys
                },
                flow_component_gpu_seconds={
                    key: tuple(phase_gpu(key, i) for i in range(measured_count))
                    for key in component_keys
                },
                flow_component_calls={
                    key: tuple(phase_count(key, i) for i in range(measured_count))
                    for key in component_keys
                },
                flow_attention_wall_seconds=tuple(
                    phase_wall("flow_attention", i) for i in range(measured_count)
                ),
                flow_attention_gpu_seconds=tuple(
                    phase_gpu("flow_attention", i) for i in range(measured_count)
                ),
                flow_attention_calls=tuple(
                    phase_count("flow_attention", i) for i in range(measured_count)
                ),
                flow_ffn_wall_seconds=tuple(
                    phase_wall("flow_ffn", i) for i in range(measured_count)
                ),
                flow_ffn_gpu_seconds=tuple(
                    phase_gpu("flow_ffn", i) for i in range(measured_count)
                ),
                flow_ffn_calls=tuple(
                    phase_count("flow_ffn", i) for i in range(measured_count)
                ),
                flow_net_wall_seconds=tuple(
                    phase_wall("flow_net", i) for i in range(measured_count)
                ),
                flow_net_gpu_seconds=tuple(
                    phase_gpu("flow_net", i) for i in range(measured_count)
                ),
                flow_net_calls=tuple(
                    phase_count("flow_net", i) for i in range(measured_count)
                ),
                mimi_decode_wall_seconds=tuple(
                    phase_wall("mimi_decode", i) for i in range(measured_count)
                ),
                mimi_decode_gpu_seconds=tuple(
                    phase_gpu("mimi_decode", i) for i in range(measured_count)
                ),
                mimi_decode_calls=tuple(
                    phase_count("mimi_decode", i) for i in range(measured_count)
                ),
                mimi_worker_wall_seconds=tuple(
                    sum(float(value) for value in trace_runs[i]["mimi_worker_wall"])
                    for i in range(measured_count)
                ),
                latent_queue_wait_seconds=tuple(
                    sum(float(value) for value in trace_runs[i]["queue_wait"])
                    for i in range(measured_count)
                ),
                latent_queue_get_calls=tuple(
                    len(trace_runs[i]["queue_get_calls"])
                    for i in range(measured_count)
                ),
                warmup_runs=warmup_count,
                peak_vram_mb=peak_vram_mb,
                peak_vram_reserved_mb=peak_vram_reserved_mb,
            )
        ]

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
