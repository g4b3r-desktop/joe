from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import wave


@dataclass(frozen=True, slots=True)
class AudioPerformance:
    """Métricas medidas pelo próprio aplicativo durante uma geração de áudio."""

    elapsed_seconds: float
    audio_duration_seconds: float
    character_count: int
    remote: bool = False


@dataclass(frozen=True, slots=True)
class TTSCudaQuantizationBenchmarkResult:
    """Repeated measurements for one CUDA Pocket TTS quantization mode."""

    quantized: bool
    model_load_seconds: float = 0.0
    quantization_seconds: float = 0.0
    voice_prepare_seconds: float = 0.0
    generation_times: tuple[float, ...] = ()
    audio_durations_seconds: tuple[float, ...] = ()
    warmup_runs: int = 0
    peak_vram_mb: float = 0.0
    peak_vram_reserved_mb: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True)
class TTSCudaPrecisionBenchmarkResult:
    """Repeated measurements for one CUDA inference precision mode."""

    precision: str
    model_load_seconds: float = 0.0
    autocast_setup_seconds: float = 0.0
    voice_prepare_seconds: float = 0.0
    generation_times: tuple[float, ...] = ()
    audio_durations_seconds: tuple[float, ...] = ()
    audio_peak_amplitudes: tuple[float, ...] = ()
    audio_rms_levels: tuple[float, ...] = ()
    warmup_runs: int = 0
    peak_vram_mb: float = 0.0
    peak_vram_reserved_mb: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True)
class TTSCudaAttentionBenchmarkResult:
    """Repeated CUDA measurements for the default and Flash-FP16 SDPA paths."""

    mode: str
    model_load_seconds: float = 0.0
    voice_prepare_seconds: float = 0.0
    generation_times: tuple[float, ...] = ()
    audio_durations_seconds: tuple[float, ...] = ()
    audio_peak_amplitudes: tuple[float, ...] = ()
    audio_rms_levels: tuple[float, ...] = ()
    warmup_runs: int = 0
    peak_vram_mb: float = 0.0
    peak_vram_reserved_mb: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True)
class TTSCudaProfileBenchmarkResult:
    """Per-generation CUDA phase timings collected from Pocket TTS worker threads."""

    model_load_seconds: float = 0.0
    voice_prepare_seconds: float = 0.0
    generation_times: tuple[float, ...] = ()
    audio_durations_seconds: tuple[float, ...] = ()
    audio_transfer_seconds: tuple[float, ...] = ()
    flow_prefill_wall_seconds: tuple[float, ...] = ()
    flow_prefill_gpu_seconds: tuple[float, ...] = ()
    flow_autoregressive_wall_seconds: tuple[float, ...] = ()
    flow_autoregressive_gpu_seconds: tuple[float, ...] = ()
    flow_step_counts: tuple[int, ...] = ()
    flow_attention_wall_seconds: tuple[float, ...] = ()
    flow_attention_gpu_seconds: tuple[float, ...] = ()
    flow_attention_calls: tuple[int, ...] = ()
    flow_ffn_wall_seconds: tuple[float, ...] = ()
    flow_ffn_gpu_seconds: tuple[float, ...] = ()
    flow_ffn_calls: tuple[int, ...] = ()
    flow_net_wall_seconds: tuple[float, ...] = ()
    flow_net_gpu_seconds: tuple[float, ...] = ()
    flow_net_calls: tuple[int, ...] = ()
    mimi_decode_wall_seconds: tuple[float, ...] = ()
    mimi_decode_gpu_seconds: tuple[float, ...] = ()
    mimi_decode_calls: tuple[int, ...] = ()
    mimi_worker_wall_seconds: tuple[float, ...] = ()
    latent_queue_wait_seconds: tuple[float, ...] = ()
    latent_queue_get_calls: tuple[int, ...] = ()
    warmup_runs: int = 0
    peak_vram_mb: float = 0.0
    peak_vram_reserved_mb: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True)
class TTSBenchmarkResult:
    """One isolated Pocket TTS benchmark run for a selected compute device."""

    device: str
    cpu_threads: int
    model_load_seconds: float = 0.0
    voice_prepare_seconds: float = 0.0
    generation_seconds: float = 0.0
    audio_duration_seconds: float = 0.0
    character_count: int = 0
    peak_vram_mb: float = 0.0
    quantized: bool = False
    repeats: int = 0
    error: str = ""


def wav_duration_seconds(path: str | Path) -> float:
    """Retorna a duração do WAV sem carregar as amostras de áudio na memória."""
    try:
        with wave.open(str(path), "rb") as audio_file:
            sample_rate = audio_file.getframerate()
            if sample_rate <= 0:
                return 0.0
            return audio_file.getnframes() / sample_rate
    except (OSError, EOFError, wave.Error):
        try:
            import soundfile as sf

            return max(0.0, float(sf.info(str(path)).duration))
        except Exception:  # noqa: BLE001
            return 0.0


def format_duration(seconds: float) -> str:
    """Formata uma duração de forma curta e consistente em toda a interface."""
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{value % 60:04.1f}"
    if minutes:
        return f"{minutes:02d}:{value % 60:04.1f}"
    return f"{value:.1f}s".replace(".", ",")


def format_elapsed_status(seconds: float) -> str:
    """Texto usado enquanto a geração ainda está em andamento."""
    return f"Tempo decorrido: {format_duration(seconds)}"


def _format_number(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def format_performance_report(metrics: AudioPerformance) -> str:
    """Monta o mesmo resumo de desempenho para as duas telas do Joe."""
    elapsed = max(0.0, float(metrics.elapsed_seconds))
    audio_duration = max(0.0, float(metrics.audio_duration_seconds))
    characters = max(0, int(metrics.character_count))

    lines = [
        "Medição realizada pelo Joe neste computador",
        (
            f"Tempo total: {format_duration(elapsed)}  •  "
            f"Duração do WAV: {format_duration(audio_duration)}"
        ),
    ]

    if elapsed > 0 and audio_duration > 0:
        realtime_speed = audio_duration / elapsed
        chars_per_second = characters / elapsed if characters else 0.0
        lines.append(
            f"Velocidade efetiva: {_format_number(realtime_speed, 2)}× tempo real  •  "
            f"Texto: {_format_number(chars_per_second)} caracteres/s"
        )
        lines.append(
            "Acima de 1× significa que o áudio foi gerado mais rápido do que sua reprodução."
        )
    else:
        lines.append(
            "Não foi possível ler a duração do WAV para calcular a velocidade efetiva."
        )

    if metrics.remote:
        lines.append(
            "No modo remoto, o cronômetro inclui envio, fila e espera pela resposta da API."
        )
    else:
        lines.append(
            "No modo local, o cronômetro inclui o carregamento do modelo quando necessário."
        )

    return "\n".join(lines)


def format_tts_benchmark_report(
    results: list[TTSBenchmarkResult],
    *,
    cpu_threads: int,
) -> str:
    """Format CPU/CUDA measurements with load, voice setup and inference separated."""
    lines = [
        "Comparação de desempenho — Pocket TTS",
        f"Mesmo texto, idioma, modelo e voz. Threads configuradas: {cpu_threads}.",
        "Carga do modelo e preparação da voz são mostradas separadamente da geração.",
        "",
    ]
    successful = []
    for result in results:
        if result.device == "cpu":
            quantization_label = "INT8" if result.quantized else "sem quantização"
            label = f"CPU — {cpu_threads} thread(s), {quantization_label}"
        elif result.device == "cuda":
            label = (
                "NVIDIA CUDA — INT8 via TorchAO"
                if result.quantized
                else "NVIDIA CUDA — sem quantização INT8"
            )
        else:
            label = result.device.upper()
        lines.append(label)
        if result.error:
            lines.append(f"  Não foi possível medir: {result.error}")
            lines.append("")
            continue
        speed = (
            result.audio_duration_seconds / result.generation_seconds
            if result.generation_seconds > 0
            else 0.0
        )
        lines.extend(
            [
                f"  Carregamento do modelo: {format_duration(result.model_load_seconds)}",
                f"  Preparação da voz: {format_duration(result.voice_prepare_seconds)}",
                (
                    f"  Geração média ({result.repeats} execução(ões)): "
                    f"{format_duration(result.generation_seconds)}"
                ),
                f"  Áudio produzido: {format_duration(result.audio_duration_seconds)}",
                f"  Velocidade: {_format_number(speed, 2)}× tempo real",
                f"  Memória de VRAM de pico: {_format_number(result.peak_vram_mb, 0)} MB",
            ]
        )
        successful.append(result)
        lines.append("")

    by_device = {result.device: result for result in successful}
    cpu = by_device.get("cpu")
    cuda = by_device.get("cuda")
    if cpu and cuda and cpu.generation_seconds > 0 and cuda.generation_seconds > 0:
        ratio = cpu.generation_seconds / cuda.generation_seconds
        if ratio > 1.02:
            lines.append(
                f"Resultado: CUDA foi {_format_number(ratio, 2)}× mais rápida na geração medida."
            )
        elif ratio < 0.98:
            lines.append(
                f"Resultado: CPU foi {_format_number(1 / ratio, 2)}× mais rápida na geração medida."
            )
        else:
            lines.append("Resultado: desempenho semelhante nos dois dispositivos.")
        cpu_quantization = (
            "INT8 ativada" if cpu.quantized else "INT8 desativada"
        )
        lines.append(
            f"Observação: CPU com {cpu_quantization}; CUDA sempre sem quantização INT8. "
            "A comparação representa as configurações reais usadas pelo Joe."
        )
    elif cpu and cuda is None:
        lines.append("Não foi possível obter uma comparação completa entre CPU e CUDA.")
    lines.append(
        "A medição não altera a configuração escolhida para as próximas gerações."
    )
    return "\n".join(lines)


def format_cuda_quantization_benchmark_report(
    results: list[TTSCudaQuantizationBenchmarkResult],
) -> str:
    """Report repeated, warmed CUDA inference timings with INT8 on and off."""
    lines = [
        "Comparação CUDA — Pocket TTS",
        "Mesmo modelo, voz e texto. Medições feitas após aquecimento, com sincronização CUDA.",
        "O cronômetro inclui geração e transferência do áudio para a RAM, como no uso real.",
        "",
    ]
    successful: dict[bool, tuple[TTSCudaQuantizationBenchmarkResult, float, float]] = {}
    for result in results:
        label = "CUDA com INT8 (TorchAO)" if result.quantized else "CUDA sem INT8"
        lines.append(label)
        if result.error:
            lines.append(f"  Não foi possível medir: {result.error}")
            lines.append("")
            continue
        timings = tuple(max(0.0, float(value)) for value in result.generation_times)
        if not timings:
            lines.append("  Nenhuma geração medida.")
            lines.append("")
            continue
        average = statistics.mean(timings)
        median = statistics.median(timings)
        deviation = statistics.stdev(timings) if len(timings) > 1 else 0.0
        audio_durations = tuple(
            max(0.0, float(value)) for value in result.audio_durations_seconds
        )
        average_audio = (
            statistics.mean(audio_durations) if audio_durations else 0.0
        )
        realtime_factor = average_audio / average if average > 0 else 0.0
        lines.extend(
            [
                f"  Aquecimento descartado: {result.warmup_runs} geração(ões)",
                f"  Gerações medidas: {len(timings)}",
                f"  Tempos individuais: {', '.join(format_duration(value) for value in timings)}",
                f"  Média: {format_duration(average)}",
                f"  Mediana: {format_duration(median)}",
                f"  Desvio padrão: {format_duration(deviation)}",
                (
                    f"  Mínimo / máximo: {format_duration(min(timings))} / "
                    f"{format_duration(max(timings))}"
                ),
                f"  Áudio médio produzido: {format_duration(average_audio)}",
                f"  Velocidade média: {_format_number(realtime_factor, 2)}× tempo real",
                f"  Carregamento + transferência para CUDA: {format_duration(result.model_load_seconds)}",
                f"  Aplicação da quantização: {format_duration(result.quantization_seconds)}",
                f"  Preparação da voz: {format_duration(result.voice_prepare_seconds)}",
                f"  Pico VRAM alocada: {_format_number(result.peak_vram_mb, 0)} MB",
                f"  Pico VRAM reservada: {_format_number(result.peak_vram_reserved_mb, 0)} MB",
                "",
            ]
        )
        successful[result.quantized] = (result, average, median)

    plain = successful.get(False)
    quantized = successful.get(True)
    if plain and quantized:
        plain_average = plain[1]
        int8_average = quantized[1]
        if plain_average > 0 and int8_average > 0:
            ratio = int8_average / plain_average
            delta_percent = (ratio - 1.0) * 100.0
            if ratio > 1.02:
                lines.append(
                    f"Resultado: INT8 ficou {_format_number(ratio, 2)}× mais lenta "
                    f"({_format_number(delta_percent, 1)}% mais tempo por geração)."
                )
                lines.append(
                    "CUDA sem quantização foi a opção mais rápida nesta medição."
                )
            elif ratio < 0.98:
                lines.append(
                    f"Resultado: INT8 foi {_format_number(1.0 / ratio, 2)}× mais rápida "
                    f"({_format_number(-delta_percent, 1)}% menos tempo por geração)."
                )
            else:
                lines.append(
                    "Resultado: os dois modos tiveram desempenho semelhante nesta medição."
                )
    else:
        lines.append(
            "Comparação inconclusiva: os dois modos precisam concluir as medições."
        )
    lines.append(
        "A comparação não altera a preferência de quantização nem o modelo escolhido "
        "para as próximas gerações."
    )
    return "\n".join(lines)

def format_cuda_precision_benchmark_report(
    results: list[TTSCudaPrecisionBenchmarkResult],
) -> str:
    """Report FP32 vs FP16 autocast measurements and basic audio integrity checks."""
    lines = [
        "Comparação CUDA — Pocket TTS: FP32 × FP16",
        "Mesmo modelo, voz e texto. FP16 usa autocast nos três pontos internos do Pocket TTS (prefill, geração autoregressiva e decoder).",
        "Os pesos permanecem FP32; apenas operações compatíveis usam precisão FP16.",
        "O cronômetro inclui geração e transferência do áudio para a RAM.",
        "",
    ]
    successful: dict[str, tuple[TTSCudaPrecisionBenchmarkResult, float]] = {}
    for result in results:
        label = (
            "CUDA FP16 autocast (pesos FP32)"
            if result.precision.upper().startswith("FP16")
            else "CUDA FP32 (referência)"
        )
        lines.append(label)
        if result.error:
            lines.append(f"  Não foi possível medir: {result.error}")
            lines.append("")
            continue
        timings = tuple(max(0.0, float(value)) for value in result.generation_times)
        if not timings:
            lines.append("  Nenhuma geração medida.")
            lines.append("")
            continue
        average = statistics.mean(timings)
        median = statistics.median(timings)
        deviation = statistics.stdev(timings) if len(timings) > 1 else 0.0
        audio_durations = tuple(
            max(0.0, float(value)) for value in result.audio_durations_seconds
        )
        average_audio = statistics.mean(audio_durations) if audio_durations else 0.0
        realtime_factor = average_audio / average if average > 0 else 0.0
        peaks = tuple(max(0.0, float(value)) for value in result.audio_peak_amplitudes)
        rms_values = tuple(max(0.0, float(value)) for value in result.audio_rms_levels)
        lines.extend(
            [
                f"  Aquecimento descartado: {result.warmup_runs} geração(ões)",
                f"  Gerações medidas: {len(timings)}",
                f"  Tempos individuais: {', '.join(format_duration(value) for value in timings)}",
                f"  Média: {format_duration(average)}",
                f"  Mediana: {format_duration(median)}",
                f"  Desvio padrão: {format_duration(deviation)}",
                (
                    f"  Mínimo / máximo: {format_duration(min(timings))} / "
                    f"{format_duration(max(timings))}"
                ),
                f"  Áudio médio produzido: {format_duration(average_audio)}",
                f"  Velocidade média: {_format_number(realtime_factor, 2)}× tempo real",
                f"  Carregamento + transferência para CUDA: {format_duration(result.model_load_seconds)}",
                f"  Preparação do autocast FP16: {format_duration(result.autocast_setup_seconds)}",
                f"  Preparação da voz: {format_duration(result.voice_prepare_seconds)}",
                (
                    "  Integridade numérica: amostras finitas (sem NaN/Inf); "
                    f"pico médio {_format_number(statistics.mean(peaks), 4) if peaks else 'n/d'}; "
                    f"RMS médio {_format_number(statistics.mean(rms_values), 4) if rms_values else 'n/d'}"
                ),
                f"  Pico VRAM alocada: {_format_number(result.peak_vram_mb, 0)} MB",
                f"  Pico VRAM reservada: {_format_number(result.peak_vram_reserved_mb, 0)} MB",
                "",
            ]
        )
        successful[result.precision] = (result, average)

    fp32 = next(
        (value for key, value in successful.items() if key.upper() == "FP32"),
        None,
    )
    fp16 = next(
        (value for key, value in successful.items() if key.upper().startswith("FP16")),
        None,
    )
    if fp32 and fp16 and fp32[1] > 0 and fp16[1] > 0:
        ratio = fp16[1] / fp32[1]
        if ratio > 1.02:
            lines.append(
                f"Resultado: FP16 ficou {_format_number(ratio, 2)}× mais lento "
                f"({_format_number((ratio - 1) * 100, 1)}% mais tempo por geração)."
            )
            lines.append("CUDA FP32 foi a opção mais rápida nesta medição.")
        elif ratio < 0.98:
            lines.append(
                f"Resultado: FP16 foi {_format_number(1 / ratio, 2)}× mais rápido "
                f"({_format_number((1 - ratio) * 100, 1)}% menos tempo por geração)."
            )
        else:
            lines.append("Resultado: FP32 e FP16 tiveram desempenho semelhante.")
    else:
        lines.append(
            "Comparação inconclusiva: FP32 e FP16 precisam concluir todas as medições."
        )
    lines.extend(
        [
            "O teste FP16 é experimental e não altera a precisão usada nas gerações normais.",
            "Pico/RMS verificam valores numéricos, mas não substituem a avaliação auditiva da voz.",
        ]
    )
    return "\n".join(lines)

def format_cuda_profile_benchmark_report(
    results: list[TTSCudaProfileBenchmarkResult],
) -> str:
    """Summarize Flow LM, Mimi decoder, queue waits and output transfer timings."""
    lines = [
        "Diagnóstico CUDA — Pocket TTS (FP32)",
        "Mesmo modelo, voz e texto. Duas gerações de aquecimento são descartadas.",
        "Tempos CPU usam relógio de parede; tempos GPU usam CUDA Events, sem sincronizar a cada etapa.",
        "",
    ]
    successful: list[tuple[TTSCudaProfileBenchmarkResult, float]] = []
    for result in results:
        lines.append("CUDA FP32 — diagnóstico por fase")
        if result.error:
            lines.extend([f"  Não foi possível diagnosticar: {result.error}", ""])
            continue
        totals = tuple(max(0.0, float(value)) for value in result.generation_times)
        if not totals:
            lines.extend(["  Nenhuma geração medida.", ""])
            continue

        def mean(values: tuple[float, ...] | tuple[int, ...]) -> float:
            return statistics.mean(values) if values else 0.0

        average_total = mean(totals)
        average_audio = mean(result.audio_durations_seconds)
        lines.extend(
            [
                f"  Gerações medidas: {len(totals)}",
                f"  Tempo total médio: {format_duration(average_total)}",
                f"  Duração média do áudio: {format_duration(average_audio)}",
                (
                    f"  Velocidade: {_format_number(average_audio / average_total, 2) if average_total else '0,00'}× "
                    "tempo real"
                ),
                "",
                "  FLOW LM",
                (
                    f"    Prefill do texto — CPU: "
                    f"{format_duration(mean(result.flow_prefill_wall_seconds))}"
                ),
                (
                    f"    Prefill do texto — GPU: "
                    f"{format_duration(mean(result.flow_prefill_gpu_seconds))}"
                ),
                (
                    f"    Geração autoregressiva — CPU: "
                    f"{format_duration(mean(result.flow_autoregressive_wall_seconds))}"
                ),
                (
                    f"    Geração autoregressiva — GPU: "
                    f"{format_duration(mean(result.flow_autoregressive_gpu_seconds))}"
                ),
                (
                    f"    Passos autoregressivos por áudio: "
                    f"{_format_number(mean(result.flow_step_counts), 1)}"
                ),
                "",
                "  FLOW LM — DETALHAMENTO INTERNO",
                (
                    f"    Atenção (self-attention) — GPU: "
                    f"{format_duration(mean(result.flow_attention_gpu_seconds))}"
                ),
                (
                    f"    Atenção — CPU: "
                    f"{format_duration(mean(result.flow_attention_wall_seconds))}; "
                    f"chamadas: {_format_number(mean(result.flow_attention_calls), 1)}"
                ),
                (
                    f"    FFN (MLP do Transformer) — GPU: "
                    f"{format_duration(mean(result.flow_ffn_gpu_seconds))}"
                ),
                (
                    f"    FFN — CPU: "
                    f"{format_duration(mean(result.flow_ffn_wall_seconds))}; "
                    f"chamadas: {_format_number(mean(result.flow_ffn_calls), 1)}"
                ),
                (
                    f"    FlowNet (projeção do latente) — GPU: "
                    f"{format_duration(mean(result.flow_net_gpu_seconds))}"
                ),
                (
                    f"    FlowNet — CPU: "
                    f"{format_duration(mean(result.flow_net_wall_seconds))}; "
                    f"chamadas: {_format_number(mean(result.flow_net_calls), 1)}"
                ),
                "",
                "  MIMI DECODER",
                (
                    f"    decode_from_latent — CPU: "
                    f"{format_duration(mean(result.mimi_decode_wall_seconds))}"
                ),
                (
                    f"    decode_from_latent — GPU: "
                    f"{format_duration(mean(result.mimi_decode_gpu_seconds))}"
                ),
                (
                    f"    Chamadas ao decoder por áudio: "
                    f"{_format_number(mean(result.mimi_decode_calls), 1)}"
                ),
                (
                    f"    Worker Mimi total (inclui espera): "
                    f"{format_duration(mean(result.mimi_worker_wall_seconds))}"
                ),
                (
                    f"    Espera acumulada na fila de latentes: "
                    f"{format_duration(mean(result.latent_queue_wait_seconds))}"
                ),
                (
                    f"    Chamadas à fila: "
                    f"{_format_number(mean(result.latent_queue_get_calls), 1)}"
                ),
                "",
                "  SAÍDA E MEMÓRIA",
                (
                    f"    Transferência áudio GPU → RAM + sincronização final: "
                    f"{format_duration(mean(result.audio_transfer_seconds))}"
                ),
                f"    Carregamento + transferência do modelo para CUDA: {format_duration(result.model_load_seconds)}",
                f"    Preparação da voz: {format_duration(result.voice_prepare_seconds)}",
                f"    Pico VRAM alocada: {_format_number(result.peak_vram_mb, 0)} MB",
                f"    Pico VRAM reservada: {_format_number(result.peak_vram_reserved_mb, 0)} MB",
                "",
            ]
        )
        successful.append((result, average_total))

    if successful:
        lines.append(
            "Interpretação: CUDA Events medem o tempo de GPU por fase. "
            "Como Flow LM e Mimi trabalham em threads paralelas, esses tempos podem se sobrepor "
            "e não devem ser somados para reconstruir o tempo total."
        )
        lines.append(
            "A espera na fila mede o bloqueio do worker ao aguardar latentes; "
            "não representa tempo de computação da GPU."
        )
        lines.append(
            "O detalhamento interno mede atenção, FFN e FlowNet com CUDA Events. "
            "Os tempos são somas por áudio e não equivalem a tempos de parede exclusivos."
        )
        lines.append(
            "O diagnóstico usa instrumentação por chamadas e eventos CUDA; "
            "há pequena sobrecarga de medição."
        )
    else:
        lines.append("Diagnóstico inconclusivo: nenhuma geração foi concluída.")
    return "\n".join(lines)



def format_cuda_tf32_benchmark_report(
    results: list[TTSCudaPrecisionBenchmarkResult],
) -> str:
    """Report IEEE FP32 against TF32 matmul without changing normal generation settings."""
    lines = [
        "Comparação CUDA — Pocket TTS: FP32 IEEE × TF32",
        "Os pesos, entradas e saídas permanecem FP32; TF32 altera apenas a precisão interna das multiplicações CUDA.",
        "Mesmo modelo, voz e texto. Duas gerações de aquecimento são descartadas.",
        "O cronômetro inclui geração e transferência do áudio para a RAM.",
        "",
    ]
    successful: dict[str, tuple[TTSCudaPrecisionBenchmarkResult, float]] = {}
    for result in results:
        label = (
            "CUDA FP32 com TF32 (Tensor Cores)"
            if result.precision.upper() == "FP32 TF32"
            else "CUDA FP32 IEEE (referência)"
        )
        lines.append(label)
        if result.error:
            lines.extend([f"  Não foi possível medir: {result.error}", ""])
            continue
        timings = tuple(max(0.0, float(value)) for value in result.generation_times)
        if not timings:
            lines.extend(["  Nenhuma geração medida.", ""])
            continue
        average = statistics.mean(timings)
        median = statistics.median(timings)
        deviation = statistics.stdev(timings) if len(timings) > 1 else 0.0
        durations = tuple(
            max(0.0, float(value)) for value in result.audio_durations_seconds
        )
        peaks = tuple(max(0.0, float(value)) for value in result.audio_peak_amplitudes)
        rms_values = tuple(max(0.0, float(value)) for value in result.audio_rms_levels)
        average_audio = statistics.mean(durations) if durations else 0.0
        realtime_factor = average_audio / average if average > 0 else 0.0
        lines.extend(
            [
                f"  Aquecimento descartado: {result.warmup_runs} geração(ões)",
                f"  Gerações medidas: {len(timings)}",
                f"  Tempos individuais: {', '.join(format_duration(value) for value in timings)}",
                f"  Média: {format_duration(average)}",
                f"  Mediana: {format_duration(median)}",
                f"  Desvio padrão: {format_duration(deviation)}",
                (
                    f"  Mínimo / máximo: {format_duration(min(timings))} / "
                    f"{format_duration(max(timings))}"
                ),
                f"  Áudio médio produzido: {format_duration(average_audio)}",
                f"  Velocidade média: {_format_number(realtime_factor, 2)}× tempo real",
                f"  Carregamento + transferência para CUDA: {format_duration(result.model_load_seconds)}",
                f"  Preparação da voz: {format_duration(result.voice_prepare_seconds)}",
                (
                    "  Integridade numérica: amostras finitas (sem NaN/Inf); "
                    f"pico médio {_format_number(statistics.mean(peaks), 4) if peaks else 'n/d'}; "
                    f"RMS médio {_format_number(statistics.mean(rms_values), 4) if rms_values else 'n/d'}"
                ),
                f"  Pico VRAM alocada: {_format_number(result.peak_vram_mb, 0)} MB",
                f"  Pico VRAM reservada: {_format_number(result.peak_vram_reserved_mb, 0)} MB",
                "",
            ]
        )
        successful[result.precision.upper()] = (result, average)

    ieee = successful.get("FP32 IEEE")
    tf32 = successful.get("FP32 TF32")
    if ieee and tf32 and ieee[1] > 0 and tf32[1] > 0:
        ratio = tf32[1] / ieee[1]
        if ratio > 1.02:
            lines.append(
                f"Resultado: TF32 ficou {_format_number(ratio, 2)}× mais lento "
                f"({_format_number((ratio - 1) * 100, 1)}% mais tempo por geração)."
            )
            lines.append("CUDA FP32 IEEE foi a opção mais rápida nesta medição.")
        elif ratio < 0.98:
            lines.append(
                f"Resultado: TF32 foi {_format_number(1 / ratio, 2)}× mais rápido "
                f"({_format_number((1 - ratio) * 100, 1)}% menos tempo por geração)."
            )
        else:
            lines.append("IEEE FP32 e TF32 tiveram desempenho semelhante nesta medição.")
    else:
        lines.append(
            "Comparação inconclusiva: IEEE FP32 e TF32 precisam concluir as medições."
        )
    lines.extend(
        [
            "TF32 é experimental e não altera a precisão usada nas gerações normais.",
            "Pico/RMS verificam valores numéricos, mas não substituem a avaliação auditiva da voz.",
            "A configuração original de precisão CUDA matmul é restaurada ao final do teste.",
        ]
    )
    return "\n".join(lines)


def format_cuda_attention_benchmark_report(
    results: list[TTSCudaAttentionBenchmarkResult],
) -> str:
    """Report default FP32 SDPA against forced Flash SDPA with FP16 Q/K/V."""
    lines = [
        "Comparação CUDA — Pocket TTS: SDPA FP32 × Flash Attention FP16",
        "O Pocket TTS já usa scaled_dot_product_attention; este teste compara os caminhos de execução.",
        "No modo Flash, somente Q/K/V entram em FP16 dentro da atenção; a saída volta para FP32.",
        "Pesos, projeções, FlowNet, Mimi e gerações normais permanecem em FP32.",
        "O backend Flash é forçado, sem fallback silencioso para o kernel matemático.",
        "Mesmo modelo, voz e texto. Duas gerações de aquecimento são descartadas.",
        "O cronômetro inclui geração e transferência do áudio para a RAM.",
        "",
    ]
    successful: dict[str, tuple[TTSCudaAttentionBenchmarkResult, float]] = {}
    for result in results:
        label = (
            "CUDA Flash Attention — SDPA FP16 somente na atenção"
            if result.mode == "Flash FP16 SDPA"
            else "CUDA FP32 — SDPA padrão (referência)"
        )
        lines.append(label)
        if result.error:
            lines.extend([f"  Não foi possível medir: {result.error}", ""])
            continue
        timings = tuple(max(0.0, float(value)) for value in result.generation_times)
        if not timings:
            lines.extend(["  Nenhuma geração medida.", ""])
            continue
        average = statistics.mean(timings)
        median = statistics.median(timings)
        deviation = statistics.stdev(timings) if len(timings) > 1 else 0.0
        durations = tuple(
            max(0.0, float(value)) for value in result.audio_durations_seconds
        )
        peaks = tuple(
            max(0.0, float(value)) for value in result.audio_peak_amplitudes
        )
        rms_values = tuple(
            max(0.0, float(value)) for value in result.audio_rms_levels
        )
        average_audio = statistics.mean(durations) if durations else 0.0
        realtime_factor = average_audio / average if average > 0 else 0.0
        lines.extend(
            [
                f"  Aquecimento descartado: {result.warmup_runs} geração(ões)",
                f"  Gerações medidas: {len(timings)}",
                f"  Tempos individuais: {', '.join(format_duration(value) for value in timings)}",
                f"  Média: {format_duration(average)}",
                f"  Mediana: {format_duration(median)}",
                f"  Desvio padrão: {format_duration(deviation)}",
                (
                    f"  Mínimo / máximo: {format_duration(min(timings))} / "
                    f"{format_duration(max(timings))}"
                ),
                f"  Áudio médio produzido: {format_duration(average_audio)}",
                f"  Velocidade média: {_format_number(realtime_factor, 2)}× tempo real",
                f"  Carregamento + transferência para CUDA: {format_duration(result.model_load_seconds)}",
                f"  Preparação da voz: {format_duration(result.voice_prepare_seconds)}",
                (
                    "  Integridade numérica: amostras finitas (sem NaN/Inf); "
                    f"pico médio {_format_number(statistics.mean(peaks), 4) if peaks else 'n/d'}; "
                    f"RMS médio {_format_number(statistics.mean(rms_values), 4) if rms_values else 'n/d'}"
                ),
                f"  Pico VRAM alocada: {_format_number(result.peak_vram_mb, 0)} MB",
                f"  Pico VRAM reservada: {_format_number(result.peak_vram_reserved_mb, 0)} MB",
                "",
            ]
        )
        successful[result.mode] = (result, average)

    reference = successful.get("FP32 SDPA")
    flash = successful.get("Flash FP16 SDPA")
    if reference and flash and reference[1] > 0 and flash[1] > 0:
        ratio = flash[1] / reference[1]
        delta_percent = (ratio - 1.0) * 100.0
        if ratio < 0.98:
            lines.append(
                f"Resultado: Flash FP16 foi {_format_number(1.0 / ratio, 2)}× mais rápido "
                f"({_format_number(-delta_percent, 1)}% menos tempo por geração)."
            )
        elif ratio > 1.02:
            lines.append(
                f"Resultado: Flash FP16 ficou {_format_number(ratio, 2)}× mais lento "
                f"({_format_number(delta_percent, 1)}% mais tempo por geração)."
            )
        else:
            lines.append(
                "Resultado: SDPA FP32 e Flash FP16 tiveram desempenho semelhante."
            )
    else:
        lines.append(
            "Comparação inconclusiva: os dois modos precisam concluir as medições."
        )
    lines.extend(
        [
            "O benchmark é experimental e não altera a precisão usada nas gerações normais.",
            "Pico/RMS verificam integridade numérica, mas não substituem a avaliação auditiva.",
            "Se Flash não suportar a forma da atenção, o teste registra o erro e não usa fallback.",
        ]
    )
    return "\n".join(lines)
