from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave


@dataclass(frozen=True, slots=True)
class AudioPerformance:
    """Métricas medidas pelo próprio aplicativo durante uma geração de áudio."""

    elapsed_seconds: float
    audio_duration_seconds: float
    character_count: int
    remote: bool = False


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
        "Uma geração de aquecimento é descartada antes das repetições cronometradas.",
        "",
    ]
    successful = []
    for result in results:
        if result.device == "cpu":
            quantization_label = "INT8" if result.quantized else "sem quantização"
            label = f"CPU — {cpu_threads} thread(s), {quantization_label}"
        elif result.device == "cuda":
            label = "NVIDIA CUDA — sem quantização INT8"
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
