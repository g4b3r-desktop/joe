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
