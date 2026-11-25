from __future__ import annotations

import wave

from joe.audio_performance import (
    AudioPerformance,
    TTSBenchmarkResult,
    format_duration,
    format_elapsed_status,
    format_performance_report,
    format_tts_benchmark_report,
    wav_duration_seconds,
)


def test_wav_duration_is_read_without_loading_audio_samples(tmp_path):
    audio_path = tmp_path / "one-second.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8000)
        audio_file.writeframes(b"\x00\x00" * 8000)

    assert wav_duration_seconds(audio_path) == 1.0


def test_invalid_audio_returns_zero_duration(tmp_path):
    invalid_audio = tmp_path / "not-a-wave.wav"
    invalid_audio.write_text("not audio", encoding="utf-8")

    assert wav_duration_seconds(invalid_audio) == 0.0


def test_duration_format_is_consistent():
    assert format_duration(65.26) == "01:05.3"
    assert format_duration(3661.26) == "01:01:01.3"
    assert format_duration(2.26) == "2,3s"


def test_elapsed_status_uses_shared_duration_format():
    assert format_elapsed_status(12.26) == "Tempo decorrido: 12,3s"


def test_performance_report_uses_desktop_measurement_for_remote_mode():
    report = format_performance_report(
        AudioPerformance(
            elapsed_seconds=2.0,
            audio_duration_seconds=6.0,
            character_count=100,
            remote=True,
        )
    )

    assert "Medição realizada pelo Joe neste computador" in report
    assert "Tempo total: 2,0s" in report
    assert "Velocidade efetiva: 3,00× tempo real" in report
    assert "Texto: 50,0 caracteres/s" in report
    assert "cronômetro inclui envio, fila e espera" in report



def test_tts_benchmark_report_compares_cpu_and_cuda_results():
    from joe.audio_performance import (
        TTSBenchmarkResult,
        format_tts_benchmark_report,
    )

    report = format_tts_benchmark_report(
        [
            TTSBenchmarkResult(
                device="cpu",
                cpu_threads=4,
                model_load_seconds=3.0,
                voice_prepare_seconds=1.0,
                generation_seconds=2.0,
                audio_duration_seconds=4.0,
                peak_vram_mb=0.0,
                quantized=True,
                repeats=2,
            ),
            TTSBenchmarkResult(
                device="cuda",
                cpu_threads=4,
                model_load_seconds=2.0,
                voice_prepare_seconds=0.5,
                generation_seconds=1.0,
                audio_duration_seconds=4.0,
                peak_vram_mb=1200.0,
                quantized=False,
                repeats=2,
            ),
        ],
        cpu_threads=4,
    )

    assert "CPU — 4 thread(s), INT8" in report
    assert "NVIDIA CUDA" in report
    assert "Carregamento do modelo" in report
    assert "Preparação da voz" in report
    assert "Memória de VRAM de pico: 1200 MB" in report
    assert "CUDA foi 2,00× mais rápida" in report
    assert "não altera a configuração escolhida" in report



def test_tts_benchmark_report_shows_when_cpu_quantization_is_disabled():
    report = format_tts_benchmark_report(
        [
            TTSBenchmarkResult(
                device="cpu",
                cpu_threads=4,
                generation_seconds=2.0,
                audio_duration_seconds=4.0,
                quantized=False,
                repeats=1,
            ),
            TTSBenchmarkResult(
                device="cuda",
                cpu_threads=4,
                generation_seconds=1.0,
                audio_duration_seconds=4.0,
                quantized=False,
                repeats=1,
            ),
        ],
        cpu_threads=4,
    )

    assert "CPU — 4 thread(s), sem quantização" in report
    assert "CPU com INT8 desativada" in report
