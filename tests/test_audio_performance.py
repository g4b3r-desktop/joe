from __future__ import annotations

import wave

from joe.audio_performance import (
    AudioPerformance,
    TTSBenchmarkResult,
    format_duration,
    format_elapsed_status,
    format_performance_report,
    format_tts_benchmark_report,
    TTSCudaQuantizationBenchmarkResult,
    format_cuda_quantization_benchmark_report,
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


def test_cuda_quantization_report_detects_four_times_slower_int8():
    report = format_cuda_quantization_benchmark_report(
        [
            TTSCudaQuantizationBenchmarkResult(
                quantized=False,
                generation_times=(1.0, 1.1, 0.9, 1.0, 1.0),
                audio_durations_seconds=(4.0, 4.0, 4.0, 4.0, 4.0),
                warmup_runs=2,
                model_load_seconds=2.0,
                peak_vram_mb=1800,
                peak_vram_reserved_mb=2200,
            ),
            TTSCudaQuantizationBenchmarkResult(
                quantized=True,
                generation_times=(4.0, 4.1, 3.9, 4.0, 4.0),
                audio_durations_seconds=(4.0, 4.0, 4.0, 4.0, 4.0),
                warmup_runs=2,
                model_load_seconds=2.2,
                quantization_seconds=0.7,
                peak_vram_mb=1300,
                peak_vram_reserved_mb=1700,
            ),
        ]
    )

    assert "Gerações medidas: 5" in report
    assert "Tempos individuais:" in report
    assert "Média:" in report
    assert "Mediana:" in report
    assert "Desvio padrão:" in report
    assert "Pico VRAM alocada:" in report
    assert "Aplicação da quantização:" in report
    assert "INT8 ficou 4,00× mais lenta" in report
    assert "CUDA sem quantização foi a opção mais rápida" in report


def test_cuda_precision_report_compares_fp32_and_fp16_autocast():
    from joe.audio_performance import (
        TTSCudaPrecisionBenchmarkResult,
        format_cuda_precision_benchmark_report,
    )

    report = format_cuda_precision_benchmark_report(
        [
            TTSCudaPrecisionBenchmarkResult(
                precision="FP32",
                generation_times=(2.0, 2.1, 1.9),
                audio_durations_seconds=(4.0, 4.0, 4.0),
                audio_peak_amplitudes=(0.5, 0.5, 0.5),
                audio_rms_levels=(0.1, 0.1, 0.1),
                warmup_runs=2,
                model_load_seconds=3.0,
                peak_vram_mb=1500,
                peak_vram_reserved_mb=1700,
            ),
            TTSCudaPrecisionBenchmarkResult(
                precision="FP16 autocast",
                generation_times=(1.0, 1.1, 0.9),
                audio_durations_seconds=(4.0, 4.0, 4.0),
                audio_peak_amplitudes=(0.5, 0.5, 0.5),
                audio_rms_levels=(0.1, 0.1, 0.1),
                warmup_runs=2,
                autocast_setup_seconds=0.01,
                model_load_seconds=3.1,
                peak_vram_mb=1450,
                peak_vram_reserved_mb=1650,
            ),
        ]
    )

    assert "CUDA FP32 (referência)" in report
    assert "CUDA FP16 autocast (pesos FP32)" in report
    assert "Integridade numérica: amostras finitas" in report
    assert "FP16 foi 2,00× mais rápido" in report
    assert "não altera a precisão usada nas gerações normais" in report


def test_cuda_precision_report_marks_incomplete_comparison():
    from joe.audio_performance import (
        TTSCudaPrecisionBenchmarkResult,
        format_cuda_precision_benchmark_report,
    )

    report = format_cuda_precision_benchmark_report(
        [
            TTSCudaPrecisionBenchmarkResult(
                precision="FP32",
                error="simulated CUDA error",
            ),
            TTSCudaPrecisionBenchmarkResult(precision="FP16 autocast"),
        ]
    )

    assert "simulated CUDA error" in report
    assert "Comparação inconclusiva" in report



def test_cuda_tf32_report_compares_ieee_and_tf32_without_changing_default():
    from joe.audio_performance import (
        TTSCudaPrecisionBenchmarkResult,
        format_cuda_tf32_benchmark_report,
    )

    report = format_cuda_tf32_benchmark_report(
        [
            TTSCudaPrecisionBenchmarkResult(
                precision="FP32 IEEE",
                generation_times=(4.0, 4.1, 3.9),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                audio_peak_amplitudes=(0.5, 0.5, 0.5),
                audio_rms_levels=(0.1, 0.1, 0.1),
                warmup_runs=2,
                peak_vram_mb=1500,
                peak_vram_reserved_mb=1650,
            ),
            TTSCudaPrecisionBenchmarkResult(
                precision="FP32 TF32",
                generation_times=(3.0, 3.1, 2.9),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                audio_peak_amplitudes=(0.49, 0.49, 0.49),
                audio_rms_levels=(0.099, 0.099, 0.099),
                warmup_runs=2,
                peak_vram_mb=1500,
                peak_vram_reserved_mb=1650,
            ),
        ]
    )

    assert "CUDA FP32 IEEE (referência)" in report
    assert "CUDA FP32 com TF32" in report
    assert "TF32 foi 1,33× mais rápido" in report
    assert "pesos, entradas e saídas permanecem FP32" in report
    assert "configuração original de precisão CUDA matmul é restaurada" in report

def test_cuda_profile_report_separates_flow_lm_mimi_and_queue_wait():
    from joe.audio_performance import (
        TTSCudaProfileBenchmarkResult,
        format_cuda_profile_benchmark_report,
    )

    report = format_cuda_profile_benchmark_report(
        [
            TTSCudaProfileBenchmarkResult(
                model_load_seconds=3.0,
                voice_prepare_seconds=0.5,
                generation_times=(4.5, 4.7, 4.6),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                audio_transfer_seconds=(0.1, 0.1, 0.1),
                flow_prefill_wall_seconds=(0.2, 0.2, 0.2),
                flow_prefill_gpu_seconds=(0.1, 0.1, 0.1),
                flow_autoregressive_wall_seconds=(2.0, 2.1, 2.0),
                flow_autoregressive_gpu_seconds=(1.5, 1.6, 1.5),
                flow_step_counts=(30, 31, 30),
                flow_attention_wall_seconds=(0.2, 0.2, 0.2),
                flow_attention_gpu_seconds=(0.15, 0.16, 0.15),
                flow_attention_calls=(30, 31, 30),
                flow_ffn_wall_seconds=(0.3, 0.3, 0.3),
                flow_ffn_gpu_seconds=(0.25, 0.26, 0.25),
                flow_ffn_calls=(30, 31, 30),
                flow_net_wall_seconds=(0.05, 0.05, 0.05),
                flow_net_gpu_seconds=(0.04, 0.04, 0.04),
                flow_net_calls=(30, 31, 30),
                mimi_decode_wall_seconds=(0.5, 0.6, 0.5),
                mimi_decode_gpu_seconds=(0.4, 0.45, 0.4),
                mimi_decode_calls=(8, 9, 8),
                mimi_worker_wall_seconds=(3.0, 3.1, 3.0),
                latent_queue_wait_seconds=(1.8, 1.9, 1.8),
                latent_queue_get_calls=(10, 11, 10),
                warmup_runs=2,
                peak_vram_mb=1526,
                peak_vram_reserved_mb=1644,
            )
        ]
    )

    assert "FLOW LM" in report
    assert "Prefill do texto — GPU" in report
    assert "Geração autoregressiva — GPU" in report
    assert "FLOW LM — DETALHAMENTO INTERNO" in report
    assert "Atenção (self-attention) — GPU" in report
    assert "FFN (MLP do Transformer) — GPU" in report
    assert "FlowNet (projeção do latente) — GPU" in report
    assert "MIMI DECODER" in report
    assert "decode_from_latent — GPU" in report
    assert "Worker Mimi total (inclui espera)" in report
    assert "Espera acumulada na fila de latentes" in report
    assert "Transferência áudio GPU → RAM" in report
    assert "não devem ser somados" in report


def test_cuda_profile_report_handles_failed_diagnostic():
    from joe.audio_performance import (
        TTSCudaProfileBenchmarkResult,
        format_cuda_profile_benchmark_report,
    )

    report = format_cuda_profile_benchmark_report(
        [TTSCudaProfileBenchmarkResult(error="missing internal Pocket TTS hooks")]
    )

    assert "missing internal Pocket TTS hooks" in report
    assert "Diagnóstico inconclusivo" in report



def test_cuda_attention_report_compares_default_sdpa_and_flash_fp16():
    from joe.audio_performance import (
        TTSCudaAttentionBenchmarkResult,
        format_cuda_attention_benchmark_report,
    )

    report = format_cuda_attention_benchmark_report(
        [
            TTSCudaAttentionBenchmarkResult(
                mode="FP32 SDPA",
                generation_times=(5.0, 5.1, 4.9),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                audio_peak_amplitudes=(0.50, 0.50, 0.50),
                audio_rms_levels=(0.10, 0.10, 0.10),
                warmup_runs=2,
                peak_vram_mb=1500,
                peak_vram_reserved_mb=1700,
            ),
            TTSCudaAttentionBenchmarkResult(
                mode="Flash FP16 SDPA",
                generation_times=(4.0, 4.1, 3.9),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                audio_peak_amplitudes=(0.49, 0.49, 0.49),
                audio_rms_levels=(0.099, 0.099, 0.099),
                warmup_runs=2,
                peak_vram_mb=1450,
                peak_vram_reserved_mb=1650,
            ),
        ]
    )

    assert "scaled_dot_product_attention" in report
    assert "Flash Attention" in report
    assert "somente Q/K/V" in report
    assert "Flash FP16 foi 1,25× mais rápido" in report
    assert "não altera a precisão usada nas gerações normais" in report


def test_cuda_attention_report_exposes_flash_backend_failures():
    from joe.audio_performance import (
        TTSCudaAttentionBenchmarkResult,
        format_cuda_attention_benchmark_report,
    )

    report = format_cuda_attention_benchmark_report(
        [
            TTSCudaAttentionBenchmarkResult(
                mode="FP32 SDPA",
                error="baseline failed",
            ),
            TTSCudaAttentionBenchmarkResult(
                mode="Flash FP16 SDPA",
                error="Flash kernel does not support this mask",
            ),
        ]
    )

    assert "Flash kernel does not support this mask" in report
    assert "Comparação Flash FP16: inconclusiva" in report
    assert "sem fallback silencioso" in report


def test_cuda_attention_report_includes_memory_efficient_backend():
    from joe.audio_performance import (
        TTSCudaAttentionBenchmarkResult,
        format_cuda_attention_benchmark_report,
    )

    report = format_cuda_attention_benchmark_report(
        [
            TTSCudaAttentionBenchmarkResult(
                mode="FP32 SDPA",
                generation_times=(5.0, 5.1, 4.9),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                warmup_runs=2,
            ),
            TTSCudaAttentionBenchmarkResult(
                mode="Flash FP16 SDPA",
                error="Flash kernel not eligible",
            ),
            TTSCudaAttentionBenchmarkResult(
                mode="Efficient FP16 SDPA",
                generation_times=(4.0, 4.1, 3.9),
                audio_durations_seconds=(6.0, 6.0, 6.0),
                warmup_runs=2,
            ),
        ]
    )

    assert "Flash kernel not eligible" in report
    assert "Memory-Efficient FP16" in report
    assert "Flash e Memory-Efficient são testados separadamente" in report
    assert "Memory-Efficient FP16 foi 1,25× mais rápido" in report
