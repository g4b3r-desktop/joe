from __future__ import annotations

from types import SimpleNamespace

from joe.system_monitor import read_hardware_metrics


def test_reads_cpu_ram_and_nvidia_metrics():
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 37.5,
        virtual_memory=lambda: SimpleNamespace(used=4 * 1024**3, total=16 * 1024**3),
    )

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout="NVIDIA GeForce RTX 3060, 42, 5120, 12288, 51\n"
        )

    metrics = read_hardware_metrics(
        psutil_module=fake_psutil,
        command_runner=fake_run,
    )

    assert metrics.cpu_percent == 37.5
    assert metrics.ram_used_mb == 4096
    assert metrics.ram_total_mb == 16384
    assert metrics.gpu_name == "NVIDIA GeForce RTX 3060"
    assert metrics.gpu_utilization_percent == 42
    assert metrics.vram_used_mb == 5120
    assert metrics.vram_total_mb == 12288
    assert metrics.gpu_temperature_c == 51


def test_missing_nvidia_smi_does_not_break_cpu_ram_metrics():
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 10,
        virtual_memory=lambda: SimpleNamespace(used=1024**3, total=8 * 1024**3),
    )

    def missing_command(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    metrics = read_hardware_metrics(
        psutil_module=fake_psutil,
        command_runner=missing_command,
    )

    assert metrics.cpu_percent == 10
    assert metrics.ram_used_mb == 1024
    assert metrics.ram_total_mb == 8192
    assert metrics.gpu_name == ""
    assert "não encontrado" in metrics.gpu_error


def test_unavailable_gpu_fields_are_reported_as_none():
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 0,
        virtual_memory=lambda: SimpleNamespace(used=0, total=1024),
    )

    metrics = read_hardware_metrics(
        psutil_module=fake_psutil,
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            stdout="NVIDIA GPU, N/A, N/A, N/A, N/A\n"
        ),
    )

    assert metrics.gpu_name == "NVIDIA GPU"
    assert metrics.gpu_utilization_percent is None
    assert metrics.vram_used_mb is None
    assert metrics.gpu_temperature_c is None
