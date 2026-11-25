from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class HardwareMetrics:
    """Snapshot of CPU/RAM and the first NVIDIA GPU, when available."""

    cpu_percent: float | None = None
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    gpu_name: str = ""
    gpu_utilization_percent: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    gpu_temperature_c: float | None = None
    gpu_error: str = ""


def _number(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_hardware_metrics(
    *,
    psutil_module: Any | None = None,
    command_runner: Callable[..., Any] | None = None,
) -> HardwareMetrics:
    """Read CPU/RAM via psutil and NVIDIA telemetry via nvidia-smi."""
    if psutil_module is None:
        import psutil as psutil_module

    cpu_percent = None
    ram_used_mb = None
    ram_total_mb = None
    try:
        cpu_percent = float(psutil_module.cpu_percent(interval=None))
    except Exception:  # noqa: BLE001
        pass
    try:
        memory = psutil_module.virtual_memory()
        ram_used_mb = float(memory.used) / (1024**2)
        ram_total_mb = float(memory.total) / (1024**2)
    except Exception:  # noqa: BLE001
        pass

    runner = command_runner or subprocess.run
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        output = str(result.stdout or "").strip()
        if not output:
            return HardwareMetrics(
                cpu_percent=cpu_percent,
                ram_used_mb=ram_used_mb,
                ram_total_mb=ram_total_mb,
                gpu_error="A NVIDIA GPU não retornou métricas.",
            )
        fields = next(csv.reader([output.splitlines()[0]]))
        if len(fields) < 5:
            raise ValueError("Resposta incompleta do nvidia-smi.")
        return HardwareMetrics(
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            gpu_name=fields[0].strip(),
            gpu_utilization_percent=_number(fields[1]),
            vram_used_mb=_number(fields[2]),
            vram_total_mb=_number(fields[3]),
            gpu_temperature_c=_number(fields[4]),
        )
    except FileNotFoundError:
        gpu_error = "nvidia-smi não encontrado; métricas da GPU indisponíveis."
    except subprocess.TimeoutExpired:
        gpu_error = "Tempo esgotado ao consultar nvidia-smi."
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).strip()
        gpu_error = f"Não foi possível consultar a GPU: {detail}" if detail else (
            "Não foi possível consultar a GPU."
        )

    return HardwareMetrics(
        cpu_percent=cpu_percent,
        ram_used_mb=ram_used_mb,
        ram_total_mb=ram_total_mb,
        gpu_error=gpu_error,
    )
