from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from uuid import uuid4


@dataclass
class GenerationTask:
    """One immutable TTS request plus its mutable in-memory execution state."""

    text: str = field(repr=False)
    speech_text: str = field(repr=False)
    language: str
    output_path: str
    model_variant: str = ""
    tts_device: str = "cpu"
    cpu_threads: int = 2
    tts_quantization: bool = True
    tts_sampler_decode_steps: int = 1
    tts_profile: str = "custom"
    voice_path: str | None = None
    voice_alias: str = ""
    remote: bool = False
    remote_url: str = ""
    remote_token: str = field(default="", repr=False)
    urgent: bool = False
    task_id: str = field(default_factory=lambda: uuid4().hex)
    sequence: int = 0
    status: str = "queued"
    progress: str = "Aguardando"
    error: str = ""
    attempts: int = 0
    created_at: float = field(default_factory=monotonic, repr=False)

    @property
    def output_name(self) -> str:
        return Path(self.output_path).name


class GenerationQueue:
    """Session-only FIFO queue with stable urgent ordering and retry support."""

    FINISHED_STATES = frozenset({"completed", "failed", "cancelled"})

    def __init__(self) -> None:
        self._tasks: dict[str, GenerationTask] = {}
        self._next_sequence = 0
        self._active_id: str | None = None

    def enqueue(
        self,
        *,
        text: str,
        speech_text: str,
        language: str,
        output_path: str,
        model_variant: str | None = None,
        tts_device: str = "cpu",
        cpu_threads: int = 2,
        tts_quantization: bool = True,
        tts_sampler_decode_steps: int = 1,
        tts_profile: str = "custom",
        voice_path: str | None = None,
        voice_alias: str = "",
        remote: bool = False,
        remote_url: str = "",
        remote_token: str = "",
        urgent: bool = False,
    ) -> GenerationTask:
        if not text.strip() or not speech_text.strip():
            raise ValueError("O texto da tarefa não pode ficar vazio.")
        if not output_path.strip():
            raise ValueError("A tarefa precisa de um caminho de saída.")
        if tts_device not in {"cpu", "cuda"}:
            raise ValueError("Dispositivo TTS inválido. Escolha cpu ou cuda.")
        if int(cpu_threads) < 1:
            raise ValueError("A quantidade de threads da CPU precisa ser maior que zero.")
        if not 1 <= int(tts_sampler_decode_steps) <= 16:
            raise ValueError("Os passos do sampler precisam ficar entre 1 e 16.")
        task = GenerationTask(
            text=text,
            speech_text=speech_text,
            language=language,
            output_path=output_path,
            model_variant=model_variant or language,
            tts_device=tts_device,
            cpu_threads=int(cpu_threads),
            tts_quantization=bool(tts_quantization),
            tts_sampler_decode_steps=int(tts_sampler_decode_steps),
            tts_profile=str(tts_profile or "custom"),
            voice_path=voice_path,
            voice_alias=voice_alias,
            remote=remote,
            remote_url=remote_url,
            remote_token=remote_token,
            urgent=urgent,
            sequence=self._next_sequence,
        )
        self._next_sequence += 1
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> GenerationTask | None:
        return self._tasks.get(task_id)

    def tasks(self) -> list[GenerationTask]:
        return sorted(self._tasks.values(), key=lambda task: task.sequence)

    def pending_count(self) -> int:
        return sum(task.status == "queued" for task in self._tasks.values())

    def active_task(self) -> GenerationTask | None:
        return self._tasks.get(self._active_id) if self._active_id else None

    def next_pending(self) -> GenerationTask | None:
        if self._active_id is not None:
            return None
        candidates = [task for task in self._tasks.values() if task.status == "queued"]
        if not candidates:
            return None
        return min(candidates, key=lambda task: (not task.urgent, task.sequence))

    def start(self, task_id: str) -> GenerationTask:
        task = self._require(task_id)
        if self._active_id is not None:
            raise RuntimeError("Já existe uma tarefa de geração em execução.")
        if task.status != "queued":
            raise ValueError("Somente tarefas aguardando podem ser iniciadas.")
        task.status = "running"
        task.progress = "Iniciando"
        self._active_id = task_id
        return task

    def update_progress(self, task_id: str, message: str) -> None:
        task = self._require(task_id)
        if task.status == "running":
            task.progress = message

    def complete(self, task_id: str) -> GenerationTask:
        task = self._require(task_id)
        self._finish_active(task_id)
        task.status = "completed"
        task.progress = "Concluído"
        task.error = ""
        return task

    def fail(self, task_id: str, message: str) -> GenerationTask:
        task = self._require(task_id)
        self._finish_active(task_id)
        task.status = "failed"
        task.progress = "Falhou"
        task.error = message.strip() or "Erro desconhecido."
        return task

    def request_cancel(self, task_id: str) -> str:
        task = self._require(task_id)
        if task.status == "queued":
            task.status = "cancelled"
            task.progress = "Cancelado antes de iniciar"
        elif task.status == "running":
            task.status = "cancelling"
            task.progress = "Solicitando cancelamento"
        return task.status

    def mark_cancelled(self, task_id: str) -> GenerationTask:
        task = self._require(task_id)
        self._finish_active(task_id)
        task.status = "cancelled"
        task.progress = "Cancelado"
        task.error = ""
        return task

    def retry(self, task_id: str, *, urgent: bool = False) -> GenerationTask:
        task = self._require(task_id)
        if task.status not in {"failed", "cancelled"}:
            raise ValueError("Somente tarefas com falha ou canceladas podem ser repetidas.")
        if self._active_id == task_id:
            raise RuntimeError("A tarefa ainda está ativa.")
        task.status = "queued"
        task.progress = "Aguardando repetição"
        task.error = ""
        task.attempts += 1
        task.urgent = urgent or task.urgent
        return task

    def prioritize(self, task_id: str) -> bool:
        task = self._require(task_id)
        if task.status != "queued" or task.urgent:
            return False
        task.urgent = True
        return True

    def remove_finished(self) -> int:
        removable = [
            task_id
            for task_id, task in self._tasks.items()
            if task.status in self.FINISHED_STATES and task_id != self._active_id
        ]
        for task_id in removable:
            del self._tasks[task_id]
        return len(removable)

    def cancel_pending(self) -> int:
        count = 0
        for task in self._tasks.values():
            if task.status == "queued":
                task.status = "cancelled"
                task.progress = "Cancelado antes de iniciar"
                count += 1
        return count

    def _finish_active(self, task_id: str) -> None:
        if self._active_id == task_id:
            self._active_id = None

    def _require(self, task_id: str) -> GenerationTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Tarefa de geração não encontrada: {task_id}") from exc
