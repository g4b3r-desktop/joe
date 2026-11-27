from __future__ import annotations

import pytest

from joe.generation_queue import GenerationQueue


def add(queue: GenerationQueue, name: str, *, urgent: bool = False):
    return queue.enqueue(
        text=name,
        speech_text=name,
        language="portuguese",
        output_path=f"/tmp/{name}.wav",
        urgent=urgent,
    )


def test_queue_is_fifo_and_urgent_tasks_jump_ahead_of_pending_tasks():
    queue = GenerationQueue()
    first = add(queue, "first")
    second = add(queue, "second")
    urgent = add(queue, "urgent", urgent=True)

    assert queue.next_pending() == urgent
    queue.start(urgent.task_id)
    assert queue.next_pending() is None
    queue.complete(urgent.task_id)

    assert queue.next_pending() == first
    queue.start(first.task_id)
    queue.complete(first.task_id)
    assert queue.next_pending() == second


def test_queue_tracks_progress_and_completes_active_task():
    queue = GenerationQueue()
    task = add(queue, "voice")
    queue.start(task.task_id)
    queue.update_progress(task.task_id, "Trecho 2 de 3")

    assert task.progress == "Trecho 2 de 3"
    assert queue.active_task() is task

    queue.complete(task.task_id)

    assert task.status == "completed"
    assert queue.active_task() is None


def test_failed_task_preserves_multiline_error_diagnostics():
    queue = GenerationQueue()
    task = add(queue, "voice")
    queue.start(task.task_id)
    diagnostic = (
        "Falha ao gerar o arquivo voice.wav.\n"
        "Erro: RuntimeError: falha no modelo\n\n"
        "Traceback técnico:\nTraceback (most recent call last): ..."
    )

    queue.fail(task.task_id, diagnostic)

    assert task.status == "failed"
    assert task.error == diagnostic


def test_failed_task_can_be_retried_and_keeps_its_request():
    queue = GenerationQueue()
    task = add(queue, "voice")
    queue.start(task.task_id)
    queue.fail(task.task_id, "Falha temporária")

    retried = queue.retry(task.task_id, urgent=True)

    assert retried is task
    assert task.status == "queued"
    assert task.error == ""
    assert task.attempts == 1
    assert task.urgent
    assert task.text == "voice"


def test_active_cancel_is_cooperative_and_pending_cancel_is_immediate():
    queue = GenerationQueue()
    active = add(queue, "active")
    pending = add(queue, "pending")
    queue.start(active.task_id)

    assert queue.request_cancel(active.task_id) == "cancelling"
    assert queue.request_cancel(pending.task_id) == "cancelled"
    queue.mark_cancelled(active.task_id)

    assert active.status == "cancelled"
    assert pending.status == "cancelled"
    assert queue.active_task() is None


def test_only_failed_or_cancelled_tasks_can_be_retried():
    queue = GenerationQueue()
    task = add(queue, "voice")

    with pytest.raises(ValueError, match="falha ou canceladas"):
        queue.retry(task.task_id)


def test_remove_finished_preserves_waiting_tasks():
    queue = GenerationQueue()
    completed = add(queue, "completed")
    waiting = add(queue, "waiting")
    queue.start(completed.task_id)
    queue.complete(completed.task_id)

    assert queue.remove_finished() == 1
    assert queue.get(completed.task_id) is None
    assert queue.get(waiting.task_id) is waiting



def test_queue_preserves_voice_alias_in_the_task_snapshot():
    queue = GenerationQueue()
    task = queue.enqueue(
        text="Olá",
        speech_text="Olá",
        language="portuguese",
        output_path="/tmp/hello.wav",
        voice_path="/voices/voice.wav",
        voice_alias="Narradora",
    )

    assert task.voice_path == "/voices/voice.wav"
    assert task.voice_alias == "Narradora"



def test_queue_preserves_voice_alias_in_the_task_snapshot():
    queue = GenerationQueue()
    task = queue.enqueue(
        text="Olá",
        speech_text="Olá",
        language="portuguese",
        output_path="/tmp/hello.wav",
        voice_path="/voices/voice.wav",
        voice_alias="Narradora",
    )

    assert task.voice_path == "/voices/voice.wav"
    assert task.voice_alias == "Narradora"


def test_queue_snapshots_selected_model_variant():
    queue = GenerationQueue()
    task = queue.enqueue(
        text="Olá",
        speech_text="Olá",
        language="portuguese",
        model_variant="portuguese_24l",
        output_path="/tmp/hello.wav",
    )

    assert task.model_variant == "portuguese_24l"



def test_queue_snapshots_tts_compute_device_and_cpu_threads():
    queue = GenerationQueue()
    task = queue.enqueue(
        text="Olá",
        speech_text="Olá",
        language="portuguese",
        output_path="/tmp/hello.wav",
        tts_device="cuda",
        cpu_threads=4,
        tts_quantization=False,
        tts_profile="performance",
    )

    assert task.tts_device == "cuda"
    assert task.cpu_threads == 4
    assert task.tts_quantization is False
    assert task.tts_profile == "performance"



def test_queue_snapshots_sampler_decode_steps_and_rejects_invalid_values():
    queue = GenerationQueue()
    task = queue.enqueue(
        text="Olá",
        speech_text="Olá",
        language="portuguese",
        output_path="/tmp/sampler.wav",
        tts_sampler_decode_steps=3,
    )

    assert task.tts_sampler_decode_steps == 3

    with pytest.raises(ValueError, match="passos do sampler"):
        queue.enqueue(
            text="Olá",
            speech_text="Olá",
            language="portuguese",
            output_path="/tmp/invalid-sampler.wav",
            tts_sampler_decode_steps=17,
        )
