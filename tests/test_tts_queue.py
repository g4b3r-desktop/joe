from __future__ import annotations

from pathlib import Path

from joe.tts_queue import TTSJobQueue


def test_queue_runs_tasks_in_fifo_order(tmp_path: Path):
    queue = TTSJobQueue()
    first = queue.enqueue("Primeiro", tmp_path / "one.wav", lambda: "one")
    second = queue.enqueue("Segundo", tmp_path / "two.wav", lambda: "two")

    assert queue.pending_count == 2
    assert queue.start_next() is first
    assert first.status == "Gerando"
    assert queue.start_next() is None
    assert queue.finish_current(first.task_id, "Concluído")
    assert queue.start_next() is second
    assert queue.finish_current(second.task_id, "Concluído")
    assert queue.active_count == 0


def test_cancel_current_is_cooperative_and_idempotent(tmp_path: Path):
    queue = TTSJobQueue()
    entry = queue.enqueue("Tarefa", tmp_path / "task.wav", lambda: None)
    queue.start_next()

    assert queue.request_cancel_current()
    assert entry.cancel_event.is_set()
    assert entry.status == "Cancelamento solicitado"
    assert not queue.request_cancel_current()
    assert queue.finish_current(entry.task_id, "Cancelado")


def test_pending_item_can_be_removed_without_affecting_current(tmp_path: Path):
    queue = TTSJobQueue()
    running = queue.enqueue("Executando", tmp_path / "run.wav", lambda: None)
    pending = queue.enqueue("Aguardando", tmp_path / "wait.wav", lambda: None)
    queue.start_next()

    assert not queue.remove(running.task_id)
    assert queue.remove(pending.task_id)
    assert queue.entries == (running,)
    assert queue.pending_count == 0


def test_finished_items_can_be_removed_from_history(tmp_path: Path):
    queue = TTSJobQueue()
    entry = queue.enqueue("Finalizada", tmp_path / "done.wav", lambda: None)
    queue.start_next()
    queue.finish_current(entry.task_id, "Falhou")

    assert queue.remove(entry.task_id)
    assert queue.entries == ()


def test_clear_pending_keeps_current_task(tmp_path: Path):
    queue = TTSJobQueue()
    current = queue.enqueue("Atual", tmp_path / "current.wav", lambda: None)
    queue.enqueue("Depois 1", tmp_path / "later1.wav", lambda: None)
    queue.enqueue("Depois 2", tmp_path / "later2.wav", lambda: None)
    queue.start_next()

    assert queue.clear_pending() == 2
    assert queue.current is current
    assert queue.entries == (current,)


def test_history_is_trimmed_without_dropping_active_tasks(tmp_path: Path):
    queue = TTSJobQueue(max_history=2)
    first = queue.enqueue("Um", tmp_path / "1.wav", lambda: None)
    queue.start_next()
    queue.finish_current(first.task_id, "Concluído")
    second = queue.enqueue("Dois", tmp_path / "2.wav", lambda: None)
    queue.start_next()
    queue.finish_current(second.task_id, "Cancelado")
    queue.enqueue("Três", tmp_path / "3.wav", lambda: None)

    assert [item.label for item in queue.entries] == ["Dois", "Três"]
