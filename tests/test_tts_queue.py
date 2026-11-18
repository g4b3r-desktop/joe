from __future__ import annotations

from joe.tts_queue import (
    FAILED_STATUS,
    PENDING_STATUS,
    RUNNING_STATUS,
    pending_queue_items,
    queue_status_counts,
    remove_queue_item,
    update_queue_item,
)


def test_pending_queue_includes_failed_for_retry_but_not_running_or_completed():
    items = [
        {"id": "a", "status": PENDING_STATUS},
        {"id": "b", "status": FAILED_STATUS},
        {"id": "c", "status": RUNNING_STATUS},
        {"id": "d", "status": "Concluído"},
    ]

    selected = pending_queue_items(items)

    assert [item["id"] for item in selected] == ["a", "b"]
    assert selected[0] is not items[0]


def test_queue_status_updates_do_not_mutate_input():
    items = [{"id": "x", "status": PENDING_STATUS}]

    updated = update_queue_item(items, "x", FAILED_STATUS, "falhou")

    assert items[0]["status"] == PENDING_STATUS
    assert updated[0]["status"] == FAILED_STATUS
    assert updated[0]["error"] == "falhou"


def test_queue_removal_and_counts():
    items = [
        {"id": "a", "status": PENDING_STATUS},
        {"id": "b", "status": FAILED_STATUS},
    ]

    remaining = remove_queue_item(items, "a")

    assert [item["id"] for item in remaining] == ["b"]
    assert queue_status_counts(remaining) == {
        "pending": 0, "running": 0, "failed": 1, "total": 1
    }
