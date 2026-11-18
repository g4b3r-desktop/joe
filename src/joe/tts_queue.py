from __future__ import annotations

from typing import Any

PENDING_STATUS = "Pendente"
RUNNING_STATUS = "Gerando"
FAILED_STATUS = "Falhou"


def pending_queue_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return snapshots of pending/failed tasks for a new sequential run."""
    return [
        dict(item)
        for item in items
        if item.get("status", PENDING_STATUS) in {PENDING_STATUS, FAILED_STATUS}
    ]


def update_queue_item(
    items: list[dict[str, Any]],
    item_id: str,
    status: str,
    error: str = "",
) -> list[dict[str, Any]]:
    """Return a queue with one task status updated."""
    updated: list[dict[str, Any]] = []
    for item in items:
        current = dict(item)
        if current.get("id") == item_id:
            current["status"] = status
            if error:
                current["error"] = error
            else:
                current.pop("error", None)
        updated.append(current)
    return updated


def remove_queue_item(items: list[dict[str, Any]], item_id: str) -> list[dict[str, Any]]:
    """Return a queue without the requested task."""
    return [item for item in items if item.get("id") != item_id]


def queue_status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """Count pending, running and failed queue entries."""
    return {
        "pending": sum(item.get("status", PENDING_STATUS) == PENDING_STATUS for item in items),
        "running": sum(item.get("status") == RUNNING_STATUS for item in items),
        "failed": sum(item.get("status") == FAILED_STATUS for item in items),
        "total": len(items),
    }
