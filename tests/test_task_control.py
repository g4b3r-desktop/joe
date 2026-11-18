from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from joe.task_control import CancellationToken


def test_cancellation_token_starts_not_requested():
    token = CancellationToken()

    assert token.is_requested is False


def test_cancellation_token_remembers_request():
    token = CancellationToken()
    token.request()

    assert token.is_requested is True


def test_cancellation_token_is_safe_to_request_from_another_thread():
    token = CancellationToken()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(token.request).result(timeout=2)

    assert token.is_requested is True
