from __future__ import annotations

import threading

from joe.local_ipc import LocalCommandServer, send_local_command


def test_send_local_command_returns_false_when_server_is_missing(tmp_path):
    assert not send_local_command("clipboard-audio", tmp_path / "missing.sock")


def test_local_command_server_receives_command(tmp_path):
    received: list[str] = []
    event = threading.Event()
    path = tmp_path / "joe.sock"

    def handle(command: str) -> None:
        received.append(command)
        event.set()

    server = LocalCommandServer(handle, path=path)
    server.start()
    try:
        assert send_local_command("clipboard-audio", path=path)
        assert event.wait(timeout=2)
        assert received == ["clipboard-audio"]
        assert path.exists()
    finally:
        server.stop()

    assert not path.exists()
