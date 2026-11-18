from __future__ import annotations

import threading

import pytest

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



def test_server_keeps_running_when_a_command_handler_raises(tmp_path):
    received: list[str] = []
    handled = threading.Event()
    path = tmp_path / "joe-handler-error.sock"

    def handle(command: str) -> None:
        received.append(command)
        if command == "raise":
            raise RuntimeError("simulated handler failure")
        handled.set()

    server = LocalCommandServer(handle, path=path)
    server.start()
    try:
        assert send_local_command("raise", path=path)
        assert send_local_command("continue", path=path)
        assert handled.wait(timeout=2)
        assert received == ["raise", "continue"]
    finally:
        server.stop()


def test_bind_failure_does_not_unlink_an_existing_path(tmp_path, monkeypatch):
    from joe import local_ipc

    path = tmp_path / "claimed.sock"
    path.write_text("owned by another process", encoding="utf-8")
    server = LocalCommandServer(lambda _command: None, path=path)

    class FailingSocket:
        def settimeout(self, _timeout):
            pass

        def bind(self, _address):
            raise OSError("address already in use")

        def close(self):
            pass

    original_exists = local_ipc.Path.exists

    def pretend_path_is_absent(candidate):
        if candidate == path:
            return False
        return original_exists(candidate)

    monkeypatch.setattr(local_ipc.socket, "socket", lambda *_args: FailingSocket())
    monkeypatch.setattr(local_ipc.Path, "exists", pretend_path_is_absent)

    with pytest.raises(OSError, match="address already in use"):
        server.start()

    assert path.read_text(encoding="utf-8") == "owned by another process"
