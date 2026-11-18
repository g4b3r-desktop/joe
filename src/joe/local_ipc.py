"""Single-instance local IPC used by the desktop hotkey helper."""
from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Callable


def command_socket_path() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime_dir) if runtime_dir and Path(runtime_dir).is_dir() else (
        Path.home() / ".cache" / "joe-audio-studio"
    )
    uid = getattr(os, "getuid", lambda: 0)()
    return base / f"joe-audio-studio-{uid}.sock"


def send_local_command(command: str, path: Path | None = None, timeout: float = 0.4) -> bool:
    if not hasattr(socket, "AF_UNIX"):
        return False
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path or command_socket_path()))
        client.sendall(command.encode("utf-8"))
        return True
    except (OSError, UnicodeError):
        return False
    finally:
        client.close()


class LocalCommandServer:
    def __init__(self, handler: Callable[[str], None], path: Path | None = None) -> None:
        self.path = path or command_socket_path()
        self.handler = handler
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("Este sistema não oferece sockets Unix para o atalho global.")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.exists():
            if send_local_command("activate", self.path, timeout=0.25):
                raise FileExistsError("Já existe uma instância do Joe em execução.")
            if self.path.is_socket():
                self.path.unlink()
            else:
                raise RuntimeError(f"O caminho IPC do Joe existe e não é um socket: {self.path}")

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.settimeout(0.4)
        try:
            server.bind(str(self.path))
            self.path.chmod(0o600)
            server.listen(8)
        except Exception:
            server.close()
            self.path.unlink(missing_ok=True)
            raise
        self._server = server
        self._thread = threading.Thread(target=self._serve, name="joe-local-command-server", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        server = self._server
        if server is None:
            return
        while not self._stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    command = connection.recv(256).decode("utf-8").strip()
                    if command:
                        self.handler(command)
                except (OSError, UnicodeError):
                    continue

    def stop(self) -> None:
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            server.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            if self.path.is_socket():
                self.path.unlink()
        except OSError:
            pass
