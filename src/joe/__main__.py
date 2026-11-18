from __future__ import annotations

import sys

from joe.local_ipc import send_local_command


def main() -> None:
    """Forward hotkey launches to Joe before importing the heavy GUI/model stack."""
    clipboard_requested = "--clipboard-audio" in sys.argv
    command = "clipboard-audio" if clipboard_requested else "activate"
    if send_local_command(command):
        return

    from joe.app import main as run_app

    run_app()


if __name__ == "__main__":
    main()
