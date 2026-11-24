from __future__ import annotations

import sys
from pathlib import Path

from joe.local_ipc import send_local_command


def main() -> None:
    """Forward desktop file-open and hotkey launches before importing the GUI stack."""
    arguments = sys.argv[1:]
    if "--clipboard-audio" in arguments:
        command = "clipboard-audio"
    else:
        project_path = next(
            (
                Path(argument).expanduser().resolve()
                for argument in arguments
                if Path(argument).suffix.lower() == ".joeaudio"
            ),
            None,
        )
        command = f"open-project:{project_path}" if project_path is not None else "activate"

    if send_local_command(command):
        return

    from joe.app import main as run_app

    run_app()


if __name__ == "__main__":
    main()
