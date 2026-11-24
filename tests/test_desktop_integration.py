from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash não está disponível")
def test_desktop_shell_scripts_have_valid_syntax():
    for relative_path in ("run.sh", "scripts/install-desktop.sh"):
        subprocess.run(
            ["bash", "-n", str(ROOT / relative_path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_desktop_installer_embedded_python_is_valid():
    installer = (ROOT / "scripts/install-desktop.sh").read_text(encoding="utf-8")
    blocks = re.findall(r"python3 - <<'PY'\s*\n(.*?)\nPY", installer, re.DOTALL)

    assert len(blocks) == 2
    for index, source in enumerate(blocks, start=1):
        compile(source, f"install-desktop.sh:python-block-{index}", "exec")
