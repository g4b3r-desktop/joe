from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "colab" / "Joe_Colab_API.ipynb"


def test_colab_notebook_json_and_python_cells_are_valid():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook.get("nbformat") == 4
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        # Remove IPython line magics (for example, %pip); remaining code must parse.
        python_source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(python_source, filename=f"{NOTEBOOK.name}:cell-{index}")
