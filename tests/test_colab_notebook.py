import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "colab" / "Joe_Colab_API.ipynb"


def test_colab_notebook_is_valid_json():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_colab_python_cells_compile():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Colab/IPython magics such as %pip are commands, not Python syntax.
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")
