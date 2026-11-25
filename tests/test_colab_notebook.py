import ast
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


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



def test_colab_api_url_accepts_domain_without_scheme_and_rejects_invalid_urls():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][8]["source"])
    module = ast.parse(source)
    parser_function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_domain_from_api_url"
    )
    isolated_module = ast.Module(body=[parser_function], type_ignores=[])
    namespace = {"re": re, "urlsplit": urlsplit}
    exec(compile(isolated_module, str(NOTEBOOK_PATH), "exec"), namespace)
    parse_domain = namespace["_domain_from_api_url"]

    assert parse_domain("api.example.com") == "api.example.com"
    assert parse_domain("https://api.example.com/") == "api.example.com"
    assert parse_domain("HTTP://api.example.com") == ""
    assert parse_domain("https://api.example.com:8443") == ""
    assert parse_domain("https://api.example.com/v1") == ""
    assert parse_domain("https://api.example.com?token=secret") == ""
