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


def test_colab_notebook_delegates_runtime_settings_to_desktop_joe():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    startup = "".join(notebook["cells"][12]["source"])
    diagnostics = "".join(notebook["cells"][14]["source"])

    assert "JOE_TTS_CPU_THREADS', '2')" in startup
    assert "JOE_TTS_SAMPLER_DECODE_STEPS', '4')" in startup
    assert "JOE_TTS_QUANTIZE_INT8', '0')" in startup
    assert "os.environ['JOE_TTS_CPU_THREADS'] = '2'" not in startup
    assert "tts_quantize_int8" in startup
    assert "tts_quantize_int8_requested" in diagnostics
    assert "INT8 usado:" in diagnostics


def test_colab_notebook_bootstraps_api_from_matching_optimization_branch():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    bootstrap = "".join(notebook["cells"][1]["source"])

    assert "BRANCH = 'feature/colab-performance-optimizations'" in bootstrap
    assert "feature/clipboard-dialog-appearance-settings" not in bootstrap


def test_desktop_opens_colab_notebook_from_matching_optimization_branch():
    app_source = NOTEBOOK_PATH.parents[1] / "src" / "joe" / "app.py"
    source = app_source.read_text(encoding="utf-8")

    assert "blob/feature/colab-performance-optimizations/colab/Joe_Colab_API.ipynb" in source
    assert "blob/feature/clipboard-dialog-appearance-settings/colab/Joe_Colab_API.ipynb" not in source
