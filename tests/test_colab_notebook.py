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



def test_colab_job_queue_rolls_back_when_executor_cannot_start_worker():
    import logging
    import threading
    import time

    import pytest

    api_path = NOTEBOOK_PATH.parent / "api_server.py"
    api_source = api_path.read_text(encoding="utf-8")
    tree = ast.parse(api_source)
    submit_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_submit_job"
    )
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            submit_function,
        ],
        type_ignores=[],
    )

    class FailingExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("worker thread unavailable")

    jobs = {}
    namespace = {
        "threading": threading,
        "time": time,
        "logger": logging.getLogger("test.joe.colab"),
        "_JOB_LOCK": threading.Lock(),
        "_JOBS": jobs,
        "_JOB_TTL_SECONDS": 1800,
        "_MAX_STORED_JOBS": 8,
        "_MAX_ACTIVE_JOBS": 2,
        "_JOB_EXECUTOR": FailingExecutor(),
        "_run_job": lambda *_args: None,
    }
    exec(compile(isolated, str(api_path), "exec"), namespace)

    with pytest.raises(RuntimeError, match="Não foi possível iniciar o worker"):
        namespace["_submit_job"](
            "tts_batch",
            lambda *_args: None,
            ["texto"],
            _pass_cancel_event=True,
        )

    assert jobs == {}


def test_colab_batch_route_exposes_unexpected_enqueue_errors():
    api_source = (NOTEBOOK_PATH.parent / "api_server.py").read_text(encoding="utf-8")
    route_start = api_source.index('@app.post("/v1/jobs/tts/batch"')
    route_end = api_source.index('@app.post("/v1/jobs/tts"', route_start)
    route = api_source[route_start:route_end]

    assert "except Exception as exc:" in route
    assert "logger.exception(" in route
    assert "status_code=503" in route
    assert "type(exc).__name__" in route
