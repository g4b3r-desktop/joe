from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Find the source checkout root, falling back to the installed package root."""
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    working_directory = Path.cwd()
    if (working_directory / "pyproject.toml").is_file():
        return working_directory

    # In a source checkout this is the repository root (src/joe/model_cache.py).
    return module_path.parents[2]

def _preserve_hf_token_path() -> None:
    """Preserva a localização original do token antes de alterar HF_HOME."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_PATH"):
        return

    previous_hf_home = os.environ.get("HF_HOME")
    candidates = []

    if previous_hf_home:
        candidates.append(Path(previous_hf_home).expanduser() / "token")

    candidates.append(
        Path.home() / ".cache" / "huggingface" / "token"
    )

    for token_path in candidates:
        if token_path.is_file():
            os.environ["HF_TOKEN_PATH"] = str(token_path)
            return

def configure_model_cache(project_root: str | Path | None = None) -> Path:
    """Point Hugging Face/Pocket TTS caches to .models inside the Joe project."""
    root = Path(project_root).expanduser().resolve() if project_root else get_project_root()
    cache_root = root / ".models" / "huggingface"
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    _preserve_hf_token_path()
    # Set these before importing huggingface_hub / Pocket TTS. Explicitly use
    # the project directory so an existing global HF_HOME cannot redirect downloads.
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    return cache_root
