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


def configure_model_cache(
    project_root: str | Path | None = None,
    *,
    cache_directory: str | Path | None = None,
) -> Path:
    """Configure Hugging Face/Pocket TTS caches.

    By default, use .models/huggingface inside the Joe checkout. A caller that
    needs a stable external location may pass cache_directory explicitly.
    """
    if project_root is not None and cache_directory is not None:
        raise ValueError("Informe project_root ou cache_directory, não ambos.")
    if cache_directory is not None:
        cache_root = Path(cache_directory).expanduser().resolve()
    else:
        root = (
            Path(project_root).expanduser().resolve()
            if project_root
            else get_project_root()
        )
        cache_root = root / ".models" / "huggingface"
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    # Set these before importing huggingface_hub / Pocket TTS. Explicitly use
    # the project directory so an existing global HF_HOME cannot redirect downloads.
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    return cache_root
