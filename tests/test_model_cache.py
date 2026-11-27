from __future__ import annotations

import os
from pathlib import Path

import pytest

from joe.model_cache import configure_model_cache


def test_configure_model_cache_uses_explicit_directory(tmp_path, monkeypatch):
    cache_root = tmp_path / ".models" / "huggingface" 
    monkeypatch.setenv("HF_HOME", "/old/hf")
    monkeypatch.setenv("HF_HUB_CACHE", "/old/hub")

    configured = configure_model_cache(cache_directory=cache_root)

    assert configured == cache_root.resolve()
    assert (configured / "hub").is_dir()
    assert os.environ["HF_HOME"] == str(configured)
    assert os.environ["HF_HUB_CACHE"] == str(configured / "hub")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(configured / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == str(configured / "hub")


def test_configure_model_cache_rejects_two_locations(tmp_path):
    with pytest.raises(ValueError, match="project_root ou cache_directory"):
        configure_model_cache(tmp_path, cache_directory=tmp_path / "other")
