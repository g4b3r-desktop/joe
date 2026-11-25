from __future__ import annotations

from joe.model_discovery import discover_local_models, default_model_search_roots


def test_discovers_official_huggingface_snapshot_and_marks_tokenizer_autodownload(tmp_path):
    root = (
        tmp_path
        / "models--kyutai--pocket-tts"
        / "snapshots"
        / "abc123"
        / "languages"
        / "portuguese"
    )
    root.mkdir(parents=True)
    checkpoint = root / "model.safetensors"
    checkpoint.write_bytes(b"pocket-tts-weights")

    models = discover_local_models([tmp_path])

    assert len(models) == 1
    model = models[0]
    assert model.language == "portuguese"
    assert model.model_variant == "portuguese"
    assert model.checkpoint == checkpoint
    assert model.upload_directory == root
    assert model.tokenizer_auto_download is True
    assert model.can_upload is True


def test_discovers_24_layer_variant_from_its_folder_name(tmp_path):
    model_dir = tmp_path / "cache" / "portuguese_24l"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    (model_dir / "tokenizer.json").write_text('{"version": 1}', encoding="utf-8")

    models = discover_local_models([tmp_path / "cache"])

    assert len(models) == 1
    assert models[0].language == "portuguese"
    assert models[0].model_variant == "portuguese_24l"
    assert models[0].upload_directory == model_dir.parent
    assert models[0].tokenizer_local is True
    assert models[0].can_upload is True


def test_ignores_unknown_and_japanese_model_folders(tmp_path):
    unknown = tmp_path / "my-custom-model"
    japanese = tmp_path / "japanese"
    unknown.mkdir()
    japanese.mkdir()
    (unknown / "model.safetensors").write_bytes(b"unknown")
    (japanese / "model.safetensors").write_bytes(b"community-model")

    assert discover_local_models([tmp_path]) == []


def test_non_huggingface_model_without_tokenizer_is_not_uploadable(tmp_path):
    model_dir = tmp_path / "portuguese"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")

    models = discover_local_models([tmp_path])

    assert len(models) == 1
    assert models[0].can_upload is False
    assert models[0].tokenizer_auto_download is False


def test_default_search_roots_include_huggingface_and_project_cache(tmp_path):
    roots = default_model_search_roots(home=tmp_path / "home", project_root=tmp_path / "project")

    assert tmp_path / "home" / ".cache" / "huggingface" / "hub" in roots
    assert tmp_path / "project" / ".models" / "huggingface" / "hub" in roots
