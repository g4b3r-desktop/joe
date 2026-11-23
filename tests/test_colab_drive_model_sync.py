from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))

from drive_model_sync import (
    ArchiveSizeLimitError,
    MODEL_ASSETS,
    ensure_portuguese_model_in_zip,
)


MODEL_MEMBER = "JOE_MODELS/pocket-tts/languages/portuguese/model.safetensors"
TOKENIZER_MEMBER = (
    "JOE_MODELS/pocket-tts-without-voice-cloning/"
    "languages/portuguese/tokenizer.json"
)
MODEL_24L_MEMBER = "JOE_MODELS/pocket-tts/languages/portuguese_24l/model.safetensors"
TOKENIZER_24L_MEMBER = (
    "JOE_MODELS/pocket-tts-without-voice-cloning/"
    "languages/portuguese_24l/tokenizer.json"
)


def _make_safetensors(path: Path) -> Path:
    header = json.dumps({
        "weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }
    }).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 4)
    return path


def _make_downloads(tmp_path: Path) -> dict[str, Path]:
    model = _make_safetensors(tmp_path / "model.safetensors")
    model_24l = _make_safetensors(tmp_path / "model_24l.safetensors")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer_24l = tmp_path / "tokenizer_24l.json"
    tokenizer_payload = json.dumps(
        {"version": "1.0", "model": {}, "pre_tokenizer": {}}
    )
    tokenizer.write_text(tokenizer_payload, encoding="utf-8")
    tokenizer_24l.write_text(tokenizer_payload, encoding="utf-8")
    return {
        "languages/portuguese/model.safetensors": model,
        "languages/portuguese/tokenizer.json": tokenizer,
        "languages/portuguese_24l/model.safetensors": model_24l,
        "languages/portuguese_24l/tokenizer.json": tokenizer_24l,
    }


def _downloader_for(files: dict[str, Path], calls: list[str]):
    def download(
        repo_id: str,
        filename: str,
        *,
        revision: str,
        token: str | None,
    ):
        assert repo_id in {
            "kyutai/pocket-tts",
            "kyutai/pocket-tts-without-voice-cloning",
        }
        assert revision
        calls.append(filename)
        return files[filename]

    return download


def test_creates_drive_zip_and_downloads_missing_assets(tmp_path):
    files = _make_downloads(tmp_path)
    calls: list[str] = []
    zip_path = tmp_path / "drive" / "JOE_MODELS.zip"

    result = ensure_portuguese_model_in_zip(
        zip_path,
        download_file=_downloader_for(files, calls),
    )

    assert result["updated"] is True
    assert set(result["downloaded"]) == {
        MODEL_MEMBER,
        TOKENIZER_MEMBER,
        MODEL_24L_MEMBER,
        TOKENIZER_24L_MEMBER,
    }
    assert len(calls) == 4
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            MODEL_MEMBER,
            TOKENIZER_MEMBER,
            MODEL_24L_MEMBER,
            TOKENIZER_24L_MEMBER,
        }


def test_existing_model_is_reused_and_only_missing_tokenizer_is_downloaded(tmp_path):
    files = _make_downloads(tmp_path)
    zip_path = tmp_path / "JOE_MODELS.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(files["languages/portuguese/model.safetensors"], MODEL_MEMBER)
        archive.writestr("user-data/keep.txt", "preservar")

    calls: list[str] = []
    result = ensure_portuguese_model_in_zip(
        zip_path,
        download_file=_downloader_for(files, calls),
    )

    assert calls == [
        "languages/portuguese/tokenizer.json",
        "languages/portuguese_24l/model.safetensors",
        "languages/portuguese_24l/tokenizer.json",
    ]
    assert result["reused"] == [MODEL_MEMBER]
    assert set(result["downloaded"]) == {
        TOKENIZER_MEMBER,
        MODEL_24L_MEMBER,
        TOKENIZER_24L_MEMBER,
    }
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.read("user-data/keep.txt") == b"preservar"
        for member in (
            MODEL_MEMBER,
            TOKENIZER_MEMBER,
            MODEL_24L_MEMBER,
            TOKENIZER_24L_MEMBER,
        ):
            assert archive.namelist().count(member) == 1


def test_complete_zip_does_not_download_or_rewrite(tmp_path):
    files = _make_downloads(tmp_path)
    zip_path = tmp_path / "JOE_MODELS.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(files["languages/portuguese/model.safetensors"], MODEL_MEMBER)
        archive.write(files["languages/portuguese/tokenizer.json"], TOKENIZER_MEMBER)
        archive.write(
            files["languages/portuguese_24l/model.safetensors"],
            MODEL_24L_MEMBER,
        )
        archive.write(
            files["languages/portuguese_24l/tokenizer.json"],
            TOKENIZER_24L_MEMBER,
        )
    original_bytes = zip_path.read_bytes()
    calls: list[str] = []

    result = ensure_portuguese_model_in_zip(
        zip_path,
        download_file=_downloader_for(files, calls),
    )

    assert result["updated"] is False
    assert calls == []
    assert zip_path.read_bytes() == original_bytes


def test_size_limit_keeps_existing_archive_unchanged(tmp_path):
    files = _make_downloads(tmp_path)
    zip_path = tmp_path / "JOE_MODELS.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("user-data/keep.txt", "original")
    original_bytes = zip_path.read_bytes()
    calls: list[str] = []

    with pytest.raises(ArchiveSizeLimitError):
        ensure_portuguese_model_in_zip(
            zip_path,
            max_archive_bytes=300,
            download_file=_downloader_for(files, calls),
        )

    assert zip_path.read_bytes() == original_bytes


def test_invalid_existing_model_is_replaced_instead_of_kept(tmp_path):
    files = _make_downloads(tmp_path)
    zip_path = tmp_path / "JOE_MODELS.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(MODEL_MEMBER, b"not a safetensors file")
        archive.write(files["languages/portuguese/tokenizer.json"], TOKENIZER_MEMBER)
        archive.write(
            files["languages/portuguese_24l/model.safetensors"],
            MODEL_24L_MEMBER,
        )
        archive.write(
            files["languages/portuguese_24l/tokenizer.json"],
            TOKENIZER_24L_MEMBER,
        )

    calls: list[str] = []
    result = ensure_portuguese_model_in_zip(
        zip_path,
        download_file=_downloader_for(files, calls),
    )

    assert calls == ["languages/portuguese/model.safetensors"]
    assert result["downloaded"] == [MODEL_MEMBER]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist().count(MODEL_MEMBER) == 1

def test_root_level_archive_keeps_root_layout(tmp_path):
    files = _make_downloads(tmp_path)
    zip_path = tmp_path / "JOE_MODELS.zip"
    root_model = MODEL_MEMBER.removeprefix("JOE_MODELS/")
    root_tokenizer = TOKENIZER_MEMBER.removeprefix("JOE_MODELS/")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(files["languages/portuguese/model.safetensors"], root_model)
        archive.write(
            files["languages/portuguese_24l/model.safetensors"],
            MODEL_24L_MEMBER.removeprefix("JOE_MODELS/"),
        )
        archive.write(
            files["languages/portuguese_24l/tokenizer.json"],
            TOKENIZER_24L_MEMBER.removeprefix("JOE_MODELS/"),
        )

    calls: list[str] = []
    result = ensure_portuguese_model_in_zip(
        zip_path,
        download_file=_downloader_for(files, calls),
    )

    assert result["downloaded"] == [root_tokenizer]
    with zipfile.ZipFile(zip_path) as archive:
        assert root_model in archive.namelist()
        assert root_tokenizer in archive.namelist()
        assert not any(name.startswith("JOE_MODELS/") for name in archive.namelist())


def test_duplicate_model_entries_are_repaired_without_duplicate_files(tmp_path):
    files = _make_downloads(tmp_path)
    zip_path = tmp_path / "JOE_MODELS.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(MODEL_MEMBER, b"old corrupt model")
        archive.write(files["languages/portuguese/model.safetensors"], MODEL_MEMBER)
        archive.write(files["languages/portuguese/tokenizer.json"], TOKENIZER_MEMBER)
        archive.write(
            files["languages/portuguese_24l/model.safetensors"],
            MODEL_24L_MEMBER,
        )
        archive.write(
            files["languages/portuguese_24l/tokenizer.json"],
            TOKENIZER_24L_MEMBER,
        )

    result = ensure_portuguese_model_in_zip(
        zip_path,
        download_file=_downloader_for(files, []),
    )

    assert result["updated"] is True
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist().count(MODEL_MEMBER) == 1
        assert archive.namelist().count(TOKENIZER_MEMBER) == 1




def test_each_tokenizer_uses_a_revision_from_its_own_repository():
    expected_revision = "00eac05ed3d16bdc3f6b5d598874019c34a89214"
    tokenizer_assets = [
        asset
        for member, asset in MODEL_ASSETS.items()
        if member.endswith("/tokenizer.json")
    ]

    assert len(tokenizer_assets) == 2
    assert all(asset[0] == "kyutai/pocket-tts-without-voice-cloning" for asset in tokenizer_assets)
    assert all(asset[2] == expected_revision for asset in tokenizer_assets)
