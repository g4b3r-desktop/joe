from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from joe.pocket_tts_models import (
    MODEL_CONFIG_BY_VARIANT,
    MODEL_VARIANTS_BY_LANGUAGE,
    default_model_variant,
)


@dataclass(frozen=True)
class LocalModel:
    """A locally discovered Pocket TTS checkpoint that can be offered for Drive upload."""

    language: str
    model_variant: str
    display_name: str
    checkpoint: Path
    upload_directory: Path
    size_bytes: int
    tokenizer_local: bool
    tokenizer_auto_download: bool

    @property
    def can_upload(self) -> bool:
        return self.tokenizer_local or self.tokenizer_auto_download

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def default_model_search_roots(
    *, home: Path | None = None, project_root: Path | None = None
) -> list[Path]:
    """Return focused cache locations; never scan the entire home directory."""
    home = (home or Path.home()).expanduser()
    project_root = project_root or Path(__file__).resolve().parents[2]
    candidates: list[Path] = []

    for variable in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(variable, "").strip()
        if value:
            candidates.append(Path(value).expanduser())

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        hf_path = Path(hf_home).expanduser()
        candidates.extend((hf_path / "hub", hf_path))

    candidates.extend(
        (
            home / ".cache" / "huggingface" / "hub",
            project_root / ".models" / "huggingface" / "hub",
        )
    )

    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _variant_details() -> dict[str, tuple[str, str]]:
    details: dict[str, tuple[str, str]] = {}
    for language, variants in MODEL_VARIANTS_BY_LANGUAGE.items():
        for variant, label, _description in variants:
            details[variant] = (language, label)
    return details


def _model_identity(
    checkpoint: Path, variants: dict[str, tuple[str, str]]
) -> tuple[str, str, str] | None:
    # O diretório imediatamente acima do checkpoint identifica a variante.
    # Nos snapshots oficiais ele é languages/<idioma>.
    folder_name = checkpoint.parent.name
    if folder_name in variants:
        language, label = variants[folder_name]
        if language != "japanese" and folder_name not in MODEL_CONFIG_BY_VARIANT:
            return language, folder_name, label
        return None

    if folder_name in MODEL_VARIANTS_BY_LANGUAGE:
        if folder_name == "japanese":
            return None
        variant = default_model_variant(folder_name)
        label = next(
            (
                item[1]
                for item in MODEL_VARIANTS_BY_LANGUAGE[folder_name]
                if item[0] == variant
            ),
            variant,
        )
        return folder_name, variant, label
    return None

def discover_local_models(search_roots: Iterable[str | Path]) -> list[LocalModel]:
    """Find known Pocket TTS checkpoints under selected folders without downloading files."""
    variants = _variant_details()
    candidates: dict[str, LocalModel] = {}
    seen_checkpoints: set[str] = set()

    for raw_root in search_roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        try:
            checkpoint_paths = root.rglob("model.safetensors")
            for checkpoint in checkpoint_paths:
                if not checkpoint.is_file():
                    continue
                try:
                    checkpoint_key = str(checkpoint.resolve())
                except OSError:
                    checkpoint_key = str(checkpoint.absolute())
                if checkpoint_key in seen_checkpoints:
                    continue
                seen_checkpoints.add(checkpoint_key)

                identity = _model_identity(checkpoint, variants)
                if identity is None:
                    continue
                language, model_variant, label = identity

                model_directory = (
                    checkpoint.parent
                    if checkpoint.parent.parent.name == "languages"
                    else (
                        checkpoint.parent.parent
                        if checkpoint.parent.name == model_variant
                        else checkpoint.parent
                    )
                )
                tokenizer_path = checkpoint.parent / "tokenizer.json"
                if not tokenizer_path.is_file():
                    tokenizer_path = model_directory / "tokenizer.json"
                tokenizer_local = tokenizer_path.is_file()

                is_official_hf_snapshot = any(
                    ancestor.name == "models--kyutai--pocket-tts"
                    for ancestor in checkpoint.parents
                ) and any(
                    ancestor.parent.name == "snapshots" for ancestor in checkpoint.parents
                )

                model = LocalModel(
                    language=language,
                    model_variant=model_variant,
                    display_name=label,
                    checkpoint=checkpoint,
                    upload_directory=model_directory,
                    size_bytes=checkpoint.stat().st_size,
                    tokenizer_local=tokenizer_local,
                    tokenizer_auto_download=(
                        is_official_hf_snapshot and not tokenizer_local
                    ),
                )
                # A cache can contain several immutable snapshots of the same variant.
                # Offer the newest checkpoint once, avoiding sequential overwrites.
                previous = candidates.get(model_variant)
                if (
                    previous is None
                    or checkpoint.stat().st_mtime_ns
                    > previous.checkpoint.stat().st_mtime_ns
                ):
                    candidates[model_variant] = model
        except OSError:
            # An inaccessible subfolder must not prevent scanning other roots.
            continue

    return sorted(
        candidates.values(),
        key=lambda item: (item.language, item.model_variant),
    )
