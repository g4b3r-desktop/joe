from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

VOICE_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}

# Google Drive is mounted through FUSE in Colab. Reading a multi-megabyte
# Safetensors file for every text chunk is expensive, so cache successful and
# failed integrity checks while the file metadata remains unchanged.
_VOICE_HASH_CACHE_TTL_SECONDS = 300.0
_VOICE_HASH_CACHE_MAX_ENTRIES = 512
_VOICE_HASH_CACHE: OrderedDict[tuple[str, int, int, int, str], tuple[float, bool]] = OrderedDict()
_VOICE_HASH_CACHE_LOCK = threading.Lock()


def safe_asset_name(value: str, *, fallback: str = "voice") -> str:
    """Create a portable single path component from a user-supplied asset name."""
    raw = Path(str(value or "")).stem.strip().casefold()
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")
    return (slug or fallback)[:64]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def voice_state_matches_sha256(path: str | Path, expected_hash: str) -> bool:
    """Validate a Drive voice-state file without re-reading it for every chunk.

    The cache key includes file size and nanosecond mtime/ctime plus the expected
    digest. A five-minute TTL also bounds stale results on Drive/FUSE mounts whose
    metadata timestamps may be coarse or delayed.
    """
    file_path = Path(path)
    expected = str(expected_hash or "").strip().lower()
    if not expected:
        return False
    stat = file_path.stat()
    key = (
        str(file_path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ctime_ns", 0)),
        expected,
    )
    now = time.monotonic()
    with _VOICE_HASH_CACHE_LOCK:
        cached = _VOICE_HASH_CACHE.get(key)
        if cached is not None and cached[0] > now:
            _VOICE_HASH_CACHE.move_to_end(key)
            return cached[1]

    valid = hmac.compare_digest(sha256_file(file_path), expected)
    with _VOICE_HASH_CACHE_LOCK:
        _VOICE_HASH_CACHE[key] = (now + _VOICE_HASH_CACHE_TTL_SECONDS, valid)
        _VOICE_HASH_CACHE.move_to_end(key)
        while len(_VOICE_HASH_CACHE) > _VOICE_HASH_CACHE_MAX_ENTRIES:
            _VOICE_HASH_CACHE.popitem(last=False)
    return valid


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_model_manifest(drive_root: str | Path, model_variant: str) -> dict[str, Any]:
    manifest_path = Path(drive_root) / "models" / model_variant / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Modelo {model_variant!r} não instalado no Google Drive. "
            "Envie a pasta do modelo pelo Joe antes de gerar áudio."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_variant") != model_variant:
        raise ValueError("O manifesto do modelo não corresponde à variante solicitada.")
    return manifest


def model_fingerprint(manifest: dict[str, Any]) -> str:
    """Fingerprint the exact checkpoint, tokenizer and runtime version used by a voice."""
    fields = {
        "model_variant": manifest.get("model_variant"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "pocket_tts_version": manifest.get("pocket_tts_version"),
    }
    raw = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_voice_state(
    drive_root: str | Path,
    model_variant: str,
    voice_id: str,
    expected_fingerprint: str,
) -> tuple[Path, dict[str, Any]]:
    if not voice_id or not re.fullmatch(r"[a-z0-9_-]{1,96}", voice_id):
        raise ValueError("Identificador de voz inválido.")
    voice_dir = Path(drive_root) / "voices" / model_variant / voice_id
    metadata_path = voice_dir / "metadata.json"
    state_path = voice_dir / "voice_state.safetensors"
    if not metadata_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Voz {voice_id!r} não encontrada no Google Drive.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_variant") != model_variant:
        raise ValueError("A voz foi preparada para outra variante de modelo.")
    if metadata.get("voice_id") != voice_id:
        raise ValueError("O arquivo de metadados não corresponde ao identificador da voz.")
    if metadata.get("model_fingerprint") != expected_fingerprint:
        raise ValueError(
            "A voz foi preparada com outros pesos/tokenizer. Reenvie o áudio original "
            "para prepará-la novamente com o modelo atual do Google Drive."
        )
    expected_state_hash = str(metadata.get("voice_state_sha256", ""))
    if not expected_state_hash or not voice_state_matches_sha256(state_path, expected_state_hash):
        raise ValueError("O estado de voz no Google Drive está corrompido ou incompleto.")
    return state_path, metadata


def voice_state_for_safetensors_export(
    voice_state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Copy Pocket TTS voice-cache tensors to CPU for Safetensors serialization."""
    import torch

    if not isinstance(voice_state, dict) or not voice_state:
        raise ValueError("O Pocket TTS retornou um estado de voz vazio ou inválido.")

    cpu_state: dict[str, dict[str, Any]] = {}
    for module_name, module_state in voice_state.items():
        if not isinstance(module_state, dict):
            raise TypeError(
                f"O estado do módulo {module_name!r} não é um dicionário de tensores."
            )
        cpu_module: dict[str, Any] = {}
        for tensor_name, tensor in module_state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"O estado {module_name}/{tensor_name} não contém um tensor PyTorch."
                )
            cpu_module[tensor_name] = tensor.detach().to(device="cpu").contiguous()
        cpu_state[module_name] = cpu_module

    return cpu_state


def config_for_uploaded_model(
    model_variant: str, model_dir: str | Path, config_dir: str | Path
) -> dict[str, Any]:
    """Build a runtime YAML from the installed Pocket TTS template and Drive files."""
    import yaml

    model_root = Path(model_dir).resolve()
    template = Path(config_dir) / f"{model_variant}.yaml"
    if not template.is_file():
        raise FileNotFoundError(
            f"O Pocket TTS instalado não contém o YAML da variante {model_variant!r}."
        )
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML-base inválido para {model_variant!r}.")
    flow_lm = data.get("flow_lm")
    lookup = flow_lm.get("lookup_table") if isinstance(flow_lm, dict) else None
    if not isinstance(lookup, dict):
        raise ValueError("O YAML-base não define flow_lm.lookup_table.")
    data["weights_path"] = str((model_root / "model.safetensors").resolve())
    data["weights_path_without_voice_cloning"] = None
    lookup["tokenizer_path"] = str((model_root / "tokenizer.json").resolve())
    return data


def validate_model_files(model_file: str | Path, tokenizer_file: str | Path) -> dict[str, Any]:
    """Validate structural formats before an uploaded bundle can replace the Drive copy."""
    from safetensors_validation import validate_safetensors_file

    checkpoint = Path(model_file)
    tokenizer = Path(tokenizer_file)
    checkpoint_info = validate_safetensors_file(checkpoint)
    # Pocket TTS voice-state exports use "module/tensor" keys, whereas a
    # model checkpoint is a normal nn.Module state_dict. Read only the already
    # validated JSON header; do not load the potentially large tensor payload.
    with checkpoint.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_size).decode("utf-8"))
    tensor_names = [
        name for name in header
        if name != "__metadata__"
    ]
    if not tensor_names or any("/" in name for name in tensor_names):
        raise ValueError(
            "O arquivo Safetensors parece ser um estado de voz preparado, não um "
            "checkpoint completo do modelo Pocket TTS. Envie os pesos model.safetensors."
        )
    parsed = json.loads(tokenizer.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("O tokenizer JSON está vazio ou inválido.")
    return {
        "checkpoint_size": checkpoint.stat().st_size,
        "tokenizer_size": tokenizer.stat().st_size,
        "tensor_count": checkpoint_info["tensor_count"],
    }


def replace_directory_atomically(staged: Path, destination: Path) -> None:
    """Replace a model/voice directory with rollback if the final rename fails."""
    staged = Path(staged)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        if had_previous and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
