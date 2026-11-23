from __future__ import annotations

import json
import os
from collections import Counter
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

MODEL_REPO_ID = "kyutai/pocket-tts"
MAX_ARCHIVE_BYTES = 4 * 1024**3  # 4 GiB

# Archive path -> file path inside the official Hugging Face repository.
MODEL_ASSETS = {
    "pocket-tts/languages/portuguese/model.safetensors": (
        "languages/portuguese/model.safetensors",
        "safetensors",
    ),
    "pocket-tts-without-voice-cloning/languages/portuguese/tokenizer.json": (
        "languages/portuguese/tokenizer.json",
        "json",
    ),
}


class ArchiveSizeLimitError(ValueError):
    """Raised when updating the model archive would exceed its configured limit."""


def _validate_stream(stream: Any, kind: str, label: str) -> None:
    if kind == "safetensors":
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{label}: arquivo Safetensors incompleto.")
        header_size = int.from_bytes(prefix, "little")
        if header_size <= 1 or header_size > 100 * 1024 * 1024:
            raise ValueError(f"{label}: cabeçalho Safetensors inválido.")
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{label}: cabeçalho Safetensors truncado.")
        header = json.loads(header_bytes)
        if not isinstance(header, dict) or not any(
            key != "__metadata__" for key in header
        ):
            raise ValueError(f"{label}: não contém tensores Safetensors.")
        # Ler até o final também valida o CRC do membro ZIP.
        while stream.read(8 * 1024 * 1024):
            pass
        return

    if kind == "json":
        payload = stream.read()
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(f"{label}: tokenizer JSON vazio ou inválido.")
        return

    raise ValueError(f"Tipo de validação desconhecido: {kind}")


def _is_valid_member(
    archive: zipfile.ZipFile, member_name: str, kind: str
) -> bool:
    try:
        with archive.open(member_name, "r") as stream:
            _validate_stream(stream, kind, member_name)
        return True
    except (
        KeyError,
        OSError,
        EOFError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ):
        return False


def _is_valid_file(path: Path, kind: str) -> bool:
    try:
        with path.open("rb") as stream:
            _validate_stream(stream, kind, str(path))
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _archive_prefix(names: list[str]) -> str:
    markers = (
        "pocket-tts/languages/portuguese/model.safetensors",
        "pocket-tts-without-voice-cloning/languages/portuguese/tokenizer.json",
    )
    for marker in markers:
        for name in names:
            if name.endswith(marker):
                return name[: -len(marker)]
    return "JOE_MODELS/"


def _default_download(
    repo_id: str, filename: str, *, token: str | None
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


def ensure_portuguese_model_in_zip(
    zip_path: str | Path,
    *,
    token: str | None = None,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
    download_file: Callable[..., str | Path] | None = None,
) -> dict[str, Any]:
    """Ensure the Portuguese Pocket TTS checkpoint and tokenizer are in a Drive ZIP.

    Existing ZIP members are preserved. Only missing or invalid model assets are
    downloaded, and the original ZIP is replaced only after the new archive has
    passed validation and the configured size limit.
    """
    destination = Path(zip_path).expanduser()
    if max_archive_bytes <= 0:
        raise ValueError("O limite do ZIP deve ser maior que zero.")

    if destination.exists() and not destination.is_file():
        raise ValueError(f"O caminho do ZIP não é um arquivo: {destination}")

    archive_exists = destination.is_file()
    if archive_exists and destination.stat().st_size > max_archive_bytes:
        raise ArchiveSizeLimitError(
            f"O ZIP atual já excede o limite de {max_archive_bytes:,} bytes."
        )

    names: list[str] = []
    if archive_exists:
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"O arquivo existente não é um ZIP válido: {destination}")
        with zipfile.ZipFile(destination, "r") as archive:
            names = archive.namelist()

    prefix = _archive_prefix(names)
    name_counts = Counter(names)
    target_members = {
        prefix + archive_path: (repo_path, kind)
        for archive_path, (repo_path, kind) in MODEL_ASSETS.items()
    }

    valid_members: set[str] = set()
    missing_members: dict[str, tuple[str, str]] = {}
    if archive_exists:
        with zipfile.ZipFile(destination, "r") as archive:
            for member_name, (repo_path, kind) in target_members.items():
                if (
                    name_counts[member_name] == 1
                    and _is_valid_member(archive, member_name, kind)
                ):
                    valid_members.add(member_name)
                else:
                    missing_members[member_name] = (repo_path, kind)
    else:
        missing_members = dict(target_members)

    if not missing_members:
        return {
            "archive_path": str(destination),
            "archive_size_bytes": destination.stat().st_size,
            "downloaded": [],
            "reused": sorted(valid_members),
            "updated": False,
        }

    selected_downloader = download_file or _default_download
    effective_token = token
    if download_file is None and not effective_token:
        try:
            from huggingface_hub import get_token

            effective_token = get_token()
        except Exception:
            effective_token = None
        if not effective_token:
            raise RuntimeError(
                "O modelo kyutai/pocket-tts exige acesso autorizado no Hugging Face. "
                "Aceite as condições em https://huggingface.co/kyutai/pocket-tts "
                "e configure o Secret HF_TOKEN no Google Colab (Notebook access ativo)."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="joe-pocket-tts-") as staging_dir:
        staging = Path(staging_dir)
        for member_name, (repo_filename, kind) in missing_members.items():
            try:
                local_path = Path(
                    selected_downloader(
                        MODEL_REPO_ID, repo_filename, token=effective_token
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Não foi possível baixar os arquivos ausentes do Hugging Face. "
                    "Confirme que o acesso ao modelo foi aceito e que HF_TOKEN tem "
                    f"permissão de leitura. Detalhe: {type(exc).__name__}: {exc}"
                ) from exc

            if not local_path.is_file() or not _is_valid_file(local_path, kind):
                raise ValueError(
                    f"O download retornou um arquivo ausente ou inválido: {repo_filename}"
                )
            staged_path = staging / f"{len(downloaded):02d}_{local_path.name}"
            shutil.copy2(local_path, staged_path)
            downloaded[member_name] = staged_path

        added_bytes = sum(path.stat().st_size for path in downloaded.values())
        current_size = destination.stat().st_size if archive_exists else 0
        estimated_size = current_size + added_bytes + 1024 * (len(downloaded) + 2)
        if estimated_size > max_archive_bytes:
            raise ArchiveSizeLimitError(
                "A atualização foi cancelada para respeitar o limite de 4 GiB. "
                f"Estimativa do ZIP atualizado: {estimated_size:,} bytes; "
                f"limite: {max_archive_bytes:,} bytes. "
                "O ZIP original no Drive foi preservado."
            )

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        temporary_zip = Path(temp_name)
        replaced_members = set(downloaded)

        try:
            with zipfile.ZipFile(
                temporary_zip, "w", allowZip64=True
            ) as output_zip:
                if archive_exists:
                    with zipfile.ZipFile(destination, "r") as input_zip:
                        for info in input_zip.infolist():
                            if info.filename in replaced_members:
                                continue
                            if info.is_dir():
                                output_zip.writestr(info, b"")
                                continue
                            with input_zip.open(info, "r") as source:
                                with output_zip.open(
                                    info, "w", force_zip64=True
                                ) as target:
                                    shutil.copyfileobj(
                                        source, target, length=8 * 1024 * 1024
                                    )

                for member_name, staged_path in downloaded.items():
                    info = zipfile.ZipInfo(member_name)
                    info.compress_type = (
                        zipfile.ZIP_STORED
                        if member_name.endswith(".safetensors")
                        else zipfile.ZIP_DEFLATED
                    )
                    info.external_attr = 0o100644 << 16
                    with staged_path.open("rb") as source:
                        with output_zip.open(
                            info, "w", force_zip64=True
                        ) as target:
                            shutil.copyfileobj(
                                source, target, length=8 * 1024 * 1024
                            )

            final_size = temporary_zip.stat().st_size
            if final_size > max_archive_bytes:
                raise ArchiveSizeLimitError(
                    "O ZIP atualizado ultrapassaria o limite de 4 GiB "
                    f"({final_size:,} bytes). O ZIP original no Drive foi preservado."
                )

            with zipfile.ZipFile(temporary_zip, "r") as check_zip:
                for member_name, (_, kind) in target_members.items():
                    if not _is_valid_member(check_zip, member_name, kind):
                        raise ValueError(
                            f"Falha na validação do arquivo dentro do ZIP: {member_name}"
                        )

            os.replace(temporary_zip, destination)
        finally:
            temporary_zip.unlink(missing_ok=True)

    return {
        "archive_path": str(destination),
        "archive_size_bytes": destination.stat().st_size,
        "downloaded": sorted(downloaded),
        "reused": sorted(valid_members),
        "updated": True,
    }
