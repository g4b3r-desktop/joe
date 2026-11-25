from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def validate_safetensors_file(path: str | Path) -> dict[str, int]:
    """Validate a Safetensors header and its tensor offsets without loading tensors."""
    source = Path(path)
    file_size = source.stat().st_size

    with source.open("rb") as stream:
        prefix = stream.read(8)
        if prefix.startswith(b"RIFF"):
            raise ValueError(
                "o conteúdo começa com RIFF/WAVE: parece ser um áudio WAV "
                "renomeado como .safetensors. Selecione o WAV original ou "
                "reexporte o estado de voz com Pocket TTS."
            )
        if prefix.startswith(b"PK\x03\x04"):
            raise ValueError(
                "o conteúdo começa com um cabeçalho ZIP: não é um estado "
                "Safetensors válido. Não renomeie um .pt/.pth/.zip para "
                ".safetensors."
            )
        if prefix.startswith(b"version "):
            raise ValueError(
                "o arquivo contém apenas um ponteiro Git LFS, não os pesos "
                "Safetensors. Baixe o arquivo real antes de enviá-lo."
            )
        if len(prefix) != 8:
            raise ValueError("arquivo muito pequeno ou truncado: faltam os 8 bytes iniciais.")

        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size < 2:
            raise ValueError(f"tamanho do cabeçalho inválido: {header_size} bytes.")
        if header_size > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"o cabeçalho declara {header_size:,} bytes, acima do limite "
                f"de {MAX_SAFETENSORS_HEADER_BYTES:,} bytes. O arquivo provavelmente "
                "não é Safetensors ou está corrompido."
            )
        if 8 + header_size > file_size:
            raise ValueError(
                f"cabeçalho truncado: declara {header_size:,} bytes, mas o arquivo "
                f"tem somente {file_size:,} bytes."
            )

        raw_header = stream.read(header_size)
        try:
            header: Any = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("o cabeçalho não contém JSON UTF-8 válido.") from exc

    if not isinstance(header, dict):
        raise ValueError("o cabeçalho JSON não é um objeto.")
    tensor_entries = {
        name: item for name, item in header.items() if name != "__metadata__"
    }
    if not tensor_entries:
        raise ValueError("o cabeçalho não contém nenhum tensor.")

    payload_size = file_size - 8 - header_size
    for name, item in tensor_entries.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("metadados de tensor inválidos no cabeçalho.")
        offsets = item.get("data_offsets")
        shape = item.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise ValueError(
                f"intervalo de dados inválido para o tensor {name!r}."
            )
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise ValueError(f"dimensões inválidas para o tensor {name!r}.")

    return {
        "file_size": file_size,
        "header_size": header_size,
        "tensor_count": len(tensor_entries),
    }
