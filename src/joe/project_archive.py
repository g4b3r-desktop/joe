from __future__ import annotations

import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


JOE_AUDIO_PROJECT_EXTENSION = ".joeaudio"
JOE_AUDIO_PROJECT_FORMAT = "joe-audio-project"
JOE_AUDIO_PROJECT_VERSION = 1


def save_joe_audio_project(
    destination: str | Path,
    *,
    original_text: str,
    phrase_texts: Sequence[str] = (),
    chunk_texts_by_index: Mapping[int, str] | None = None,
    full_audio_path: str | Path,
    phrase_audio_paths: Mapping[int, str | Path] | None = None,
    recognized_text: str = "",
    recognized_segments: Sequence[Mapping[str, object]] = (),
    performance: Mapping[str, object] | None = None,
) -> Path:
    """Save text and generated WAV files as one compressed Joe project.

    The .joeaudio file is a ZIP container with a versioned JSON manifest,
    UTF-8 text files, the complete WAV, and available per-phrase WAV files.
    The destination is replaced atomically only after the archive is complete.
    """
    target = Path(destination).expanduser()
    full_audio = Path(full_audio_path).expanduser()
    if not full_audio.is_file():
        raise FileNotFoundError(f"Áudio completo não encontrado: {full_audio}")

    target.parent.mkdir(parents=True, exist_ok=True)
    chunk_texts = {
        int(index): str(text)
        for index, text in (chunk_texts_by_index or {}).items()
    }
    if not chunk_texts:
        chunk_texts = {
            index: str(text)
            for index, text in enumerate(phrase_texts, start=1)
        }

    phrase_audio_files: list[dict[str, object]] = []
    phrase_sources: list[tuple[Path, str]] = []
    for index, source_value in sorted((phrase_audio_paths or {}).items()):
        source = Path(source_value).expanduser()
        if not source.is_file():
            continue
        member = f"audio/phrases/{int(index):03d}.wav"
        phrase_sources.append((source, member))
        phrase_audio_files.append({
            "index": int(index),
            "text": chunk_texts.get(int(index), ""),
            "file": member,
        })

    text_files = {
        "original": "texts/original.txt",
        "phrases": "texts/phrases.json",
    }
    if recognized_text:
        text_files["recognized"] = "texts/recognized.txt"
    if recognized_segments:
        text_files["recognized_segments"] = "texts/recognized_segments.json"

    audio_files = {
        "full": "audio/full.wav",
        "phrases": phrase_audio_files,
    }
    manifest: dict[str, object] = {
        "format": JOE_AUDIO_PROJECT_FORMAT,
        "format_version": JOE_AUDIO_PROJECT_VERSION,
        "application": "Joe Audio Studio",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "texts": text_files,
        "audio": audio_files,
    }
    if performance:
        manifest["performance"] = dict(performance)

    phrase_payload = [
        {"index": index, "text": text}
        for index, text in sorted(chunk_texts.items())
    ]
    segments_payload = [dict(segment) for segment in recognized_segments]

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.comment = b"Joe Audio Studio project"
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            archive.writestr("texts/original.txt", str(original_text).encode("utf-8"))
            archive.writestr(
                "texts/phrases.json",
                json.dumps(phrase_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            if recognized_text:
                archive.writestr(
                    "texts/recognized.txt",
                    str(recognized_text).encode("utf-8"),
                )
            if recognized_segments:
                archive.writestr(
                    "texts/recognized_segments.json",
                    json.dumps(
                        segments_payload, ensure_ascii=False, indent=2
                    ).encode("utf-8"),
                )
            archive.write(full_audio, "audio/full.wav")
            for source, member in phrase_sources:
                archive.write(source, member)

        # Catch truncated or invalid ZIPs before replacing an existing project.
        with zipfile.ZipFile(temporary, "r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise zipfile.BadZipFile(
                    f"Falha ao validar o arquivo compactado: {corrupt_member}"
                )
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return target


def load_joe_audio_project(
    source: str | Path,
    extraction_directory: str | Path,
) -> dict[str, object]:
    """Validate and load a .joeaudio project into a caller-owned temp directory.

    Archive members are copied individually; ZIP extraction is never performed
    wholesale, preventing path traversal from untrusted project files.
    """
    project = Path(source).expanduser()
    destination = Path(extraction_directory).expanduser()
    if not project.is_file():
        raise FileNotFoundError(f"Projeto Joe não encontrado: {project}")
    destination.mkdir(parents=True, exist_ok=True)

    def safe_member(name: object) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("O projeto contém um caminho de arquivo inválido.")
        member = Path(name)
        # ZIP names are POSIX paths regardless of the host operating system.
        from pathlib import PurePosixPath
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or "\\" in name:
            raise ValueError("O projeto contém um caminho inseguro.")
        return posix.as_posix()

    with zipfile.ZipFile(project, "r") as archive:
        if archive.testzip() is not None:
            raise zipfile.BadZipFile("O projeto está corrompido.")
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Manifesto do projeto ausente ou inválido.") from exc

        if not isinstance(manifest, dict):
            raise ValueError("Manifesto do projeto inválido.")
        if manifest.get("format") != JOE_AUDIO_PROJECT_FORMAT:
            raise ValueError("Este arquivo não é um projeto do Joe.")
        version = manifest.get("format_version")
        if version != JOE_AUDIO_PROJECT_VERSION:
            raise ValueError(
                f"Versão do projeto não suportada: {version!r}."
            )

        texts = manifest.get("texts")
        audio = manifest.get("audio")
        if not isinstance(texts, dict) or not isinstance(audio, dict):
            raise ValueError("Manifesto sem as seções de texto e áudio.")

        full_audio_member = safe_member(audio.get("full"))
        if full_audio_member not in archive.namelist():
            raise ValueError("O projeto não contém o áudio completo.")
        phrase_audio = audio.get("phrases", [])
        if not isinstance(phrase_audio, list):
            raise ValueError("Lista de áudios de frases inválida.")

        required_members = {
            "manifest.json",
            safe_member(texts.get("original")),
            safe_member(texts.get("phrases")),
            full_audio_member,
        }
        phrase_members: list[tuple[int, str, str]] = []
        for entry in phrase_audio:
            if not isinstance(entry, dict):
                raise ValueError("Entrada de áudio de frase inválida.")
            index = entry.get("index")
            if not isinstance(index, int) or index < 1:
                raise ValueError("Índice de frase inválido.")
            member = safe_member(entry.get("file"))
            if member not in archive.namelist():
                raise ValueError(f"Áudio da frase {index} não encontrado.")
            phrase_members.append((index, member, str(entry.get("text", ""))))
            required_members.add(member)

        optional_text_members = {}
        for key in ("recognized", "recognized_segments"):
            if key in texts:
                member = safe_member(texts[key])
                if member not in archive.namelist():
                    raise ValueError(f"Arquivo de texto {key} não encontrado.")
                optional_text_members[key] = member
                required_members.add(member)

        # Reject duplicate member names and extract only manifest-declared files.
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("O projeto contém nomes de arquivos duplicados.")
        extracted: dict[str, Path] = {}
        for member in required_members:
            safe_name = safe_member(member)
            output = destination.joinpath(*safe_name.split("/"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(member))
            extracted[member] = output

        try:
            original_text = extracted[safe_member(texts["original"])].read_text(
                encoding="utf-8"
            )
            phrase_payload = json.loads(
                extracted[safe_member(texts["phrases"])].read_text(encoding="utf-8")
            )
        except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Os textos do projeto estão inválidos.") from exc

        if not isinstance(phrase_payload, list):
            raise ValueError("Lista de frases inválida.")
        phrase_texts: dict[int, str] = {}
        for entry in phrase_payload:
            if not isinstance(entry, dict):
                raise ValueError("Entrada de texto de frase inválida.")
            index = entry.get("index")
            if not isinstance(index, int) or index < 1:
                raise ValueError("Índice de texto de frase inválido.")
            phrase_texts[index] = str(entry.get("text", ""))

        recognized_text = ""
        if "recognized" in optional_text_members:
            recognized_text = extracted[optional_text_members["recognized"]].read_text(
                encoding="utf-8"
            )
        recognized_segments = []
        if "recognized_segments" in optional_text_members:
            try:
                recognized_segments = json.loads(
                    extracted[optional_text_members["recognized_segments"]].read_text(
                        encoding="utf-8"
                    )
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Segmentos reconhecidos inválidos.") from exc
            if not isinstance(recognized_segments, list):
                raise ValueError("Lista de segmentos reconhecidos inválida.")

    phrase_paths = {
        index: extracted[member]
        for index, member, _ in phrase_members
    }
    return {
        "manifest": manifest,
        "original_text": original_text,
        "phrase_texts": phrase_texts,
        "full_audio_path": extracted[full_audio_member],
        "phrase_audio_paths": phrase_paths,
        "recognized_text": recognized_text,
        "recognized_segments": recognized_segments,
        "performance": manifest.get("performance", {}),
    }
