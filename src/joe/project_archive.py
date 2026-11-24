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
