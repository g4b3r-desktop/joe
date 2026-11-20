from __future__ import annotations

import re
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def _split_long_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > max_chars:
            pieces.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += extra
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_text_into_chunks(text: str, max_chars: int = 280) -> list[str]:
    """Split speech text near sentence/paragraph boundaries without cutting words."""
    if max_chars < 40:
        raise ValueError("O tamanho dos blocos precisa ser de pelo menos 40 caracteres.")
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    sentences = [
        part.strip()
        for part in _SENTENCE_SPLIT.split(cleaned)
        if part.strip()
    ]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(_split_long_text(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def merge_wav_files(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    pause_ms: int = 60,
) -> Path:
    """Join WAV chunks into one PCM16 file, normalizing sample rate and channels."""
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("Nenhum trecho de áudio foi fornecido.")
    if pause_ms < 0:
        raise ValueError("A pausa entre trechos não pode ser negativa.")

    first_info = sf.info(paths[0])
    sample_rate = int(first_info.samplerate)
    channels = int(first_info.channels)
    if channels not in (1, 2):
        raise ValueError("O áudio precisa ter um ou dois canais.")

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pause = np.zeros((round(sample_rate * pause_ms / 1000), channels), dtype=np.float32)

    with sf.SoundFile(
        destination,
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        format="WAV",
        subtype="PCM_16",
    ) as merged:
        for index, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(f"Trecho de áudio não encontrado: {path}")
            audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
            if audio.size == 0:
                raise ValueError(f"O trecho de áudio está vazio: {path}")

            if source_rate != sample_rate:
                divisor = gcd(int(source_rate), sample_rate)
                audio = resample_poly(
                    audio,
                    sample_rate // divisor,
                    int(source_rate) // divisor,
                    axis=0,
                ).astype(np.float32, copy=False)

            if audio.shape[1] == 1 and channels == 2:
                audio = np.repeat(audio, 2, axis=1)
            elif audio.shape[1] == 2 and channels == 1:
                audio = audio.mean(axis=1, keepdims=True)
            elif audio.shape[1] != channels:
                raise ValueError(f"Número de canais incompatível no trecho: {path}")

            merged.write(audio)
            if pause.shape[0] and index < len(paths) - 1:
                merged.write(pause)

    return destination
