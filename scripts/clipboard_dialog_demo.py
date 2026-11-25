#!/usr/bin/env python3
"""Abre a janela de atalho com frases e WAVs sintéticos, sem executar TTS."""

from __future__ import annotations

import argparse
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QApplication

from joe.audio_performance import AudioPerformance, wav_duration_seconds
from joe.ui.clipboard_audio_dialog import ClipboardAudioDialog


def write_test_wav(path: Path, frequency: float, duration: float = 0.48) -> None:
    sample_rate = 22050
    frame_count = int(sample_rate * duration)
    amplitude = 6500
    frames = bytearray()
    for frame in range(frame_count):
        sample = int(
            amplitude * math.sin(2.0 * math.pi * frequency * frame / sample_rate)
        )
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(bytes(frames))


def merge_test_wavs(paths: list[Path], destination: Path) -> None:
    """Une os pequenos WAVs de teste para habilitar Salvar e ouvir tudo."""
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    silence = b"\x00\x00" * round(params.framerate * 0.06)
    with wave.open(str(destination), "wb") as merged:
        merged.setparams(params)
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as source:
                merged.writeframes(source.readframes(source.getnframes()))
            if index < len(paths) - 1:
                merged.writeframes(silence)


def build_phrases(count: int) -> list[str]:
    samples = [
        "Esta é uma frase de demonstração para testar a quebra automática de linha.",
        "A janela mantém a frase atual destacada enquanto o áudio é reproduzido.",
        "As próximas frases aparecem como gerando até que o respectivo trecho fique pronto.",
        "Use as setas para navegar e a tecla Enter para ouvir a frase selecionada.",
        "Clique com o botão direito para copiar o texto ou tocar a partir desta frase.",
    ]
    return [
        f"{samples[index % len(samples)]} (frase {index + 1:03d})"
        for index in range(count)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phrases",
        type=int,
        default=35,
        help="quantidade de frases de teste (padrão: 35)",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="abre com apenas duas frases",
    )
    args = parser.parse_args()
    count = 2 if args.short else max(1, args.phrases)

    app = QApplication(sys.argv)
    app.setOrganizationName("g4b3r-desktop")
    app.setApplicationName("JoeAudioStudio")

    temp_dir = tempfile.TemporaryDirectory(prefix="joe-clipboard-dialog-demo-")
    root = Path(temp_dir.name)
    phrases = build_phrases(count)
    original_text = " ".join(phrases)
    chunk_paths = []
    for index in range(count):
        path = root / f"phrase_{index + 1:03d}.wav"
        write_test_wav(path, 260 + index * 7)
        chunk_paths.append(path)

    final_wav = root / "complete.wav"
    merge_test_wavs(chunk_paths, final_wav)
    total_duration = wav_duration_seconds(final_wav)

    player = QMediaPlayer()
    audio_output = QAudioOutput()
    audio_output.setVolume(0.18)
    player.setAudioOutput(audio_output)

    dialog = ClipboardAudioDialog(
        player,
        str(Path.home() / "Áudios"),
        phrases=phrases,
    )
    dialog.temp_dir = root
    dialog.set_text_preview(original_text)
    dialog.status_label.setText("Demonstração: trechos sintéticos, sem geração TTS.")
    dialog.show()

    performance = AudioPerformance(
        elapsed_seconds=max(total_duration * 2.7, 1.0),
        audio_duration_seconds=total_duration,
        character_count=len(original_text),
        remote=False,
    )

    def add_phrase(index: int = 1) -> None:
        if index > count:
            dialog.set_ready(
                {"path": str(final_wav), "performance": performance}
            )
            dialog.status_label.setText(
                "Demonstração pronta. Os sons são tons sintéticos, não fala."
            )
            return
        dialog.add_chunk(index, chunk_paths[index - 1], count, phrases[index - 1])
        QTimer.singleShot(180, lambda: add_phrase(index + 1))

    QTimer.singleShot(250, add_phrase)
    app.aboutToQuit.connect(temp_dir.cleanup)
    result = app.exec()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
