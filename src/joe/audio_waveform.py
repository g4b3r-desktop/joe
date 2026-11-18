from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_waveform_peaks(
    audio_path: str | Path,
    *,
    max_points: int = 1200,
    block_frames: int = 65536,
) -> list[float]:
    """Stream an audio file and calculate a normalized min-memory amplitude envelope."""
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {path}")
    if max_points < 8:
        raise ValueError("A forma de onda precisa de pelo menos 8 pontos.")
    if block_frames < 256:
        raise ValueError("O bloco de leitura precisa ter pelo menos 256 frames.")

    info = sf.info(path)
    if info.frames <= 0 or info.channels <= 0:
        return []

    point_count = min(int(max_points), int(info.frames))
    edges = np.linspace(0, info.frames, point_count + 1, dtype=np.int64)
    peaks = np.zeros(point_count, dtype=np.float32)

    frame_position = 0
    bucket = 0
    with sf.SoundFile(path) as audio_file:
        for block in audio_file.blocks(
            blocksize=block_frames,
            dtype="float32",
            always_2d=True,
        ):
            block_end = frame_position + len(block)
            while bucket < point_count and edges[bucket] < block_end:
                overlap_start = max(frame_position, int(edges[bucket]))
                overlap_end = min(block_end, int(edges[bucket + 1]))
                if overlap_end > overlap_start:
                    sample_slice = block[
                        overlap_start - frame_position : overlap_end - frame_position
                    ]
                    peaks[bucket] = max(
                        peaks[bucket],
                        float(np.max(np.abs(sample_slice))),
                    )
                if edges[bucket + 1] <= block_end:
                    bucket += 1
                else:
                    break
            frame_position = block_end

    maximum = float(peaks.max(initial=0.0))
    if maximum > 0:
        peaks /= maximum
    return [float(value) for value in peaks]
