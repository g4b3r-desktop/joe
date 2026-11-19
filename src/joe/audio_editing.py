from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 with a stable (frames, channels) shape."""
    samples, sample_rate = sf.read(
        Path(path).expanduser(), dtype="float32", always_2d=True
    )
    if sample_rate <= 0 or samples.size == 0:
        raise ValueError("O arquivo não contém áudio válido.")
    return samples, int(sample_rate)


def waveform_peaks(samples: np.ndarray, point_count: int = 1400) -> list[float]:
    """Return a compact absolute-peak envelope suitable for drawing a waveform."""
    audio = np.asarray(samples)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2 or audio.shape[0] == 0:
        return []
    if point_count < 1:
        raise ValueError("point_count deve ser maior que zero.")
    bins = min(point_count, audio.shape[0])
    boundaries = np.linspace(0, audio.shape[0], bins + 1, dtype=np.int64)
    return [
        min(1.0, float(np.max(np.abs(audio[boundaries[i]:boundaries[i + 1]]))))
        for i in range(bins)
    ]


def find_silence_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -40.0,
    minimum_silence_seconds: float = 0.5,
    frame_seconds: float = 0.01,
) -> list[tuple[int, int]]:
    """Find quiet intervals and return their [start, end) sample boundaries."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError("O áudio está vazio.")
    if sample_rate <= 0:
        raise ValueError("A taxa de amostragem deve ser positiva.")
    if not -120.0 <= threshold_db <= 0.0:
        raise ValueError("O limite de silêncio deve ficar entre -120 e 0 dB.")
    if minimum_silence_seconds <= 0.0 or frame_seconds <= 0.0:
        raise ValueError("As durações devem ser maiores que zero.")

    original_length = audio.shape[0]
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = int(np.ceil(original_length / frame_size))
    padded_length = frame_count * frame_size
    if padded_length != original_length:
        audio = np.pad(audio, ((0, padded_length - original_length), (0, 0)))
    frames = audio.reshape(frame_count, frame_size, audio.shape[1])
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=(1, 2)))
    quiet = rms <= (10.0 ** (threshold_db / 20.0))
    transitions = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    minimum_samples = int(round(minimum_silence_seconds * sample_rate))

    regions: list[tuple[int, int]] = []
    for first, last in zip(starts, ends, strict=True):
        start = int(first * frame_size)
        end = min(int(last * frame_size), original_length)
        if end - start >= minimum_samples:
            regions.append((start, end))
    return regions


def remove_long_silences(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -40.0,
    minimum_silence_seconds: float = 0.5,
    keep_silence_seconds: float = 0.15,
) -> tuple[np.ndarray, float, int]:
    """Shorten long quiet intervals while retaining a small natural pause."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if keep_silence_seconds < 0.0:
        raise ValueError("O tempo de silêncio a preservar não pode ser negativo.")
    regions = find_silence_regions(
        audio, sample_rate, threshold_db=threshold_db,
        minimum_silence_seconds=minimum_silence_seconds,
    )
    if not regions:
        return audio.copy(), 0.0, 0
    if float(np.max(np.abs(audio))) <= 10.0 ** (threshold_db / 20.0):
        raise ValueError("O arquivo parece conter apenas silêncio; nada foi alterado.")

    keep_samples = int(round(keep_silence_seconds * sample_rate))
    parts: list[np.ndarray] = []
    cursor = 0
    removed_samples = 0
    total_samples = audio.shape[0]
    for start, end in regions:
        if start > cursor:
            parts.append(audio[cursor:start])
        silence_length = end - start
        keep = min(keep_samples, silence_length)
        if start == 0:
            if keep:
                parts.append(audio[end - keep:end])
        elif end >= total_samples:
            if keep:
                parts.append(audio[start:start + keep])
        else:
            left_keep = keep // 2
            right_keep = keep - left_keep
            if left_keep:
                parts.append(audio[start:start + left_keep])
            if right_keep:
                parts.append(audio[end - right_keep:end])
        removed_samples += silence_length - keep
        cursor = end
    if cursor < total_samples:
        parts.append(audio[cursor:])
    if not parts:
        raise ValueError("Não foi possível preservar conteúdo de áudio após a análise.")
    result = np.concatenate(parts, axis=0)
    return result, removed_samples / float(sample_rate), len(regions)


def _ensure_distinct_paths(source: str | Path, destination: str | Path) -> None:
    if Path(source).expanduser().resolve() == Path(destination).expanduser().resolve():
        raise ValueError("O arquivo original não pode ser sobrescrito.")


def _write_wav(destination: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    output = Path(destination).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output


def export_trimmed_audio(
    source: str | Path,
    destination: str | Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    """Export a selected time range to a new WAV, leaving the source untouched."""
    _ensure_distinct_paths(source, destination)
    if start_seconds < 0.0 or end_seconds <= start_seconds:
        raise ValueError("O intervalo de corte é inválido.")
    samples, sample_rate = load_audio(source)
    start = min(int(round(start_seconds * sample_rate)), samples.shape[0])
    end = min(int(round(end_seconds * sample_rate)), samples.shape[0])
    if end <= start:
        raise ValueError("O intervalo selecionado não contém áudio.")
    return _write_wav(destination, samples[start:end], sample_rate)


def export_without_silence(
    source: str | Path,
    destination: str | Path,
    *,
    threshold_db: float = -40.0,
    minimum_silence_seconds: float = 0.5,
    keep_silence_seconds: float = 0.15,
) -> tuple[Path, float, int, float, float]:
    """Export a silence-reduced WAV and return removed time and duration statistics."""
    _ensure_distinct_paths(source, destination)
    samples, sample_rate = load_audio(source)
    original_duration = samples.shape[0] / float(sample_rate)
    processed, removed_seconds, count = remove_long_silences(
        samples, sample_rate, threshold_db=threshold_db,
        minimum_silence_seconds=minimum_silence_seconds,
        keep_silence_seconds=keep_silence_seconds,
    )
    output = _write_wav(destination, processed, sample_rate)
    final_duration = processed.shape[0] / float(sample_rate)
    return output, removed_seconds, count, original_duration, final_duration
