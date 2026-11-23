"""Helpers for preserving uploaded media formats in the Colab worker."""

from pathlib import Path


def safe_upload_suffix(filename: str | None) -> str:
    """Return a safe suffix, preserving Pocket TTS voice-state files.

    ``.safetensors`` is intentionally accepted even though it is longer than
    the short audio suffixes. The Colab API writes uploads to a temporary file
    using this suffix, and Pocket TTS selects its Safetensors loader by suffix.
    """
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix == ".safetensors":
        return suffix
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        return ".wav"
    return suffix
