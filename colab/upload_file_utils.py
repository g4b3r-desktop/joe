"""Helpers for preserving uploaded media formats in the Colab worker."""

from pathlib import Path


def safe_upload_suffix(
    filename: str | None, *, allow_safetensors: bool = False
) -> str:
    """Return a safe suffix, optionally preserving Pocket TTS voice states.

    Safetensors is allowed only for the TTS voice upload. The transcription
    endpoint continues to accept audio formats only.
    """
    suffix = Path(filename or "audio.wav").suffix.lower()
    if allow_safetensors and suffix == ".safetensors":
        return suffix
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        return ".wav"
    return suffix
