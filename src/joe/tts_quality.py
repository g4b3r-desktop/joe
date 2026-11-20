from __future__ import annotations

from inspect import Parameter, signature
from typing import Any

# Pocket TTS defaults to 50 tokens per chunk, or 25 when continuity is enabled.
# Shorter chunks plus the previous chunk's audio context favor completeness.
DEFAULT_MAX_TOKENS = 25
DEFAULT_TEACHER_FORCE_PREVIOUS_CHUNK = True


def generate_audio_with_quality(
    model: Any,
    model_state: Any,
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    teacher_force_previous_chunk: bool = DEFAULT_TEACHER_FORCE_PREVIOUS_CHUNK,
) -> Any:
    """Generate speech with short token chunks and previous-chunk continuity.

    Options are passed only when exposed by the installed Pocket TTS API,
    preserving compatibility with older releases that accept only state and text.
    """
    method = model.generate_audio
    try:
        parameters = signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    options: dict[str, Any] = {}
    if "max_tokens" in parameters or accepts_kwargs:
        options["max_tokens"] = max_tokens
    if "teacher_force_previous_chunk" in parameters or accepts_kwargs:
        options["teacher_force_previous_chunk"] = teacher_force_previous_chunk

    return method(model_state, text, **options)
