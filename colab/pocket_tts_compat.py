from __future__ import annotations

from typing import Any

JAPANESE_MODEL_CONFIG = "hf://takuma104/pocket-tts-ja-cc/japanese.yaml"
JAPANESE_DEFAULT_VOICE = "hf://takuma104/pocket-tts-ja-cc/voices/female_03.wav"

SUPPORTED_TTS_MODEL_VARIANTS = {
    "portuguese": {"portuguese", "portuguese_24l"},
    "english": {
        "english", "english_2026-01", "english_2026-04", "english_2026-09",
        "english_2026-09_2", "english_drifting_26-09",
        "english_18m_2026-10", "english_18m_refl_2026-10",
        "english_7m_2026-10", "english_7m_refl_2026-10",
        "english_2m_2026-10", "english_2m_refl_2026-10",
    },
    "spanish": {"spanish", "spanish_24l"},
    "french": {"french", "french_24l"},
    "italian": {"italian", "italian_24l"},
    "german": {"german", "german_24l"},
    "dutch": {"dutch", "dutch_24l"},
    # Only the compact six-layer Japanese community model is exposed.
    "japanese": {"japanese"},
}


def load_tts_model(
    model_class: Any,
    language: str,
    model_variant: str,
    *,
    quantize: bool,
    sampler_decode_steps: int | None = None,
    temp: float | None = None,
) -> Any:
    """Load a built-in Pocket TTS variant or the Japanese community YAML model."""
    generation_options: dict[str, Any] = {}
    if sampler_decode_steps is not None:
        generation_options["sampler_decode_steps"] = sampler_decode_steps
    if temp is not None:
        generation_options["temp"] = temp

    if language == "japanese" and model_variant == "japanese":
        return model_class.load_model(
            config=JAPANESE_MODEL_CONFIG,
            quantize=quantize,
            **generation_options,
        )
    return model_class.load_model(
        language=model_variant,
        quantize=quantize,
        **generation_options,
    )


def default_voice_for_language(language: str, configured_voice: str | None) -> str | None:
    """Prefer a Japanese preset for Japanese; otherwise keep the configured default."""
    if language == "japanese":
        return JAPANESE_DEFAULT_VOICE
    return configured_voice
