from colab.pocket_tts_compat import (
    JAPANESE_DEFAULT_VOICE,
    JAPANESE_MODEL_CONFIG,
    SUPPORTED_TTS_MODEL_VARIANTS,
    default_voice_for_language,
    load_tts_model,
)


def test_colab_catalog_exposes_only_compact_japanese_variant():
    assert SUPPORTED_TTS_MODEL_VARIANTS["japanese"] == {"japanese"}


def test_colab_loads_japanese_from_community_yaml_config():
    calls = []

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            calls.append(kwargs)
            return "loaded"

    result = load_tts_model(
        FakeTTSModel,
        "japanese",
        "japanese",
        quantize=False,
    )

    assert result == "loaded"
    assert calls == [{"config": JAPANESE_MODEL_CONFIG, "quantize": False}]


def test_colab_uses_japanese_preset_voice_unless_user_configured_one():
    assert default_voice_for_language("japanese", None) == JAPANESE_DEFAULT_VOICE
    assert default_voice_for_language("japanese", "/voices/custom.wav") == JAPANESE_DEFAULT_VOICE
    assert default_voice_for_language("portuguese", "/voices/custom.wav") == "/voices/custom.wav"



def test_colab_forwards_quality_sampler_steps_when_configured():
    calls = []

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            calls.append(kwargs)
            return "loaded"

    result = load_tts_model(
        FakeTTSModel,
        "portuguese",
        "portuguese_24l",
        quantize=False,
        sampler_decode_steps=5,
    )

    assert result == "loaded"
    assert calls == [
        {
            "language": "portuguese_24l",
            "quantize": False,
            "sampler_decode_steps": 5,
        }
    ]
