from __future__ import annotations

from joe.tts_quality import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEACHER_FORCE_PREVIOUS_CHUNK,
    generate_audio_with_quality,
)


class ModernPocketTTS:
    def __init__(self):
        self.received = None

    def generate_audio(
        self,
        model_state,
        text,
        max_tokens=None,
        teacher_force_previous_chunk=False,
    ):
        self.received = {
            "model_state": model_state,
            "text": text,
            "max_tokens": max_tokens,
            "teacher_force_previous_chunk": teacher_force_previous_chunk,
        }
        return "audio"


class LegacyPocketTTS:
    def generate_audio(self, model_state, text):
        return (model_state, text)


def test_quality_profile_uses_shorter_chunks_and_previous_audio_context():
    model = ModernPocketTTS()

    result = generate_audio_with_quality(model, "voice-state", "Texto de teste.")

    assert result == "audio"
    assert model.received == {
        "model_state": "voice-state",
        "text": "Texto de teste.",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "teacher_force_previous_chunk": DEFAULT_TEACHER_FORCE_PREVIOUS_CHUNK,
    }
    assert DEFAULT_MAX_TOKENS == 25
    assert DEFAULT_TEACHER_FORCE_PREVIOUS_CHUNK is True


def test_quality_profile_remains_compatible_with_older_pocket_tts_api():
    model = LegacyPocketTTS()

    assert generate_audio_with_quality(
        model, "voice-state", "Texto de teste."
    ) == ("voice-state", "Texto de teste.")


def test_quality_profile_passes_options_to_keyword_argument_api():
    class FlexiblePocketTTS:
        def __init__(self):
            self.options = None

        def generate_audio(self, model_state, text, **kwargs):
            self.options = kwargs
            return "audio"

    model = FlexiblePocketTTS()
    generate_audio_with_quality(model, "voice", "texto")

    assert model.options == {
        "max_tokens": 25,
        "teacher_force_previous_chunk": True,
    }
