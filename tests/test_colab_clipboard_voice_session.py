from __future__ import annotations

from pathlib import Path

import pytest

from joe.services.colab_api import ColabAPIError, ColabVoiceSession


class FakeColabClient:
    def __init__(self, voice_id: str = "voice-drive-123") -> None:
        self.prepared_calls = []
        self.generated_calls = []
        self.voice_id = voice_id

    def prepare_voice(self, audio_path, language, model_variant, name=None):
        self.prepared_calls.append((Path(audio_path), language, model_variant, name))
        return {"voice_id": self.voice_id}

    def generate_audio(
        self,
        text,
        language,
        voice_path=None,
        cancel_event=None,
        model_variant=None,
        voice_id=None,
    ):
        self.generated_calls.append({
            "text": text,
            "language": language,
            "voice_path": Path(voice_path) if voice_path else None,
            "model_variant": model_variant,
            "voice_id": voice_id,
        })
        return b"RIFF" + b"\\x00" * 4 + b"WAVE"


def test_clipboard_voice_session_prepares_once_and_reuses_drive_voice_id(tmp_path):
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"reference-audio")
    client = FakeColabClient()
    session = ColabVoiceSession(
        client,
        audio_path,
        "portuguese",
        "portuguese",
        name="Minha voz",
    )

    # The worker preflights the voice once, before generating the first phrase.
    assert session.prepare() == "voice-drive-123"
    assert session.prepare() == "voice-drive-123"

    first = session.generate_audio("Primeira frase")
    second = session.generate_audio("Segunda frase")

    assert first.startswith(b"RIFF")
    assert second.startswith(b"RIFF")
    assert len(client.prepared_calls) == 1
    assert client.prepared_calls[0] == (
        audio_path,
        "portuguese",
        "portuguese",
        "Minha voz",
    )
    assert [call["text"] for call in client.generated_calls] == [
        "Primeira frase",
        "Segunda frase",
    ]
    assert all(call["voice_id"] == "voice-drive-123" for call in client.generated_calls)
    assert all(call["voice_path"] == audio_path for call in client.generated_calls)


def test_clipboard_voice_session_without_custom_voice_uses_colab_default():
    client = FakeColabClient()
    session = ColabVoiceSession(
        client,
        None,
        "english",
        "english",
    )

    session.prepare()
    audio = session.generate_audio("Hello there")

    assert audio.startswith(b"RIFF")
    assert client.prepared_calls == []
    assert client.generated_calls == [{
        "text": "Hello there",
        "language": "english",
        "voice_path": None,
        "model_variant": "english",
        "voice_id": None,
    }]


def test_clipboard_voice_session_requires_voice_id_from_colab(tmp_path):
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"reference-audio")
    client = FakeColabClient(voice_id="")

    session = ColabVoiceSession(client, audio_path, "portuguese", "portuguese")

    with pytest.raises(ColabAPIError, match="identificador da voz"):
        session.prepare()
    assert client.generated_calls == []
