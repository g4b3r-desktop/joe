from __future__ import annotations

import hashlib
import io
from pathlib import Path

import httpx

from joe.services.colab_api import ColabAPIClient


def _client():
    return ColabAPIClient("https://example.trycloudflare.com", "x" * 40)


def test_upload_model_bundle_sends_checkpoint_and_tokenizer_in_chunks(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    checkpoint = model_dir / "model.safetensors"
    tokenizer = model_dir / "tokenizer.json"
    checkpoint.write_bytes(b"model-weights")
    tokenizer.write_text('{"version":"1"}', encoding="utf-8")

    client = _client()
    calls = []
    chunk_payloads = []

    def request(method, endpoint, **kwargs):
        calls.append((method, endpoint))
        if endpoint == "/v1/assets/uploads":
            payload = kwargs["json"]
            assert payload["kind"] == "model"
            assert payload["model_variant"] == "portuguese_24l"
            assert payload["files"]["checkpoint"]["sha256"] == hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            return httpx.Response(200, json={"upload_id": "a" * 32})
        if method == "PUT":
            chunk = kwargs["files"]["chunk"][1].read()
            chunk_payloads.append((endpoint, chunk))
            return httpx.Response(200, json={"chunk_index": 0})
        if endpoint.endswith("/commit"):
            return httpx.Response(
                200,
                json={"kind": "model", "model_variant": "portuguese_24l"},
            )
        raise AssertionError((method, endpoint))

    monkeypatch.setattr(client, "_request", request)

    result = client.upload_model_bundle(
        model_dir, "portuguese", "portuguese_24l"
    )

    assert result["kind"] == "model"
    assert [item[1] for item in chunk_payloads] == [
        checkpoint.read_bytes(),
        tokenizer.read_bytes(),
    ]
    assert len([call for call in calls if call[0] == "PUT"]) == 2
    assert calls[-1] == (
        "POST",
        f"/v1/assets/uploads/{'a' * 32}/commit",
    )


def test_prepare_voice_uploads_reference_audio_and_returns_voice_id(monkeypatch, tmp_path):
    audio = tmp_path / "Kessya.wav"
    audio.write_bytes(b"fake wav reference")
    client = _client()
    monkeypatch.setattr(client, "list_assets", lambda: {"models": [], "voices": []})
    submitted_chunks = []

    def request(method, endpoint, **kwargs):
        if endpoint == "/v1/assets/uploads":
            payload = kwargs["json"]
            assert payload["kind"] == "voice"
            assert payload["name"] == "Kessya"
            assert payload["file"]["sha256"] == hashlib.sha256(
                audio.read_bytes()
            ).hexdigest()
            return httpx.Response(200, json={"upload_id": "b" * 32})
        if method == "PUT":
            submitted_chunks.append(kwargs["files"]["chunk"][1].read())
            return httpx.Response(200, json={"chunk_index": 0})
        if endpoint.endswith("/commit"):
            return httpx.Response(
                200,
                json={
                    "kind": "voice",
                    "voice_id": "kessya-1234",
                    "reused": False,
                },
            )
        raise AssertionError((method, endpoint))

    monkeypatch.setattr(client, "_request", request)

    result = client.prepare_voice(
        audio, "portuguese", "portuguese_24l", name="Kessya"
    )

    assert result["voice_id"] == "kessya-1234"
    assert submitted_chunks == [audio.read_bytes()]


def test_prepare_voice_rejects_safetensors_without_uploading(tmp_path):
    voice_state = tmp_path / "voice.safetensors"
    voice_state.write_bytes(b"prepared-state")

    client = _client()

    try:
        client.prepare_voice(voice_state, "portuguese", "portuguese_24l")
    except ValueError as exc:
        assert "áudio original" in str(exc)
    else:
        raise AssertionError("A direct Drive flow must require the original reference audio.")


def test_generate_audio_sends_drive_voice_id_without_file_upload(monkeypatch):
    client = _client()
    submitted = {}

    def request(method, endpoint, **kwargs):
        if endpoint == "/v1/jobs/tts":
            submitted.update(kwargs["data"])
            submitted["files"] = kwargs["files"]
            return httpx.Response(202, json={"job_id": "job-voice-id"})
        if endpoint == "/v1/jobs/job-voice-id":
            return httpx.Response(200, json={"status": "completed"})
        if endpoint == "/v1/jobs/job-voice-id/result":
            return httpx.Response(200, content=b"RIFF" + b"\x00" * 4 + b"WAVE")
        raise AssertionError((method, endpoint))

    monkeypatch.setattr(client, "_request", request)

    audio = client.generate_audio(
        "Olá",
        "portuguese",
        model_variant="portuguese_24l",
        voice_id="kessya-1234",
    )

    assert audio.startswith(b"RIFF")
    assert submitted["voice_id"] == "kessya-1234"
    assert submitted["files"] == []
