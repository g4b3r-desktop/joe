import pytest

from joe.services.colab_api import ColabAPIClient


def test_colab_client_requires_https_url():
    with pytest.raises(ValueError, match="HTTPS"):
        ColabAPIClient("http://example.com", "x" * 40)


def test_colab_client_requires_token():
    with pytest.raises(ValueError, match="token"):
        ColabAPIClient("https://example.trycloudflare.com", "")


def test_colab_client_normalizes_url():
    client = ColabAPIClient("https://example.trycloudflare.com/", "x" * 40)
    assert client.base_url == "https://example.trycloudflare.com"



def test_colab_tts_rejects_prepared_voice_state_files(tmp_path):
    from joe.services.colab_api import ColabAPIError

    prepared_voice = tmp_path / "voice.safetensors"
    prepared_voice.write_bytes(b"prepared state")
    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)

    with pytest.raises(ColabAPIError, match="áudio original"):
        client.generate_audio("Olá", "portuguese", prepared_voice)


def test_colab_tts_rejects_non_wav_results(monkeypatch):
    import httpx

    from joe.services.colab_api import ColabAPIError

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: httpx.Response(202, json={"job_id": "job-123"}),
    )
    monkeypatch.setattr(
        client,
        "_wait_for_result",
        lambda _job_id: httpx.Response(200, content=b"not a wave file"),
    )

    with pytest.raises(ColabAPIError, match="RIFF/WAVE"):
        client.generate_audio("Olá", "portuguese")
