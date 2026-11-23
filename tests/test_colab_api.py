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



def test_colab_tts_uploads_prepared_voice_state_files(monkeypatch, tmp_path):
    import httpx

    prepared_voice = tmp_path / "jane.safetensors"
    prepared_voice.write_bytes(b"prepared pocket tts state")
    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    submitted = {}

    def request(method, endpoint, **kwargs):
        assert method == "POST"
        assert endpoint == "/v1/jobs/tts"
        uploaded = kwargs["files"][0]
        submitted["field"] = uploaded[0]
        submitted["filename"] = uploaded[1][0]
        submitted["content_type"] = uploaded[1][2]
        submitted["content"] = uploaded[1][1].read()
        submitted["data"] = kwargs["data"]
        return httpx.Response(202, json={"job_id": "job-voice-state"})

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(
        client,
        "_wait_for_result",
        lambda *_args, **_kwargs: httpx.Response(
            200, content=b"RIFF" + b"\\x00" * 4 + b"WAVE"
        ),
    )

    result = client.generate_audio(
        "Olá",
        "portuguese",
        prepared_voice,
        model_variant="portuguese",
    )

    assert result.startswith(b"RIFF")
    assert submitted["field"] == "voice"
    assert submitted["filename"] == "jane.safetensors"
    assert submitted["content_type"] == "application/octet-stream"
    assert submitted["content"] == b"prepared pocket tts state"
    assert submitted["data"]["language"] == "portuguese"
    assert submitted["data"]["model_variant"] == "portuguese"


def test_colab_tts_reports_missing_voice_file_clearly(tmp_path):
    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    with pytest.raises(FileNotFoundError, match="Arquivo de voz não encontrado"):
        client.generate_audio(
            "Olá", "portuguese", tmp_path / "missing.safetensors"
        )


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
        lambda _job_id, **_kwargs: httpx.Response(200, content=b"not a wave file"),
    )

    with pytest.raises(ColabAPIError, match="RIFF/WAVE"):
        client.generate_audio("Olá", "portuguese")



def test_colab_job_polling_retries_transient_connection_errors(monkeypatch):
    import httpx

    from joe.services.colab_api import ColabAPIError

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    calls = []

    def request(method, endpoint, **_kwargs):
        calls.append((method, endpoint))
        if len(calls) == 1:
            raise ColabAPIError("temporary tunnel failure", retryable=True)
        if endpoint == "/v1/jobs/job-123":
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, content=b"RIFF" + b"\x00" * 4 + b"WAVE")

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr("joe.services.colab_api.time.sleep", lambda _seconds: None)

    result = client._wait_for_result("job-123", job_timeout=1)

    assert result.content[0:4] == b"RIFF"
    assert len(calls) == 3


def test_colab_job_polling_does_not_retry_authentication_errors(monkeypatch):
    from joe.services.colab_api import ColabAPIError

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    calls = []

    def request(*_args, **_kwargs):
        calls.append(True)
        raise ColabAPIError("Token recusado")

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr("joe.services.colab_api.time.sleep", lambda _seconds: None)

    with pytest.raises(ColabAPIError, match="Token recusado"):
        client._wait_for_result("job-123", job_timeout=1)

    assert len(calls) == 1



def test_colab_job_polling_requests_remote_cancellation(monkeypatch):
    from threading import Event

    from joe.services.colab_api import ColabAPIError

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    cancel_event = Event()
    cancel_calls = []

    def request(method, endpoint, **_kwargs):
        if method == "GET":
            cancel_event.set()
            return __import__("httpx").Response(200, json={"status": "running"})
        cancel_calls.append((method, endpoint))
        return __import__("httpx").Response(200, json={"status": "cancelling"})

    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(ColabAPIError, match="Geração cancelada") as error:
        client._wait_for_result("job-123", job_timeout=1, cancel_event=cancel_event)

    assert error.value.cancelled
    assert cancel_calls == [("DELETE", "/v1/jobs/job-123")]


def test_colab_job_polling_stops_when_remote_reports_cancelled(monkeypatch):
    import httpx

    from joe.services.colab_api import ColabAPIError

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: httpx.Response(200, json={"status": "cancelled"}),
    )

    with pytest.raises(ColabAPIError, match="tarefa remota foi cancelada") as error:
        client._wait_for_result("job-123", job_timeout=1)

    assert error.value.cancelled


def test_colab_client_sends_selected_pocket_tts_model_variant(monkeypatch):
    import httpx

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    submitted = {}

    def request(_method, _endpoint, **kwargs):
        submitted.update(kwargs["data"])
        return httpx.Response(202, json={"job_id": "job-model-variant"})

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(
        client,
        "_wait_for_result",
        lambda *_args, **_kwargs: httpx.Response(
            200, content=b"RIFF" + b"\x00" * 4 + b"WAVE"
        ),
    )

    client.generate_audio(
        "Olá",
        "portuguese",
        model_variant="portuguese_24l",
    )

    assert submitted["language"] == "portuguese"
    assert submitted["model_variant"] == "portuguese_24l"
