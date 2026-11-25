import pytest

from joe.services.colab_api import ColabAPIClient, split_tts_batches


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
            200, content=b"RIFF" + bytes(4) + b"WAVE"
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


def test_colab_tts_settings_send_cpu_int8_flag(monkeypatch):
    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    submitted = {}

    def request(method, endpoint, **kwargs):
        submitted.update(method=method, endpoint=endpoint, json=kwargs.get("json"))
        import httpx
        return httpx.Response(200, json={"ok": True, "tts_quantize_int8": True})

    monkeypatch.setattr(client, "_request", request)
    result = client.update_tts_settings(2, 4, quantize_int8=True)

    assert result["tts_quantize_int8"] is True
    assert submitted["method"] == "POST"
    assert submitted["endpoint"] == "/v1/settings/tts"
    assert submitted["json"] == {
        "cpu_threads": 2,
        "sampler_decode_steps": 4,
        "quantize_int8": True,
    }
    client.close()


def test_colab_batch_downloads_one_zip_for_all_segments(monkeypatch):
    import io
    import zipfile

    import httpx

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    archive_buffer = io.BytesIO()
    wav_one = b"RIFF" + bytes(4) + b"WAVE" + b"first"
    wav_two = b"RIFF" + bytes(4) + b"WAVE" + b"second"
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("segment_0001.wav", wav_one)
        archive.writestr("segment_0002.wav", wav_two)
    submitted = {}

    def request(method, endpoint, **kwargs):
        submitted.update(method=method, endpoint=endpoint, json=kwargs.get("json"))
        return httpx.Response(202, json={"job_id": "batch-123"})

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(
        client,
        "_wait_for_result",
        lambda job_id, **kwargs: httpx.Response(200, content=archive_buffer.getvalue()),
    )

    result = client.generate_audio_batch(
        ["Olá.", "Tudo bem?"],
        "portuguese",
        model_variant="portuguese_24l",
        voice_id="voice-123",
    )

    assert result == [wav_one, wav_two]
    assert submitted["method"] == "POST"
    assert submitted["endpoint"] == "/v1/jobs/tts/batch"
    assert submitted["json"] == {
        "texts": ["Olá.", "Tudo bem?"],
        "language": "portuguese",
        "model_variant": "portuguese_24l",
        "voice_id": "voice-123",
    }
    client.close()


def test_colab_client_reuses_persistent_httpx_pool(monkeypatch):
    import httpx

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    pool = client._get_http_client()
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, id(pool)))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(pool, "request", request)
    assert client.test_connection()["ok"] is True
    assert client.test_connection()["ok"] is True
    assert client._get_http_client() is pool
    assert len(calls) == 2
    assert {call[2] for call in calls} == {id(pool)}
    client.close()
    assert pool.is_closed


def test_split_tts_batches_respects_character_and_segment_limits():
    texts = ["a" * 8_000, "b" * 8_000, "c" * 8_000]

    batches = split_tts_batches(texts, max_characters=18_000, max_segments=2)

    assert batches == [[texts[0], texts[1]], [texts[2]]]
    assert all(sum(map(len, batch)) <= 18_000 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def test_split_tts_batches_rejects_segment_larger_than_batch_limit():
    import pytest

    with pytest.raises(ValueError, match="excede o limite"):
        split_tts_batches(["texto muito longo"], max_characters=4)



def test_split_tts_batches_supports_full_20k_request_and_512_phrases():
    texts = ["frase"] * 513

    batches = split_tts_batches(texts)

    assert [len(batch) for batch in batches] == [512, 1]
    assert all(sum(map(len, batch)) <= 20_000 for batch in batches)

    full_limit = split_tts_batches(["a" * 20_000])
    assert full_limit == [["a" * 20_000]]



def test_colab_batch_falls_back_to_legacy_tts_route_on_404(monkeypatch):
    from joe.services.colab_api import ColabAPIError

    client = ColabAPIClient("https://example.trycloudflare.com", "x" * 40)
    batch_requests = []
    legacy_requests = []

    def request(method, endpoint, **_kwargs):
        batch_requests.append((method, endpoint))
        raise ColabAPIError("API do Colab respondeu HTTP 404: Not Found")

    def legacy_generate(
        text,
        language,
        voice_path=None,
        cancel_event=None,
        model_variant=None,
        voice_id=None,
    ):
        legacy_requests.append(
            {
                "text": text,
                "language": language,
                "voice_path": voice_path,
                "model_variant": model_variant,
                "voice_id": voice_id,
            }
        )
        return b"RIFF" + bytes(4) + b"WAVE" + text.encode("utf-8")

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "generate_audio", legacy_generate)

    result = client.generate_audio_batch(
        ["Primeira frase.", "Segunda frase.", "Terceira frase."],
        "portuguese",
        model_variant="portuguese_24l",
        voice_id="voice-prepared-in-drive",
    )

    assert len(batch_requests) == 1
    assert batch_requests == [("POST", "/v1/jobs/tts/batch")]
    assert len(result) == 3
    assert [item["text"] for item in legacy_requests] == [
        "Primeira frase.",
        "Segunda frase.",
        "Terceira frase.",
    ]
    assert all(item["language"] == "portuguese" for item in legacy_requests)
    assert all(item["model_variant"] == "portuguese_24l" for item in legacy_requests)
    assert all(
        item["voice_id"] == "voice-prepared-in-drive"
        and item["voice_path"] is None
        for item in legacy_requests
    )
    client.close()
