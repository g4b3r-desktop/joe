from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx


class ColabAPIError(RuntimeError):
    """Raised when the remote Joe API cannot complete a request."""


class ColabAPIClient:
    """HTTP client for a Joe API started in Google Colab."""

    def __init__(self, base_url: str, token: str, timeout: float = 1800.0) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.token = token.strip()
        if not self.base_url.startswith("https://"):
            raise ValueError("A API remota exige a URL HTTPS trycloudflare exibida pelo notebook.")
        if not self.token:
            raise ValueError("Informe o token de acesso gerado pelo notebook do Colab.")
        self.timeout = httpx.Timeout(
            timeout,
            connect=20.0,
            read=45.0,
            write=300.0,
            pool=20.0,
        )

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.token}"
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
                response = client.request(
                    method,
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    **kwargs,
                )
        except httpx.RequestError as exc:
            raise ColabAPIError(
                "Não foi possível alcançar a API do Colab. Verifique se o notebook "
                "continua executando e se a URL do túnel ainda é válida."
            ) from exc

        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            if response.status_code in {401, 403}:
                raise ColabAPIError("Token recusado pela API do Colab. Confira o token.")
            raise ColabAPIError(f"API do Colab respondeu HTTP {response.status_code}: {detail}")
        return response

    def test_connection(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def _wait_for_result(self, job_id: str, job_timeout: float = 1800.0) -> httpx.Response:
        deadline = time.monotonic() + job_timeout
        while time.monotonic() < deadline:
            status = self._request("GET", f"/v1/jobs/{job_id}").json()
            state = status.get("status")
            if state == "completed":
                return self._request("GET", f"/v1/jobs/{job_id}/result")
            if state == "failed":
                raise ColabAPIError(
                    f"Falha na tarefa remota: {status.get('error') or 'erro desconhecido'}"
                )
            time.sleep(1.2)
        raise ColabAPIError(
            "A inferência no Colab excedeu 30 minutos. Confira o estado do runtime "
            "e tente novamente com um arquivo menor."
        )

    def generate_audio(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
    ) -> bytes:
        data = {"text": text, "language": language}
        files: list[Any] = []
        opened = None
        try:
            if voice_path:
                path = Path(voice_path)
                opened = path.open("rb")
                files = [("voice", (path.name, opened, "application/octet-stream"))]
            response = self._request("POST", "/v1/jobs/tts", data=data, files=files)
        finally:
            if opened:
                opened.close()
        job_id = response.json().get("job_id")
        if not job_id:
            raise ColabAPIError("A API não retornou o identificador da tarefa TTS.")
        return self._wait_for_result(job_id).content

    def transcribe(
        self,
        audio_path: str | Path,
        timestamps: bool = True,
    ) -> dict[str, Any]:
        path = Path(audio_path)
        with path.open("rb") as audio_file:
            response = self._request(
                "POST",
                "/v1/jobs/transcribe",
                data={"timestamps": "segment" if timestamps else "none"},
                files={"audio": (path.name, audio_file, "application/octet-stream")},
            )
        job_id = response.json().get("job_id")
        if not job_id:
            raise ColabAPIError("A API não retornou o identificador da tarefa de transcrição.")
        return self._wait_for_result(job_id).json()
