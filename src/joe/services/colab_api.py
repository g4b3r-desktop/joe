from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any

import httpx


class ColabAPIError(RuntimeError):
    """Raised when the remote Joe API cannot complete a request."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        cancelled: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.cancelled = cancelled


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
                "continua executando e se a URL do túnel ainda é válida.",
                retryable=True,
            ) from exc

        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            if response.status_code in {401, 403}:
                raise ColabAPIError("Token recusado pela API do Colab. Confira o token.")
            retryable = response.status_code in {408, 425, 429, 500, 502, 503, 504}
            raise ColabAPIError(
                f"API do Colab respondeu HTTP {response.status_code}: {detail}",
                retryable=retryable,
            )
        return response

    def test_connection(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Ask the Colab runtime to stop waiting work or discard a running result."""
        return self._request("DELETE", f"/v1/jobs/{job_id}").json()

    def _wait_for_result(
        self,
        job_id: str,
        job_timeout: float = 1800.0,
        cancel_event: Event | None = None,
    ) -> httpx.Response:
        deadline = time.monotonic() + job_timeout
        transient_failures = 0

        def cancel_remote_job() -> None:
            try:
                self.cancel_job(job_id)
            except ColabAPIError:
                # The desktop cancellation remains final if the tunnel disappeared.
                pass
            raise ColabAPIError(
                "Geração cancelada. Uma inferência já iniciada no Colab pode terminar "
                "no runtime, mas o resultado será descartado.",
                cancelled=True,
            )

        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                cancel_remote_job()
            try:
                status = self._request("GET", f"/v1/jobs/{job_id}").json()
                state = status.get("status")
                if state == "completed":
                    if cancel_event is not None and cancel_event.is_set():
                        cancel_remote_job()
                    return self._request("GET", f"/v1/jobs/{job_id}/result")
                if state == "failed":
                    raise ColabAPIError(
                        f"Falha na tarefa remota: {status.get('error') or 'erro desconhecido'}"
                    )
                if state == "cancelled":
                    raise ColabAPIError("A tarefa remota foi cancelada.", cancelled=True)
                transient_failures = 0
                delay = 1.2
            except ColabAPIError as exc:
                if exc.cancelled or not exc.retryable:
                    raise
                transient_failures += 1
                delay = min(1.2 * (2 ** (transient_failures - 1)), 8.0)

            remaining = deadline - time.monotonic()
            if remaining > 0:
                wait_seconds = min(delay, remaining)
                if cancel_event is not None:
                    cancel_event.wait(wait_seconds)
                else:
                    time.sleep(wait_seconds)
        raise ColabAPIError(
            "A inferência no Colab excedeu 30 minutos. Confira o estado do runtime "
            "e tente novamente com um arquivo menor."
        )

    @staticmethod
    def _asset_file_info(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        if size <= 0:
            raise ValueError(f"Arquivo vazio não pode ser enviado: {path}")
        return {"name": path.name, "size": size, "sha256": digest.hexdigest()}

    def _send_asset_chunks(self, upload_id: str, asset_key: str, path: Path) -> None:
        chunk_size = 8 * 1024 * 1024
        with path.open("rb") as stream:
            index = 0
            while True:
                data = stream.read(chunk_size)
                if not data:
                    break
                response = self._request(
                    "PUT",
                    f"/v1/assets/uploads/{upload_id}/{asset_key}/{index}",
                    files={
                        "chunk": (
                            f"{asset_key}-{index:06d}.part",
                            __import__("io").BytesIO(data),
                            "application/octet-stream",
                        )
                    },
                )
                result = response.json()
                if result.get("chunk_index") != index:
                    raise ColabAPIError(
                        f"O Colab confirmou um bloco inesperado do arquivo {asset_key}."
                    )
                index += 1

    def upload_model_bundle(
        self,
        model_directory: str | Path,
        language: str,
        model_variant: str,
    ) -> dict[str, Any]:
        """Upload checkpoint + tokenizer individually in resumable-sized chunks."""
        root = Path(model_directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"Pasta do modelo não encontrada: {root}")
        checkpoint = root / "model.safetensors"
        tokenizer = root / "tokenizer.json"
        for file_path in (checkpoint, tokenizer):
            if not file_path.is_file():
                raise FileNotFoundError(
                    "A pasta do modelo precisa conter model.safetensors e tokenizer.json "
                    f"no nível principal. Arquivo ausente: {file_path}"
                )
        payload = {
            "kind": "model",
            "language": language,
            "model_variant": model_variant,
            "files": {
                "checkpoint": self._asset_file_info(checkpoint),
                "tokenizer": self._asset_file_info(tokenizer),
            },
        }
        upload = self._request("POST", "/v1/assets/uploads", json=payload).json()
        upload_id = str(upload.get("upload_id", ""))
        if not upload_id:
            raise ColabAPIError("O Colab não retornou o identificador de upload do modelo.")
        self._send_asset_chunks(upload_id, "checkpoint", checkpoint)
        self._send_asset_chunks(upload_id, "tokenizer", tokenizer)
        result = self._request(
            "POST", f"/v1/assets/uploads/{upload_id}/commit"
        ).json()
        return result

    def prepare_voice(
        self,
        audio_path: str | Path,
        language: str,
        model_variant: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Upload original reference audio and prepare/cache its voice state on Colab."""
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Áudio de referência não encontrado: {path}")
        if path.suffix.lower() == ".safetensors":
            raise ValueError(
                "No armazenamento direto, selecione o WAV/FLAC/MP3/OGG original. "
                "O estado Safetensors será preparado pelo próprio Colab."
            )
        payload = {
            "kind": "voice",
            "language": language,
            "model_variant": model_variant,
            "name": name or path.stem,
            "file": self._asset_file_info(path),
        }
        upload = self._request("POST", "/v1/assets/uploads", json=payload).json()
        upload_id = str(upload.get("upload_id", ""))
        if not upload_id:
            raise ColabAPIError("O Colab não retornou o identificador de upload da voz.")
        self._send_asset_chunks(upload_id, "voice", path)
        return self._request(
            "POST", f"/v1/assets/uploads/{upload_id}/commit"
        ).json()

    def list_assets(self) -> dict[str, Any]:
        """Return the models and prepared voices currently persisted in Google Drive."""
        return self._request("GET", "/v1/assets").json()

    def generate_audio(
        self,
        text: str,
        language: str,
        voice_path: str | Path | None = None,
        cancel_event: Event | None = None,
        model_variant: str | None = None,
        voice_id: str | None = None,
    ) -> bytes:
        data = {
            "text": text,
            "language": language,
            "model_variant": model_variant or language,
        }
        if voice_id:
            data["voice_id"] = voice_id
        files: list[Any] = []
        opened = None
        try:
            if voice_path:
                path = Path(voice_path).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"Arquivo de voz não encontrado: {path}")
                opened = path.open("rb")
                # Pocket TTS accepts both reference recordings and exported
                # .safetensors voice states through get_state_for_audio_prompt().
                # Keep the original suffix so the Colab worker can detect the format.
                files = [("voice", (path.name, opened, "application/octet-stream"))]
            response = self._request("POST", "/v1/jobs/tts", data=data, files=files)
        finally:
            if opened:
                opened.close()
        job_id = response.json().get("job_id")
        if not job_id:
            raise ColabAPIError("A API não retornou o identificador da tarefa TTS.")
        audio = self._wait_for_result(job_id, cancel_event=cancel_event).content
        if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            raise ColabAPIError(
                "A API remota não retornou um WAV válido (cabeçalho RIFF/WAVE ausente). "
                "Confira o erro da tarefa no Colab e o formato da voz enviada. "
                "São aceitos áudios de referência e estados de voz .safetensors."
            )
        return audio

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
