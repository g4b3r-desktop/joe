from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path
from threading import Event, Lock
from typing import Any

import httpx


MAX_TTS_TEXT_CHARS = 20_000
MAX_TTS_BATCH_SEGMENTS = 512


def split_tts_batches(
    texts: list[str],
    *,
    max_characters: int = MAX_TTS_TEXT_CHARS,
    max_segments: int = MAX_TTS_BATCH_SEGMENTS,
) -> list[list[str]]:
    """Split TTS text chunks into requests within the Colab API batch limits."""
    if max_characters < 1 or max_segments < 1:
        raise ValueError("Os limites do lote precisam ser maiores que zero.")
    batches: list[list[str]] = []
    current: list[str] = []
    current_characters = 0
    for value in texts:
        text = str(value).strip()
        if not text:
            raise ValueError("O lote de áudio contém um segmento vazio.")
        if len(text) > max_characters:
            raise ValueError(
                f"Um segmento contém {len(text)} caracteres e excede o limite de {max_characters}."
            )
        if current and (
            len(current) >= max_segments
            or current_characters + len(text) > max_characters
        ):
            batches.append(current)
            current = []
            current_characters = 0
        current.append(text)
        current_characters += len(text)
    if current:
        batches.append(current)
    return batches


def _download_hf_tokenizer(filename: str, revision: str, cache_dir: Path) -> Path:
    """Download one tokenizer file from the exact Hugging Face model revision."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "O tokenizer.json não está no cache local e huggingface_hub não está "
            "instalado. Atualize as dependências do Joe para permitir o download automático."
        ) from exc

    try:
        downloaded = hf_hub_download(
            repo_id="kyutai/pocket-tts",
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            "O Joe encontrou os pesos no cache do Hugging Face, mas não conseguiu "
            "obter o tokenizer.json da mesma revisão. Confira sua conexão, execute "
            "'hf auth login' e aceite as condições de acesso ao repositório "
            "'kyutai/pocket-tts' no Hugging Face. Detalhe: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return Path(downloaded)


def _find_model_checkpoint(root: Path, language: str, model_variant: str) -> Path:
    """Find a model checkpoint in a bundle, language folder, or HF snapshot."""
    candidates = [
        root / "model.safetensors",
        root / "languages" / model_variant / "model.safetensors",
        root / model_variant / "model.safetensors",
        root / "languages" / language / "model.safetensors",
        root / language / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # A snapshot root can contain multiple language folders. Prefer an exact
    # variant folder, then the language folder; never choose an arbitrary model.
    for folder_name in (model_variant, language):
        matches = sorted({
            path for path in root.glob(f"**/{folder_name}/model.safetensors")
            if path.is_file()
        })
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Foram encontrados vários checkpoints para '{folder_name}' em {root}. "
                "Selecione a pasta languages/<idioma> do snapshot desejado."
            )

    raise FileNotFoundError(
        "Não encontrei model.safetensors na pasta selecionada. Selecione a pasta "
        "languages/<idioma> do cache Hugging Face, a pasta do snapshot ou uma pasta "
        "que contenha model.safetensors."
    )


def _resolve_model_bundle_files(
    root: Path, language: str, model_variant: str
) -> tuple[Path, Path]:
    """Resolve the checkpoint and its matching tokenizer, downloading only if needed."""
    checkpoint = _find_model_checkpoint(root, language, model_variant)
    tokenizer = checkpoint.parent / "tokenizer.json"
    if tokenizer.is_file():
        return checkpoint, tokenizer

    # A generic hand-made bundle may keep the tokenizer one level above the
    # checkpoint. Do not use this fallback for languages/<id>, where tokenizers
    # are language-specific.
    is_language_folder = checkpoint.parent.parent.name == "languages"
    root_tokenizer = root / "tokenizer.json"
    if not is_language_folder and root_tokenizer.is_file():
        return checkpoint, root_tokenizer

    # HF snapshots are immutable revisions. Derive the tokenizer path from the
    # checkpoint's own path and fetch it from that exact commit, never from main.
    snapshot_root = next(
        (
            ancestor for ancestor in checkpoint.parents
            if ancestor.parent.name == "snapshots"
        ),
        None,
    )
    is_official_hf_cache = any(
        ancestor.name == "models--kyutai--pocket-tts"
        for ancestor in checkpoint.parents
    )
    if snapshot_root is not None and is_official_hf_cache:
        relative_folder = checkpoint.parent.relative_to(snapshot_root)
        tokenizer_filename = (relative_folder / "tokenizer.json").as_posix()
        cached_candidate = snapshot_root / relative_folder / "tokenizer.json"
        if cached_candidate.is_file():
            return checkpoint, cached_candidate
        cache_dir = snapshot_root.parents[2]
        tokenizer = _download_hf_tokenizer(
            tokenizer_filename, snapshot_root.name, cache_dir
        )
        if not tokenizer.is_file():
            raise FileNotFoundError(
                f"O Hugging Face não retornou o tokenizer esperado: {tokenizer}"
            )
        return checkpoint, tokenizer

    raise FileNotFoundError(
        "O arquivo tokenizer.json não foi encontrado junto do modelo. Selecione a "
        "pasta completa do modelo ou use o snapshot do cache Hugging Face para que "
        "o Joe baixe automaticamente o tokenizer da mesma revisão."
    )


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
        self._client_lock = Lock()
        self._http_client: httpx.Client | None = self._create_http_client()

    def _create_http_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=12,
                max_keepalive_connections=8,
                keepalive_expiry=45.0,
            ),
        )

    def _get_http_client(self) -> httpx.Client:
        with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = self._create_http_client()
            return self._http_client

    def close(self) -> None:
        """Close the persistent HTTP pool when its owning task/session ends."""
        with self._client_lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    def __enter__(self) -> "ColabAPIClient":
        self._get_http_client()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self._get_http_client().request(
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

            if response.status_code == 530:
                # Cloudflare 1033 means the named/quick tunnel cannot currently
                # resolve a connected cloudflared origin. The Colab job may still
                # be running, so let _wait_for_result retry status/result polling
                # instead of failing the shortcut on the first tunnel interruption.
                body_text = response.text.lower()
                if "1033" in body_text:
                    detail = (
                        "O Cloudflare não conseguiu localizar uma conexão ativa "
                        "para o túnel do Colab (erro 1033). O trabalho remoto pode "
                        "continuar executando; o Joe tentará consultar novamente "
                        "quando o túnel voltar."
                    )
                else:
                    detail = (
                        "O Cloudflare retornou HTTP 530. O túnel do Colab está "
                        "temporariamente indisponível; o Joe tentará consultar novamente."
                    )

            retryable = response.status_code in {408, 425, 429, 500, 502, 503, 504, 530}
            raise ColabAPIError(
                f"API do Colab respondeu HTTP {response.status_code}: {detail}",
                retryable=retryable,
            )
        return response

    def test_connection(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def update_tts_settings(
        self,
        cpu_threads: int,
        sampler_decode_steps: int,
        quantize_int8: bool = False,
    ) -> dict[str, Any]:
        """Apply CPU threads, sampler quality and CPU-only INT8 on Colab."""
        return self._request(
            "POST",
            "/v1/settings/tts",
            json={
                "cpu_threads": int(cpu_threads),
                "sampler_decode_steps": int(sampler_decode_steps),
                "quantize_int8": bool(quantize_int8),
            },
        ).json()

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
                delay = 0.75
            except ColabAPIError as exc:
                if exc.cancelled or not exc.retryable:
                    raise
                transient_failures += 1
                delay = min(0.75 * (2 ** (transient_failures - 1)), 8.0)

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
        checkpoint, tokenizer = _resolve_model_bundle_files(
            root, language, model_variant
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

    @staticmethod
    def _drive_model_fingerprint(manifest: dict[str, Any]) -> str:
        fields = {
            "model_variant": manifest.get("model_variant"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "tokenizer_sha256": manifest.get("tokenizer_sha256"),
            "pocket_tts_version": manifest.get("pocket_tts_version"),
        }
        raw = __import__("json").dumps(
            fields, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

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
                "No armazenamento direto, selecione o áudio original (WAV/FLAC/MP3/OGG). "
                "O estado Safetensors será preparado pelo próprio Colab."
            )
        file_info = self._asset_file_info(path)
        # Avoid retransmitting and re-encoding a voice already stored with this
        # exact checkpoint/tokenizer/Pocket TTS version in the remote Drive.
        inventory = self.list_assets()
        models = inventory.get("models", [])
        if not isinstance(models, list):
            raise ColabAPIError(
                "A API do Colab retornou um inventário de modelos inválido. "
                "Reinicie a API usando o notebook atualizado."
            )
        model_manifest = next(
            (
                item for item in models
                if isinstance(item, dict) and item.get("model_variant") == model_variant
            ),
            None,
        )
        if not model_manifest:
            available = sorted(
                str(item.get("model_variant"))
                for item in models
                if isinstance(item, dict) and item.get("model_variant")
            )
            invalid = inventory.get("invalid_models", [])
            detail = (
                "; manifestos inválidos: "
                + ", ".join(
                    f"{item.get('model_variant')}: {item.get('error')}"
                    for item in invalid
                    if isinstance(item, dict)
                )
                if invalid else ""
            )
            variants = ", ".join(available) if available else "nenhuma"
            raise ColabAPIError(
                f"O modelo '{model_variant}' não foi encontrado no Google Drive "
                f"(variantes válidas: {variants}). Envie primeiro a pasta do modelo "
                f"pela aba Google Colab / API do Joe e confirme que ela aparece em "
                f"GET /v1/assets.{detail}"
            )
        expected_fingerprint = self._drive_model_fingerprint(model_manifest)
        voices = inventory.get("voices", [])
        if not isinstance(voices, list):
            voices = []
        for item in voices:
            if (
                isinstance(item, dict)
                and item.get("model_variant") == model_variant
                and item.get("source_sha256") == file_info["sha256"]
                and item.get("model_fingerprint") == expected_fingerprint
                and item.get("state_valid") is True
            ):
                return {
                    "kind": "voice",
                    "voice_id": item.get("voice_id"),
                    "name": item.get("name"),
                    "model_variant": model_variant,
                    "reused": True,
                }

        payload = {
            "kind": "voice",
            "language": language,
            "model_variant": model_variant,
            "name": name or path.stem,
            "file": file_info,
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

    def generate_audio_batch(
        self,
        texts: list[str],
        language: str,
        cancel_event: Event | None = None,
        model_variant: str | None = None,
        voice_id: str | None = None,
    ) -> list[bytes]:
        """Generate all phrases in as few remote jobs as the API limits allow."""
        normalized = [str(item).strip() for item in texts]
        if not normalized or any(not item for item in normalized):
            raise ValueError("O lote de áudio precisa conter textos não vazios.")

        batches = split_tts_batches(normalized)
        all_audio_segments: list[bytes] = []
        batch_endpoint_available = True
        for batch in batches:
            if cancel_event is not None and cancel_event.is_set():
                raise ColabAPIError("Geração cancelada pelo Joe.", cancelled=True)

            # Older Colab notebooks expose /v1/jobs/tts but not the newer
            # /v1/jobs/tts/batch route. A 404 is definitive, so switch to the
            # compatible one-job-per-phrase path for this and remaining batches.
            if not batch_endpoint_available:
                for text in batch:
                    if cancel_event is not None and cancel_event.is_set():
                        raise ColabAPIError("Geração cancelada pelo Joe.", cancelled=True)
                    all_audio_segments.append(
                        self.generate_audio(
                            text,
                            language,
                            cancel_event=cancel_event,
                            model_variant=model_variant,
                            voice_id=voice_id,
                        )
                    )
                continue

            payload: dict[str, Any] = {
                "texts": batch,
                "language": language,
                "model_variant": model_variant or language,
            }
            if voice_id:
                payload["voice_id"] = voice_id
            try:
                upload = self._request(
                    "POST", "/v1/jobs/tts/batch", json=payload
                ).json()
            except ColabAPIError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                batch_endpoint_available = False
                for text in batch:
                    if cancel_event is not None and cancel_event.is_set():
                        raise ColabAPIError("Geração cancelada pelo Joe.", cancelled=True)
                    all_audio_segments.append(
                        self.generate_audio(
                            text,
                            language,
                            cancel_event=cancel_event,
                            model_variant=model_variant,
                            voice_id=voice_id,
                        )
                    )
                continue

            job_id = str(upload.get("job_id", ""))
            if not job_id:
                raise ColabAPIError("O Colab não retornou o identificador do lote de áudio.")
            response = self._wait_for_result(job_id, cancel_event=cancel_event)
            try:
                with zipfile.ZipFile(io.BytesIO(response.content), mode="r") as archive:
                    names = set(archive.namelist())
                    batch_audio_segments: list[bytes] = []
                    for index in range(1, len(batch) + 1):
                        name = f"segment_{index:04d}.wav"
                        if name not in names:
                            raise ColabAPIError(
                                f"O pacote de áudio do Colab não contém o segmento {index}."
                            )
                        audio_data = archive.read(name)
                        if not audio_data.startswith(b"RIFF") or audio_data[8:12] != b"WAVE":
                            raise ColabAPIError(
                                f"O segmento {index} recebido do Colab não é um WAV RIFF/WAVE válido."
                            )
                        batch_audio_segments.append(audio_data)
            except zipfile.BadZipFile as exc:
                raise ColabAPIError(
                    "O Colab retornou um pacote de áudio em lote inválido."
                ) from exc
            all_audio_segments.extend(batch_audio_segments)
        return all_audio_segments

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
            # A Drive-prepared voice is identified by voice_id. Do not also
            # attach the original local file: the API intentionally rejects
            # multipart voice uploads in direct-Drive mode.
            if voice_path and not voice_id:
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

class ColabVoiceSession:
    """Prepare one reference voice in Drive and reuse its voice_id across TTS chunks."""

    def __init__(
        self,
        client: ColabAPIClient,
        voice_path: str | Path | None,
        language: str,
        model_variant: str,
        name: str | None = None,
    ) -> None:
        self.client = client
        self.voice_path = Path(voice_path).expanduser() if voice_path else None
        self.language = language
        self.model_variant = model_variant
        self.name = name
        self._prepared = False
        self._voice_id: str | None = None

    @property
    def voice_id(self) -> str | None:
        return self._voice_id

    def prepare(self) -> str | None:
        """Upload/process the reference audio at most once per shortcut generation."""
        if self._prepared:
            return self._voice_id
        if self.voice_path is None:
            self._prepared = True
            return None

        prepared = self.client.prepare_voice(
            self.voice_path,
            self.language,
            self.model_variant,
            name=self.name,
        )
        voice_id = str(prepared.get("voice_id") or "").strip()
        if not voice_id:
            raise ColabAPIError(
                "O Colab não retornou o identificador da voz preparada no Drive. "
                "Confira a preparação da voz no notebook."
            )
        self._voice_id = voice_id
        self._prepared = True
        return voice_id

    def generate_audio_batch(
        self,
        texts: list[str],
        cancel_event: Event | None = None,
    ) -> list[bytes]:
        """Generate shortcut phrases in one request using the prepared Drive voice."""
        self.prepare()
        return self.client.generate_audio_batch(
            texts,
            self.language,
            cancel_event=cancel_event,
            model_variant=self.model_variant,
            voice_id=self._voice_id,
        )

    def generate_audio(
        self,
        text: str,
        cancel_event: Event | None = None,
    ) -> bytes:
        """Generate one chunk using the already prepared Drive voice, never multipart."""
        self.prepare()
        return self.client.generate_audio(
            text,
            self.language,
            voice_path=self.voice_path,
            cancel_event=cancel_event,
            model_variant=self.model_variant,
            voice_id=self._voice_id,
        )
