from __future__ import annotations

import hmac
import hashlib
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import moondream as md
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pocket_tts import TTSModel
from pocket_tts_compat import (
    SUPPORTED_TTS_MODEL_VARIANTS,
    default_voice_for_language,
    load_tts_model,
)
from safetensors_validation import validate_safetensors_file
from upload_file_utils import safe_upload_suffix
import os, torch
torch.set_num_threads(os.cpu_count() or 2)

API_TOKEN = os.environ.get("JOE_API_TOKEN", "")
if len(API_TOKEN) < 32:
    raise RuntimeError("Defina JOE_API_TOKEN com pelo menos 32 caracteres aleatórios.")

MAX_AUDIO_BYTES = 80 * 1024 * 1024
MAX_TEXT_CHARS = 20_000
PARAKEET_MODEL_ID = "moondream/parakeet-redux"
DEFAULT_VOICE = os.environ.get("JOE_TTS_DEFAULT_VOICE") or None
USE_DRIVE_MODELS = os.environ.get("JOE_TTS_USE_DRIVE_MODELS") == "1"

app = FastAPI(
    title="Joe Colab Inference API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_tts_lock = threading.Lock()
_asr_lock = threading.Lock()
_tts_models: dict[str, Any] = {}
_tts_voice_states: dict[tuple[str, str], Any] = {}
_asr_context: Any = None
_asr_model: Any = None
_asr_processor: Any = None
_asr_backend = "não carregado"
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="joe-inference")
_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_MAX_ACTIVE_JOBS = 2
_MAX_STORED_JOBS = 8
_JOB_TTL_SECONDS = 1800


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token obrigatório.")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido.")



async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="O arquivo de áudio está vazio.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Áudio excede o limite de 80 MB.")
    return data


def _get_tts_model(language: str, model_variant: str) -> Any:
    model = _tts_models.get(model_variant)
    if model is None:
        # The Colab worker is sequential; retain only the selected variant in memory.
        _tts_voice_states.clear()
        _tts_models.clear()
        local_config = os.environ.get("JOE_TTS_CONFIG_PATH")
        local_24l_config = os.environ.get("JOE_TTS_24L_CONFIG_PATH")
        selected_local_config = None
        if language == "portuguese" and model_variant == "portuguese":
            selected_local_config = local_config
        elif language == "portuguese" and model_variant == "portuguese_24l":
            selected_local_config = local_24l_config

        if language != "japanese" and USE_DRIVE_MODELS:
            if selected_local_config:
                if not Path(selected_local_config).expanduser().is_file():
                    raise FileNotFoundError(
                        "Configuração local do Pocket TTS não encontrada: "
                        f"{selected_local_config}"
                    )
            elif language == "portuguese":
                raise RuntimeError(
                    "O Pocket TTS está configurado para usar modelos do Google Drive, "
                    f"mas a configuração da variante {model_variant!r} não foi aplicada."
                )
            elif not local_config or not Path(local_config).expanduser().is_file():
                raise RuntimeError(
                    "O Pocket TTS está configurado para usar modelos do Google Drive, "
                    "mas a configuração local não foi aplicada."
                )

        # Each Portuguese variant uses its own YAML and its matching checkpoint,
        # tokenizer and voice-state architecture. Japanese remains a community
        # model loaded from its published YAML; other variants use built-in IDs.
        if selected_local_config:
            config_path = Path(selected_local_config).expanduser()
            model = TTSModel.load_model(config=str(config_path), quantize=False)
        else:
            model = load_tts_model(
                TTSModel, language, model_variant, quantize=False
            )
        _tts_models[model_variant] = model
    return model


def _generate_tts(
    text: str,
    language: str,
    model_variant: str,
    voice_file: str | None,
    voice_name: str | None = None,
) -> bytes:
    voice_source = voice_file or default_voice_for_language(language, DEFAULT_VOICE)
    if not voice_source:
        raise RuntimeError(
            "Nenhuma voz de referência local foi configurada. "
            "Envie um WAV pelo Joe ou configure um WAV no Google Drive."
        )
    if isinstance(voice_source, str) and not voice_source.startswith(("hf://", "https://")):
        voice_path = Path(voice_source).expanduser()
        if not voice_path.is_file():
            raise FileNotFoundError(f"Arquivo de voz local não encontrado: {voice_path}")
        voice_source = str(voice_path)

    if voice_name:
        voice_label = voice_name.replace("\\", "/").rsplit("/", 1)[-1]
    elif isinstance(voice_source, str) and not voice_source.startswith(("hf://", "https://")):
        voice_label = Path(voice_source).name
    else:
        voice_label = "voz predefinida"

    voice_identity = (
        hashlib.sha256(Path(voice_source).read_bytes()).hexdigest()
        if isinstance(voice_source, str) and Path(voice_source).is_file()
        else str(voice_source)
    )
    with _tts_lock:
        try:
            model = _get_tts_model(language, model_variant)
        except Exception as exc:
            raise RuntimeError(
                "Falha ao carregar o checkpoint/configuração do modelo Pocket TTS "
                f"'{model_variant}'. Confira os pesos model.safetensors e o YAML "
                f"dessa variante. Detalhe: {type(exc).__name__}: {exc}"
            ) from exc

        key = (model_variant, voice_identity)
        voice_state = _tts_voice_states.get(key)
        if voice_state is None:
            is_local_file = (
                isinstance(voice_source, str)
                and not voice_source.startswith(("hf://", "https://"))
                and Path(voice_source).is_file()
            )
            is_safetensors = (
                is_local_file and Path(voice_source).suffix.lower() == ".safetensors"
            )
            if is_safetensors:
                try:
                    validate_safetensors_file(voice_source)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"O arquivo de voz '{voice_label}' não é um estado "
                        f"Safetensors válido: {exc}"
                    ) from exc
            try:
                voice_state = model.get_state_for_audio_prompt(voice_source)
            except Exception as exc:
                voice_kind = (
                    "estado .safetensors"
                    if is_safetensors
                    else "áudio/URL de referência"
                )
                raise RuntimeError(
                    f"Falha ao carregar a voz '{voice_label}' ({voice_kind}) "
                    f"com o modelo '{model_variant}'. Para estados preparados, "
                    "confirme que a voz foi exportada usando a mesma variante. "
                    f"Detalhe: {type(exc).__name__}: {exc}"
                ) from exc
            _tts_voice_states[key] = voice_state

        try:
            audio = model.generate_audio(voice_state, text)
        except Exception as exc:
            raise RuntimeError(
                f"Falha na síntese do texto com o modelo '{model_variant}' "
                f"e a voz '{voice_label}'. Detalhe: {type(exc).__name__}: {exc}"
            ) from exc
        samples = audio.detach().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav") as output:
        sf.write(output.name, samples, int(model.sample_rate), subtype="PCM_16")
        output.seek(0)
        return Path(output.name).read_bytes()


def _get_asr_model() -> Any:
    global _asr_context, _asr_model, _asr_processor, _asr_backend
    if _asr_model is not None:
        return _asr_model

    if torch.cuda.is_available():
        _asr_context = md.photon(PARAKEET_MODEL_ID, device="cuda")
        _asr_model = _asr_context.__enter__()
        _asr_backend = f"Parakeet Redux / Photon CUDA ({torch.cuda.get_device_name(0)})"
        return _asr_model

    # Compatibility fallback when this Colab runtime has no GPU.
    from transformers import AutoModel, AutoProcessor

    _asr_processor = AutoProcessor.from_pretrained(
        PARAKEET_MODEL_ID,
        trust_remote_code=True,
    )
    _asr_model = AutoModel.from_pretrained(
        PARAKEET_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to("cpu").eval()
    _asr_backend = "Parakeet Redux / Transformers CPU"
    return _asr_model


def _transcribe(audio_path: str, timestamps: str) -> dict[str, Any]:
    with _asr_lock:
        model = _get_asr_model()
        if _asr_backend.startswith("Parakeet Redux / Photon"):
            if timestamps in {"segment", "word"}:
                result = model.transcribe(audio=audio_path, timestamps=timestamps)
            else:
                result = model.transcribe(audio=audio_path)
        else:
            import numpy as np
            import soundfile as audio_io
            from math import gcd
            from scipy.signal import resample_poly

            waveform, sample_rate = audio_io.read(
                audio_path, dtype="float32", always_2d=False
            )
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            waveform = np.asarray(waveform, dtype=np.float32)
            if sample_rate != 16_000:
                divisor = gcd(int(sample_rate), 16_000)
                waveform = resample_poly(
                    waveform,
                    16_000 // divisor,
                    int(sample_rate) // divisor,
                ).astype(np.float32)
            result = model.transcribe(waveform, _asr_processor)

    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    if not isinstance(result, dict):
        result = {"text": str(result), "segments": []}
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": result.get("segments", []),
        "backend": _asr_backend,
        "device": "CUDA" if "CUDA" in _asr_backend else "CPU",
    }

@app.get("/health")
def health(_: None = Depends(require_token)) -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA disponível"
    return {
        "ok": True,
        "service": "joe-colab-api",
        "tts": "kyutai/pocket-tts",
        "asr": PARAKEET_MODEL_ID,
        "asr_loaded": _asr_model is not None,
        "asr_backend": _asr_backend,
        "tts_loaded_languages": sorted(_tts_models.keys()),
        "tts_loaded_model_variants": sorted(_tts_models.keys()),
        "tts_local_drive_config": bool(os.environ.get("JOE_TTS_CONFIG_PATH")),
        "tts_local_drive_24l_config": bool(os.environ.get("JOE_TTS_24L_CONFIG_PATH")),
        "tts_drive_models_required": USE_DRIVE_MODELS,
        "tts_default_voice_local": bool(DEFAULT_VOICE and Path(DEFAULT_VOICE).is_file()),
        "hf_cache_configured": bool(os.environ.get("HF_HOME")),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
    }


def _run_job(job_id: str, function: Any, args: tuple[Any, ...]) -> None:
    with _JOB_LOCK:
        job = _JOBS[job_id]
        if job["cancel_requested"]:
            job["status"] = "cancelled"
            return
        job["status"] = "running"
    try:
        result = function(*args)
    except Exception as exc:
        with _JOB_LOCK:
            job = _JOBS[job_id]
            if job["cancel_requested"]:
                job["status"] = "cancelled"
                job["result"] = None
                job["error"] = None
            else:
                job["status"] = "failed"
                job["error"] = str(exc)
    else:
        with _JOB_LOCK:
            job = _JOBS[job_id]
            if job["cancel_requested"]:
                job["status"] = "cancelled"
                job["result"] = None
                job["error"] = None
            else:
                job["status"] = "completed"
                job["result"] = result


def _submit_job(kind: str, function: Any, *args: Any) -> str:
    now = time.monotonic()
    with _JOB_LOCK:
        expired = [
            job_id for job_id, job in _JOBS.items()
            if job["status"] in {"completed", "failed", "cancelled"}
            and now - job["created_at"] > _JOB_TTL_SECONDS
        ]
        for job_id in expired:
            _JOBS.pop(job_id, None)
        while len(_JOBS) >= _MAX_STORED_JOBS:
            completed = [
                (job["created_at"], job_id)
                for job_id, job in _JOBS.items()
                if job["status"] in {"completed", "failed"}
            ]
            if not completed:
                raise HTTPException(status_code=429, detail="Armazenamento de tarefas cheio.")
            _, oldest_job_id = min(completed)
            _JOBS.pop(oldest_job_id, None)
        active = sum(
            job["status"] in {"queued", "running", "cancelling"}
            for job in _JOBS.values()
        )
        if active >= _MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="Fila de inferência cheia. Tente novamente mais tarde.")
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "kind": kind,
            "status": "queued",
            "created_at": now,
            "result": None,
            "error": None,
            "cancel_requested": False,
            "future": None,
        }
    future = _JOB_EXECUTOR.submit(_run_job, job_id, function, args)
    with _JOB_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["future"] = future
    return job_id


def _generate_tts_job(
    text: str,
    language: str,
    model_variant: str,
    voice_data: bytes | None,
    voice_suffix: str,
    voice_filename: str | None,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="joe-tts-job-") as temp_dir:
        voice_path = None
        if voice_data is not None:
            voice_path = str(Path(temp_dir) / f"voice{voice_suffix}")
            Path(voice_path).write_bytes(voice_data)
        return _generate_tts(
            text, language, model_variant, voice_path, voice_name=voice_filename
        )


def _transcribe_job(audio_data: bytes, suffix: str, timestamps: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="joe-asr-job-") as temp_dir:
        audio_path = Path(temp_dir) / f"input{suffix}"
        audio_path.write_bytes(audio_data)
        return _transcribe(str(audio_path), timestamps)


@app.post("/v1/jobs/tts", status_code=202, dependencies=[Depends(require_token)])
async def submit_tts_job(
    text: str = Form(...),
    language: str = Form("portuguese"),
    model_variant: str | None = Form(default=None),
    voice: UploadFile | None = File(default=None),
) -> dict[str, str]:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="O texto não pode ficar vazio.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Texto excede o limite de 20.000 caracteres.")
    if language not in SUPPORTED_TTS_MODEL_VARIANTS:
        raise HTTPException(status_code=422, detail="Idioma não suportado pelo Pocket TTS.")
    selected_model = (model_variant or language).strip()
    if selected_model not in SUPPORTED_TTS_MODEL_VARIANTS[language]:
        raise HTTPException(
            status_code=422,
            detail="A versão selecionada não está disponível para o idioma informado.",
        )
    voice_data = await _read_upload(voice) if voice is not None else None
    suffix = safe_upload_suffix(voice.filename, allow_safetensors=True) if voice is not None else ".wav"
    voice_filename = (
        (voice.filename or f"voice{suffix}").replace("\\", "/").rsplit("/", 1)[-1]
        if voice is not None else None
    )
    job_id = _submit_job(
        "tts",
        _generate_tts_job,
        text,
        language,
        selected_model,
        voice_data,
        suffix,
        voice_filename,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/v1/jobs/transcribe", status_code=202, dependencies=[Depends(require_token)])
async def submit_transcription_job(
    audio: UploadFile = File(...),
    timestamps: str = Form("segment"),
) -> dict[str, str]:
    if timestamps not in {"none", "segment", "word"}:
        raise HTTPException(status_code=422, detail="timestamps deve ser none, segment ou word.")
    audio_data = await _read_upload(audio)
    suffix = safe_upload_suffix(audio.filename)
    job_id = _submit_job("transcription", _transcribe_job, audio_data, suffix, timestamps)
    return {"job_id": job_id, "status": "queued"}


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
def cancel_job(job_id: str) -> dict[str, str]:
    """Cancel queued work or discard the result after an active inference returns."""
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada ou expirada.")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return {"job_id": job_id, "status": job["status"]}
        job["cancel_requested"] = True
        future = job.get("future")
        if future is not None and future.cancel():
            job["status"] = "cancelled"
        else:
            job["status"] = "cancelling"
        return {"job_id": job_id, "status": job["status"]}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job_status(job_id: str) -> dict[str, Any]:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada ou expirada.")
        return {
            "job_id": job_id,
            "kind": job["kind"],
            "status": job["status"],
            "error": job["error"],
        }


@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(require_token)])
def get_job_result(job_id: str) -> Any:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada ou expirada.")
        if job["status"] in {"queued", "running", "cancelling"}:
            raise HTTPException(status_code=409, detail="A tarefa ainda está em andamento.")
        if job["status"] == "cancelled":
            raise HTTPException(status_code=410, detail="A tarefa foi cancelada.")
        if job["status"] == "failed":
            message = job["error"] or "Falha desconhecida na inferência."
            raise HTTPException(status_code=500, detail=message)
        kind = job["kind"]
        result = job["result"]
        # Keep completed results available until TTL/eviction so the desktop can
        # retry a download after a transient tunnel/network interruption.
    if kind == "tts":
        return Response(
            content=result,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="joe-generated.wav"'},
        )
    return result
