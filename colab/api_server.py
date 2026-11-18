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

API_TOKEN = os.environ.get("JOE_API_TOKEN", "")
if len(API_TOKEN) < 32:
    raise RuntimeError("Defina JOE_API_TOKEN com pelo menos 32 caracteres aleatórios.")

MAX_AUDIO_BYTES = 80 * 1024 * 1024
MAX_TEXT_CHARS = 20_000
PARAKEET_MODEL_ID = "moondream/parakeet-redux"
DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

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


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "audio.wav").suffix.lower()
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        return ".wav"
    return suffix


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="O arquivo de áudio está vazio.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Áudio excede o limite de 80 MB.")
    return data


def _get_tts_model(language: str) -> Any:
    model = _tts_models.get(language)
    if model is None:
        model = TTSModel.load_model(language=language, quantize=True)
        _tts_models[language] = model
    return model


def _generate_tts(text: str, language: str, voice_file: str | None) -> bytes:
    voice_source = voice_file or DEFAULT_VOICE
    voice_identity = (
        hashlib.sha256(Path(voice_file).read_bytes()).hexdigest()
        if voice_file else DEFAULT_VOICE
    )
    with _tts_lock:
        model = _get_tts_model(language)
        key = (language, voice_identity)
        voice_state = _tts_voice_states.get(key)
        if voice_state is None:
            voice_state = model.get_state_for_audio_prompt(voice_source)
            _tts_voice_states[key] = voice_state
        audio = model.generate_audio(voice_state, text)
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
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
    }


def _run_job(job_id: str, function: Any, args: tuple[Any, ...]) -> None:
    with _JOB_LOCK:
        _JOBS[job_id]["status"] = "running"
    try:
        result = function(*args)
    except Exception as exc:
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(exc)
    else:
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["result"] = result


def _submit_job(kind: str, function: Any, *args: Any) -> str:
    now = time.monotonic()
    with _JOB_LOCK:
        expired = [
            job_id for job_id, job in _JOBS.items()
            if job["status"] in {"completed", "failed"}
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
        active = sum(job["status"] in {"queued", "running"} for job in _JOBS.values())
        if active >= _MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="Fila de inferência cheia. Tente novamente mais tarde.")
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "kind": kind,
            "status": "queued",
            "created_at": now,
            "result": None,
            "error": None,
        }
    _JOB_EXECUTOR.submit(_run_job, job_id, function, args)
    return job_id


def _generate_tts_job(
    text: str, language: str, voice_data: bytes | None, voice_suffix: str
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="joe-tts-job-") as temp_dir:
        voice_path = None
        if voice_data is not None:
            voice_path = str(Path(temp_dir) / f"voice{voice_suffix}")
            Path(voice_path).write_bytes(voice_data)
        return _generate_tts(text, language, voice_path)


def _transcribe_job(audio_data: bytes, suffix: str, timestamps: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="joe-asr-job-") as temp_dir:
        audio_path = Path(temp_dir) / f"input{suffix}"
        audio_path.write_bytes(audio_data)
        return _transcribe(str(audio_path), timestamps)


@app.post("/v1/jobs/tts", status_code=202, dependencies=[Depends(require_token)])
async def submit_tts_job(
    text: str = Form(...),
    language: str = Form("portuguese"),
    voice: UploadFile | None = File(default=None),
) -> dict[str, str]:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="O texto não pode ficar vazio.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Texto excede o limite de 20.000 caracteres.")
    allowed_languages = {
        "english", "french", "german", "portuguese",
        "italian", "spanish", "dutch",
    }
    if language not in allowed_languages:
        raise HTTPException(status_code=422, detail="Idioma não suportado pelo Pocket TTS.")
    voice_data = await _read_upload(voice) if voice is not None else None
    suffix = _safe_suffix(voice.filename) if voice is not None else ".wav"
    job_id = _submit_job("tts", _generate_tts_job, text, language, voice_data, suffix)
    return {"job_id": job_id, "status": "queued"}


@app.post("/v1/jobs/transcribe", status_code=202, dependencies=[Depends(require_token)])
async def submit_transcription_job(
    audio: UploadFile = File(...),
    timestamps: str = Form("segment"),
) -> dict[str, str]:
    if timestamps not in {"none", "segment", "word"}:
        raise HTTPException(status_code=422, detail="timestamps deve ser none, segment ou word.")
    audio_data = await _read_upload(audio)
    suffix = _safe_suffix(audio.filename)
    job_id = _submit_job("transcription", _transcribe_job, audio_data, suffix, timestamps)
    return {"job_id": job_id, "status": "queued"}


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
        if job["status"] in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="A tarefa ainda está em andamento.")
        if job["status"] == "failed":
            message = job["error"] or "Falha desconhecida na inferência."
            _JOBS.pop(job_id, None)
            raise HTTPException(status_code=500, detail=message)
        kind = job["kind"]
        result = job["result"]
        _JOBS.pop(job_id, None)
    if kind == "tts":
        return Response(
            content=result,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="joe-generated.wav"'},
        )
    return result
