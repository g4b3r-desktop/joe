from __future__ import annotations

import hmac
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import moondream as md
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

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
    with _tts_lock:
        model = _get_tts_model(language)
        key = (language, voice_source)
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
    global _asr_context, _asr_model
    if _asr_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _asr_context = md.photon(PARAKEET_MODEL_ID, device=device)
        _asr_model = _asr_context.__enter__()
    return _asr_model


def _transcribe(audio_path: str, timestamps: str) -> dict[str, Any]:
    with _asr_lock:
        speech = _get_asr_model()
        if timestamps in {"segment", "word"}:
            result = speech.transcribe(audio=audio_path, timestamps=timestamps)
        else:
            result = speech.transcribe(audio=audio_path)
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": result.get("segments", []),
        "backend": "Parakeet Redux / Photon",
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
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
        "asr": PARakeet_MODEL_ID,
        "asr_loaded": _asr_model is not None,
        "tts_loaded_languages": sorted(_tts_models.keys()),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
    }


@app.post("/v1/tts", dependencies=[Depends(require_token)])
async def text_to_speech(
    text: str = Form(...),
    language: str = Form("portuguese"),
    voice: UploadFile | None = File(default=None),
) -> Response:
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

    voice_path = None
    with tempfile.TemporaryDirectory(prefix="joe-tts-") as temp_dir:
        if voice is not None:
            voice_data = await _read_upload(voice)
            voice_path = str(Path(temp_dir) / f"voice{_safe_suffix(voice.filename)}")
            Path(voice_path).write_bytes(voice_data)
        try:
            wav_data = await run_in_threadpool(
                _generate_tts, text, language, voice_path
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Falha no Pocket TTS: {exc}") from exc

    return Response(
        content=wav_data,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="joe-generated.wav"'},
    )


@app.post("/v1/transcribe", dependencies=[Depends(require_token)])
async def speech_to_text(
    audio: UploadFile = File(...),
    timestamps: str = Form("segment"),
) -> dict[str, Any]:
    if timestamps not in {"none", "segment", "word"}:
        raise HTTPException(status_code=422, detail="timestamps deve ser none, segment ou word.")
    audio_data = await _read_upload(audio)

    with tempfile.TemporaryDirectory(prefix="joe-asr-") as temp_dir:
        audio_path = Path(temp_dir) / f"input{_safe_suffix(audio.filename)}"
        audio_path.write_bytes(audio_data)
        try:
            return await run_in_threadpool(_transcribe, str(audio_path), timestamps)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na transcrição Parakeet Redux: {exc}",
            ) from exc
