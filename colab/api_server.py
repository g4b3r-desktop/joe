from __future__ import annotations

import gc
import hmac
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import moondream as md
import soundfile as sf
import torch
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pocket_tts import TTSModel
from pocket_tts_compat import (
    SUPPORTED_TTS_MODEL_VARIANTS,
    default_voice_for_language,
    load_tts_model,
)
from safetensors_validation import validate_safetensors_file
from upload_file_utils import safe_upload_suffix
from drive_assets import (
    VOICE_AUDIO_SUFFIXES,
    atomic_write_json,
    config_for_uploaded_model,
    load_model_manifest,
    model_fingerprint,
    replace_directory_atomically,
    resolve_voice_state,
    safe_asset_name,
    sha256_file,
    validate_model_files,
)
from tts_runtime import (
    configure_torch_threads,
    ensure_supported_torch,
    inspect_audio_samples,
    resolve_tts_device,
)

API_TOKEN = os.environ.get("JOE_API_TOKEN", "")
if len(API_TOKEN) < 32:
    raise RuntimeError("Defina JOE_API_TOKEN com pelo menos 32 caracteres aleatórios.")

MAX_AUDIO_BYTES = 80 * 1024 * 1024
MAX_TEXT_CHARS = 20_000
PARAKEET_MODEL_ID = "moondream/parakeet-redux"
DEFAULT_VOICE = os.environ.get("JOE_TTS_DEFAULT_VOICE") or None
USE_DRIVE_MODELS = os.environ.get("JOE_TTS_USE_DRIVE_MODELS") == "1"
DIRECT_DRIVE_ASSETS = os.environ.get("JOE_TTS_DIRECT_DRIVE") == "1"
DRIVE_ASSETS_ROOT = Path(os.environ.get("JOE_TTS_DRIVE_ROOT", "/content/drive/MyDrive/JOE_TTS")).expanduser()
ASSET_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
MAX_MODEL_ASSET_BYTES = 4_000_000_000
MAX_VOICE_ASSET_BYTES = 80 * 1024 * 1024

# Pocket TTS remains CPU-first, but this Colab worker can explicitly move the
# nn.Module to CUDA. Auto prefers CUDA when available and safely falls back.
TTS_DEVICE_MODE = os.environ.get("JOE_TTS_DEVICE", "auto").strip().lower()
TTS_SAMPLER_ERROR: str | None = None
try:
    TTS_SAMPLER_DECODE_STEPS = int(
        os.environ.get("JOE_TTS_SAMPLER_DECODE_STEPS", "5")
    )
    if not 1 <= TTS_SAMPLER_DECODE_STEPS <= 16:
        raise ValueError("O número de passos do sampler deve ficar entre 1 e 16.")
except ValueError as exc:
    TTS_SAMPLER_ERROR = str(exc)
    TTS_SAMPLER_DECODE_STEPS = 5
TTS_DEVICE_ERROR: str | None = None
TTS_VERSION_ERROR: str | None = None
TTS_THREADS_ERROR: str | None = None
try:
    ensure_supported_torch(torch.__version__)
except RuntimeError as exc:
    TTS_VERSION_ERROR = str(exc)
try:
    TTS_CPU_THREADS = configure_torch_threads(
        torch, os.environ.get("JOE_TTS_CPU_THREADS", "1")
    )
except (TypeError, ValueError) as exc:
    TTS_THREADS_ERROR = str(exc)
    TTS_CPU_THREADS = configure_torch_threads(torch, 1)
try:
    TTS_DEVICE_SELECTED = resolve_tts_device(
        TTS_DEVICE_MODE, torch.cuda.is_available()
    )
except (RuntimeError, ValueError) as exc:
    TTS_DEVICE_ERROR = str(exc)
    TTS_DEVICE_SELECTED = "cpu"

app = FastAPI(
    title="Joe Colab Inference API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_tts_lock = threading.Lock()
_asr_lock = threading.Lock()
_tts_models: dict[str, Any] = {}
_tts_loaded_devices: dict[str, str] = {}
_tts_device_fallback_reason: str | None = None
_last_tts_metrics: dict[str, Any] | None = None
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


def _load_tts_model_for_inference(loader: Any, model_variant: str) -> Any:
    global _tts_device_fallback_reason

    if TTS_VERSION_ERROR:
        raise RuntimeError(TTS_VERSION_ERROR)
    if TTS_SAMPLER_ERROR:
        raise RuntimeError(TTS_SAMPLER_ERROR)
    if TTS_THREADS_ERROR:
        raise RuntimeError(TTS_THREADS_ERROR)
    if TTS_DEVICE_ERROR:
        raise RuntimeError(TTS_DEVICE_ERROR)

    model = loader()
    selected_device = TTS_DEVICE_SELECTED
    try:
        model.to(selected_device)
    except Exception as exc:
        if TTS_DEVICE_MODE != "auto" or selected_device != "cuda":
            raise RuntimeError(
                f"Não foi possível mover o Pocket TTS para {selected_device}: "
                f"{type(exc).__name__}: {exc}. Selecione CPU ou Automático."
            ) from exc
        # Auto is the only mode allowed to fall back after CUDA allocation fails.
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        model = loader()
        model.to("cpu")
        selected_device = "cpu"
        _tts_device_fallback_reason = (
            "CUDA estava disponível, mas o carregamento na GPU falhou; "
            f"o modelo foi colocado na CPU ({type(exc).__name__}: {exc})."
        )

    model.eval()
    actual_device = str(getattr(model, "device", selected_device))
    if "cuda" in actual_device.lower():
        actual_device = "cuda"
    else:
        actual_device = "cpu"
    _tts_loaded_devices[model_variant] = actual_device
    return model


def _get_tts_model(language: str, model_variant: str) -> Any:
    model = _tts_models.get(model_variant)
    if model is None:
        # The worker is sequential; retain only the selected variant in memory.
        _tts_voice_states.clear()
        _tts_models.clear()
        _tts_loaded_devices.clear()

        if DIRECT_DRIVE_ASSETS:
            drive_config = DRIVE_ASSETS_ROOT / "models" / model_variant / "config.yaml"
            if not drive_config.is_file():
                raise FileNotFoundError(
                    f"Modelo {model_variant!r} ainda não foi enviado ao Google Drive. "
                    "Use 'Enviar modelo local ao Google Drive' na aba Colab do Joe."
                )
            loader = lambda: TTSModel.load_model(
                config=str(drive_config),
                quantize=False,
                sampler_decode_steps=TTS_SAMPLER_DECODE_STEPS,
            )
        else:
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

            if selected_local_config:
                config_path = Path(selected_local_config).expanduser()
                loader = lambda: TTSModel.load_model(
                    config=str(config_path),
                    quantize=False,
                    sampler_decode_steps=TTS_SAMPLER_DECODE_STEPS,
                )
            else:
                loader = lambda: load_tts_model(
                    TTSModel,
                    language,
                    model_variant,
                    quantize=False,
                    sampler_decode_steps=TTS_SAMPLER_DECODE_STEPS,
                )

        model = _load_tts_model_for_inference(loader, model_variant)
        _tts_models[model_variant] = model
    return model

def _generate_tts(
    text: str,
    language: str,
    model_variant: str,
    voice_file: str | None,
    voice_name: str | None = None,
    voice_id: str | None = None,
) -> bytes:
    if voice_id:
        if not DIRECT_DRIVE_ASSETS:
            raise RuntimeError("IDs de voz do Google Drive exigem o modo de armazenamento direto.")
        manifest = load_model_manifest(DRIVE_ASSETS_ROOT, model_variant)
        state_path, voice_metadata = resolve_voice_state(
            DRIVE_ASSETS_ROOT,
            model_variant,
            voice_id,
            model_fingerprint(manifest),
        )
        voice_source = str(state_path)
        voice_name = voice_name or str(voice_metadata.get("name") or voice_id)
    else:
        voice_source = voice_file or default_voice_for_language(language, DEFAULT_VOICE)
    if not voice_source:
        raise RuntimeError(
            "Nenhuma voz foi selecionada. Escolha um áudio de referência no Joe; "
            "no modo Drive direto, o Colab o prepara e salva automaticamente."
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

        global _last_tts_metrics
        effective_device = _tts_loaded_devices.get(model_variant, TTS_DEVICE_SELECTED)
        if effective_device == "cuda":
            torch.cuda.synchronize()
        generation_started = time.perf_counter()
        try:
            # generate_audio() already uses torch.no_grad() in Pocket TTS.
            audio = model.generate_audio(voice_state, text)
            if effective_device == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                f"Falha na síntese do texto com o modelo '{model_variant}' "
                f"e a voz '{voice_label}' no dispositivo {effective_device}. "
                f"Detalhe: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed_seconds = max(time.perf_counter() - generation_started, 0.000001)
        samples = audio.detach().to("cpu").float().numpy()
        sample_rate = int(model.sample_rate)
        signal_metrics = inspect_audio_samples(samples, sample_rate)
        _last_tts_metrics = {
            **signal_metrics,
            "elapsed_seconds": elapsed_seconds,
            "realtime_factor": signal_metrics["duration_seconds"] / elapsed_seconds,
            "device": effective_device,
            "cpu_threads": TTS_CPU_THREADS,
            "language": language,
            "model_variant": model_variant,
            "voice": voice_label,
            "text_characters": len(text),
            "sample_rate": sample_rate,
            "torch_version": str(torch.__version__),
        }
        print(
            "[JOE_TTS] "
            f"device={effective_device} threads={TTS_CPU_THREADS} "
            f"variant={model_variant} audio={signal_metrics['duration_seconds']:.2f}s "
            f"elapsed={elapsed_seconds:.2f}s "
            f"RTF={_last_tts_metrics['realtime_factor']:.2f} "
            f"peak={signal_metrics['peak']:.5f} rms={signal_metrics['rms']:.5f}"
        )

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


def _require_direct_drive() -> Path:
    if not DIRECT_DRIVE_ASSETS:
        raise HTTPException(
            status_code=409,
            detail="O notebook não habilitou o armazenamento direto no Google Drive. "
            "Reinicie pelo notebook novo do Joe.",
        )
    if not DRIVE_ASSETS_ROOT.is_dir():
        raise HTTPException(
            status_code=503,
            detail="Google Drive não está montado. Reinicie a célula de configuração do Drive.",
        )
    return DRIVE_ASSETS_ROOT


def _validate_asset_variant(language: str, model_variant: str) -> None:
    if language not in SUPPORTED_TTS_MODEL_VARIANTS:
        raise HTTPException(status_code=422, detail="Idioma TTS não suportado.")
    if model_variant not in SUPPORTED_TTS_MODEL_VARIANTS[language]:
        raise HTTPException(
            status_code=422,
            detail="A variante selecionada não pertence ao idioma informado.",
        )


def _upload_session_path(root: Path, upload_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise HTTPException(status_code=400, detail="Identificador de upload inválido.")
    path = root / ".uploads" / upload_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Sessão de upload não encontrada ou expirada.")
    return path


def _assemble_uploaded_file(
    session_dir: Path, key: str, metadata: dict[str, Any], destination: Path
) -> None:
    size = int(metadata["size"])
    expected_hash = str(metadata["sha256"])
    chunk_count = math.ceil(size / ASSET_UPLOAD_CHUNK_BYTES)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with destination.open("wb") as output:
        for index in range(chunk_count):
            part = session_dir / f"{key}.{index:06d}.part"
            if not part.is_file():
                raise HTTPException(
                    status_code=409,
                    detail=f"Upload incompleto: falta o bloco {index + 1}/{chunk_count} do arquivo {key}.",
                )
            with part.open("rb") as source:
                while True:
                    data = source.read(1024 * 1024)
                    if not data:
                        break
                    output.write(data)
                    digest.update(data)
                    written += len(data)
    if written != size or not hmac.compare_digest(digest.hexdigest(), expected_hash):
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Arquivo {key} incompleto ou com SHA-256 diferente do informado.",
        )


@app.post("/v1/assets/uploads", dependencies=[Depends(require_token)])
def start_asset_upload(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    root = _require_direct_drive()
    kind = str(payload.get("kind", "")).strip()
    language = str(payload.get("language", "")).strip()
    model_variant = str(payload.get("model_variant", "")).strip()
    _validate_asset_variant(language, model_variant)

    if kind == "model":
        if model_variant == "japanese":
            raise HTTPException(
                status_code=422,
                detail="O upload de modelos comunitários japoneses ainda não é suportado por este instalador.",
            )
        files = payload.get("files")
        if not isinstance(files, dict) or set(files) != {"checkpoint", "tokenizer"}:
            raise HTTPException(
                status_code=422,
                detail="Envie exatamente os arquivos checkpoint (model.safetensors) e tokenizer (tokenizer.json).",
            )
        total_size = 0
        normalized_files: dict[str, dict[str, Any]] = {}
        for key, maximum in (
            ("checkpoint", MAX_MODEL_ASSET_BYTES),
            ("tokenizer", 100 * 1024 * 1024),
        ):
            item = files[key]
            if not isinstance(item, dict):
                raise HTTPException(status_code=422, detail=f"Metadados inválidos para {key}.")
            size = item.get("size")
            checksum = str(item.get("sha256", "")).lower()
            if not isinstance(size, int) or size <= 0 or size > maximum:
                raise HTTPException(status_code=413, detail=f"Tamanho inválido ou acima do limite para {key}.")
            if not re.fullmatch(r"[a-f0-9]{64}", checksum):
                raise HTTPException(status_code=422, detail=f"SHA-256 inválido para {key}.")
            normalized_files[key] = {
                "size": size,
                "sha256": checksum,
                "name": "model.safetensors" if key == "checkpoint" else "tokenizer.json",
            }
            total_size += size
        if total_size > MAX_MODEL_ASSET_BYTES:
            raise HTTPException(status_code=413, detail="O conjunto do modelo excede o limite de 4 GB.")
        metadata = {
            "kind": kind,
            "language": language,
            "model_variant": model_variant,
            "files": normalized_files,
        }
    elif kind == "voice":
        item = payload.get("file")
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="Metadados do áudio de referência ausentes.")
        size = item.get("size")
        checksum = str(item.get("sha256", "")).lower()
        filename = Path(str(item.get("name", "reference.wav"))).name
        suffix = Path(filename).suffix.lower()
        if suffix not in VOICE_AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Formato de referência não suportado. Use WAV, FLAC, MP3 ou OGG; não envie .safetensors.",
            )
        if not isinstance(size, int) or size <= 0 or size > MAX_VOICE_ASSET_BYTES:
            raise HTTPException(status_code=413, detail="Áudio de referência vazio ou acima de 80 MB.")
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise HTTPException(status_code=422, detail="SHA-256 inválido para o áudio.")
        metadata = {
            "kind": kind,
            "language": language,
            "model_variant": model_variant,
            "name": safe_asset_name(str(payload.get("name") or Path(filename).stem)),
            "file": {"size": size, "sha256": checksum, "name": filename, "suffix": suffix},
        }
        try:
            load_model_manifest(root, model_variant)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"O modelo '{model_variant}' não está instalado no Google Drive "
                    "ou não possui manifest.json. Use 'Enviar modelo local ao Google Drive' "
                    "na aba Google Colab / API do Joe e tente novamente."
                ),
            ) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Não foi possível validar o manifesto do modelo '{model_variant}' "
                    f"no Google Drive ({type(exc).__name__}: {exc}). "
                    "Confira se a pasta JOE_TTS/models/<variante> contém manifest.json "
                    "válido; se necessário, envie novamente o modelo pelo Joe."
                ),
            ) from exc
    else:
        raise HTTPException(status_code=422, detail="Tipo de upload desconhecido.")

    upload_id = uuid.uuid4().hex
    session_dir = root / ".uploads" / upload_id
    try:
        session_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(session_dir / "upload.json", metadata)
    except OSError as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(
            status_code=503,
            detail=(
                "O Google Drive não permitiu criar a sessão temporária de upload "
                f"em JOE_TTS/.uploads ({type(exc).__name__}: {exc}). "
                "Verifique se a célula de montagem do Drive continua ativa e se a pasta "
                "JOE_TTS permite gravação."
            ),
        ) from exc
    return {
        "upload_id": upload_id,
        "chunk_bytes": ASSET_UPLOAD_CHUNK_BYTES,
        "status": "receiving",
    }


@app.put(
    "/v1/assets/uploads/{upload_id}/{asset_key}/{chunk_index}",
    dependencies=[Depends(require_token)],
)
async def receive_asset_chunk(
    upload_id: str,
    asset_key: str,
    chunk_index: int,
    chunk: UploadFile = File(...),
) -> dict[str, Any]:
    root = _require_direct_drive()
    session_dir = _upload_session_path(root, upload_id)
    metadata = json.loads((session_dir / "upload.json").read_text(encoding="utf-8"))
    files = metadata.get("files", {}) if metadata.get("kind") == "model" else {
        "voice": metadata.get("file", {})
    }
    if asset_key not in files:
        raise HTTPException(status_code=404, detail="Arquivo não pertence a esta sessão de upload.")
    file_info = files[asset_key]
    size = int(file_info["size"])
    chunk_count = math.ceil(size / ASSET_UPLOAD_CHUNK_BYTES)
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise HTTPException(status_code=416, detail="Índice de bloco fora do intervalo.")
    data = await chunk.read(ASSET_UPLOAD_CHUNK_BYTES + 1)
    expected_size = min(ASSET_UPLOAD_CHUNK_BYTES, size - chunk_index * ASSET_UPLOAD_CHUNK_BYTES)
    if len(data) != expected_size:
        raise HTTPException(
            status_code=422,
            detail=f"Bloco com tamanho incorreto: esperado {expected_size} bytes, recebido {len(data)}.",
        )
    destination = session_dir / f"{asset_key}.{chunk_index:06d}.part"
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return {
        "upload_id": upload_id,
        "file": asset_key,
        "chunk_index": chunk_index,
        "received_bytes": len(data),
        "total_chunks": chunk_count,
    }


@app.post("/v1/assets/uploads/{upload_id}/commit", dependencies=[Depends(require_token)])
def commit_asset_upload(upload_id: str) -> dict[str, Any]:
    root = _require_direct_drive()
    session_dir = _upload_session_path(root, upload_id)
    metadata = json.loads((session_dir / "upload.json").read_text(encoding="utf-8"))
    kind = metadata["kind"]
    language = metadata["language"]
    model_variant = metadata["model_variant"]
    _validate_asset_variant(language, model_variant)

    if kind == "model":
        staging = root / ".staging" / f"model-{upload_id}"
        destination = root / "models" / model_variant
        staging.mkdir(parents=True, exist_ok=False)
        try:
            checkpoint = staging / "model.safetensors"
            tokenizer = staging / "tokenizer.json"
            _assemble_uploaded_file(session_dir, "checkpoint", metadata["files"]["checkpoint"], checkpoint)
            _assemble_uploaded_file(session_dir, "tokenizer", metadata["files"]["tokenizer"], tokenizer)
            validation = validate_model_files(checkpoint, tokenizer)
            ensure_supported_torch(torch.__version__)
            from pocket_tts.utils.config import CONFIGS_DIR
            import yaml

            # Load the exact staged checkpoint and tokenizer through Pocket TTS.
            # This checks the actual model architecture/state-dict compatibility,
            # not only whether the files are syntactically valid.
            validation_data = config_for_uploaded_model(
                model_variant, staging, CONFIGS_DIR
            )
            validation_config = staging / "validation-config.yaml"
            validation_config.write_text(
                yaml.safe_dump(validation_data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            validation_model = TTSModel.load_model(
                config=str(validation_config),
                quantize=False,
                sampler_decode_steps=TTS_SAMPLER_DECODE_STEPS,
            )
            if not getattr(validation_model, "has_voice_cloning", True):
                raise ValueError(
                    "O checkpoint foi carregado, mas não contém os pesos de clonagem de voz."
                )
            del validation_model
            gc.collect()
            validation_config.unlink(missing_ok=True)

            # The installed YAML must reference the final Drive directory, not staging.
            config_data = config_for_uploaded_model(model_variant, destination, CONFIGS_DIR)
            config_path = staging / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            model_version = importlib.metadata.version("pocket-tts")
            manifest = {
                "schema": 1,
                "language": language,
                "model_variant": model_variant,
                "checkpoint_sha256": metadata["files"]["checkpoint"]["sha256"],
                "tokenizer_sha256": metadata["files"]["tokenizer"]["sha256"],
                "checkpoint_size": validation["checkpoint_size"],
                "tokenizer_size": validation["tokenizer_size"],
                "tensor_count": validation["tensor_count"],
                "pocket_tts_version": model_version,
                "torch_version": str(torch.__version__),
                "validated_by_pocket_tts_loader": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(staging / "manifest.json", manifest)
            replace_directory_atomically(staging, destination)
            with _tts_lock:
                _tts_voice_states.clear()
                _tts_models.clear()
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)
        return {
            "kind": "model",
            "language": language,
            "model_variant": model_variant,
            "manifest": manifest,
            "installed_path": str(destination),
        }

    if kind == "voice":
        staging = root / ".staging" / f"voice-{upload_id}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            audio_name = metadata["file"]["name"]
            audio_path = staging / f"reference{metadata['file']['suffix']}"
            _assemble_uploaded_file(session_dir, "voice", metadata["file"], audio_path)
            model_manifest = load_model_manifest(root, model_variant)
            fingerprint = model_fingerprint(model_manifest)
            voice_id = (
                f"{metadata['name']}-{metadata['file']['sha256'][:12]}-{fingerprint[:8]}"
            )
            destination = root / "voices" / model_variant / voice_id
            existing_metadata = destination / "metadata.json"
            if existing_metadata.is_file():
                existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
                if (
                    existing.get("source_sha256") == metadata["file"]["sha256"]
                    and existing.get("model_fingerprint") == fingerprint
                    and (destination / "voice_state.safetensors").is_file()
                ):
                    shutil.rmtree(staging, ignore_errors=True)
                    return {
                        "kind": "voice",
                        "voice_id": voice_id,
                        "name": existing.get("name", metadata["name"]),
                        "model_variant": model_variant,
                        "reused": True,
                        "metadata": existing,
                    }

            with _tts_lock:
                model = _get_tts_model(language, model_variant)
                if not getattr(model, "has_voice_cloning", True):
                    raise RuntimeError(
                        f"O modelo {model_variant!r} carregado não tem pesos de clonagem de voz."
                    )
                voice_state = model.get_state_for_audio_prompt(str(audio_path), truncate=True)
                from pocket_tts import export_model_state
                cpu_voice_state = voice_state_for_safetensors_export(voice_state)
                export_model_state(cpu_voice_state, staging / "voice_state.safetensors")
                del cpu_voice_state

            from safetensors_validation import validate_safetensors_file
            validate_safetensors_file(staging / "voice_state.safetensors")
            voice_metadata = {
                "schema": 1,
                "voice_id": voice_id,
                "name": metadata["name"],
                "language": language,
                "model_variant": model_variant,
                "model_fingerprint": fingerprint,
                "source_name": audio_name,
                "source_sha256": metadata["file"]["sha256"],
                "voice_state_sha256": sha256_file(staging / "voice_state.safetensors"),
                "pocket_tts_version": model_manifest["pocket_tts_version"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(staging / "metadata.json", voice_metadata)
            replace_directory_atomically(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)
        return {
            "kind": "voice",
            "voice_id": voice_id,
            "name": metadata["name"],
            "model_variant": model_variant,
            "reused": False,
            "metadata": voice_metadata,
        }

    shutil.rmtree(session_dir, ignore_errors=True)
    raise HTTPException(status_code=422, detail="Tipo de upload desconhecido.")


@app.get("/v1/assets", dependencies=[Depends(require_token)])
def list_drive_assets() -> dict[str, Any]:
    root = _require_direct_drive()
    models = []
    voices = []
    models_root = root / "models"
    if models_root.is_dir():
        for manifest_path in models_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["model_fingerprint"] = model_fingerprint(manifest)
                models.append(manifest)
            except (OSError, ValueError):
                continue
    voices_root = root / "voices"
    if voices_root.is_dir():
        for metadata_path in voices_root.glob("*/*/metadata.json"):
            try:
                voice_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                state_path = metadata_path.parent / "voice_state.safetensors"
                expected_hash = str(voice_metadata.get("voice_state_sha256", ""))
                voice_metadata["state_valid"] = bool(
                    state_path.is_file()
                    and expected_hash
                    and hmac.compare_digest(sha256_file(state_path), expected_hash)
                )
                voices.append(voice_metadata)
            except (OSError, ValueError):
                continue
    return {"root": str(root), "models": models, "voices": voices}


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
        "direct_drive_assets_enabled": DIRECT_DRIVE_ASSETS and DRIVE_ASSETS_ROOT.is_dir(),
        "drive_assets_root": str(DRIVE_ASSETS_ROOT) if DIRECT_DRIVE_ASSETS else None,
        "tts_direct_drive_models": sorted(
            path.parent.name for path in (DRIVE_ASSETS_ROOT / "models").glob("*/manifest.json")
        ) if DIRECT_DRIVE_ASSETS and (DRIVE_ASSETS_ROOT / "models").is_dir() else [],
        "tts_direct_drive_voice_count": len(
            list((DRIVE_ASSETS_ROOT / "voices").glob("*/*/metadata.json"))
        ) if DIRECT_DRIVE_ASSETS and (DRIVE_ASSETS_ROOT / "voices").is_dir() else 0,
        "tts_default_voice_local": bool(DEFAULT_VOICE and Path(DEFAULT_VOICE).is_file()),
        "hf_cache_configured": bool(os.environ.get("HF_HOME")),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "torch_version": str(torch.__version__),
        "tts_device_mode": TTS_DEVICE_MODE,
        "tts_device_selected": TTS_DEVICE_SELECTED,
        "tts_loaded_devices": dict(_tts_loaded_devices),
        "tts_cpu_threads": TTS_CPU_THREADS,
        "tts_sampler_decode_steps": TTS_SAMPLER_DECODE_STEPS,
        "tts_sampler_error": TTS_SAMPLER_ERROR,
        "tts_device_error": TTS_DEVICE_ERROR,
        "tts_runtime_version_error": TTS_VERSION_ERROR,
        "tts_threads_error": TTS_THREADS_ERROR,
        "tts_device_fallback_reason": _tts_device_fallback_reason,
        "last_tts_generation": dict(_last_tts_metrics) if _last_tts_metrics else None,
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
    voice_id: str | None,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="joe-tts-job-") as temp_dir:
        voice_path = None
        if voice_data is not None:
            voice_path = str(Path(temp_dir) / f"voice{voice_suffix}")
            Path(voice_path).write_bytes(voice_data)
        return _generate_tts(
            text,
            language,
            model_variant,
            voice_path,
            voice_name=voice_filename,
            voice_id=voice_id,
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
    voice_id: str | None = Form(default=None),
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
    if voice is not None and voice_id:
        raise HTTPException(status_code=422, detail="Envie o arquivo de voz ou voice_id, não ambos.")
    if DIRECT_DRIVE_ASSETS and voice is not None:
        raise HTTPException(
            status_code=409,
            detail="O modo Drive direto exige preparar o áudio original no Colab. "
            "Use o Joe atualizado, que salva o áudio no Drive antes da geração.",
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
        voice_id,
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
