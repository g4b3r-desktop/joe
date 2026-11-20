from __future__ import annotations

import tempfile
from pathlib import Path

from joe.tts_quality import generate_audio_with_quality

DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"


class PocketTTSService:
    """Lazy, reusable adapter around Kyutai Pocket TTS."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._voice_states: dict[tuple[str, str], object] = {}

    def generate(
        self,
        text: str,
        language: str,
        output_path: str | Path,
        voice_path: str | Path | None = None,
    ) -> Path:
        if not text.strip():
            raise ValueError("Digite algum texto antes de gerar o áudio.")

        try:
            import soundfile as sf
            import torch
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Dependências do Pocket TTS ausentes. Execute ./scripts/install-linux.sh."
            ) from exc

        # Keep inference bounded on older desktop CPUs.
        try:
            torch.set_num_threads(2)
        except RuntimeError:
            pass

        model = self._models.get(language)
        if model is None:
            model = TTSModel.load_model(language=language, quantize=True)
            self._models[language] = model

        voice_source = str(voice_path) if voice_path else DEFAULT_VOICE
        state_key = (language, voice_source)
        voice_state = self._voice_states.get(state_key)
        if voice_state is None:
            voice_state = model.get_state_for_audio_prompt(voice_source)
            self._voice_states[state_key] = voice_state

        audio = generate_audio_with_quality(model, voice_state, text.strip())
        samples = audio.detach().cpu().numpy()

        destination = Path(output_path).expanduser()
        return _write_wav_atomically(sf, samples, int(model.sample_rate), destination)

    def export_voice_state(
        self,
        voice_audio_path: str | Path,
        output_path: str | Path,
        language: str,
    ) -> Path:
        """Encode a reference recording once and save its reusable KV state."""
        source = Path(voice_audio_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Áudio de referência não encontrado: {source}")
        destination = Path(output_path).expanduser()
        if destination.suffix.lower() != ".safetensors":
            raise ValueError("O estado de voz precisa ser salvo com a extensão .safetensors.")

        try:
            from pocket_tts import export_model_state, TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "A versão instalada do Pocket TTS não oferece exportação de estado de voz."
            ) from exc
        import torch

        try:
            torch.set_num_threads(2)
        except RuntimeError:
            pass

        model = self._models.get(language)
        if model is None:
            model = TTSModel.load_model(language=language, quantize=True)
            self._models[language] = model

        source_key = str(source.resolve())
        state_key = (language, source_key)
        voice_state = self._voice_states.get(state_key)
        if voice_state is None:
            voice_state = model.get_state_for_audio_prompt(source_key)
            self._voice_states[state_key] = voice_state

        destination.parent.mkdir(parents=True, exist_ok=True)
        export_model_state(voice_state, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("O Pocket TTS não criou um arquivo .safetensors válido.")
        return destination

    def close(self) -> None:
        self._voice_states.clear()
        self._models.clear()



def _write_wav_atomically(soundfile_module, samples, sample_rate: int, destination: Path) -> Path:
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=".wav",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        soundfile_module.write(
            temporary_path,
            samples,
            int(sample_rate),
            subtype="PCM_16",
        )
        info = soundfile_module.info(temporary_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError("O Pocket TTS produziu um WAV vazio ou inválido.")
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination
