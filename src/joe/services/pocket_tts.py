from __future__ import annotations

from pathlib import Path

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

        audio = model.generate_audio(voice_state, text.strip())
        samples = audio.detach().cpu().numpy()

        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, samples, int(model.sample_rate), subtype="PCM_16")
        return destination

    def close(self) -> None:
        self._voice_states.clear()
        self._models.clear()
