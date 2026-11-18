---
name: joe-audio-studio
description: Use this skill when explaining, developing, debugging, testing, or documenting Joe Audio Studio in the g4b3r-desktop/joe repository. Covers desktop TTS/STT, voice libraries and favorites, clipboard hotkey, local/Colab backends, audio performance, installation, and project conventions.
---

# Joe Audio Studio — project skill

Use this skill as project-specific context for work on **Joe Audio Studio**, the desktop application that converts text to speech and transcribes audio. The repository is `g4b3r-desktop/joe`.

## Source of truth

Before changing code, inspect the current branch and the relevant files. This skill is a maintained project guide, not a substitute for the live source.

When sources disagree, use this order:
1. Current implementation and tests.
2. Current notebook/API implementation.
3. Current README and workflow documentation.
4. This skill and historical conversation context.

Do not assume a historical commit, UI detail, dependency version, or test result is still current. Verify it.

## Project rules

- The active development branch is `feature/colab-model-api`. Confirm that it still exists and is the intended branch before writing.
- Never commit feature work to `main`. Do not force-push, reset, or rewrite branch history.
- Preserve existing features, user settings, saved voice files, generated audio, model caches, and checkpoints. Do not delete or replace user data to simplify an implementation.
- Prefer small, focused changes. Do not combine unrelated refactors with a feature request.
- Keep the desktop UI and user-facing messages in Brazilian Portuguese unless the user requests another language.
- For a feature request, implement the requested behavior when repository access permits; do not stop at a proposal. Add or update tests and documentation where appropriate.
- Do not claim that code was run, tests passed, or a commit was pushed unless tool output verifies it. Distinguish static review, compilation, unit tests, and CI status.
- Treat Colab URLs and Bearer tokens as secrets. Do not print, commit, log, or include real credentials in examples. The current app stores the configured token in local user settings without encryption; warn before recommending sharing those settings.
- Never upload model weights, generated audio, user recordings, local environment files, or credentials to Git.

## Understand the product first

Joe has three main desktop areas:
- **Texto → Áudio:** type/paste text, choose a voice and language, generate WAV, and play the latest output.
- **Áudio → Texto:** select an audio file, transcribe it, review/edit the transcript, and save/copy the result; SRT is available when timestamped segments exist.
- **Google Colab / API:** choose local or remote inference, configure the temporary HTTPS tunnel and token, test the connection, and open the Colab notebook.

Less frequently changed TTS options live in a separate **⚙ Configurações** dialog. The main TTS screen also exposes **★ Favoritos**. Voice selection is handled by a recursive library with format filtering and immediate preview during mouse or keyboard navigation.

## Architecture and file map

Read [references/project-map.md](references/project-map.md) for module responsibilities and test locations.

Key modules:
- `src/joe/app.py`: PySide6 desktop UI, tabs, dialogs, QSettings bindings, workers, playback, global hotkey and orchestration.
- `src/joe/services/pocket_tts.py`: local Pocket TTS loading, voice conditioning/cache, WAV generation, and reusable `.safetensors` voice-state export.
- `src/joe/services/parakeet.py`: Parakeet Redux transcription with Photon when compatible, otherwise the Transformers reference backend; supports explicit CPU/CUDA preference.
- `src/joe/services/colab_api.py`: authenticated HTTP client, asynchronous job polling, remote TTS and transcription.
- `colab/api_server.py`: FastAPI service run inside the Colab runtime.
- `colab/Joe_Colab_API.ipynb`: notebook that prepares models/runtime and starts the API plus Cloudflare Quick Tunnel.
- `src/joe/audio_chunks.py`: sentence-aware text splitting and WAV chunk merging.
- `src/joe/audio_performance.py`: desktop-side timing, WAV duration and speed report.
- `src/joe/local_ipc.py`: private Unix-domain socket used to forward shortcut launches to the running Joe instance.
- `src/joe/voice_favorites.py`: normalization and JSON persistence helpers for favorite voice paths and aliases.
- `src/joe/text_normalization.py`: language-aware speech text preparation.
- `scripts/install-linux.sh` and `scripts/install-desktop.sh`: Linux dependencies, launcher, icon and Xfce shortcut setup.

## Required working method

1. Inspect the current branch head and working tree. Fetch the latest branch state before planning a commit.
2. Read the smallest relevant parts of implementation, tests, and documentation. Follow existing conventions rather than guessing.
3. Implement the smallest coherent change. Preserve the current API and settings keys unless migration is explicitly needed.
4. Add regression tests for the behavior and edge cases. Prefer testing pure helpers independently of Qt/model downloads.
5. Run focused tests first, then the full available checks:
   - `python -m pytest`
   - `ruff check .`
   - `python -m compileall -q src colab`
6. Inspect the final diff. Ensure no model weights, recordings, secrets, cache folders, generated WAVs, or local environment files were added.
7. Commit to `feature/colab-model-api` only. Re-fetch the branch head before updating it; use a normal fast-forward update, not force.
8. Report the commit, files changed, tests actually run and any failures. Give the user commands to update their Linux checkout.

See [references/development-workflow.md](references/development-workflow.md) for commands and Git safety details.

## Feature behavior and constraints

Read [references/feature-reference.md](references/feature-reference.md) before modifying voice handling, the clipboard shortcut, TTS performance, or the Colab integration.

Important invariants:
- The original typed text stays intact; speech normalization is performed on a prepared copy.
- Voice favorites are keyed by canonical file path and have a user-editable alias.
- A `.safetensors` file is a prepared voice state, not playable audio.
- In the voice library, a single click and Up/Down navigation preview playable audio; a double click or explicit selection action chooses the voice.
- The clipboard shortcut is `Ctrl+Shift+Espaço`; it should open the compact audio window, not unexpectedly show a second main window.
- Closing the compact clipboard window removes temporary generated chunks/audio; a file explicitly saved by the user remains permanent.
- Performance is measured by the desktop app. Remote timing includes upload, queueing and polling/waiting; local timing includes model loading when it occurs.
- Colab connectivity monitoring does not keep a runtime awake or prevent provider disconnections.
- Local voice `.safetensors` support does not imply that the current remote API accepts that state format.

## User-facing update style

Use clear Brazilian Portuguese. Prefer direct, actionable instructions. When a change is committed, include:
- What changed.
- The verified commit link.
- Exact update commands, normally:

```bash
git checkout feature/colab-model-api
git pull origin feature/colab-model-api
.venv/bin/pip install -e .
bash run.sh
```

Do not state that the user’s local machine was updated; a GitHub commit only updates the remote branch.
