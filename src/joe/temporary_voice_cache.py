"""Persistent temporary snapshots of voice files used by the global shortcut."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path


class TemporaryVoiceFileCache:
    """Keep a selected voice snapshot until the user chooses a different voice.

    The cache lives in the user's cache directory so repeated one-shot shortcut
    launches reuse the same copy. Retired files are kept until no worker uses
    them, then removed. The original voice file is never modified.
    """

    MANIFEST_NAME = "current-voice.json"

    def __init__(self, cache_directory: str | Path | None = None) -> None:
        if cache_directory is None:
            cache_root = Path(
                os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            ).expanduser()
            cache_directory = cache_root / "joe-audio-studio" / "shortcut-voice"

        self._directory = Path(cache_directory).expanduser()
        self._source_key: str | None = None
        self._cached_path: Path | None = None
        self._retired_paths: set[Path] = set()
        self._load_existing_cache()

    @staticmethod
    def _canonical_source(source: str | Path) -> Path:
        return Path(source).expanduser().resolve(strict=False)

    @property
    def cached_path(self) -> Path | None:
        """Expose the active cached file for diagnostics and tests."""
        return self._cached_path

    def _load_existing_cache(self) -> None:
        manifest = self._directory / self.MANIFEST_NAME
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            source_key = str(data["source"])
            cached_name = str(data["file"])
            if not source_key or Path(cached_name).name != cached_name:
                return
            cached_path = self._directory / cached_name
            if cached_path.is_file():
                self._source_key = source_key
                self._cached_path = cached_path
        except (OSError, ValueError, TypeError, KeyError):
            return

    def _write_manifest(self, source_key: str, cached_path: Path) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        manifest = self._directory / self.MANIFEST_NAME
        temporary = self._directory / f".{self.MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
        payload = {"source": source_key, "file": cached_path.name}
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, source: str | Path | None) -> Path | None:
        """Return the stable snapshot, copying the source only after a voice change."""
        if source is None or not str(source).strip():
            self.invalidate()
            self.cleanup_retired()
            return None

        source_path = self._canonical_source(source)
        source_key = str(source_path)
        if (
            source_key == self._source_key
            and self._cached_path is not None
            and self._cached_path.is_file()
        ):
            return self._cached_path

        if not source_path.is_file():
            raise FileNotFoundError(f"Arquivo de voz não encontrado: {source_path}")

        # A shortcut generation is serialized, so a new snapshot is requested
        # only after a previous worker has stopped using the retired copy.
        self.invalidate()
        self.cleanup_retired()
        self._directory.mkdir(parents=True, exist_ok=True)

        suffix = source_path.suffix or ".bin"
        token = uuid.uuid4().hex
        destination = self._directory / f"voice-{token}{suffix}"
        partial = self._directory / f".voice-{token}.part"
        try:
            shutil.copy2(source_path, partial)
            if partial.stat().st_size != source_path.stat().st_size:
                raise OSError("A cópia temporária da voz ficou incompleta.")
            os.replace(partial, destination)
            self._write_manifest(source_key, destination)
        except Exception:
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise

        self._source_key = source_key
        self._cached_path = destination
        return destination

    def invalidate(self) -> None:
        """Forget the selected snapshot but defer deletion for active workers."""
        if self._cached_path is not None:
            self._retired_paths.add(self._cached_path)
        self._cached_path = None
        self._source_key = None
        manifest = self._directory / self.MANIFEST_NAME
        manifest.unlink(missing_ok=True)

    def cleanup_retired(self) -> None:
        """Delete snapshots that are no longer referenced by a running worker."""
        for path in tuple(self._retired_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            self._retired_paths.discard(path)

    def close(self) -> None:
        """Clean retired snapshots while preserving the active cache for next launch."""
        self.cleanup_retired()
