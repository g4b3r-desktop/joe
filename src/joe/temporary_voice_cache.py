"""Temporary snapshots of voice files used by the global shortcut."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path


class TemporaryVoiceFileCache:
    """Keep one selected voice snapshot until the user changes the voice.

    Old snapshots are retired immediately on selection changes, but their files
    are removed only when the caller confirms no generation is using them.
    """

    def __init__(self) -> None:
        self._directory: Path | None = None
        self._source_key: str | None = None
        self._cached_path: Path | None = None
        self._retired_paths: set[Path] = set()

    @staticmethod
    def _canonical_source(source: str | Path) -> Path:
        return Path(source).expanduser().resolve(strict=False)

    def get(self, source: str | Path | None) -> Path | None:
        """Return a stable temporary copy, creating it only for a new selection."""
        if source is None or not str(source).strip():
            return None

        source_path = self._canonical_source(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Arquivo de voz não encontrado: {source_path}")

        source_key = str(source_path)
        if (
            source_key == self._source_key
            and self._cached_path is not None
            and self._cached_path.is_file()
        ):
            return self._cached_path

        # get() is called only when the previous shortcut generation is idle.
        self.invalidate()
        self.cleanup_retired()

        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix="joe-shortcut-voice-"))

        suffix = source_path.suffix or ".bin"
        destination = self._directory / f"voice-{uuid.uuid4().hex}{suffix}"
        shutil.copy2(source_path, destination)
        if destination.stat().st_size != source_path.stat().st_size:
            destination.unlink(missing_ok=True)
            raise OSError("A cópia temporária da voz ficou incompleta.")

        self._source_key = source_key
        self._cached_path = destination
        return destination

    def invalidate(self) -> None:
        """Forget the selected snapshot but defer deletion for active workers."""
        if self._cached_path is not None:
            self._retired_paths.add(self._cached_path)
        self._cached_path = None
        self._source_key = None

    def cleanup_retired(self) -> None:
        """Delete snapshots that are no longer referenced by a running worker."""
        for path in tuple(self._retired_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            self._retired_paths.discard(path)

    def close(self) -> None:
        """Remove all shortcut voice snapshots when the Joe process exits."""
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
        self._directory = None
        self._source_key = None
        self._cached_path = None
        self._retired_paths.clear()
