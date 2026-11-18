from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from joe.audio_performance import wav_duration_seconds


@dataclass(frozen=True, slots=True)
class AudioHistoryEntry:
    entry_id: int
    path: Path
    created_at: str
    duration_seconds: float
    size_bytes: int
    voice_path: str
    voice_alias: str
    backend: str
    elapsed_seconds: float
    realtime_speed: float
    chars_per_second: float
    text: str | None


class AudioHistoryStore:
    """Persistent local index of generated audio files; deleting an entry never deletes audio."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    voice_path TEXT NOT NULL DEFAULT '',
                    voice_alias TEXT NOT NULL DEFAULT '',
                    backend TEXT NOT NULL DEFAULT '',
                    elapsed_seconds REAL NOT NULL DEFAULT 0,
                    realtime_speed REAL NOT NULL DEFAULT 0,
                    chars_per_second REAL NOT NULL DEFAULT 0,
                    text TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audio_history_created "
                "ON audio_history(created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audio_history_path "
                "ON audio_history(path)"
            )

    def add(
        self,
        audio_path: str | Path,
        *,
        voice_path: str | Path | None = None,
        voice_alias: str = "",
        backend: str = "",
        elapsed_seconds: float = 0.0,
        chars_per_second: float = 0.0,
        text: str | None = None,
    ) -> int:
        """Index an existing WAV. The original text is stored only when explicitly supplied."""
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Áudio não encontrado para o histórico: {path}")
        if path.suffix.lower() != ".wav":
            raise ValueError("O histórico do Joe aceita arquivos WAV gerados.")

        duration = wav_duration_seconds(path)
        size = path.stat().st_size
        elapsed = max(0.0, float(elapsed_seconds))
        speed = duration / elapsed if elapsed > 0 and duration > 0 else 0.0
        normalized_voice = str(Path(voice_path).expanduser()) if voice_path else ""
        alias = str(voice_alias or "").strip()
        record_text = str(text) if text is not None else None
        created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audio_history (
                    path, created_at, duration_seconds, size_bytes,
                    voice_path, voice_alias, backend, elapsed_seconds,
                    realtime_speed, chars_per_second, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(path.resolve()),
                    created_at,
                    duration,
                    size,
                    normalized_voice,
                    alias,
                    str(backend or ""),
                    elapsed,
                    speed,
                    max(0.0, float(chars_per_second)),
                    record_text,
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _entry(row: sqlite3.Row) -> AudioHistoryEntry:
        return AudioHistoryEntry(
            entry_id=int(row["id"]),
            path=Path(row["path"]),
            created_at=str(row["created_at"]),
            duration_seconds=float(row["duration_seconds"]),
            size_bytes=int(row["size_bytes"]),
            voice_path=str(row["voice_path"]),
            voice_alias=str(row["voice_alias"]),
            backend=str(row["backend"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            realtime_speed=float(row["realtime_speed"]),
            chars_per_second=float(row["chars_per_second"]),
            text=row["text"],
        )

    def list_entries(
        self,
        *,
        search: str = "",
        limit: int = 500,
    ) -> list[AudioHistoryEntry]:
        """Return newest entries first. Search never requires stored source text."""
        safe_limit = min(max(int(limit), 1), 5000)
        query = str(search or "").strip()
        with self._connect() as connection:
            if query:
                pattern = f"%{query}%"
                rows = connection.execute(
                    """
                    SELECT * FROM audio_history
                    WHERE path LIKE ? OR voice_alias LIKE ? OR voice_path LIKE ?
                       OR backend LIKE ? OR COALESCE(text, '') LIKE ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (pattern, pattern, pattern, pattern, pattern, safe_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM audio_history
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        return [self._entry(row) for row in rows]

    def get(self, entry_id: int) -> AudioHistoryEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM audio_history WHERE id = ?",
                (int(entry_id),),
            ).fetchone()
        return self._entry(row) if row is not None else None

    def remove_entry(self, entry_id: int) -> bool:
        """Remove metadata only. The WAV is intentionally never deleted here."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM audio_history WHERE id = ?",
                (int(entry_id),),
            )
            return cursor.rowcount > 0
