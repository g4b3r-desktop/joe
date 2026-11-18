from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioHistoryEntry:
    id: int
    audio_path: str
    file_name: str
    created_at: str
    duration_seconds: float
    size_bytes: int
    voice_path: str
    voice_alias: str
    language: str
    generation_seconds: float
    realtime_speed: float
    backend: str
    text_content: str | None

    @property
    def voice_label(self) -> str:
        return self.voice_alias or (Path(self.voice_path).stem if self.voice_path else "Voz padrão")

    @property
    def source_label(self) -> str:
        return "Google Colab" if self.backend.casefold() == "google colab" else "Local"


class AudioHistoryStore:
    """SQLite index for generated WAV files; audio remains in its chosen folder."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    voice_path TEXT NOT NULL DEFAULT '',
                    voice_alias TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT '',
                    generation_seconds REAL NOT NULL DEFAULT 0,
                    realtime_speed REAL NOT NULL DEFAULT 0,
                    backend TEXT NOT NULL DEFAULT 'Local',
                    text_content TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audio_history_created_at "
                "ON audio_history(created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audio_history_voice "
                "ON audio_history(voice_alias, voice_path)"
            )

    @staticmethod
    def _entry(row: sqlite3.Row) -> AudioHistoryEntry:
        return AudioHistoryEntry(
            id=int(row["id"]),
            audio_path=str(row["audio_path"]),
            file_name=str(row["file_name"]),
            created_at=str(row["created_at"]),
            duration_seconds=float(row["duration_seconds"]),
            size_bytes=int(row["size_bytes"]),
            voice_path=str(row["voice_path"]),
            voice_alias=str(row["voice_alias"]),
            language=str(row["language"]),
            generation_seconds=float(row["generation_seconds"]),
            realtime_speed=float(row["realtime_speed"]),
            backend=str(row["backend"]),
            text_content=row["text_content"],
        )

    def add_generated_audio(
        self,
        audio_path: str | Path,
        *,
        duration_seconds: float,
        generation_seconds: float,
        voice_path: str | Path | None = None,
        voice_alias: str = "",
        language: str = "",
        backend: str = "Local",
        text_content: str | None = None,
        created_at: datetime | None = None,
    ) -> AudioHistoryEntry:
        path = Path(audio_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(f"Áudio do histórico não encontrado: {path}")
        stat = path.stat()
        if stat.st_size <= 0:
            raise ValueError("Não é possível registrar um áudio vazio no histórico.")

        duration = max(0.0, float(duration_seconds))
        elapsed = max(0.0, float(generation_seconds))
        speed = duration / elapsed if elapsed > 0 and duration > 0 else 0.0
        timestamp = (created_at or datetime.now().astimezone()).astimezone().isoformat(
            timespec="seconds"
        )
        voice = str(Path(voice_path).expanduser()) if voice_path else ""
        alias = str(voice_alias or "").strip() or (
            Path(voice).stem if voice else "Voz padrão"
        )
        language = str(language or "").strip()
        remote_sources = {"google colab", "colab", "remote"}
        source = (
            "Google Colab"
            if str(backend).casefold() in remote_sources
            else "Local"
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audio_history (
                    audio_path, file_name, created_at, duration_seconds, size_bytes,
                    voice_path, voice_alias, language, generation_seconds,
                    realtime_speed, backend, text_content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audio_path) DO UPDATE SET
                    file_name=excluded.file_name,
                    created_at=excluded.created_at,
                    duration_seconds=excluded.duration_seconds,
                    size_bytes=excluded.size_bytes,
                    voice_path=excluded.voice_path,
                    voice_alias=excluded.voice_alias,
                    language=excluded.language,
                    generation_seconds=excluded.generation_seconds,
                    realtime_speed=excluded.realtime_speed,
                    backend=excluded.backend,
                    text_content=excluded.text_content
                """,
                (
                    str(path),
                    path.name,
                    timestamp,
                    duration,
                    stat.st_size,
                    voice,
                    alias,
                    language,
                    elapsed,
                    speed,
                    source,
                    text_content,
                ),
            )
            row = connection.execute(
                "SELECT * FROM audio_history WHERE audio_path = ?", (str(path),)
            ).fetchone()
        if row is None:
            raise RuntimeError("Não foi possível confirmar o registro no histórico.")
        return self._entry(row)

    def get(self, entry_id: int) -> AudioHistoryEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM audio_history WHERE id = ?", (int(entry_id),)
            ).fetchone()
        return self._entry(row) if row is not None else None

    def list_entries(
        self,
        *,
        search: str = "",
        voice: str = "",
        date_from: str = "",
        date_to: str = "",
        min_duration: float | None = None,
        max_duration: float | None = None,
    ) -> list[AudioHistoryEntry]:
        clauses: list[str] = []
        values: list[object] = []

        search = str(search or "").strip()
        if search:
            pattern = f"%{search}%"
            clauses.append(
                "(file_name LIKE ? OR audio_path LIKE ? OR voice_alias LIKE ? "
                "OR voice_path LIKE ? OR language LIKE ? OR COALESCE(text_content, '') LIKE ?)"
            )
            values.extend([pattern] * 6)

        if voice:
            clauses.append(
                "COALESCE(NULLIF(voice_alias, ''), NULLIF(voice_path, ''), 'Voz padrão') = ?"
            )
            values.append(voice)
        if date_from:
            clauses.append("substr(created_at, 1, 10) >= ?")
            values.append(date_from)
        if date_to:
            clauses.append("substr(created_at, 1, 10) <= ?")
            values.append(date_to)
        if min_duration is not None and float(min_duration) > 0:
            clauses.append("duration_seconds >= ?")
            values.append(float(min_duration))
        if max_duration is not None and float(max_duration) > 0:
            clauses.append("duration_seconds <= ?")
            values.append(float(max_duration))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM audio_history {where} ORDER BY created_at DESC, id DESC",
                values,
            ).fetchall()
        return [self._entry(row) for row in rows]

    def available_voices(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT
                    COALESCE(NULLIF(voice_alias, ''), NULLIF(voice_path, ''), 'Voz padrão') AS label
                FROM audio_history
                ORDER BY label COLLATE NOCASE
                """
            ).fetchall()
        return [str(row["label"]) for row in rows]

    def rename_audio(self, entry_id: int, new_name: str) -> AudioHistoryEntry:
        entry = self.get(entry_id)
        if entry is None:
            raise KeyError(f"Registro de histórico não encontrado: {entry_id}")
        source = Path(entry.audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"O áudio não está mais na pasta: {source}")

        candidate = str(new_name).strip()
        if not candidate or Path(candidate).name != candidate or candidate in {".", ".."}:
            raise ValueError("Informe somente um nome de arquivo, sem pastas.")
        if Path(candidate).suffix.lower() != source.suffix.lower():
            candidate = f"{candidate}{source.suffix}"
        destination = source.with_name(candidate)
        if destination == source:
            return entry
        if destination.exists():
            raise FileExistsError(f"Já existe um arquivo com esse nome: {destination.name}")

        source.rename(destination)
        try:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE audio_history SET audio_path = ?, file_name = ? WHERE id = ?",
                    (str(destination), destination.name, int(entry_id)),
                )
        except Exception:
            try:
                destination.rename(source)
            except OSError:
                pass
            raise
        updated = self.get(entry_id)
        if updated is None:
            raise RuntimeError("O registro renomeado não pôde ser recuperado.")
        return updated

    def delete(self, entry_id: int, *, delete_file: bool = True) -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        path = Path(entry.audio_path)
        if delete_file and path.exists():
            path.unlink()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM audio_history WHERE id = ?", (int(entry_id),)
            )
        return cursor.rowcount > 0

    def export_copy(self, entry_id: int, destination: str | Path) -> Path:
        entry = self.get(entry_id)
        if entry is None:
            raise KeyError(f"Registro de histórico não encontrado: {entry_id}")
        source = Path(entry.audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"O áudio não está mais na pasta: {source}")
        target = Path(destination).expanduser()
        if target.exists() and target.resolve() == source.resolve():
            raise ValueError("Escolha um destino diferente do arquivo original.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
