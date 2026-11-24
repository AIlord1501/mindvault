"""Versioned, transactional SQLite schema migration runner."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.db.database import DatabaseManager

_MIGRATION_FILE_RE = re.compile(r"^(\d+)_([a-zA-Z0-9_-]+)\.sql$")
DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


@dataclass(frozen=True)
class MigrationFile:
    """Represents a single SQL migration script on disk."""

    version: int
    name: str
    path: Path
    sql: str
    checksum: str


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script into complete executable statements without implicit commits."""
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not buffer and (not stripped or stripped.startswith("--")):
            continue
        buffer.append(line)
        candidate = "\n".join(buffer)
        if sqlite3.complete_statement(candidate):
            cleaned = candidate.strip()
            if cleaned and cleaned != ";":
                statements.append(cleaned)
            buffer.clear()
    remainder = "\n".join(buffer).strip()
    if remainder and not remainder.startswith("--"):
        statements.append(remainder)
    return statements


class MigrationRunner:
    """Applies ordered SQL migrations and tracks state in `schema_migrations`."""

    def __init__(
        self,
        db: DatabaseManager,
        migrations_dir: Optional[Path] = None,
    ) -> None:
        self._db = db
        self._migrations_dir = migrations_dir or DEFAULT_MIGRATIONS_DIR

    def _ensure_tracking_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )

    def discover(self) -> list[MigrationFile]:
        """Discover and sort SQL migration files by numeric version."""
        if not self._migrations_dir.exists():
            return []

        found: list[MigrationFile] = []
        seen_versions: set[int] = set()
        for path in sorted(self._migrations_dir.iterdir()):
            if not path.is_file():
                continue
            match = _MIGRATION_FILE_RE.match(path.name)
            if not match:
                continue
            version = int(match.group(1))
            name = match.group(2)
            if version in seen_versions:
                raise RuntimeError(
                    f"Duplicate migration version {version} in {self._migrations_dir}"
                )
            seen_versions.add(version)
            sql = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            found.append(
                MigrationFile(
                    version=version,
                    name=name,
                    path=path,
                    sql=sql,
                    checksum=checksum,
                )
            )

        found.sort(key=lambda item: item.version)
        return found

    def applied_versions(self) -> list[int]:
        """Return sorted list of applied migration version numbers."""
        with self._db.connection() as conn:
            self._ensure_tracking_table(conn)
            rows = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version ASC"
            ).fetchall()
            return [int(row["version"]) for row in rows]

    def current_version(self) -> int:
        """Return highest applied migration version, or 0 if none applied."""
        versions = self.applied_versions()
        return versions[-1] if versions else 0

    def apply_pending(self) -> list[int]:
        """Apply all unapplied migrations atomically and return applied versions."""
        migrations = self.discover()
        newly_applied: list[int] = []

        with self._db.transaction(immediate=True) as conn:
            self._ensure_tracking_table(conn)
            rows = conn.execute(
                "SELECT version, checksum FROM schema_migrations ORDER BY version ASC"
            ).fetchall()
            applied_checksums = {int(row["version"]): str(row["checksum"]) for row in rows}

            for migration in migrations:
                if migration.version in applied_checksums:
                    if applied_checksums[migration.version] != migration.checksum:
                        raise RuntimeError(
                            f"Checksum mismatch for already-applied migration {migration.version} "
                            f"({migration.name})."
                        )
                    continue

                for statement in _split_sql_statements(migration.sql):
                    conn.execute(statement)

                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, name, checksum, applied_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        migration.version,
                        migration.name,
                        migration.checksum,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                newly_applied.append(migration.version)

        return newly_applied
