"""SQLite connection and transaction management for MindVault."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


class DatabaseManager:
    """Manages local SQLite connections with strict pragmas and graceful shutdown."""

    def __init__(self, db_path: Path, busy_timeout_ms: int = 5000) -> None:
        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        self._connections: set[sqlite3.Connection] = set()
        self._closed = False
        self.recovered_from_corruption: bool = False
        self.quarantined_path: Optional[str] = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _create_connection(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("DatabaseManager is closed.")

        is_memory = str(self._db_path) == ":memory:"
        if not is_memory:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(self._db_path),
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)};")
        if not is_memory:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection and close it on exit."""
        conn = self._create_connection()
        with self._lock:
            self._connections.add(conn)
        try:
            yield conn
        finally:
            with self._lock:
                self._connections.discard(conn)
            try:
                conn.close()
            except sqlite3.Error:
                pass

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a connection wrapped in an explicit SQLite transaction."""
        begin_sql = "BEGIN IMMEDIATE" if immediate else "BEGIN"
        with self.connection() as conn:
            conn.execute(begin_sql)
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def health_check(self) -> tuple[bool, Optional[str]]:
        """Verify database reachability with a lightweight read query."""
        if self._closed:
            return False, "Database manager is closed."
        try:
            with self.connection() as conn:
                row = conn.execute("SELECT 1 AS ok").fetchone()
                return (row is not None and row["ok"] == 1), None
        except sqlite3.Error as exc:
            return False, str(exc)

    def ensure_healthy_or_recover(self) -> dict[str, Any]:
        """Verify SQLite integrity via quick_check, quarantining and replacing corrupt DBs."""
        is_memory = str(self._db_path) == ":memory:"
        if is_memory or not self._db_path.exists():
            return {
                "healthy": True,
                "recovered": False,
                "quarantined_path": None,
                "error": None,
            }

        failure_reason: Optional[str] = None
        try:
            with self.connection() as conn:
                row = conn.execute("PRAGMA quick_check;").fetchone()
                status_val = str(row[0]).strip().lower() if row is not None else ""
                if status_val != "ok":
                    failure_reason = f"SQLite PRAGMA quick_check returned '{status_val}'."
        except sqlite3.Error as exc:
            failure_reason = str(exc)

        if failure_reason is None:
            return {
                "healthy": True,
                "recovered": False,
                "quarantined_path": None,
                "error": None,
            }

        with self._lock:
            active = list(self._connections)
            self._connections.clear()
        for conn in active:
            try:
                conn.close()
            except sqlite3.Error:
                pass

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup_path = self._db_path.with_name(f"{self._db_path.name}.corrupt.{stamp}.bak")
        if self._db_path.exists():
            self._db_path.replace(backup_path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self._db_path}{suffix}")
            if sidecar.exists():
                try:
                    sidecar.replace(Path(f"{backup_path}{suffix}"))
                except OSError:
                    pass

        self.recovered_from_corruption = True
        self.quarantined_path = str(backup_path)
        return {
            "healthy": True,
            "recovered": True,
            "quarantined_path": str(backup_path),
            "error": failure_reason,
        }

    def close(self) -> None:
        """Close all tracked SQLite connections during graceful shutdown."""
        with self._lock:
            self._closed = True
            active = list(self._connections)
            self._connections.clear()
        for conn in active:
            try:
                conn.close()
            except sqlite3.Error:
                pass
