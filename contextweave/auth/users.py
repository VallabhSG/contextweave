"""API-key user registry backing private workspaces."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from contextweave.config import settings
from contextweave.timeutils import utcnow

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    api_key_hash TEXT NOT NULL UNIQUE,
    name TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class UserStore:
    """SQLite registry mapping hashed API keys to user ids.

    Keys are stored only as SHA-256 hashes; the plaintext key exists
    exactly once, in the register response.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            root = Path(settings.data_dir)
            root.mkdir(parents=True, exist_ok=True)
            db_path = str(root / "users.db")
        self._db_path = db_path
        with self._conn() as conn:
            conn.executescript(USERS_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_user(self, name: str = "") -> tuple[str, str]:
        """Create a user; returns (user_id, api_key)."""
        user_id = uuid.uuid4().hex[:12]
        api_key = f"cw_{secrets.token_hex(20)}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO users (id, api_key_hash, name, created_at) VALUES (?, ?, ?, ?)",
                (user_id, _hash_key(api_key), name[:80], utcnow().isoformat()),
            )
        return user_id, api_key

    def verify(self, api_key: str) -> str | None:
        """Return the user id for a valid API key, else None."""
        if not api_key:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE api_key_hash = ?", (_hash_key(api_key),)
            ).fetchone()
        return row["id"] if row else None
