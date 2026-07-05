"""Digest subscriptions: who gets the daily email, when, and opt-out state.

Account-level data, so it lives beside the user registry (users.db in
SQLite mode, cw_digest_subscriptions in Postgres mode) — never inside a
workspace, and never touched by a memory wipe.
"""

from __future__ import annotations

import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from contextweave.config import settings
from contextweave.timeutils import utcnow

SUBSCRIPTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS digest_subscriptions (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    send_hour_utc INTEGER NOT NULL,
    unsubscribe_token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_sent_on TEXT
);
"""


class SubscriptionStore:
    """SQLite subscription registry (shares users.db with the user store)."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            root = Path(settings.data_dir)
            root.mkdir(parents=True, exist_ok=True)
            db_path = str(root / "users.db")
        self._db_path = db_path
        with self._conn() as conn:
            conn.executescript(SUBSCRIPTIONS_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def subscribe(self, user_id: str, email: str, send_hour_utc: int) -> str:
        """Create or update a subscription; returns the unsubscribe token."""
        token = secrets.token_urlsafe(24)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO digest_subscriptions "
                "(user_id, email, send_hour_utc, unsubscribe_token, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (user_id) DO UPDATE SET email = excluded.email, "
                "send_hour_utc = excluded.send_hour_utc, "
                "unsubscribe_token = excluded.unsubscribe_token",
                (user_id, email, send_hour_utc, token, utcnow().isoformat()),
            )
        return token

    def unsubscribe(self, user_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM digest_subscriptions WHERE user_id = ?", (user_id,))
        return cur.rowcount > 0

    def unsubscribe_by_token(self, token: str) -> bool:
        if not token:
            return False
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM digest_subscriptions WHERE unsubscribe_token = ?", (token,)
            )
        return cur.rowcount > 0

    def get(self, user_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT email, send_hour_utc, unsubscribe_token, last_sent_on "
                "FROM digest_subscriptions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def due(self, hour_utc: int, today_iso: str) -> list[dict]:
        """Subscriptions scheduled for this hour that haven't been sent today."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT user_id, email, unsubscribe_token FROM digest_subscriptions "
                "WHERE send_hour_utc = ? AND (last_sent_on IS NULL OR last_sent_on != ?)",
                (hour_utc, today_iso),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_sent(self, user_id: str, today_iso: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE digest_subscriptions SET last_sent_on = ? WHERE user_id = ?",
                (today_iso, user_id),
            )


def get_subscription_store():
    """Registry matching the active storage backend (Postgres or SQLite)."""
    if settings.database_url:
        from contextweave.storage.postgres import PgSubscriptionStore

        return PgSubscriptionStore()
    return SubscriptionStore()
