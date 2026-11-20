"""Persistent conversation storage on SQLite (spec section 10).

Conversations and messages survive restarts. Schema changes are applied
through an explicit migration list keyed by ``PRAGMA user_version``.
A backup is a plain file copy of the database (see scripts/backup.py).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MIGRATIONS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT '',
        sources_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id, id);
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for i, sql in enumerate(_MIGRATIONS, start=1):
            if version < i:
                self._conn.executescript(sql)
                self._conn.execute(f"PRAGMA user_version={i}")
                self._conn.commit()

    # -- conversations -------------------------------------------------
    def create_conversation(self, title: str = "") -> dict[str, Any]:
        cid = uuid.uuid4().hex
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (cid, title or "محادثة جديدة", now, now))
        return self.get_conversation(cid)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE id=?", (conversation_id,))
        return cur.rowcount > 0

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (title, _now(), conversation_id))

    # -- messages ------------------------------------------------------
    def add_message(self, conversation_id: str, role: str, content: str,
                    state: str = "", sources: list[dict] | None = None) -> dict[str, Any]:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO messages(conversation_id, role, content, state, sources_json, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (conversation_id, role, content, state,
                 json.dumps(sources or [], ensure_ascii=False), now))
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            message_id = cur.lastrowid
        return {"id": message_id, "conversation_id": conversation_id, "role": role,
                "content": content, "state": state, "sources": sources or [],
                "created_at": now}

    def get_messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
            (conversation_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json") or "[]")
            result.append(item)
        return result

    def count_messages(self, conversation_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?",
            (conversation_id,)).fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._conn.close()
