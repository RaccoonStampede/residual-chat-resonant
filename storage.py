#!/usr/bin/env python3
"""
storage.py — SQLite persistence layer for ResidualChat-Resonant.

Tables:
  sessions        — chat sessions (id, created_at, updated_at, metadata JSON)
  messages        — individual turns (session_id, role, content, timestamp, metadata JSON)
  fragments       — trained fragments (text, signature_hash, session_id, timestamp)
  state_snapshots — periodic coherent_state snapshots for session recovery
"""

import json
import logging
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StorageManager:
    """Handles all SQLite DB operations for ResidualChat-Resonant."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS fragments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        text            TEXT NOT NULL,
        signature_hash  TEXT NOT NULL,
        session_id      TEXT,
        timestamp       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS state_snapshots (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        state_json  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_fragments_session ON fragments(session_id);
    CREATE INDEX IF NOT EXISTS idx_snapshots_session ON state_snapshots(session_id);
    """

    def __init__(self, db_path: str = "./data/residual.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)
        logger.debug("DB schema initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def create_session(self, metadata: Optional[Dict] = None) -> str:
        session_id = str(uuid.uuid4())
        now = _now()
        meta_json = json.dumps(metadata or {})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
                (session_id, now, now, meta_json),
            )
        logger.debug("Created session %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        return d

    def update_session(self, session_id: str, metadata: Dict):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ?, metadata = ? WHERE id = ?",
                (_now(), json.dumps(metadata), session_id),
            )

    def touch_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_now(), session_id),
            )

    def delete_session(self, session_id: str):
        """Soft-delete by marking metadata deleted=true."""
        session = self.get_session(session_id)
        if session is None:
            return
        meta = session["metadata"]
        meta["deleted"] = True
        meta["deleted_at"] = _now()
        self.update_session(session_id, meta)

    def list_sessions(
        self,
        page: int = 1,
        per_page: int = 20,
        include_deleted: bool = False,
    ) -> Tuple[List[Dict], int]:
        offset = (page - 1) * per_page
        with self._connect() as conn:
            if include_deleted:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (per_page, offset),
                ).fetchall()
                total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            else:
                # Exclude rows where metadata JSON contains "deleted":true
                rows = conn.execute(
                    "SELECT * FROM sessions "
                    "WHERE json_extract(metadata, '$.deleted') IS NOT 1 "
                    "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (per_page, offset),
                ).fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) FROM sessions "
                    "WHERE json_extract(metadata, '$.deleted') IS NOT 1"
                ).fetchone()[0]
        sessions = []
        for row in rows:
            d = dict(row)
            d["metadata"] = json.loads(d["metadata"])
            sessions.append(d)
        return sessions, total

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, now, json.dumps(metadata or {})),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            return cur.lastrowid

    def get_messages(self, session_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    def update_message_metadata(self, message_id: int, metadata: Dict):
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), message_id),
            )

    # ------------------------------------------------------------------
    # Fragment operations
    # ------------------------------------------------------------------

    def save_fragment(
        self,
        text: str,
        signature_hash: str,
        session_id: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO fragments (text, signature_hash, session_id, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (text, signature_hash, session_id, _now()),
            )
            return cur.lastrowid

    def get_fragments(
        self,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        query = "SELECT * FROM fragments WHERE 1=1"
        params: List[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if search:
            query += " AND text LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def fragment_exists(self, signature_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM fragments WHERE signature_hash = ? LIMIT 1",
                (signature_hash,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # State snapshot operations
    # ------------------------------------------------------------------

    def save_snapshot(self, session_id: str, state: Dict):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO state_snapshots (session_id, timestamp, state_json) VALUES (?, ?, ?)",
                (session_id, _now(), json.dumps(state)),
            )

    def get_latest_snapshot(self, session_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM state_snapshots WHERE session_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["state"] = json.loads(d["state_json"])
        return d

    # ------------------------------------------------------------------
    # Session export helpers
    # ------------------------------------------------------------------

    def export_session_json(self, session_id: str) -> Dict:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        messages = self.get_messages(session_id)
        fragments = self.get_fragments(session_id=session_id)
        snapshot = self.get_latest_snapshot(session_id)
        return {
            "session": session,
            "messages": messages,
            "fragments": fragments,
            "snapshot": snapshot,
            "exported_at": _now(),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        with self._connect() as conn:
            total_sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            total_messages = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            total_fragments = conn.execute(
                "SELECT COUNT(*) FROM fragments"
            ).fetchone()[0]
            avg_fragments = conn.execute(
                "SELECT AVG(cnt) FROM ("
                "  SELECT COUNT(*) as cnt FROM fragments GROUP BY session_id"
                ")"
            ).fetchone()[0]
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_fragments": total_fragments,
            "avg_fragments_per_session": round(avg_fragments or 0, 2),
        }

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(self, backup_dir: Optional[str] = None) -> str:
        if backup_dir is None:
            backup_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = os.path.join(backup_dir, f"residual_backup_{timestamp}.db")
        shutil.copy2(self.db_path, dest)
        logger.info("DB backed up to %s", dest)
        return dest
