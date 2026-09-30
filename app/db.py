"""Task history in SQLite (app/history.db)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


class History:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, request TEXT, created REAL, status TEXT, stop_reason TEXT,
                mode TEXT, events TEXT, files TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, tid: str, request: str, mode: str) -> None:
        with _LOCK, self._conn() as c:
            c.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                      (tid, request, time.time(), "running", "", mode, "[]", "{}"))

    def finish(self, tid: str, status: str, stop_reason: str, events: list, files: dict) -> None:
        with _LOCK, self._conn() as c:
            c.execute("UPDATE tasks SET status=?, stop_reason=?, events=?, files=? WHERE id=?",
                      (status, stop_reason, json.dumps(events), json.dumps(files), tid))

    def list(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, request, created, status, mode FROM tasks ORDER BY created DESC LIMIT ?",
                             (limit,)).fetchall()
        return [dict(zip(["id", "request", "created", "status", "mode"], r)) for r in rows]

    def get(self, tid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT id, request, created, status, stop_reason, mode, events, files FROM tasks WHERE id=?",
                          (tid,)).fetchone()
        if r is None:
            return None
        d = dict(zip(["id", "request", "created", "status", "stop_reason", "mode", "events", "files"], r))
        d["events"], d["files"] = json.loads(d["events"]), json.loads(d["files"])
        return d
