"""Optional completion counts, isolated by login and positive/negative role."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class FrequencyStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS usage (
                    user TEXT, name TEXT, kind TEXT, negative INTEGER,
                    count INTEGER NOT NULL, last_used TEXT NOT NULL,
                    PRIMARY KEY (user, name, kind, negative))""")
                yield connection
        finally:
            connection.close()

    def increase(self, user, name, kind, negative):
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO usage VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(user, name, kind, negative) DO UPDATE
                SET count = count + 1, last_used = excluded.last_used""",
                (user, name, kind, int(negative), now),
            )

    def get(self, user):
        if not self.path.exists():
            return []
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT name, kind, negative, count, last_used FROM usage WHERE user = ?",
                (user,),
            ).fetchall()
        return [
            dict(zip(("name", "kind", "negative", "count", "last_used"), row))
            for row in rows
        ]

    def clear(self, user):
        if self.path.exists():
            with self.lock, self.connect() as connection:
                connection.execute("DELETE FROM usage WHERE user = ?", (user,))
