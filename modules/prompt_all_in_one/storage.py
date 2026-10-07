"""Per-login durable history, favorites and server-only provider credentials."""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self._initialized = False
        self._lock = threading.Lock()

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                connection.executescript("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY, user TEXT NOT NULL, scope TEXT NOT NULL,
                    prompt TEXT NOT NULL, tags TEXT NOT NULL, name TEXT NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0, position REAL NOT NULL,
                    time REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS entries_owner ON entries(user, scope);
                CREATE TABLE IF NOT EXISTS providers (
                    user TEXT NOT NULL, provider TEXT NOT NULL, config TEXT NOT NULL,
                    PRIMARY KEY(user, provider));
                """)
        finally:
            connection.close()
        if os.name != "nt":
            self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                connection.row_factory = sqlite3.Row
                yield connection
        finally:
            connection.close()

    @staticmethod
    def document(row):
        result = dict(row)
        result.pop("user", None)
        result["tags"] = json.loads(result["tags"])
        result["favorite"] = bool(result["favorite"])
        return result

    def list(self, user, scope, favorites=False):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM entries WHERE user=? AND scope=? AND favorite=? ORDER BY position DESC LIMIT 1000",
                (user, scope, int(favorites)),
            ).fetchall()
        return [self.document(row) for row in rows]

    def add(self, user, scope, prompt, tags, name="", favorite=False, limit=100):
        now = time.time()
        with self.connect() as connection:
            latest = connection.execute(
                "SELECT * FROM entries WHERE user=? AND scope=? AND favorite=? ORDER BY position DESC LIMIT 1",
                (user, scope, int(favorite)),
            ).fetchone()
            encoded = json.dumps(tags, ensure_ascii=False)
            if (
                latest
                and latest["prompt"] == prompt
                and latest["tags"] == encoded
                and not favorite
            ):
                return self.document(latest)
            identifier = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    user,
                    scope,
                    prompt,
                    encoded,
                    name,
                    int(favorite),
                    now,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM entries WHERE id IN (SELECT id FROM entries WHERE user=? AND scope=? AND favorite=0 ORDER BY position DESC LIMIT -1 OFFSET ?)",
                (user, scope, limit),
            )
            row = connection.execute(
                "SELECT * FROM entries WHERE id=?", (identifier,)
            ).fetchone()
        return self.document(row)

    def update(self, user, identifier, name, position):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM entries WHERE id=? AND user=?", (identifier, user)
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE entries SET name=?, position=? WHERE id=? AND user=?",
                (
                    row["name"] if name is None else name,
                    row["position"] if position is None else position,
                    identifier,
                    user,
                ),
            )
        return True

    def delete(self, user, scope, favorites=False, identifier=None):
        with self.connect() as connection:
            if identifier:
                connection.execute(
                    "DELETE FROM entries WHERE id=? AND user=? AND scope=? AND favorite=?",
                    (identifier, user, scope, int(favorites)),
                )
            else:
                connection.execute(
                    "DELETE FROM entries WHERE user=? AND scope=? AND favorite=?",
                    (user, scope, int(favorites)),
                )

    def provider(self, user, name):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT config FROM providers WHERE user=? AND provider=?", (user, name)
            ).fetchone()
        return json.loads(row[0]) if row else {}

    def save_provider(self, user, name, values):
        # Empty secret keeps the existing credential; explicit clear erases it.
        previous = self.provider(user, name)
        clear = values.pop("clear_key", False)
        from .providers.base import METADATA

        secret_fields = {
            field["key"]
            for field in METADATA.get(name, {}).get("fields", [])
            if field["secret"]
        }
        options = values.get("options", {})
        if not clear:
            for field in secret_fields:
                if not options.get(field):
                    options[field] = previous.get("options", {}).get(field, "")
        else:
            for field in secret_fields:
                options[field] = ""
        values["options"] = options
        if clear:
            values["key"] = ""
        elif not values.get("key"):
            values["key"] = previous.get("key", "")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO providers VALUES (?,?,?)",
                (user, name, json.dumps(values)),
            )
        return self.public_provider(user, name)

    def public_provider(self, user, name):
        config = self.provider(user, name)
        from .providers.base import METADATA

        secret_fields = {
            field["key"]
            for field in METADATA.get(name, {}).get("fields", [])
            if field["secret"]
        }
        options = config.get("options", {})
        public_options = {
            key: value for key, value in options.items() if key not in secret_fields
        }
        return {
            **{
                key: value
                for key, value in config.items()
                if key not in ("key", "options")
            },
            "options": public_options,
            "fields": METADATA.get(name, {}).get("fields", []),
            "configured_fields": [key for key in secret_fields if options.get(key)],
            "configured": bool(
                config.get("key") or any(options.get(key) for key in secret_fields)
            ),
        }
