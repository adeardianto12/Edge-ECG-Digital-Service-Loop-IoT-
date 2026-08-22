"""SQLite WAL audit store with event idempotency and recoverable outbox state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class GatewayStore:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                endpoint_id TEXT NOT NULL,
                received_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outbox (
                event_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at TEXT,
                terminal_status TEXT NOT NULL DEFAULT 'pending'
            );
        """)
        self.connection.commit()

    def existing_response(self, event_id: str) -> dict | None:
        row = self.connection.execute("SELECT response_json FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return json.loads(row["response_json"]) if row else None

    def record(self, event_id: str, endpoint_id: str, received_at: str, payload: dict, status: str, response: dict) -> None:
        self.connection.execute(
            "INSERT INTO events(event_id, endpoint_id, received_at, payload_json, status, response_json) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, endpoint_id, received_at, json.dumps(payload), status, json.dumps(response)),
        )
        self.connection.commit()

    def queue_ack(self, event_id: str, topic: str, payload: dict) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO outbox(event_id, topic, payload_json, attempts, terminal_status) VALUES (?, ?, ?, 0, 'pending')",
            (event_id, topic, json.dumps(payload)),
        )
        self.connection.commit()

    def pending_outbox(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM outbox WHERE terminal_status = 'pending' ORDER BY rowid"))

    def mark_delivered(self, event_id: str) -> None:
        self.connection.execute("UPDATE outbox SET terminal_status = 'delivered' WHERE event_id = ?", (event_id,))
        self.connection.commit()

    def mark_attempt(self, event_id: str, terminal_status: str) -> None:
        self.connection.execute("UPDATE outbox SET attempts = attempts + 1, terminal_status = ? WHERE event_id = ?", (terminal_status, event_id))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def audit_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
