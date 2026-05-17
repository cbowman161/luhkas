import sqlite3
import json
from datetime import datetime

from config import DB_PATH


class EventLog:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE,
            event_type TEXT,
            message TEXT,
            data TEXT,
            read INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            level TEXT,
            message TEXT,
            data TEXT,
            read INTEGER,
            created_at TEXT
        )
        """)

        self.conn.commit()

    def notify(self, job_id, level, message, data=None):
        self.conn.execute(
            """
            INSERT INTO notifications (job_id, level, message, data, read, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                level,
                message,
                json.dumps(data or {}, default=str),
                0,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def unread_notifications(self):
        rows = self.conn.execute(
            """
            SELECT id, job_id, level, message, data, created_at
            FROM notifications
            WHERE read=0
            ORDER BY id ASC
            """
        ).fetchall()

        return [
            {
                "id": row[0],
                "job_id": row[1],
                "level": row[2],
                "message": row[3],
                "data": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
            }
            for row in rows
        ]


    def mark_notifications_read(self, notification_ids):
        if not notification_ids:
            return

        placeholders = ",".join("?" for _ in notification_ids)
        self.conn.execute(
            f"UPDATE notifications SET read=1 WHERE id IN ({placeholders})",
            notification_ids,
        )
        self.conn.commit()

    def write(self, job_id, event_type, message, data=None):
        now = datetime.utcnow().isoformat()

        self.conn.execute(
            """
            INSERT INTO events (
                job_id, event_type, message, data, read, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                event_type=excluded.event_type,
                message=excluded.message,
                data=excluded.data,
                read=0,
                updated_at=excluded.updated_at
            """,
            (
                job_id,
                event_type,
                message,
                json.dumps(data or {}, default=str),
                0,
                now,
                now,
            ),
        )
        self.conn.commit()

    def unread(self):
        rows = self.conn.execute(
            """
            SELECT id, job_id, event_type, message, data, read, created_at, updated_at
            FROM events
            WHERE read=0
            ORDER BY updated_at ASC
            """
        ).fetchall()

        return [
            {
                "id": row[0],
                "job_id": row[1],
                "event_type": row[2],
                "message": row[3],
                "data": json.loads(row[4]) if row[4] else {},
                "read": bool(row[5]),
                "created_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def mark_read(self, event_ids):
        if not event_ids:
            return

        placeholders = ",".join("?" for _ in event_ids)
        self.conn.execute(
            f"UPDATE events SET read=1 WHERE id IN ({placeholders})",
            event_ids,
        )
        self.conn.commit()

    def all_for_job(self, job_id):
        rows = self.conn.execute(
            """
            SELECT id, job_id, event_type, message, data, read, created_at
            FROM events
            WHERE job_id=?
            ORDER BY id ASC
            """,
            (job_id,),
        ).fetchall()

        return [
            {
                "id": row[0],
                "job_id": row[1],
                "event_type": row[2],
                "message": row[3],
                "data": json.loads(row[4]) if row[4] else {},
                "read": bool(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]