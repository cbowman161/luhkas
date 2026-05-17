import sqlite3
import json
from datetime import datetime

from config import DB_PATH


class StateStore:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            goal TEXT,
            status TEXT,
            result TEXT,
            created_at TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            data TEXT,
            timestamp TEXT
        )
        """)

        self.conn.commit()

    def create_task(self, task_id, goal):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO tasks (id, goal, status, result, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, goal, "running", None, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def update_task_status(self, task_id, status):
        self.conn.execute(
            "UPDATE tasks SET status=? WHERE id=?",
            (status, task_id),
        )
        self.conn.commit()

    def set_task_result(self, task_id, result):
        status = result.get("status", "completed") if isinstance(result, dict) else "completed"

        if status == "success":
            status = "completed"

        self.conn.execute(
            "UPDATE tasks SET result=?, status=? WHERE id=?",
            (json.dumps(result), status, task_id),
        )
        self.conn.commit()

    def get_task(self, task_id):
        row = self.conn.execute(
            "SELECT id, goal, status, result, created_at FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "goal": row[1],
            "status": row[2],
            "result": json.loads(row[3]) if row[3] else None,
            "created_at": row[4],
        }

    def list_tasks(self):
        rows = self.conn.execute(
            """
            SELECT id, goal, status, result, created_at
            FROM tasks
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [
            {
                "id": row[0],
                "goal": row[1],
                "status": row[2],
                "result": json.loads(row[3]) if row[3] else None,
                "created_at": row[4],
            }
            for row in rows
        ]

    def add_history(self, task_id, data):
        self.conn.execute(
            """
            INSERT INTO task_history (task_id, data, timestamp)
            VALUES (?, ?, ?)
            """,
            (task_id, json.dumps(data), datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def get_history(self, task_id):
        rows = self.conn.execute(
            """
            SELECT data
            FROM task_history
            WHERE task_id=?
            ORDER BY id ASC
            """,
            (task_id,),
        ).fetchall()

        return [json.loads(row[0]) for row in rows]