import threading
import uuid
import sqlite3
import json
from datetime import datetime

from config import DB_PATH


class JobManager:
    def __init__(self, event_log, db_path=DB_PATH):
        self.event_log = event_log
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT,
            result TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """)
        self.conn.commit()

    def create_job(self, title):
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        self.conn.execute(
            """
            INSERT INTO jobs (id, title, status, result, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, title, "queued", None, now, now),
        )
        self.conn.commit()

        self.event_log.write(job_id, "queued", f"Queued job: {title}")
        return job_id

    def update_job(self, job_id, status, result=None):
        self.conn.execute(
            """
            UPDATE jobs
            SET status=?, result=?, updated_at=?
            WHERE id=?
            """,
            (
                status,
                json.dumps(result) if result is not None else None,
                datetime.utcnow().isoformat(),
                job_id,
            ),
        )
        self.conn.commit()

    def start_background(self, title, target, *args, **kwargs):
        job_id = self.create_job(title)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, title, target, args, kwargs),
            daemon=True,
        )
        thread.start()

        return job_id

    def _run_job(self, job_id, title, target, args, kwargs):
        self.update_job(job_id, "running")
        self.event_log.write(job_id, "running", f"Started job: {title}")

        try:
            result = target(job_id, *args, **kwargs)

            self.update_job(job_id, "completed", result)
            self.event_log.write(
                job_id,
                "completed",
                f"Completed job: {title}",
                result,
            )

        except Exception as e:
            result = {"error": str(e)}

            self.update_job(job_id, "failed", result)
            self.event_log.write(
                job_id,
                "failed",
                f"Failed job: {title}: {str(e)}",
                result,
            )

    def list_jobs(self):
        rows = self.conn.execute(
            """
            SELECT id, title, status, result, created_at, updated_at
            FROM jobs
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [
            {
                "id": row[0],
                "title": row[1],
                "status": row[2],
                "result": json.loads(row[3]) if row[3] else None,
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]