from __future__ import annotations

import os
import sqlite3
import threading
import time


class TelemetryLogger:
    def __init__(self, db_path: str = "config/telemetry.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                ts   REAL NOT NULL,
                v    REAL,
                ax   REAL, ay REAL, az REAL,
                gx   REAL, gy REAL, gz REAL,
                mx   REAL, my REAL, mz REAL,
                odl  REAL, odr REAL,
                L    REAL, R   REAL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts ON telemetry(ts)"
        )
        self._conn.commit()

    def log(self, tel: dict):
        with self._lock:
            self._conn.execute(
                "INSERT INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    tel.get("v"),
                    tel.get("ax"), tel.get("ay"), tel.get("az"),
                    tel.get("gx"), tel.get("gy"), tel.get("gz"),
                    tel.get("mx"), tel.get("my"), tel.get("mz"),
                    tel.get("odl"), tel.get("odr"),
                    tel.get("L"), tel.get("R"),
                ),
            )
            self._conn.commit()

    def recent(self, seconds: float = 60.0) -> list:
        cols = ["ts","v","ax","ay","az","gx","gy","gz","mx","my","mz","odl","odr","L","R"]
        cutoff = time.time() - seconds
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM telemetry WHERE ts > ? ORDER BY ts", (cutoff,)
            ).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        with self._lock:
            self._conn.close()
