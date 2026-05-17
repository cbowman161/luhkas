"""File-backed chat session log used by camera-node UI services."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path


class ChatLog:
    def __init__(self, path: Path, max_entries: int) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries = deque(maxlen=max_entries)
        self._seq = 0

    def add(self, role: str, text: str, source: str = "chat", **meta) -> dict:
        entry = {
            "id": 0,
            "timestamp": time.time(),
            "role": role,
            "source": source,
            "text": str(text),
            "meta": {key: value for key, value in meta.items() if value is not None},
        }
        with self._lock:
            self._seq += 1
            entry["id"] = self._seq
            self._entries.append(entry)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception:
            pass
        return entry

    def snapshot(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            entries = list(self._entries)
        if limit is not None and limit > 0:
            entries = entries[-limit:]
        return entries

    def init_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.add("system", "session_start", source="session", path=str(self.path))
