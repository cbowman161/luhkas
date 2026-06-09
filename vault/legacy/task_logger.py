import json
import os
import traceback
from datetime import datetime

from config import TASK_LOG_DIR


class TaskLogger:
    def __init__(self, root=TASK_LOG_DIR):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def path_for(self, task_id):
        safe = str(task_id or "unknown").replace("/", "_")
        return os.path.join(self.root, f"{safe}.log")

    def write(self, task_id, message, data=None):
        path = self.path_for(task_id)
        record = {
            "time": datetime.utcnow().isoformat(),
            "task_id": task_id,
            "message": message,
            "data": data,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def exception(self, task_id, message):
        self.write(task_id, message, {"traceback": traceback.format_exc()})
