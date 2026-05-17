import json
import uuid
from pathlib import Path
from typing import Any, Dict

from .config import TASKS_DIR


def create_workspace() -> tuple[str, Path]:
    task_id = str(uuid.uuid4())
    root = TASKS_DIR / task_id
    for sub in ['src', 'tests', 'artifacts', 'logs']:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return task_id, root


def ensure_workspace(task_id: str) -> Path:
    root = TASKS_DIR / task_id
    if not root.exists():
        raise FileNotFoundError(f'Workspace not found: {root}')
    return root


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))
