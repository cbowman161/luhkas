from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class BuildTask:
    task_id: str
    goal: str
    state: str
    workspace: str
    message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkOrder:
    goal: str
    capability_name: str
    entrypoint: str
    files: List[Dict[str, str]]
    test_command: str
    self_test_command: str
    success_criteria: List[str]
    notes: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedFile:
    path: str
    content: str


@dataclass
class BuildFiles:
    files: List[GeneratedFile]
    notes: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'files': [{'path': f.path, 'content': f.content} for f in self.files],
            'notes': self.notes,
        }
