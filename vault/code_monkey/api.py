from typing import Any, Dict

from .build_manager import BuildManager


def submit(goal: str) -> Dict[str, Any]:
    return BuildManager().submit(goal)


def plan(task_id: str) -> Dict[str, Any]:
    return BuildManager().plan(task_id)


def build(task_id: str) -> Dict[str, Any]:
    return BuildManager().build(task_id)


def submit_and_plan(goal: str) -> Dict[str, Any]:
    return BuildManager().submit_and_plan(goal)


def submit_and_build(goal: str) -> Dict[str, Any]:
    return BuildManager().submit_and_build(goal)


def status(task_id: str) -> Dict[str, Any]:
    return BuildManager().status(task_id)


def list_tasks() -> Dict[str, Any]:
    return BuildManager().list_tasks()


def events(task_id: str) -> Dict[str, Any]:
    return BuildManager().events(task_id)


def session(task_id: str) -> Dict[str, Any]:
    return BuildManager().session(task_id)
