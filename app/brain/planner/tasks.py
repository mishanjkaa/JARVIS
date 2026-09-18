import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_tasks_path(tasks_path: Optional[Path] = None) -> Path:
    if tasks_path is not None:
        return tasks_path
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "tasks.json"


def _load_tasks(tasks_path: Path) -> list[dict[str, object]]:
    if not tasks_path.exists():
        return []
    try:
        content = tasks_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    if not content.strip():
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _save_tasks(tasks_path: Path, data: list[dict[str, object]]) -> None:
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_task(text: str, tasks_path: Optional[Path] = None) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    path = _get_tasks_path(tasks_path)
    tasks = _load_tasks(path)
    task_id = len(tasks) + 1
    task = {
        "id": task_id,
        "text": cleaned,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "completed": False,
        "completed_at": None,
    }
    tasks.append(task)
    _save_tasks(path, tasks)
    logger.info("Task added")
    return task_id


def list_tasks(tasks_path: Optional[Path] = None) -> str:
    path = _get_tasks_path(tasks_path)
    tasks = _load_tasks(path)
    if not tasks:
        return "No tasks stored."
    lines = []
    for item in tasks:
        if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("text"), str):
            completed = item.get("completed", False)
            status = "done" if completed else "pending"
            lines.append(f"{item['id']}. {item['text']} [{status}]")
    return "\n".join(lines)


def complete_task(task_id: int, tasks_path: Optional[Path] = None) -> str:
    path = _get_tasks_path(tasks_path)
    tasks = _load_tasks(path)
    for item in tasks:
        if isinstance(item, dict) and item.get("id") == task_id:
            item["completed"] = True
            item["completed_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            _save_tasks(path, tasks)
            logger.info("Task completed")
            return f"Completed task {task_id}."
    return f"Task {task_id} not found."


def delete_task(task_id: int, tasks_path: Optional[Path] = None) -> str:
    path = _get_tasks_path(tasks_path)
    tasks = _load_tasks(path)
    remaining = [item for item in tasks if not (isinstance(item, dict) and item.get("id") == task_id)]
    if len(remaining) == len(tasks):
        return f"Task {task_id} not found."
    _save_tasks(path, remaining)
    logger.info("Task deleted")
    return f"Deleted task {task_id}."
