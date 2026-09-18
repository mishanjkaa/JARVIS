import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.brain.planner.tasks import (
    add_task,
    complete_task,
    delete_task,
    list_tasks,
)


class TasksTests(unittest.TestCase):
    def _temp_dir(self) -> Path:
        temp_root = Path.cwd() / ".tmp-tests"
        temp_root.mkdir(exist_ok=True)
        temp_dir = temp_root / f"tasks-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def test_add_and_list_tasks(self) -> None:
        tasks_path = self._temp_dir() / "tasks.json"
        task_id = add_task("buy milk", tasks_path)
        self.assertEqual(task_id, 1)
        self.assertEqual(list_tasks(tasks_path), "1. buy milk [pending]")

    def test_complete_and_delete_task(self) -> None:
        tasks_path = self._temp_dir() / "tasks.json"
        add_task("read docs", tasks_path)
        complete_task(1, tasks_path)
        self.assertIn("[done]", list_tasks(tasks_path))
        self.assertEqual(delete_task(1, tasks_path), "Deleted task 1.")

    def test_malformed_json_is_recovered(self) -> None:
        tasks_path = self._temp_dir() / "tasks.json"
        tasks_path.write_text("{not valid json", encoding="utf-8")
        self.assertEqual(add_task("recovered", tasks_path), 1)
        self.assertEqual(list_tasks(tasks_path), "1. recovered [pending]")
