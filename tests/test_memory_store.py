import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.brain.memory.store import (
    forget_memory,
    get_memory_file_path,
    list_memories,
    recall_memory,
    save_memory,
)


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.temp_dir = self.temp_root / f"memory-{uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        self.memory_file = self.temp_dir / "memory.json"

    def test_save_and_recall_value(self) -> None:
        result = save_memory("favorite_ide", "VS Code", self.memory_file)
        self.assertEqual(result, "Saved memory for 'favorite_ide'.")
        self.assertEqual(recall_memory("favorite_ide", self.memory_file), "VS Code")

    def test_update_existing_value_preserves_created_at(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        first = self.memory_file.read_text(encoding="utf-8")
        self.assertIn("VS Code", first)

        result = save_memory("favorite_ide", "PyCharm", self.memory_file)
        self.assertEqual(result, "Updated memory for 'favorite_ide'.")

        data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        entry = data["favorite_ide"]
        self.assertEqual(entry["value"], "PyCharm")
        self.assertIn("created_at", entry)
        self.assertIn("updated_at", entry)
        self.assertEqual(entry["source"], "user")

    def test_forget_value(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        result = forget_memory("favorite_ide", self.memory_file)
        self.assertEqual(result, "Forgot memory for 'favorite_ide'.")
        self.assertEqual(recall_memory("favorite_ide", self.memory_file), "No memory found for 'favorite_ide'.")

    def test_list_memories(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        save_memory("name", "Misha", self.memory_file)
        result = list_memories(self.memory_file)
        self.assertIn("favorite_ide: VS Code", result)
        self.assertIn("name: Misha", result)

    def test_missing_key(self) -> None:
        self.assertEqual(recall_memory("missing", self.memory_file), "No memory found for 'missing'.")

    def test_malformed_json_recovers_safely(self) -> None:
        self.memory_file.write_text("{not valid json", encoding="utf-8")
        self.assertEqual(recall_memory("favorite_ide", self.memory_file), "No memory found for 'favorite_ide'.")
        self.assertEqual(save_memory("favorite_ide", "VS Code", self.memory_file), "Saved memory for 'favorite_ide'.")

    def test_normalizes_keys_and_rejects_empty_values(self) -> None:
        self.assertEqual(save_memory(" favorite_ide ", "VS Code", self.memory_file), "Saved memory for 'favorite_ide'.")
        self.assertEqual(recall_memory("FAVORITE_IDE", self.memory_file), "VS Code")
        self.assertEqual(save_memory("name", "   ", self.memory_file), "Please provide a non-empty memory value.")
        self.assertEqual(save_memory("   ", "VS Code", self.memory_file), "Please provide a non-empty memory key.")

    def test_default_path_uses_repository_data_folder(self) -> None:
        path = get_memory_file_path()
        self.assertTrue(path.name == "memory.json")
        self.assertTrue(path.parent.name == "data")


if __name__ == "__main__":
    unittest.main()
