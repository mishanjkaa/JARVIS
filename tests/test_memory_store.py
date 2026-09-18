import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.brain.memory.store import (
    CATEGORY_LEARNED_PATTERN,
    CATEGORY_PREFERENCE,
    SOURCE_AI_PROPOSED,
    SOURCE_USER,
    find_similar_keys,
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

    def test_category_defaults_to_fact_for_backward_compatible_calls(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        self.assertEqual(data["favorite_ide"]["category"], "fact")

    def test_category_can_be_set_to_preference(self) -> None:
        save_memory("theme", "dark", self.memory_file, category=CATEGORY_PREFERENCE)
        data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        self.assertEqual(data["theme"]["category"], CATEGORY_PREFERENCE)

    def test_rejects_invalid_category(self) -> None:
        result = save_memory("theme", "dark", self.memory_file, category="not_a_real_category")
        self.assertEqual(result, "Please provide a valid memory category.")
        self.assertEqual(recall_memory("theme", self.memory_file), "No memory found for 'theme'.")

    def test_source_defaults_to_user(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        self.assertEqual(data["favorite_ide"]["source"], SOURCE_USER)

    def test_source_ai_proposed_is_recorded_separately_from_user(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file, source=SOURCE_USER)
        save_memory("build_flag", "release", self.memory_file, source=SOURCE_AI_PROPOSED)
        data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        self.assertEqual(data["favorite_ide"]["source"], SOURCE_USER)
        self.assertEqual(data["build_flag"]["source"], SOURCE_AI_PROPOSED)

    def test_learned_pattern_capture_can_be_disabled(self) -> None:
        result = save_memory(
            "retry_pattern",
            "back off after 3 failures",
            self.memory_file,
            category=CATEGORY_LEARNED_PATTERN,
            source=SOURCE_AI_PROPOSED,
            learned_capture_enabled=False,
        )
        self.assertEqual(result, "Learned-pattern memory capture is disabled.")
        self.assertEqual(recall_memory("retry_pattern", self.memory_file), "No memory found for 'retry_pattern'.")

    def test_learned_pattern_capture_allowed_when_enabled(self) -> None:
        result = save_memory(
            "retry_pattern",
            "back off after 3 failures",
            self.memory_file,
            category=CATEGORY_LEARNED_PATTERN,
            source=SOURCE_AI_PROPOSED,
            learned_capture_enabled=True,
        )
        self.assertEqual(result, "Saved memory for 'retry_pattern'.")

    def test_max_entries_soft_cap_blocks_new_keys_but_allows_updates(self) -> None:
        save_memory("key_one", "value one", self.memory_file, max_entries=1)
        blocked = save_memory("key_two", "value two", self.memory_file, max_entries=1)
        self.assertEqual(blocked, "Memory limit reached (1 entries). Forget an existing memory before adding a new one.")
        self.assertEqual(recall_memory("key_two", self.memory_file), "No memory found for 'key_two'.")

        updated = save_memory("key_one", "updated value", self.memory_file, max_entries=1)
        self.assertEqual(updated, "Updated memory for 'key_one'.")
        self.assertEqual(recall_memory("key_one", self.memory_file), "updated value")

    def test_list_memories_shows_category_and_source(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        result = list_memories(self.memory_file)
        self.assertIn("favorite_ide: VS Code", result)
        self.assertIn("[fact/user]", result)

    def test_list_memories_filters_by_category_and_source(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file, category="fact", source=SOURCE_USER)
        save_memory(
            "retry_pattern",
            "back off after 3 failures",
            self.memory_file,
            category=CATEGORY_LEARNED_PATTERN,
            source=SOURCE_AI_PROPOSED,
        )
        learned_only = list_memories(self.memory_file, category=CATEGORY_LEARNED_PATTERN, source=SOURCE_AI_PROPOSED)
        self.assertIn("retry_pattern", learned_only)
        self.assertNotIn("favorite_ide", learned_only)

    def test_find_similar_keys_matches_substring_and_prefix(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        save_memory("favorite_food", "pizza", self.memory_file)
        save_memory("unrelated", "value", self.memory_file)
        suggestions = find_similar_keys("favorite", self.memory_file)
        self.assertIn("favorite_ide", suggestions)
        self.assertIn("favorite_food", suggestions)
        self.assertNotIn("unrelated", suggestions)

    def test_find_similar_keys_respects_limit(self) -> None:
        for index in range(5):
            save_memory(f"favorite_{index}", "value", self.memory_file)
        suggestions = find_similar_keys("favorite", self.memory_file, limit=2)
        self.assertEqual(len(suggestions), 2)

    def test_find_similar_keys_returns_empty_for_no_match(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        self.assertEqual(find_similar_keys("zzz_unmatched", self.memory_file), [])

    def test_find_similar_keys_excludes_exact_match(self) -> None:
        save_memory("favorite_ide", "VS Code", self.memory_file)
        self.assertEqual(find_similar_keys("favorite_ide", self.memory_file), [])


if __name__ == "__main__":
    unittest.main()
