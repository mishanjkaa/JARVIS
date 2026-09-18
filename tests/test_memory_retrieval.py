import unittest
from unittest.mock import patch

from app.brain.memory.store import DEFAULT_CATEGORY, SOURCE_AI_PROPOSED
from app.brain.tools.registry import ToolRegistry


class MemoryRetrievalToolTests(unittest.TestCase):
    """Covers the AI-facing tool-call path for memory.remember/memory.recall/memory.forget.

    These exercise app.brain.tools.builtin_tools handlers through the real ToolRegistry
    (the same entry point ToolExecutor uses), with the store layer mocked out so no test
    touches the real app/data/memory.json file or global runtime config state.
    """

    def setUp(self) -> None:
        self.registry = ToolRegistry()

    @patch("app.brain.tools.builtin_tools.get_effective_runtime_config")
    @patch("app.brain.tools.builtin_tools.save_memory")
    def test_remember_tool_tags_source_as_ai_proposed(self, save_memory_mock, config_mock) -> None:
        config_mock.return_value = {"memory_max_entries": 500, "memory_learned_capture_enabled": True}
        save_memory_mock.return_value = "Saved memory for 'project'."
        handler = self.registry.get("memory.remember").handler
        result = handler({"key": "project", "value": "JARVIS"})
        self.assertTrue(result.success)
        save_memory_mock.assert_called_once_with(
            "project",
            "JARVIS",
            category=DEFAULT_CATEGORY,
            source=SOURCE_AI_PROPOSED,
            max_entries=500,
            learned_capture_enabled=True,
        )

    @patch("app.brain.tools.builtin_tools.get_effective_runtime_config")
    @patch("app.brain.tools.builtin_tools.save_memory")
    def test_remember_tool_forwards_explicit_category(self, save_memory_mock, config_mock) -> None:
        config_mock.return_value = {"memory_max_entries": 500, "memory_learned_capture_enabled": True}
        save_memory_mock.return_value = "Saved memory for 'theme'."
        handler = self.registry.get("memory.remember").handler
        handler({"key": "theme", "value": "dark", "category": "preference"})
        save_memory_mock.assert_called_once_with(
            "theme",
            "dark",
            category="preference",
            source=SOURCE_AI_PROPOSED,
            max_entries=500,
            learned_capture_enabled=True,
        )

    @patch("app.brain.tools.builtin_tools.get_effective_runtime_config")
    @patch("app.brain.tools.builtin_tools.save_memory")
    def test_remember_tool_rejects_unknown_category_without_writing(self, save_memory_mock, config_mock) -> None:
        config_mock.return_value = {"memory_max_entries": 500, "memory_learned_capture_enabled": True}
        handler = self.registry.get("memory.remember").handler
        result = handler({"key": "theme", "value": "dark", "category": "not_a_real_category"})
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Invalid memory category.")
        save_memory_mock.assert_not_called()

    @patch("app.brain.tools.builtin_tools.get_effective_runtime_config")
    @patch("app.brain.tools.builtin_tools.save_memory")
    def test_remember_tool_surfaces_learned_capture_disabled_as_failure(self, save_memory_mock, config_mock) -> None:
        config_mock.return_value = {"memory_max_entries": 500, "memory_learned_capture_enabled": False}
        save_memory_mock.return_value = "Learned-pattern memory capture is disabled."
        handler = self.registry.get("memory.remember").handler
        result = handler({"key": "retry_pattern", "value": "back off after 3 failures", "category": "learned_pattern"})
        self.assertFalse(result.success)
        save_memory_mock.assert_called_once_with(
            "retry_pattern",
            "back off after 3 failures",
            category="learned_pattern",
            source=SOURCE_AI_PROPOSED,
            max_entries=500,
            learned_capture_enabled=False,
        )

    @patch("app.brain.tools.builtin_tools.recall_memory")
    def test_recall_tool_contract_is_unchanged_on_a_miss(self, recall_memory_mock) -> None:
        recall_memory_mock.return_value = "No memory found for 'missing'."
        handler = self.registry.get("memory.recall").handler
        result = handler({"key": "missing"})
        self.assertTrue(result.success)
        self.assertEqual(result.message, "No memory found for 'missing'.")
        self.assertNotIn("Did you mean", result.message)

    @patch("app.brain.tools.builtin_tools.forget_memory")
    def test_forget_tool_is_registered_and_delegates_to_the_store(self, forget_memory_mock) -> None:
        forget_memory_mock.return_value = "Forgot memory for 'project'."
        handler = self.registry.get("memory.forget").handler
        result = handler({"key": "project"})
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Forgot memory for 'project'.")
        forget_memory_mock.assert_called_once_with("project")

    def test_forget_tool_is_persistent_write_like_remember(self) -> None:
        remember_definition = self.registry.get("memory.remember")
        forget_definition = self.registry.get("memory.forget")
        self.assertEqual(remember_definition.risk_level, "persistent_write")
        self.assertEqual(forget_definition.risk_level, "persistent_write")

    def test_no_memory_list_tool_is_registered_for_the_ai(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.get("memory.list")


if __name__ == "__main__":
    unittest.main()
