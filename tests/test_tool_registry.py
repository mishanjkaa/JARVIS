import unittest

from app.brain.tools.registry import ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def test_registry_contains_expected_tools(self) -> None:
        registry = ToolRegistry()
        self.assertIn("calculator.calculate", registry.list_names())
        self.assertIn("memory.remember", registry.list_names())
        self.assertIn("filesystem.create_text_file", registry.list_names())
        self.assertIn("terminal.execute", registry.list_names())
        self.assertIn("browser.open_url", registry.list_names())
        self.assertEqual(registry.count() > 0, True)
