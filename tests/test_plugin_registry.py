import unittest

from app.brain.plugins.registry import PluginRegistry
from app.brain.plugins.models import PluginDefinition


class PluginRegistryTests(unittest.TestCase):
    def test_duplicate_tool_name_rejected(self) -> None:
        registry = PluginRegistry()
        plugin = PluginDefinition(plugin_id="demo", display_name="Demo", version="1.0", min_jarvis_version="2.5.0", module_path="app.plugins.example_plugin", tools=[{"name": "example.echo", "description": "Echo"}], risk_levels=["read_only"], enabled=True)
        registry.register_plugin(plugin)
        duplicate = PluginDefinition(plugin_id="demo2", display_name="Demo2", version="1.0", min_jarvis_version="2.5.0", module_path="app.plugins.example_plugin", tools=[{"name": "example.echo", "description": "Echo 2"}], risk_levels=["read_only"], enabled=True)
        self.assertFalse(registry.can_register_plugin(duplicate))
