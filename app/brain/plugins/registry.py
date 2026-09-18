from __future__ import annotations

from app.brain.plugins.models import PluginDefinition
from app.brain.plugins.state import get_plugin_state


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginDefinition] = {}
        self._tool_names: set[str] = set()

    def can_register_plugin(self, plugin: PluginDefinition) -> bool:
        return all(tool.get("name") not in self._tool_names for tool in plugin.tools)

    def register_plugin(self, plugin: PluginDefinition) -> bool:
        if not self.can_register_plugin(plugin):
            return False
        self._plugins[plugin.plugin_id] = plugin
        for tool in plugin.tools:
            self._tool_names.add(tool["name"])
        return True

    def enable(self, plugin_id: str) -> bool:
        state = get_plugin_state()
        if plugin_id not in state.enabled_trusted_plugin_ids:
            state.enabled_trusted_plugin_ids.append(plugin_id)
        state.load_statuses[plugin_id] = "enabled"
        return True

    def disable(self, plugin_id: str) -> None:
        state = get_plugin_state()
        state.enabled_trusted_plugin_ids = [value for value in state.enabled_trusted_plugin_ids if value != plugin_id]
        state.load_statuses[plugin_id] = "disabled"


_REGISTRY = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    return _REGISTRY
