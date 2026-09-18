from __future__ import annotations

from app.brain.plugins.models import PluginManifest


def validate_manifest(manifest: PluginManifest, trusted_id: str) -> None:
    if manifest.plugin_id != trusted_id or not manifest.display_name or len(manifest.tools) > 20:
        raise ValueError("plugin manifest rejected")
    if any(not isinstance(name, str) or not name.startswith(trusted_id + ".") for name in manifest.tools):
        raise ValueError("plugin tool name rejected")
    if set(manifest.risk_levels) != set(manifest.tools) or any(value != "read_only" for value in manifest.risk_levels.values()):
        raise ValueError("plugin risk declaration rejected")

from typing import Any

from app.brain.plugins.models import PluginDefinition


def validate_plugin_definition(definition: PluginDefinition) -> None:
    if not isinstance(definition, PluginDefinition):
        raise ValueError("plugin definition invalid")
    if not definition.plugin_id or not definition.display_name:
        raise ValueError("plugin metadata invalid")
    if definition.module_path.startswith(".") or definition.module_path.startswith("/"):
        raise ValueError("module path invalid")
    if not definition.tools:
        raise ValueError("plugin must declare tools")
    for tool in definition.tools:
        if not isinstance(tool, dict) or "name" not in tool or "description" not in tool:
            raise ValueError("tool declaration invalid")
