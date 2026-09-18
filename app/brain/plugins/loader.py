from __future__ import annotations

import importlib
from pathlib import Path

from app.brain.plugins.manifest import read_manifest
from app.brain.plugins.state import set_plugin_status
from app.brain.plugins.trusted_plugins import TRUSTED_PLUGINS
from app.brain.plugins.validation import validate_manifest


def load_trusted_plugin(plugin_id: str):
    trusted = TRUSTED_PLUGINS.get(plugin_id)
    if trusted is None:
        raise ValueError("plugin is not trusted")
    try:
        manifest = read_manifest(Path(__file__).resolve().parents[3] / trusted["manifest"])
        validate_manifest(manifest, plugin_id)
        module = importlib.import_module(trusted["module"])
        declared = set(manifest.tools)
        tools = module.build_tools()
        if {tool.name for tool in tools} != declared:
            raise ValueError("plugin tools do not match manifest")
        set_plugin_status(plugin_id, "loaded", True)
        return tools
    except Exception:
        set_plugin_status(plugin_id, "load_failed")
        raise ValueError("plugin could not be loaded")

import importlib
from typing import Any

from app.brain.plugins.manifest import load_manifest
from app.brain.plugins.models import PluginDefinition, PluginLoadResult
from app.brain.plugins.state import get_plugin_state
from app.brain.plugins.trusted_plugins import TRUSTED_PLUGINS
from app.brain.plugins.validation import validate_plugin_definition


def load_plugin(plugin_id: str) -> PluginLoadResult:
    if plugin_id not in TRUSTED_PLUGINS:
        return PluginLoadResult(plugin_id=plugin_id, loaded=False, status="unknown")
    state = get_plugin_state()
    if plugin_id not in state.enabled_trusted_plugin_ids:
        return PluginLoadResult(plugin_id=plugin_id, loaded=False, status="disabled")
    metadata = TRUSTED_PLUGINS[plugin_id]
    manifest_path = metadata["manifest"]
    manifest = load_manifest(manifest_path)
    definition = PluginDefinition(
        plugin_id=manifest["plugin_id"],
        display_name=manifest["display_name"],
        version=manifest["version"],
        min_jarvis_version=manifest["min_jarvis_version"],
        module_path=metadata["module"],
        tools=manifest.get("tools", []),
        risk_levels=manifest.get("risk_levels", []),
        enabled=True,
    )
    try:
        validate_plugin_definition(definition)
        module = importlib.import_module(definition.module_path)
        if not hasattr(module, "register"):
            raise AttributeError("register")
        state.loaded_plugin_ids.append(plugin_id)
        state.load_statuses[plugin_id] = "loaded"
        return PluginLoadResult(plugin_id=plugin_id, loaded=True, status="loaded")
    except Exception:
        state.load_statuses[plugin_id] = "failed"
        return PluginLoadResult(plugin_id=plugin_id, loaded=False, status="failed")
