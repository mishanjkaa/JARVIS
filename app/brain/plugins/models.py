from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginManifest:
    plugin_id: str
    display_name: str
    version: str
    minimum_jarvis_version: str
    tools: tuple[str, ...]
    risk_levels: dict[str, str]


@dataclass
class PluginDefinition:
    plugin_id: str
    display_name: str
    version: str
    min_jarvis_version: str
    module_path: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    risk_levels: list[str] = field(default_factory=list)
    enabled: bool = False


@dataclass
class PluginLoadResult:
    plugin_id: str
    loaded: bool
    status: str
