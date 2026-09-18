from __future__ import annotations

_enabled: set[str] = set()
_loaded: set[str] = set()
_status: dict[str, str] = {}


def set_plugin_status(plugin_id: str, status: str, loaded: bool = False) -> None:
    _status[plugin_id] = status
    if loaded:
        _loaded.add(plugin_id)


def set_enabled_plugins(plugin_ids: set[str]) -> None:
    _enabled.clear()
    _enabled.update(plugin_ids)


def get_plugin_state() -> tuple[set[str], set[str], dict[str, str]]:
    return set(_enabled), set(_loaded), dict(_status)


def reset_plugin_state() -> None:
    _enabled.clear(); _loaded.clear(); _status.clear()

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginState:
    enabled_trusted_plugin_ids: list[str] = field(default_factory=list)
    loaded_plugin_ids: list[str] = field(default_factory=list)
    load_statuses: dict[str, str] = field(default_factory=dict)


STATE = PluginState()


def get_plugin_state() -> PluginState:
    return STATE


def reset_plugin_state() -> None:
    STATE.enabled_trusted_plugin_ids = []
    STATE.loaded_plugin_ids = []
    STATE.load_statuses = {}
