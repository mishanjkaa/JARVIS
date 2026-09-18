from __future__ import annotations

import json
from pathlib import Path

from app.brain.plugins.models import PluginManifest


def read_manifest(path: Path) -> PluginManifest:
    if path.stat().st_size > 16_384:
        raise ValueError("manifest too large")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"plugin_id", "display_name", "version", "minimum_jarvis_version", "tools", "risk_levels"}:
        raise ValueError("manifest schema rejected")
    if not isinstance(payload["tools"], list) or not all(isinstance(item, str) for item in payload["tools"]):
        raise ValueError("manifest tools rejected")
    return PluginManifest(payload["plugin_id"], payload["display_name"], payload["version"], payload["minimum_jarvis_version"], tuple(payload["tools"]), payload["risk_levels"])

import json
from pathlib import Path
from typing import Any


class PluginManifestError(ValueError):
    """Raised when a plugin manifest is invalid."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise PluginManifestError("manifest missing")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PluginManifestError("manifest must be an object")
    if not isinstance(data.get("plugin_id"), str) or not data.get("plugin_id", "").strip():
        raise PluginManifestError("plugin_id invalid")
    if not isinstance(data.get("display_name"), str) or not data.get("display_name", "").strip():
        raise PluginManifestError("display_name invalid")
    if not isinstance(data.get("version"), str) or not data.get("version", "").strip():
        raise PluginManifestError("version invalid")
    if not isinstance(data.get("min_jarvis_version"), str) or not data.get("min_jarvis_version", "").strip():
        raise PluginManifestError("min_jarvis_version invalid")
    if not isinstance(data.get("tools"), list):
        raise PluginManifestError("tools invalid")
    if not isinstance(data.get("risk_levels"), list):
        raise PluginManifestError("risk_levels invalid")
    return data
