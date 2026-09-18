from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import DEFAULT_SETTINGS

_CONFIG_PATH_OVERRIDE: Path | None = None


def get_config_path() -> Path:
    if _CONFIG_PATH_OVERRIDE is not None:
        return _CONFIG_PATH_OVERRIDE
    return Path(__file__).resolve().parent / "config.json"


def set_config_path_override(path: Path | str | None) -> None:
    global _CONFIG_PATH_OVERRIDE
    _CONFIG_PATH_OVERRIDE = Path(path).resolve() if path is not None else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path).resolve() if path is not None else get_config_path()
    raw = _load_json(path)
    merged = dict(DEFAULT_SETTINGS)
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool, list)) or value is None:
            merged[key] = value
    if not isinstance(merged.get("enabled_plugin_ids"), list) or not all(isinstance(item, str) for item in merged["enabled_plugin_ids"]):
        merged["enabled_plugin_ids"] = list(DEFAULT_SETTINGS["enabled_plugin_ids"])
    numeric_ranges = {
        "plan_approval_timeout_seconds": (1, 3600),
        "max_conversation_entries": (1, 100),
        "max_timeline_entries": (1, 1000),
        "max_persistent_writes_per_plan": (0, 2),
        "max_external_actions_per_plan": (0, 2),
        "agent_max_steps": (1, 20),
        "agent_result_size_limit": (40, 2000),
        "filesystem_max_file_size": (1, 4_194_304),
        "filesystem_max_read_size": (1, 1_048_576),
        "filesystem_max_write_size": (1, 1_048_576),
        "terminal_timeout_seconds": (1, 300),
        "terminal_max_stdout_chars": (100, 200000),
        "terminal_max_stderr_chars": (100, 200000),
        "terminal_history_limit": (1, 100),
        "browser_navigation_timeout_seconds": (1, 300),
        "browser_extract_text_max_chars": (100, 50000),
        "browser_max_elements": (1, 200),
        "vision_timeout_seconds": (1, 300),
        "vision_max_file_size": (1, 16_777_216),
        "vision_max_width": (1, 16_384),
        "vision_max_height": (1, 16_384),
        "vision_max_pixels": (1, 16_777_216),
        "vision_max_ocr_chars": (1, 20_000),
        "vision_max_regions": (1, 50),
        "vision_evidence_retention_seconds": (1, 3600),
        "vision_browser_capture_ttl_seconds": (1, 3600),
        "vision_browser_capture_max_bytes": (1, 16_777_216),
        "vision_browser_capture_max_width": (1, 4096),
        "vision_browser_capture_max_height": (1, 4096),
        "vision_browser_capture_max_pixels": (1, 16_777_216),
        "vision_browser_capture_min_candidate_width_pixels": (1, 2000),
        "vision_browser_capture_min_candidate_height_pixels": (1, 2000),
        "vision_browser_capture_min_candidate_area_pixels": (1, 4_000_000),
        "vision_browser_capture_verification_context_scale_percent": (100, 1000),
        "vision_browser_capture_verification_context_min_width_pixels": (1, 4000),
        "vision_browser_capture_verification_context_min_height_pixels": (1, 4000),
        "vision_browser_capture_verification_context_max_area_percent": (1, 100),
        "intelligence_timeout_seconds": (1, 300),
        "intelligence_max_plan_steps": (1, 20),
        "intelligence_max_context_chars": (1000, 100000),
        "intelligence_max_recent_messages": (1, 50),
        "intelligence_max_planning_attempts": (1, 4),
        "memory_max_entries": (1, 100_000),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        value = merged.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            merged[key] = DEFAULT_SETTINGS[key]
    return merged
