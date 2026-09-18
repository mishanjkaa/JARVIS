from __future__ import annotations

from typing import Any

from app.brain.configuration.config_writer import write_config
from app.brain.configuration.runtime_config import get_effective_runtime_config, replace_runtime_config, reset_runtime_config
from app.brain.configuration.runtime_config import validate_change
from app.brain.vision.state import invalidate_vision_readiness_cache
from config.config_loader import get_config_path, load_config


def _effective_config() -> dict[str, Any]:
    return get_effective_runtime_config()


def config_show() -> str:
    config = _effective_config()
    return "\n".join(f"{key}: {config[key]}" for key in sorted(config))


def config_get(key: str) -> str:
    config = _effective_config()
    if key not in config:
        return "Configuration key not found."
    return str(config[key])


def config_set(key: str, value: Any) -> str:
    try:
        validated_value = validate_change(key, value)
    except ValueError:
        return "Configuration change rejected."
    config_path = get_config_path()
    candidate = load_config(config_path)
    candidate[key] = validated_value
    if not write_config(config_path, candidate, backup_path=config_path.with_name("config.backup.json")):
        return "Configuration change rejected."
    reloaded = load_config(config_path)
    if reloaded.get(key) != candidate.get(key):
        return "Configuration change rejected."
    replace_runtime_config(reloaded, status="updated")
    if key.startswith("vision_"):
        invalidate_vision_readiness_cache()
    return f"Configuration updated: {key}."


def config_reset(key: str) -> str:
    defaults = reset_runtime_config()
    if key not in defaults:
        return "Configuration key not found."
    return f"Configuration reset: {key}."


def config_reload() -> str:
    reset_runtime_config()
    return "Configuration reloaded."
