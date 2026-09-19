from __future__ import annotations

from typing import Any

from app.brain.configuration.config_writer import write_config
from app.brain.configuration.runtime_config import get_effective_runtime_config, replace_runtime_config, reset_runtime_config
from app.brain.configuration.runtime_config import validate_change
from app.brain.vision.state import invalidate_vision_readiness_cache
from config.config_loader import get_config_path, load_config

# RFC-010: location_shared_secret is a credential, not a tunable setting. It is excluded
# from runtime_config.MUTABLE_KEYS (so it can never be set via `config set`, only by editing
# config/config.json directly) and redacted here so `config show`/`config get` never echo it
# back in plaintext. vision_gemini_api_key (2026-09-19 "screen understanding" discussion,
# see config/settings.py) is the same shape of credential and gets the same treatment.
_REDACTED_CONFIG_KEYS = {"location_shared_secret", "vision_gemini_api_key"}


def _effective_config() -> dict[str, Any]:
    return get_effective_runtime_config()


def config_show() -> str:
    config = _effective_config()
    return "\n".join(f"{key}: {'<redacted>' if key in _REDACTED_CONFIG_KEYS else config[key]}" for key in sorted(config))


def config_get(key: str) -> str:
    config = _effective_config()
    if key not in config:
        return "Configuration key not found."
    if key in _REDACTED_CONFIG_KEYS:
        return "<redacted>"
    return str(config[key])


def config_set(key: str, value: Any) -> str:
    try:
        validated_value = validate_change(key, value)
    except ValueError:
        return "Configuration change rejected."
    merged = get_effective_runtime_config()
    config_path = get_config_path()
    candidate = load_config(config_path)
    candidate[key] = validated_value
    if not write_config(config_path, candidate, backup_path=config_path.with_name("config.backup.json")):
        return "Configuration change rejected."
    reloaded = load_config(config_path)
    if reloaded.get(key) != candidate.get(key):
        return "Configuration change rejected."
    merged[key] = reloaded[key]
    replace_runtime_config(merged, status="updated")
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
