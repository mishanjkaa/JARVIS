from __future__ import annotations

from typing import Any

from app.brain.configuration.state import get_runtime_config_state, reset_runtime_config_state

MUTABLE_KEYS = {
    "assistant_name", "language", "voice_enabled", "ai_enabled", "ollama_model",
    "ai_timeout_seconds", "ai_max_plan_steps", "ai_allow_conversation", "show_plan_preview",
    "agent_enabled", "agent_max_steps", "agent_result_size_limit", "agent_allow_persistent_actions",
    "filesystem_enabled", "filesystem_max_file_size", "filesystem_max_read_size", "filesystem_max_write_size", "filesystem_soft_delete", "filesystem_allow_full_disk_access",
    "developer_mode", "auto_execute_low_risk", "auto_execute_medium_project",
    "terminal_enabled", "terminal_default_working_directory", "terminal_timeout_seconds",
    "terminal_max_stdout_chars", "terminal_max_stderr_chars", "terminal_history_limit",
    "terminal_allow_python", "terminal_allow_git_read_only", "terminal_allow_package_install_with_approval",
    "browser_enabled", "browser_backend", "browser_headless_default", "browser_navigation_timeout_seconds",
    "browser_extract_text_max_chars", "browser_max_elements", "browser_allow_http", "browser_screenshot_overwrite",
    "vision_enabled", "vision_provider", "vision_model", "vision_ollama_base_url", "vision_timeout_seconds",
    "vision_max_file_size", "vision_max_width", "vision_max_height", "vision_max_pixels", "vision_max_ocr_chars",
    "vision_max_regions", "vision_evidence_retention_seconds", "vision_browser_capture_enabled",
    "vision_browser_capture_ttl_seconds", "vision_browser_capture_max_bytes", "vision_browser_capture_max_width",
    "vision_browser_capture_max_height", "vision_browser_capture_max_pixels",
    "vision_browser_capture_min_candidate_width_pixels", "vision_browser_capture_min_candidate_height_pixels",
    "vision_browser_capture_min_candidate_area_pixels",
    "vision_browser_capture_verification_context_scale_percent", "vision_browser_capture_verification_context_min_width_pixels",
    "vision_browser_capture_verification_context_min_height_pixels", "vision_browser_capture_verification_context_max_area_percent",
    "intelligence_enabled", "intelligence_provider", "intelligence_model", "intelligence_timeout_seconds",
    "intelligence_max_plan_steps", "intelligence_max_context_chars", "intelligence_max_recent_messages",
    "intelligence_require_structured_output", "intelligence_allow_goal_evaluation",
    "intelligence_allow_heuristic_fallback", "intelligence_fail_closed", "intelligence_max_planning_attempts",
    "memory_max_entries", "memory_learned_capture_enabled",
    "vision_desktop_capture_enabled", "vision_desktop_capture_ttl_seconds",
    "vision_desktop_capture_max_bytes", "vision_desktop_capture_max_width",
    "vision_desktop_capture_max_height", "vision_desktop_capture_max_pixels",
    # RFC-010: location_shared_secret is deliberately excluded from this allowlist. It is a
    # credential, not a tunable setting, so it can only be set by editing config/config.json
    # directly and is redacted (never echoed) by config_show()/config_get().
    "location_enabled", "location_bind_host", "location_port", "location_stale_after_seconds", "osrm_base_url",
    "voice_stt_model", "voice_tts_voice", "voice_require_speaker_verification", "voice_verification_threshold",
    "voice_input_device", "voice_input_sample_rate", "voice_input_channels",
    "voice_talk_auto_stop_on_silence", "voice_talk_silence_timeout_seconds", "voice_talk_min_duration_seconds",
    "voice_listen_on_startup",
}


def validate_change(key: str, value: object) -> object:
    if key not in MUTABLE_KEYS:
        raise ValueError("configuration key is not allowlisted")
    if key in {"voice_enabled", "voice_require_speaker_verification", "voice_talk_auto_stop_on_silence", "voice_listen_on_startup", "ai_enabled", "ai_allow_conversation", "show_plan_preview", "agent_enabled", "agent_allow_persistent_actions", "filesystem_enabled", "filesystem_soft_delete", "filesystem_allow_full_disk_access", "developer_mode", "auto_execute_low_risk", "auto_execute_medium_project", "terminal_enabled", "terminal_allow_python", "terminal_allow_git_read_only", "terminal_allow_package_install_with_approval", "browser_enabled", "browser_headless_default", "browser_allow_http", "browser_screenshot_overwrite", "vision_enabled", "vision_browser_capture_enabled", "intelligence_enabled", "intelligence_require_structured_output", "intelligence_allow_goal_evaluation", "intelligence_allow_heuristic_fallback", "intelligence_fail_closed", "memory_learned_capture_enabled", "vision_desktop_capture_enabled", "location_enabled"} and not isinstance(value, bool):
        raise ValueError("configuration value has invalid type")
    if key in {"ai_timeout_seconds", "ai_max_plan_steps", "agent_max_steps", "agent_result_size_limit", "filesystem_max_file_size", "filesystem_max_read_size", "filesystem_max_write_size", "terminal_timeout_seconds", "terminal_max_stdout_chars", "terminal_max_stderr_chars", "terminal_history_limit", "browser_navigation_timeout_seconds", "browser_extract_text_max_chars", "browser_max_elements", "vision_timeout_seconds", "vision_max_file_size", "vision_max_width", "vision_max_height", "vision_max_pixels", "vision_max_ocr_chars", "vision_max_regions", "vision_evidence_retention_seconds", "vision_browser_capture_ttl_seconds", "vision_browser_capture_max_bytes", "vision_browser_capture_max_width", "vision_browser_capture_max_height", "vision_browser_capture_max_pixels", "vision_browser_capture_min_candidate_width_pixels", "vision_browser_capture_min_candidate_height_pixels", "vision_browser_capture_min_candidate_area_pixels", "vision_browser_capture_verification_context_scale_percent", "vision_browser_capture_verification_context_min_width_pixels", "vision_browser_capture_verification_context_min_height_pixels", "vision_browser_capture_verification_context_max_area_percent", "intelligence_timeout_seconds", "intelligence_max_plan_steps", "intelligence_max_context_chars", "intelligence_max_recent_messages", "intelligence_max_planning_attempts", "memory_max_entries", "vision_desktop_capture_ttl_seconds", "vision_desktop_capture_max_bytes", "vision_desktop_capture_max_width", "vision_desktop_capture_max_height", "vision_desktop_capture_max_pixels", "location_port", "location_stale_after_seconds"} and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError("configuration value has invalid range")
    if key in {"assistant_name", "language", "ollama_model", "terminal_default_working_directory", "browser_backend", "vision_provider", "vision_ollama_base_url", "intelligence_provider", "intelligence_model", "osrm_base_url", "voice_stt_model"} and (not isinstance(value, str) or not value.strip() or len(value) > 260):
        raise ValueError("configuration value has invalid type")
    if key == "vision_model" and (not isinstance(value, str) or len(value) > 260):
        raise ValueError("configuration value has invalid type")
    if key in {"location_bind_host", "voice_tts_voice"} and (not isinstance(value, str) or len(value) > 260):
        raise ValueError("configuration value has invalid type")
    if key == "voice_verification_threshold" and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0):
        raise ValueError("configuration value has invalid range")
    if key == "voice_input_device" and (not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 255):
        raise ValueError("configuration value has invalid range")
    if key == "voice_input_sample_rate" and (not isinstance(value, int) or isinstance(value, bool) or not 8000 <= value <= 192000):
        raise ValueError("configuration value has invalid range")
    if key == "voice_input_channels" and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 8):
        raise ValueError("configuration value has invalid range")
    if key in {"voice_talk_silence_timeout_seconds", "voice_talk_min_duration_seconds"} and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.2 <= float(value) <= 10.0):
        raise ValueError("configuration value has invalid range")
    return value
_ALLOWED_KEYS = {
    "assistant_name",
    "language",
    "voice_enabled",
    "ai_enabled",
    "ollama_model",
    "ai_timeout_seconds",
    "ai_max_plan_steps",
    "ai_allow_conversation",
    "show_plan_preview",
    "agent_enabled",
    "agent_max_steps",
    "agent_result_size_limit",
    "agent_allow_persistent_actions",
    "filesystem_enabled",
    "filesystem_max_file_size",
    "filesystem_max_read_size",
    "filesystem_max_write_size",
    "filesystem_soft_delete",
    "filesystem_allow_full_disk_access",
    "developer_mode",
    "auto_execute_low_risk",
    "auto_execute_medium_project",
    "terminal_enabled",
    "terminal_default_working_directory",
    "terminal_timeout_seconds",
    "terminal_max_stdout_chars",
    "terminal_max_stderr_chars",
    "terminal_history_limit",
    "terminal_allow_python",
    "terminal_allow_git_read_only",
    "terminal_allow_package_install_with_approval",
    "browser_enabled",
    "browser_backend",
    "browser_headless_default",
    "browser_navigation_timeout_seconds",
    "browser_extract_text_max_chars",
    "browser_max_elements",
    "browser_allow_http",
    "browser_screenshot_overwrite",
    "vision_enabled",
    "vision_provider",
    "vision_model",
    "vision_ollama_base_url",
    "vision_timeout_seconds",
    "vision_max_file_size",
    "vision_max_width",
    "vision_max_height",
    "vision_max_pixels",
    "vision_max_ocr_chars",
    "vision_max_regions",
    "vision_evidence_retention_seconds",
    "vision_browser_capture_enabled",
    "vision_browser_capture_ttl_seconds",
    "vision_browser_capture_max_bytes",
    "vision_browser_capture_max_width",
    "vision_browser_capture_max_height",
    "vision_browser_capture_max_pixels",
    "vision_browser_capture_min_candidate_width_pixels",
    "vision_browser_capture_min_candidate_height_pixels",
    "vision_browser_capture_min_candidate_area_pixels",
    "vision_browser_capture_verification_context_scale_percent",
    "vision_browser_capture_verification_context_min_width_pixels",
    "vision_browser_capture_verification_context_min_height_pixels",
    "vision_browser_capture_verification_context_max_area_percent",
    "intelligence_enabled",
    "intelligence_provider",
    "intelligence_model",
    "intelligence_timeout_seconds",
    "intelligence_max_plan_steps",
    "intelligence_max_context_chars",
    "intelligence_max_recent_messages",
    "intelligence_require_structured_output",
    "intelligence_allow_goal_evaluation",
    "intelligence_allow_heuristic_fallback",
    "intelligence_fail_closed",
    "intelligence_max_planning_attempts",
    "memory_max_entries",
    "memory_learned_capture_enabled",
    "vision_desktop_capture_enabled",
    "vision_desktop_capture_ttl_seconds",
    "vision_desktop_capture_max_bytes",
    "vision_desktop_capture_max_width",
    "vision_desktop_capture_max_height",
    "vision_desktop_capture_max_pixels",
    "location_enabled",
    "location_bind_host",
    "location_port",
    "location_stale_after_seconds",
    "osrm_base_url",
    "voice_stt_model",
    "voice_tts_voice",
    "voice_require_speaker_verification",
    "voice_verification_threshold",
    "voice_input_device",
    "voice_input_sample_rate",
    "voice_input_channels",
    "voice_talk_auto_stop_on_silence",
    "voice_talk_silence_timeout_seconds",
    "voice_talk_min_duration_seconds",
    "voice_listen_on_startup",
}


def get_runtime_config() -> dict[str, Any]:
    state = get_runtime_config_state()
    if state.snapshot:
        return dict(state.snapshot)
    return {
        "assistant_name": "JARVIS",
        "language": "en",
        "voice_enabled": True,
        "ai_enabled": True,
        "ollama_model": "llama3.2",
        "ai_timeout_seconds": 20,
        "ai_max_plan_steps": 5,
        "ai_allow_conversation": True,
        "show_plan_preview": True,
        "agent_enabled": True,
        "agent_max_steps": 10,
        "agent_result_size_limit": 200,
        "agent_allow_persistent_actions": True,
        "filesystem_enabled": True,
        "filesystem_max_file_size": 1048576,
        "filesystem_max_read_size": 65536,
        "filesystem_max_write_size": 65536,
        "filesystem_soft_delete": True,
        "filesystem_allow_full_disk_access": False,
        "developer_mode": False,
        "auto_execute_low_risk": True,
        "auto_execute_medium_project": True,
        "terminal_enabled": True,
        "terminal_default_working_directory": ".",
        "terminal_timeout_seconds": 30,
        "terminal_max_stdout_chars": 20000,
        "terminal_max_stderr_chars": 20000,
        "terminal_history_limit": 25,
        "terminal_allow_python": True,
        "terminal_allow_git_read_only": True,
        "terminal_allow_package_install_with_approval": True,
        "browser_enabled": True,
        "browser_backend": "playwright",
        "browser_headless_default": True,
        "browser_navigation_timeout_seconds": 30,
        "browser_extract_text_max_chars": 4000,
        "browser_max_elements": 40,
        "browser_allow_http": False,
        "browser_screenshot_overwrite": False,
        "vision_enabled": True,
        "vision_provider": "ollama",
        "vision_model": "",
        "vision_ollama_base_url": "http://127.0.0.1:11434",
        "vision_timeout_seconds": 90,
        "vision_max_file_size": 4_194_304,
        "vision_max_width": 4096,
        "vision_max_height": 4096,
        "vision_max_pixels": 4_194_304,
        "vision_max_ocr_chars": 4000,
        "vision_max_regions": 8,
        "vision_evidence_retention_seconds": 300,
        "vision_browser_capture_enabled": True,
        "vision_browser_capture_ttl_seconds": 180,
        "vision_browser_capture_max_bytes": 2_000_000,
        "vision_browser_capture_max_width": 1920,
        "vision_browser_capture_max_height": 1080,
        "vision_browser_capture_max_pixels": 2_073_600,
        "vision_browser_capture_min_candidate_width_pixels": 12,
        "vision_browser_capture_min_candidate_height_pixels": 12,
        "vision_browser_capture_min_candidate_area_pixels": 144,
        "vision_browser_capture_verification_context_scale_percent": 400,
    "vision_browser_capture_verification_context_min_width_pixels": 320,
    "vision_browser_capture_verification_context_min_height_pixels": 320,
        "vision_browser_capture_verification_context_max_area_percent": 40,
        "intelligence_enabled": True,
        "intelligence_provider": "ollama",
        "intelligence_model": "llama3.2",
        "intelligence_timeout_seconds": 60,
        "intelligence_max_plan_steps": 10,
        "intelligence_max_context_chars": 30000,
        "intelligence_max_recent_messages": 12,
        "intelligence_require_structured_output": True,
        "intelligence_allow_goal_evaluation": True,
        "intelligence_allow_heuristic_fallback": False,
        "intelligence_fail_closed": True,
        "intelligence_max_planning_attempts": 2,
        "memory_max_entries": 500,
        "memory_learned_capture_enabled": True,
        "vision_desktop_capture_enabled": True,
        "vision_desktop_capture_ttl_seconds": 180,
        "vision_desktop_capture_max_bytes": 6_000_000,
        "vision_desktop_capture_max_width": 3840,
        "vision_desktop_capture_max_height": 2160,
        "vision_desktop_capture_max_pixels": 8_294_400,
        "location_enabled": False,
        "location_bind_host": "",
        "location_port": 8766,
        "location_shared_secret": "",
        "location_stale_after_seconds": 120,
        "osrm_base_url": "http://router.project-osrm.org",
        "voice_stt_model": "small",
        "voice_tts_voice": "",
        "voice_require_speaker_verification": False,
        "voice_verification_threshold": 0.4,
        "voice_input_device": 1,
        "voice_input_sample_rate": 44100,
        "voice_input_channels": 4,
        "voice_talk_auto_stop_on_silence": True,
        "voice_talk_silence_timeout_seconds": 1.0,
        "voice_talk_min_duration_seconds": 1.0,
        "voice_listen_on_startup": True,
    }


def get_runtime_config_snapshot() -> dict[str, Any]:
    return dict(get_runtime_config_state().snapshot)


def get_effective_runtime_config() -> dict[str, Any]:
    from config.config_loader import load_config

    config = load_config()
    snapshot = get_runtime_config_state().snapshot
    if snapshot:
        config.update(snapshot)
    return config


def replace_runtime_config(config: dict[str, Any], *, status: str = "updated") -> dict[str, Any]:
    state = get_runtime_config_state()
    state.snapshot = dict(config)
    state.generation += 1
    state.last_safe_status = status
    return dict(state.snapshot)


def set_runtime_config_value(key: str, value: Any) -> dict[str, Any]:
    if key not in _ALLOWED_KEYS:
        raise ValueError("key not allowlisted")
    if key in {"assistant_name", "language", "ollama_model", "terminal_default_working_directory", "browser_backend", "vision_provider", "vision_ollama_base_url", "intelligence_provider", "intelligence_model", "osrm_base_url", "voice_stt_model"} and not isinstance(value, str):
        raise ValueError("invalid value")
    if key in {"vision_model", "location_bind_host", "voice_tts_voice"} and not isinstance(value, str):
        raise ValueError("invalid value")
    if key == "voice_verification_threshold" and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0):
        raise ValueError("invalid value")
    if key == "voice_input_device" and (not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 255):
        raise ValueError("invalid value")
    if key == "voice_input_sample_rate" and (not isinstance(value, int) or isinstance(value, bool) or not 8000 <= value <= 192000):
        raise ValueError("invalid value")
    if key == "voice_input_channels" and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 8):
        raise ValueError("invalid value")
    if key in {"voice_talk_silence_timeout_seconds", "voice_talk_min_duration_seconds"} and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.2 <= float(value) <= 10.0):
        raise ValueError("invalid value")
    if key in {"voice_enabled", "voice_require_speaker_verification", "voice_talk_auto_stop_on_silence", "voice_listen_on_startup", "ai_enabled", "ai_allow_conversation", "show_plan_preview", "agent_enabled", "agent_allow_persistent_actions", "filesystem_enabled", "filesystem_soft_delete", "filesystem_allow_full_disk_access", "developer_mode", "auto_execute_low_risk", "auto_execute_medium_project", "terminal_enabled", "terminal_allow_python", "terminal_allow_git_read_only", "terminal_allow_package_install_with_approval", "browser_enabled", "browser_headless_default", "browser_allow_http", "browser_screenshot_overwrite", "vision_enabled", "vision_browser_capture_enabled", "intelligence_enabled", "intelligence_require_structured_output", "intelligence_allow_goal_evaluation", "intelligence_allow_heuristic_fallback", "intelligence_fail_closed", "memory_learned_capture_enabled", "vision_desktop_capture_enabled", "location_enabled"} and not isinstance(value, bool):
        raise ValueError("invalid value")
    if key in {"ai_timeout_seconds", "ai_max_plan_steps", "agent_max_steps", "agent_result_size_limit", "filesystem_max_file_size", "filesystem_max_read_size", "filesystem_max_write_size", "terminal_timeout_seconds", "terminal_max_stdout_chars", "terminal_max_stderr_chars", "terminal_history_limit", "browser_navigation_timeout_seconds", "browser_extract_text_max_chars", "browser_max_elements", "vision_timeout_seconds", "vision_max_file_size", "vision_max_width", "vision_max_height", "vision_max_pixels", "vision_max_ocr_chars", "vision_max_regions", "vision_evidence_retention_seconds", "vision_browser_capture_ttl_seconds", "vision_browser_capture_max_bytes", "vision_browser_capture_max_width", "vision_browser_capture_max_height", "vision_browser_capture_max_pixels", "vision_browser_capture_min_candidate_width_pixels", "vision_browser_capture_min_candidate_height_pixels", "vision_browser_capture_min_candidate_area_pixels", "vision_browser_capture_verification_context_scale_percent", "vision_browser_capture_verification_context_min_width_pixels", "vision_browser_capture_verification_context_min_height_pixels", "vision_browser_capture_verification_context_max_area_percent", "intelligence_timeout_seconds", "intelligence_max_plan_steps", "intelligence_max_context_chars", "intelligence_max_recent_messages", "intelligence_max_planning_attempts", "memory_max_entries", "vision_desktop_capture_ttl_seconds", "vision_desktop_capture_max_bytes", "vision_desktop_capture_max_width", "vision_desktop_capture_max_height", "vision_desktop_capture_max_pixels", "location_port", "location_stale_after_seconds"} and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError("invalid value")
    config = get_effective_runtime_config()
    config[key] = value
    return replace_runtime_config(config, status="updated")


def reset_runtime_config() -> dict[str, Any]:
    reset_runtime_config_state()
    return get_runtime_config()
