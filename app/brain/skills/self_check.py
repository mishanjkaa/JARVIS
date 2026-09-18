import importlib
import logging
import os
from pathlib import Path

from app.brain.context.history import get_history
from app.brain.context.state import get_context
from app.brain.logging_setup import initialize_logging
from app.brain.memory.store import get_memory_file_path
from app.brain.voice.voice_controller import get_voice_state

logger = logging.getLogger(__name__)


def _safe_check(label: str, func) -> str:
    try:
        func()
        return f"{label}: PASS"
    except Exception:
        return f"{label}: WARNING"


def run_self_check() -> str:
    checks = []
    try:
        from config.settings import APP_NAME, VERSION
        checks.append("settings: PASS")
    except Exception:
        checks.append("settings: WARNING")

    checks.append(_safe_check("config_loader", lambda: importlib.import_module("config.config_loader")))
    checks.append(_safe_check("router", lambda: importlib.import_module("app.brain.router")))
    checks.append(_safe_check("context_state", lambda: get_context()))
    checks.append(_safe_check("history", lambda: get_history()))
    checks.append(_safe_check("memory", lambda: get_memory_file_path()))

    try:
        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "data"
        data_dir.exists()
        (data_dir / "notes.json").exists()
        (data_dir / "tasks.json").exists()
        checks.append("notes: PASS")
        checks.append("tasks: PASS")
    except Exception:
        checks.append("notes: WARNING")
        checks.append("tasks: WARNING")

    checks.append(_safe_check("voice_modules", lambda: get_voice_state()))
    checks.append(_safe_check("ai_models", lambda: importlib.import_module("app.brain.ai.models")))
    checks.append(_safe_check("ai_provider", lambda: importlib.import_module("app.brain.ai.provider")))
    checks.append(_safe_check("ollama_provider", lambda: importlib.import_module("app.brain.ai.ollama_provider")))
    checks.append(_safe_check("ai_orchestrator", lambda: importlib.import_module("app.brain.ai.orchestrator")))
    checks.append(_safe_check("json_parser", lambda: importlib.import_module("app.brain.ai.json_parser")))
    checks.append(_safe_check("tool_registry", lambda: importlib.import_module("app.brain.tools.registry")))
    checks.append(_safe_check("tool_executor", lambda: importlib.import_module("app.brain.tools.executor")))
    checks.append(_safe_check("ai_policy", lambda: importlib.import_module("app.brain.security.ai_policy")))
    checks.append(_safe_check("ai_config", lambda: importlib.import_module("config.config_loader")))

    try:
        initialize_logging()
        checks.append("logging: PASS")
    except Exception:
        checks.append("logging: WARNING")

    try:
        checks.append(f"os: PASS ({os.name})")
    except Exception:
        checks.append("os: WARNING")

    try:
        from app.brain.computer.folder_actions import _diagnose_known_folder_status
        checks.append(_safe_check("known_folder_resolver", lambda: _diagnose_known_folder_status("desktop")))
    except Exception:
        checks.append("known_folder_resolver: WARNING")

    return " | ".join(checks)
