import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def execute_power_action(action: str) -> str:
    normalized = action.strip().lower()
    if normalized not in {"lock", "shutdown", "restart"}:
        return "Unsupported action."

    try:
        logger.info("Power action requested: %s", normalized)
        if normalized == "lock" and os.name == "nt":
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], shell=False, check=False)
            return "Lock request sent."
        if normalized == "shutdown" and os.name == "nt":
            subprocess.run(["shutdown", "/s", "/t", "0"], shell=False, check=False)
            return "Shutdown request sent."
        if normalized == "restart" and os.name == "nt":
            subprocess.run(["shutdown", "/r", "/t", "0"], shell=False, check=False)
            return "Restart request sent."
        return "This action is only available on Windows."
    except Exception:
        logger.exception("Power action failed")
        return "I could not complete that action right now."
