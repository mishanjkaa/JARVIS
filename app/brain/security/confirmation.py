import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_PENDING_ACTION: Optional[str] = None
_PENDING_AT: Optional[float] = None
_TIMEOUT_SECONDS = 30


def request_confirmation(action: str) -> str:
    normalized = action.strip().lower()
    if not normalized:
        return "Please provide an action."
    global _PENDING_ACTION, _PENDING_AT
    _PENDING_ACTION = normalized
    _PENDING_AT = time.monotonic()
    logger.info("Pending action requested")
    return f"Please confirm: {normalized}"


def get_pending_action() -> Optional[str]:
    if _PENDING_ACTION is None or _PENDING_AT is None:
        return None
    if time.monotonic() - _PENDING_AT > _TIMEOUT_SECONDS:
        clear_pending_action()
        return None
    return _PENDING_ACTION


def confirm_pending_action(action: str) -> bool:
    pending = get_pending_action()
    if pending is None:
        return False
    normalized = action.strip().lower()
    if normalized != pending:
        return False
    clear_pending_action()
    return True


def cancel_pending_action() -> None:
    clear_pending_action()


def clear_pending_action() -> None:
    global _PENDING_ACTION, _PENDING_AT
    _PENDING_ACTION = None
    _PENDING_AT = None
