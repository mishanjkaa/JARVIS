import logging
from typing import List

logger = logging.getLogger(__name__)

_HISTORY: List[str] = []
_MAX_HISTORY = 50


def add_history_entry(label: str) -> None:
    if not label or not isinstance(label, str):
        return
    _HISTORY.append(label)
    if len(_HISTORY) > _MAX_HISTORY:
        del _HISTORY[0]


def get_history() -> List[str]:
    return list(_HISTORY)


def clear_history() -> None:
    _HISTORY.clear()


def record_safe_command(category: str) -> None:
    safe_category = category.strip().lower()
    if not safe_category:
        return
    add_history_entry(safe_category)
