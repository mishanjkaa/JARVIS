import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RuntimeContext:
    def __init__(self) -> None:
        self.last_command_category: Optional[str] = None
        self.last_calculator_result: Optional[str] = None
        self.last_opened_application: Optional[str] = None
        self.last_opened_folder: Optional[str] = None
        self.last_web_search: Optional[str] = None


CONTEXT = RuntimeContext()


def reset_context() -> None:
    CONTEXT.last_command_category = None
    CONTEXT.last_calculator_result = None
    CONTEXT.last_opened_application = None
    CONTEXT.last_opened_folder = None
    CONTEXT.last_web_search = None


def update_context(**kwargs: object) -> None:
    for key, value in kwargs.items():
        if key == "last_command_category" and isinstance(value, str):
            CONTEXT.last_command_category = value
        elif key == "last_calculator_result" and isinstance(value, str):
            CONTEXT.last_calculator_result = value
        elif key == "last_opened_application" and isinstance(value, str):
            CONTEXT.last_opened_application = value
        elif key == "last_opened_folder" and isinstance(value, str):
            CONTEXT.last_opened_folder = value
        elif key == "last_web_search" and isinstance(value, str):
            CONTEXT.last_web_search = value


def get_context() -> RuntimeContext:
    return CONTEXT
