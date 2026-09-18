import logging
from pathlib import Path

from config.settings import LOG_FILE, LOG_LEVEL


def initialize_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    log_file_path = Path(LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_file_path.resolve():
                return

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
