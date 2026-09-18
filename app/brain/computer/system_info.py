import logging
import platform
import shutil
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["bytes", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} TB"


def get_system_info() -> str:
    try:
        logger.info("System info requested")
        os_name = platform.system() or "Unknown"
        release = platform.release() or "Unknown"
        architecture = platform.machine() or "Unknown"
        python_version = platform.python_version() or "Unknown"
        return f"OS: {os_name}; Release: {release}; Architecture: {architecture}; Python: {python_version}"
    except Exception:
        logger.exception("System info request failed")
        return "I could not read system information right now."


def get_disk_space() -> str:
    try:
        logger.info("Disk space requested")
        home_path = Path.home()
        drive = home_path.anchor or home_path.drive or "/"
        if not drive and home_path.parts:
            drive = home_path.parts[0]
        if not drive:
            drive = "/"
        total, used, free = shutil.disk_usage(drive)
        return (
            f"Total: {format_size(total)}; "
            f"Used: {format_size(used)}; "
            f"Free: {format_size(free)}"
        )
    except Exception:
        logger.exception("Disk space request failed")
        return "I could not read the disk space right now."


def get_computer_name() -> str:
    try:
        logger.info("Computer name requested")
        hostname = socket.gethostname()
        return f"Computer name: {hostname}"
    except Exception:
        logger.exception("Computer name request failed")
        return "I could not read the computer name right now."
