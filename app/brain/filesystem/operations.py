from __future__ import annotations

from pathlib import Path
import shutil

from app.brain.filesystem.errors import FilesystemLimitError, FilesystemOperationError
from app.brain.filesystem.models import FilesystemMetadata


def list_directory_entries(path: Path, *, recursive: bool, max_depth: int) -> list[str]:
    if not path.exists():
        raise FilesystemOperationError("Directory not found.")
    if not path.is_dir():
        raise FilesystemOperationError("Directory not found.")
    entries: list[str] = []
    if recursive:
        _walk_directory(path, path, entries, current_depth=0, max_depth=max_depth)
    else:
        for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            entries.append(child.name + ("/" if child.is_dir() else ""))
    return entries


def create_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FilesystemOperationError("Unable to create directory.") from error


def create_text_file(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError as error:
        raise FilesystemOperationError("Unable to create file.") from error


def read_text_file(path: Path, *, max_read_size: int) -> str:
    if not path.exists() or not path.is_file():
        raise FilesystemOperationError("File not found.")
    if path.stat().st_size > max_read_size:
        raise FilesystemLimitError("File is too large to read safely.")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise FilesystemOperationError("Only UTF-8 text files are supported.") from error


def write_text_file(path: Path, text: str, *, max_write_size: int) -> None:
    encoded = text.encode("utf-8")
    if len(encoded) > max_write_size:
        raise FilesystemLimitError("Write content is too large.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as error:
        raise FilesystemOperationError("Unable to write file.") from error


def append_text_file(path: Path, text: str, *, max_write_size: int, max_file_size: int) -> None:
    encoded = text.encode("utf-8")
    if len(encoded) > max_write_size:
        raise FilesystemLimitError("Write content is too large.")
    current_size = path.stat().st_size if path.exists() else 0
    if current_size + len(encoded) > max_file_size:
        raise FilesystemLimitError("File is too large to modify safely.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as error:
        raise FilesystemOperationError("Unable to append to file.") from error


def rename_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FilesystemOperationError("File not found.")
    if source.is_dir():
        raise FilesystemOperationError("Only files are supported for this operation.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    except OSError as error:
        raise FilesystemOperationError("Unable to rename file.") from error


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FilesystemOperationError("File not found.")
    if source.is_dir():
        raise FilesystemOperationError("Only files are supported for this operation.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError as error:
        raise FilesystemOperationError("Unable to copy file.") from error


def move_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FilesystemOperationError("File not found.")
    if source.is_dir():
        raise FilesystemOperationError("Only files are supported for this operation.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    except OSError as error:
        raise FilesystemOperationError("Unable to move file.") from error


def soft_delete(source: Path, trash_destination: Path) -> None:
    if not source.exists():
        raise FilesystemOperationError("File not found.")
    if source.is_dir():
        raise FilesystemOperationError("Only files are supported for this operation.")
    try:
        trash_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(trash_destination))
    except OSError as error:
        raise FilesystemOperationError("Unable to move file to Trash.") from error


def exists(path: Path) -> bool:
    return path.exists()


def metadata(path: Path) -> FilesystemMetadata:
    return FilesystemMetadata(
        relative_path=path.name,
        exists=path.exists(),
        is_file=path.is_file(),
        is_directory=path.is_dir(),
        size_bytes=path.stat().st_size if path.exists() and path.is_file() else 0,
    )


def _walk_directory(root: Path, current: Path, entries: list[str], *, current_depth: int, max_depth: int) -> None:
    if current_depth > max_depth:
        raise FilesystemLimitError("Directory traversal exceeded the maximum depth.")
    for child in sorted(current.iterdir(), key=lambda item: item.name.lower()):
        relative = child.relative_to(root).as_posix()
        entries.append(relative + ("/" if child.is_dir() else ""))
        if child.is_dir():
            _walk_directory(root, child, entries, current_depth=current_depth + 1, max_depth=max_depth)
