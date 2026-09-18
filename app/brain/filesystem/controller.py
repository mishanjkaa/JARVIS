from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.errors import FilesystemDisabledError, FilesystemError, FilesystemLimitError, FilesystemOperationError, FilesystemPathError
from app.brain.filesystem.models import FilesystemOperationResult, FilesystemResolvedPath
from app.brain.filesystem.operations import (
    append_text_file,
    copy_path,
    create_directory,
    create_text_file,
    exists as path_exists,
    list_directory_entries,
    metadata as path_metadata,
    move_path,
    read_text_file,
    rename_path,
    soft_delete,
    write_text_file,
)
from app.brain.filesystem.path_policy import relative_audit_path, resolve_path
from app.brain.filesystem.state import get_filesystem_state, set_current_directory, set_last_touched_file
from config.config_loader import load_config

_TRASH_DIR_NAME = "Trash"


class FilesystemController:
    def effective_config(self) -> dict[str, Any]:
        return get_effective_runtime_config()

    def ensure_enabled(self) -> None:
        if not self.effective_config().get("filesystem_enabled", True):
            raise FilesystemDisabledError("Filesystem runtime is disabled.")

    def list_directory(self, path: str | None = None, *, recursive: bool = False, max_depth: int = 1) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=True)
        entries = list_directory_entries(resolved.absolute_path, recursive=recursive, max_depth=max_depth)
        self._remember_directory(resolved.absolute_path)
        self._audit("filesystem_read", resolved)
        message = "Directory is empty." if not entries else "\n".join(entries)
        return FilesystemOperationResult(True, message, message, {"display_value": message})

    def create_directory(self, path: str) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=True)
        create_directory(resolved.absolute_path)
        self._remember_directory(resolved.absolute_path)
        self._audit("filesystem_create", resolved)
        message = f"Created directory {resolved.relative_path}."
        return FilesystemOperationResult(True, message, resolved.relative_path, {"display_value": resolved.relative_path})

    def create_text_file(self, path: str) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=False)
        create_text_file(resolved.absolute_path)
        self._enforce_file_limit(resolved.absolute_path)
        self._remember_file(resolved.absolute_path)
        self._audit("filesystem_create", resolved)
        message = f"Created file {resolved.relative_path}."
        return FilesystemOperationResult(True, message, resolved.relative_path, {"display_value": resolved.relative_path})

    def read_text_file(self, path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=False)
        text = read_text_file(resolved.absolute_path, max_read_size=int(self.effective_config()["filesystem_max_read_size"]))
        self._remember_file(resolved.absolute_path)
        self._audit("filesystem_read", resolved)
        return FilesystemOperationResult(True, text, text, {"display_value": text, "path": resolved.relative_path})

    def write_text_file(self, text: str, path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=False)
        write_text_file(resolved.absolute_path, text, max_write_size=int(self.effective_config()["filesystem_max_write_size"]))
        self._enforce_file_limit(resolved.absolute_path)
        self._remember_file(resolved.absolute_path)
        self._audit("filesystem_write", resolved)
        message = f"Wrote text to {resolved.relative_path}."
        return FilesystemOperationResult(True, message, resolved.relative_path, {"display_value": resolved.relative_path})

    def append_text_file(self, text: str, path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=False)
        append_text_file(
            resolved.absolute_path,
            text,
            max_write_size=int(self.effective_config()["filesystem_max_write_size"]),
            max_file_size=int(self.effective_config()["filesystem_max_file_size"]),
        )
        self._remember_file(resolved.absolute_path)
        self._audit("filesystem_append", resolved)
        message = f"Appended text to {resolved.relative_path}."
        return FilesystemOperationResult(True, message, resolved.relative_path, {"display_value": resolved.relative_path})

    def rename_path(self, source_path: str | None, destination_path: str) -> FilesystemOperationResult:
        self.ensure_enabled()
        source = resolve_path(source_path, prefer_directory=False)
        destination = resolve_path(destination_path, prefer_directory=False)
        rename_path(source.absolute_path, destination.absolute_path)
        self._remember_file(destination.absolute_path)
        self._audit("filesystem_rename", destination)
        message = f"Renamed {source.relative_path} to {destination.relative_path}."
        return FilesystemOperationResult(True, message, destination.relative_path, {"display_value": destination.relative_path})

    def copy_path(self, source_path: str | None, destination_path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        source = resolve_path(source_path, prefer_directory=False)
        if destination_path:
            destination = resolve_path(destination_path, prefer_directory=False)
        else:
            destination_absolute = source.absolute_path.with_name(f"{source.absolute_path.stem}_copy{source.absolute_path.suffix}")
            destination = FilesystemResolvedPath(
                root=source.root,
                absolute_path=destination_absolute,
                relative_path=self._relative_path(destination_absolute, source.root),
            )
        copy_path(source.absolute_path, destination.absolute_path)
        self._remember_file(destination.absolute_path)
        self._audit("filesystem_copy", destination)
        message = f"Copied {source.relative_path} to {destination.relative_path}."
        return FilesystemOperationResult(True, message, destination.relative_path, {"display_value": destination.relative_path})

    def move_path(self, source_path: str | None, destination_path: str) -> FilesystemOperationResult:
        self.ensure_enabled()
        source = resolve_path(source_path, prefer_directory=False)
        destination = resolve_path(destination_path, prefer_directory=False)
        move_path(source.absolute_path, destination.absolute_path)
        self._remember_file(destination.absolute_path)
        self._audit("filesystem_move", destination)
        message = f"Moved {source.relative_path} to {destination.relative_path}."
        return FilesystemOperationResult(True, message, destination.relative_path, {"display_value": destination.relative_path})

    def delete_path(self, path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        config = self.effective_config()
        if not config.get("filesystem_soft_delete", True):
            raise FilesystemOperationError("Soft delete is disabled.")
        source = resolve_path(path, prefer_directory=False)
        trash_destination = self._trash_destination(source.absolute_path, source.root)
        soft_delete(source.absolute_path, trash_destination)
        set_last_touched_file(None)
        self._audit("filesystem_delete", source)
        message = f"Moved {source.relative_path} to Trash."
        return FilesystemOperationResult(True, message, self._relative_path(trash_destination, source.root), {"display_value": self._relative_path(trash_destination, source.root)})

    def exists(self, path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=False)
        exists_value = path_exists(resolved.absolute_path)
        message = f"Exists: {'true' if exists_value else 'false'}"
        return FilesystemOperationResult(True, message, "true" if exists_value else "false", {"display_value": "true" if exists_value else "false"})

    def metadata(self, path: str | None = None) -> FilesystemOperationResult:
        self.ensure_enabled()
        resolved = resolve_path(path, prefer_directory=False)
        info = path_metadata(resolved.absolute_path)
        info.relative_path = resolved.relative_path
        message = info.to_display()
        return FilesystemOperationResult(True, message, message, {"display_value": message})

    def _enforce_file_limit(self, path: Path) -> None:
        if path.exists() and path.is_file() and path.stat().st_size > int(self.effective_config()["filesystem_max_file_size"]):
            raise FilesystemLimitError("File is too large to modify safely.")

    def _trash_destination(self, source: Path, root: Path) -> Path:
        relative = source.resolve().relative_to(root.resolve())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        target = root / _TRASH_DIR_NAME / relative.parent / f"{relative.stem}_{timestamp}{relative.suffix}"
        return target

    def _remember_file(self, path: Path) -> None:
        set_last_touched_file(path)
        set_current_directory(path.parent)

    def _remember_directory(self, path: Path) -> None:
        set_current_directory(path)

    def _audit(self, event_type: str, resolved) -> None:
        record_audit_event(event_type, message=resolved.relative_path)

    def _relative_path(self, path: Path, root: Path) -> str:
        return relative_audit_path(path, root)


_CONTROLLER = FilesystemController()


def get_filesystem_controller() -> FilesystemController:
    return _CONTROLLER
