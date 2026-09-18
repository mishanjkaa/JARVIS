from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilesystemRuntimeState:
    trusted_roots: list[Path] = field(default_factory=list)
    current_directory: Path | None = None
    last_touched_file: Path | None = None


_STATE = FilesystemRuntimeState()


def get_filesystem_state() -> FilesystemRuntimeState:
    return _STATE


def reset_filesystem_state() -> None:
    _STATE.trusted_roots = []
    _STATE.current_directory = None
    _STATE.last_touched_file = None


def set_trusted_roots(roots: list[Path]) -> None:
    _STATE.trusted_roots = list(roots)


def set_current_directory(path: Path | None) -> None:
    _STATE.current_directory = path


def set_last_touched_file(path: Path | None) -> None:
    _STATE.last_touched_file = path
