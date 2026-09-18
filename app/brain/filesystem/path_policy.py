from __future__ import annotations

import os
from pathlib import Path

from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.models import FilesystemResolvedPath
from app.brain.filesystem.state import get_filesystem_state

_WINDOWS_FORBIDDEN_SEGMENTS = {
    "windows",
    "program files",
    "program files (x86)",
    "system32",
    "users",
}
_ROOT_ANCHORED_RELATIVE_NAMES = {"trash"}


def default_project_root() -> Path:
    return _canonicalize(Path(__file__).parents[3])


def get_trusted_roots() -> list[Path]:
    state = get_filesystem_state()
    if state.trusted_roots:
        return [_canonicalize(root) for root in state.trusted_roots]
    return [default_project_root()]


def resolve_path(user_path: str | None, *, prefer_directory: bool = False, allow_missing: bool = True) -> FilesystemResolvedPath:
    raw_value = "" if user_path is None else str(user_path).strip()
    roots = get_trusted_roots()
    base_path = _explicit_base_path() if raw_value else _base_path(prefer_directory=prefer_directory)

    if raw_value:
        candidate = Path(raw_value)
        _reject_unsafe_input(raw_value, candidate)
        if candidate.is_absolute():
            joined = candidate
        elif candidate.parts and candidate.parts[0].lower() in _ROOT_ANCHORED_RELATIVE_NAMES:
            joined = roots[0] / candidate
        elif _is_already_root_relative(candidate, base_path, roots):
            joined = roots[0] / candidate
        else:
            joined = base_path / candidate
    else:
        joined = base_path

    resolved = _resolve_with_existing_ancestors(joined)
    root = _match_root(resolved, roots)
    relative_path = resolved.relative_to(root).as_posix()
    return FilesystemResolvedPath(root=root, absolute_path=resolved, relative_path="." if not relative_path else relative_path)


def relative_audit_path(path: Path, root: Path) -> str:
    try:
        relative = _canonicalize(path).relative_to(_canonicalize(root))
    except Exception:
        return path.name
    return relative.as_posix() or "."


def _base_path(*, prefer_directory: bool) -> Path:
    state = get_filesystem_state()
    if prefer_directory and state.current_directory is not None:
        return state.current_directory
    if not prefer_directory and state.last_touched_file is not None:
        return state.last_touched_file
    if state.current_directory is not None:
        return state.current_directory
    return get_trusted_roots()[0]


def _explicit_base_path() -> Path:
    state = get_filesystem_state()
    if state.current_directory is not None:
        return state.current_directory
    if state.last_touched_file is not None:
        return state.last_touched_file.parent
    return get_trusted_roots()[0]


def _is_already_root_relative(candidate: Path, base_path: Path, roots: list[Path]) -> bool:
    if candidate.is_absolute():
        return False
    for root in roots:
        normalized_root = _canonicalize(root)
        normalized_base = _canonicalize(base_path)
        try:
            base_relative = normalized_base.relative_to(normalized_root)
        except ValueError:
            continue
        if not base_relative.parts:
            continue
        if len(candidate.parts) < len(base_relative.parts):
            continue
        if tuple(part.lower() for part in candidate.parts[: len(base_relative.parts)]) == tuple(part.lower() for part in base_relative.parts):
            return True
    return False


def _reject_unsafe_input(raw_value: str, candidate: Path) -> None:
    if raw_value.startswith("\\\\"):
        raise FilesystemPathError("Network paths are not allowed.")
    if raw_value.startswith("//"):
        raise FilesystemPathError("Network paths are not allowed.")
    if any(part == ".." for part in candidate.parts):
        raise FilesystemPathError("That path is not allowed.")
    anchor = candidate.anchor.lower()
    if anchor and len(candidate.parts) <= 1:
        raise FilesystemPathError("Drive roots are not allowed.")
    lower_value = raw_value.lower()
    if any(segment in lower_value for segment in ("c:\\windows", "c:\\program files", "system32")):
        raise FilesystemPathError("That path is not allowed.")
    for part in candidate.parts:
        if part.lower() in _WINDOWS_FORBIDDEN_SEGMENTS and candidate.is_absolute():
            raise FilesystemPathError("That path is not allowed.")


def _resolve_with_existing_ancestors(path: Path) -> Path:
    normalized = _canonicalize(path)
    pending_parts: list[str] = []
    current = normalized
    while not current.exists() and current != current.parent:
        pending_parts.append(current.name)
        current = current.parent
    _reject_symlink_ancestor(current)
    resolved = _canonicalize(current)
    for part in reversed(pending_parts):
        resolved = resolved / part
        if resolved.exists():
            _reject_symlink_ancestor(resolved)
    return _canonicalize(resolved)


def _match_root(resolved_path: Path, roots: list[Path]) -> Path:
    normalized_path = _canonicalize(resolved_path)
    for root in roots:
        try:
            normalized_root = _canonicalize(root)
            normalized_path.relative_to(normalized_root)
            return normalized_root
        except ValueError:
            continue
    raise FilesystemPathError("That path is outside trusted roots.")


def _canonicalize(path: Path) -> Path:
    return Path(os.path.normpath(os.path.abspath(str(path))))


def _reject_symlink_ancestor(path: Path) -> None:
    current = _canonicalize(path)
    while True:
        if current.exists() and current.is_symlink():
            raise FilesystemPathError("Symlink paths are not allowed.")
        if current == current.parent:
            return
        current = current.parent
