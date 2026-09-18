from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.path_policy import get_trusted_roots
from app.brain.terminal.errors import TerminalPolicyError, TerminalWorkingDirectoryError
from app.brain.terminal.models import TerminalCommandRequest, TerminalPolicyDecision
from config.config_loader import load_config

_OUTPUT_TRUNCATED_MARKER = "\n[output truncated]"
_FORBIDDEN_EXECUTABLE_NAMES = {
    "bash",
    "cmd",
    "command.com",
    "diskpart",
    "format",
    "net",
    "netsh",
    "powershell",
    "pwsh",
    "reg",
    "regedit",
    "runas",
    "sc",
    "sh",
    "shutdown",
    "taskkill",
    "wsl",
}
_SHELL_OPERATOR_PATTERNS = (
    "&&",
    "||",
    ">>",
    "$(",
    "${",
    "%comspec%",
    "`",
)
_DANGEROUS_GIT_SUBCOMMANDS = {
    "add",
    "checkout",
    "clean",
    "commit",
    "merge",
    "push",
    "rebase",
    "reset",
    "restore",
    "switch",
}
_READ_ONLY_GIT_SUBCOMMANDS = {"status", "diff", "log", "branch", "rev-parse", "show", "remote"}
_SAFE_PACKAGE_SPEC_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?(?:(==|>=|<=|~=|>|<)[A-Za-z0-9*_.-]+)?$")
_SUPPORTED_OPERATION_TYPES = ("python", "git_read_only", "package_install")


def effective_terminal_config() -> dict[str, Any]:
    config = get_effective_runtime_config()
    return config


def configured_python_executable() -> str:
    return _canonicalize(Path(sys.executable)).as_posix()


def configured_git_executable() -> str:
    git_path = shutil.which("git") or shutil.which("git.exe") or "git"
    return git_path


def default_terminal_working_directory(config: dict[str, Any] | None = None) -> str:
    effective = effective_terminal_config() if config is None else dict(config)
    configured = effective.get("terminal_default_working_directory")
    trusted_roots = get_trusted_roots()
    if configured:
        try:
            resolved = _validate_directory(str(configured))
            return resolved.as_posix()
        except TerminalWorkingDirectoryError:
            pass
    return trusted_roots[0].as_posix()


def supported_operation_types() -> tuple[str, ...]:
    return _SUPPORTED_OPERATION_TYPES


def read_only_git_subcommands() -> tuple[str, ...]:
    return tuple(sorted(_READ_ONLY_GIT_SUBCOMMANDS))


def decision_from_arguments(arguments: dict[str, Any], config: dict[str, Any] | None = None) -> TerminalPolicyDecision:
    request = request_from_arguments(arguments, config=config)
    return evaluate_request(request, config=config)


def request_from_arguments(arguments: dict[str, Any], config: dict[str, Any] | None = None) -> TerminalCommandRequest:
    effective = effective_terminal_config() if config is None else dict(config)
    executable = str(arguments.get("executable", "")).strip()
    if not executable:
        raise TerminalPolicyError("Executable is required.")
    raw_arguments = arguments.get("arguments", [])
    if not isinstance(raw_arguments, list) or not all(isinstance(item, str) for item in raw_arguments):
        raise TerminalPolicyError("Arguments are invalid.")
    working_directory = _normalize_working_directory_argument(arguments.get("working_directory"), effective)
    timeout_seconds = int(arguments.get("timeout_seconds") or effective.get("terminal_timeout_seconds", 30))
    operation_type = str(arguments.get("operation_type") or "python").strip() or "python"
    raw_command = str(arguments.get("raw_command") or build_display_command(executable, raw_arguments))
    expected_exit_code = int(arguments.get("expected_exit_code", 0))
    return TerminalCommandRequest(
        executable=executable,
        arguments=list(raw_arguments),
        working_directory=working_directory,
        timeout_seconds=timeout_seconds,
        operation_type=operation_type,
        raw_command=raw_command,
        expected_exit_code=expected_exit_code,
    )


def _normalize_working_directory_argument(value: Any, config: dict[str, Any]) -> str:
    default_directory = default_terminal_working_directory(config)
    if value is None:
        return default_directory
    raw_value = str(value).strip()
    if not raw_value:
        return default_directory
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return raw_value
    return (Path(default_directory) / candidate).as_posix()


def evaluate_request(request: TerminalCommandRequest, config: dict[str, Any] | None = None) -> TerminalPolicyDecision:
    effective = effective_terminal_config() if config is None else dict(config)
    if not effective.get("terminal_enabled", True):
        return _reject(request, "Terminal runtime is disabled.")
    safe_display = build_display_command(request.executable, request.arguments)
    if _contains_shell_syntax(request.raw_command) or any(_contains_shell_syntax(value) for value in [request.executable, *request.arguments]):
        return _reject(request, "Shell syntax is not allowed.", safe_display=safe_display)
    working_directory = _validate_directory(request.working_directory)
    executable_name = Path(request.executable).name.lower()
    if executable_name in _FORBIDDEN_EXECUTABLE_NAMES:
        return _reject(request, "That executable is not allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())

    if request.operation_type == "python":
        return _evaluate_python_request(request, effective, safe_display, working_directory)
    if request.operation_type == "git_read_only":
        return _evaluate_git_request(request, effective, safe_display, working_directory)
    if request.operation_type == "package_install":
        return _evaluate_package_install_request(request, effective, safe_display, working_directory)
    return _reject(request, "Unsupported terminal operation.", safe_display=safe_display, working_directory=working_directory.as_posix())


def build_display_command(executable: str, arguments: list[str]) -> str:
    return " ".join([executable, *arguments]).strip()


def truncate_output(value: str, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    if max_chars <= 0:
        return _OUTPUT_TRUNCATED_MARKER.strip()
    if len(value) <= max_chars:
        return value
    marker = _OUTPUT_TRUNCATED_MARKER
    available = max(0, max_chars - len(marker))
    return value[:available] + marker


def _evaluate_python_request(
    request: TerminalCommandRequest,
    config: dict[str, Any],
    safe_display: str,
    working_directory: Path,
) -> TerminalPolicyDecision:
    if not config.get("terminal_allow_python", True):
        return _reject(request, "Python terminal execution is disabled.", safe_display=safe_display, working_directory=working_directory.as_posix())
    executable_path = configured_python_executable()
    if Path(request.executable).name.lower() not in {"python", Path(executable_path).name.lower()}:
        return _reject(request, "Unknown executable.", safe_display=safe_display, working_directory=working_directory.as_posix())
    for argument in request.arguments:
        if argument == "-c":
            return _reject(request, "Inline Python execution is not allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())
    _validate_python_arguments(request.arguments, working_directory)
    risk_level = "HIGH" if _is_python_package_install(request.arguments) else "MEDIUM"
    reason = "package installation changes the environment" if risk_level == "HIGH" else "project-local Python execution"
    return TerminalPolicyDecision(
        allowed=True,
        risk_level=risk_level,
        reason=reason,
        safe_display_command=build_display_command(executable_path, request.arguments),
        executable_path=executable_path,
        arguments=list(request.arguments),
        working_directory=working_directory.as_posix(),
        project_scoped=True,
    )


def _evaluate_git_request(
    request: TerminalCommandRequest,
    config: dict[str, Any],
    safe_display: str,
    working_directory: Path,
) -> TerminalPolicyDecision:
    if not config.get("terminal_allow_git_read_only", True):
        return _reject(request, "Git terminal execution is disabled.", safe_display=safe_display, working_directory=working_directory.as_posix())
    executable_path = configured_git_executable()
    if Path(request.executable).name.lower() not in {"git", Path(executable_path).name.lower()}:
        return _reject(request, "Unknown executable.", safe_display=safe_display, working_directory=working_directory.as_posix())
    if not request.arguments:
        return _reject(request, "Git arguments are required.", safe_display=safe_display, working_directory=working_directory.as_posix())
    first = request.arguments[0].lower()
    if first in _DANGEROUS_GIT_SUBCOMMANDS:
        return _reject(request, "Destructive Git operations are not allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())
    if first not in _READ_ONLY_GIT_SUBCOMMANDS:
        return _reject(request, "Only read-only Git operations are allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())
    if first == "branch" and not _is_safe_git_branch_request(request.arguments[1:]):
        return _reject(request, "Only read-only Git branch inspection is allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())
    if first == "remote":
        if request.arguments[1:] not in [["-v"], ["--verbose"]]:
            return _reject(request, "Only git remote -v is allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())
    return TerminalPolicyDecision(
        allowed=True,
        risk_level="MEDIUM",
        reason="read-only Git inspection",
        safe_display_command=build_display_command(executable_path, request.arguments),
        executable_path=executable_path,
        arguments=list(request.arguments),
        working_directory=working_directory.as_posix(),
        project_scoped=True,
    )


def _evaluate_package_install_request(
    request: TerminalCommandRequest,
    config: dict[str, Any],
    safe_display: str,
    working_directory: Path,
) -> TerminalPolicyDecision:
    if not config.get("terminal_allow_package_install_with_approval", True):
        return _reject(request, "Package installation is disabled.", safe_display=safe_display, working_directory=working_directory.as_posix())
    executable_path = configured_python_executable()
    if request.arguments[:3] != ["-m", "pip", "install"] or len(request.arguments) < 4:
        return _reject(request, "Only python -m pip install is supported.", safe_display=safe_display, working_directory=working_directory.as_posix())
    package_specs = request.arguments[3:]
    for spec in package_specs:
        if not _SAFE_PACKAGE_SPEC_RE.match(spec):
            return _reject(request, "Package specification is not allowed.", safe_display=safe_display, working_directory=working_directory.as_posix())
    return TerminalPolicyDecision(
        allowed=True,
        risk_level="HIGH",
        reason="package installation changes the environment",
        safe_display_command=build_display_command(executable_path, request.arguments),
        executable_path=executable_path,
        arguments=list(request.arguments),
        working_directory=working_directory.as_posix(),
        project_scoped=True,
    )


def _validate_python_arguments(arguments: list[str], working_directory: Path) -> None:
    if not arguments:
        return
    path_like_arguments = [value for value in arguments if _looks_like_path_argument(value)]
    for value in path_like_arguments:
        _validate_local_path_argument(value, working_directory)
    if arguments[:2] == ["-m", "unittest"]:
        return
    if arguments == ["--version"]:
        return


def _looks_like_path_argument(value: str) -> bool:
    if value.startswith("-"):
        return False
    return any(marker in value for marker in ("/", "\\", ".py"))


def _validate_local_path_argument(value: str, working_directory: Path) -> None:
    candidate = Path(value)
    joined = candidate if candidate.is_absolute() else working_directory / candidate
    normalized = _canonicalize(joined)
    _reject_symlink_ancestor(normalized)
    if not _is_inside_trusted_roots(normalized):
        raise TerminalPolicyError("That path is not allowed.")
    if normalized.name.lower() in {"system32", "windows"}:
        raise TerminalPolicyError("That path is not allowed.")


def _validate_directory(value: str) -> Path:
    if not value:
        raise TerminalWorkingDirectoryError("Working directory is invalid.")
    candidate = _canonicalize(Path(value))
    if not candidate.exists() or not candidate.is_dir():
        raise TerminalWorkingDirectoryError("Working directory is invalid.")
    _reject_symlink_ancestor(candidate)
    if not _is_inside_trusted_roots(candidate):
        raise TerminalWorkingDirectoryError("Working directory is invalid.")
    return candidate


def _contains_shell_syntax(value: str) -> bool:
    lowered = value.lower()
    if any(token in lowered for token in _SHELL_OPERATOR_PATTERNS):
        return True
    if re.search(r"(^|[\s])([|;`])(?!\S)", value):
        return True
    if re.search(r"(^|[\s])(&|>|<)(?=$|[\s])", value):
        return True
    return False


def _is_python_package_install(arguments: list[str]) -> bool:
    return len(arguments) >= 3 and arguments[:3] == ["-m", "pip", "install"]


def _is_safe_git_branch_request(arguments: list[str]) -> bool:
    if not arguments:
        return True
    allowed_flags = {"--list", "-a", "--all", "-r", "--remotes", "--show-current", "-v", "-vv"}
    return all(argument in allowed_flags for argument in arguments)


def _canonicalize(path: Path) -> Path:
    return Path(os.path.normpath(os.path.abspath(str(path))))


def _is_inside_trusted_roots(path: Path) -> bool:
    for root in get_trusted_roots():
        normalized_root = _canonicalize(root)
        try:
            path.relative_to(normalized_root)
            return True
        except ValueError:
            continue
    return False


def _reject_symlink_ancestor(path: Path) -> None:
    current = _canonicalize(path)
    while True:
        if current.exists() and current.is_symlink():
            raise TerminalPolicyError("That path is not allowed.")
        if current == current.parent:
            return
        current = current.parent


def _reject(
    request: TerminalCommandRequest,
    reason: str,
    *,
    safe_display: str | None = None,
    working_directory: str = "",
) -> TerminalPolicyDecision:
    return TerminalPolicyDecision(
        allowed=False,
        risk_level="HIGH",
        reason=reason,
        safe_display_command=safe_display or build_display_command(request.executable, request.arguments),
        executable_path=request.executable,
        arguments=list(request.arguments),
        working_directory=working_directory or request.working_directory,
        project_scoped=False,
        rejection_reason=reason,
    )
