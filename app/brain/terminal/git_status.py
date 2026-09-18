from __future__ import annotations

from dataclasses import dataclass, field

_NOT_REPOSITORY_MARKERS = (
    "not a git repository",
    "fatal: not a git repository",
)
_TRUNCATED_MARKER = "[output truncated]"


@dataclass
class GitUntrackedEvidence:
    operation: str = "git_untracked_files"
    success: bool = False
    repository_detected: bool = False
    untracked_files: list[str] = field(default_factory=list)
    exit_code: int | None = None
    output_truncated: bool = False
    error_category: str = ""
    error_reason: str = ""

    def to_reference_fields(self) -> dict[str, object]:
        return {
            "terminal_operation": self.operation,
            "success": self.success,
            "repository_detected": self.repository_detected,
            "git_untracked_files": list(self.untracked_files),
            "exit_code": self.exit_code if self.exit_code is not None else -1,
            "output_truncated": self.output_truncated,
            "git_error_category": self.error_category,
            "git_error_reason": self.error_reason,
        }


def extract_git_untracked_evidence(arguments: list[str], stdout: str, stderr: str, exit_code: int | None) -> GitUntrackedEvidence | None:
    if not _is_git_untracked_status_request(arguments):
        return None
    output_truncated = _TRUNCATED_MARKER in stdout or _TRUNCATED_MARKER in stderr
    if exit_code == 0:
        return GitUntrackedEvidence(
            success=True,
            repository_detected=True,
            untracked_files=_parse_porcelain_untracked(stdout),
            exit_code=exit_code,
            output_truncated=output_truncated,
        )
    error_text = _first_error_line(stderr) or _first_error_line(stdout) or "git command failed"
    lowered_error = error_text.lower()
    if any(marker in lowered_error for marker in _NOT_REPOSITORY_MARKERS):
        return GitUntrackedEvidence(
            success=False,
            repository_detected=False,
            exit_code=exit_code,
            output_truncated=output_truncated,
            error_category="not_git_repository",
            error_reason="Not a Git repository.",
        )
    return GitUntrackedEvidence(
        success=False,
        repository_detected=True,
        exit_code=exit_code,
        output_truncated=output_truncated,
        error_category="git_command_failed",
        error_reason=error_text[:160],
    )


def _is_git_untracked_status_request(arguments: list[str]) -> bool:
    if not arguments or str(arguments[0]).lower() != "status":
        return False
    normalized = [str(item).lower() for item in arguments[1:]]
    return "--porcelain" in normalized


def _parse_porcelain_untracked(stdout: str) -> list[str]:
    paths: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.startswith("?? "):
            continue
        path = line[3:].strip()
        if path:
            paths.append(path)
    return paths


def _first_error_line(value: str) -> str:
    for line in value.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""
