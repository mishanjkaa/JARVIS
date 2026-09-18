from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TerminalExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TerminalCommandRequest:
    executable: str
    arguments: list[str] = field(default_factory=list)
    working_directory: str = "."
    timeout_seconds: int = 30
    operation_type: str = "python"
    raw_command: str = ""
    expected_exit_code: int = 0


@dataclass
class TerminalPolicyDecision:
    allowed: bool
    risk_level: str
    reason: str
    safe_display_command: str
    executable_path: str
    arguments: list[str]
    working_directory: str
    project_scoped: bool
    rejection_reason: str = ""


@dataclass
class TerminalExecutionResult:
    display_command: str
    executable: str
    arguments: list[str]
    working_directory: str
    stdout: str
    stderr: str
    exit_code: int | None
    started_at: str
    completed_at: str
    duration_seconds: float
    timed_out: bool
    cancelled: bool
    status: TerminalExecutionStatus
    policy_reason: str = ""

    def to_display(self) -> str:
        status_line = {
            TerminalExecutionStatus.COMPLETED: "Command completed successfully.",
            TerminalExecutionStatus.FAILED: "Command failed.",
            TerminalExecutionStatus.TIMED_OUT: "Command timed out.",
            TerminalExecutionStatus.CANCELLED: "Command cancelled.",
            TerminalExecutionStatus.REJECTED: "Command rejected.",
            TerminalExecutionStatus.RUNNING: "Command is running.",
            TerminalExecutionStatus.PENDING: "Command pending.",
        }[self.status]
        lines = [status_line]
        if self.exit_code is not None:
            lines.append(f"Exit code: {self.exit_code}")
        if self.stdout:
            lines.append("")
            lines.append("stdout:")
            lines.append(self.stdout)
        if self.stderr:
            lines.append("")
            lines.append("stderr:")
            lines.append(self.stderr)
        return "\n".join(lines)


@dataclass
class TerminalExecutionRecord:
    display_command: str
    status: str
    exit_code: int | None
    duration_seconds: float
    timestamp: str

