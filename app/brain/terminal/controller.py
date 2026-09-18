from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_runtime_config
from app.brain.terminal.errors import TerminalError
from app.brain.terminal.executor import execute_subprocess
from app.brain.terminal.models import TerminalExecutionRecord, TerminalExecutionResult, TerminalExecutionStatus
from app.brain.terminal.policy import build_display_command, decision_from_arguments, effective_terminal_config, request_from_arguments
from app.brain.terminal.state import get_terminal_state


class TerminalController:
    def effective_config(self) -> dict[str, Any]:
        return effective_terminal_config()

    def execute_from_arguments(self, arguments: dict[str, Any]) -> TerminalExecutionResult:
        config = self.effective_config()
        try:
            request = request_from_arguments(arguments, config=config)
            decision = decision_from_arguments(arguments, config=config)
        except TerminalError as error:
            result = self._rejected_result(arguments, str(error))
            record_audit_event("command_rejected", message=result.policy_reason)
            self._store_result(result)
            return result
        record_audit_event("command_planned", message=decision.safe_display_command)
        if decision.risk_level in {"MEDIUM", "HIGH"}:
            record_audit_event("approval_requested", message=f"{decision.risk_level} {decision.safe_display_command}")
        if not decision.allowed:
            record_audit_event("command_rejected", message=decision.rejection_reason or decision.reason)
            now = datetime.now(timezone.utc).isoformat()
            result = TerminalExecutionResult(
                display_command=decision.safe_display_command,
                executable=decision.executable_path,
                arguments=list(decision.arguments),
                working_directory=decision.working_directory,
                stdout="",
                stderr=decision.rejection_reason or decision.reason,
                exit_code=None,
                started_at=now,
                completed_at=now,
                duration_seconds=0.0,
                timed_out=False,
                cancelled=False,
                status=TerminalExecutionStatus.REJECTED,
                policy_reason=decision.rejection_reason or decision.reason,
            )
            self._store_result(result)
            return result

        record_audit_event("command_approved", message=decision.safe_display_command)
        state = get_terminal_state()
        with state.lock:
            state.active_request = request
            state.active_result = None
            state.cancellation_requested = False
            state.last_safe_status = f"Running: {decision.safe_display_command}"
        record_audit_event("command_started", message=decision.safe_display_command)
        result = execute_subprocess(
            request,
            executable_path=decision.executable_path,
            working_directory=decision.working_directory,
            max_stdout_chars=int(config.get("terminal_max_stdout_chars", 20000)),
            max_stderr_chars=int(config.get("terminal_max_stderr_chars", 20000)),
            state=state,
        )
        result.policy_reason = decision.reason
        if result.status == TerminalExecutionStatus.COMPLETED:
            record_audit_event("command_completed", message=decision.safe_display_command)
        elif result.status == TerminalExecutionStatus.TIMED_OUT:
            record_audit_event("command_timed_out", message=decision.safe_display_command)
        elif result.status == TerminalExecutionStatus.CANCELLED:
            record_audit_event("command_cancelled", message=decision.safe_display_command)
        else:
            record_audit_event("command_failed", message=decision.safe_display_command)
        self._store_result(result)
        return result

    def record_policy_rejection(self, arguments: dict[str, Any], reason: str) -> TerminalExecutionResult:
        result = self._rejected_result(arguments, reason)
        record_audit_event("command_rejected", message=result.policy_reason)
        self._store_result(result)
        return result

    def status_message(self) -> str:
        config = self.effective_config()
        enabled = "enabled" if config.get("terminal_enabled", True) else "disabled"
        state = get_terminal_state()
        with state.lock:
            if state.active_request is None or state.active_process is None or state.active_process.poll() is not None:
                return f"Terminal {enabled}.\nNo command currently running."
            command = state.active_result.display_command if state.active_result is not None else f"{state.active_request.executable} {' '.join(state.active_request.arguments)}".strip()
            return f"Terminal {enabled}.\nRunning: {command}"

    def history_message(self) -> str:
        state = get_terminal_state()
        with state.lock:
            if not state.recent_history:
                return "No terminal history yet."
            lines = []
            for record in state.recent_history:
                lines.append(
                    f"{record.timestamp} | {record.status} | exit={record.exit_code if record.exit_code is not None else '-'} | duration={record.duration_seconds:.3f}s | {record.display_command}"
                )
            return "\n".join(lines)

    def cancel_active_execution(self, *, reason: str = "cancel") -> str:
        state = get_terminal_state()
        with state.lock:
            process = state.active_process
            if process is None or process.poll() is not None:
                return "No terminal command is running."
            state.cancellation_requested = True
            try:
                process.terminate()
            except OSError:
                pass
        if reason == "emergency":
            record_audit_event("emergency_stop_applied", message="Terminal execution terminated")
            return "Emergency stop engaged."
        record_audit_event("command_cancelled", message="Terminal execution cancellation requested")
        return "Terminal cancellation requested."

    def _rejected_result(self, arguments: dict[str, Any], reason: str) -> TerminalExecutionResult:
        executable = str(arguments.get("executable") or "").strip()
        raw_arguments = arguments.get("arguments", [])
        safe_arguments = [item for item in raw_arguments if isinstance(item, str)]
        working_directory = str(arguments.get("working_directory") or "").strip()
        now = datetime.now(timezone.utc).isoformat()
        return TerminalExecutionResult(
            display_command=build_display_command(executable, safe_arguments),
            executable=executable,
            arguments=safe_arguments,
            working_directory=working_directory,
            stdout="",
            stderr=reason,
            exit_code=None,
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            timed_out=False,
            cancelled=False,
            status=TerminalExecutionStatus.REJECTED,
            policy_reason=reason,
        )

    def _store_result(self, result: TerminalExecutionResult) -> None:
        state = get_terminal_state()
        history_limit = int(self.effective_config().get("terminal_history_limit", 25))
        with state.lock:
            state.active_result = result
            state.active_request = None
            state.active_process = None
            state.cancellation_requested = False
            state.last_safe_status = result.status.value
            state.recent_history.append(
                TerminalExecutionRecord(
                    display_command=result.display_command,
                    status=result.status.value,
                    exit_code=result.exit_code,
                    duration_seconds=result.duration_seconds,
                    timestamp=result.completed_at,
                )
            )
            del state.recent_history[:-history_limit]


_CONTROLLER = TerminalController()


def get_terminal_controller() -> TerminalController:
    return _CONTROLLER
