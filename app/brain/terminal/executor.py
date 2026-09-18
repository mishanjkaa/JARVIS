from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone

from app.brain.terminal.models import TerminalCommandRequest, TerminalExecutionResult, TerminalExecutionStatus
from app.brain.terminal.policy import truncate_output
from app.brain.terminal.state import TerminalRuntimeState


def execute_subprocess(
    request: TerminalCommandRequest,
    *,
    executable_path: str,
    working_directory: str,
    max_stdout_chars: int,
    max_stderr_chars: int,
    state: TerminalRuntimeState,
) -> TerminalExecutionResult:
    started = datetime.now(timezone.utc)
    process = subprocess.Popen(
        [executable_path, *request.arguments],
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    with state.lock:
        state.active_process = process

    stdout = ""
    stderr = ""
    status = TerminalExecutionStatus.RUNNING
    exit_code: int | None = None
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + max(1, request.timeout_seconds)

    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            exit_code = process.returncode
            with state.lock:
                cancelled = cancelled or state.cancellation_requested
            if cancelled:
                status = TerminalExecutionStatus.CANCELLED
            elif timed_out:
                status = TerminalExecutionStatus.TIMED_OUT
            elif exit_code == request.expected_exit_code:
                status = TerminalExecutionStatus.COMPLETED
            else:
                status = TerminalExecutionStatus.FAILED
            break
        except subprocess.TimeoutExpired:
            with state.lock:
                if state.cancellation_requested:
                    cancelled = True
                    _terminate_process(process)
                    continue
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process(process)
                continue

    completed = datetime.now(timezone.utc)
    duration = max(0.0, (completed - started).total_seconds())
    return TerminalExecutionResult(
        display_command=" ".join([executable_path, *request.arguments]).strip(),
        executable=executable_path,
        arguments=list(request.arguments),
        working_directory=working_directory,
        stdout=truncate_output(stdout or "", max_stdout_chars),
        stderr=truncate_output(stderr or "", max_stderr_chars),
        exit_code=exit_code,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        duration_seconds=round(duration, 3),
        timed_out=timed_out,
        cancelled=cancelled,
        status=status,
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)
