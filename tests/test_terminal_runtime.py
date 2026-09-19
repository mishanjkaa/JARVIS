from __future__ import annotations

import os
import subprocess
import shutil
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.agent.controller import get_agent_controller
from app.brain.agent.state import get_agent_runtime_state, reset_agent_runtime_state
from app.brain.audit.audit_log import get_audit_entries, reset_audit_log
from app.brain.ai.models import AIIntent, AIResponse, ProviderStatus, ProviderStatusCategory
from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.filesystem.state import reset_filesystem_state, set_trusted_roots
from app.brain.planner.approval import store_pending_plan
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.planner_v2 import create_plan_from_request
from app.brain.planner.state import get_planner_state, reset_planner_state
from app.brain.risk.analyzer import analyze_plan
from app.brain.risk.models import PolicyOutcome, RiskLevel
from app.brain.router import _CONVERSATION_RUNTIME, route_command
from app.brain.terminal.controller import get_terminal_controller
from app.brain.terminal.models import TerminalExecutionResult, TerminalExecutionStatus
from app.brain.terminal.policy import configured_git_executable, configured_python_executable, decision_from_arguments
from app.brain.terminal.state import get_terminal_state, reset_terminal_state


class _FakeConversationProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.model = "stub-model"
        self.timeout = 1.0
        self.base_url = "http://localhost"

    def generate_text(self, prompt: str) -> AIResponse:
        self.prompts.append(prompt)
        return AIResponse(
            category="conversation",
            message="unexpected conversation call",
            intent=AIIntent.CONVERSATION,
            status=ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama"),
        )


class TerminalRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.root = (self.temp_root / f"terminal-{uuid4().hex}").absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.controller = get_terminal_controller()
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_planner_state()
        reset_filesystem_state()
        reset_terminal_state()
        reset_audit_log()
        set_trusted_roots([self.root])

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_planner_state()
        reset_filesystem_state()
        reset_terminal_state()
        reset_audit_log()
        shutil.rmtree(self.root, ignore_errors=True)

    def _terminal_arguments(
        self,
        *,
        executable: str = "python",
        arguments: list[str] | None = None,
        working_directory: str | None = None,
        timeout_seconds: int = 30,
        operation_type: str = "python",
        raw_command: str | None = None,
    ) -> dict[str, object]:
        args = list(arguments or [])
        command_text = raw_command or " ".join([executable, *args]).strip()
        return {
            "executable": executable,
            "arguments": args,
            "working_directory": working_directory or str(self.root),
            "timeout_seconds": timeout_seconds,
            "operation_type": operation_type,
            "raw_command": command_text,
        }

    def _write_script(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _init_git_repo(self) -> None:
        git_path = configured_git_executable()
        if shutil.which(git_path) is None and not Path(git_path).exists():
            self.skipTest("git is not available in this environment")
        subprocess.run([git_path, "init"], cwd=self.root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _wait_for_active_process(self, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            process = get_terminal_state().active_process
            if process is not None and process.poll() is None:
                return
            time.sleep(0.05)
        self.fail("Terminal process did not become active in time.")

    def test_terminal_status_commands_are_safe(self) -> None:
        self.assertEqual(route_command("terminal status"), "Terminal enabled.\nNo command currently running.")
        self.assertEqual(route_command("terminal history"), "No terminal history yet.")
        self.assertEqual(route_command("terminal cancel"), "No terminal command is running.")

    def test_terminal_result_display_is_readable(self) -> None:
        result = TerminalExecutionResult(
            display_command="python --version",
            executable="python",
            arguments=["--version"],
            working_directory=str(self.root),
            stdout="Python 3",
            stderr="",
            exit_code=0,
            started_at="2026-07-23T00:00:00+00:00",
            completed_at="2026-07-23T00:00:01+00:00",
            duration_seconds=1.0,
            timed_out=False,
            cancelled=False,
            status=TerminalExecutionStatus.COMPLETED,
        )
        display = result.to_display()
        self.assertIn("Command completed successfully.", display)
        self.assertIn("Exit code: 0", display)
        self.assertIn("stdout:", display)

    def test_configured_python_executable_matches_runtime(self) -> None:
        self.assertTrue(Path(configured_python_executable()).exists())

    def test_python_version_execution(self) -> None:
        result = self.controller.execute_from_arguments(self._terminal_arguments(arguments=["--version"], raw_command="python --version"))
        self.assertEqual(result.status, TerminalExecutionStatus.COMPLETED)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Python", result.to_display())

    def test_project_local_python_script_execution(self) -> None:
        script = self._write_script("hello.py", "print('hello from terminal runtime')\n")
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(arguments=[script.name], raw_command=f"python {script.name}")
        )
        self.assertEqual(result.status, TerminalExecutionStatus.COMPLETED)
        self.assertIn("hello from terminal runtime", result.stdout)

    def test_unittest_execution_through_router(self) -> None:
        self._write_script(
            "tests/test_sample.py",
            "import unittest\n\n\nclass SampleTest(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(2 + 2, 4)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        )
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        response = route_command("Run the unit tests")
        self.assertIn("Command completed successfully.", response)
        self.assertIn("OK", response)

    def test_non_zero_exit_code_marks_command_failed(self) -> None:
        script = self._write_script("fail.py", "import sys\nprint('boom', file=sys.stderr)\nraise SystemExit(3)\n")
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(arguments=[script.name], raw_command=f"python {script.name}")
        )
        self.assertEqual(result.status, TerminalExecutionStatus.FAILED)
        self.assertEqual(result.exit_code, 3)
        self.assertIn("boom", result.stderr)

    def test_output_truncation_is_bounded(self) -> None:
        set_runtime_config_value("terminal_max_stdout_chars", 80)
        set_runtime_config_value("terminal_max_stderr_chars", 80)
        script = self._write_script(
            "truncate.py",
            "import sys\nprint('A' * 200)\nprint('B' * 200, file=sys.stderr)\n",
        )
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(arguments=[script.name], raw_command=f"python {script.name}")
        )
        self.assertEqual(result.status, TerminalExecutionStatus.COMPLETED)
        self.assertIn("[output truncated]", result.stdout)
        self.assertIn("[output truncated]", result.stderr)

    def test_large_stdout_router_response_stays_useful(self) -> None:
        set_runtime_config_value("developer_mode", True)
        set_runtime_config_value("terminal_max_stdout_chars", 180)
        script = self._write_script("big_stdout.py", "print('A' * 1000)\n")
        response = route_command(f"python {script.name}")
        self.assertIn("Command completed successfully.", response)
        self.assertIn("Exit code: 0", response)
        self.assertIn("Duration:", response)
        self.assertIn("[output truncated]", response)
        self.assertNotEqual(response, "Tool result too large.")

    def test_large_stderr_router_response_stays_useful(self) -> None:
        set_runtime_config_value("developer_mode", True)
        set_runtime_config_value("terminal_max_stderr_chars", 180)
        script = self._write_script("big_stderr.py", "import sys\nprint('B' * 1000, file=sys.stderr)\nraise SystemExit(2)\n")
        response = route_command(f"python {script.name}")
        self.assertIn("Command failed.", response)
        self.assertIn("Exit code: 2", response)
        self.assertIn("stderr:", response)
        self.assertIn("[output truncated]", response)
        self.assertNotEqual(response, "Tool result too large.")

    def test_large_unittest_output_does_not_collapse_to_generic_size_error(self) -> None:
        set_runtime_config_value("developer_mode", True)
        set_runtime_config_value("terminal_max_stdout_chars", 180)
        set_runtime_config_value("terminal_max_stderr_chars", 180)
        self._write_script(
            "mini_tests/test_loud.py",
            "import unittest\n\n\nclass LoudTest(unittest.TestCase):\n    def test_failure(self):\n        self.fail('Z' * 1000)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        )
        response = route_command('python -m unittest discover -s mini_tests -p "test_*.py"')
        self.assertIn("Command failed.", response)
        self.assertIn("Exit code:", response)
        self.assertIn("[output truncated]", response)
        self.assertNotEqual(response, "Tool result too large.")

    def test_large_git_status_output_does_not_collapse_to_generic_size_error(self) -> None:
        self._init_git_repo()
        set_runtime_config_value("developer_mode", True)
        set_runtime_config_value("terminal_max_stdout_chars", 120)
        for index in range(60):
            self._write_script(f"file_{index}.txt", f"file {index}\n")
        response = route_command("show git status")
        self.assertIn("Command completed successfully.", response)
        self.assertIn("Exit code: 0", response)
        self.assertIn("[output truncated]", response)
        self.assertNotEqual(response, "Tool result too large.")

    def test_timeout_marks_execution_and_cleans_up_process(self) -> None:
        script = self._write_script("sleep.py", "import time\ntime.sleep(5)\nprint('done')\n")
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(arguments=[script.name], raw_command=f"python {script.name}", timeout_seconds=1)
        )
        self.assertEqual(result.status, TerminalExecutionStatus.TIMED_OUT)
        self.assertTrue(result.timed_out)
        self.assertIsNone(get_terminal_state().active_process)

    def test_terminal_cancel_stops_later_execution(self) -> None:
        script = self._write_script("cancel_me.py", "import time\ntime.sleep(5)\nprint('done')\n")
        holder: dict[str, TerminalExecutionResult] = {}

        thread = threading.Thread(
            target=lambda: holder.setdefault(
                "result",
                self.controller.execute_from_arguments(
                    self._terminal_arguments(arguments=[script.name], raw_command=f"python {script.name}", timeout_seconds=10)
                ),
            )
        )
        thread.start()
        self._wait_for_active_process()
        self.assertEqual(route_command("terminal cancel"), "Terminal cancellation requested.")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(holder["result"].status, TerminalExecutionStatus.CANCELLED)
        self.assertTrue(holder["result"].cancelled)
        self.assertIsNone(get_terminal_state().active_process)

    def test_emergency_stop_terminates_process_and_blocks_new_terminal_execution(self) -> None:
        script = self._write_script("long_run.py", "import time\ntime.sleep(5)\nprint('done')\n")
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        holder: dict[str, str] = {}
        thread = threading.Thread(target=lambda: holder.setdefault("response", route_command(f"python {script.name}")))
        thread.start()
        self._wait_for_active_process()
        self.assertEqual(route_command("emergency stop"), "Emergency stop engaged.")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(route_command("agent status"), "Agent state: emergency_stopped.")
        blocked = route_command("python --version")
        self.assertEqual(blocked, "Agent runtime is emergency stopped.")

    def test_terminal_history_is_bounded(self) -> None:
        set_runtime_config_value("terminal_history_limit", 2)
        for _ in range(3):
            result = self.controller.execute_from_arguments(self._terminal_arguments(arguments=["--version"], raw_command="python --version"))
            self.assertEqual(result.status, TerminalExecutionStatus.COMPLETED)
        history_lines = self.controller.history_message().splitlines()
        self.assertEqual(len(history_lines), 2)
        self.assertIn("python", history_lines[-1].lower())

    def test_working_directory_must_stay_inside_trusted_roots(self) -> None:
        outside = self.root.parent
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(working_directory=str(outside), raw_command="python --version", arguments=["--version"])
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("Working directory is invalid.", result.stderr)

    def test_relative_working_directory_is_resolved_against_trusted_default_root(self) -> None:
        outside_cwd = (self.temp_root / f"outside-cwd-{uuid4().hex}").absolute()
        outside_cwd.mkdir(parents=True, exist_ok=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(outside_cwd)
            decision = decision_from_arguments(
                self._terminal_arguments(
                    executable="python",
                    arguments=["--version"],
                    working_directory=".",
                    operation_type="python",
                    raw_command="python --version",
                )
            )
        finally:
            os.chdir(previous_cwd)
            shutil.rmtree(outside_cwd, ignore_errors=True)
        self.assertTrue(decision.allowed)
        self.assertEqual(Path(decision.working_directory), self.root)

    def test_relative_path_into_a_system_directory_under_a_broad_trusted_root_is_rejected(self) -> None:
        # The historical check here only looked at the path argument's FINAL component
        # ("evil.py", not "system32"), and only against a raw absolute-looking string --
        # neither catches a RELATIVE argument that resolves into a system directory once
        # joined to a broad trusted root (e.g. the whole system drive, once
        # filesystem_allow_full_disk_access is enabled elsewhere). Reproduced here without
        # that config by trusting a folder that itself contains a "Windows/System32" path,
        # which is exactly the shape a widened trusted root would produce.
        (self.root / "Windows" / "System32").mkdir(parents=True, exist_ok=True)
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(arguments=["Windows/System32/evil.py"], raw_command="python Windows/System32/evil.py")
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("That path is not allowed.", result.stderr)

    def test_trusted_root_enforcement_rejects_traversal(self) -> None:
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(arguments=["../outside.py"], raw_command="python ../outside.py")
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("That path is not allowed.", result.stderr)

    def test_symlink_escape_is_rejected(self) -> None:
        outside_dir = (self.temp_root / f"outside-{uuid4().hex}").absolute()
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_script = outside_dir / "outside.py"
        outside_script.write_text("print('escaped')\n", encoding="utf-8")
        link_path = self.root / "link"
        try:
            try:
                link_path.symlink_to(outside_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not available in this environment")
            result = self.controller.execute_from_arguments(
                self._terminal_arguments(arguments=["link/outside.py"], raw_command="python link/outside.py")
            )
            self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
            self.assertIn("That path is not allowed.", result.stderr)
        finally:
            if link_path.exists() or link_path.is_symlink():
                try:
                    link_path.unlink()
                except OSError:
                    pass
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_shell_operator_commands_are_rejected(self) -> None:
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(
                arguments=["test.py", "&&", "del", "important.txt"],
                raw_command="python test.py && del important.txt",
            )
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("Shell syntax is not allowed.", result.stderr)

    def test_pipe_and_redirection_commands_are_rejected(self) -> None:
        pipe_result = self.controller.execute_from_arguments(
            self._terminal_arguments(
                arguments=["status", "|", "more"],
                executable="git",
                operation_type="git_read_only",
                raw_command="git status | more",
            )
        )
        redirect_result = self.controller.execute_from_arguments(
            self._terminal_arguments(
                arguments=["status", ">", "status.txt"],
                executable="git",
                operation_type="git_read_only",
                raw_command="git status > status.txt",
            )
        )
        self.assertEqual(pipe_result.status, TerminalExecutionStatus.REJECTED)
        self.assertEqual(redirect_result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("Shell syntax is not allowed.", pipe_result.stderr)
        self.assertIn("Shell syntax is not allowed.", redirect_result.stderr)

    def test_unknown_executable_is_rejected(self) -> None:
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(
                executable="unknown-program",
                arguments=["argument"],
                operation_type="unsupported",
                raw_command="unknown-program argument",
            )
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("Unsupported terminal operation.", result.stderr)

    def test_git_read_only_commands_are_medium_risk(self) -> None:
        status_decision = decision_from_arguments(
            self._terminal_arguments(executable="git", arguments=["status"], operation_type="git_read_only", raw_command="git status")
        )
        porcelain_decision = decision_from_arguments(
            self._terminal_arguments(executable="git", arguments=["status", "--porcelain"], operation_type="git_read_only", raw_command="git status --porcelain")
        )
        branch_decision = decision_from_arguments(
            self._terminal_arguments(executable="git", arguments=["branch"], operation_type="git_read_only", raw_command="git branch")
        )
        self.assertTrue(status_decision.allowed)
        self.assertTrue(porcelain_decision.allowed)
        self.assertEqual(status_decision.risk_level, "MEDIUM")
        self.assertEqual(porcelain_decision.risk_level, "MEDIUM")
        self.assertTrue(branch_decision.allowed)
        self.assertEqual(branch_decision.risk_level, "MEDIUM")

    def test_unsupported_operation_type_is_rejected(self) -> None:
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(
                executable="git",
                arguments=["status", "--porcelain"],
                operation_type="shell",
                raw_command="git status --porcelain",
            )
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("Unsupported terminal operation.", result.stderr)

    def test_destructive_git_commands_are_rejected(self) -> None:
        result = self.controller.execute_from_arguments(
            self._terminal_arguments(executable="git", arguments=["reset", "--hard"], operation_type="git_read_only", raw_command="git reset --hard")
        )
        self.assertEqual(result.status, TerminalExecutionStatus.REJECTED)
        self.assertIn("Destructive Git operations are not allowed.", result.stderr)

    def test_package_install_is_high_risk(self) -> None:
        decision = decision_from_arguments(
            self._terminal_arguments(
                arguments=["-m", "pip", "install", "requests>=2.30"],
                operation_type="package_install",
                raw_command="python -m pip install requests>=2.30",
            )
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.risk_level, "HIGH")

    def test_package_install_approval_path_stays_safe(self) -> None:
        set_runtime_config_value("developer_mode", True)
        response = route_command("Install Python package requests")
        self.assertIn("HIGH RISK operation.", response)
        self.assertIn("args: ['-m', 'pip', 'install', 'requests']", response)
        self.assertIn("working directory:", response)
        self.assertIn("package installation changes the environment.", response)
        set_runtime_config_value("terminal_allow_package_install_with_approval", False)
        approved = route_command("approve plan")
        self.assertIn("Command rejected.", approved)
        self.assertIn("Package installation is disabled.", approved)

    def test_developer_mode_auto_executes_medium_terminal_commands(self) -> None:
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        response = route_command("python --version")
        self.assertIn("Command completed successfully.", response)
        self.assertIn("Python", response)

    def test_high_risk_terminal_plan_still_requires_approval_in_developer_mode(self) -> None:
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        response = route_command("pip install requests")
        self.assertIn("HIGH RISK operation.", response)
        self.assertIn("Approval required.", response)

    def test_mixed_plan_uses_highest_risk_step(self) -> None:
        assessment = analyze_plan(
            AgentPlan(
                steps=[
                    AgentStep(1, "filesystem.create_text_file", {"path": "requirements.txt"}, risk_level="persistent_write"),
                    AgentStep(
                        2,
                        "terminal.execute",
                        {
                            "executable": "python",
                            "arguments": ["-m", "pip", "install", "requests"],
                            "working_directory": str(self.root),
                            "timeout_seconds": 30,
                            "operation_type": "package_install",
                            "raw_command": "python -m pip install requests",
                        },
                        risk_level="sensitive",
                    ),
                ],
                original_request="Modify requirements then install package.",
            )
        )
        self.assertEqual(assessment.level, RiskLevel.HIGH)
        self.assertEqual(assessment.policy_outcome, PolicyOutcome.APPROVAL_REQUIRED)

    def test_planner_routes_terminal_requests_to_structured_steps(self) -> None:
        package_plan = create_plan_from_request("Install package requests")
        python_plan = create_plan_from_request("python --version")
        unit_test_plan = create_plan_from_request("Run the unit tests")
        self.assertEqual(package_plan.steps[0].tool_name, "terminal.execute")
        self.assertEqual(package_plan.steps[0].arguments["arguments"], ["-m", "pip", "install", "requests"])
        self.assertEqual(python_plan.steps[0].arguments["arguments"], ["--version"])
        self.assertEqual(unit_test_plan.steps[0].arguments["arguments"][:2], ["-m", "unittest"])

    def test_router_integrates_terminal_history(self) -> None:
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        route_command("python --version")
        history = route_command("terminal history")
        self.assertIn("completed", history)
        self.assertIn("python", history.lower())

    def test_agent_runtime_executes_terminal_step_after_filesystem_steps(self) -> None:
        plan = AgentPlan(
            steps=[
                AgentStep(1, "filesystem.create_text_file", {"path": "runner.py"}, risk_level="persistent_write"),
                AgentStep(2, "filesystem.write_text_file", {"path": "runner.py", "text": "print('agent terminal ok')"}, risk_level="persistent_write"),
                AgentStep(
                    3,
                    "terminal.execute",
                    {
                        "executable": "python",
                        "arguments": ["runner.py"],
                        "working_directory": str(self.root),
                        "timeout_seconds": 30,
                        "operation_type": "python",
                        "raw_command": "python runner.py",
                    },
                    risk_level="local_safe",
                ),
            ],
            original_request="Create a script and run it.",
        )
        self.assertIn("MEDIUM risk", store_pending_plan(plan))
        result = route_command("approve plan")
        self.assertIn("Command completed successfully.", result)
        self.assertIn("agent terminal ok", result)

    def test_agent_execution_with_truncated_terminal_output_stays_useful(self) -> None:
        set_runtime_config_value("terminal_max_stdout_chars", 180)
        script = self._write_script("agent_big.py", "print('C' * 1000)\n")
        plan = AgentPlan(
            steps=[
                AgentStep(
                    1,
                    "terminal.execute",
                    {
                        "executable": "python",
                        "arguments": [script.name],
                        "working_directory": str(self.root),
                        "timeout_seconds": 30,
                        "operation_type": "python",
                        "raw_command": f"python {script.name}",
                    },
                    risk_level="local_safe",
                )
            ],
            original_request="Run a loud script.",
        )
        self.assertIn("MEDIUM risk", store_pending_plan(plan))
        response = route_command("approve plan")
        self.assertIn("Command completed successfully.", response)
        self.assertIn("[output truncated]", response)
        self.assertNotEqual(response, "Tool result too large.")

    def test_audit_events_are_generated_for_terminal_execution(self) -> None:
        result = self.controller.execute_from_arguments(self._terminal_arguments(arguments=["--version"], raw_command="python --version"))
        self.assertEqual(result.status, TerminalExecutionStatus.COMPLETED)
        event_types = [entry.event_type for entry in get_audit_entries()]
        self.assertIn("command_planned", event_types)
        self.assertIn("approval_requested", event_types)
        self.assertIn("command_approved", event_types)
        self.assertIn("command_started", event_types)
        self.assertIn("command_completed", event_types)

    def test_write_and_append_parsing_regression(self) -> None:
        self.assertEqual(route_command("Create folder RiskTest"), "Created directory RiskTest.")
        self.assertEqual(route_command("Create file RiskTest/notes.txt"), "Created file RiskTest/notes.txt.")
        self.assertEqual(route_command("Write Hello to RiskTest/notes.txt"), "Wrote text to RiskTest/notes.txt.")
        self.assertEqual(route_command('Append "World" to RiskTest/notes.txt'), "Appended text to RiskTest/notes.txt.")
        self.assertEqual(route_command("Read RiskTest/notes.txt"), "HelloWorld")

    def test_filesystem_traversal_routing_regression(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeConversationProvider()
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            response = route_command("Read ../../Windows/System32/drivers/etc/hosts")
        self.assertEqual(response, "That path is not allowed.")
        self.assertEqual(provider.prompts, [])

    def test_package_install_routing_regression(self) -> None:
        phrases = (
            "Install Python package requests",
            "Install package requests",
            "pip install requests",
            "python -m pip install requests",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                with patch("app.brain.router._build_ai_response") as ai_mock:
                    response = route_command(phrase)
                self.assertIn("HIGH RISK operation.", response)
                self.assertIn("Approval required.", response)
                ai_mock.assert_not_called()
                plan = create_plan_from_request(phrase)
                self.assertEqual(plan.steps[0].tool_name, "terminal.execute")
                self.assertEqual(plan.steps[0].arguments["arguments"][:3], ["-m", "pip", "install"])
                route_command("cancel plan")

    def test_policy_rejection_is_immediate_and_does_not_leave_pending_state(self) -> None:
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        response = route_command("python test.py && del important.txt")
        self.assertEqual(response, "Command rejected by terminal policy.\nReason: Shell syntax is not allowed.")
        self.assertEqual(route_command("show pending plan"), "No pending plan.")
        self.assertEqual(route_command("agent status"), "Agent state: idle.")
        follow_up = route_command("python --version")
        self.assertIn("Command completed successfully.", follow_up)

    def test_policy_rejected_commands_cover_security_acceptance_cases(self) -> None:
        cases = (
            ("python test.py && del important.txt", "Shell syntax is not allowed."),
            ("powershell -Command Get-ChildItem", "That executable is not allowed."),
            ("cmd /c dir", "That executable is not allowed."),
            ("git reset --hard", "Destructive Git operations are not allowed."),
            ("git clean -fd", "Destructive Git operations are not allowed."),
            ("shutdown /s", "That executable is not allowed."),
            ("format C:", "That executable is not allowed."),
            ("python ../../outside.py", "That path is not allowed."),
            ("python C:/Windows/System32/example.py", "That path is not allowed."),
            ("pip install requests && shutdown /s", "Shell syntax is not allowed."),
            ("unknown-program argument", "Unsupported terminal operation."),
        )
        for command, reason in cases:
            with self.subTest(command=command):
                response = route_command(command)
                self.assertEqual(response, f"Command rejected by terminal policy.\nReason: {reason}")
                self.assertEqual(route_command("show pending plan"), "No pending plan.")
                self.assertEqual(route_command("agent status"), "Agent state: idle.")
                self.assertIsNone(get_planner_state().pending_plan)
                self.assertIsNone(get_agent_runtime_state().current_task)

    def test_rejected_destructive_git_does_not_block_filesystem_command(self) -> None:
        response = route_command("git reset --hard")
        self.assertIn("Command rejected by terminal policy.", response)
        self.assertEqual(route_command("Create file RiskTest/after-rejection.txt"), "Created file RiskTest/after-rejection.txt.")

    def test_medium_risk_plan_cancellation_clears_state_and_allows_new_work(self) -> None:
        self.assertEqual(route_command("Create file notes.txt"), "Created file notes.txt.")
        pending = route_command("Delete notes.txt")
        self.assertIn("This plan is MEDIUM risk.", pending)
        self.assertEqual(route_command("cancel plan"), "Pending plan cancelled.")
        self.assertEqual(route_command("agent status"), "Agent state: idle.")
        self.assertIsNone(get_planner_state().pending_plan)
        self.assertIsNone(get_agent_runtime_state().current_task)
        self.assertEqual(route_command("Create file notes2.txt"), "Created file notes2.txt.")

    def test_high_risk_plan_cancellation_aliases_are_safe(self) -> None:
        pending = route_command("Install Python package requests")
        self.assertIn("HIGH RISK operation.", pending)
        self.assertEqual(route_command("reject plan"), "Pending plan cancelled.")
        self.assertEqual(route_command("cancel plan"), "No plan is currently pending.")
        self.assertEqual(route_command("agent status"), "Agent state: idle.")
        self.assertIsNone(get_planner_state().pending_plan)
        self.assertIsNone(get_agent_runtime_state().current_task)

    def test_cancel_pending_package_install_does_not_execute_and_terminal_stays_usable(self) -> None:
        self.assertIn("HIGH RISK operation.", route_command("Install package requests"))
        self.assertEqual(route_command("cancel plan"), "Pending plan cancelled.")
        history = route_command("terminal history")
        self.assertNotIn("pip install requests", history)
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        follow_up = route_command("python --version")
        self.assertIn("Command completed successfully.", follow_up)

    def test_rejected_plan_cancellation_is_harmless(self) -> None:
        self.assertIn("Command rejected by terminal policy.", route_command("cmd /c dir"))
        self.assertEqual(route_command("cancel plan"), "No plan is currently pending.")

    def test_plan_cancellation_records_audit_and_rejected_commands_appear_in_history(self) -> None:
        self.assertIn("HIGH RISK operation.", route_command("Install Python package requests"))
        self.assertEqual(route_command("cancel plan"), "Pending plan cancelled.")
        self.assertIn("Command rejected by terminal policy.", route_command("powershell -Command Get-ChildItem"))
        event_types = [entry.event_type for entry in get_audit_entries()]
        self.assertIn("plan_cancelled", event_types)
        self.assertIn("agent_cancelled", event_types)
        self.assertIn("command_rejected", event_types)
        history = route_command("terminal history")
        self.assertIn("rejected", history)


if __name__ == "__main__":
    unittest.main()
