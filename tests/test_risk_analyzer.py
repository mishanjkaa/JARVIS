import os
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.brain.agent.state import reset_agent_runtime_state
from app.brain.configuration.runtime_config import reset_runtime_config
from app.brain.filesystem.state import reset_filesystem_state, set_trusted_roots
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.planner_v2 import create_plan_from_request
from app.brain.planner.state import reset_planner_state
from app.brain.risk.analyzer import analyze_plan
from app.brain.risk.models import RiskLevel
from app.brain.router import route_command


class RiskAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.root = (self.temp_root / f"risk-{uuid4().hex}").absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_planner_state()
        reset_filesystem_state()
        set_trusted_roots([self.root])

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_planner_state()
        reset_filesystem_state()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_low_risk_auto_execution(self) -> None:
        self.assertEqual(route_command("Create folder Test"), "Created directory Test.")
        self.assertTrue((self.root / "Test").exists())

    def test_medium_risk_requires_approval(self) -> None:
        self.assertEqual(route_command("Create file notes.txt"), "Created file notes.txt.")
        response = route_command("Delete notes.txt")
        self.assertIn("This plan is MEDIUM risk.", response)
        self.assertTrue((self.root / "notes.txt").exists())

    def test_high_risk_requires_approval(self) -> None:
        response = route_command("Install Python package requests")
        self.assertIn("HIGH RISK operation.", response)
        self.assertIn("Approval required.", response)

    def test_developer_mode_off_keeps_medium_pending(self) -> None:
        self.assertEqual(route_command("Create file notes.txt"), "Created file notes.txt.")
        self.assertEqual(route_command("developer mode off"), "Developer mode disabled.")
        response = route_command("Delete notes.txt")
        self.assertIn("This plan is MEDIUM risk.", response)
        self.assertTrue((self.root / "notes.txt").exists())

    def test_developer_mode_on_auto_executes_medium_project_operations(self) -> None:
        self.assertEqual(route_command("Create file notes.txt"), "Created file notes.txt.")
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        self.assertIn("Moved notes.txt to Trash.", route_command("Delete notes.txt"))
        self.assertFalse((self.root / "notes.txt").exists())

    def test_risk_analyzer_correctness(self) -> None:
        low_plan = AgentPlan(steps=[AgentStep(1, "filesystem.read_text_file", {"path": "notes.txt"})], original_request="read")
        medium_plan = AgentPlan(steps=[AgentStep(1, "filesystem.delete_path", {"path": "notes.txt"}, risk_level="persistent_write")], original_request="delete")
        high_plan = AgentPlan(
            steps=[
                AgentStep(
                    1,
                    "terminal.execute",
                    {
                        "executable": "python",
                        "arguments": ["-m", "pip", "install", "requests"],
                        "working_directory": ".",
                        "timeout_seconds": 30,
                        "operation_type": "package_install",
                        "raw_command": "Install Python package requests",
                    },
                    risk_level="sensitive",
                )
            ],
            original_request="install",
        )

        self.assertEqual(analyze_plan(low_plan).level, RiskLevel.LOW)
        self.assertEqual(analyze_plan(medium_plan).level, RiskLevel.MEDIUM)
        self.assertEqual(analyze_plan(high_plan).level, RiskLevel.HIGH)

    def test_planner_integration_for_high_risk_placeholder(self) -> None:
        plan = create_plan_from_request("Install Python package requests")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].tool_name, "terminal.execute")
        self.assertEqual(plan.steps[0].arguments["arguments"], ["-m", "pip", "install", "requests"])

    def test_agent_integration_low_risk_auto_executes(self) -> None:
        self.assertEqual(route_command("Create folder Project"), "Created directory Project.")
        self.assertEqual(route_command("agent status"), "Agent state: completed.")

    def test_delete_requires_approval(self) -> None:
        self.assertEqual(route_command("Create file notes.txt"), "Created file notes.txt.")
        self.assertIn("Pending plan:", route_command("Delete notes.txt"))
        self.assertEqual(route_command("approve plan"), "Moved notes.txt to Trash.")

    def test_terminal_placeholder_remains_high_risk(self) -> None:
        assessment = analyze_plan(
            AgentPlan(
                steps=[
                    AgentStep(
                        1,
                        "terminal.execute",
                        {
                            "executable": "python",
                            "arguments": ["-m", "pip", "install", "requests"],
                            "working_directory": ".",
                            "timeout_seconds": 30,
                            "operation_type": "package_install",
                            "raw_command": "pip install requests",
                        },
                        risk_level="sensitive",
                    )
                ],
                original_request="install",
            )
        )
        self.assertEqual(assessment.level, RiskLevel.HIGH)

    def test_memory_forget_requires_same_approval_as_memory_remember(self) -> None:
        remember_assessment = analyze_plan(
            AgentPlan(
                steps=[AgentStep(1, "memory.remember", {"key": "project", "value": "JARVIS"}, risk_level="persistent_write")],
                original_request="remember project = JARVIS",
            )
        )
        forget_assessment = analyze_plan(
            AgentPlan(
                steps=[AgentStep(1, "memory.forget", {"key": "project"}, risk_level="persistent_write")],
                original_request="forget project",
            )
        )
        self.assertEqual(remember_assessment.level, RiskLevel.MEDIUM)
        self.assertEqual(forget_assessment.level, RiskLevel.MEDIUM)
        self.assertEqual(remember_assessment.auto_execute, forget_assessment.auto_execute)

    def test_desktop_capture_never_auto_executes_even_with_developer_mode_and_medium_project(self) -> None:
        # RFC-007C requires MEDIUM approval for desktop/window capture, always. Unlike
        # filesystem operations, desktop.* tools are deliberately absent from every
        # project-scoping exemption in _is_project_scoped(), so they fall through to its
        # catch-all `return False`. This proves that holds even when both settings that
        # would otherwise auto-execute a MEDIUM-risk plan are enabled.
        permissive_config = {"developer_mode": True, "auto_execute_medium_project": True}
        screen_assessment = analyze_plan(
            AgentPlan(
                steps=[AgentStep(1, "desktop.capture_screen", {}, risk_level="persistent_write")],
                original_request="screenshot the desktop",
            ),
            config=permissive_config,
        )
        window_assessment = analyze_plan(
            AgentPlan(
                steps=[AgentStep(1, "desktop.capture_window", {"window_id": 1, "window_title": "Notepad"}, risk_level="persistent_write")],
                original_request="screenshot window Notepad",
            ),
            config=permissive_config,
        )
        self.assertEqual(screen_assessment.level, RiskLevel.MEDIUM)
        self.assertFalse(screen_assessment.project_scoped)
        self.assertFalse(screen_assessment.auto_execute)
        self.assertEqual(window_assessment.level, RiskLevel.MEDIUM)
        self.assertFalse(window_assessment.project_scoped)
        self.assertFalse(window_assessment.auto_execute)
