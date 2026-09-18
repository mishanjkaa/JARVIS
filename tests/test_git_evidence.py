from __future__ import annotations

import unittest

from app.brain.agent.models import AgentLifecycleState, AgentTaskRecord
from app.brain.intelligence.goal_evaluator import evaluate_goal, render_goal_evaluation
from app.brain.intelligence.models import DynamicPlan, DynamicPlanStep
from app.brain.intelligence.task_interpreter import interpret_task
from app.brain.planner.plan_models import AgentPlan
from app.brain.terminal.git_status import extract_git_untracked_evidence


class GitEvidenceTests(unittest.TestCase):
    def _git_task(self):
        return interpret_task("Show me which files in git are untracked.")

    def _git_plan(self) -> DynamicPlan:
        task = self._git_task()
        return DynamicPlan(
            goal="Inspect git status",
            success_criteria=["stdout shows untracked files"],
            steps=[
                DynamicPlanStep(
                    "terminal.execute",
                    {
                        "executable": "git",
                        "arguments": ["status", "--porcelain"],
                        "timeout_seconds": 30,
                        "operation_type": "git_read_only",
                        "raw_command": "git status --porcelain",
                    },
                    description="Inspect Git status with porcelain output.",
                    expected_result="stdout shows untracked files",
                )
            ],
            original_request=task.raw_input,
        )

    def _task_record(self, *, message: str, reference_fields: dict[str, object], state: AgentLifecycleState = AgentLifecycleState.COMPLETED) -> AgentTaskRecord:
        task = self._git_task()
        return AgentTaskRecord(
            task_id=1,
            plan=AgentPlan(original_request=task.raw_input),
            original_user_goal=task.goal,
            total_steps=1,
            state=state,
            failure_reason=message if state == AgentLifecycleState.FAILED else "",
            step_results=[
                {
                    "tool_name": "terminal.execute",
                    "message": message,
                    "reference_fields": reference_fields,
                }
            ],
        )

    def test_porcelain_one_untracked_file(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], "?? welcome.py\n", "", 0)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.untracked_files, ["welcome.py"])

    def test_porcelain_several_untracked_files(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], "?? welcome.py\n?? sample_message.py\n", "", 0)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.untracked_files, ["welcome.py", "sample_message.py"])

    def test_porcelain_path_with_spaces(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], "?? notes folder/welcome draft.py\n", "", 0)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.untracked_files, ["notes folder/welcome draft.py"])

    def test_no_untracked_files(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], " M tracked.py\n", "", 0)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.untracked_files, [])

    def test_tracked_modified_files_are_not_reported_as_untracked(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], " M tracked.py\nM  tracked2.py\n", "", 0)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.untracked_files, [])

    def test_mixed_porcelain_extracts_only_double_question_mark_entries(self) -> None:
        stdout = " M tracked.py\n?? welcome.py\nA  committed.py\n?? notes with spaces.txt\n"
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], stdout, "", 0)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.untracked_files, ["welcome.py", "notes with spaces.txt"])

    def test_non_git_directory_is_categorized(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], "", "fatal: not a git repository (or any of the parent directories): .git\n", 128)
        self.assertIsNotNone(evidence)
        self.assertFalse(evidence.repository_detected)
        self.assertEqual(evidence.error_category, "not_git_repository")
        self.assertEqual(evidence.error_reason, "Not a Git repository.")

    def test_non_zero_git_exit_code_is_not_treated_as_success(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], "", "fatal: git status failed\n", 2)
        self.assertIsNotNone(evidence)
        self.assertFalse(evidence.success)
        self.assertEqual(evidence.error_category, "git_command_failed")

    def test_empty_stdout_means_no_untracked_only_after_successful_repository_detection(self) -> None:
        evidence = extract_git_untracked_evidence(["status", "--porcelain"], "", "", 0)
        self.assertIsNotNone(evidence)
        self.assertTrue(evidence.repository_detected)
        self.assertEqual(evidence.untracked_files, [])

    def test_final_answer_contains_exact_evidence_backed_paths(self) -> None:
        task = self._git_task()
        plan = self._git_plan()
        record = self._task_record(
            message="Command completed successfully.\nExit code: 0\n\nstdout:\n?? welcome.py\n?? sample_message.py",
            reference_fields={
                "terminal_operation": "git_untracked_files",
                "success": True,
                "repository_detected": True,
                "git_untracked_files": ["welcome.py", "sample_message.py"],
                "exit_code": 0,
                "output_truncated": False,
                "git_error_category": "",
                "git_error_reason": "",
            },
        )
        evaluation = evaluate_goal(plan, record, task=task)
        rendered = render_goal_evaluation(plan, evaluation, task=task)
        self.assertEqual(rendered, "Untracked files:\n- welcome.py\n- sample_message.py")

    def test_goal_evaluator_rejects_generic_answer_without_git_evidence(self) -> None:
        task = self._git_task()
        plan = self._git_plan()
        record = self._task_record(
            message="Command completed successfully.\nExit code: 0\n\nstdout:\nstatus checked",
            reference_fields={},
        )
        evaluation = evaluate_goal(plan, record, task=task)
        self.assertNotEqual(evaluation.status.value, "completed")

    def test_goal_evaluator_does_not_invent_filenames(self) -> None:
        task = self._git_task()
        plan = self._git_plan()
        record = self._task_record(
            message="Command completed successfully.\nExit code: 0\n\nstdout:\n?? actual.py\nFake summary mentions invented.py",
            reference_fields={
                "terminal_operation": "git_untracked_files",
                "success": True,
                "repository_detected": True,
                "git_untracked_files": ["actual.py"],
                "exit_code": 0,
                "output_truncated": False,
                "git_error_category": "",
                "git_error_reason": "",
            },
        )
        evaluation = evaluate_goal(plan, record, task=task)
        rendered = render_goal_evaluation(plan, evaluation, task=task)
        self.assertEqual(rendered, "Untracked files:\n- actual.py")
        self.assertNotIn("invented.py", rendered)

    def test_failed_git_status_renders_categorized_reason(self) -> None:
        task = self._git_task()
        plan = self._git_plan()
        record = self._task_record(
            message="Command failed.\nExit code: 128\n\nstderr:\nfatal: not a git repository (or any of the parent directories): .git",
            reference_fields={
                "terminal_operation": "git_untracked_files",
                "success": False,
                "repository_detected": False,
                "git_untracked_files": [],
                "exit_code": 128,
                "output_truncated": False,
                "git_error_category": "not_git_repository",
                "git_error_reason": "Not a Git repository.",
            },
            state=AgentLifecycleState.FAILED,
        )
        evaluation = evaluate_goal(plan, record, task=task)
        rendered = render_goal_evaluation(plan, evaluation, task=task)
        self.assertEqual(rendered, "Git status could not be completed.\nReason: Not a Git repository.")


if __name__ == "__main__":
    unittest.main()
