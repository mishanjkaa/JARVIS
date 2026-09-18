from __future__ import annotations

from pathlib import Path
from typing import Any

from app.brain.browser.risk import classify_browser_step
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.path_policy import resolve_path
from app.brain.planner.plan_models import AgentPlan
from app.brain.risk.models import PlanRiskAssessment, PolicyOutcome, RiskLevel
from app.brain.risk.rules import HIGH_RISK_PREFIXES, HIGH_RISK_TOOLS, LOW_RISK_TOOLS, MEDIUM_RISK_TOOLS
from app.brain.terminal.policy import decision_from_arguments as terminal_decision_from_arguments
from app.brain.terminal.errors import TerminalError
from config.config_loader import load_config


def analyze_plan(plan: AgentPlan, config: dict[str, Any] | None = None) -> PlanRiskAssessment:
    effective_config = _effective_config(config)
    reasons: list[str] = []
    level = RiskLevel.LOW
    policy_outcome = PolicyOutcome.ALLOWED

    filesystem_write_count = 0
    filesystem_move_count = 0
    filesystem_delete_count = 0

    for step in plan.steps:
        tool_name = step.tool_name
        if _is_high_risk_tool(tool_name):
            level = RiskLevel.HIGH
            reasons.append("contains a high-risk operation")
            break
        if tool_name == "terminal.execute":
            try:
                decision = terminal_decision_from_arguments(step.arguments, config=effective_config)
                if not decision.allowed:
                    policy_outcome = PolicyOutcome.REJECTED
                    level = RiskLevel.HIGH
                    reasons = [decision.rejection_reason or decision.reason]
                    break
                terminal_level = RiskLevel(decision.risk_level)
                level = _elevate(level, terminal_level)
                reasons.append(decision.reason)
            except TerminalError as error:
                policy_outcome = PolicyOutcome.REJECTED
                level = RiskLevel.HIGH
                reasons = [str(error).strip() or "That command is not allowed."]
                break
            continue
        browser_assessment = classify_browser_step(tool_name, step.arguments)
        if browser_assessment is not None:
            browser_level, browser_reason = browser_assessment
            level = _elevate(level, browser_level)
            reasons.append(browser_reason)
            continue
        if tool_name == "filesystem.delete_path":
            filesystem_delete_count += 1
            level = _elevate(level, RiskLevel.MEDIUM)
        elif tool_name == "filesystem.move_path":
            filesystem_move_count += 1
        elif tool_name.startswith("filesystem.") and tool_name not in {
            "filesystem.exists",
            "filesystem.list_directory",
            "filesystem.metadata",
            "filesystem.read_text_file",
        }:
            filesystem_write_count += 1
        elif tool_name in MEDIUM_RISK_TOOLS:
            level = _elevate(level, RiskLevel.MEDIUM)
        elif tool_name not in LOW_RISK_TOOLS:
            level = _elevate(level, RiskLevel.HIGH)
            reasons.append("contains an unknown or future-sensitive operation")
            break

    if policy_outcome == PolicyOutcome.REJECTED:
        return PlanRiskAssessment(
            level=level,
            auto_execute=False,
            requires_approval=False,
            project_scoped=False,
            policy_outcome=policy_outcome,
            reasons=reasons,
        )

    if level != RiskLevel.HIGH:
        if filesystem_delete_count:
            reasons.append("includes soft delete operations")
            level = _elevate(level, RiskLevel.MEDIUM)
        if filesystem_move_count > 1:
            reasons.append("moves multiple files")
            level = _elevate(level, RiskLevel.MEDIUM)
        if filesystem_write_count > 4:
            reasons.append("modifies many project files")
            level = _elevate(level, RiskLevel.MEDIUM)

    project_scoped = _is_project_scoped(plan)
    auto_execute = _auto_execute(level, project_scoped, effective_config)
    requires_approval = not auto_execute
    policy_outcome = PolicyOutcome.ALLOWED if auto_execute else PolicyOutcome.APPROVAL_REQUIRED

    if level == RiskLevel.LOW and not reasons:
        reasons.append("contains only low-risk operations")
    elif level == RiskLevel.MEDIUM and not reasons:
        reasons.append("contains medium-risk operations")
    elif level == RiskLevel.HIGH and not reasons:
        reasons.append("contains high-risk operations")

    return PlanRiskAssessment(
        level=level,
        auto_execute=auto_execute,
        requires_approval=requires_approval,
        project_scoped=project_scoped,
        policy_outcome=policy_outcome,
        reasons=reasons,
    )


def _effective_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    merged = get_effective_runtime_config()
    return merged


def _is_high_risk_tool(tool_name: str) -> bool:
    if tool_name in HIGH_RISK_TOOLS:
        return True
    return any(tool_name.startswith(prefix) for prefix in HIGH_RISK_PREFIXES)


def _elevate(current: RiskLevel, new: RiskLevel) -> RiskLevel:
    order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
    return new if order[new] > order[current] else current


def _auto_execute(level: RiskLevel, project_scoped: bool, config: dict[str, Any]) -> bool:
    if level == RiskLevel.LOW:
        return bool(config.get("auto_execute_low_risk", True))
    if level == RiskLevel.MEDIUM:
        return bool(config.get("developer_mode", False)) and bool(config.get("auto_execute_medium_project", True)) and project_scoped
    return False


def _is_project_scoped(plan: AgentPlan) -> bool:
    if not plan.steps:
        return True
    for step in plan.steps:
        if step.tool_name.startswith("filesystem."):
            if not _filesystem_step_is_project_scoped(step.arguments):
                return False
            continue
        if step.tool_name == "terminal.execute":
            try:
                decision = terminal_decision_from_arguments(step.arguments)
            except TerminalError:
                return False
            if not decision.allowed or not decision.project_scoped:
                return False
            continue
        if step.tool_name.startswith("browser."):
            continue
        if step.tool_name.startswith("vision."):
            if not _vision_step_is_project_scoped(step.arguments):
                return False
            continue
        if step.tool_name in {
            "calculator.calculate",
            "computer.open_application",
            "computer.open_known_folder",
            "internet.search",
            "memory.recall",
            "notes.list",
            "system.get_date",
            "system.get_time",
            "system.system_info",
            "tasks.list",
        }:
            continue
        return False
    return True


def _filesystem_step_is_project_scoped(arguments: dict[str, Any]) -> bool:
    path_fields = ("path", "source_path", "destination_path")
    for field_name in path_fields:
        value = arguments.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            return False
        try:
            resolved = resolve_path(value, prefer_directory=False)
        except FilesystemPathError:
            return False
        if not _is_within_root(resolved.absolute_path, resolved.root):
            return False
    return True


def _vision_step_is_project_scoped(arguments: dict[str, Any]) -> bool:
    capture_id = arguments.get("capture_id")
    if isinstance(capture_id, (str, dict)):
        return True
    path = arguments.get("path")
    if not isinstance(path, str):
        return False
    try:
        resolved = resolve_path(path, prefer_directory=False)
    except FilesystemPathError:
        return False
    return _is_within_root(resolved.absolute_path, resolved.root)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
