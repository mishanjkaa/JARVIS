from __future__ import annotations

from app.brain.planner.plan_models import AgentPlan
from app.brain.browser.risk import classify_browser_step
from app.brain.risk.models import PlanRiskAssessment, RiskLevel
from app.brain.terminal.policy import decision_from_arguments as terminal_decision_from_arguments
from app.brain.terminal.errors import TerminalError
import re
from urllib.parse import urlparse


def summarize_plan(plan: AgentPlan, assessment: PlanRiskAssessment | None = None) -> str:
    lines = ["Pending plan:"]
    step_levels = _effective_step_levels(plan)
    for index, step in enumerate(plan.steps, 1):
        description = step.user_visible_description or "Perform a registered action."
        lines.append(f"{index}. {description} [{step_levels[index - 1]}]")
        if step.tool_name == "terminal.execute":
            executable = step.arguments.get("executable", "")
            arguments = step.arguments.get("arguments", [])
            working_directory = step.arguments.get("working_directory", ".")
            if executable:
                lines.append(f"   executable: {executable}")
            if isinstance(arguments, list):
                lines.append(f"   args: {arguments}")
            lines.append(f"   working directory: {working_directory}")
        if step.tool_name == "browser.open_url":
            lines.append(f"   url: {step.arguments.get('url', '')}")
            lines.append(f"   wait until: {step.arguments.get('wait_until', 'domcontentloaded')}")
        if step.tool_name == "browser.input_text":
            target = (
                step.arguments.get("label_hint")
                or step.arguments.get("placeholder_hint")
                or step.arguments.get("name_hint")
                or step.arguments.get("control_type", "form control")
            )
            text_value = step.arguments.get("text")
            text_length = _redacted_text_length(text_value) if isinstance(text_value, str) else None
            if text_length is None:
                text_length = len(text_value) if isinstance(text_value, str) else 0
            lines.append(f"   target: {target}")
            lines.append(f"   text: [redacted text length={text_length}]")
        if step.tool_name == "browser.clear_input":
            target = (
                step.arguments.get("label_hint")
                or step.arguments.get("placeholder_hint")
                or step.arguments.get("name_hint")
                or step.arguments.get("control_type", "form control")
            )
            lines.append(f"   target: {target}")
        if step.tool_name == "browser.submit_form":
            target = (
                step.arguments.get("label_hint")
                or step.arguments.get("placeholder_hint")
                or step.arguments.get("name_hint")
                or step.arguments.get("submit_text_hint")
                or step.arguments.get("form_text_hint")
                or "requested form"
            )
            lines.append(f"   target: {target}")
            current_origin = _current_browser_origin(plan, index)
            if current_origin:
                lines.append(f"   current origin: {current_origin}")
            if step.arguments.get("allowed_destination_origin"):
                lines.append(f"   approved destination origin: {step.arguments.get('allowed_destination_origin')}")
            if step.arguments.get("page_context"):
                lines.append(f"   context: {step.arguments.get('page_context')}")
        if step.tool_name == "browser.take_screenshot":
            lines.append(f"   path: {step.arguments.get('path', '')}")
    if assessment is not None:
        if assessment.level == RiskLevel.HIGH:
            lines.append("HIGH RISK operation.")
            lines.append("Approval required.")
        elif assessment.level == RiskLevel.MEDIUM:
            lines.append("This plan is MEDIUM risk.")
            lines.append("Approve?")
        else:
            lines.append("This plan is LOW risk.")
        for reason in _approval_reasons(plan, assessment)[:4]:
            lines.append(f"- {reason}.")
    elif any(step.risk_level == "persistent_write" for step in plan.steps):
        lines.append("Approval required because this plan writes persistent data.")
    return "\n".join(lines)


def _effective_step_levels(plan: AgentPlan) -> list[str]:
    labels: list[str] = []
    for step in plan.steps:
        label = _fallback_step_level(step.risk_level)
        browser_assessment = classify_browser_step(step.tool_name, step.arguments)
        if browser_assessment is not None:
            labels.append(browser_assessment[0].value)
            continue
        if step.tool_name == "terminal.execute":
            try:
                decision = terminal_decision_from_arguments(step.arguments)
            except TerminalError:
                labels.append("HIGH")
                continue
            labels.append(str(decision.risk_level).upper())
            continue
        labels.append(label)
    return labels


def _fallback_step_level(risk_level: str) -> str:
    normalized = str(risk_level or "").strip().lower()
    mapping = {
        "read_only": "LOW",
        "local_safe": "LOW",
        "persistent_write": "MEDIUM",
        "external_navigation": "MEDIUM",
        "sensitive": "HIGH",
        "destructive": "HIGH",
    }
    return mapping.get(normalized, normalized.upper() or "UNKNOWN")


def _approval_reasons(plan: AgentPlan, assessment: PlanRiskAssessment) -> list[str]:
    prioritized: list[str] = []
    for step in plan.steps:
        browser_assessment = classify_browser_step(step.tool_name, step.arguments)
        if browser_assessment is not None and browser_assessment[0] == assessment.level:
            prioritized.append(browser_assessment[1])
            continue
        if step.tool_name == "terminal.execute":
            try:
                decision = terminal_decision_from_arguments(step.arguments)
            except TerminalError:
                continue
            if str(decision.risk_level).upper() == assessment.level.value:
                prioritized.append(decision.reason)
    combined = prioritized + list(assessment.reasons)
    deduped: list[str] = []
    for reason in combined:
        text = str(reason).strip().rstrip(".")
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _current_browser_origin(plan: AgentPlan, step_index: int) -> str:
    for candidate in reversed(plan.steps[: step_index - 1]):
        if candidate.tool_name not in {"browser.open_url", "browser.open_new_tab"}:
            continue
        url = str(candidate.arguments.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    return ""


_REDACTED_TEXT_PATTERN = re.compile(r"\[redacted text length=(\d+)\]")


def _redacted_text_length(value: str) -> int | None:
    if not isinstance(value, str):
        return None
    match = _REDACTED_TEXT_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    return int(match.group(1))
