from __future__ import annotations

import re
import shlex

from app.brain.intent.models import IntentClassification
from app.brain.planner.goal import Goal
from app.brain.planner.plan_models import AgentPlan


def build_goal(classification: IntentClassification, requested_outcome: str) -> Goal:
    return Goal(user_request_category=classification.intent.value, requested_outcome=requested_outcome[:200], maximum_steps=5)


def build_plan(goal: Goal, steps: tuple = ()) -> AgentPlan:
    return AgentPlan(steps=list(steps), goal=goal, original_request=goal.requested_outcome)

from app.brain.intent.classifier import IntentClassifier
from app.brain.intent.models import IntentCategory
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.terminal.policy import default_terminal_working_directory

_CALCULATION_WITH_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(?:calculate|calc|work out)\s+(?P<expression>.+?)\s+\b(?:and|then)\b\s+(?P<follow_up>.+)\s*$",
    re.IGNORECASE,
)
_PERSISTENT_FOLLOWUP_PATTERN = re.compile(
    r"\b(save|record|note|make a note|write (?:it|that|the result|the answer) down)\b",
    re.IGNORECASE,
)
_FILESYSTEM_SEQUENCE_SPLIT = re.compile(r"\s+\b(?:and|then)\b\s+", re.IGNORECASE)
_CALCULATION_PREFIX_PATTERN = re.compile(r"^\s*(?:calculate|calc|work out|what is)\s+(.+?)\s*$", re.IGNORECASE)
_WRITE_QUOTED_PATTERN = re.compile(r'^write\s+"((?:[^"\\]|\\.)*)"\s+to\s+(.+)$', re.IGNORECASE)
_WRITE_UNQUOTED_PATTERN = re.compile(r"^write\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)
_APPEND_QUOTED_PATTERN = re.compile(r'^append\s+"((?:[^"\\]|\\.)*)"\s+to\s+(.+)$', re.IGNORECASE)
_APPEND_UNQUOTED_PATTERN = re.compile(r"^append\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)
_FORBIDDEN_TERMINAL_EXECUTABLES = {
    "bash",
    "cmd",
    "format",
    "git",
    "pip",
    "pip3",
    "powershell",
    "pwsh",
    "python",
    "python.exe",
    "py",
    "sh",
    "shutdown",
    "unknown-program",
    "wsl",
}


class PlannerV2:
    """Create a simple, validated plan from a user request."""

    def __init__(self) -> None:
        self.classifier = IntentClassifier()

    def build_plan(self, raw_input: str) -> AgentPlan:
        result = self.classifier.classify(raw_input)
        calculation_with_follow_up = _extract_calculation_with_persistent_follow_up(raw_input)
        if calculation_with_follow_up is not None:
            expression = calculation_with_follow_up
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="calculator.calculate", arguments={"expression": expression}, risk_level="read_only", user_visible_description="Calculate an expression."),
                AgentStep(step_id=2, tool_name="notes.create", arguments={"text": {"from_step": 1, "field": "display_value"}}, depends_on=[1], risk_level="persistent_write", user_visible_description="Create a note using the result."),
            ], original_request=raw_input)
        terminal_plan = _build_terminal_plan(raw_input)
        if terminal_plan is not None:
            terminal_plan.original_request = raw_input
            return terminal_plan
        filesystem_plan = _build_filesystem_plan(raw_input)
        if filesystem_plan is not None:
            filesystem_plan.original_request = raw_input
            return filesystem_plan
        if result.intent == IntentCategory.CALCULATION:
            expression = _extract_calculation_expression(raw_input)
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="calculator.calculate", arguments={"expression": expression}, risk_level="read_only", user_visible_description="Calculate an expression."),
            ], original_request=raw_input)
        if result.intent == IntentCategory.NOTES_WRITE:
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="notes.create", arguments={"text": raw_input}, risk_level="persistent_write", user_visible_description="Create a note."),
            ], original_request=raw_input)
        if result.intent == IntentCategory.TASKS_WRITE:
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="tasks.create", arguments={"text": raw_input}, risk_level="persistent_write", user_visible_description="Create a task."),
            ], original_request=raw_input)
        if result.intent == IntentCategory.WEB_SEARCH:
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="internet.search", arguments={"query": raw_input}, risk_level="external_navigation", user_visible_description="Search the web."),
            ], original_request=raw_input)
        if result.intent == IntentCategory.OPEN_FOLDER:
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="computer.open_known_folder", arguments={"folder": "downloads"}, risk_level="local_safe", user_visible_description="Open a folder."),
            ], original_request=raw_input)
        if result.intent == IntentCategory.OPEN_APPLICATION:
            return AgentPlan(steps=[
                AgentStep(step_id=1, tool_name="computer.open_application", arguments={"application": "calculator"}, risk_level="local_safe", user_visible_description="Open an application."),
            ], original_request=raw_input)
        return AgentPlan(steps=[
            AgentStep(step_id=1, tool_name="system.system_info", arguments={}, risk_level="read_only", user_visible_description="Provide system information."),
        ], original_request=raw_input)


def create_plan_from_request(raw_input: str) -> AgentPlan:
    return PlannerV2().build_plan(raw_input)


def _extract_calculation_with_persistent_follow_up(raw_input: str) -> str | None:
    match = _CALCULATION_WITH_FOLLOWUP_PATTERN.match(raw_input)
    if match is None:
        return None
    if _PERSISTENT_FOLLOWUP_PATTERN.search(match.group("follow_up")) is None:
        return None
    expression = match.group("expression").strip()
    return expression or None


def _extract_calculation_expression(raw_input: str) -> str:
    match = _CALCULATION_PREFIX_PATTERN.match(raw_input)
    expression = raw_input.strip() if match is None else match.group(1).strip()
    expression = re.sub(r"\bmultiplied by\b", "*", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\btimes\b", "*", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bplus\b", "+", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bminus\b", "-", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bdivided by\b", "/", expression, flags=re.IGNORECASE)
    return " ".join(expression.split())


def _build_filesystem_plan(raw_input: str) -> AgentPlan | None:
    steps: list[AgentStep] = []
    for index, part in enumerate(_FILESYSTEM_SEQUENCE_SPLIT.split(raw_input.strip()), 1):
        parsed = _parse_filesystem_step(part)
        if parsed is None:
            if index == 1:
                return None
            return None
        steps.append(
            AgentStep(
                step_id=index,
                tool_name=parsed["tool_name"],
                arguments=parsed["arguments"],
                risk_level=parsed["risk_level"],
                user_visible_description=parsed["description"],
            )
        )
    return AgentPlan(steps=steps) if steps else None


def _parse_filesystem_step(raw_part: str) -> dict[str, object] | None:
    value = raw_part.strip()
    lower = value.lower()

    create_directory = re.match(r"^(?:create|make)\s+(?:a\s+)?(?:folder|directory)(?:\s+named)?\s+(.+)$", value, re.IGNORECASE)
    if create_directory:
        path = create_directory.group(1).strip().strip('"')
        return {"tool_name": "filesystem.create_directory", "arguments": {"path": path}, "risk_level": "persistent_write", "description": "Create a directory."}

    create_file = re.match(r"^(?:create|make)\s+(?:a\s+)?(?:file\s+)?([^\"]+\.\w+|[A-Za-z0-9._\\/-]+)$", value, re.IGNORECASE)
    if create_file and any(token in lower for token in ("file", ".txt", ".md", ".json", ".log")):
        path = create_file.group(1).strip().strip('"')
        return {"tool_name": "filesystem.create_text_file", "arguments": {"path": path}, "risk_level": "persistent_write", "description": "Create a text file."}

    create_readme = re.match(r"^(?:create|make)\s+(readme(?:\.md)?)$", value, re.IGNORECASE)
    if create_readme:
        path = create_readme.group(1).strip().strip('"')
        return {"tool_name": "filesystem.create_text_file", "arguments": {"path": path}, "risk_level": "persistent_write", "description": "Create a text file."}

    write_arguments = _parse_write_or_append(value, verb="write")
    if write_arguments is not None:
        return {"tool_name": "filesystem.write_text_file", "arguments": write_arguments, "risk_level": "persistent_write", "description": "Write text to a file."}

    append_arguments = _parse_write_or_append(value, verb="append")
    if append_arguments is not None:
        return {"tool_name": "filesystem.append_text_file", "arguments": append_arguments, "risk_level": "persistent_write", "description": "Append text to a file."}

    read_match = re.match(r"^(?:read|show)\s+(?:file\s+)?(.+)$", value, re.IGNORECASE)
    if read_match and not lower.startswith("show my"):
        path = read_match.group(1).strip().strip('"')
        return {"tool_name": "filesystem.read_text_file", "arguments": {"path": path}, "risk_level": "read_only", "description": "Read a text file."}

    rename_match = re.match(r"^rename\s+(.+?)\s+to\s+(.+)$", value, re.IGNORECASE)
    if rename_match:
        return {
            "tool_name": "filesystem.rename_path",
            "arguments": {"source_path": rename_match.group(1).strip().strip('"'), "destination_path": rename_match.group(2).strip().strip('"')},
            "risk_level": "persistent_write",
            "description": "Rename a path.",
        }

    move_match = re.match(r"^move\s+(.+?)\s+to\s+(.+)$", value, re.IGNORECASE)
    if move_match:
        return {
            "tool_name": "filesystem.move_path",
            "arguments": {"source_path": move_match.group(1).strip().strip('"'), "destination_path": move_match.group(2).strip().strip('"')},
            "risk_level": "persistent_write",
            "description": "Move a path.",
        }

    copy_match = re.match(r"^copy\s+(.+?)(?:\s+to\s+(.+))?$", value, re.IGNORECASE)
    if copy_match:
        arguments = {"source_path": copy_match.group(1).strip().strip('"')}
        if copy_match.group(2):
            arguments["destination_path"] = copy_match.group(2).strip().strip('"')
        return {"tool_name": "filesystem.copy_path", "arguments": arguments, "risk_level": "persistent_write", "description": "Copy a path."}

    delete_match = re.match(r"^(?:delete|remove)\s+(?:file\s+)?(.+)$", value, re.IGNORECASE)
    if delete_match:
        return {"tool_name": "filesystem.delete_path", "arguments": {"path": delete_match.group(1).strip().strip('"')}, "risk_level": "persistent_write", "description": "Soft delete a path."}

    list_match = re.match(r"^(?:list|show)\s+(?:directory|folder|files)(?:\s+(.+))?$", value, re.IGNORECASE)
    if list_match:
        arguments: dict[str, object] = {}
        if list_match.group(1):
            arguments["path"] = list_match.group(1).strip().strip('"')
        return {"tool_name": "filesystem.list_directory", "arguments": arguments, "risk_level": "read_only", "description": "List a directory."}

    exists_match = re.match(r"^(?:does|check whether)\s+(.+?)\s+exist[s]?$", value, re.IGNORECASE)
    if exists_match:
        return {"tool_name": "filesystem.exists", "arguments": {"path": exists_match.group(1).strip().strip('"')}, "risk_level": "read_only", "description": "Check whether a path exists."}

    metadata_match = re.match(r"^(?:metadata|details)\s+(?:for\s+)?(.+)$", value, re.IGNORECASE)
    if metadata_match:
        return {"tool_name": "filesystem.metadata", "arguments": {"path": metadata_match.group(1).strip().strip('"')}, "risk_level": "read_only", "description": "Read path metadata."}

    return None


def _parse_write_or_append(value: str, *, verb: str) -> dict[str, str] | None:
    if verb == "write":
        quoted_pattern = _WRITE_QUOTED_PATTERN
        unquoted_pattern = _WRITE_UNQUOTED_PATTERN
    else:
        quoted_pattern = _APPEND_QUOTED_PATTERN
        unquoted_pattern = _APPEND_UNQUOTED_PATTERN

    quoted_match = quoted_pattern.match(value)
    if quoted_match:
        return {"text": quoted_match.group(1).replace('\\"', '"').strip(), "path": quoted_match.group(2).strip().strip('"')}

    unquoted_match = unquoted_pattern.match(value)
    if unquoted_match:
        return {"text": unquoted_match.group(1).strip(), "path": unquoted_match.group(2).strip().strip('"')}

    fallback = re.match(rf"^{verb}\s+\"?([^\"]+)\"?$", value, re.IGNORECASE)
    if fallback:
        return {"text": fallback.group(1).strip()}
    return None


def _build_terminal_plan(raw_input: str) -> AgentPlan | None:
    request = _parse_terminal_request(raw_input)
    if request is None:
        return None
    return AgentPlan(steps=[
        AgentStep(
            step_id=1,
            tool_name="terminal.execute",
            arguments=request,
            risk_level="local_safe",
            user_visible_description="Run a terminal command.",
        )
    ])


def _parse_terminal_request(raw_input: str) -> dict[str, object] | None:
    value = raw_input.strip()
    lower = value.lower()

    if lower in {"run unit tests", "run the unit tests"}:
        return _terminal_arguments(raw_input, "python", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], operation_type="python")

    if lower.startswith("show git status"):
        return _terminal_arguments(raw_input, "git", ["status"], operation_type="git_read_only")

    if lower.startswith("install python package ") or lower.startswith("install package "):
        package_value = value.split(" ", 3)[-1].strip()
        package_specs = package_value.split()
        return _terminal_arguments(raw_input, "python", ["-m", "pip", "install", *package_specs], operation_type="package_install")

    command_text = value[4:].strip() if lower.startswith("run ") else value
    try:
        tokens = shlex.split(command_text)
    except ValueError:
        return None
    if not tokens:
        return None

    executable = tokens[0].lower()
    arguments = tokens[1:]
    if executable == "python":
        if arguments[:3] == ["-m", "pip", "install"]:
            return _terminal_arguments(raw_input, "python", arguments, operation_type="package_install")
        return _terminal_arguments(raw_input, "python", arguments, operation_type="python")
    if executable == "py":
        if arguments[:3] == ["-m", "pip", "install"]:
            return _terminal_arguments(raw_input, "python", arguments, operation_type="package_install")
        return _terminal_arguments(raw_input, "python", arguments, operation_type="python")
    if executable == "git":
        return _terminal_arguments(raw_input, "git", arguments, operation_type="git_read_only")
    if executable in {"pip", "pip3"} and arguments and arguments[0].lower() == "install":
        return _terminal_arguments(raw_input, "python", ["-m", "pip", *arguments], operation_type="package_install")
    if executable in _FORBIDDEN_TERMINAL_EXECUTABLES or executable.endswith(".exe"):
        return _terminal_arguments(raw_input, executable, arguments, operation_type="unsupported")
    return None


def _terminal_arguments(raw_input: str, executable: str, arguments: list[str], *, operation_type: str) -> dict[str, object]:
    return {
        "executable": executable,
        "arguments": arguments,
        "working_directory": default_terminal_working_directory(),
        "timeout_seconds": 30,
        "operation_type": operation_type,
        "raw_command": raw_input,
    }
