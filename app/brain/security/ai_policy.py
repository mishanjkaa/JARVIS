from __future__ import annotations

from typing import Any

RISK_LEVELS = {"read_only", "local_safe", "external_navigation", "persistent_write", "sensitive", "destructive"}
PERSISTENT_WRITE_TOOLS = {
    "memory.remember", "memory.forget", "notes.create", "notes.delete", "tasks.create", "tasks.complete", "tasks.delete",
    "filesystem.create_directory", "filesystem.create_text_file", "filesystem.write_text_file", "filesystem.append_text_file",
    "filesystem.rename_path", "filesystem.copy_path", "filesystem.move_path", "filesystem.delete_path",
}
EXTERNAL_TOOLS = {"internet.search", "internet.open_youtube", "internet.open_github"}
OPEN_TOOLS = {"computer.open_application", "computer.open_known_folder"}


class PolicyError(ValueError):
    pass


def validate_risk_level(risk_level: Any) -> str:
    if not isinstance(risk_level, str):
        raise PolicyError("risk level must be a string")
    normalized = risk_level.strip().lower()
    if normalized not in RISK_LEVELS:
        raise PolicyError("unknown risk level")
    return normalized


def validate_plan_step(step: Any) -> None:
    if not isinstance(step, dict):
        raise PolicyError("step must be a mapping")
    if not isinstance(step.get("tool_name"), str) or not step["tool_name"].strip():
        raise PolicyError("step tool_name missing")
    if not isinstance(step.get("arguments"), dict):
        raise PolicyError("step arguments invalid")
    if "reference" in step and step["reference"] is not None:
        reference = step["reference"]
        if not isinstance(reference, dict):
            raise PolicyError("reference must be a mapping")
        if set(reference.keys()) != {"from_step", "field"}:
            raise PolicyError("reference must use from_step and field")
        if not isinstance(reference.get("from_step"), int):
            raise PolicyError("reference from_step must be an int")
        if reference.get("field") != "display_value":
            raise PolicyError("reference field not allowlisted")


def validate_plan(plan: Any, *, max_steps: int = 5) -> None:
    if not isinstance(plan, list):
        raise PolicyError("plan must be a list")
    if len(plan) > max_steps:
        raise PolicyError("plan too long")
    seen_ids = set()
    for index, step in enumerate(plan):
        validate_plan_step(step)
        step_id = step.get("id")
        if not isinstance(step_id, int):
            raise PolicyError("step id must be an int")
        if step_id in seen_ids:
            raise PolicyError("duplicate step id")
        seen_ids.add(step_id)
        tool_name = step.get("tool_name")
        if tool_name in {"power.request_lock", "power.request_restart", "power.request_shutdown"}:
            raise PolicyError("power actions must use confirmation wrapper")
        if step.get("reference") is not None:
            reference = step["reference"]
            if reference.get("from_step", 0) > index + 1:
                raise PolicyError("reference points forward")
            if reference.get("from_step", 0) <= 0:
                raise PolicyError("reference points to invalid step")
    if len(seen_ids) != len(plan):
        raise PolicyError("invalid step identifiers")


def validate_plan_policy(plan: Any, *, max_steps: int = 5, max_persistent_writes: int = 2, max_external_actions: int = 2) -> None:
    """Reject unsafe plan combinations before any step can execute."""
    validate_plan(plan, max_steps=max_steps)
    persistent = 0
    external = 0
    opens = 0
    power = 0
    for step in plan:
        tool_name = step.get("tool_name")
        if not isinstance(tool_name, str) or tool_name.startswith(("config.", "plugin.")):
            raise PolicyError("tool is not permitted in an AI plan")
        if tool_name.startswith("power."):
            power += 1
        if tool_name in PERSISTENT_WRITE_TOOLS:
            persistent += 1
        if tool_name in EXTERNAL_TOOLS:
            external += 1
        if tool_name in OPEN_TOOLS:
            opens += 1
        arguments = step.get("arguments", {})
        if not isinstance(arguments, dict):
            raise PolicyError("arguments invalid")
        if tool_name == "terminal.execute":
            for value in arguments.values():
                if isinstance(value, str) and ("${" in value or "import " in value or "__" in value or "shell" in value.lower()):
                    raise PolicyError("unsafe argument")
    if persistent > max_persistent_writes or external > max_external_actions or opens > 2:
        raise PolicyError("plan action limit exceeded")
    if power and len(plan) != 1:
        raise PolicyError("power request cannot be mixed")
