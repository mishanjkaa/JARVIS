from __future__ import annotations

from typing import Any

from app.brain.browser.models import ALLOWED_WAIT_UNTIL
from app.brain.intelligence.models import GoalEvaluationStatus, TaskIntent, ToolCatalogEntry
from app.brain.planner.result_references import allows_result_reference
from app.brain.terminal.policy import supported_operation_types
from app.brain.tools.validators import ALLOWED_APP_NAMES, ALLOWED_FOLDER_NAMES

_CONFIDENCE_ENUM = ["low", "medium", "high"]
_AMBIGUITY_ENUM = ["low", "medium", "high"]
_REFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["from_step", "field"],
    "additionalProperties": False,
    "properties": {
        "from_step": {"type": "integer", "minimum": 1},
        "field": {"type": "string"},
    },
}


TASK_INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "intent",
        "confidence",
        "goal",
        "expected_result",
        "referenced_paths",
        "execution_requested",
        "ambiguity_level",
        "language",
        "constraints",
        "requested_artifacts",
        "requested_contents",
        "requested_output_texts",
        "requested_summary",
        "read_only_task",
        "destructive_scope_unclear",
        "requires_execution",
        "requires_verification",
        "requires_stdout_match",
        "requires_tests",
        "requires_code_write",
        "requested_operation",
    ],
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "enum": [item.value for item in TaskIntent]},
        "confidence": {"type": "string", "enum": _CONFIDENCE_ENUM},
        "goal": {"type": "string"},
        "expected_result": {"type": "string"},
        "referenced_paths": {"type": "array", "items": {"type": "string"}},
        "execution_requested": {"type": "boolean"},
        "ambiguity_level": {"type": "string", "enum": _AMBIGUITY_ENUM},
        "language": {"type": "string"},
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "value"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                },
            },
        },
        "requested_artifacts": {"type": "array", "items": {"type": "string"}},
        "requested_contents": {"type": "array", "items": {"type": "string"}},
        "requested_output_texts": {"type": "array", "items": {"type": "string"}},
        "requested_summary": {"type": "boolean"},
        "read_only_task": {"type": "boolean"},
        "destructive_scope_unclear": {"type": "boolean"},
        "requires_execution": {"type": "boolean"},
        "requires_verification": {"type": "boolean"},
        "requires_stdout_match": {"type": "boolean"},
        "requires_tests": {"type": "boolean"},
        "requires_code_write": {"type": "boolean"},
        "requested_operation": {"type": "string"},
    },
}


GOAL_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "summary", "evidence"],
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": [item.value for item in GoalEvaluationStatus]},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}


def build_plan_output_schema(tool_catalog: list[ToolCatalogEntry], max_steps: int) -> dict[str, Any]:
    step_variants = [_step_schema_for_tool(entry) for entry in tool_catalog if entry.enabled]
    return {
        "type": "object",
        "required": ["goal", "success_criteria", "steps"],
        "additionalProperties": False,
        "properties": {
            "goal": {"type": "string"},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "steps": {
                "type": "array",
                "maxItems": max_steps,
                "items": {"oneOf": step_variants} if step_variants else {"type": "object"},
            },
        },
    }


def validate_structured_payload(payload: Any, schema: dict[str, Any]) -> None:
    _validate_value(payload, schema, path="$")


def _validate_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        _validate_one_of(value, one_of, path=path)
        return

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: expected {schema['const']!r}")

    schema_type = schema.get("type")
    if schema_type == "object":
        _validate_object(value, schema, path=path)
    elif schema_type == "array":
        _validate_array(value, schema, path=path)
    elif schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean")
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path}: expected integer")
        minimum = schema.get("minimum")
        if isinstance(minimum, int) and value < minimum:
            raise ValueError(f"{path}: integer below minimum")
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: expected number")

    allowed_values = schema.get("enum")
    if isinstance(allowed_values, list) and value not in allowed_values:
        raise ValueError(f"{path}: invalid enum value")


def _validate_object(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    for key in required:
        if key not in value:
            raise ValueError(f"{path}.{key}: missing required field")
    if schema.get("additionalProperties", True) is False:
        extra_keys = set(value) - set(properties)
        if extra_keys:
            extra = sorted(extra_keys)[0]
            raise ValueError(f"{path}.{extra}: unexpected field")
    for key, property_schema in properties.items():
        if key not in value or not isinstance(property_schema, dict):
            continue
        _validate_value(value[key], property_schema, path=f"{path}.{key}")


def _validate_array(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected array")
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        raise ValueError(f"{path}: too many items")
    item_schema = schema.get("items")
    if not isinstance(item_schema, dict):
        return
    for index, item in enumerate(value):
        _validate_value(item, item_schema, path=f"{path}[{index}]")


def _step_schema_for_tool(entry: ToolCatalogEntry) -> dict[str, Any]:
    argument_properties: dict[str, Any] = {}
    required_arguments: list[str] = []
    for argument_name, spec in entry.argument_schema.items():
        argument_properties[argument_name] = _argument_property_schema(entry.name, argument_name, spec)
        if spec.get("required", True):
            required_arguments.append(argument_name)
    return {
        "type": "object",
        "required": ["tool", "arguments", "description", "depends_on", "expected_result"],
        "additionalProperties": False,
        "properties": {
            "tool": {"type": "string", "const": entry.name},
            "arguments": {
                "type": "object",
                "properties": argument_properties,
                "required": required_arguments,
                "additionalProperties": False,
            },
            "description": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "expected_result": {"type": "string"},
        },
    }


def _argument_property_schema(tool_name: str, argument_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    field_type = spec.get("type")
    if argument_name == "operation_type":
        return _maybe_with_reference(tool_name, argument_name, {"type": "string", "enum": list(supported_operation_types())})
    if argument_name == "wait_until":
        return _maybe_with_reference(tool_name, argument_name, {"type": "string", "enum": list(ALLOWED_WAIT_UNTIL)})
    if field_type == "string_list":
        item_schema: dict[str, Any] = {"type": "string"}
        return {"type": "array", "items": item_schema}
    if field_type == "integer":
        return _maybe_with_reference(tool_name, argument_name, {"type": "integer"})
    if field_type == "bool":
        return _maybe_with_reference(tool_name, argument_name, {"type": "boolean"})
    if field_type == "number":
        return _maybe_with_reference(tool_name, argument_name, {"type": "number"})
    if field_type == "app":
        return _maybe_with_reference(tool_name, argument_name, {"type": "string", "enum": sorted(ALLOWED_APP_NAMES)})
    if field_type == "folder":
        return _maybe_with_reference(tool_name, argument_name, {"type": "string", "enum": sorted(ALLOWED_FOLDER_NAMES)})
    return _maybe_with_reference(tool_name, argument_name, {"type": "string"})


def _maybe_with_reference(tool_name: str, argument_name: str, base_schema: dict[str, Any]) -> dict[str, Any]:
    if not allows_result_reference(tool_name, argument_name):
        return base_schema
    return {"oneOf": [base_schema, _REFERENCE_SCHEMA]}


def _validate_one_of(value: Any, variants: list[Any], *, path: str) -> None:
    matching_variants = _matching_const_variants(value, variants)
    if matching_variants is not None:
        if not matching_variants:
            raise ValueError(f"{path}.tool: invalid enum value")
        errors: list[str] = []
        for variant in matching_variants:
            try:
                _validate_value(value, variant, path=path)
                return
            except ValueError as error:
                errors.append(str(error))
        raise ValueError(errors[0] if errors else f"{path}: no valid schema variant")

    errors: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        try:
            _validate_value(value, variant, path=path)
            return
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError(errors[0])
    raise ValueError(f"{path}: no valid schema variant")


def _matching_const_variants(value: Any, variants: list[Any]) -> list[dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    tool_value = value.get("tool")
    if not isinstance(tool_value, str):
        return None
    const_variants: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        properties = variant.get("properties")
        if not isinstance(properties, dict):
            return None
        tool_property = properties.get("tool")
        if not isinstance(tool_property, dict) or "const" not in tool_property:
            return None
        const_variants.append(variant)
    return [variant for variant in const_variants if variant["properties"]["tool"].get("const") == tool_value]
