from __future__ import annotations

import re
from typing import Any

CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
ALLOWED_APP_NAMES = {"notepad", "calculator", "browser"}
ALLOWED_FOLDER_NAMES = {"desktop", "downloads", "documents", "pictures", "music", "videos"}
MAX_TEXT_LENGTH = 200
MAX_QUERY_LENGTH = 160
MAX_PATH_LENGTH = 260
MAX_LIST_ITEMS = 20


def validate_text(value: Any, *, field_name: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{field_name} too long")
    if CONTROL_RE.search(value):
        raise ValueError(f"{field_name} contains control characters")
    return value.strip()


def validate_query(value: Any) -> str:
    return validate_text(value, field_name="query", max_length=MAX_QUERY_LENGTH)


def validate_app_name(value: Any) -> str:
    app_name = validate_text(value, field_name="application", max_length=40)
    if app_name.lower() not in ALLOWED_APP_NAMES:
        raise ValueError("application not allowlisted")
    return app_name.lower()


def validate_folder_name(value: Any) -> str:
    folder_name = validate_text(value, field_name="folder", max_length=40)
    if folder_name.lower() not in ALLOWED_FOLDER_NAMES:
        raise ValueError("folder not allowlisted")
    return folder_name.lower()


def validate_path_text(value: Any, *, field_name: str = "path") -> str:
    return validate_text(value, field_name=field_name, max_length=MAX_PATH_LENGTH)


def validate_string_list(value: Any, *, field_name: str, max_items: int = MAX_LIST_ITEMS, item_max_length: int = MAX_TEXT_LENGTH) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if len(value) > max_items:
        raise ValueError(f"{field_name} too long")
    return [validate_text(item, field_name=field_name, max_length=item_max_length) for item in value]


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("arguments must be a mapping")
    if not isinstance(schema, dict):
        raise ValueError("invalid schema")
    for name, spec in schema.items():
        required = spec.get("required", True)
        if name not in args:
            if required:
                raise ValueError(f"missing argument: {name}")
            continue
        if args[name] is None and not required:
            continue
        if spec.get("type") == "text":
            args[name] = validate_text(args[name], field_name=name, max_length=spec.get("max_length", MAX_TEXT_LENGTH))
        elif spec.get("type") == "query":
            args[name] = validate_query(args[name])
        elif spec.get("type") == "app":
            args[name] = validate_app_name(args[name])
        elif spec.get("type") == "folder":
            args[name] = validate_folder_name(args[name])
        elif spec.get("type") == "path":
            args[name] = validate_path_text(args[name], field_name=name)
        elif spec.get("type") == "bool":
            if not isinstance(args[name], bool):
                raise ValueError(f"{name} must be boolean")
        elif spec.get("type") == "integer":
            if not isinstance(args[name], int) or isinstance(args[name], bool):
                raise ValueError(f"{name} must be an integer")
        elif spec.get("type") == "string_list":
            args[name] = validate_string_list(
                args[name],
                field_name=name,
                max_items=spec.get("max_items", MAX_LIST_ITEMS),
                item_max_length=spec.get("item_max_length", MAX_TEXT_LENGTH),
            )
        elif spec.get("type") == "number":
            if not isinstance(args[name], (int, float)):
                raise ValueError(f"{name} must be numeric")
        else:
            raise ValueError(f"unsupported schema type for {name}")
    for extra in args:
        if extra not in schema:
            raise ValueError("unexpected argument")
    return args
