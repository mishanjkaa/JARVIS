from __future__ import annotations

import json
import re
from typing import Any

MAX_RESPONSE_CHARS = 12000
_FULL_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", flags=re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str, max_chars: int = MAX_RESPONSE_CHARS) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("structured output is empty")
    if len(cleaned) > max_chars:
        raise ValueError("structured output is too large")

    fenced = _FULL_CODE_FENCE_PATTERN.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    candidate = _extract_single_object(cleaned)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("structured output JSON decode failed") from error
    if not isinstance(parsed, dict):
        raise ValueError("structured output root type mismatch")
    return parsed


def _extract_single_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("structured output root type mismatch")
    depth = 0
    in_string = False
    escape = False
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        raise ValueError("structured output was truncated")
    before = text[:start].strip()
    after = text[end:].strip()
    if before:
        raise ValueError("structured output contained text before JSON")
    if after:
        if "{" in after:
            raise ValueError("structured output contained multiple JSON values")
        raise ValueError("structured output contained text after JSON")
    return text[start:end]
