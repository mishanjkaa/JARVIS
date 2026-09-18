from __future__ import annotations

import json
from typing import Any


def normalize_vision_query(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace('"', " ").replace("'", " ")
    tokens = [token for token in normalized.split() if token not in {"the", "a", "an"}]
    return " ".join(tokens)


def looks_like_json_container_text(value: str) -> bool:
    raw = str(value or "").strip()
    return bool(raw) and raw[0] in "{["


def canonicalize_vision_find_query_text(value: Any, *, expected_query: str) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if not looks_like_json_container_text(raw):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"text"}:
        return None
    text = parsed.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    normalized_text = normalize_vision_query(text)
    normalized_expected = normalize_vision_query(expected_query)
    if normalized_expected and normalized_expected not in normalized_text:
        return None
    return text.strip()
