from __future__ import annotations

import re


def sanitize_summary(entries: list[str], limit: int = 8) -> list[str]:
    safe: list[str] = []
    for entry in entries[-limit:]:
        if not isinstance(entry, str):
            continue
        value = re.sub(r"[\r\n\x00-\x1f\x7f]+", " ", entry).strip()
        if value:
            safe.append(value[:160])
    return safe


def sanitize_for_prompt(text: str, max_length: int = 240) -> str:
    if not isinstance(text, str):
        return ""
    value = re.sub(r"[\r\n\x00-\x1f\x7f]+", " ", text).strip()
    return " ".join(value.split())[:max_length]
