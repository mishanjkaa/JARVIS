from __future__ import annotations

MAX_MEMORY_READS = 3


def validate_memory_key(key: object) -> str:
    if not isinstance(key, str) or not key.strip() or len(key.strip()) > 80 or any(char in key for char in "\r\n\x00"):
        raise ValueError("memory key rejected")
    return key.strip().lower()
from __future__ import annotations


def can_retrieve_memory(key: str) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower()
    if not normalized:
        return False
    if len(normalized) > 80:
        return False
    if any(ch.isspace() for ch in normalized):
        return False
    return True
