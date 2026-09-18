from __future__ import annotations

from app.brain.memory.memory_policy import validate_memory_key
from app.brain.memory.store import recall_memory


def retrieve_one(key: str) -> str:
    return recall_memory(validate_memory_key(key))
from __future__ import annotations

from app.brain.memory.memory_policy import can_retrieve_memory
from app.brain.memory.store import recall_memory


def retrieve_memory_value(key: str) -> str:
    if not can_retrieve_memory(key):
        raise ValueError("memory key invalid")
    return recall_memory(key)
