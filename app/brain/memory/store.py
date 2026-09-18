import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_memory_file_path(memory_file: Optional[Path] = None) -> Path:
    if memory_file is not None:
        return memory_file

    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "memory.json"


def _load_memory_data(memory_file: Path) -> dict[str, dict[str, str]]:
    if not memory_file.exists():
        return {}

    try:
        content = memory_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}

    if not content.strip():
        return {}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def _save_memory_data(memory_file: Path, data: dict[str, dict[str, str]]) -> None:
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_memory(key: str, value: str, memory_file: Optional[Path] = None) -> str:
    normalized_key = key.strip().lower()
    normalized_value = value.strip()

    if not normalized_key:
        return "Please provide a non-empty memory key."

    if not normalized_value:
        return "Please provide a non-empty memory value."

    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if normalized_key in data:
        created_at = data[normalized_key].get("created_at", now)
        data[normalized_key] = {
            "value": normalized_value,
            "created_at": created_at,
            "updated_at": now,
            "source": "user",
        }
        _save_memory_data(path, data)
        return f"Updated memory for '{normalized_key}'."

    data[normalized_key] = {
        "value": normalized_value,
        "created_at": now,
        "updated_at": now,
        "source": "user",
    }
    _save_memory_data(path, data)
    return f"Saved memory for '{normalized_key}'."


def recall_memory(key: str, memory_file: Optional[Path] = None) -> str:
    normalized_key = key.strip().lower()

    if not normalized_key:
        return "Please provide a non-empty memory key."

    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    if normalized_key not in data:
        return f"No memory found for '{normalized_key}'."

    return data[normalized_key]["value"]


def forget_memory(key: str, memory_file: Optional[Path] = None) -> str:
    normalized_key = key.strip().lower()

    if not normalized_key:
        return "Please provide a non-empty memory key."

    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    if normalized_key not in data:
        return f"No memory found for '{normalized_key}'."

    del data[normalized_key]
    _save_memory_data(path, data)
    return f"Forgot memory for '{normalized_key}'."


def list_memories(memory_file: Optional[Path] = None) -> str:
    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    if not data:
        return "No memories stored."

    lines = [f"{key}: {entry['value']}" for key, entry in sorted(data.items())]
    return "\n".join(lines)
