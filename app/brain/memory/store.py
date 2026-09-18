import json
from datetime import datetime
from pathlib import Path
from typing import Optional

SOURCE_USER = "user"
SOURCE_AI_PROPOSED = "ai_proposed"
VALID_SOURCES = {SOURCE_USER, SOURCE_AI_PROPOSED}

CATEGORY_FACT = "fact"
CATEGORY_PREFERENCE = "preference"
CATEGORY_LEARNED_PATTERN = "learned_pattern"
VALID_CATEGORIES = {CATEGORY_FACT, CATEGORY_PREFERENCE, CATEGORY_LEARNED_PATTERN}
DEFAULT_CATEGORY = CATEGORY_FACT

MAX_SUGGESTED_KEYS = 3


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


def _entry_category(entry: dict[str, str]) -> str:
    category = entry.get("category", DEFAULT_CATEGORY)
    return category if category in VALID_CATEGORIES else DEFAULT_CATEGORY


def _entry_source(entry: dict[str, str]) -> str:
    source = entry.get("source", SOURCE_USER)
    return source if source in VALID_SOURCES else SOURCE_USER


def save_memory(
    key: str,
    value: str,
    memory_file: Optional[Path] = None,
    *,
    category: str = DEFAULT_CATEGORY,
    source: str = SOURCE_USER,
    max_entries: Optional[int] = None,
    learned_capture_enabled: bool = True,
) -> str:
    normalized_key = key.strip().lower()
    normalized_value = value.strip()

    if not normalized_key:
        return "Please provide a non-empty memory key."

    if not normalized_value:
        return "Please provide a non-empty memory value."

    if category not in VALID_CATEGORIES:
        return "Please provide a valid memory category."

    if source not in VALID_SOURCES:
        return "Please provide a valid memory source."

    if category == CATEGORY_LEARNED_PATTERN and not learned_capture_enabled:
        return "Learned-pattern memory capture is disabled."

    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if normalized_key in data:
        created_at = data[normalized_key].get("created_at", now)
        data[normalized_key] = {
            "value": normalized_value,
            "created_at": created_at,
            "updated_at": now,
            "source": source,
            "category": category,
        }
        _save_memory_data(path, data)
        return f"Updated memory for '{normalized_key}'."

    if max_entries is not None and len(data) >= max_entries:
        return f"Memory limit reached ({max_entries} entries). Forget an existing memory before adding a new one."

    data[normalized_key] = {
        "value": normalized_value,
        "created_at": now,
        "updated_at": now,
        "source": source,
        "category": category,
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


def find_similar_keys(key: str, memory_file: Optional[Path] = None, *, limit: int = MAX_SUGGESTED_KEYS) -> list[str]:
    """Return up to `limit` stored key names that look related to `key`.

    This is explicit, non-semantic key-name matching (substring/prefix only) used to
    build a "did you mean" hint on the deterministic recall command. It never returns
    memory values, and it is not used by the AI-facing memory.recall tool.
    """
    normalized_key = key.strip().lower()
    if not normalized_key:
        return []

    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    candidates = [
        stored_key
        for stored_key in data
        if stored_key != normalized_key and (normalized_key in stored_key or stored_key in normalized_key)
    ]
    candidates.sort(key=lambda candidate: (not candidate.startswith(normalized_key), len(candidate), candidate))
    return candidates[:limit]


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


def list_memories(
    memory_file: Optional[Path] = None,
    *,
    category: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    path = get_memory_file_path(memory_file)
    data = _load_memory_data(path)

    items = sorted(data.items())
    if category is not None:
        items = [(key, entry) for key, entry in items if _entry_category(entry) == category]
    if source is not None:
        items = [(key, entry) for key, entry in items if _entry_source(entry) == source]

    if not items:
        return "No memories stored."

    lines = [
        f"{key}: {entry['value']} [{_entry_category(entry)}/{_entry_source(entry)}]"
        for key, entry in items
    ]
    return "\n".join(lines)
