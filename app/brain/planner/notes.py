import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_notes_path(notes_path: Optional[Path] = None) -> Path:
    if notes_path is not None:
        return notes_path
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "notes.json"


def _load_notes(notes_path: Path) -> list[dict[str, object]]:
    if not notes_path.exists():
        return []
    try:
        content = notes_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    if not content.strip():
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _save_notes(notes_path: Path, data: list[dict[str, object]]) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_note(text: str, notes_path: Optional[Path] = None) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    path = _get_notes_path(notes_path)
    notes = _load_notes(path)
    note_id = len(notes) + 1
    note = {"id": note_id, "text": cleaned, "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
    notes.append(note)
    _save_notes(path, notes)
    logger.info("Note added")
    return note_id


def list_notes(notes_path: Optional[Path] = None) -> str:
    path = _get_notes_path(notes_path)
    notes = _load_notes(path)
    if not notes:
        return "No notes stored."
    lines = [f"{item['id']}. {item['text']}" for item in notes if isinstance(item, dict) and isinstance(item.get('id'), int) and isinstance(item.get('text'), str)]
    return "\n".join(lines)


def delete_note(note_id: int, notes_path: Optional[Path] = None) -> str:
    path = _get_notes_path(notes_path)
    notes = _load_notes(path)
    remaining = [item for item in notes if not (isinstance(item, dict) and item.get("id") == note_id)]
    if len(remaining) == len(notes):
        return f"Note {note_id} not found."
    _save_notes(path, remaining)
    logger.info("Note deleted")
    return f"Deleted note {note_id}."
