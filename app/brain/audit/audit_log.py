from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

_CONTROL_RE = re.compile(r"[\r\n\x00-\x1f\x7f]+")
_AUDIT_ENTRIES: list["AuditEntry"] = []
_MAX_AUDIT_ENTRIES = 200


@dataclass
class AuditEntry:
    event_type: str
    timestamp: str
    task_id: int | None = None
    step_index: int | None = None
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "message": self.message,
        }


def _sanitize_message(message: str, max_length: int = 160) -> str:
    if not isinstance(message, str):
        return ""
    cleaned = _CONTROL_RE.sub(" ", message).strip()
    return cleaned[:max_length]


def record_audit_event(event_type: str, *, task_id: int | None = None, step_index: int | None = None, message: str = "") -> None:
    if not isinstance(event_type, str) or not event_type.strip():
        return
    entry = AuditEntry(
        event_type=event_type.strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_id=task_id if isinstance(task_id, int) else None,
        step_index=step_index if isinstance(step_index, int) else None,
        message=_sanitize_message(message),
    )
    _AUDIT_ENTRIES.append(entry)
    del _AUDIT_ENTRIES[:-_MAX_AUDIT_ENTRIES]


def get_audit_entries(limit: int | None = None) -> list[AuditEntry]:
    if limit is None:
        return list(_AUDIT_ENTRIES)
    return list(_AUDIT_ENTRIES[-limit:])


def reset_audit_log() -> None:
    _AUDIT_ENTRIES.clear()
