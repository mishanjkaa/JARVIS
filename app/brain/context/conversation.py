from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationEntry:
    summary: str


@dataclass
class ConversationState:
    entries: list[ConversationEntry] = field(default_factory=list)

    def add_entry(self, summary: str, maximum: int = 20) -> None:
        if not isinstance(summary, str) or not summary.strip():
            return
        self.entries.append(ConversationEntry(summary=summary.strip()[:160]))
        if len(self.entries) > maximum:
            self.entries = self.entries[-maximum:]

    def clear(self) -> None:
        self.entries.clear()


STATE = ConversationState()


def get_conversation_state() -> ConversationState:
    return STATE


def reset_conversation_state() -> None:
    STATE.clear()


def add_conversation_summary(summary: str, maximum: int = 20) -> None:
    get_conversation_state().add_entry(summary, maximum=maximum)


def get_conversation_summaries(limit: int | None = None) -> list[str]:
    entries = [entry.summary for entry in get_conversation_state().entries]
    if limit is None:
        return entries
    return entries[-limit:]


def add_summary(summary: str, maximum: int = 20) -> None:
    add_conversation_summary(summary, maximum=maximum)


def get_summaries(limit: int = 8) -> list[str]:
    return get_conversation_summaries(limit=limit)


def clear_conversation() -> None:
    STATE.clear()


def reset_conversation() -> None:
    clear_conversation()
