from __future__ import annotations

import re

from app.brain.filesystem.state import get_filesystem_state
from app.brain.intent.models import ConfidenceCategory, Intent, IntentClassification


def classify_obvious(text: str) -> IntentClassification | None:
    value = text.strip().lower()
    if not value or len(value) > 400 or re.search(r"[\x00-\x1f\x7f]", value):
        return None
    multi = bool(re.search(r"\b(and|then)\b", value))
    if re.search(r"\b(calculate|work out|what is)\b", value) and re.search(r"\d", value):
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.CALCULATION, ConfidenceCategory.HIGH, True, multi, text)
    if re.search(r"\b(search|find information|look up)\b", value):
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.WEB_SEARCH, ConfidenceCategory.HIGH, True, multi, text)
    if re.search(r"\b(open|launch)\b", value) and re.search(r"\b(downloads|documents|desktop|pictures|music|videos)\b", value):
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.OPEN_FOLDER, ConfidenceCategory.HIGH, True, multi, text)
    if re.search(r"\b(open|launch)\b", value) and re.search(r"\b(calculator|notepad|browser)\b", value):
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.OPEN_APPLICATION, ConfidenceCategory.HIGH, True, multi, text)
    if _looks_like_terminal_request(value):
        return IntentClassification(Intent.MULTI_STEP, ConfidenceCategory.HIGH, True, multi, text)
    if value.startswith(("write ", "append ")) and get_filesystem_state().last_touched_file is not None:
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.FILESYSTEM, ConfidenceCategory.HIGH, True, multi, text)
    if re.match(r"^(read|show)\s+", value) and any(token in value for token in ("..", "/", "\\")):
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.FILESYSTEM, ConfidenceCategory.HIGH, True, multi, text)
    if re.search(r"\b(create|make|write|append|read|rename|copy|move|delete|list|show)\b", value) and re.search(r"\b(file|folder|directory|txt|trash)\b", value):
        return IntentClassification(Intent.MULTI_STEP if multi else Intent.FILESYSTEM, ConfidenceCategory.HIGH, True, multi, text)
    if value.startswith(("remember ", "make a note", "add a task", "forget ")):
        intent = Intent.MEMORY_WRITE if value.startswith("remember ") else Intent.NOTES_WRITE if value.startswith("make a note") else Intent.TASKS_WRITE
        return IntentClassification(intent, ConfidenceCategory.HIGH, True, multi, text)
    if re.search(r"\b(show|list)\b.*\b(tasks|notes)\b", value):
        return IntentClassification(Intent.TASKS_READ if "task" in value else Intent.NOTES_READ, ConfidenceCategory.HIGH, True, multi, text)
    if re.search(r"\b(what did i calculate|what did you remember|recall)\b", value):
        return IntentClassification(Intent.MEMORY_READ, ConfidenceCategory.HIGH, True, False, text)
    if value.startswith(("lock ", "restart ", "shutdown ")):
        return IntentClassification(Intent.POWER_REQUEST, ConfidenceCategory.HIGH, True, False, text)
    return None


def _looks_like_terminal_request(value: str) -> bool:
    if re.search(r"\b(install|pip install|npm install|pnpm install|yarn add|apt install|brew install|terminal|python\b|git\b|powershell\b|pwsh\b|cmd\b|bash\b|sh\b|wsl\b|shutdown\b|format\b|unknown-program\b)\b", value):
        return True
    if value.startswith("run the unit tests") or value.startswith("run unit tests") or value.startswith("show git status"):
        return True
    tokens = value.split()
    if len(tokens) < 2:
        return False
    first = tokens[0]
    if first == "run" and len(tokens) >= 3:
        command = tokens[1]
        return command in {"python", "py", "git", "pip", "pip3", "powershell", "pwsh", "cmd", "bash", "sh", "wsl"} or command.endswith(".exe") or command == "unknown-program"
    return first.endswith(".exe")

import re
from typing import Callable

from app.brain.intent.models import ConfidenceCategory, IntentCategory


Rule = Callable[[str], tuple[IntentCategory, ConfidenceCategory, bool, bool] | None]


def _match(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def build_rule_map() -> list[tuple[tuple[str, ...], IntentCategory, ConfidenceCategory, bool, bool]]:
    return [
        (("calculate", "times", "plus", "minus", "divide", "multiply", "expression", "math", "work out"), IntentCategory.CALCULATION, ConfidenceCategory.HIGH, True, False),
        (("remember", "memory", "project uses python", "project version"), IntentCategory.MEMORY_WRITE, ConfidenceCategory.HIGH, True, False),
        (("recall", "what project", "what did i ask you to remember", "remembered"), IntentCategory.MEMORY_READ, ConfidenceCategory.HIGH, True, False),
        (("create file", "write file", "append file", "read file", "rename file", "copy file", "move file", "delete file", "create folder", "create directory", "trash"), IntentCategory.FILESYSTEM, ConfidenceCategory.HIGH, True, False),
        (("note", "make a note", "write a note"), IntentCategory.NOTES_WRITE, ConfidenceCategory.HIGH, True, False),
        (("task", "todo", "to do"), IntentCategory.TASKS_WRITE, ConfidenceCategory.HIGH, True, False),
        (("show my tasks", "list tasks", "my tasks", "tasks"), IntentCategory.TASKS_READ, ConfidenceCategory.HIGH, True, False),
        (("search", "find information", "information about"), IntentCategory.WEB_SEARCH, ConfidenceCategory.HIGH, True, True),
        (("open downloads", "downloads", "open documents", "documents", "open pictures", "pictures", "open music", "music", "open videos", "videos"), IntentCategory.OPEN_FOLDER, ConfidenceCategory.HIGH, True, False),
        (("open notepad", "open calculator", "open browser", "open github", "open youtube"), IntentCategory.OPEN_APPLICATION, ConfidenceCategory.HIGH, True, False),
        (("system info", "disk space", "computer name", "what time", "what date"), IntentCategory.SYSTEM_INFORMATION, ConfidenceCategory.HIGH, True, False),
    ]


def classify_with_rules(text: str) -> tuple[IntentCategory, ConfidenceCategory, bool, bool] | None:
    normalized = " ".join(text.lower().split())
    for patterns, intent, confidence, tools_required, multi_step_required in build_rule_map():
        if any(_match(pattern, normalized) for pattern in patterns):
            return intent, confidence, tools_required, multi_step_required
    return None
