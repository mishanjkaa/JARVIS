from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfidenceCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Intent(str, Enum):
    CALCULATION = "calculation"
    WEB_SEARCH = "web_search"
    OPEN_FOLDER = "open_folder"
    OPEN_APPLICATION = "open_application"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    NOTES_WRITE = "notes_write"
    NOTES_READ = "notes_read"
    TASKS_WRITE = "tasks_write"
    TASKS_READ = "tasks_read"
    POWER_REQUEST = "power_request"
    SYSTEM_INFORMATION = "system_information"
    FILESYSTEM = "filesystem"
    MULTI_STEP = "multi_step"
    UNSUPPORTED = "unsupported"


class IntentCategory(str, Enum):
    CALCULATION = "calculation"
    WEB_SEARCH = "web_search"
    OPEN_FOLDER = "open_folder"
    OPEN_APPLICATION = "open_application"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    NOTES_WRITE = "notes_write"
    NOTES_READ = "notes_read"
    TASKS_WRITE = "tasks_write"
    TASKS_READ = "tasks_read"
    POWER_REQUEST = "power_request"
    SYSTEM_INFORMATION = "system_information"
    FILESYSTEM = "filesystem"
    MULTI_STEP = "multi_step"
    UNSUPPORTED = "unsupported"


@dataclass
class IntentClassification:
    intent: Intent
    confidence_category: ConfidenceCategory
    tools_required: bool
    multi_step_required: bool
    original_text: str


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence_category: ConfidenceCategory
    tools_required: bool
    multi_step_required: bool
    original_text: str
    normalized_text: str
