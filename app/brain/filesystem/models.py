from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FilesystemResolvedPath:
    root: Path
    absolute_path: Path
    relative_path: str


@dataclass
class FilesystemOperationResult:
    success: bool
    message: str
    display_value: str = ""
    reference_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilesystemMetadata:
    relative_path: str
    exists: bool
    is_file: bool
    is_directory: bool
    size_bytes: int

    def to_display(self) -> str:
        return (
            f"path={self.relative_path} | exists={str(self.exists).lower()} | "
            f"file={str(self.is_file).lower()} | directory={str(self.is_directory).lower()} | "
            f"size_bytes={self.size_bytes}"
        )
