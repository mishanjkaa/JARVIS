from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_config(path: Path | str, data: dict[str, Any], *, backup_path: Path | str | None = None) -> bool:
    config_path = Path(path)
    backup = Path(backup_path) if backup_path is not None else config_path.with_name("config.backup.json")
    temp_name = ""
    try:
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_handle, temp_name = tempfile.mkstemp(prefix="jarvis-config-", suffix=".json", dir=str(config_path.parent))
        with os.fdopen(temp_handle, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if config_path.exists():
            backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(temp_name, config_path)
        return True
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        return False
