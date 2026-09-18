import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.brain.planner.notes import (
    add_note,
    delete_note,
    list_notes,
)


class NotesTests(unittest.TestCase):
    def _temp_dir(self) -> Path:
        temp_root = Path.cwd() / ".tmp-tests"
        temp_root.mkdir(exist_ok=True)
        temp_dir = temp_root / f"notes-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def test_add_and_list_notes(self) -> None:
        notes_path = self._temp_dir() / "notes.json"
        note_id = add_note("hello world", notes_path)
        self.assertEqual(note_id, 1)
        self.assertEqual(list_notes(notes_path), "1. hello world")

    def test_delete_note(self) -> None:
        notes_path = self._temp_dir() / "notes.json"
        add_note("first", notes_path)
        add_note("second", notes_path)
        deleted = delete_note(1, notes_path)
        self.assertEqual(deleted, "Deleted note 1.")
        self.assertEqual(list_notes(notes_path), "2. second")

    def test_malformed_json_is_recovered(self) -> None:
        notes_path = self._temp_dir() / "notes.json"
        notes_path.write_text("{not valid json", encoding="utf-8")
        self.assertEqual(add_note("recovered", notes_path), 1)
        self.assertEqual(list_notes(notes_path), "1. recovered")
