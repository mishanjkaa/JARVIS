import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.agent.state import reset_agent_runtime_state
from app.brain.audit.audit_log import get_audit_entries, reset_audit_log
from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.filesystem.controller import get_filesystem_controller
from app.brain.filesystem.errors import FilesystemDisabledError, FilesystemError, FilesystemPathError
from app.brain.filesystem.path_policy import resolve_path
from app.brain.filesystem.state import reset_filesystem_state, set_trusted_roots
from app.brain.planner.approval import cancel_pending_plan
from app.brain.planner.state import reset_planner_state
from app.brain.router import _CONVERSATION_RUNTIME, route_command


class _FakeConversationProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, prompt: str):
        self.prompts.append(prompt)
        raise AssertionError("filesystem requests should not call conversation mode")


class FilesystemRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.root = (self.temp_root / f"case-{uuid4().hex}").absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        reset_runtime_config()
        reset_filesystem_state()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_planner_state()
        set_trusted_roots([self.root])

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_filesystem_state()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_planner_state()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_allowed_root(self) -> None:
        result = get_filesystem_controller().create_text_file("notes.txt")
        self.assertEqual(result.display_value, "notes.txt")
        self.assertTrue((self.root / "notes.txt").exists())

    def test_rejected_root(self) -> None:
        outside = self.root.parent / "outside.txt"
        with self.assertRaises(FilesystemPathError):
            resolve_path(str(outside), prefer_directory=False)

    def test_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(FilesystemPathError):
            get_filesystem_controller().create_text_file("../escape.txt")

    def test_symlink_escape_is_rejected(self) -> None:
        outside_dir = (self.temp_root / f"outside-{uuid4().hex}").absolute()
        outside_dir.mkdir(parents=True, exist_ok=True)
        target_file = outside_dir / "escape.txt"
        link_path = self.root / "link"
        try:
            try:
                link_path.symlink_to(outside_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not available in this environment")
            with self.assertRaises(FilesystemPathError):
                get_filesystem_controller().create_text_file("link/escape.txt")
        finally:
            if link_path.exists() or link_path.is_symlink():
                try:
                    link_path.unlink()
                except OSError:
                    pass
            if target_file.exists():
                target_file.unlink()
            try:
                outside_dir.rmdir()
            except OSError:
                pass

    def test_soft_delete_moves_file_to_trash(self) -> None:
        controller = get_filesystem_controller()
        controller.create_text_file("notes.txt")
        controller.write_text_file("Hello", "notes.txt")
        result = controller.delete_path("notes.txt")
        self.assertIn("Trash", result.message)
        self.assertFalse((self.root / "notes.txt").exists())
        self.assertTrue(any(path.name.startswith("notes_") for path in (self.root / "Trash").rglob("*") if path.is_file()))

    def test_copy_move_rename_and_append(self) -> None:
        controller = get_filesystem_controller()
        controller.create_text_file("notes.txt")
        controller.write_text_file("Hello", "notes.txt")
        controller.append_text_file("World", "notes.txt")
        self.assertEqual(controller.read_text_file("notes.txt").message, "HelloWorld")
        self.assertIn("report.txt", controller.rename_path("notes.txt", "report.txt").message)
        self.assertTrue((self.root / "report.txt").exists())
        self.assertIn("report_copy.txt", controller.copy_path("report.txt").message)
        self.assertTrue((self.root / "report_copy.txt").exists())
        self.assertIn("archive.txt", controller.move_path("report_copy.txt", "archive.txt").message)
        self.assertTrue((self.root / "archive.txt").exists())

    def test_large_file_rejection(self) -> None:
        set_runtime_config_value("filesystem_max_write_size", 4)
        controller = get_filesystem_controller()
        controller.create_text_file("notes.txt")
        with self.assertRaises(FilesystemError):
            controller.write_text_file("HelloWorld", "notes.txt")

    def test_disabled_filesystem(self) -> None:
        set_runtime_config_value("filesystem_enabled", False)
        with self.assertRaises(FilesystemDisabledError):
            get_filesystem_controller().create_text_file("notes.txt")

    def test_result_passing_and_agent_integration(self) -> None:
        fake_provider = _FakeConversationProvider()
        with patch.object(_CONVERSATION_RUNTIME, "provider", fake_provider):
            self.assertEqual(route_command("Create folder TestProject"), "Created directory TestProject.")
            self.assertEqual(route_command("Create file notes.txt"), "Created file TestProject/notes.txt.")
            self.assertEqual(route_command("Write Hello"), "Wrote text to TestProject/notes.txt.")
            self.assertEqual(route_command("Append World"), "Appended text to TestProject/notes.txt.")
            self.assertEqual(route_command("Read notes.txt"), "HelloWorld")
        self.assertEqual(fake_provider.prompts, [])

    def test_planner_integration(self) -> None:
        self.assertEqual(route_command("Create folder Reports"), "Created directory Reports.")
        self.assertTrue((self.root / "Reports").exists())

    def test_audit_events_are_generated(self) -> None:
        controller = get_filesystem_controller()
        controller.create_text_file("notes.txt")
        controller.write_text_file("Hello", "notes.txt")
        controller.append_text_file("World", "notes.txt")
        controller.copy_path("notes.txt")
        controller.rename_path("notes.txt", "report.txt")
        controller.move_path("notes_copy.txt", "archive.txt")
        controller.delete_path("report.txt")
        event_types = [entry.event_type for entry in get_audit_entries()]
        self.assertIn("filesystem_create", event_types)
        self.assertIn("filesystem_write", event_types)
        self.assertIn("filesystem_append", event_types)
        self.assertIn("filesystem_copy", event_types)
        self.assertIn("filesystem_rename", event_types)
        self.assertIn("filesystem_move", event_types)
        self.assertIn("filesystem_delete", event_types)

    def test_manual_acceptance_flow(self) -> None:
        self.assertEqual(route_command("Create folder TestProject"), "Created directory TestProject.")
        self.assertEqual(route_command("Create file notes.txt"), "Created file TestProject/notes.txt.")
        self.assertEqual(route_command("Write Hello"), "Wrote text to TestProject/notes.txt.")
        self.assertEqual(route_command("Append World"), "Appended text to TestProject/notes.txt.")
        self.assertEqual(route_command("Read notes.txt"), "HelloWorld")
        self.assertEqual(route_command("Rename notes.txt to report.txt"), "Renamed TestProject/notes.txt to TestProject/report.txt.")
        self.assertEqual(route_command("Copy report.txt"), "Copied TestProject/report.txt to TestProject/report_copy.txt.")
        self.assertIn("Pending plan", route_command("Delete report.txt"))
        self.assertIn("Moved TestProject/report.txt to Trash.", route_command("approve plan"))
        trash_listing = route_command("List directory Trash")
        self.assertIn("TestProject", trash_listing)

    def test_explicit_relative_path_is_not_duplicated(self) -> None:
        controller = get_filesystem_controller()
        self.assertEqual(controller.create_directory("TestProject").message, "Created directory TestProject.")
        self.assertEqual(
            controller.create_text_file("TestProject/notes.txt").message,
            "Created file TestProject/notes.txt.",
        )
        controller.write_text_file("Hello")
        controller.append_text_file("World")
        self.assertEqual(controller.read_text_file("TestProject/notes.txt").message, "HelloWorld")
        self.assertEqual(
            controller.rename_path("TestProject/notes.txt", "TestProject/report.txt").message,
            "Renamed TestProject/notes.txt to TestProject/report.txt.",
        )
        self.assertTrue((self.root / "TestProject" / "report.txt").exists())
        self.assertFalse((self.root / "TestProject" / "TestProject" / "notes.txt").exists())

    def test_manual_acceptance_flow_with_explicit_paths(self) -> None:
        self.assertEqual(route_command("Create folder TestProject"), "Created directory TestProject.")
        self.assertEqual(route_command("Create file TestProject/notes.txt"), "Created file TestProject/notes.txt.")
        self.assertEqual(route_command("Write Hello"), "Wrote text to TestProject/notes.txt.")
        self.assertEqual(route_command("Append World"), "Appended text to TestProject/notes.txt.")
        self.assertEqual(route_command("Read TestProject/notes.txt"), "HelloWorld")
        self.assertEqual(route_command("Rename TestProject/notes.txt to TestProject/report.txt"), "Renamed TestProject/notes.txt to TestProject/report.txt.")

    def test_cancelled_plan_never_writes(self) -> None:
        self.assertEqual(route_command("Create file notes.txt"), "Created file notes.txt.")
        self.assertIn("Pending plan", route_command("Delete notes.txt"))
        self.assertEqual(cancel_pending_plan(), "Pending plan cancelled.")
        self.assertTrue((self.root / "notes.txt").exists())
