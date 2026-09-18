from __future__ import annotations

import io
import shutil
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from app.brain.agent.state import get_agent_runtime_state, reset_agent_runtime_state
from app.brain.audit.audit_log import reset_audit_log
from app.brain.configuration.runtime_config import get_effective_runtime_config, reset_runtime_config, set_runtime_config_value
from app.brain.router import route_command
from app.brain.tools.registry import ToolRegistry
from app.brain.vision.controller import get_vision_controller, reset_vision_controller
from app.brain.vision.desktop_capture_backend import RawCapture, WindowInfo
from app.brain.vision.errors import VisionCaptureUnsupportedError, VisionImageError, VisionPolicyError
from app.brain.vision.state import get_vision_state, reset_vision_state
from tests.test_vision_runtime import _FakeVisionProvider


def _fake_window(window_id: int = 111, title: str = "Notepad - Untitled") -> WindowInfo:
    return WindowInfo(window_id=window_id, title=title, left=0, top=0, right=800, bottom=600)


def _fake_raw_capture(width: int = 800, height: int = 600) -> RawCapture:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buffer, format="PNG")
    return RawCapture(image_bytes=buffer.getvalue(), width=width, height=height)


class DesktopCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.root = (self.temp_root / f"desktop-{uuid4().hex}").absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_vision_state()
        set_runtime_config_value("vision_model", "fake-vision")
        set_runtime_config_value("vision_enabled", True)
        set_runtime_config_value("vision_desktop_capture_enabled", True)
        self.fake_provider = _FakeVisionProvider()
        reset_vision_controller(provider=self.fake_provider)
        self.registry = ToolRegistry()

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_vision_state()
        reset_vision_controller()
        shutil.rmtree(self.root, ignore_errors=True)

    # --- Happy paths: capture then analyze --------------------------------

    @patch("app.brain.vision.desktop_capture_backend.capture_full_desktop")
    def test_capture_desktop_screen_then_describe(self, capture_mock) -> None:
        capture_mock.return_value = _fake_raw_capture()
        capture = get_vision_controller().capture_desktop_screen()
        self.assertEqual(capture.source_type, "desktop_screen")
        self.assertEqual(capture.window_title, "")
        evidence = get_vision_controller().describe_desktop_capture(capture_id=capture.capture_id)
        self.assertTrue(evidence.success)
        self.assertEqual(evidence.description, self.fake_provider.description)

    @patch("app.brain.vision.desktop_capture_backend.capture_window")
    @patch("app.brain.vision.desktop_capture_backend.find_window")
    def test_capture_desktop_window_then_extract_text(self, find_window_mock, capture_window_mock) -> None:
        find_window_mock.return_value = _fake_window()
        capture_window_mock.return_value = _fake_raw_capture()
        capture = get_vision_controller().capture_desktop_window(window_id=111, window_title="Notepad - Untitled")
        self.assertEqual(capture.source_type, "desktop_window")
        self.assertEqual(capture.window_title, "Notepad - Untitled")
        evidence = get_vision_controller().extract_text_from_desktop_capture(capture_id=capture.capture_id)
        self.assertTrue(evidence.success)
        self.assertEqual(evidence.extracted_text, self.fake_provider.extracted_text)

    @patch("app.brain.vision.desktop_capture_backend.capture_full_desktop")
    def test_find_visual_element_in_desktop_capture(self, capture_mock) -> None:
        capture_mock.return_value = _fake_raw_capture()
        capture = get_vision_controller().capture_desktop_screen()
        evidence = get_vision_controller().find_visual_element_in_desktop_capture(capture_id=capture.capture_id, query="red circle")
        self.assertTrue(evidence.success)
        self.assertEqual(len(evidence.visual_regions), 1)

    @patch("app.brain.vision.desktop_capture_backend.list_windows")
    def test_list_windows_reports_titles_and_ids(self, list_windows_mock) -> None:
        list_windows_mock.return_value = [_fake_window(111, "Notepad - Untitled"), _fake_window(222, "Calculator")]
        result = get_vision_controller().list_windows()
        self.assertTrue(result.success)
        titles = [window["title"] for window in result.windows]
        self.assertIn("Notepad - Untitled", titles)
        self.assertIn("Calculator", titles)

    # --- Window identity revalidation (HWND reuse) -------------------------

    @patch("app.brain.vision.desktop_capture_backend.find_window")
    def test_capture_window_rejects_when_title_changed_since_approval(self, find_window_mock) -> None:
        # The plan was approved against window_id 111 titled "Notepad - Untitled". By the
        # time it executes, Windows has recycled that HWND for an unrelated window. Existence
        # and visibility alone would pass; only the title check catches this.
        find_window_mock.return_value = WindowInfo(window_id=111, title="Banking App - Recycled Handle", left=0, top=0, right=800, bottom=600)
        with self.assertRaises(VisionPolicyError) as context:
            get_vision_controller().capture_desktop_window(window_id=111, window_title="Notepad - Untitled")
        self.assertIn("changed since this plan was approved", str(context.exception))

    @patch("app.brain.vision.desktop_capture_backend.find_window")
    def test_capture_window_rejects_when_window_closed(self, find_window_mock) -> None:
        find_window_mock.return_value = None
        with self.assertRaises(VisionImageError):
            get_vision_controller().capture_desktop_window(window_id=111, window_title="Notepad - Untitled")

    @patch("app.brain.vision.desktop_capture_backend.capture_window")
    @patch("app.brain.vision.desktop_capture_backend.find_window")
    def test_capture_window_succeeds_when_title_still_matches(self, find_window_mock, capture_window_mock) -> None:
        find_window_mock.return_value = _fake_window(111, "Notepad - Untitled")
        capture_window_mock.return_value = _fake_raw_capture()
        capture = get_vision_controller().capture_desktop_window(window_id=111, window_title="Notepad - Untitled")
        self.assertEqual(capture.window_title, "Notepad - Untitled")

    # --- TTL expiration -----------------------------------------------------

    @patch("app.brain.vision.desktop_capture_backend.capture_full_desktop")
    def test_expired_desktop_capture_is_rejected_safely(self, capture_mock) -> None:
        capture_mock.return_value = _fake_raw_capture()
        capture = get_vision_controller().capture_desktop_screen()
        state = get_vision_state()
        stored = state.desktop_captures[capture.capture_id]
        state.desktop_captures[capture.capture_id] = replace(
            stored,
            expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        )
        evidence = get_vision_controller().describe_desktop_capture(capture_id=capture.capture_id)
        self.assertFalse(evidence.success)
        self.assertEqual(evidence.error_category, "expired")

    # --- Task ownership scoping ----------------------------------------------

    @patch("app.brain.vision.desktop_capture_backend.capture_full_desktop")
    def test_desktop_capture_owned_by_a_different_task_is_rejected(self, capture_mock) -> None:
        capture_mock.return_value = _fake_raw_capture()
        capture = get_vision_controller().capture_desktop_screen()
        state = get_vision_state()
        stored = state.desktop_captures[capture.capture_id]
        state.desktop_captures[capture.capture_id] = replace(stored, owner_agent_task_id=999999)
        evidence = get_vision_controller().describe_desktop_capture(capture_id=capture.capture_id)
        self.assertFalse(evidence.success)
        self.assertEqual(evidence.error_category, "policy")

    # --- No leakage of bytes/paths -------------------------------------------

    @patch("app.brain.vision.desktop_capture_backend.capture_full_desktop")
    def test_capture_tool_result_never_exposes_bytes(self, capture_mock) -> None:
        capture_mock.return_value = _fake_raw_capture()
        handler = self.registry.get("desktop.capture_screen").handler
        result = handler({})
        self.assertTrue(result.success)
        serialized = str(result.to_dict())
        self.assertNotIn("iVBOR", serialized)
        self.assertNotIn(str(self.temp_root), serialized)

    @patch("app.brain.vision.desktop_capture_backend.capture_full_desktop")
    def test_desktop_captures_status_exposes_metadata_only(self, capture_mock) -> None:
        capture_mock.return_value = _fake_raw_capture()
        get_vision_controller().capture_desktop_screen()
        status = route_command("desktop captures")
        self.assertIn("Temporary desktop captures:", status)
        self.assertNotIn("iVBOR", status)
        cleared = route_command("desktop clear captures")
        self.assertIn("Cleared 1 temporary desktop capture", cleared)
        self.assertEqual(get_vision_state().desktop_captures, {})

    # --- Non-Windows platform: fail cleanly, never silently no-op -----------

    @patch("app.brain.vision.desktop_capture_backend.is_supported", return_value=False)
    def test_capture_screen_fails_cleanly_on_unsupported_platform(self, _is_supported_mock) -> None:
        with self.assertRaises(VisionCaptureUnsupportedError):
            get_vision_controller().capture_desktop_screen()

    @patch("app.brain.vision.desktop_capture_backend.is_supported", return_value=False)
    def test_capture_screen_tool_reports_unsupported_platform_as_failure(self, _is_supported_mock) -> None:
        handler = self.registry.get("desktop.capture_screen").handler
        result = handler({})
        self.assertFalse(result.success)
        self.assertIn("Windows", result.message)

    # --- Tool registration / risk shape --------------------------------------

    def test_desktop_tools_are_registered(self) -> None:
        names = self.registry.list_names()
        for name in [
            "desktop.list_windows",
            "desktop.capture_screen",
            "desktop.capture_window",
            "vision.describe_desktop_capture",
            "vision.extract_text_from_desktop_capture",
            "vision.find_visual_element_in_desktop_capture",
        ]:
            self.assertIn(name, names)

    def test_capture_tools_are_persistent_write_risk_level(self) -> None:
        self.assertEqual(self.registry.get("desktop.capture_screen").risk_level, "persistent_write")
        self.assertEqual(self.registry.get("desktop.capture_window").risk_level, "persistent_write")
        self.assertEqual(self.registry.get("desktop.list_windows").risk_level, "read_only")


if __name__ == "__main__":
    unittest.main()
