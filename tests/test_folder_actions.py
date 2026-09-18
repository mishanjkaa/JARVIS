import unittest
from pathlib import Path
from unittest.mock import patch

from app.brain.computer.folder_actions import _get_known_folder_status, open_known_folder


class FolderActionsTests(unittest.TestCase):
    @patch("app.brain.computer.folder_actions.os.startfile")
    @patch("app.brain.computer.folder_actions.os.name", "nt")
    @patch("app.brain.computer.folder_actions._resolve_known_folder_path")
    @patch("app.brain.computer.folder_actions.Path.exists")
    @patch("app.brain.computer.folder_actions.Path.is_dir")
    @patch("app.brain.computer.folder_actions.Path.home")
    def test_open_known_folder_success(self, home_mock, is_dir_mock, exists_mock, resolve_mock, startfile_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")
        exists_mock.return_value = True
        is_dir_mock.return_value = True
        resolve_mock.return_value = Path("C:/Users/Test/Desktop")

        result = open_known_folder("desktop")

        self.assertEqual(result, "Opened Desktop.")
        startfile_mock.assert_called_once()

    @patch("app.brain.computer.folder_actions.os.startfile")
    @patch("app.brain.computer.folder_actions.os.name", "nt")
    @patch("app.brain.computer.folder_actions._resolve_known_folder_path")
    @patch("app.brain.computer.folder_actions.Path.exists")
    @patch("app.brain.computer.folder_actions.Path.is_dir")
    @patch("app.brain.computer.folder_actions.Path.home")
    def test_open_known_folder_missing_folder(self, home_mock, is_dir_mock, exists_mock, resolve_mock, startfile_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")
        exists_mock.return_value = False
        is_dir_mock.return_value = False
        resolve_mock.return_value = None

        result = open_known_folder("downloads")

        self.assertEqual(result, "The Downloads folder is not available.")
        startfile_mock.assert_not_called()

    @patch("app.brain.computer.folder_actions.os.startfile")
    @patch("app.brain.computer.folder_actions.os.name", "nt")
    @patch("app.brain.computer.folder_actions.Path.home")
    def test_open_known_folder_rejects_unknown_name(self, home_mock, startfile_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")

        result = open_known_folder("secret")

        self.assertEqual(result, "I can only open known folders.")
        startfile_mock.assert_not_called()

    @patch("app.brain.computer.folder_actions.os.startfile")
    @patch("app.brain.computer.folder_actions.os.name", "nt")
    @patch("app.brain.computer.folder_actions._resolve_known_folder_path")
    @patch("app.brain.computer.folder_actions.Path.exists")
    @patch("app.brain.computer.folder_actions.Path.is_dir")
    @patch("app.brain.computer.folder_actions.Path.home")
    def test_open_known_folder_uses_redirected_path(self, home_mock, is_dir_mock, exists_mock, resolve_mock, startfile_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")
        exists_mock.return_value = True
        is_dir_mock.return_value = True
        resolve_mock.return_value = Path("C:/Users/Test/OneDrive/Desktop")

        result = open_known_folder("desktop")

        self.assertEqual(result, "Opened Desktop.")
        startfile_mock.assert_called_once()

    @patch("app.brain.computer.folder_actions.os.startfile")
    @patch("app.brain.computer.folder_actions.os.name", "nt")
    @patch("app.brain.computer.folder_actions._resolve_known_folder_path")
    @patch("app.brain.computer.folder_actions.Path.exists")
    @patch("app.brain.computer.folder_actions.Path.is_dir")
    @patch("app.brain.computer.folder_actions.Path.home")
    def test_open_known_folder_falls_back_to_onedrive_desktop(self, home_mock, is_dir_mock, exists_mock, resolve_mock, startfile_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")
        resolve_mock.return_value = None
        exists_mock.return_value = True
        is_dir_mock.return_value = True

        result = open_known_folder("desktop")

        self.assertEqual(result, "Opened Desktop.")
        startfile_mock.assert_called_once()

    @patch("app.brain.computer.folder_actions.os.name", "nt")
    def test_known_folder_status_uses_safe_categories(self) -> None:
        self.assertIn(_get_known_folder_status("desktop"), {"api_failed", "api_available", "unavailable"})


if __name__ == "__main__":
    unittest.main()
