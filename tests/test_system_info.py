import unittest
from pathlib import Path
from unittest.mock import patch

from app.brain.computer.system_info import (
    format_size,
    get_computer_name,
    get_disk_space,
    get_system_info,
)


class SystemInfoTests(unittest.TestCase):
    def test_format_size_handles_common_units(self) -> None:
        self.assertEqual(format_size(512), "512.00 bytes")
        self.assertEqual(format_size(2048), "2.00 KB")
        self.assertEqual(format_size(2 * 1024 * 1024), "2.00 MB")
        self.assertEqual(format_size(3 * 1024 * 1024 * 1024), "3.00 GB")
        self.assertEqual(format_size(5 * 1024 * 1024 * 1024 * 1024), "5.00 TB")

    @patch("app.brain.computer.system_info.platform.system")
    @patch("app.brain.computer.system_info.platform.release")
    @patch("app.brain.computer.system_info.platform.machine")
    @patch("app.brain.computer.system_info.platform.python_version")
    def test_system_info_summary_is_concise(self, python_version_mock, machine_mock, release_mock, system_mock) -> None:
        system_mock.return_value = "Windows"
        release_mock.return_value = "11"
        machine_mock.return_value = "AMD64"
        python_version_mock.return_value = "3.11.4"

        result = get_system_info()

        self.assertIn("OS: Windows", result)
        self.assertIn("Release: 11", result)
        self.assertIn("Architecture: AMD64", result)
        self.assertIn("Python: 3.11.4", result)

    @patch("app.brain.computer.system_info.shutil.disk_usage")
    @patch("app.brain.computer.system_info.Path.home")
    def test_disk_space_uses_readable_units(self, home_mock, disk_usage_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")
        disk_usage_mock.return_value = (3 * 1024 * 1024 * 1024, 1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)

        result = get_disk_space()

        self.assertIn("Total: 3.00 GB", result)
        self.assertIn("Used: 1.00 GB", result)
        self.assertIn("Free: 2.00 GB", result)

    @patch("app.brain.computer.system_info.socket.gethostname")
    def test_computer_name_returns_hostname(self, gethostname_mock) -> None:
        gethostname_mock.return_value = "DESKTOP-123"
        self.assertEqual(get_computer_name(), "Computer name: DESKTOP-123")

    @patch("app.brain.computer.system_info.shutil.disk_usage")
    @patch("app.brain.computer.system_info.Path.home")
    def test_disk_space_handles_failures(self, home_mock, disk_usage_mock) -> None:
        home_mock.return_value = Path("C:/Users/Test")
        disk_usage_mock.side_effect = OSError("boom")

        result = get_disk_space()

        self.assertEqual(result, "I could not read the disk space right now.")


if __name__ == "__main__":
    unittest.main()
