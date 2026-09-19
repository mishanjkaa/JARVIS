"""Owner-requested boundary widening (2026-09-19 discussion): JARVIS's filesystem and
terminal tools were previously confined to just their own project folder no matter what
(get_trusted_roots() had no config knob at all). The owner asked for real access to her
whole system drive (Desktop, Documents, Downloads, ...), which meant this project's
Windows/System32/Program Files/other-users'-profile protections -- previously only ever
checked against a RAW, literal path argument -- needed to actually hold up once trusted
roots got that broad. See app/brain/filesystem/path_policy.py's
is_forbidden_system_location() docstring for the exact loophole this closes: a relative
argument resolving into a system directory only once joined to a broad trusted root,
which the old check never saw.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.path_policy import (
    _canonicalize,
    default_project_root,
    get_trusted_roots,
    is_forbidden_system_location,
    resolve_path,
)
from app.brain.filesystem.state import reset_filesystem_state, set_trusted_roots


class IsForbiddenSystemLocationTests(unittest.TestCase):
    def test_blocks_windows_directory_anywhere_in_the_path(self) -> None:
        self.assertTrue(is_forbidden_system_location(Path("/", "anywhere", "Windows", "System32", "evil.py")))

    def test_blocks_the_full_path_not_just_the_final_component(self) -> None:
        # The historical bug this replaces only checked the leaf name ("evil.py"), which
        # a file plainly living inside System32 sails straight through.
        self.assertTrue(is_forbidden_system_location(Path("/", "Windows", "System32", "evil.py")))

    def test_blocks_program_files(self) -> None:
        self.assertTrue(is_forbidden_system_location(Path("/", "Program Files", "Something", "app.exe")))
        self.assertTrue(is_forbidden_system_location(Path("/", "Program Files (x86)", "Something", "app.exe")))

    def test_blocks_another_accounts_profile(self) -> None:
        with patch.object(Path, "home", return_value=Path("/", "Users", "owner")):
            self.assertTrue(is_forbidden_system_location(Path("/", "Users", "someone_else", "secrets.txt")))

    def test_allows_the_owners_own_home_directory_even_though_it_contains_users(self) -> None:
        # The assistant owner's own default trusted root legitimately lives under
        # Users\<name> -- blocking that segment outright (the historical behavior) would
        # break everyday operation, not just protect anything.
        with patch.object(Path, "home", return_value=Path("/", "Users", "owner")):
            self.assertFalse(is_forbidden_system_location(Path("/", "Users", "owner", "Desktop", "notes.txt")))

    def test_allows_an_ordinary_path(self) -> None:
        self.assertFalse(is_forbidden_system_location(Path("/", "data", "projects", "jarvis", "notes.txt")))


class GetTrustedRootsFullDiskAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()
        reset_filesystem_state()

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_filesystem_state()

    def test_defaults_to_just_the_project_folder(self) -> None:
        self.assertEqual(get_trusted_roots(), [default_project_root()])

    def test_full_disk_access_widens_to_the_drive_root(self) -> None:
        set_runtime_config_value("filesystem_allow_full_disk_access", True)
        project_root = default_project_root()
        expected = Path(project_root.anchor) if project_root.anchor else project_root
        self.assertEqual(get_trusted_roots(), [_canonicalize(expected)])

    def test_explicit_trusted_roots_override_the_config_flag(self) -> None:
        # set_trusted_roots() is the test-only escape hatch (and, in principle, a future
        # more targeted config); an explicit override always wins over the coarse
        # all-or-nothing full-disk-access flag.
        set_runtime_config_value("filesystem_allow_full_disk_access", True)
        custom_root = Path.cwd() / ".tmp-tests" / "explicit-root"
        set_trusted_roots([custom_root])
        self.assertEqual(get_trusted_roots(), [_canonicalize(custom_root)])


class ResolvePathClosesTheRelativePathLoopholeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()
        reset_filesystem_state()

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_filesystem_state()

    def test_relative_path_into_windows_is_rejected_even_with_full_disk_access(self) -> None:
        set_runtime_config_value("filesystem_allow_full_disk_access", True)
        with self.assertRaises(FilesystemPathError):
            resolve_path("Windows/System32/evil.py")

    def test_ordinary_relative_path_still_resolves_normally_with_full_disk_access(self) -> None:
        set_runtime_config_value("filesystem_allow_full_disk_access", True)
        result = resolve_path("some_file.txt")
        self.assertEqual(result.relative_path, "some_file.txt")


if __name__ == "__main__":
    unittest.main()
