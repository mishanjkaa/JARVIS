import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.configuration.runtime_config import get_effective_runtime_config, get_runtime_config, get_runtime_config_snapshot, replace_runtime_config, reset_runtime_config
from app.brain.configuration.config_writer import write_config
from app.brain.configuration.state import get_runtime_config_state
from app.brain.router import route_command
from config.config_loader import load_config, set_config_path_override


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()
        self.temp_dir = self._temp_dir()
        self.config_path = self.temp_dir / "config.json"
        self.config_path.write_text("{}", encoding="utf-8")
        set_config_path_override(self.config_path)

    def tearDown(self) -> None:
        set_config_path_override(None)
        reset_runtime_config()

    def _temp_dir(self) -> Path:
        temp_root = Path.cwd() / ".tmp-tests"
        temp_root.mkdir(exist_ok=True)
        temp_dir = temp_root / f"config-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def test_runtime_config_allowlist(self) -> None:
        config = get_runtime_config()
        self.assertIn("assistant_name", config)
        self.assertIn("vision_enabled", config)
        self.assertIn("vision_provider", config)
        self.assertIn("vision_model", config)
        self.assertIn("vision_ollama_base_url", config)
        self.assertNotIn("secret", config)

    def test_config_writer_atomicity(self) -> None:
        tmpdir = self._temp_dir()
        config_path = tmpdir / "config.json"
        config_path.write_text('{"assistant_name": "JARVIS"}', encoding="utf-8")
        success = write_config(config_path, {"assistant_name": "JARVIS 2.5"}, backup_path=tmpdir / "config.backup.json")
        self.assertTrue(success)
        self.assertEqual((tmpdir / "config.backup.json").exists(), True)
        self.assertIn("assistant_name", config_path.read_text(encoding="utf-8"))

    def test_vision_model_persists_across_simulated_restart(self) -> None:
        self.assertEqual(route_command("config set vision_model qwen2.5vl:3b"), "Configuration updated: vision_model.")
        self.assertEqual(load_config()["vision_model"], "qwen2.5vl:3b")
        reset_runtime_config()
        self.assertEqual(get_effective_runtime_config()["vision_model"], "qwen2.5vl:3b")

    def test_full_config_set_then_restart_loads_persisted_vision_model(self) -> None:
        self.assertEqual(route_command("config set vision_model qwen2.5vl:3b"), "Configuration updated: vision_model.")
        replace_runtime_config({"vision_model": ""}, status="updated")
        reset_runtime_config()
        self.assertEqual(route_command("config get vision_model"), "qwen2.5vl:3b")

    def test_existing_mutable_configuration_keys_still_persist(self) -> None:
        self.assertEqual(route_command("config set ai_enabled true"), "Configuration updated: ai_enabled.")
        self.assertEqual(load_config()["ai_enabled"], True)
        reset_runtime_config()
        self.assertEqual(get_effective_runtime_config()["ai_enabled"], True)

    def test_browser_capture_configuration_keys_persist(self) -> None:
        self.assertEqual(route_command("config set vision_browser_capture_ttl_seconds 240"), "Configuration updated: vision_browser_capture_ttl_seconds.")
        self.assertEqual(route_command("config set vision_browser_capture_enabled false"), "Configuration updated: vision_browser_capture_enabled.")
        self.assertEqual(load_config()["vision_browser_capture_ttl_seconds"], 240)
        self.assertEqual(load_config()["vision_browser_capture_enabled"], False)
        reset_runtime_config()
        config = get_effective_runtime_config()
        self.assertEqual(config["vision_browser_capture_ttl_seconds"], 240)
        self.assertEqual(config["vision_browser_capture_enabled"], False)

    def test_failed_writes_preserve_last_valid_configuration(self) -> None:
        self.assertEqual(route_command("config set vision_model qwen2.5vl:3b"), "Configuration updated: vision_model.")
        with patch("app.brain.configuration.config_commands.write_config", return_value=False):
            self.assertEqual(route_command("config set vision_model broken-model"), "Configuration change rejected.")
        self.assertEqual(load_config()["vision_model"], "qwen2.5vl:3b")

    def test_failed_write_does_not_update_runtime_snapshot(self) -> None:
        replace_runtime_config({"assistant_name": "Snapshot", "vision_model": "qwen2.5vl:3b"}, status="updated")
        before_snapshot = get_runtime_config_snapshot()
        before_generation = get_runtime_config_state().generation
        with patch("app.brain.configuration.config_commands.write_config", return_value=False):
            self.assertEqual(route_command("config set vision_model broken-model"), "Configuration change rejected.")
        self.assertEqual(get_runtime_config_snapshot(), before_snapshot)
        self.assertEqual(get_runtime_config_state().generation, before_generation)
