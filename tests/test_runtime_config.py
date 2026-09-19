import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.configuration.runtime_config import get_effective_runtime_config, get_runtime_config, get_runtime_config_snapshot, replace_runtime_config, reset_runtime_config, set_runtime_config_value
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

    def test_voice_input_device_settings_persist(self) -> None:
        # RFC-009 mic pipeline fix: the native capture device/rate/channel-count are
        # configurable, defaulting to 1/44100/4 -- see app.brain.voice.audio_io.
        self.assertEqual(get_effective_runtime_config()["voice_input_device"], 1)
        self.assertEqual(get_effective_runtime_config()["voice_input_sample_rate"], 44100)
        self.assertEqual(get_effective_runtime_config()["voice_input_channels"], 4)
        self.assertEqual(route_command("config set voice_input_device 2"), "Configuration updated: voice_input_device.")
        self.assertEqual(route_command("config set voice_input_sample_rate 48000"), "Configuration updated: voice_input_sample_rate.")
        self.assertEqual(route_command("config set voice_input_channels 2"), "Configuration updated: voice_input_channels.")
        self.assertEqual(load_config()["voice_input_device"], 2)
        self.assertEqual(load_config()["voice_input_sample_rate"], 48000)
        self.assertEqual(load_config()["voice_input_channels"], 2)
        reset_runtime_config()
        config = get_effective_runtime_config()
        self.assertEqual(config["voice_input_device"], 2)
        self.assertEqual(config["voice_input_sample_rate"], 48000)
        self.assertEqual(config["voice_input_channels"], 2)

    def test_voice_require_speaker_verification_defaults_off_and_is_configurable(self) -> None:
        # Owner's explicit request: plain voice control, no speaker recognition. Speaker
        # verification is opt-in now (default False), not opt-out, but stays available for
        # anyone who wants RFC-009's original hard-boundary behavior back.
        self.assertEqual(get_effective_runtime_config()["voice_require_speaker_verification"], False)
        self.assertEqual(route_command("config set voice_require_speaker_verification true"), "Configuration updated: voice_require_speaker_verification.")
        self.assertEqual(load_config()["voice_require_speaker_verification"], True)
        reset_runtime_config()
        self.assertEqual(get_effective_runtime_config()["voice_require_speaker_verification"], True)
        self.assertEqual(route_command("config set voice_require_speaker_verification not-a-bool"), "Configuration change rejected.")

    def test_voice_verification_threshold_default_is_data_driven_recalibration(self) -> None:
        # Regression test for the recalibration itself. First pass: the original 0.75
        # default rejected a real enrolled owner's own genuine voice (live similarity
        # 0.7311, logged on real hardware), which motivated 0.75 -> 0.6. Second pass, after
        # the mic/enrollment pipeline bugs this investigation found were fixed (native
        # capture, SpeechBrain's Windows symlink crash, silence trimming, stale enrollment
        # vs. current pipeline): a *freshly re-enrolled* profile under the fully-fixed
        # pipeline still saw genuine-owner live attempts land at 0.5883, 0.4662, and 0.2318
        # -- i.e. 0.6 was still too high for this real deployment even with every mechanical
        # cause eliminated. The default must stay below the worst *clean* genuine-match
        # score observed (0.4662) while remaining above speechbrain's own 0.25 reference
        # boundary for this model.
        self.assertEqual(get_effective_runtime_config()["voice_verification_threshold"], 0.4)
        self.assertLess(get_effective_runtime_config()["voice_verification_threshold"], 0.4662)
        self.assertGreater(get_effective_runtime_config()["voice_verification_threshold"], 0.25)
        self.assertEqual(route_command("config set voice_verification_threshold 0.5"), "Configuration updated: voice_verification_threshold.")
        self.assertEqual(load_config()["voice_verification_threshold"], 0.5)
        reset_runtime_config()
        self.assertEqual(get_effective_runtime_config()["voice_verification_threshold"], 0.5)

    def test_config_set_parses_float_values_not_just_bool_and_int(self) -> None:
        # Found while making voice_verification_threshold's new default retunable from the
        # CLI: `config set` never parsed a float at all (only "true"/"false" and a
        # plain-digit int), so any float-typed setting -- currently only
        # voice_verification_threshold -- silently returned "Configuration change
        # rejected." for a value like "0.65", with nothing indicating the real cause was a
        # parsing gap rather than a bad value. String-valued keys must still round-trip as
        # plain strings, including ones with a decimal-looking value it can't parse.
        self.assertEqual(route_command("config set voice_verification_threshold 0.65"), "Configuration updated: voice_verification_threshold.")
        self.assertEqual(load_config()["voice_verification_threshold"], 0.65)
        self.assertEqual(route_command("config set vision_model qwen2.5vl:3b"), "Configuration updated: vision_model.")
        self.assertEqual(load_config()["vision_model"], "qwen2.5vl:3b")

    def test_voice_input_device_accepts_zero_but_rejects_out_of_range(self) -> None:
        # Device index 0 is a legitimate PortAudio device (the first one) and must not be
        # treated as falsy/invalid the way most other voice_* integer settings treat 0.
        self.assertEqual(route_command("config set voice_input_device 0"), "Configuration updated: voice_input_device.")
        self.assertEqual(load_config()["voice_input_device"], 0)
        self.assertEqual(route_command("config set voice_input_device -1"), "Configuration change rejected.")
        self.assertEqual(route_command("config set voice_input_channels 0"), "Configuration change rejected.")
        self.assertEqual(route_command("config set voice_input_sample_rate 1000"), "Configuration change rejected.")

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

    def test_config_set_preserves_active_session_only_override(self) -> None:
        # Regression test for a bug where config_set() replaced the entire in-memory
        # runtime snapshot with file-only content, silently discarding session-only
        # toggles (like "ai on") that are never written to config.json.
        self.assertEqual(route_command("ai on"), "AI enabled for this session.")
        self.assertEqual(route_command("ai status"), "AI is enabled.")
        self.assertEqual(route_command("config set ollama_model llama3.2:latest"), "Configuration updated: ollama_model.")
        self.assertEqual(route_command("ai status"), "AI is enabled.")
        self.assertEqual(load_config()["ollama_model"], "llama3.2:latest")

    def test_config_set_preserves_multiple_session_only_overrides(self) -> None:
        self.assertEqual(route_command("ai on"), "AI enabled for this session.")
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        self.assertEqual(route_command("config set ollama_model llama3.2:latest"), "Configuration updated: ollama_model.")
        self.assertEqual(route_command("ai status"), "AI is enabled.")
        self.assertEqual(route_command("developer mode status"), "Developer mode is enabled.")

    def test_config_set_on_same_key_overrides_stale_session_value(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        self.assertEqual(get_effective_runtime_config()["ai_enabled"], True)
        self.assertEqual(route_command("config set ai_enabled false"), "Configuration updated: ai_enabled.")
        self.assertEqual(get_effective_runtime_config()["ai_enabled"], False)
        self.assertEqual(route_command("ai status"), "AI is disabled.")
