"""RFC-009 follow-up: the owner asked not to have to type 'voice talk' before every
utterance, and to have JARVIS able to hear and act on commands as soon as it starts (no
'ai on'/'voice on'/'voice talk' typed first). Covers three things added for that:

- `ai_enabled`/`voice_enabled` now default to True (config/settings.py and the
  runtime_config fallbacks) instead of False -- see test_runtime_config.py for the
  round-trip/allowlist coverage of the new `voice_listen_on_startup` key specifically;
  this file focuses on the new *behavior* those defaults and the key enable.
- `app.brain.router._voice_listen_loop` / the "voice listen" command: a continuous
  push-to-talk loop (not a true wake-word engine, which RFC-009 leaves out of scope) that
  keeps calling VoiceController.push_to_talk_local() until a recognized stop phrase,
  Ctrl+C, or a structural voice error ends it.
- `main._maybe_start_voice_listening_on_launch`: entering that same loop automatically at
  startup when `voice_enabled` and `voice_listen_on_startup` both say to.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.voice.errors import VoiceCaptureError, VoiceDisabledError, VoiceProviderError
from app.brain.voice.models import VoiceTurnResult
from config.config_loader import set_config_path_override


class _ScriptedVoiceController:
    """Stands in for the real VoiceController. Each push_to_talk_local() call pops the
    next scripted item: an Exception instance is raised (mirroring a real capture/STT
    failure), a string is treated as this turn's transcript and handed to the loop's own
    `route_text` callback -- exactly what handle_voice_turn really does -- so these tests
    exercise the loop's actual stop-phrase-interception and error-handling logic, not a
    re-implementation of it."""

    def __init__(self, script: list) -> None:
        self._script = list(script)

    def push_to_talk_local(self, *, route_text):
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        transcript = item
        reply_text = route_text(transcript)
        return VoiceTurnResult(transcript=transcript, reply_text=reply_text)


class _ImmediateKeyboardInterruptController:
    def push_to_talk_local(self, *, route_text):
        raise KeyboardInterrupt()


class VoiceListenLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()

    def tearDown(self) -> None:
        reset_runtime_config()

    @patch("app.brain.router.route_command")
    @patch("app.brain.router.get_voice_controller")
    def test_stop_phrase_is_intercepted_before_routing(self, get_voice_controller_mock, route_command_mock) -> None:
        from app.brain.router import _voice_listen_loop

        get_voice_controller_mock.return_value = _ScriptedVoiceController(["open notepad", "stop listening"])
        route_command_mock.return_value = "Opened Notepad."

        result = _voice_listen_loop()

        self.assertEqual(result, "Stopped continuous voice listening.")
        # The stop phrase itself must never reach the real command router -- only the
        # genuine command before it -- otherwise it would be routed as an unrelated
        # conversational turn (and spoken back) instead of cleanly ending the loop.
        route_command_mock.assert_called_once_with("open notepad")

    @patch("app.brain.router.route_command")
    @patch("app.brain.router.get_voice_controller")
    def test_no_speech_recognized_is_swallowed_and_loop_continues(self, get_voice_controller_mock, route_command_mock) -> None:
        from app.brain.router import _voice_listen_loop

        get_voice_controller_mock.return_value = _ScriptedVoiceController([
            VoiceProviderError("No speech was recognized."),
            VoiceProviderError("No speech was recognized."),
            "open browser",
            "stop listening",
        ])
        route_command_mock.return_value = "Opened the default web browser."

        result = _voice_listen_loop()

        self.assertEqual(result, "Stopped continuous voice listening.")
        route_command_mock.assert_called_once_with("open browser")

    @patch("app.brain.router.get_voice_controller")
    def test_capture_error_ends_the_loop_instead_of_retrying_forever(self, get_voice_controller_mock) -> None:
        from app.brain.router import _voice_listen_loop

        error = VoiceCaptureError("Microphone recording failed on voice input device 1 (44100 Hz, 4 channel(s)).")
        get_voice_controller_mock.return_value = _ScriptedVoiceController([error])

        result = _voice_listen_loop()

        self.assertEqual(result, str(error))

    @patch("app.brain.router.get_voice_controller")
    def test_disabled_error_ends_the_loop(self, get_voice_controller_mock) -> None:
        from app.brain.router import _voice_listen_loop

        error = VoiceDisabledError("Voice is disabled. Enable it with 'voice on'.")
        get_voice_controller_mock.return_value = _ScriptedVoiceController([error])

        result = _voice_listen_loop()

        self.assertEqual(result, str(error))

    @patch("app.brain.router.get_voice_controller")
    def test_keyboard_interrupt_stops_the_loop_cleanly(self, get_voice_controller_mock) -> None:
        from app.brain.router import _voice_listen_loop

        get_voice_controller_mock.return_value = _ImmediateKeyboardInterruptController()

        result = _voice_listen_loop()

        self.assertEqual(result, "Stopped continuous voice listening.")

    @patch("app.brain.router.route_command")
    @patch("app.brain.router.get_voice_controller")
    def test_shutdown_reply_propagates_out_of_the_loop(self, get_voice_controller_mock, route_command_mock) -> None:
        from app.brain.router import _voice_listen_loop

        get_voice_controller_mock.return_value = _ScriptedVoiceController(["exit"])
        route_command_mock.return_value = "shutdown"

        result = _voice_listen_loop()

        self.assertEqual(result, "shutdown")

    @patch("app.brain.router._voice_listen_loop", return_value="Stopped continuous voice listening.")
    def test_voice_listen_command_dispatches_to_the_loop(self, loop_mock) -> None:
        from app.brain.router import route_command

        self.assertEqual(route_command("voice listen"), "Stopped continuous voice listening.")
        loop_mock.assert_called_once_with()

    def test_voice_stop_listening_typed_outside_a_loop_is_a_no_op(self) -> None:
        from app.brain.router import route_command

        # Typing this only ever reaches route_command when no loop is actually running --
        # a running loop blocks this exact code path until it returns -- so there is
        # nothing to stop; this just avoids "unrecognized command" confusion.
        self.assertEqual(route_command("voice stop listening"), "Not currently listening continuously.")


class VoiceListenOnStartupDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()
        # "config set" (unlike "voice on"/"ai on") persists to disk via the real config
        # path -- without overriding it to a throwaway temp file here the way
        # test_runtime_config.py's RuntimeConfigTests does, this test's own
        # test_voice_listen_on_startup_is_configurable would permanently write
        # voice_listen_on_startup=false into this checkout's real config/config.json,
        # corrupting the "defaults to True" assumption for every test that runs after it
        # in the same process (found the hard way: passed alone, failed as part of the
        # full suite).
        temp_root = Path.cwd() / ".tmp-tests"
        temp_root.mkdir(exist_ok=True)
        temp_dir = temp_root / f"voice-listen-config-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        config_path = temp_dir / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        set_config_path_override(config_path)

    def tearDown(self) -> None:
        set_config_path_override(None)
        reset_runtime_config()

    def test_ai_and_voice_and_listen_on_startup_default_to_true(self) -> None:
        from app.brain.configuration.runtime_config import get_effective_runtime_config

        # Owner's explicit request: JARVIS should be immediately usable at startup, with
        # no "ai on"/"voice on" typed first, and able to hear commands with no "voice
        # talk" typed first either.
        config = get_effective_runtime_config()
        self.assertEqual(config["ai_enabled"], True)
        self.assertEqual(config["voice_enabled"], True)
        self.assertEqual(config["voice_listen_on_startup"], True)

    def test_voice_listen_on_startup_is_configurable(self) -> None:
        from app.brain.router import route_command
        from config.config_loader import load_config

        self.assertEqual(route_command("config set voice_listen_on_startup false"), "Configuration updated: voice_listen_on_startup.")
        self.assertEqual(load_config()["voice_listen_on_startup"], False)
        self.assertEqual(route_command("config set voice_listen_on_startup not-a-bool"), "Configuration change rejected.")


class MaybeStartVoiceListeningOnLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()

    def tearDown(self) -> None:
        reset_runtime_config()

    def test_skips_the_loop_when_voice_is_disabled(self) -> None:
        import main

        set_runtime_config_value("voice_enabled", False)
        set_runtime_config_value("voice_listen_on_startup", True)
        with patch("app.brain.router._voice_listen_loop") as loop_mock:
            self.assertTrue(main._maybe_start_voice_listening_on_launch())
        loop_mock.assert_not_called()

    def test_skips_the_loop_when_startup_flag_is_off(self) -> None:
        import main

        set_runtime_config_value("voice_enabled", True)
        set_runtime_config_value("voice_listen_on_startup", False)
        with patch("app.brain.router._voice_listen_loop") as loop_mock:
            self.assertTrue(main._maybe_start_voice_listening_on_launch())
        loop_mock.assert_not_called()

    def test_starts_the_loop_and_falls_back_to_typing_when_it_ends_normally(self) -> None:
        import main

        set_runtime_config_value("voice_enabled", True)
        set_runtime_config_value("voice_listen_on_startup", True)
        with patch("app.brain.router._voice_listen_loop", return_value="Stopped continuous voice listening.") as loop_mock:
            self.assertTrue(main._maybe_start_voice_listening_on_launch())
        loop_mock.assert_called_once_with()

    def test_returns_false_when_owner_exits_during_startup_listening(self) -> None:
        import main

        set_runtime_config_value("voice_enabled", True)
        set_runtime_config_value("voice_listen_on_startup", True)
        with patch("app.brain.router._voice_listen_loop", return_value="shutdown"):
            self.assertFalse(main._maybe_start_voice_listening_on_launch())


if __name__ == "__main__":
    unittest.main()
