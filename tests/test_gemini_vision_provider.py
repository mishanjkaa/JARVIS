"""Owner-requested (2026-09-19 "screen understanding" discussion): a cloud vision provider
using Google's free-tier Gemini API, for real screen-reading accuracy the local Ollama
vision models can't match on her hardware. Mirrors the transport-level test shape already
used for the local Ollama vision provider in tests/test_vision_runtime.py -- same mocked
urllib.request.urlopen pattern, same VisionEvidence/VisionProviderStatus expectations --
since GeminiVisionProvider reuses the exact same schemas/evidence-building helpers.
"""

from __future__ import annotations

import json
import os
import unittest
import urllib.error
from datetime import datetime, timezone
from unittest.mock import patch

from app.brain.configuration.config_commands import config_get, config_set, config_show
from app.brain.configuration.runtime_config import get_effective_runtime_config, replace_runtime_config, reset_runtime_config, set_runtime_config_value
from app.brain.vision.controller import get_vision_controller, reset_vision_controller
from app.brain.vision.errors import VisionProviderError, VisionProviderUnavailableError
from app.brain.vision.gemini_provider import GeminiVisionProvider
from app.brain.vision.models import LoadedVisionImage, VisionFrame
from app.brain.vision.ollama_provider import OllamaVisionProvider


def _tiny_png_image() -> LoadedVisionImage:
    tiny_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
        b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    now = datetime.now(timezone.utc).isoformat()
    frame = VisionFrame(
        frame_id="frame-test",
        source_type="file",
        safe_display_name="test.png",
        source_hash="test-hash",
        mime_type="image/png",
        width=1,
        height=1,
        created_at=now,
        expires_at=now,
        temporary_copy=False,
        trust_classification="trusted_root_file",
        file_size_bytes=len(tiny_png),
    )
    return LoadedVisionImage(
        frame=frame,
        original_path="",
        safe_display_name="test.png",
        mime_type="image/png",
        width=1,
        height=1,
        source_hash="test-hash",
        file_size_bytes=len(tiny_png),
        image_bytes=tiny_png,
        temp_copy_path="",
    )


class _MockHttpResponse:
    def __init__(self, payload, *, status: int = 200) -> None:
        body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self._body = body.encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(status: int) -> urllib.error.HTTPError:
    import io

    return urllib.error.HTTPError(url="https://generativelanguage.googleapis.com/", code=status, msg="error", hdrs=None, fp=io.BytesIO(b"{}"))


def _generate_content_response(payload: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


class GeminiVisionProviderConfigurationTests(unittest.TestCase):
    def test_uses_config_key_when_present(self) -> None:
        provider = GeminiVisionProvider(api_key="from-config", model="gemini-2.0-flash")
        self.assertEqual(provider.api_key, "from-config")

    def test_falls_back_to_environment_variable_when_config_key_is_blank(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}, clear=False):
            provider = GeminiVisionProvider(api_key="", model="gemini-2.0-flash")
            self.assertEqual(provider.api_key, "from-env")

    def test_config_key_takes_priority_over_environment_variable(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}, clear=False):
            provider = GeminiVisionProvider(api_key="from-config", model="gemini-2.0-flash")
            self.assertEqual(provider.api_key, "from-config")

    def test_status_reports_not_configured_without_a_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            provider = GeminiVisionProvider(api_key="", model="gemini-2.0-flash")
            status = provider.status()
            self.assertFalse(status.configured)
            self.assertIn("API key", status.generation_detail)

    def test_status_reports_not_configured_without_a_model(self) -> None:
        provider = GeminiVisionProvider(api_key="a-key", model="")
        status = provider.status()
        self.assertFalse(status.configured)


class GeminiVisionProviderModelCapabilityTests(unittest.TestCase):
    def test_known_vision_model_prefix_is_recognized(self) -> None:
        provider = GeminiVisionProvider(api_key="a-key", model="gemini-2.0-flash")
        with patch("urllib.request.urlopen", return_value=_MockHttpResponse({"name": "models/gemini-2.0-flash"})):
            installed, capable = provider._model_capabilities()
        self.assertTrue(installed)
        self.assertTrue(capable)

    def test_unknown_model_name_is_not_reported_vision_capable(self) -> None:
        provider = GeminiVisionProvider(api_key="a-key", model="some-future-text-only-model")
        with patch("urllib.request.urlopen", return_value=_MockHttpResponse({"name": "models/some-future-text-only-model"})):
            installed, capable = provider._model_capabilities()
        self.assertTrue(installed)
        self.assertFalse(capable)

    def test_missing_model_on_the_account_is_reported_as_not_installed(self) -> None:
        provider = GeminiVisionProvider(api_key="a-key", model="gemini-2.0-flash")
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            installed, capable = provider._model_capabilities()
        self.assertFalse(installed)
        self.assertFalse(capable)


class GeminiVisionProviderRequestTests(unittest.TestCase):
    def _ready_provider(self) -> GeminiVisionProvider:
        return GeminiVisionProvider(api_key="a-key", model="gemini-2.0-flash")

    def test_describe_image_happy_path(self) -> None:
        provider = self._ready_provider()
        image = _tiny_png_image()
        response = _generate_content_response(
            {"description": "A blue button.", "confidence": 0.9, "warnings": [], "regions": []}
        )
        with patch("urllib.request.urlopen", return_value=_MockHttpResponse(response)):
            evidence = provider.describe_image(image, detail_level="normal")
        self.assertEqual(evidence.provider, "gemini")
        self.assertEqual(evidence.description, "A blue button.")
        self.assertAlmostEqual(evidence.confidence, 0.9)

    def test_rate_limit_response_raises_a_clear_unavailable_error(self) -> None:
        provider = self._ready_provider()
        image = _tiny_png_image()
        with patch("urllib.request.urlopen", side_effect=_http_error(429)):
            with self.assertRaises(VisionProviderUnavailableError) as error:
                provider.describe_image(image, detail_level="normal")
        self.assertIn("rate limit", str(error.exception).lower())

    def test_unauthorized_response_raises_a_clear_unavailable_error(self) -> None:
        provider = self._ready_provider()
        image = _tiny_png_image()
        with patch("urllib.request.urlopen", side_effect=_http_error(403)):
            with self.assertRaises(VisionProviderUnavailableError) as error:
                provider.describe_image(image, detail_level="normal")
        self.assertIn("rejected", str(error.exception).lower())

    def test_malformed_response_text_raises_provider_error(self) -> None:
        provider = self._ready_provider()
        image = _tiny_png_image()
        with patch("urllib.request.urlopen", return_value=_MockHttpResponse({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})):
            with self.assertRaises(VisionProviderError) as error:
                provider.describe_image(image, detail_level="normal")
        self.assertIn("malformed", str(error.exception).lower())

    def test_missing_api_key_fails_before_any_network_call(self) -> None:
        provider = GeminiVisionProvider(api_key="", model="gemini-2.0-flash")
        image = _tiny_png_image()
        with patch("urllib.request.urlopen") as urlopen_mock:
            with self.assertRaises(VisionProviderUnavailableError):
                provider.describe_image(image, detail_level="normal")
            urlopen_mock.assert_not_called()


class VisionControllerGeminiDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()
        reset_vision_controller()

    def tearDown(self) -> None:
        reset_runtime_config()
        reset_vision_controller()

    def test_defaults_to_ollama_provider(self) -> None:
        controller = get_vision_controller()
        self.assertIsInstance(controller.provider(), OllamaVisionProvider)

    def test_gemini_provider_selected_when_configured(self) -> None:
        set_runtime_config_value("vision_provider", "gemini")
        set_runtime_config_value("vision_model", "gemini-2.0-flash")
        controller = get_vision_controller()
        provider = controller.provider()
        self.assertIsInstance(provider, GeminiVisionProvider)
        self.assertEqual(provider.model, "gemini-2.0-flash")


class VisionGeminiApiKeyIsACredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()

    def tearDown(self) -> None:
        reset_runtime_config()

    def test_cannot_be_set_via_config_set(self) -> None:
        result = config_set("vision_gemini_api_key", "a-real-key")
        self.assertEqual(result, "Configuration change rejected.")

    def test_is_redacted_by_config_get_when_present_on_disk(self) -> None:
        # Simulates the owner having edited config.json directly, the only supported way to
        # set this credential -- see config/settings.py's comment on this key.
        config = get_effective_runtime_config()
        config["vision_gemini_api_key"] = "a-real-key"
        replace_runtime_config(config)
        self.assertEqual(config_get("vision_gemini_api_key"), "<redacted>")

    def test_is_redacted_by_config_show(self) -> None:
        config = get_effective_runtime_config()
        config["vision_gemini_api_key"] = "a-real-key"
        replace_runtime_config(config)
        output = config_show()
        self.assertIn("vision_gemini_api_key: <redacted>", output)
        self.assertNotIn("a-real-key", output)


if __name__ == "__main__":
    unittest.main()
