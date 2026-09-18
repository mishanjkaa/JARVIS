from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Mapping

from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.location.server import register_get_route, register_route
from app.brain.voice.audio_io import resample_audio, wav_bytes_to_samples
from app.brain.voice.controller import get_voice_controller
from app.brain.voice.errors import VoiceCaptureError, VoiceError
from app.brain.voice.verification import VOICE_SAMPLE_RATE_HZ

# RFC-009: the phone-facing voice route rides the same Tailscale-bound listener RFC-010
# stood up (see app.brain.location.server's module docstring) -- one bind host, one
# location_shared_secret, one port to allow in Tailscale ACLs, rather than a second
# listener for voice specifically.
VOICE_TURN_PATH = "/voice/turn"
VOICE_CLIENT_PATH = "/voice/client"

_CLIENT_PAGE_PATH = Path(__file__).resolve().parent / "static" / "voice_client.html"


def _extract_bearer_token(headers: Mapping[str, str]) -> str:
    value = headers.get("Authorization", "") or ""
    if not value.startswith("Bearer "):
        return ""
    return value[len("Bearer "):].strip()


def _json_response(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def handle_voice_turn_request(*, headers: Mapping[str, str], raw_body: bytes) -> tuple[int, bytes]:
    """The phone push-to-talk POST /voice/turn handler, kept independent of
    BaseHTTPRequestHandler so it can be unit-tested without opening a real socket. The
    request body is one WAV recording; the response is JSON carrying the transcript, the
    reply text, and the reply audio as base64 WAV. A rejected request (bad/missing shared
    secret) never reaches speaker verification or transcription."""
    config = get_effective_runtime_config()
    expected_secret = str(config.get("location_shared_secret", ""))
    token = _extract_bearer_token(headers)
    if not expected_secret or token != expected_secret:
        record_audit_event("voice_turn_rejected", message="unauthenticated /voice/turn POST")
        return 401, _json_response({"result": "unauthorized"})

    try:
        samples, sample_rate = wav_bytes_to_samples(raw_body)
        samples = resample_audio(samples, sample_rate, VOICE_SAMPLE_RATE_HZ)
    except VoiceCaptureError:
        return 400, _json_response({"result": "invalid body"})

    try:
        from app.brain.router import route_command

        result = get_voice_controller().handle_voice_turn(samples, route_text=route_command)
    except VoiceError as error:
        return 200, _json_response({"result": "no_action", "error": str(error)})
    except Exception:
        return 200, _json_response({"result": "no_action", "error": "Voice turn failed safely."})

    return 200, _json_response(
        {
            "result": "ok",
            "transcript": result.transcript,
            "reply_text": result.reply_text,
            "reply_audio_wav_base64": base64.b64encode(result.reply_audio_wav).decode("ascii") if result.reply_audio_wav else "",
        }
    )


def serve_voice_client_page() -> tuple[int, bytes, str]:
    try:
        body = _CLIENT_PAGE_PATH.read_bytes()
    except OSError:
        return 404, _json_response({"result": "not found"}), "application/json"
    return 200, body, "text/html; charset=utf-8"


register_route(VOICE_TURN_PATH, lambda headers, raw_body: handle_voice_turn_request(headers=headers, raw_body=raw_body))
register_get_route(VOICE_CLIENT_PATH, serve_voice_client_page)
