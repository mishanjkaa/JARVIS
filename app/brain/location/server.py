from __future__ import annotations

import json
import logging
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.location.controller import get_location_controller
from app.brain.location.errors import LocationBindError

OVERLAND_PATH = "/location/overland"
# Generous for a normal Overland batch (a handful of points); guards the handler against
# an oversized body from anything that isn't the real app.
_MAX_BODY_BYTES = 262_144

_server_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None


def is_bind_host_present(host: str) -> bool:
    """True only if `host` is a real, non-wildcard address of a local network interface.
    Used both to validate location_bind_host before binding and to report Tailscale
    connectivity in `location status` (the configured Tailscale IP is only reachable while
    the Tailscale interface holding it is up)."""
    if not host or not host.strip() or host.strip() in {"0.0.0.0", "::"}:
        return False
    normalized = host.strip()
    family = socket.AF_INET6 if ":" in normalized else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.bind((normalized, 0))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def validate_bind_host(host: str) -> None:
    if not host or not host.strip():
        raise LocationBindError("location_bind_host is not configured. Set it to this machine's Tailscale IP.")
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::"}:
        raise LocationBindError("location_bind_host must not be a wildcard bind address (0.0.0.0/::); it must be this machine's Tailscale IP.")
    if not is_bind_host_present(normalized):
        raise LocationBindError(f"'{normalized}' is not an address of any network interface on this machine.")


def _extract_bearer_token(headers: Mapping[str, str]) -> str:
    value = headers.get("Authorization", "") or ""
    if not value.startswith("Bearer "):
        return ""
    return value[len("Bearer "):].strip()


def _freshest_feature_index(features: list[Any]) -> int:
    # Overland batches are typically chronological already, but picking the point with the
    # latest parseable timestamp is more robust than blindly trusting array order; falling
    # back to the last item keeps behavior sane if no timestamp parses.
    best_index = len(features) - 1
    best_timestamp: datetime | None = None
    for index, feature in enumerate(features):
        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        raw_timestamp = properties.get("timestamp") if isinstance(properties, dict) else None
        if not isinstance(raw_timestamp, str):
            continue
        try:
            parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if best_timestamp is None or parsed > best_timestamp:
            best_timestamp = parsed
            best_index = index
    return best_index


def _json_response(payload: dict[str, str]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def handle_overland_request(*, headers: Mapping[str, str], raw_body: bytes) -> tuple[int, bytes]:
    """The Overland POST /location/overland handler, kept independent of BaseHTTPRequestHandler
    so it can be unit-tested without opening a real socket. A rejected request (bad or
    missing shared secret, malformed body) never touches stored state."""
    config = get_effective_runtime_config()
    expected_secret = str(config.get("location_shared_secret", ""))
    token = _extract_bearer_token(headers)
    if not expected_secret or token != expected_secret:
        record_audit_event("location_point_rejected", message="unauthenticated Overland POST")
        return 401, _json_response({"result": "unauthorized"})

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 400, _json_response({"result": "invalid body"})

    locations = payload.get("locations") if isinstance(payload, dict) else None
    if not isinstance(locations, list) or not locations:
        return 400, _json_response({"result": "invalid body"})

    feature = locations[_freshest_feature_index(locations)]
    if not isinstance(feature, dict):
        return 400, _json_response({"result": "invalid body"})
    geometry = feature.get("geometry")
    properties = feature.get("properties")
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list) or len(coordinates) != 2:
        return 400, _json_response({"result": "invalid body"})
    try:
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return 400, _json_response({"result": "invalid body"})

    properties = properties if isinstance(properties, dict) else {}
    device_id = str(properties.get("device_id") or "default")
    raw_accuracy = properties.get("horizontal_accuracy")
    accuracy_meters = float(raw_accuracy) if isinstance(raw_accuracy, (int, float)) else None
    raw_timestamp = properties.get("timestamp")
    captured_at = raw_timestamp if isinstance(raw_timestamp, str) else ""

    get_location_controller().ingest_overland_point(
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        accuracy_meters=accuracy_meters,
        captured_at=captured_at,
    )
    record_audit_event("location_point_received", message=f"device {device_id}")
    return 200, _json_response({"result": "ok"})


class _OverlandRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return  # audit_log already records accept/reject outcomes; skip stderr access logs

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != OVERLAND_PATH:
            self._respond(404, _json_response({"result": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._respond(400, _json_response({"result": "invalid body"}))
            return
        raw_body = self.rfile.read(length)
        status_code, response_body = handle_overland_request(headers=self.headers, raw_body=raw_body)
        self._respond(status_code, response_body)

    def _respond(self, status_code: int, body: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_location_server() -> str:
    global _server, _server_thread
    with _server_lock:
        if _server is not None:
            return "Location server is already running."
        config = get_effective_runtime_config()
        if not bool(config.get("location_enabled", False)):
            return "Location is disabled; not starting the location server."
        host = str(config.get("location_bind_host", ""))
        validate_bind_host(host)
        port = int(config.get("location_port", 8766))
        server = ThreadingHTTPServer((host, port), _OverlandRequestHandler)
        thread = threading.Thread(target=server.serve_forever, name="jarvis-location-server", daemon=True)
        thread.start()
        _server = server
        _server_thread = thread
        record_audit_event("location_server_started", message=f"{host}:{port}")
        return f"Location server listening on {host}:{port}."


def stop_location_server() -> str:
    global _server, _server_thread
    with _server_lock:
        if _server is None:
            return "Location server is not running."
        _server.shutdown()
        _server.server_close()
        _server = None
        _server_thread = None
        record_audit_event("location_server_stopped", message="location server stopped")
        return "Location server stopped."


def is_location_server_running() -> bool:
    with _server_lock:
        return _server is not None


def start_location_server_if_enabled() -> None:
    """Call once at process startup (see main.py). Never silently falls back to a public
    bind: if location is enabled but location_bind_host is missing, a wildcard address, or
    not present on this machine, this logs a clear error and leaves the rest of JARVIS
    running rather than crashing or binding somewhere unsafe."""
    config = get_effective_runtime_config()
    if not bool(config.get("location_enabled", False)):
        return
    try:
        start_location_server()
    except LocationBindError as error:
        logging.getLogger(__name__).error("Location server did not start: %s", error)
