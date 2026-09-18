from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from app.brain.ai.json_parser import extract_json_object
from app.brain.ai.models import AIIntent, AIResponse, ProviderStatus, ProviderStatusCategory, ToolCall
from app.brain.intelligence.errors import IntelligenceProviderRequestError
from app.brain.intelligence.structured_output import validate_structured_payload

_GENERATION_READINESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ok"],
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
    },
}


class OllamaProvider:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.2", timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def status(self) -> ProviderStatus:
        return ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama")

    def check_health(self) -> bool:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 2.0)) as response:
                return 200 <= getattr(response, "status", 200) < 300
        except (urllib.error.URLError, OSError, TimeoutError, socket.timeout):
            return False

    def check_generation_ready(self) -> tuple[bool, str]:
        try:
            payload = self.request_json('Return JSON only: {"ok": true}', schema=_GENERATION_READINESS_SCHEMA)
        except IntelligenceProviderRequestError as error:
            return False, error.safe_detail
        except ValueError as error:
            return False, _sanitize_detail(str(error), limit=160)
        return (payload.get("ok") is True), "structured generation succeeded"

    def generate_text(self, prompt: str) -> AIResponse:
        try:
            body = self._request(prompt)
            payload = json.loads(body)
            message = payload.get("response", "") if isinstance(payload, dict) else ""
            if not isinstance(message, str):
                message = ""
            return AIResponse(category="conversation", message=message.strip(), intent=AIIntent.CONVERSATION, status=self.status())
        except (TimeoutError, socket.timeout):
            return AIResponse(
                category="fallback",
                message="",
                status=ProviderStatus(category=ProviderStatusCategory.TIMEOUT, provider="ollama"),
            )
        except IntelligenceProviderRequestError as error:
            if error.category == "provider_timeout":
                status = ProviderStatusCategory.TIMEOUT
            elif error.category == "provider_connection_failed":
                status = ProviderStatusCategory.UNAVAILABLE
            else:
                status = ProviderStatusCategory.ERROR
            return AIResponse(
                category="fallback",
                message="",
                status=ProviderStatus(category=status, provider="ollama"),
            )
        except urllib.error.URLError as error:
            status = ProviderStatusCategory.TIMEOUT if isinstance(getattr(error, "reason", None), TimeoutError) else ProviderStatusCategory.UNAVAILABLE
            return AIResponse(
                category="fallback",
                message="",
                status=ProviderStatus(category=status, provider="ollama"),
            )
        except (ValueError, OSError, json.JSONDecodeError):
            return AIResponse(
                category="fallback",
                message="",
                status=ProviderStatus(category=ProviderStatusCategory.ERROR, provider="ollama"),
            )

    def generate_structured(self, prompt: str) -> AIResponse:
        try:
            payload = self.request_json(prompt)
            return self._build_response(payload)
        except IntelligenceProviderRequestError as error:
            if error.category == "provider_timeout":
                status = ProviderStatusCategory.TIMEOUT
            elif error.category == "provider_connection_failed":
                status = ProviderStatusCategory.UNAVAILABLE
            else:
                status = ProviderStatusCategory.ERROR
            return AIResponse(category="fallback", message="I am offline right now.", status=ProviderStatus(category=status, provider="ollama"))
        except (TimeoutError, socket.timeout):
            return AIResponse(category="fallback", message="I am offline right now.", status=ProviderStatus(category=ProviderStatusCategory.TIMEOUT, provider="ollama"))
        except urllib.error.URLError:
            return AIResponse(category="fallback", message="I am offline right now.", status=ProviderStatus(category=ProviderStatusCategory.UNAVAILABLE, provider="ollama"))
        except (ValueError, OSError):
            return AIResponse(category="fallback", message="I am offline right now.", status=ProviderStatus(category=ProviderStatusCategory.ERROR, provider="ollama"))

    def request_json(self, prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        body = self._request(prompt, structured=True, schema=schema)
        envelope = self._decode_envelope(body)
        payload = self._extract_structured_payload(envelope)
        if schema is not None:
            validate_structured_payload(payload, schema)
        return payload

    def _request(self, prompt: str, *, structured: bool = False, schema: dict[str, Any] | None = None) -> str:
        payload_body: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if schema is not None:
            payload_body["format"] = schema
        elif structured:
            payload_body["format"] = "json"
        payload = json.dumps(payload_body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read().decode("utf-8")
            return data
        except urllib.error.HTTPError as error:
            raise IntelligenceProviderRequestError(
                "provider_http_error",
                _sanitize_http_error_detail(error),
                status_code=error.code,
            ) from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise IntelligenceProviderRequestError("provider_timeout", "Ollama request timed out") from error
            raise IntelligenceProviderRequestError(
                "provider_connection_failed",
                _sanitize_detail(f"Ollama request failed: {error.reason if hasattr(error, 'reason') else error}", limit=160),
            ) from error
        except (TimeoutError, socket.timeout) as error:
            raise IntelligenceProviderRequestError("provider_timeout", "Ollama request timed out") from error

    def _decode_envelope(self, body: str) -> dict[str, Any]:
        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError("provider envelope JSON decode failed") from error
        if not isinstance(envelope, dict):
            raise ValueError("provider envelope is invalid")
        return envelope

    def _extract_structured_payload(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if "response" not in envelope and any(key in envelope for key in {"category", "goal", "steps", "intent", "status"}):
            return envelope
        response_payload = envelope.get("response", "")
        if isinstance(response_payload, dict):
            return response_payload
        if not isinstance(response_payload, str):
            raise ValueError("provider response is invalid")
        return extract_json_object(response_payload)

    def _build_response(self, payload: dict) -> AIResponse:
        if isinstance(payload.get("category"), str):
            category = payload["category"]
        else:
            category = "fallback"
        message = payload.get("message", "") if isinstance(payload.get("message"), str) else ""
        intent_value = payload.get("intent") if isinstance(payload.get("intent"), str) else None
        intent = AIIntent(intent_value) if intent_value in {item.value for item in AIIntent} else None
        tool_call_payload = payload.get("tool_call") if isinstance(payload.get("tool_call"), dict) else None
        tool_call = None
        if tool_call_payload is not None:
            tool_call_name = tool_call_payload.get("name") if isinstance(tool_call_payload.get("name"), str) else ""
            tool_args = tool_call_payload.get("arguments") if isinstance(tool_call_payload.get("arguments"), dict) else {}
            tool_call = ToolCall(name=tool_call_name, arguments=tool_args)
        return AIResponse(category=category, message=message, intent=intent, tool_call=tool_call, status=self.status())


def _sanitize_http_error_detail(error: urllib.error.HTTPError) -> str:
    body = ""
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    finally:
        try:
            error.close()
        except Exception:
            pass
    detail = _extract_error_message(body) or f"Ollama generate request returned HTTP {error.code}"
    return _sanitize_detail(f"HTTP {error.code} from Ollama generate endpoint: {detail}", limit=220)


def _extract_error_message(body: str) -> str:
    if not body.strip():
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return body


def _sanitize_detail(text: str, *, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 14] + "...[truncated]"
