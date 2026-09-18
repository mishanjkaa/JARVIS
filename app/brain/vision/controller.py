from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from PIL import Image

from app.brain.agent.state import get_agent_runtime_state
from app.brain.audit.audit_log import record_audit_event
from app.brain.intelligence.vision_query import normalize_vision_query
from app.brain.configuration.runtime_config import get_effective_runtime_config, get_runtime_config_state
from app.brain.vision import desktop_capture_backend
from app.brain.vision.errors import VisionCaptureUnsupportedError, VisionDisabledError, VisionEvidenceExpiredError, VisionImageError, VisionPolicyError, VisionProviderError, VisionProviderUnavailableError
from app.brain.vision.image_loader import bbox_to_pixel_rect, cleanup_loaded_image, crop_loaded_image, load_local_image
from app.brain.vision.models import BrowserCaptureRecord, DesktopCaptureRecord, DesktopWindowListResult, LoadedVisionImage, VISION_DETAIL_LEVELS, VisionBoundingBox, VisionCropObservation, VisionEvidence, VisionFrame, VisionObservedObject, VisionProviderStatus, VisionRegion
from app.brain.vision.ollama_provider import OllamaVisionProvider
from app.brain.vision.provider import VisionProvider
from app.brain.vision.state import get_vision_state


class VisionController:
    def __init__(self, *, provider: VisionProvider | None = None) -> None:
        self._provider_override = provider

    def effective_config(self) -> dict[str, Any]:
        return get_effective_runtime_config()

    def provider(self) -> VisionProvider:
        if self._provider_override is not None:
            return self._provider_override
        config = self.effective_config()
        return OllamaVisionProvider(
            base_url=str(config.get("vision_ollama_base_url", "http://127.0.0.1:11434")),
            model=str(config.get("vision_model", "")),
            timeout=float(config.get("vision_timeout_seconds", 45)),
        )

    def status(self) -> VisionProviderStatus:
        status = replace(self.provider().status())
        return self._apply_cached_readiness(status)

    def status_message(self) -> str:
        config = self.effective_config()
        status = self.status()
        lines = [
            f"Vision enabled: {'yes' if config.get('vision_enabled', True) else 'no'}",
            f"Provider: {status.provider_name}",
            f"Model: {status.model}",
            f"Provider configured: {'yes' if status.configured else 'no'}",
            f"Provider reachable: {'yes' if status.provider_reachable else 'no'}",
            f"Model installed: {'yes' if status.model_installed else 'no'}",
            f"Image capability ready: {'yes' if status.image_capability_ready else 'no'}",
            f"Generation ready: {self._format_readiness(status.generation_ready)}",
        ]
        if status.generation_detail:
            lines.append(f"Detail: {status.generation_detail}")
        return "\n".join(lines)

    def provider_status_message(self) -> str:
        status = self.status()
        return "\n".join(
            [
                f"Vision provider: {status.provider_name}",
                f"Model: {status.model}",
                f"Provider configured: {'yes' if status.configured else 'no'}",
                f"Provider reachable: {'yes' if status.provider_reachable else 'no'}",
                f"Model installed: {'yes' if status.model_installed else 'no'}",
                f"Image capability ready: {'yes' if status.image_capability_ready else 'no'}",
            ]
        )

    def browser_status_message(self) -> str:
        config = self.effective_config()
        from app.brain.browser.controller import get_browser_controller

        browser_ready = get_browser_controller().runtime_ready()
        status = self.status()
        state = get_vision_state()
        with state.lock:
            active_captures = len(state.captures)
        integration_enabled = bool(config.get("vision_enabled", True) and config.get("vision_browser_capture_enabled", True))
        overall_ready = (
            integration_enabled
            and browser_ready
            and status.configured
            and status.base_url_allowed
            and status.provider_reachable
            and status.model_installed
            and status.image_capability_ready
        )
        lines = [
            f"Integration enabled: {'yes' if integration_enabled else 'no'}",
            f"Browser Runtime ready: {'yes' if browser_ready else 'no'}",
            f"Vision Runtime ready: {'yes' if status.configured and status.base_url_allowed and status.provider_reachable and status.model_installed and status.image_capability_ready else 'no'}",
            f"Provider: {status.provider_name}",
            f"Model: {status.model}",
            "Viewport capture supported: yes" if integration_enabled else "Viewport capture supported: no",
            f"Active temporary captures: {active_captures}",
            f"Capture TTL: {int(config.get('vision_browser_capture_ttl_seconds', 180))}s",
            f"Overall ready: {'yes' if overall_ready else 'no'}",
        ]
        return "\n".join(lines)

    def captures_message(self) -> str:
        removed = self.cleanup_expired_captures()
        state = get_vision_state()
        with state.lock:
            captures = sorted(state.captures.values(), key=lambda item: item.captured_at)
        if not captures:
            return "No temporary browser captures."
        lines = []
        if removed:
            lines.append(f"Expired captures cleaned: {removed}")
        lines.append("Temporary browser captures:")
        now = datetime.now(timezone.utc)
        for capture in captures[:20]:
            expired = _parse_iso(capture.expires_at) <= now
            lines.append(
                " | ".join(
                    [
                        _display_capture_id(capture.capture_id),
                        capture.source_type,
                        f"{capture.session_id}/{capture.tab_id}",
                        capture.origin or "(no origin)",
                        f"age={_age_seconds(capture.captured_at, now)}s",
                        "expired" if expired else "active",
                    ]
                )
            )
        return "\n".join(lines)

    def clear_browser_captures(self) -> str:
        removed = self._remove_captures(lambda _capture: True)
        return f"Cleared {removed} temporary browser capture{'s' if removed != 1 else ''}."

    def provider_check_message(self) -> str:
        status = self.status()
        generation_ready = status.generation_ready
        detail = status.generation_detail
        if status.configured and status.base_url_allowed and status.provider_reachable and status.model_installed and status.image_capability_ready:
            generation_ready, detail = self._perform_generation_check()
            status = self.status()
            generation_ready = status.generation_ready
            detail = status.generation_detail
        lines = [
            f"Vision provider: {status.provider_name}",
            f"Model: {status.model}",
            f"Provider configured: {'yes' if status.configured else 'no'}",
            f"Provider reachable: {'yes' if status.provider_reachable else 'no'}",
            f"Model installed: {'yes' if status.model_installed else 'no'}",
            f"Image capability ready: {'yes' if status.image_capability_ready else 'no'}",
            f"Generation ready: {self._format_readiness(generation_ready)}",
        ]
        if detail:
            lines.append(f"Generation detail: {detail}")
        return "\n".join(lines)

    def describe_image(self, *, path: str, detail_level: str = "normal") -> VisionEvidence:
        if detail_level not in VISION_DETAIL_LEVELS:
            detail_level = "normal"
        return self._run_with_image("describe_image", path=path, handler=lambda image: self.provider().describe_image(image, detail_level=detail_level))

    def extract_text(self, *, path: str, language_hint: str = "", max_characters: int | None = None) -> VisionEvidence:
        config = self.effective_config()
        char_limit = int(max_characters or config.get("vision_max_ocr_chars", 4000))
        return self._run_with_image(
            "extract_text",
            path=path,
            handler=lambda image: self.provider().extract_text(image, language_hint=language_hint, max_characters=char_limit),
        )

    def find_visual_element(self, *, path: str, query: str, max_results: int = 3) -> VisionEvidence:
        return self._run_with_image(
            "find_visual_element",
            path=path,
            handler=lambda image: self.provider().find_visual_element(image, query=query, max_results=max(1, min(max_results, int(self.effective_config().get("vision_max_regions", 8))))),
        )

    def store_browser_capture(
        self,
        *,
        owner_request_id: int | None,
        owner_agent_task_id: int | None,
        session_id: str,
        tab_id: str,
        url: str,
        origin: str,
        page_version: int,
        width: int,
        height: int,
        screenshot_pixel_width: int = 0,
        screenshot_pixel_height: int = 0,
        visual_viewport_width: float = 0.0,
        visual_viewport_height: float = 0.0,
        visual_viewport_offset_left: float = 0.0,
        visual_viewport_offset_top: float = 0.0,
        scroll_x: float = 0.0,
        scroll_y: float = 0.0,
        device_pixel_ratio: float = 0.0,
        device_scale_factor: float = 0.0,
        screenshot_scale_option: str = "",
        viewport_only: bool = True,
        mime_type: str,
        image_bytes: bytes,
        dom_elements: list[dict[str, Any]] | None = None,
    ) -> BrowserCaptureRecord:
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            raise VisionDisabledError("Vision runtime is disabled.")
        if not bool(config.get("vision_browser_capture_enabled", True)):
            raise VisionPolicyError("Browser visual evidence is disabled.")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise VisionImageError("Browser capture data is invalid.")
        max_bytes = int(config.get("vision_browser_capture_max_bytes", 2_000_000))
        max_width = int(config.get("vision_browser_capture_max_width", 1920))
        max_height = int(config.get("vision_browser_capture_max_height", 1080))
        max_pixels = int(config.get("vision_browser_capture_max_pixels", 2_073_600))
        if len(image_bytes) > max_bytes:
            raise VisionImageError("Browser capture exceeds the configured size limit.")
        if width <= 0 or height <= 0:
            raise VisionImageError("Browser capture dimensions are invalid.")
        if width > max_width or height > max_height or width * height > max_pixels:
            raise VisionImageError("Browser capture exceeds the configured viewport limits.")
        if str(mime_type).strip().lower() != "image/png":
            raise VisionImageError("Browser captures must use PNG format.")
        now = datetime.now(timezone.utc)
        capture_id = self._next_capture_id()
        digest = hashlib.sha256(bytes(image_bytes)).hexdigest()
        record = BrowserCaptureRecord(
            capture_id=capture_id,
            owner_request_id=owner_request_id,
            owner_agent_task_id=owner_agent_task_id,
            source_type="browser_viewport",
            session_id=session_id,
            tab_id=tab_id,
            url=str(url or "").strip(),
            origin=str(origin or "").strip(),
            page_version=int(page_version),
            viewport_width=int(width),
            viewport_height=int(height),
            captured_at=now.isoformat(),
            expires_at=(now + _capture_ttl(config)).isoformat(),
            image_format="png",
            source_hash=digest,
            byte_size=len(image_bytes),
            mime_type="image/png",
            safe_display_name=f"browser-capture-{capture_id[-8:]}.png",
            screenshot_pixel_width=int(screenshot_pixel_width or width),
            screenshot_pixel_height=int(screenshot_pixel_height or height),
            visual_viewport_width=float(visual_viewport_width or width),
            visual_viewport_height=float(visual_viewport_height or height),
            visual_viewport_offset_left=float(visual_viewport_offset_left or 0.0),
            visual_viewport_offset_top=float(visual_viewport_offset_top or 0.0),
            scroll_x=float(scroll_x or 0.0),
            scroll_y=float(scroll_y or 0.0),
            device_pixel_ratio=float(device_pixel_ratio or 0.0),
            device_scale_factor=float(device_scale_factor or 0.0),
            screenshot_scale_option=str(screenshot_scale_option or "").strip(),
            viewport_only=bool(viewport_only),
            dom_elements=list(dom_elements or [])[:24],
            image_bytes=bytes(image_bytes),
        )
        state = get_vision_state()
        with state.lock:
            state.captures[capture_id] = record
            state.last_safe_status = "browser_capture_ready"
        record_audit_event(
            "vision_browser_capture_created",
            task_id=owner_agent_task_id,
            message=f"{capture_id}:{record.origin or '(no origin)'}",
        )
        return record

    def describe_browser_capture(self, *, capture_id: str, detail_level: str = "normal") -> VisionEvidence:
        if detail_level not in VISION_DETAIL_LEVELS:
            detail_level = "normal"
        return self._run_with_capture(
            "describe_browser_capture",
            capture_id=capture_id,
            handler=lambda image: self.provider().describe_image(image, detail_level=detail_level),
        )

    def extract_text_from_browser_capture(
        self,
        *,
        capture_id: str,
        language_hint: str = "",
        max_characters: int | None = None,
    ) -> VisionEvidence:
        config = self.effective_config()
        char_limit = int(max_characters or config.get("vision_max_ocr_chars", 4000))
        return self._run_with_capture(
            "extract_text_from_browser_capture",
            capture_id=capture_id,
            handler=lambda image: self.provider().extract_text(
                image,
                language_hint=language_hint,
                max_characters=char_limit,
            ),
        )

    def find_visual_element_in_browser_capture(
        self,
        *,
        capture_id: str,
        query: str,
        max_results: int = 3,
    ) -> VisionEvidence:
        limit = max(1, min(max_results, int(self.effective_config().get("vision_max_regions", 8))))
        return self._run_with_capture(
            "find_visual_element_in_browser_capture",
            capture_id=capture_id,
            browser_visual_query=query,
            handler=lambda image: self.provider().find_visual_element(
                image,
                query=query,
                max_results=limit,
                strict_localization=True,
            ),
        )

    def list_windows(self) -> DesktopWindowListResult:
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            return DesktopWindowListResult(success=False, error_reason="Vision runtime is disabled.")
        if not bool(config.get("vision_desktop_capture_enabled", True)):
            return DesktopWindowListResult(success=False, error_reason="Desktop capture is disabled.")
        if not desktop_capture_backend.is_supported():
            return DesktopWindowListResult(success=False, error_reason="Desktop capture is only supported on Windows.")
        try:
            windows = desktop_capture_backend.list_windows()
        except Exception:
            return DesktopWindowListResult(success=False, error_reason="Could not list open windows.")
        return DesktopWindowListResult(
            success=True,
            windows=[{"window_id": window.window_id, "title": window.title} for window in windows],
        )

    def capture_desktop_screen(self) -> DesktopCaptureRecord:
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            raise VisionDisabledError("Vision runtime is disabled.")
        if not bool(config.get("vision_desktop_capture_enabled", True)):
            raise VisionPolicyError("Desktop capture is disabled.")
        owner_request_id, owner_agent_task_id = self._capture_owner_ids()
        raw = desktop_capture_backend.capture_full_desktop()
        return self._store_desktop_capture(
            raw,
            source_type="desktop_screen",
            window_id=None,
            window_title="",
            owner_request_id=owner_request_id,
            owner_agent_task_id=owner_agent_task_id,
            config=config,
        )

    def capture_desktop_window(self, *, window_id: int, window_title: str) -> DesktopCaptureRecord:
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            raise VisionDisabledError("Vision runtime is disabled.")
        if not bool(config.get("vision_desktop_capture_enabled", True)):
            raise VisionPolicyError("Desktop capture is disabled.")
        # Revalidate the target window against what the user actually approved: existence,
        # visibility, AND title. Windows recycles HWND values after a window closes, so
        # checking existence/visibility alone would risk silently capturing whatever new
        # window now happens to hold that handle while still claiming to fulfill the
        # originally-approved request.
        current = desktop_capture_backend.find_window(window_id)
        if current is None:
            raise VisionImageError("The target window is no longer available.")
        expected_title = str(window_title or "").strip()
        if current.title.strip() != expected_title:
            raise VisionPolicyError("The target window changed since this plan was approved.")
        owner_request_id, owner_agent_task_id = self._capture_owner_ids()
        raw = desktop_capture_backend.capture_window(window_id)
        return self._store_desktop_capture(
            raw,
            source_type="desktop_window",
            window_id=window_id,
            window_title=current.title,
            owner_request_id=owner_request_id,
            owner_agent_task_id=owner_agent_task_id,
            config=config,
        )

    def describe_desktop_capture(self, *, capture_id: str, detail_level: str = "normal") -> VisionEvidence:
        if detail_level not in VISION_DETAIL_LEVELS:
            detail_level = "normal"
        return self._run_with_desktop_capture(
            "describe_desktop_capture",
            capture_id=capture_id,
            handler=lambda image: self.provider().describe_image(image, detail_level=detail_level),
        )

    def extract_text_from_desktop_capture(
        self,
        *,
        capture_id: str,
        language_hint: str = "",
        max_characters: int | None = None,
    ) -> VisionEvidence:
        config = self.effective_config()
        char_limit = int(max_characters or config.get("vision_max_ocr_chars", 4000))
        return self._run_with_desktop_capture(
            "extract_text_from_desktop_capture",
            capture_id=capture_id,
            handler=lambda image: self.provider().extract_text(
                image,
                language_hint=language_hint,
                max_characters=char_limit,
            ),
        )

    def find_visual_element_in_desktop_capture(
        self,
        *,
        capture_id: str,
        query: str,
        max_results: int = 3,
    ) -> VisionEvidence:
        limit = max(1, min(max_results, int(self.effective_config().get("vision_max_regions", 8))))
        return self._run_with_desktop_capture(
            "find_visual_element_in_desktop_capture",
            capture_id=capture_id,
            handler=lambda image: self.provider().find_visual_element(image, query=query, max_results=limit),
        )

    def desktop_captures_message(self) -> str:
        removed = self.cleanup_expired_desktop_captures()
        state = get_vision_state()
        with state.lock:
            captures = sorted(state.desktop_captures.values(), key=lambda item: item.captured_at)
        if not captures:
            return "No temporary desktop captures."
        lines = []
        if removed:
            lines.append(f"Expired captures cleaned: {removed}")
        lines.append("Temporary desktop captures:")
        now = datetime.now(timezone.utc)
        for capture in captures[:20]:
            expired = _parse_iso(capture.expires_at) <= now
            lines.append(
                " | ".join(
                    [
                        _display_capture_id(capture.capture_id),
                        capture.source_type,
                        capture.window_title or "(full desktop)",
                        f"age={_age_seconds(capture.captured_at, now)}s",
                        "expired" if expired else "active",
                    ]
                )
            )
        return "\n".join(lines)

    def clear_desktop_captures(self) -> str:
        removed = self._remove_desktop_captures(lambda _capture: True)
        return f"Cleared {removed} temporary desktop capture{'s' if removed != 1 else ''}."

    def cleanup_expired_evidence(self) -> int:
        state = get_vision_state()
        removed = 0
        now = datetime.now(timezone.utc)
        with state.lock:
            for evidence_id, evidence in list(state.evidences.items()):
                try:
                    expires_at = datetime.fromisoformat(evidence.expires_at)
                except ValueError:
                    del state.evidences[evidence_id]
                    removed += 1
                    continue
                if expires_at <= now:
                    del state.evidences[evidence_id]
                    removed += 1
        return removed

    def cleanup_expired_captures(self) -> int:
        now = datetime.now(timezone.utc)
        return self._remove_captures(lambda capture: _parse_iso(capture.expires_at) <= now)

    def cleanup_captures_for_session(self, session_id: str) -> int:
        return self._remove_captures(lambda capture: capture.session_id == session_id)

    def cleanup_captures_for_task(self, *, owner_request_id: int | None = None, owner_agent_task_id: int | None = None) -> int:
        return self._remove_captures(
            lambda capture: (
                owner_agent_task_id is not None
                and capture.owner_agent_task_id == owner_agent_task_id
            )
            or (
                owner_request_id is not None
                and capture.owner_request_id == owner_request_id
            )
        )

    def cleanup_expired_desktop_captures(self) -> int:
        now = datetime.now(timezone.utc)
        return self._remove_desktop_captures(lambda capture: _parse_iso(capture.expires_at) <= now)

    def cleanup_desktop_captures_for_task(self, *, owner_request_id: int | None = None, owner_agent_task_id: int | None = None) -> int:
        return self._remove_desktop_captures(
            lambda capture: (
                owner_agent_task_id is not None
                and capture.owner_agent_task_id == owner_agent_task_id
            )
            or (
                owner_request_id is not None
                and capture.owner_request_id == owner_request_id
            )
        )

    def get_current_evidence(self, evidence_id: str) -> VisionEvidence:
        self.cleanup_expired_evidence()
        state = get_vision_state()
        with state.lock:
            evidence = state.evidences.get(evidence_id)
        if evidence is None:
            raise VisionEvidenceExpiredError("Vision evidence is no longer available.")
        return evidence

    def _run_with_image(self, operation: str, *, path: str, handler) -> VisionEvidence:
        self.cleanup_expired_evidence()
        self.cleanup_expired_captures()
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            return self._failure(operation, error_category="disabled", error_reason="Vision runtime is disabled.")
        image = None
        try:
            image = load_local_image(path)
            evidence = handler(image)
            self._store_evidence(evidence)
            record_audit_event(f"vision_{operation}", message=image.safe_display_name)
            return evidence
        except VisionDisabledError as error:
            return self._failure(operation, error_category="disabled", error_reason=str(error))
        except VisionPolicyError as error:
            return self._failure(operation, error_category="policy", error_reason=str(error))
        except VisionEvidenceExpiredError as error:
            return self._failure(operation, error_category="expired", error_reason=str(error))
        except VisionImageError as error:
            return self._failure(operation, error_category="image_error", error_reason=str(error))
        except VisionProviderUnavailableError as error:
            return self._failure(operation, error_category="provider_unavailable", error_reason=str(error))
        except VisionProviderError as error:
            return self._failure(operation, error_category="provider_error", error_reason=str(error))
        finally:
            cleanup_loaded_image(image)

    def _run_with_capture(self, operation: str, *, capture_id: str, handler, browser_visual_query: str = "") -> VisionEvidence:
        self.cleanup_expired_evidence()
        self.cleanup_expired_captures()
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            return self._failure(operation, error_category="disabled", error_reason="Vision runtime is disabled.")
        try:
            capture = self._require_browser_capture(capture_id)
            image = self._loaded_image_from_capture(capture)
            evidence = handler(image)
            evidence = self._bind_capture_evidence(evidence, capture)
            if operation == "find_visual_element_in_browser_capture":
                evidence = self._verify_browser_capture_visual_match(evidence, capture, query=browser_visual_query)
            self._store_evidence(evidence)
            record_audit_event(
                f"vision_{operation}",
                task_id=capture.owner_agent_task_id,
                message=f"{capture.capture_id}:{capture.origin or '(no origin)'}",
            )
            return evidence
        except VisionDisabledError as error:
            return self._failure(operation, error_category="disabled", error_reason=str(error))
        except VisionEvidenceExpiredError as error:
            return self._failure(operation, error_category="expired", error_reason=str(error))
        except VisionPolicyError as error:
            return self._failure(operation, error_category="policy", error_reason=str(error))
        except VisionImageError as error:
            return self._failure(operation, error_category="image_error", error_reason=str(error))
        except VisionProviderUnavailableError as error:
            return self._failure(operation, error_category="provider_unavailable", error_reason=str(error))
        except VisionProviderError as error:
            return self._failure(operation, error_category="provider_error", error_reason=str(error))

    def _store_evidence(self, evidence: VisionEvidence) -> None:
        state = get_vision_state()
        with state.lock:
            state.evidences[evidence.evidence_id] = evidence
            state.last_safe_status = evidence.operation

    def _perform_generation_check(self) -> tuple[bool | None, str]:
        provider = self.provider()
        if not hasattr(provider, "check_generation_ready"):
            return None, "Generation readiness check is unavailable."
        started = time.monotonic()
        ready, detail = provider.check_generation_ready()  # type: ignore[attr-defined]
        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        self._store_readiness_cache(ready, detail, elapsed_ms=elapsed_ms)
        return ready, detail

    def _apply_cached_readiness(self, status: VisionProviderStatus) -> VisionProviderStatus:
        state = get_vision_state()
        config_state = get_runtime_config_state()
        signature = self._readiness_signature()
        with state.lock:
            cache_valid = (
                state.readiness_signature == signature
                and state.readiness_generation == config_state.generation
                and state.generation_ready is not None
            )
            if cache_valid:
                status.generation_ready = state.generation_ready
                status.generation_detail = state.generation_detail
                return status
            if status.configured and status.base_url_allowed and status.provider_reachable and status.model_installed and status.image_capability_ready:
                status.generation_ready = None
                status.generation_detail = "Generation readiness has not been checked yet."
                return status
            state.readiness_signature = signature
            state.readiness_generation = config_state.generation
            state.generation_ready = status.generation_ready
            state.generation_detail = status.generation_detail
            state.generation_checked_at = ""
            state.generation_elapsed_ms = None
        return status

    def _store_readiness_cache(self, ready: bool | None, detail: str, *, elapsed_ms: int | None) -> None:
        state = get_vision_state()
        config_state = get_runtime_config_state()
        with state.lock:
            state.readiness_signature = self._readiness_signature()
            state.readiness_generation = config_state.generation
            state.generation_ready = ready
            state.generation_detail = detail[:220]
            state.generation_checked_at = datetime.now(timezone.utc).isoformat()
            state.generation_elapsed_ms = elapsed_ms

    def _readiness_signature(self) -> str:
        provider = self.provider()
        return "|".join(
            [
                getattr(provider, "name", "unknown"),
                str(getattr(provider, "base_url", "")),
                str(getattr(provider, "model", "")),
                str(getattr(provider, "timeout", "")),
            ]
        )

    def _format_readiness(self, ready: bool | None) -> str:
        if ready is None:
            return "unknown"
        return "yes" if ready else "no"

    def _next_capture_id(self) -> str:
        state = get_vision_state()
        with state.lock:
            capture_id = f"capture-{state.next_capture_id}-{uuid4().hex[:8]}"
            state.next_capture_id += 1
            return capture_id

    def _remove_captures(self, predicate) -> int:
        state = get_vision_state()
        removed = 0
        with state.lock:
            for capture_id, capture in list(state.captures.items()):
                if not predicate(capture):
                    continue
                del state.captures[capture_id]
                removed += 1
        return removed

    def _capture_owner_ids(self) -> tuple[int | None, int | None]:
        owner_request_id: int | None = None
        try:
            from app.brain.intelligence.controller import get_intelligence_controller

            request = get_intelligence_controller().state.current_request
            if request is not None:
                owner_request_id = request.request_id
        except Exception:
            owner_request_id = None
        runtime_state = get_agent_runtime_state()
        owner_agent_task_id = runtime_state.current_task.task_id if runtime_state.current_task is not None else None
        return owner_request_id, owner_agent_task_id

    def _next_desktop_capture_id(self) -> str:
        state = get_vision_state()
        with state.lock:
            capture_id = f"desktop-capture-{state.next_desktop_capture_id}-{uuid4().hex[:8]}"
            state.next_desktop_capture_id += 1
            return capture_id

    def _remove_desktop_captures(self, predicate) -> int:
        state = get_vision_state()
        removed = 0
        with state.lock:
            for capture_id, capture in list(state.desktop_captures.items()):
                if not predicate(capture):
                    continue
                del state.desktop_captures[capture_id]
                removed += 1
        return removed

    def _store_desktop_capture(
        self,
        raw: desktop_capture_backend.RawCapture,
        *,
        source_type: str,
        window_id: int | None,
        window_title: str,
        owner_request_id: int | None,
        owner_agent_task_id: int | None,
        config: dict[str, Any],
    ) -> DesktopCaptureRecord:
        max_bytes = int(config.get("vision_desktop_capture_max_bytes", 6_000_000))
        max_width = int(config.get("vision_desktop_capture_max_width", 3840))
        max_height = int(config.get("vision_desktop_capture_max_height", 2160))
        max_pixels = int(config.get("vision_desktop_capture_max_pixels", 8_294_400))
        if not isinstance(raw.image_bytes, (bytes, bytearray)) or not raw.image_bytes:
            raise VisionImageError("Desktop capture data is invalid.")
        if len(raw.image_bytes) > max_bytes:
            raise VisionImageError("Desktop capture exceeds the configured size limit.")
        if raw.width <= 0 or raw.height <= 0:
            raise VisionImageError("Desktop capture dimensions are invalid.")
        if raw.width > max_width or raw.height > max_height or raw.width * raw.height > max_pixels:
            raise VisionImageError("Desktop capture exceeds the configured dimension limits.")
        now = datetime.now(timezone.utc)
        capture_id = self._next_desktop_capture_id()
        digest = hashlib.sha256(bytes(raw.image_bytes)).hexdigest()
        record = DesktopCaptureRecord(
            capture_id=capture_id,
            owner_request_id=owner_request_id,
            owner_agent_task_id=owner_agent_task_id,
            source_type=source_type,
            window_id=window_id,
            window_title=window_title,
            width=int(raw.width),
            height=int(raw.height),
            captured_at=now.isoformat(),
            expires_at=(now + _desktop_capture_ttl(config)).isoformat(),
            image_format="png",
            source_hash=digest,
            byte_size=len(raw.image_bytes),
            mime_type=raw.mime_type,
            safe_display_name=f"desktop-capture-{capture_id[-8:]}.png",
            image_bytes=bytes(raw.image_bytes),
        )
        state = get_vision_state()
        with state.lock:
            state.desktop_captures[capture_id] = record
            state.last_safe_status = "desktop_capture_ready"
        record_audit_event(
            "vision_desktop_capture_created",
            task_id=owner_agent_task_id,
            message=f"{capture_id}:{source_type}",
        )
        return record

    def _require_desktop_capture(self, capture_id: str) -> DesktopCaptureRecord:
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise VisionPolicyError("Desktop capture ID is required.")
        state = get_vision_state()
        with state.lock:
            capture = state.desktop_captures.get(capture_id.strip())
        if capture is None:
            raise VisionEvidenceExpiredError("Desktop capture is no longer available.")
        now = datetime.now(timezone.utc)
        if _parse_iso(capture.expires_at) <= now:
            self.cleanup_expired_desktop_captures()
            raise VisionEvidenceExpiredError("Desktop capture is no longer available.")
        runtime_state = get_agent_runtime_state()
        active_task = runtime_state.current_task
        if capture.owner_agent_task_id is not None:
            active_or_recent = []
            if active_task is not None:
                active_or_recent.append(active_task.task_id)
            active_or_recent.extend(task.task_id for task in runtime_state.archived_tasks[-5:])
            if capture.owner_agent_task_id not in active_or_recent:
                raise VisionPolicyError("Desktop capture belongs to a different task.")
        return capture

    def _loaded_image_from_desktop_capture(self, capture: DesktopCaptureRecord) -> LoadedVisionImage:
        frame = VisionFrame(
            frame_id=f"frame-{capture.capture_id}",
            source_type=capture.source_type,
            safe_display_name=capture.safe_display_name,
            source_hash=capture.source_hash,
            mime_type=capture.mime_type,
            width=capture.width,
            height=capture.height,
            created_at=capture.captured_at,
            expires_at=capture.expires_at,
            temporary_copy=False,
            trust_classification="desktop_capture",
            file_size_bytes=capture.byte_size,
        )
        return LoadedVisionImage(
            frame=frame,
            original_path="",
            safe_display_name=capture.safe_display_name,
            mime_type=capture.mime_type,
            width=capture.width,
            height=capture.height,
            source_hash=capture.source_hash,
            file_size_bytes=capture.byte_size,
            image_bytes=bytes(capture.image_bytes),
            temp_copy_path="",
        )

    def _run_with_desktop_capture(self, operation: str, *, capture_id: str, handler) -> VisionEvidence:
        self.cleanup_expired_evidence()
        self.cleanup_expired_desktop_captures()
        config = self.effective_config()
        if not bool(config.get("vision_enabled", True)):
            return self._failure(operation, error_category="disabled", error_reason="Vision runtime is disabled.")
        try:
            capture = self._require_desktop_capture(capture_id)
            image = self._loaded_image_from_desktop_capture(capture)
            evidence = handler(image)
            evidence = replace(evidence, instruction_trust="none", factual_credibility="unevaluated")
            self._store_evidence(evidence)
            record_audit_event(
                f"vision_{operation}",
                task_id=capture.owner_agent_task_id,
                message=capture.capture_id,
            )
            return evidence
        except VisionDisabledError as error:
            return self._failure(operation, error_category="disabled", error_reason=str(error))
        except VisionEvidenceExpiredError as error:
            return self._failure(operation, error_category="expired", error_reason=str(error))
        except VisionPolicyError as error:
            return self._failure(operation, error_category="policy", error_reason=str(error))
        except VisionImageError as error:
            return self._failure(operation, error_category="image_error", error_reason=str(error))
        except VisionProviderUnavailableError as error:
            return self._failure(operation, error_category="provider_unavailable", error_reason=str(error))
        except VisionProviderError as error:
            return self._failure(operation, error_category="provider_error", error_reason=str(error))

    def _require_browser_capture(self, capture_id: str) -> BrowserCaptureRecord:
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise VisionPolicyError("Browser capture ID is required.")
        state = get_vision_state()
        with state.lock:
            capture = state.captures.get(capture_id.strip())
        if capture is None:
            raise VisionEvidenceExpiredError("Browser capture is no longer available.")
        now = datetime.now(timezone.utc)
        if _parse_iso(capture.expires_at) <= now:
            self.cleanup_expired_captures()
            raise VisionEvidenceExpiredError("Browser capture is no longer available.")
        runtime_state = get_agent_runtime_state()
        active_task = runtime_state.current_task
        if capture.owner_agent_task_id is not None:
            active_or_recent = []
            if active_task is not None:
                active_or_recent.append(active_task.task_id)
            active_or_recent.extend(task.task_id for task in runtime_state.archived_tasks[-5:])
            if capture.owner_agent_task_id not in active_or_recent:
                raise VisionPolicyError("Browser capture belongs to a different task.")
        from app.brain.browser.controller import get_browser_controller

        freshness_error = get_browser_controller().validate_capture_freshness(
            session_id=capture.session_id,
            tab_id=capture.tab_id,
            url=capture.url,
            origin=capture.origin,
            page_version=capture.page_version,
        )
        if freshness_error:
            raise VisionPolicyError(freshness_error)
        return capture

    def _loaded_image_from_capture(self, capture: BrowserCaptureRecord) -> LoadedVisionImage:
        frame = VisionFrame(
            frame_id=f"frame-{capture.capture_id}",
            source_type="browser_viewport",
            safe_display_name=capture.safe_display_name,
            source_hash=capture.source_hash,
            mime_type=capture.mime_type,
            width=_capture_image_width(capture),
            height=_capture_image_height(capture),
            created_at=capture.captured_at,
            expires_at=capture.expires_at,
            temporary_copy=False,
            trust_classification="browser_viewport_capture",
            file_size_bytes=capture.byte_size,
        )
        return LoadedVisionImage(
            frame=frame,
            original_path="",
            safe_display_name=capture.safe_display_name,
            mime_type=capture.mime_type,
            width=_capture_image_width(capture),
            height=_capture_image_height(capture),
            source_hash=capture.source_hash,
            file_size_bytes=capture.byte_size,
            image_bytes=bytes(capture.image_bytes),
            temp_copy_path="",
        )

    def _bind_capture_evidence(self, evidence: VisionEvidence, capture: BrowserCaptureRecord) -> VisionEvidence:
        warnings = list(evidence.warnings)
        captcha_suspected, captcha_reason = _detect_captcha(evidence)
        if captcha_suspected and captcha_reason:
            warnings.append(captcha_reason)
        return replace(
            evidence,
            source_url=capture.url,
            source_origin=capture.origin,
            instruction_trust="none",
            factual_credibility="unevaluated",
            captcha_suspected=captcha_suspected,
            captcha_reason=captcha_reason,
            warnings=warnings[:8],
        )

    def _verify_browser_capture_visual_match(self, evidence: VisionEvidence, capture: BrowserCaptureRecord, *, query: str) -> VisionEvidence:
        normalized_query = normalize_vision_query(query)
        diagnostics = _initial_browser_visual_grounding_diagnostics(
            capture=capture,
            normalized_query=normalized_query,
            provider_diagnostics=evidence.grounding_diagnostics,
            locator_candidate_count=len(evidence.visual_regions),
        )
        dom_grounded_region = _dom_grounded_region_for_query(capture, normalized_query)
        diagnostics["dom_grounding_attempted"] = True
        if _query_prefers_dom_grounding(normalized_query) and dom_grounded_region is not None:
            diagnostics["dom_grounding_result"] = "matched"
            _mark_dom_grounding_diagnostics_verified(diagnostics, normalized_query=normalized_query)
            return replace(
                evidence,
                success=True,
                grounded=True,
                no_match=False,
                match_outcome="found",
                error_category="",
                error_reason="",
                confidence=dom_grounded_region.confidence,
                visual_regions=[dom_grounded_region],
                grounding_diagnostics=diagnostics,
            )
        if (evidence.no_match or not evidence.visual_regions) and dom_grounded_region is not None:
            diagnostics["dom_grounding_result"] = "matched"
            _mark_dom_grounding_diagnostics_verified(diagnostics, normalized_query=normalized_query)
            return replace(
                evidence,
                success=True,
                grounded=True,
                no_match=False,
                match_outcome="found",
                error_category="",
                error_reason="",
                confidence=dom_grounded_region.confidence,
                visual_regions=[dom_grounded_region],
                grounding_diagnostics=diagnostics,
            )
        if evidence.no_match or not evidence.visual_regions:
            diagnostics["dom_grounding_result"] = "no_match"
            diagnostics["final_grounding_state"] = "no_candidate"
            diagnostics["final_verification_type"] = ""
            return replace(
                evidence,
                success=True,
                grounded=True,
                no_match=True,
                match_outcome="not_found",
                error_category="",
                error_reason="",
                confidence=0.0,
                visual_regions=[],
                grounding_diagnostics=diagnostics,
            )
        verified_regions: list[VisionRegion] = []
        uncertain_regions: list[VisionRegion] = []
        candidate_detected = False
        warnings = list(evidence.warnings)
        disagreement_detected = False
        selected_diagnostics: dict[str, Any] | None = None
        for region in evidence.visual_regions:
            candidate_detected = True
            classification, verified_region, reason, candidate_diagnostics = self._classify_browser_capture_candidate(
                region,
                capture=capture,
                normalized_query=normalized_query,
            )
            if selected_diagnostics is None:
                selected_diagnostics = candidate_diagnostics
            if classification == "found" and verified_region is not None:
                selected_diagnostics = candidate_diagnostics
                verified_regions.append(verified_region)
                continue
            if classification == "uncertain" and verified_region is not None:
                uncertain_regions.append(verified_region)
                if reason:
                    warnings.append(reason)
                if verified_region.verification == "dom_disagreement":
                    disagreement_detected = True
        if selected_diagnostics:
            diagnostics.update(selected_diagnostics)
        if not verified_regions and dom_grounded_region is not None and not disagreement_detected:
            diagnostics["dom_grounding_result"] = "matched"
            _mark_dom_grounding_diagnostics_verified(diagnostics, normalized_query=normalized_query)
            verified_regions.append(dom_grounded_region)
        elif disagreement_detected:
            diagnostics["dom_grounding_result"] = "disagreement"
        else:
            diagnostics["dom_grounding_result"] = "no_match"
        if not verified_regions:
            fallback_region, fallback_reason, fallback_diagnostics = self._verify_browser_capture_with_full_frame_observer(
                capture=capture,
                normalized_query=normalized_query,
                dom_grounded_region=dom_grounded_region,
            )
            diagnostics.update(fallback_diagnostics)
            if fallback_region is not None:
                verified_regions.append(fallback_region)
            elif fallback_reason:
                warnings.append(fallback_reason)
        if verified_regions:
            diagnostics["final_grounding_state"] = str(verified_regions[0].verification or "verified")
            diagnostics["final_verification_type"] = str(verified_regions[0].verification or "")
            if not _browser_visual_positive_verification_is_consistent(verified_regions[0], diagnostics):
                diagnostics["final_grounding_state"] = "candidate_unverified"
                diagnostics["final_verification_type"] = ""
                return replace(
                    evidence,
                    success=True,
                    grounded=True,
                    no_match=False,
                    match_outcome="uncertain",
                    error_category="",
                    error_reason="Potential browser visual matches could not be verified.",
                    confidence=0.0,
                    visual_regions=[],
                    warnings=(warnings + ["Internal grounding evidence was inconsistent and was rejected safely."])[:8],
                    grounding_diagnostics=diagnostics,
                )
            return replace(
                evidence,
                success=True,
                grounded=True,
                no_match=False,
                match_outcome="found",
                error_category="",
                error_reason="",
                confidence=max((region.confidence for region in verified_regions), default=0.0),
                visual_regions=verified_regions,
                warnings=warnings[:8],
                grounding_diagnostics=diagnostics,
            )
        if uncertain_regions:
            if disagreement_detected:
                warnings.append("Vision and DOM evidence disagree about the requested element.")
            diagnostics["final_grounding_state"] = str(uncertain_regions[0].verification or "candidate_unverified")
            diagnostics["final_verification_type"] = ""
            return replace(
                evidence,
                success=True,
                grounded=True,
                no_match=False,
                match_outcome="uncertain",
                error_category="",
                error_reason="Potential browser visual matches could not be verified.",
                visual_regions=uncertain_regions,
                warnings=warnings[:8],
                grounding_diagnostics=diagnostics,
            )
        if candidate_detected:
            warnings.append("Candidate region was not supported by observed crop evidence.")
            diagnostics["final_grounding_state"] = str(diagnostics.get("final_grounding_state") or "candidate_unverified")
            diagnostics["final_verification_type"] = ""
            return replace(
                evidence,
                success=True,
                grounded=True,
                no_match=False,
                match_outcome="uncertain",
                error_category="",
                error_reason="Potential browser visual matches could not be verified.",
                confidence=0.0,
                visual_regions=[],
                warnings=warnings[:8],
                grounding_diagnostics=diagnostics,
            )
        diagnostics["final_grounding_state"] = "no_candidate"
        diagnostics["final_verification_type"] = ""
        return replace(
            evidence,
            success=True,
            grounded=True,
            no_match=True,
            match_outcome="not_found",
            error_category="",
            error_reason="",
            confidence=0.0,
            visual_regions=[],
            warnings=warnings[:8],
            grounding_diagnostics=diagnostics,
        )

    def _classify_browser_capture_candidate(
        self,
        region: VisionRegion,
        *,
        capture: BrowserCaptureRecord,
        normalized_query: str,
    ) -> tuple[str, VisionRegion | None, str, dict[str, Any]]:
        diagnostics = _base_candidate_grounding_diagnostics(region=region, capture=capture)
        bbox = region.bounding_box
        if bbox is None or not _valid_region_bbox(bbox):
            diagnostics["geometry_validation_result"] = "rejected"
            diagnostics["geometry_rejection_reason"] = "Potential match lacked a valid visual region."
            diagnostics["final_grounding_state"] = "candidate_rejected"
            return "uncertain", replace(region, verification="candidate_rejected"), "Potential match lacked a valid visual region.", diagnostics
        dom_match = _best_dom_match_for_region(capture.dom_elements, bbox, capture=capture)
        if dom_match is not None:
            correlated = _region_from_dom_match(region, dom_match, capture=capture)
            if _dom_match_supports_query(dom_match, normalized_query):
                diagnostics["geometry_validation_result"] = "accepted"
                diagnostics["geometry_rejection_reason"] = ""
                diagnostics["dom_grounding_result"] = "matched"
                _mark_dom_grounding_diagnostics_verified(diagnostics, normalized_query=normalized_query, preserve_attempts=False)
                return "found", correlated, "", diagnostics
            diagnostics["geometry_validation_result"] = "accepted"
            diagnostics["geometry_rejection_reason"] = ""
            diagnostics["dom_grounding_result"] = "disagreement"
            diagnostics["final_grounding_state"] = "dom_disagreement"
            return "uncertain", replace(correlated, verification="dom_disagreement"), "Potential match disagreed with DOM evidence.", diagnostics
        plausibility_reason = _candidate_bbox_plausibility_reason(
            bbox,
            capture=capture,
            config=self.effective_config(),
            normalized_query=normalized_query,
        )
        if plausibility_reason:
            diagnostics["geometry_validation_result"] = "rejected"
            diagnostics["geometry_rejection_reason"] = plausibility_reason
            diagnostics["final_grounding_state"] = "candidate_rejected"
            return "uncertain", replace(region, verification="candidate_rejected"), plausibility_reason, diagnostics
        diagnostics["geometry_validation_result"] = "accepted"
        diagnostics["geometry_rejection_reason"] = ""
        diagnostics["dom_grounding_result"] = "no_match"
        crop_region, crop_reason, crop_diagnostics = self._verify_candidate_region_with_crop(
            region,
            capture=capture,
            normalized_query=normalized_query,
        )
        diagnostics.update(crop_diagnostics)
        if crop_region is not None:
            diagnostics["final_grounding_state"] = "crop_verified"
            diagnostics["final_verification_type"] = "crop_verified"
            return "found", crop_region, "", diagnostics
        diagnostics["final_grounding_state"] = str(diagnostics.get("final_grounding_state") or "candidate_unverified")
        return "uncertain", replace(region, verification="candidate_unverified"), crop_reason or "Candidate region was not supported by observed crop evidence.", diagnostics

    def _verify_candidate_region_with_crop(
        self,
        region: VisionRegion,
        *,
        capture: BrowserCaptureRecord,
        normalized_query: str,
    ) -> tuple[VisionRegion | None, str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "crop_verification_attempted": False,
            "crop_context_padding_pixels": "",
            "crop_pixel_dimensions": "",
            "crop_observer_outcome": "",
            "source_capture_identity_match": "",
            "background_only": "",
            "object_fully_visible": "",
            "canonical_dominant_colors": [],
            "canonical_observed_shapes": [],
            "canonical_object_categories": [],
            "required_target_properties": [],
            "matched_target_properties": [],
            "missing_target_properties": [],
        }
        if region.bounding_box is None:
            return None, "Potential match lacked a valid visual region.", diagnostics
        context_bbox = _expand_verification_context_bbox(
            region.bounding_box,
            capture=capture,
            config=self.effective_config(),
        )
        diagnostics["verification_context_normalized_box"] = context_bbox.to_dict()
        context_left, context_top, context_right, context_bottom = _bbox_to_capture_image_pixel_rect(context_bbox, capture)
        crop_width = max(1, context_right - context_left)
        crop_height = max(1, context_bottom - context_top)
        diagnostics["verification_context_pixel_box"] = _pixel_box_dict(context_left, context_top, context_right, context_bottom)
        diagnostics["crop_pixel_dimensions"] = f"{crop_width}x{crop_height}"
        candidate_left, candidate_top, candidate_right, candidate_bottom = _bbox_to_capture_image_pixel_rect(region.bounding_box, capture)
        diagnostics["crop_context_padding_pixels"] = (
            f"left={max(0, candidate_left - context_left)}, top={max(0, candidate_top - context_top)}, "
            f"right={max(0, context_right - candidate_right)}, bottom={max(0, context_bottom - candidate_bottom)}"
        )
        diagnostics["crop_verification_attempted"] = True
        crop_image: LoadedVisionImage | None = None
        try:
            base_image = self._loaded_image_from_capture(capture)
            crop_image = crop_loaded_image(base_image, context_bbox, safe_display_name="browser-crop.png")
            diagnostics["source_capture_identity_match"] = "yes" if base_image.source_hash == capture.source_hash else "no"
            diagnostics.update(_bounded_pixel_statistics(crop_image))
            observation = self.provider().verify_visual_crop(crop_image)
        except (VisionImageError, VisionProviderError, VisionProviderUnavailableError) as error:
            diagnostics["crop_observer_outcome"] = "error"
            diagnostics["final_grounding_state"] = "candidate_unverified"
            return None, f"Candidate region was not supported by observed crop evidence. ({str(error)[:120]})", diagnostics
        finally:
            cleanup_loaded_image(crop_image)
        observer_region, match_diagnostics, reason = _verify_observed_crop_objects(
            observation,
            normalized_query=normalized_query,
            capture=capture,
            context_bbox=context_bbox,
            crop_width=crop_width,
            crop_height=crop_height,
            viewport_width=_capture_image_width(capture),
            viewport_height=_capture_image_height(capture),
        )
        diagnostics.update(match_diagnostics)
        if observer_region is None:
            diagnostics["final_grounding_state"] = str(diagnostics.get("final_grounding_state") or "candidate_unverified")
            return None, reason, diagnostics
        merged_attributes = dict(region.attributes)
        merged_attributes.update(observer_region.attributes)
        merged_attributes.setdefault("relative_location", _relative_location(observer_region.bounding_box))
        visible_text = observer_region.visible_text.strip()
        label = str(observer_region.label or "").strip() or str(region.label or "").strip() or "match"
        return (
            replace(
                observer_region,
                label=label,
                visible_text=visible_text,
                verification="crop_verified",
                attributes=merged_attributes,
            ),
            "",
            diagnostics,
        )

    def _verify_browser_capture_with_full_frame_observer(
        self,
        *,
        capture: BrowserCaptureRecord,
        normalized_query: str,
        dom_grounded_region: VisionRegion | None,
    ) -> tuple[VisionRegion | None, str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "full_frame_verification_attempted": False,
            "full_frame_observer_outcome": "",
        }
        if dom_grounded_region is not None or _query_prefers_dom_grounding(normalized_query):
            return None, "", diagnostics
        requirements = _parse_visual_target_requirements(normalized_query)
        if not requirements["colors"] and not requirements["shapes"] and not requirements["categories"]:
            return None, "", diagnostics
        diagnostics["full_frame_verification_attempted"] = True
        base_image: LoadedVisionImage | None = None
        try:
            base_image = self._loaded_image_from_capture(capture)
            diagnostics.update(_bounded_pixel_statistics(base_image, prefix="full_frame_"))
            observation = self.provider().observe_full_frame(base_image)
        except (VisionImageError, VisionProviderError, VisionProviderUnavailableError) as error:
            diagnostics["full_frame_observer_outcome"] = "error"
            return None, f"Candidate region was not supported by observed crop evidence. ({str(error)[:120]})", diagnostics
        finally:
            cleanup_loaded_image(base_image)
        full_frame_bbox = VisionBoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        observer_region, match_diagnostics, reason = _verify_observed_crop_objects(
            observation,
            normalized_query=normalized_query,
            capture=capture,
            context_bbox=full_frame_bbox,
            crop_width=_capture_image_width(capture),
            crop_height=_capture_image_height(capture),
            viewport_width=_capture_image_width(capture),
            viewport_height=_capture_image_height(capture),
        )
        for key in (
            "background_only",
            "object_fully_visible",
            "canonical_dominant_colors",
            "canonical_observed_shapes",
            "canonical_object_categories",
            "matched_target_properties",
            "missing_target_properties",
            "required_target_properties",
        ):
            if key in match_diagnostics:
                diagnostics[key] = match_diagnostics[key]
        if observer_region is None:
            diagnostics["full_frame_observer_outcome"] = str(match_diagnostics.get("crop_observer_outcome") or "object_mismatch")
            return None, reason, diagnostics
        diagnostics["full_frame_observer_outcome"] = "matched"
        attributes = dict(observer_region.attributes)
        attributes.pop("verification_context", None)
        return replace(observer_region, verification="pixel_verified", attributes=attributes), "", diagnostics

    def _failure(self, operation: str, *, error_category: str, error_reason: str) -> VisionEvidence:
        now = datetime.now(timezone.utc).isoformat()
        return VisionEvidence(
            evidence_id=f"vision-error-{operation}",
            frame_id="",
            operation=operation,
            success=False,
            provider=self.provider().name if self._provider_override is not None else "ollama",
            model=getattr(self.provider(), "model", ""),
            source_hash="",
            grounded=False,
            created_at=now,
            expires_at=now,
            confidence=0.0,
            warnings=[],
            error_category=error_category,
            error_reason=error_reason[:160],
            instruction_trust="none",
            factual_credibility="unevaluated",
        )


_CONTROLLER = VisionController()


def get_vision_controller() -> VisionController:
    return _CONTROLLER


def reset_vision_controller(*, provider: VisionProvider | None = None) -> VisionController:
    global _CONTROLLER
    _CONTROLLER = VisionController(provider=provider)
    return _CONTROLLER


def _capture_ttl(config: dict[str, Any]):
    from datetime import timedelta

    return timedelta(seconds=int(config.get("vision_browser_capture_ttl_seconds", 180)))


def _desktop_capture_ttl(config: dict[str, Any]):
    from datetime import timedelta

    return timedelta(seconds=int(config.get("vision_desktop_capture_ttl_seconds", 180)))


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _display_capture_id(capture_id: str) -> str:
    value = str(capture_id or "").strip()
    if len(value) <= 18:
        return value or "(unknown capture)"
    return value[:18] + "..."


def _age_seconds(timestamp: str, now: datetime) -> int:
    created = _parse_iso(timestamp)
    delta = now - created
    return max(0, int(delta.total_seconds()))


def _detect_captcha(evidence: VisionEvidence) -> tuple[bool, str]:
    observed = " ".join(
        part
        for part in (
            evidence.description,
            evidence.extracted_text,
            " ".join(block.text for block in evidence.ocr_blocks),
            " ".join(region.label for region in evidence.visual_regions),
        )
        if isinstance(part, str) and part.strip()
    ).lower()
    if "captcha" not in observed:
        return False, ""
    return True, "CAPTCHA may be present. Complete it manually, then tell me to continue."


def _candidate_bbox_plausibility_reason(
    bbox: VisionBoundingBox,
    *,
    capture: BrowserCaptureRecord,
    config: dict[str, Any],
    normalized_query: str,
) -> str:
    pixel_width = bbox.width * max(1, _capture_image_width(capture))
    pixel_height = bbox.height * max(1, _capture_image_height(capture))
    pixel_area = pixel_width * pixel_height
    min_width = int(config.get("vision_browser_capture_min_candidate_width_pixels", 12))
    min_height = int(config.get("vision_browser_capture_min_candidate_height_pixels", 12))
    min_area = int(config.get("vision_browser_capture_min_candidate_area_pixels", 144))
    if pixel_width < min_width or pixel_height < min_height or pixel_area < min_area:
        return "Candidate region was too small to verify safely."
    requirements = _parse_visual_target_requirements(normalized_query)
    if "circle" in requirements.get("shapes", set()):
        aspect_ratio = pixel_width / max(pixel_height, 1.0)
        if aspect_ratio < 0.65 or aspect_ratio > 1.35:
            return "Candidate region aspect ratio was not plausible for a circle."
    return ""


def _initial_browser_visual_grounding_diagnostics(
    *,
    capture: BrowserCaptureRecord,
    normalized_query: str,
    provider_diagnostics: dict[str, Any],
    locator_candidate_count: int,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "dom_grounding_attempted": False,
        "dom_grounding_result": "not_attempted",
        "locator_outcome": str(provider_diagnostics.get("locator_outcome") or "").strip() or "unknown",
        "locator_candidates": max(0, int(provider_diagnostics.get("locator_candidate_count") or locator_candidate_count or 0)),
        "locator_coordinate_frame": "screenshot_image",
        "candidate_coordinate_space": "",
        "candidate_normalized_box": "",
        "candidate_pixel_box": "",
        "candidate_css_pixel_box": "",
        "viewport_dimensions": f"{int(capture.viewport_width)}x{int(capture.viewport_height)}",
        "viewport_css_dimensions": f"{int(capture.viewport_width)}x{int(capture.viewport_height)}",
        "visual_viewport_dimensions": f"{int(round(_capture_visual_viewport_width(capture)))}x{int(round(_capture_visual_viewport_height(capture)))}",
        "visual_viewport_offsets": f"x={round(float(capture.visual_viewport_offset_left or 0.0), 3)}, y={round(float(capture.visual_viewport_offset_top or 0.0), 3)}",
        "page_scroll": f"x={round(float(capture.scroll_x or 0.0), 3)}, y={round(float(capture.scroll_y or 0.0), 3)}",
        "device_pixel_ratio": round(float(capture.device_pixel_ratio or 0.0), 4),
        "device_scale_factor": round(float(capture.device_scale_factor or 0.0), 4),
        "decoded_screenshot_dimensions": f"{_capture_image_width(capture)}x{_capture_image_height(capture)}",
        "screenshot_scale_option": str(capture.screenshot_scale_option or "").strip() or "device",
        "viewport_only": "yes" if capture.viewport_only else "no",
        "scale_x": round(_capture_scale_x(capture), 6),
        "scale_y": round(_capture_scale_y(capture), 6),
        "geometry_validation_result": "",
        "geometry_rejection_reason": "",
        "crop_verification_attempted": False,
        "crop_pixel_dimensions": "",
        "crop_context_padding_pixels": "",
        "verification_context_normalized_box": "",
        "verification_context_pixel_box": "",
        "actual_crop_dimensions": "",
        "crop_observer_outcome": "",
        "full_frame_verification_attempted": False,
        "full_frame_observer_outcome": "",
        "background_only": "",
        "object_fully_visible": "",
        "canonical_dominant_colors": [],
        "canonical_observed_shapes": [],
        "canonical_object_categories": [],
        "required_target_properties": _required_property_labels(_parse_visual_target_requirements(normalized_query)),
        "matched_target_properties": [],
        "missing_target_properties": _required_property_labels(_parse_visual_target_requirements(normalized_query)),
        "final_grounding_state": "",
        "final_verification_type": "",
        "retry_count": max(0, int(provider_diagnostics.get("locator_retry_count") or 0)),
    }
    return diagnostics


def _mark_dom_grounding_diagnostics_verified(
    diagnostics: dict[str, Any],
    *,
    normalized_query: str,
    preserve_attempts: bool = True,
) -> None:
    requirements = _parse_visual_target_requirements(normalized_query)
    required_properties = _required_property_labels(requirements)
    diagnostics["required_target_properties"] = list(required_properties)
    diagnostics["matched_target_properties"] = list(required_properties)
    diagnostics["missing_target_properties"] = []
    diagnostics["final_grounding_state"] = "dom_verified"
    diagnostics["final_verification_type"] = "dom_verified"
    if preserve_attempts:
        diagnostics["crop_verification_attempted"] = False
        diagnostics["full_frame_verification_attempted"] = False


def _base_candidate_grounding_diagnostics(*, region: VisionRegion, capture: BrowserCaptureRecord) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "candidate_coordinate_space": str(region.attributes.get("coordinate_space") or "").strip(),
        "candidate_normalized_box": region.bounding_box.to_dict() if region.bounding_box is not None else "",
        "candidate_pixel_box": "",
        "candidate_css_pixel_box": "",
    }
    if region.bounding_box is not None and _valid_region_bbox(region.bounding_box):
        left, top, right, bottom = _bbox_to_capture_image_pixel_rect(region.bounding_box, capture)
        diagnostics["candidate_pixel_box"] = _pixel_box_dict(left, top, right, bottom)
        css_left, css_top, css_right, css_bottom = _bbox_to_capture_css_pixel_rect(region.bounding_box, capture)
        diagnostics["candidate_css_pixel_box"] = _pixel_box_dict(css_left, css_top, css_right, css_bottom)
    return diagnostics


def _expand_verification_context_bbox(
    candidate_bbox: VisionBoundingBox,
    *,
    capture: BrowserCaptureRecord,
    config: dict[str, Any],
) -> VisionBoundingBox:
    viewport_width = max(1, _capture_image_width(capture))
    viewport_height = max(1, _capture_image_height(capture))
    candidate_width_px = candidate_bbox.width * viewport_width
    candidate_height_px = candidate_bbox.height * viewport_height
    scale = max(1.0, int(config.get("vision_browser_capture_verification_context_scale_percent", 400)) / 100.0)
    target_width_px = max(
        candidate_width_px * scale,
        float(int(config.get("vision_browser_capture_verification_context_min_width_pixels", 280))),
    )
    target_height_px = max(
        candidate_height_px * scale,
        float(int(config.get("vision_browser_capture_verification_context_min_height_pixels", 280))),
    )
    max_area = viewport_width * viewport_height * (int(config.get("vision_browser_capture_verification_context_max_area_percent", 40)) / 100.0)
    current_area = target_width_px * target_height_px
    if current_area > max_area > 0:
        shrink = (max_area / current_area) ** 0.5
        target_width_px *= shrink
        target_height_px *= shrink
    target_width_px = min(target_width_px, float(viewport_width))
    target_height_px = min(target_height_px, float(viewport_height))
    center_x_px = (candidate_bbox.x + candidate_bbox.width / 2.0) * viewport_width
    center_y_px = (candidate_bbox.y + candidate_bbox.height / 2.0) * viewport_height
    left = center_x_px - target_width_px / 2.0
    top = center_y_px - target_height_px / 2.0
    left = min(max(0.0, left), max(0.0, viewport_width - target_width_px))
    top = min(max(0.0, top), max(0.0, viewport_height - target_height_px))
    return VisionBoundingBox(
        x=left / viewport_width,
        y=top / viewport_height,
        width=target_width_px / viewport_width,
        height=target_height_px / viewport_height,
    )


def _pixel_box_dict(left: int, top: int, right: int, bottom: int) -> dict[str, int]:
    return {
        "x1": int(left),
        "y1": int(top),
        "x2": int(right),
        "y2": int(bottom),
        "width": max(0, int(right - left)),
        "height": max(0, int(bottom - top)),
    }


def _verify_observed_crop_objects(
    observation: VisionCropObservation,
    *,
    normalized_query: str,
    capture: BrowserCaptureRecord,
    context_bbox: VisionBoundingBox,
    crop_width: int,
    crop_height: int,
    viewport_width: int,
    viewport_height: int,
) -> tuple[VisionRegion | None, dict[str, Any], str]:
    requirements = _parse_visual_target_requirements(normalized_query)
    diagnostics: dict[str, Any] = {
        "crop_observer_outcome": "no_objects",
        "background_only": "" if observation.background_only is None else ("yes" if observation.background_only else "no"),
        "object_fully_visible": "",
        "canonical_dominant_colors": [],
        "canonical_observed_shapes": [],
        "canonical_object_categories": [],
        "matched_target_properties": [],
        "missing_target_properties": _required_property_labels(requirements),
    }
    if not observation.objects:
        diagnostics["final_grounding_state"] = "candidate_unverified"
        return None, diagnostics, "Candidate region was not supported by observed crop evidence."
    best_object: VisionObservedObject | None = None
    best_match: dict[str, Any] | None = None
    for item in observation.objects[:4]:
        match = _match_observed_object(item, requirements=requirements, normalized_query=normalized_query)
        if best_object is None:
            best_object = item
            best_match = match
        if match["matched"]:
            best_object = item
            best_match = match
            break
    if best_object is None or best_match is None:
        diagnostics["final_grounding_state"] = "candidate_unverified"
        return None, diagnostics, "Candidate region was not supported by observed crop evidence."
    diagnostics["crop_observer_outcome"] = "matched" if best_match["matched"] else "object_mismatch"
    diagnostics["object_fully_visible"] = "" if best_object.object_fully_visible is None else ("yes" if best_object.object_fully_visible else "no")
    diagnostics["canonical_dominant_colors"] = sorted(best_match["observed_colors"])
    diagnostics["canonical_observed_shapes"] = sorted(best_match["observed_shapes"])
    diagnostics["canonical_object_categories"] = sorted(best_match["observed_categories"])
    diagnostics["matched_target_properties"] = best_match["matched_properties"]
    diagnostics["missing_target_properties"] = best_match["missing_properties"]
    if not best_match["matched"]:
        diagnostics["final_grounding_state"] = "candidate_unverified"
        return None, diagnostics, "Candidate region was not supported by observed crop evidence."
    mapped_bbox = _map_crop_bbox_to_viewport(
        best_object.bounding_box,
        context_bbox=context_bbox,
        crop_width=crop_width,
        crop_height=crop_height,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    if mapped_bbox is None or not _valid_region_bbox(mapped_bbox):
        diagnostics["final_grounding_state"] = "candidate_rejected"
        return None, diagnostics, "Candidate region was not supported by observed crop evidence."
    attributes = {
        "observed_colors": ", ".join(sorted(best_match["observed_colors"])),
        "observed_shapes": ", ".join(sorted(best_match["observed_shapes"])),
        "observed_categories": ", ".join(sorted(best_match["observed_categories"])),
        "relative_location": _relative_location(mapped_bbox),
        "verification_context": "expanded_crop",
    }
    attributes = {key: value for key, value in attributes.items() if value}
    label = normalized_query if best_match["required_properties"] else (best_object.summary.strip() or "match")
    region = VisionRegion(
        label=label,
        confidence=1.0,
        bounding_box=mapped_bbox,
        visible_text=best_object.visible_text.strip(),
        verification="crop_verified",
        attributes=attributes,
    )
    diagnostics["final_grounding_state"] = "crop_verified"
    diagnostics["final_verification_type"] = "crop_verified"
    return region, diagnostics, ""


def _map_crop_bbox_to_viewport(
    crop_bbox: VisionBoundingBox | None,
    *,
    context_bbox: VisionBoundingBox,
    crop_width: int,
    crop_height: int,
    viewport_width: int,
    viewport_height: int,
) -> VisionBoundingBox | None:
    if crop_bbox is None or not _valid_region_bbox(crop_bbox):
        return None
    context_left, context_top, context_right, context_bottom = bbox_to_pixel_rect(
        context_bbox,
        width=viewport_width,
        height=viewport_height,
    )
    pixel_left = context_left + crop_bbox.x * crop_width
    pixel_top = context_top + crop_bbox.y * crop_height
    pixel_right = context_left + (crop_bbox.x + crop_bbox.width) * crop_width
    pixel_bottom = context_top + (crop_bbox.y + crop_bbox.height) * crop_height
    if pixel_right <= pixel_left or pixel_bottom <= pixel_top:
        return None
    if pixel_left < 0 or pixel_top < 0 or pixel_right > context_right + 1 or pixel_bottom > context_bottom + 1:
        return None
    return VisionBoundingBox(
        x=float(pixel_left) / max(1, viewport_width),
        y=float(pixel_top) / max(1, viewport_height),
        width=float(pixel_right - pixel_left) / max(1, viewport_width),
        height=float(pixel_bottom - pixel_top) / max(1, viewport_height),
    )


def _capture_image_width(capture: BrowserCaptureRecord) -> int:
    return max(1, int(capture.screenshot_pixel_width or capture.viewport_width or 1))


def _capture_image_height(capture: BrowserCaptureRecord) -> int:
    return max(1, int(capture.screenshot_pixel_height or capture.viewport_height or 1))


def _capture_visual_viewport_width(capture: BrowserCaptureRecord) -> float:
    return float(capture.visual_viewport_width or capture.viewport_width or 1.0)


def _capture_visual_viewport_height(capture: BrowserCaptureRecord) -> float:
    return float(capture.visual_viewport_height or capture.viewport_height or 1.0)


def _capture_scale_x(capture: BrowserCaptureRecord) -> float:
    return float(_capture_image_width(capture)) / max(_capture_visual_viewport_width(capture), 1.0)


def _capture_scale_y(capture: BrowserCaptureRecord) -> float:
    return float(_capture_image_height(capture)) / max(_capture_visual_viewport_height(capture), 1.0)


def _bbox_to_capture_image_pixel_rect(bbox: VisionBoundingBox, capture: BrowserCaptureRecord) -> tuple[int, int, int, int]:
    return bbox_to_pixel_rect(bbox, width=_capture_image_width(capture), height=_capture_image_height(capture))


def _bbox_to_capture_css_pixel_rect(bbox: VisionBoundingBox, capture: BrowserCaptureRecord) -> tuple[int, int, int, int]:
    image_left, image_top, image_right, image_bottom = _bbox_to_capture_image_pixel_rect(bbox, capture)
    scale_x = _capture_scale_x(capture)
    scale_y = _capture_scale_y(capture)
    offset_left = float(capture.visual_viewport_offset_left or 0.0)
    offset_top = float(capture.visual_viewport_offset_top or 0.0)
    left = int(round((image_left / max(scale_x, 1e-9)) + offset_left))
    top = int(round((image_top / max(scale_y, 1e-9)) + offset_top))
    right = int(round((image_right / max(scale_x, 1e-9)) + offset_left))
    bottom = int(round((image_bottom / max(scale_y, 1e-9)) + offset_top))
    return left, top, right, bottom


def _dom_bbox_to_capture_image_bbox(dom_bbox: VisionBoundingBox | None, capture: BrowserCaptureRecord) -> VisionBoundingBox | None:
    if dom_bbox is None or not _valid_region_bbox(dom_bbox):
        return None
    css_width = max(1.0, float(capture.viewport_width or _capture_visual_viewport_width(capture) or 1.0))
    css_height = max(1.0, float(capture.viewport_height or _capture_visual_viewport_height(capture) or 1.0))
    offset_left = float(capture.visual_viewport_offset_left or 0.0)
    offset_top = float(capture.visual_viewport_offset_top or 0.0)
    scale_x = _capture_scale_x(capture)
    scale_y = _capture_scale_y(capture)
    image_width = float(_capture_image_width(capture))
    image_height = float(_capture_image_height(capture))
    css_left = (dom_bbox.x * css_width) - offset_left
    css_top = (dom_bbox.y * css_height) - offset_top
    css_right = css_left + (dom_bbox.width * css_width)
    css_bottom = css_top + (dom_bbox.height * css_height)
    image_left = css_left * scale_x
    image_top = css_top * scale_y
    image_right = css_right * scale_x
    image_bottom = css_bottom * scale_y
    if image_right <= image_left or image_bottom <= image_top:
        return None
    if image_left < 0 or image_top < 0 or image_right > image_width + 1 or image_bottom > image_height + 1:
        return None
    return VisionBoundingBox(
        x=image_left / image_width,
        y=image_top / image_height,
        width=(image_right - image_left) / image_width,
        height=(image_bottom - image_top) / image_height,
    )


def _query_prefers_dom_grounding(normalized_query: str) -> bool:
    query = normalized_query.strip()
    if not query:
        return False
    if query == "main heading":
        return True
    return any(
        token in query
        for token in (
            "heading",
            "button",
            "link",
            "search field",
            "search box",
            "textbox",
            "text box",
            "input field",
            "search input",
            "field",
        )
    )


def _browser_visual_positive_verification_is_consistent(region: VisionRegion, diagnostics: dict[str, Any]) -> bool:
    verification = str(region.verification or "").strip()
    if verification == "dom_verified":
        return str(diagnostics.get("dom_grounding_result") or "").strip() == "matched"
    if verification == "crop_verified":
        return str(diagnostics.get("crop_observer_outcome") or "").strip() == "matched"
    if verification == "pixel_verified":
        return str(diagnostics.get("full_frame_observer_outcome") or "").strip() == "matched"
    return False


def _bounded_pixel_statistics(image: LoadedVisionImage, *, prefix: str = "crop_") -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(image.image_bytes)) as decoded:
            rgb = decoded.convert("RGB")
            width, height = rgb.size
            total = max(1, width * height)
            red_pixels = 0
            white_pixels = 0
            non_background_pixels = 0
            pixels = rgb.load()
            for y in range(height):
                for x in range(width):
                    red, green, blue = pixels[x, y]
                    is_white = red >= 240 and green >= 240 and blue >= 240
                    is_red = red >= 140 and red > green * 1.2 and red > blue * 1.2
                    if is_white:
                        white_pixels += 1
                    else:
                        non_background_pixels += 1
                    if is_red:
                        red_pixels += 1
    except Exception:
        return {}
    return {
        f"{prefix}red_pixel_ratio": round(red_pixels / total, 6),
        f"{prefix}white_pixel_ratio": round(white_pixels / total, 6),
        f"{prefix}non_background_ratio": round(non_background_pixels / total, 6),
    }


def _parse_visual_target_requirements(normalized_query: str) -> dict[str, Any]:
    tokens = [token for token in normalized_query.split() if token]
    colors = set()
    shapes = set()
    categories = set()
    for token in tokens:
        colors.update(_canonicalize_color_token(token))
        shapes.update(_canonicalize_shape_token(token))
        categories.update(_canonicalize_category_token(token))
    return {
        "colors": colors,
        "shapes": shapes,
        "categories": categories,
        "text": normalized_query if any(token in normalized_query for token in ("text ", "word ", "label ", "heading", "button", "link", "field")) else "",
    }


def _required_property_labels(requirements: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    labels.extend(f"color:{item}" for item in sorted(requirements.get("colors", set())))
    labels.extend(f"shape:{item}" for item in sorted(requirements.get("shapes", set())))
    labels.extend(f"category:{item}" for item in sorted(requirements.get("categories", set())))
    if requirements.get("text"):
        labels.append(f"text:{requirements['text']}")
    return labels


def _match_observed_object(
    item: VisionObservedObject,
    *,
    requirements: dict[str, Any],
    normalized_query: str,
) -> dict[str, Any]:
    observed_colors = _canonicalize_color_values(item.dominant_colors, item.summary, item.object_categories)
    observed_shapes = _canonicalize_shape_values(item.shapes, item.summary, item.object_categories)
    observed_categories = _canonicalize_category_values(item.object_categories, item.summary)
    haystack = normalize_vision_query(
        " ".join(
            part
            for part in (
                item.summary,
                item.visible_text,
                " ".join(item.object_categories),
                " ".join(item.shapes),
                " ".join(item.dominant_colors),
            )
            if isinstance(part, str) and part.strip()
        )
    )
    matched_properties: list[str] = []
    missing_properties: list[str] = []
    for color in sorted(requirements["colors"]):
        if color in observed_colors:
            matched_properties.append(f"color:{color}")
        else:
            missing_properties.append(f"color:{color}")
    for shape in sorted(requirements["shapes"]):
        if shape in observed_shapes:
            matched_properties.append(f"shape:{shape}")
        else:
            missing_properties.append(f"shape:{shape}")
    for category in sorted(requirements["categories"]):
        if category in observed_categories:
            matched_properties.append(f"category:{category}")
        else:
            missing_properties.append(f"category:{category}")
    if requirements["text"]:
        if requirements["text"] in haystack:
            matched_properties.append(f"text:{requirements['text']}")
        else:
            missing_properties.append(f"text:{requirements['text']}")
    if item.object_fully_visible is False and requirements["shapes"]:
        for shape in sorted(requirements["shapes"]):
            label = f"shape:{shape}"
            if label in matched_properties:
                matched_properties.remove(label)
            if label not in missing_properties:
                missing_properties.append(label)
    if not requirements["colors"] and not requirements["shapes"] and not requirements["categories"] and not requirements["text"]:
        if normalized_query and normalized_query not in haystack:
            missing_properties.append(f"text:{normalized_query}")
    return {
        "matched": not missing_properties,
        "required_properties": _required_property_labels(requirements),
        "matched_properties": matched_properties,
        "missing_properties": missing_properties,
        "observed_colors": observed_colors,
        "observed_shapes": observed_shapes,
        "observed_categories": observed_categories,
    }


def _canonicalize_color_values(values: list[str], *extra_sources: Any) -> set[str]:
    observed: set[str] = set()
    for value in values:
        observed.update(_canonicalize_color_token(normalize_vision_query(value)))
    for source in extra_sources:
        observed.update(_canonicalize_color_tokens_from_text(normalize_vision_query(" ".join(source) if isinstance(source, list) else str(source or ""))))
    return observed


def _canonicalize_shape_values(values: list[str], *extra_sources: Any) -> set[str]:
    observed: set[str] = set()
    for value in values:
        observed.update(_canonicalize_shape_token(normalize_vision_query(value)))
    for source in extra_sources:
        observed.update(_canonicalize_shape_tokens_from_text(normalize_vision_query(" ".join(source) if isinstance(source, list) else str(source or ""))))
    return observed


def _canonicalize_category_values(values: list[str], *extra_sources: Any) -> set[str]:
    observed: set[str] = set()
    for value in values:
        observed.update(_canonicalize_category_token(normalize_vision_query(value)))
    for source in extra_sources:
        text = normalize_vision_query(" ".join(source) if isinstance(source, list) else str(source or ""))
        for token in text.split():
            observed.update(_canonicalize_category_token(token))
    return observed


def _canonicalize_color_token(token: str) -> set[str]:
    aliases = {
        "red": "red",
        "crimson": "red",
        "scarlet": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "orange": "orange",
        "purple": "purple",
        "black": "black",
        "white": "white",
        "gray": "gray",
        "grey": "gray",
        "brown": "brown",
    }
    normalized = token.strip().lower()
    if not normalized:
        return set()
    if normalized in aliases:
        return {aliases[normalized]}
    if normalized.startswith("#") and len(normalized) == 7:
        try:
            red = int(normalized[1:3], 16)
            green = int(normalized[3:5], 16)
            blue = int(normalized[5:7], 16)
        except ValueError:
            return set()
        if max(red, green, blue) < 40:
            return {"black"}
        if min(red, green, blue) > 220:
            return {"white"}
        if abs(red - green) < 20 and abs(green - blue) < 20:
            return {"gray"}
        if red >= 120 and red > green * 1.25 and red > blue * 1.25:
            return {"red"}
        if blue >= 120 and blue > red * 1.15 and blue > green * 1.15:
            return {"blue"}
        if green >= 120 and green > red * 1.15 and green > blue * 1.15:
            return {"green"}
        if red >= 150 and green >= 120 and blue < 100:
            return {"orange"}
        if red >= 140 and blue >= 120 and green < 120:
            return {"purple"}
    return set()


def _canonicalize_color_tokens_from_text(text: str) -> set[str]:
    observed: set[str] = set()
    for token in text.split():
        observed.update(_canonicalize_color_token(token))
    return observed


def _canonicalize_shape_token(token: str) -> set[str]:
    aliases = {
        "circle": "circle",
        "circular": "circle",
        "round": "circle",
        "disk": "circle",
        "disc": "circle",
        "square": "square",
        "rectangle": "rectangle",
        "rectangular": "rectangle",
        "triangle": "triangle",
        "triangular": "triangle",
        "oval": "oval",
        "elliptical": "oval",
        "star": "star",
        "line": "line",
    }
    normalized = token.strip().lower()
    return {aliases[normalized]} if normalized in aliases else set()


def _canonicalize_shape_tokens_from_text(text: str) -> set[str]:
    observed: set[str] = set()
    for token in text.split():
        observed.update(_canonicalize_shape_token(token))
    return observed


def _canonicalize_category_token(token: str) -> set[str]:
    aliases = {
        "shape": "shape",
        "icon": "icon",
        "logo": "logo",
        "image": "image",
        "photo": "image",
        "illustration": "image",
        "text": "text",
        "heading": "heading",
        "button": "button",
        "link": "link",
        "field": "field",
    }
    normalized = token.strip().lower()
    return {aliases[normalized]} if normalized in aliases else set()


def _dom_grounded_region_for_query(capture: BrowserCaptureRecord, normalized_query: str) -> VisionRegion | None:
    dom_match = _best_dom_match_for_query(capture.dom_elements, normalized_query)
    if dom_match is None:
        return None
    bbox = _dom_bbox_to_capture_image_bbox(_dict_bbox(dom_match.get("bounding_box")), capture)
    if bbox is None or not _valid_region_bbox(bbox):
        return None
    visible_text = str(dom_match.get("visible_text") or dom_match.get("accessible_name") or "").strip()
    label = visible_text or str(dom_match.get("element_type") or "match").strip()
    attributes: dict[str, str] = {"relative_location": _relative_location(bbox)}
    for key in ("role", "tag", "accessible_name", "element_type"):
        value = str(dom_match.get(key) or "").strip()
        if value:
            attributes[key] = value
    return VisionRegion(
        label=label,
        confidence=1.0,
        bounding_box=bbox,
        visible_text=visible_text,
        verification="dom_verified",
        attributes=attributes,
    )


def _valid_region_bbox(bbox: VisionBoundingBox) -> bool:
    if min(bbox.x, bbox.y, bbox.width, bbox.height) < 0:
        return False
    if bbox.width <= 0 or bbox.height <= 0:
        return False
    if bbox.x >= 1 or bbox.y >= 1:
        return False
    if bbox.x + bbox.width > 1.000001 or bbox.y + bbox.height > 1.000001:
        return False
    return True


def _best_dom_match_for_region(dom_elements: list[dict[str, Any]], bbox: VisionBoundingBox, *, capture: BrowserCaptureRecord) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for item in dom_elements[:24]:
        if not isinstance(item, dict):
            continue
        dom_bbox = _dom_bbox_to_capture_image_bbox(_dict_bbox(item.get("bounding_box")), capture)
        if dom_bbox is None:
            continue
        overlap = _bbox_overlap_ratio(bbox, dom_bbox)
        if overlap <= 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, item)
    return best[1] if best is not None and best[0] >= 0.1 else None


def _best_dom_match_for_query(dom_elements: list[dict[str, Any]], normalized_query: str) -> dict[str, Any] | None:
    best: tuple[tuple[int, float, float], dict[str, Any]] | None = None
    for item in dom_elements[:24]:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("visible", True)):
            continue
        bbox = _dict_bbox(item.get("bounding_box"))
        if bbox is None or not _valid_region_bbox(bbox):
            continue
        score = _dom_query_match_score(item, normalized_query)
        if score is None:
            continue
        ranking = (score, -bbox.y, -bbox.x)
        if best is None or ranking > best[0]:
            best = (ranking, item)
    return best[1] if best is not None else None


def _region_from_dom_match(region: VisionRegion, dom_match: dict[str, Any], *, capture: BrowserCaptureRecord) -> VisionRegion:
    visible_text = str(dom_match.get("visible_text") or dom_match.get("accessible_name") or "").strip()
    label = visible_text or str(region.label or "").strip() or str(dom_match.get("element_type") or "match").strip()
    attributes = dict(region.attributes)
    role = str(dom_match.get("role") or "").strip()
    tag = str(dom_match.get("tag") or "").strip()
    accessible_name = str(dom_match.get("accessible_name") or "").strip()
    element_type = str(dom_match.get("element_type") or "").strip()
    if role:
        attributes["role"] = role
    if tag:
        attributes["tag"] = tag
    if accessible_name:
        attributes["accessible_name"] = accessible_name
    if element_type:
        attributes["element_type"] = element_type
    dom_bbox = _dom_bbox_to_capture_image_bbox(_dict_bbox(dom_match.get("bounding_box")), capture)
    if dom_bbox is not None:
        attributes.setdefault("relative_location", _relative_location(dom_bbox))
    return replace(
        region,
        label=label,
        visible_text=visible_text,
        bounding_box=dom_bbox or region.bounding_box,
        verification="dom_verified",
        attributes=attributes,
    )


def _dom_match_supports_query(dom_match: dict[str, Any], normalized_query: str) -> bool:
    query = normalized_query.strip()
    if not query:
        return False
    haystacks = [
        normalize_vision_query(str(dom_match.get("element_type") or "")),
        normalize_vision_query(str(dom_match.get("role") or "")),
        normalize_vision_query(str(dom_match.get("tag") or "")),
        normalize_vision_query(str(dom_match.get("visible_text") or "")),
        normalize_vision_query(str(dom_match.get("accessible_name") or "")),
    ]
    if "heading" in query:
        return any(token in haystacks[0] or token in haystacks[1] or token in haystacks[2] for token in ("heading", "h1", "h2", "h3", "h4", "h5", "h6"))
    if any(token in query for token in ("search field", "search box", "input", "textbox", "text box", "field")):
        return any(token in haystacks[0] or token in haystacks[1] or token in haystacks[2] for token in ("input", "textbox", "searchbox"))
    if query in haystacks[3] or query in haystacks[4]:
        return True
    return any(part and part in " ".join(haystacks) for part in query.split())


def _dom_query_match_score(dom_match: dict[str, Any], normalized_query: str) -> int | None:
    query = normalized_query.strip()
    if not query:
        return None
    element_type = normalize_vision_query(str(dom_match.get("element_type") or ""))
    role = normalize_vision_query(str(dom_match.get("role") or ""))
    tag = normalize_vision_query(str(dom_match.get("tag") or ""))
    visible_text = normalize_vision_query(str(dom_match.get("visible_text") or ""))
    accessible_name = normalize_vision_query(str(dom_match.get("accessible_name") or ""))
    combined_name = " ".join(part for part in (visible_text, accessible_name) if part)
    if query in {visible_text, accessible_name}:
        return 300
    if query == "main heading":
        if tag == "h1":
            return 280
        if element_type == "heading" or role == "heading":
            return 260
        return None
    if query.endswith(" heading"):
        if element_type == "heading" or role == "heading" or tag.startswith("h"):
            target = query[: -len(" heading")].strip()
            if target and target in combined_name:
                return 250
        return None
    if query.endswith(" link"):
        target = query[: -len(" link")].strip()
        if element_type == "link" and target and target in combined_name:
            return 240
        return None
    if query.endswith(" button"):
        target = query[: -len(" button")].strip()
        if (element_type == "button" or role == "button") and target and target in combined_name:
            return 240
        return None
    if any(token in query for token in ("search field", "search box", "textbox", "text box", "input field", "search input")):
        if any(token in " ".join((element_type, role, tag)) for token in ("input", "textbox", "searchbox")):
            return 230
        return None
    if _dom_match_supports_query(dom_match, normalized_query):
        return 180
    return None


def _has_observable_visual_attributes(region: VisionRegion, normalized_query: str) -> bool:
    meaningful_values: list[str] = []
    for key, value in region.attributes.items():
        normalized_key = str(key or "").strip().lower()
        if normalized_key in {"query", "command", "action", "tool", "instruction", "url", "path"}:
            return False
        normalized_value = normalize_vision_query(str(value or ""))
        if not normalized_value or normalized_value == normalized_query:
            continue
        meaningful_values.append(normalized_value)
    if region.visible_text and normalize_vision_query(region.visible_text) != normalized_query:
        meaningful_values.append(normalize_vision_query(region.visible_text))
    return bool(meaningful_values)


def _dict_bbox(value: Any) -> VisionBoundingBox | None:
    if not isinstance(value, dict):
        return None
    try:
        return VisionBoundingBox(
            x=float(value.get("x")),
            y=float(value.get("y")),
            width=float(value.get("width")),
            height=float(value.get("height")),
        )
    except (TypeError, ValueError):
        return None


def _bbox_overlap_ratio(first: VisionBoundingBox, second: VisionBoundingBox) -> float:
    first_x2 = first.x + first.width
    first_y2 = first.y + first.height
    second_x2 = second.x + second.width
    second_y2 = second.y + second.height
    intersect_x1 = max(first.x, second.x)
    intersect_y1 = max(first.y, second.y)
    intersect_x2 = min(first_x2, second_x2)
    intersect_y2 = min(first_y2, second_y2)
    if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
        return 0.0
    intersection = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    first_area = first.width * first.height
    second_area = second.width * second.height
    denominator = max(first_area, second_area, 1e-9)
    return intersection / denominator


def _relative_location(bbox: VisionBoundingBox) -> str:
    center_x = bbox.x + (bbox.width / 2.0)
    center_y = bbox.y + (bbox.height / 2.0)
    horizontal = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
    vertical = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"
    return f"{vertical}-{horizontal}"
