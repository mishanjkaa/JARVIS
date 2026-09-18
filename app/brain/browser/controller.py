from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.brain.agent.state import get_agent_runtime_state
from app.brain.audit.audit_log import record_audit_event
from app.brain.browser.backends import BrowserBackend, PlaywrightBrowserBackend
from app.brain.browser.errors import (
    BrowserCancelledError,
    BrowserDisabledError,
    BrowserOperationError,
    BrowserPolicyError,
    BrowserSessionError,
    BrowserTimeoutError,
    BrowserUnavailableError,
)
from app.brain.browser.models import (
    ALLOWED_CLICK_TARGET_TYPES,
    ALLOWED_ELEMENT_TYPES,
    ALLOWED_FORM_CONTROL_TYPES,
    ALLOWED_SCROLL_DIRECTIONS,
    ALLOWED_SCROLL_TARGET_TYPES,
    ALLOWED_SCREENSHOT_EXTENSIONS,
    ALLOWED_TAB_TARGETS,
    ALLOWED_WAIT_UNTIL,
    BrowserBackendStatus,
    BrowserElementInspectionResult,
    BrowserElementMetadata,
    BrowserEvidence,
    BrowserFormInspectionResult,
    BrowserInteractionResult,
    BrowserNavigationResult,
    BrowserPageInfoResult,
    BrowserFormActionResult,
    BrowserScrollResult,
    BrowserScreenshotResult,
    BrowserSessionRecord,
    BrowserSessionStatus,
    BrowserTabOpenResult,
    BrowserTabRecord,
    BrowserTextResult,
    BrowserViewportCaptureResult,
)
from app.brain.browser.state import get_browser_state
from app.brain.browser.url_policy import BrowserUrlPolicy
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.path_policy import relative_audit_path, resolve_path
from config.config_loader import load_config


class BrowserController:
    def __init__(self, *, backend: BrowserBackend | None = None, policy: BrowserUrlPolicy | None = None) -> None:
        self.backend = backend or PlaywrightBrowserBackend()
        self.policy = policy or BrowserUrlPolicy()

    def effective_config(self) -> dict[str, Any]:
        return get_effective_runtime_config()

    def status(self) -> BrowserBackendStatus:
        return self.backend.status()

    def runtime_ready(self) -> bool:
        config = self.effective_config()
        if not config.get("browser_enabled", True):
            return False
        return self.status().runtime_ready

    def status_message(self) -> str:
        config = self.effective_config()
        status = self.status()
        state = get_browser_state()
        lines = [
            f"Browser enabled: {'yes' if config.get('browser_enabled', True) else 'no'}",
            f"Backend: {status.backend_name}",
            f"Python package available: {'yes' if status.python_package_available else 'no'}",
            f"Browser binary available: {'yes' if status.browser_binary_available else 'no'}",
            f"Active sessions: {len(state.sessions)}",
            f"Runtime ready: {'yes' if status.runtime_ready and config.get('browser_enabled', True) else 'no'}",
        ]
        if status.installation_guidance and not status.runtime_ready:
            lines.append("Installation guidance:")
            lines.extend(status.installation_guidance.splitlines())
        return "\n".join(lines)

    def sessions_message(self) -> str:
        state = get_browser_state()
        with state.lock:
            if not state.sessions:
                return "No browser sessions."
            return "\n".join(session.summary() for session in state.sessions.values())

    def tabs_message(self) -> str:
        session = self._focused_session()
        if session is None:
            return "No active browser session."
        return self._format_tabs(session)

    def planning_context(self) -> dict[str, Any]:
        session = self._focused_session()
        if session is None:
            return {"session_count": 0, "active_session": "", "active_tab": "", "tabs": []}
        active_tab = self._active_tab(session)
        return {
            "session_count": 1,
            "active_session": session.session_id,
            "active_tab": active_tab.tab_id if active_tab is not None else "",
            "tabs": self._tab_snapshots(session),
            "current_url": active_tab.current_url if active_tab is not None else "",
            "current_title": active_tab.current_title if active_tab is not None else "",
        }

    def close_all_sessions(self, *, reason: str = "manual") -> str:
        state = get_browser_state()
        closed = 0
        with state.lock:
            sessions = list(state.sessions.values())
        for session in sessions:
            evidence = self.close_session(session.session_id)
            if evidence.success:
                closed += 1
        if reason == "emergency":
            return "Browser sessions closed by emergency stop."
        return f"Closed {closed} browser session{'s' if closed != 1 else ''}."

    def cancel_current_operation(self, *, reason: str = "cancel") -> str:
        state = get_browser_state()
        with state.lock:
            session_id = state.active_session_id
            if not session_id:
                return "No browser operation is running."
            state.cancellation_requested = True
            session = state.sessions.get(session_id)
            if session is not None:
                self.backend.cancel_operation(session.backend_handle)
        record_audit_event("browser_cancelled", message=f"{reason}:{session_id}")
        if reason == "emergency":
            return "Browser emergency stop applied."
        return "Browser cancellation requested."

    def start_session(self, *, headless: bool | None = None) -> BrowserEvidence:
        config = self.effective_config()
        try:
            self._ensure_enabled(config)
        except BrowserDisabledError as error:
            return self._failure("start_session", error_category="disabled", error_reason=str(error))
        status = self.status()
        if not status.runtime_ready:
            return self._failure(
                "start_session",
                error_category="unavailable",
                error_reason="Browser runtime is unavailable right now.",
                browser_backend=status.backend_name,
            )
        headless_value = bool(config.get("browser_headless_default", True) if headless is None else headless)
        session_id = self._next_session_id()
        try:
            handle = self.backend.start_session(headless=headless_value)
        except BrowserUnavailableError as error:
            return self._failure("start_session", error_category="unavailable", error_reason=str(error), browser_backend=status.backend_name)
        now = datetime.now(timezone.utc).isoformat()
        session = BrowserSessionRecord(
            session_id=session_id,
            created_at=now,
            status=BrowserSessionStatus.READY,
            audit_correlation_id=uuid4().hex,
            headless=headless_value,
            backend_handle=handle,
        )
        initial_tab = self._create_tab_record(session, handle.page, created_at=now)
        session.tabs[initial_tab.tab_id] = initial_tab
        session.tab_order.append(initial_tab.tab_id)
        session.active_tab_id = initial_tab.tab_id
        self._sync_session_from_active_tab(session)
        state = get_browser_state()
        with state.lock:
            state.sessions[session_id] = session
            state.focused_session_id = session_id
            state.last_safe_status = f"Browser session {session_id} ready."
        record_audit_event("browser_session_started", message=session_id)
        return BrowserEvidence(
            browser_operation="start_session",
            success=True,
            session_id=session_id,
            session_created=True,
            status=session.status.value,
            tab_id=initial_tab.tab_id,
            active_tab_id=initial_tab.tab_id,
            tab_count=1,
            tabs=self._tab_snapshots(session),
            page_version=initial_tab.page_version,
            history_length=0,
            headless=headless_value,
            browser_backend=status.backend_name,
        )

    def get_active_session(self) -> BrowserEvidence:
        session = self._focused_session()
        if session is None:
            return self._failure("get_active_session", error_category="invalid_session", error_reason="No active browser session.")
        active_tab = self._active_tab(session)
        return BrowserEvidence(
            browser_operation="get_active_session",
            success=True,
            session_id=session.session_id,
            status=session.status.value,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            tab_count=len(session.tabs),
            tabs=self._tab_snapshots(session),
            page_version=active_tab.page_version if active_tab is not None else 0,
            url=active_tab.current_url if active_tab is not None else "",
            title=active_tab.current_title if active_tab is not None else "",
            browser_backend=self.status().backend_name,
        )

    def close_session(self, session_id: str) -> BrowserEvidence:
        session = self._get_session(session_id)
        if session is None:
            return self._failure("close_session", session_id=session_id, error_category="invalid_session", error_reason="Unknown browser session.")
        try:
            self.backend.close_session(session.backend_handle)
        except Exception as error:
            return self._failure("close_session", session_id=session_id, error_category="backend_error", error_reason=str(error) or "Could not close browser session.")
        try:
            from app.brain.vision.controller import get_vision_controller

            get_vision_controller().cleanup_captures_for_session(session_id)
        except Exception:
            pass
        state = get_browser_state()
        with state.lock:
            session.status = BrowserSessionStatus.CLOSED
            state.sessions.pop(session.session_id, None)
            if state.focused_session_id == session_id:
                state.focused_session_id = ""
            if state.active_session_id == session_id:
                state.active_session_id = ""
                state.active_operation = ""
                state.cancellation_requested = False
            state.last_safe_status = f"Browser session {session_id} closed."
        record_audit_event("browser_session_closed", message=session_id)
        return BrowserEvidence(
            browser_operation="close_session",
            success=True,
            session_id=session_id,
            status="closed",
            history_length=len(session.navigation_history),
            browser_backend=self.status().backend_name,
        )

    def open_url(self, *, session_id: str, url: str, wait_until: str, timeout_seconds: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "open_url")
        if isinstance(session, BrowserEvidence):
            return session
        try:
            wait_state = self._validate_wait_until(wait_until)
            timeout_value = self._validate_timeout(timeout_seconds)
            requested_url = self.policy.validate_url(url)
        except (BrowserPolicyError, BrowserOperationError) as error:
            return self._failure("open_url", session_id=session_id, requested_url=url, error_category="policy_rejected", error_reason=str(error))

        def _navigate() -> BrowserNavigationResult:
            return self.backend.open_url(
                session.backend_handle,
                url=requested_url,
                wait_until=wait_state,
                timeout_seconds=timeout_value,
                is_cancelled=self._is_cancelled,
            )

        outcome = self._run_session_operation(session, "open_url", _navigate)
        if isinstance(outcome, BrowserEvidence):
            return outcome
        try:
            redirect_chain = outcome.redirect_chain or [requested_url, outcome.final_url]
            validated_chain = self.policy.validate_redirect_chain([value for value in redirect_chain if value])
        except BrowserPolicyError as error:
            self.close_session(session.session_id)
            return self._failure(
                "open_url",
                session_id=session_id,
                requested_url=requested_url,
                final_url=outcome.final_url,
                error_category="policy_rejected",
                error_reason=str(error),
                browser_backend=self.status().backend_name,
            )
        self._record_navigation(session, outcome.final_url, outcome.title)
        active_tab = self._active_tab(session)
        record_audit_event("browser_navigation_completed", message=f"{requested_url} -> {outcome.final_url}")
        return BrowserEvidence(
            browser_operation="open_url",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            tab_count=len(session.tabs),
            requested_url=requested_url,
            final_url=outcome.final_url,
            title=outcome.title,
            page_version=active_tab.page_version if active_tab is not None else 0,
            redirect_chain=validated_chain,
            redirect_count=max(0, len(validated_chain) - 1),
            load_state=outcome.load_state,
            browser_backend=self.status().backend_name,
        )

    def get_page_info(self, *, session_id: str) -> BrowserEvidence:
        session = self._require_live_session(session_id, "get_page_info")
        if isinstance(session, BrowserEvidence):
            return session
        outcome = self._run_session_operation(session, "get_page_info", lambda: self.backend.get_page_info(session.backend_handle))
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.current_url = outcome.url
            active_tab.current_title = outcome.title
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="get_page_info",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=active_tab.page_version if active_tab is not None else 0,
            load_state=outcome.load_state,
            history_length=len(session.navigation_history),
            browser_backend=self.status().backend_name,
        )

    def extract_visible_text(self, *, session_id: str, max_characters: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "extract_visible_text")
        if isinstance(session, BrowserEvidence):
            return session
        limit = self._validate_text_limit(max_characters)
        outcome = self._run_session_operation(session, "extract_visible_text", lambda: self.backend.extract_visible_text(session.backend_handle))
        if isinstance(outcome, BrowserEvidence):
            return outcome
        text = str(outcome.text or "").strip()
        truncated = len(text) > limit
        safe_text = text[:limit]
        return BrowserEvidence(
            browser_operation="extract_visible_text",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=session.page_version,
            text=safe_text,
            text_truncated=truncated,
            character_count=len(text),
            browser_backend=self.status().backend_name,
        )

    def inspect_elements(self, *, session_id: str, element_types: list[str], max_elements: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "inspect_elements")
        if isinstance(session, BrowserEvidence):
            return session
        selected_types = self._validate_element_types(element_types)
        element_limit = self._validate_element_limit(max_elements)
        prefix = f"{session.session_id}-p{session.page_version}-e"
        outcome = self._run_session_operation(
            session,
            "inspect_elements",
            lambda: self.backend.inspect_elements(
                session.backend_handle,
                element_types=selected_types,
                max_elements=element_limit,
                element_id_prefix=prefix,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        serialized = [element.to_dict() for element in outcome.elements[:element_limit]]
        return BrowserEvidence(
            browser_operation="inspect_elements",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=session.page_version,
            elements=serialized,
            element_count=len(serialized),
            browser_backend=self.status().backend_name,
        )

    def inspect_form_controls(
        self,
        *,
        session_id: str,
        max_controls: int,
        tab_id: str = "",
        page_version: int = 0,
    ) -> BrowserEvidence:
        session = self._require_live_session(session_id, "inspect_form_controls")
        if isinstance(session, BrowserEvidence):
            return session
        stale = self._validate_page_binding(session, operation="inspect_form_controls", tab_id=tab_id, page_version=page_version)
        if stale is not None:
            return stale
        control_limit = self._validate_element_limit(max_controls)
        prefix = f"{session.session_id}-p{session.page_version}-f"
        outcome = self._run_session_operation(
            session,
            "inspect_form_controls",
            lambda: self.backend.inspect_form_controls(
                session.backend_handle,
                max_controls=control_limit,
                control_id_prefix=prefix,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        serialized = [control.to_dict() for control in outcome.controls[:control_limit]]
        return BrowserEvidence(
            browser_operation="inspect_form_controls",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=session.page_version,
            controls=serialized,
            control_count=len(serialized),
            browser_backend=self.status().backend_name,
        )

    def take_screenshot(self, *, session_id: str, path: str, full_page: bool) -> BrowserEvidence:
        session = self._require_live_session(session_id, "take_screenshot")
        if isinstance(session, BrowserEvidence):
            return session
        config = self.effective_config()
        try:
            resolved = resolve_path(path, prefer_directory=False, allow_missing=True)
            extension = resolved.absolute_path.suffix.lower()
            if extension not in ALLOWED_SCREENSHOT_EXTENSIONS:
                raise BrowserPolicyError("Unsupported screenshot image format.")
            if resolved.absolute_path.exists() and not config.get("browser_screenshot_overwrite", False):
                raise BrowserPolicyError("Screenshot path already exists.")
        except (BrowserPolicyError, FilesystemPathError) as error:
            return self._failure("take_screenshot", session_id=session_id, error_category="policy_rejected", error_reason=str(error))
        outcome = self._run_session_operation(
            session,
            "take_screenshot",
            lambda: self.backend.take_screenshot(session.backend_handle, target_path=resolved.absolute_path, full_page=full_page),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        audit_path = relative_audit_path(resolved.absolute_path, resolved.root)
        record_audit_event("browser_screenshot_saved", message=audit_path)
        return BrowserEvidence(
            browser_operation="take_screenshot",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            screenshot_path=audit_path,
            full_page=full_page,
            browser_backend=self.status().backend_name,
        )

    def capture_view(self, *, session_id: str, tab_id: str = "") -> BrowserEvidence:
        session = self._require_live_session(session_id, "capture_view")
        if isinstance(session, BrowserEvidence):
            return session
        stale = self._validate_page_binding(session, operation="capture_view", tab_id=tab_id, page_version=0)
        if stale is not None:
            return stale
        outcome = self._run_session_operation(
            session,
            "capture_view",
            lambda: self.backend.capture_viewport(session.backend_handle),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is None:
            return self._failure("capture_view", session_id=session_id, error_category="invalid_session", error_reason="No active browser tab is available.")
        owner_request_id, owner_agent_task_id = self._capture_owner_ids()
        origin = self._origin_for(active_tab.current_url)
        dom_elements = self._capture_dom_elements(session)
        try:
            from app.brain.vision.controller import get_vision_controller

            capture = get_vision_controller().store_browser_capture(
                owner_request_id=owner_request_id,
                owner_agent_task_id=owner_agent_task_id,
                session_id=session.session_id,
                tab_id=active_tab.tab_id,
                url=active_tab.current_url,
                origin=origin,
                page_version=active_tab.page_version,
                width=outcome.width,
                height=outcome.height,
                screenshot_pixel_width=outcome.screenshot_pixel_width,
                screenshot_pixel_height=outcome.screenshot_pixel_height,
                visual_viewport_width=outcome.visual_viewport_width,
                visual_viewport_height=outcome.visual_viewport_height,
                visual_viewport_offset_left=outcome.visual_viewport_offset_left,
                visual_viewport_offset_top=outcome.visual_viewport_offset_top,
                scroll_x=outcome.scroll_x,
                scroll_y=outcome.scroll_y,
                device_pixel_ratio=outcome.device_pixel_ratio,
                device_scale_factor=outcome.device_scale_factor,
                screenshot_scale_option=outcome.screenshot_scale,
                viewport_only=outcome.viewport_only,
                mime_type=outcome.mime_type,
                image_bytes=outcome.image_bytes,
                dom_elements=dom_elements,
            )
        except Exception as error:
            return self._failure(
                "capture_view",
                session_id=session_id,
                error_category="capture_failed",
                error_reason=str(error) or "Browser capture could not be stored safely.",
            )
        record_audit_event("browser_capture_created", task_id=owner_agent_task_id, message=f"{session.session_id}:{origin or '(no origin)'}")
        return BrowserEvidence(
            browser_operation="capture_view",
            success=True,
            session_id=session.session_id,
            capture_id=capture.capture_id,
            tab_id=active_tab.tab_id,
            active_tab_id=session.active_tab_id,
            url=active_tab.current_url,
            title=active_tab.current_title,
            origin=origin,
            page_version=active_tab.page_version,
            captured_at=capture.captured_at,
            expires_at=capture.expires_at,
            viewport_width=outcome.width,
            viewport_height=outcome.height,
            screenshot_pixel_width=outcome.screenshot_pixel_width,
            screenshot_pixel_height=outcome.screenshot_pixel_height,
            visual_viewport_width=outcome.visual_viewport_width,
            visual_viewport_height=outcome.visual_viewport_height,
            visual_viewport_offset_left=outcome.visual_viewport_offset_left,
            visual_viewport_offset_top=outcome.visual_viewport_offset_top,
            scroll_x=outcome.scroll_x,
            scroll_y=outcome.scroll_y,
            device_pixel_ratio=outcome.device_pixel_ratio,
            device_scale_factor=outcome.device_scale_factor,
            screenshot_scale=outcome.screenshot_scale,
            viewport_only=outcome.viewport_only,
            browser_backend=self.status().backend_name,
        )

    def go_back(self, *, session_id: str) -> BrowserEvidence:
        session = self._require_live_session(session_id, "go_back")
        if isinstance(session, BrowserEvidence):
            return session
        outcome = self._run_session_operation(session, "go_back", lambda: self.backend.go_back(session.backend_handle))
        if isinstance(outcome, BrowserEvidence):
            return outcome
        self._move_history(session, direction=-1, fallback_url=outcome.final_url, title=outcome.title)
        active_tab = self._active_tab(session)
        return BrowserEvidence(
            browser_operation="go_back",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            final_url=session.current_url,
            title=session.current_title,
            history_length=len(session.navigation_history),
            browser_backend=self.status().backend_name,
        )

    def go_forward(self, *, session_id: str) -> BrowserEvidence:
        session = self._require_live_session(session_id, "go_forward")
        if isinstance(session, BrowserEvidence):
            return session
        outcome = self._run_session_operation(session, "go_forward", lambda: self.backend.go_forward(session.backend_handle))
        if isinstance(outcome, BrowserEvidence):
            return outcome
        self._move_history(session, direction=1, fallback_url=outcome.final_url, title=outcome.title)
        active_tab = self._active_tab(session)
        return BrowserEvidence(
            browser_operation="go_forward",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            final_url=session.current_url,
            title=session.current_title,
            history_length=len(session.navigation_history),
            browser_backend=self.status().backend_name,
        )

    def wait_for_page(self, *, session_id: str, wait_until: str, timeout_seconds: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "wait_for_page")
        if isinstance(session, BrowserEvidence):
            return session
        try:
            wait_state = self._validate_wait_until(wait_until)
            timeout_value = self._validate_timeout(timeout_seconds)
        except BrowserOperationError as error:
            return self._failure("wait_for_page", session_id=session_id, error_category="invalid_arguments", error_reason=str(error))
        outcome = self._run_session_operation(
            session,
            "wait_for_page",
            lambda: self.backend.wait_for_page(session.backend_handle, wait_until=wait_state, timeout_seconds=timeout_value, is_cancelled=self._is_cancelled),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.current_url = outcome.url
            active_tab.current_title = outcome.title
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="wait_for_page",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            final_url=outcome.url,
            title=outcome.title,
            load_state=outcome.load_state,
            history_length=len(session.navigation_history),
            browser_backend=self.status().backend_name,
        )

    def inspect_clickable_elements(self, *, session_id: str, max_elements: int) -> BrowserEvidence:
        return self.inspect_elements(session_id=session_id, element_types=["links", "buttons"], max_elements=max_elements)

    def scroll_page(self, *, session_id: str, direction: str, amount: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "scroll_page")
        if isinstance(session, BrowserEvidence):
            return session
        scroll_direction = self._validate_scroll_direction(direction)
        scroll_amount = self._validate_scroll_amount(amount)
        outcome = self._run_session_operation(
            session,
            "scroll_page",
            lambda: self.backend.scroll_page(session.backend_handle, direction=scroll_direction, amount=scroll_amount),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.scroll_y = outcome.scroll_y
            active_tab.page_version += 1
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="scroll_page",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=active_tab.page_version if active_tab is not None else 0,
            scroll_y=outcome.scroll_y,
            browser_backend=self.status().backend_name,
        )

    def scroll_to_element(
        self,
        *,
        session_id: str,
        target_type: str,
        text_hint: str = "",
        href_hint: str = "",
        ordinal: int = 0,
    ) -> BrowserEvidence:
        session = self._require_live_session(session_id, "scroll_to_element")
        if isinstance(session, BrowserEvidence):
            return session
        validated_target = self._validate_scroll_target_type(target_type)
        outcome = self._run_session_operation(
            session,
            "scroll_to_element",
            lambda: self.backend.scroll_to_element(
                session.backend_handle,
                target_type=validated_target,
                text_hint=text_hint.strip(),
                href_hint=href_hint.strip(),
                ordinal=max(0, ordinal),
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.scroll_y = outcome.scroll_y
            active_tab.page_version += 1
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="scroll_to_element",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=active_tab.page_version if active_tab is not None else 0,
            scroll_y=outcome.scroll_y,
            target_description=outcome.target_description,
            browser_backend=self.status().backend_name,
        )

    def click_element(
        self,
        *,
        session_id: str,
        target_type: str,
        text_hint: str = "",
        href_hint: str = "",
        ordinal: int = 0,
        wait_until: str = "domcontentloaded",
        timeout_seconds: int = 30,
    ) -> BrowserEvidence:
        session = self._require_live_session(session_id, "click_element")
        if isinstance(session, BrowserEvidence):
            return session
        validated_target = self._validate_click_target_type(target_type)
        wait_state = self._validate_wait_until(wait_until)
        timeout_value = self._validate_timeout(timeout_seconds)
        outcome = self._run_session_operation(
            session,
            "click_element",
            lambda: self.backend.click_element(
                session.backend_handle,
                target_type=validated_target,
                text_hint=text_hint.strip(),
                href_hint=href_hint.strip(),
                ordinal=max(0, ordinal),
                wait_until=wait_state,
                timeout_seconds=timeout_value,
                is_cancelled=self._is_cancelled,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if outcome.navigated:
            self._record_navigation(session, outcome.url, outcome.title)
            active_tab = self._active_tab(session)
        elif active_tab is not None:
            active_tab.current_url = outcome.url
            active_tab.current_title = outcome.title
            active_tab.page_version += 1
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="click_element",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            final_url=outcome.url,
            title=outcome.title,
            page_version=active_tab.page_version if active_tab is not None else 0,
            target_description=outcome.target_description,
            target_type=outcome.target_type,
            navigated=outcome.navigated,
            browser_backend=self.status().backend_name,
        )

    def input_text(
        self,
        *,
        session_id: str,
        control_type: str,
        text: str,
        label_hint: str = "",
        placeholder_hint: str = "",
        name_hint: str = "",
        ordinal: int = 0,
        tab_id: str = "",
        page_version: int = 0,
    ) -> BrowserEvidence:
        session = self._require_live_session(session_id, "input_text")
        if isinstance(session, BrowserEvidence):
            return session
        stale = self._validate_page_binding(session, operation="input_text", tab_id=tab_id, page_version=page_version)
        if stale is not None:
            return stale
        validated_control = self._validate_form_control_type(control_type)
        if self._looks_sensitive_form_target(label_hint, placeholder_hint, name_hint):
            return self._failure("input_text", session_id=session_id, error_category="policy_rejected", error_reason="Sensitive form fields are not supported.")
        outcome = self._run_session_operation(
            session,
            "input_text",
            lambda: self.backend.input_text(
                session.backend_handle,
                control_type=validated_control,
                label_hint=label_hint.strip(),
                placeholder_hint=placeholder_hint.strip(),
                name_hint=name_hint.strip(),
                ordinal=max(0, ordinal),
                text=text,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.page_version += 1
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="input_text",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=session.page_version,
            target_description=outcome.target_description,
            target_type=outcome.control_type,
            field_changed=outcome.field_changed,
            text_length=outcome.text_length,
            browser_backend=self.status().backend_name,
        )

    def clear_input(
        self,
        *,
        session_id: str,
        control_type: str,
        label_hint: str = "",
        placeholder_hint: str = "",
        name_hint: str = "",
        ordinal: int = 0,
        tab_id: str = "",
        page_version: int = 0,
    ) -> BrowserEvidence:
        session = self._require_live_session(session_id, "clear_input")
        if isinstance(session, BrowserEvidence):
            return session
        stale = self._validate_page_binding(session, operation="clear_input", tab_id=tab_id, page_version=page_version)
        if stale is not None:
            return stale
        validated_control = self._validate_form_control_type(control_type)
        if self._looks_sensitive_form_target(label_hint, placeholder_hint, name_hint):
            return self._failure("clear_input", session_id=session_id, error_category="policy_rejected", error_reason="Sensitive form fields are not supported.")
        outcome = self._run_session_operation(
            session,
            "clear_input",
            lambda: self.backend.clear_input(
                session.backend_handle,
                control_type=validated_control,
                label_hint=label_hint.strip(),
                placeholder_hint=placeholder_hint.strip(),
                name_hint=name_hint.strip(),
                ordinal=max(0, ordinal),
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.page_version += 1
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="clear_input",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            url=outcome.url,
            title=outcome.title,
            page_version=session.page_version,
            target_description=outcome.target_description,
            target_type=outcome.control_type,
            field_changed=outcome.field_changed,
            text_length=0,
            browser_backend=self.status().backend_name,
        )

    def submit_form(
        self,
        *,
        session_id: str,
        label_hint: str = "",
        placeholder_hint: str = "",
        name_hint: str = "",
        form_text_hint: str = "",
        submit_text_hint: str = "",
        ordinal: int = 0,
        wait_until: str = "domcontentloaded",
        timeout_seconds: int = 30,
        allowed_destination_origin: str = "",
        page_context: str = "",
        tab_id: str = "",
        page_version: int = 0,
    ) -> BrowserEvidence:
        session = self._require_live_session(session_id, "submit_form")
        if isinstance(session, BrowserEvidence):
            return session
        stale = self._validate_page_binding(session, operation="submit_form", tab_id=tab_id, page_version=page_version)
        if stale is not None:
            return stale
        if self._looks_sensitive_form_target(label_hint, placeholder_hint, name_hint, submit_text_hint, page_context):
            return self._failure("submit_form", session_id=session_id, error_category="policy_rejected", error_reason="Sensitive or authenticated form submission is not supported.")
        wait_state = self._validate_wait_until(wait_until)
        timeout_value = self._validate_timeout(timeout_seconds)
        preview = self.inspect_form_controls(session_id=session_id, max_controls=self._validate_element_limit(40), tab_id=tab_id, page_version=page_version)
        if not preview.success:
            return preview
        target_preview = self._match_form_control_preview(
            preview.controls,
            label_hint=label_hint,
            placeholder_hint=placeholder_hint,
            name_hint=name_hint,
            form_text_hint=form_text_hint,
            submit_text_hint=submit_text_hint,
            ordinal=ordinal,
        )
        if isinstance(target_preview, BrowserEvidence):
            return target_preview
        action_url = str(target_preview.get("form_action") or "").strip()
        action_origin = self._validated_form_origin(action_url, session.current_url)
        current_origin = self._origin_for(session.current_url)
        if action_origin and current_origin and action_origin != current_origin:
            allowed = str(allowed_destination_origin or "").strip().lower()
            if not allowed or allowed != action_origin.lower():
                return self._failure("submit_form", session_id=session_id, error_category="policy_rejected", error_reason="Unexpected cross-origin form submission is not allowed.")
        outcome = self._run_session_operation(
            session,
            "submit_form",
            lambda: self.backend.submit_form(
                session.backend_handle,
                label_hint=label_hint.strip(),
                placeholder_hint=placeholder_hint.strip(),
                name_hint=name_hint.strip(),
                form_text_hint=form_text_hint.strip(),
                submit_text_hint=submit_text_hint.strip(),
                ordinal=max(0, ordinal),
                wait_until=wait_state,
                timeout_seconds=timeout_value,
                is_cancelled=self._is_cancelled,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            self._record_navigation(session, outcome.url, outcome.title)
            active_tab = self._active_tab(session)
        confirmation_limit = self._validate_text_limit(4000)
        confirmation_text = str(outcome.confirmation_text or "").strip()
        truncated = len(confirmation_text) > confirmation_limit
        safe_confirmation = confirmation_text[:confirmation_limit]
        return BrowserEvidence(
            browser_operation="submit_form",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            final_url=outcome.url,
            title=outcome.title,
            page_version=active_tab.page_version if active_tab is not None else session.page_version,
            target_description=outcome.target_description,
            target_type=outcome.control_type,
            submitted=outcome.submitted,
            navigated=outcome.navigated,
            form_action=action_origin,
            form_method=outcome.form_method,
            confirmation_text=safe_confirmation,
            confirmation_text_truncated=truncated,
            browser_backend=self.status().backend_name,
        )

    def reload_page(self, *, session_id: str, wait_until: str, timeout_seconds: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "reload_page")
        if isinstance(session, BrowserEvidence):
            return session
        wait_state = self._validate_wait_until(wait_until)
        timeout_value = self._validate_timeout(timeout_seconds)
        outcome = self._run_session_operation(
            session,
            "reload_page",
            lambda: self.backend.reload_page(
                session.backend_handle,
                wait_until=wait_state,
                timeout_seconds=timeout_value,
                is_cancelled=self._is_cancelled,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        active_tab = self._active_tab(session)
        if active_tab is not None:
            active_tab.current_url = outcome.final_url
            active_tab.current_title = outcome.title
            active_tab.page_version += 1
            self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="reload_page",
            success=True,
            session_id=session.session_id,
            tab_id=active_tab.tab_id if active_tab is not None else "",
            active_tab_id=session.active_tab_id,
            final_url=outcome.final_url,
            title=outcome.title,
            load_state=outcome.load_state,
            browser_backend=self.status().backend_name,
        )

    def open_new_tab(self, *, session_id: str, url: str, wait_until: str, timeout_seconds: int) -> BrowserEvidence:
        session = self._require_live_session(session_id, "open_new_tab")
        if isinstance(session, BrowserEvidence):
            return session
        try:
            wait_state = self._validate_wait_until(wait_until)
            timeout_value = self._validate_timeout(timeout_seconds)
            requested_url = self.policy.validate_url(url)
        except (BrowserPolicyError, BrowserOperationError) as error:
            return self._failure("open_new_tab", session_id=session_id, requested_url=url, error_category="policy_rejected", error_reason=str(error))
        outcome = self._run_session_operation(
            session,
            "open_new_tab",
            lambda: self.backend.open_new_tab(
                session.backend_handle,
                url=requested_url,
                wait_until=wait_state,
                timeout_seconds=timeout_value,
                is_cancelled=self._is_cancelled,
            ),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        try:
            redirect_chain = outcome.redirect_chain or [requested_url, outcome.final_url]
            validated_chain = self.policy.validate_redirect_chain([value for value in redirect_chain if value])
        except BrowserPolicyError as error:
            return self._failure(
                "open_new_tab",
                session_id=session_id,
                requested_url=requested_url,
                final_url=outcome.final_url,
                error_category="policy_rejected",
                error_reason=str(error),
                browser_backend=self.status().backend_name,
            )
        new_tab = self._create_tab_record(session, outcome.tab_handle)
        session.tabs[new_tab.tab_id] = new_tab
        session.tab_order.append(new_tab.tab_id)
        session.active_tab_id = new_tab.tab_id
        self._record_navigation(session, outcome.final_url, outcome.title)
        record_audit_event("browser_tab_opened", message=f"{session.session_id}:{new_tab.tab_id}")
        return BrowserEvidence(
            browser_operation="open_new_tab",
            success=True,
            session_id=session.session_id,
            tab_id=new_tab.tab_id,
            active_tab_id=session.active_tab_id,
            tab_count=len(session.tabs),
            tabs=self._tab_snapshots(session),
            requested_url=requested_url,
            final_url=outcome.final_url,
            title=outcome.title,
            redirect_chain=validated_chain,
            redirect_count=max(0, len(validated_chain) - 1),
            load_state=outcome.load_state,
            page_version=new_tab.page_version,
            browser_backend=self.status().backend_name,
        )

    def switch_tab(self, *, session_id: str, target: str = "current", tab_id: str = "") -> BrowserEvidence:
        session = self._require_live_session(session_id, "switch_tab")
        if isinstance(session, BrowserEvidence):
            return session
        target_tab = self._resolve_tab_target(session, target=target, tab_id=tab_id, operation="switch_tab")
        if isinstance(target_tab, BrowserEvidence):
            return target_tab
        outcome = self._run_session_operation(
            session,
            "switch_tab",
            lambda: self.backend.switch_tab(session.backend_handle, tab_handle=target_tab.backend_handle),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        session.active_tab_id = target_tab.tab_id
        target_tab.current_url = outcome.url
        target_tab.current_title = outcome.title
        self._sync_session_from_active_tab(session)
        record_audit_event("browser_tab_switched", message=f"{session.session_id}:{target_tab.tab_id}")
        return BrowserEvidence(
            browser_operation="switch_tab",
            success=True,
            session_id=session.session_id,
            tab_id=target_tab.tab_id,
            active_tab_id=target_tab.tab_id,
            tab_count=len(session.tabs),
            tabs=self._tab_snapshots(session),
            url=outcome.url,
            title=outcome.title,
            page_version=target_tab.page_version,
            browser_backend=self.status().backend_name,
        )

    def list_tabs(self, *, session_id: str) -> BrowserEvidence:
        session = self._require_live_session(session_id, "list_tabs")
        if isinstance(session, BrowserEvidence):
            return session
        return BrowserEvidence(
            browser_operation="list_tabs",
            success=True,
            session_id=session.session_id,
            tab_id=session.active_tab_id,
            active_tab_id=session.active_tab_id,
            tab_count=len(session.tabs),
            tabs=self._tab_snapshots(session),
            page_version=session.page_version,
            browser_backend=self.status().backend_name,
        )

    def close_tab(self, *, session_id: str, target: str = "current", tab_id: str = "") -> BrowserEvidence:
        session = self._require_live_session(session_id, "close_tab")
        if isinstance(session, BrowserEvidence):
            return session
        target_tab = self._resolve_tab_target(session, target=target, tab_id=tab_id, operation="close_tab")
        if isinstance(target_tab, BrowserEvidence):
            return target_tab
        successor_id = self._successor_tab_id(session, target_tab.tab_id)
        outcome = self._run_session_operation(
            session,
            "close_tab",
            lambda: self.backend.close_tab(session.backend_handle, tab_handle=target_tab.backend_handle),
        )
        if isinstance(outcome, BrowserEvidence):
            return outcome
        session.tabs.pop(target_tab.tab_id, None)
        session.tab_order = [item for item in session.tab_order if item != target_tab.tab_id]
        record_audit_event("browser_tab_closed", message=f"{session.session_id}:{target_tab.tab_id}")
        if not session.tab_order:
            closed = self.close_session(session.session_id)
            closed.browser_operation = "close_tab"
            closed.tab_id = target_tab.tab_id
            closed.tab_count = 0
            closed.tabs = []
            return closed
        session.active_tab_id = successor_id or session.tab_order[-1]
        active_tab = self._active_tab(session)
        if active_tab is not None:
            switch_outcome = self._run_session_operation(
                session,
                "switch_tab",
                lambda: self.backend.switch_tab(session.backend_handle, tab_handle=active_tab.backend_handle),
            )
            if isinstance(switch_outcome, BrowserEvidence):
                return switch_outcome
            active_tab.current_url = switch_outcome.url
            active_tab.current_title = switch_outcome.title
        self._sync_session_from_active_tab(session)
        return BrowserEvidence(
            browser_operation="close_tab",
            success=True,
            session_id=session.session_id,
            tab_id=target_tab.tab_id,
            active_tab_id=session.active_tab_id,
            tab_count=len(session.tabs),
            tabs=self._tab_snapshots(session),
            url=session.current_url,
            title=session.current_title,
            page_version=session.page_version,
            browser_backend=self.status().backend_name,
        )

    def _ensure_enabled(self, config: dict[str, Any]) -> None:
        if not config.get("browser_enabled", True):
            raise BrowserDisabledError("Browser runtime is disabled.")
        if get_agent_runtime_state().emergency_stop_active:
            raise BrowserDisabledError("Browser runtime is unavailable during emergency stop.")

    def _next_session_id(self) -> str:
        state = get_browser_state()
        with state.lock:
            session_id = f"browser-{state.next_session_id}"
            state.next_session_id += 1
            return session_id

    def _get_session(self, session_id: str) -> BrowserSessionRecord | None:
        state = get_browser_state()
        with state.lock:
            return state.sessions.get(session_id)

    def _focused_session(self) -> BrowserSessionRecord | None:
        state = get_browser_state()
        with state.lock:
            if state.focused_session_id:
                session = state.sessions.get(state.focused_session_id)
                if session is not None and session.status != BrowserSessionStatus.CLOSED:
                    return session
            live_sessions = [session for session in state.sessions.values() if session.status != BrowserSessionStatus.CLOSED]
            if len(live_sessions) == 1:
                state.focused_session_id = live_sessions[0].session_id
                return live_sessions[0]
            return None

    def _active_tab(self, session: BrowserSessionRecord) -> BrowserTabRecord | None:
        if session.active_tab_id and session.active_tab_id in session.tabs:
            return session.tabs[session.active_tab_id]
        for tab_id in session.tab_order:
            tab = session.tabs.get(tab_id)
            if tab is not None:
                session.active_tab_id = tab.tab_id
                return tab
        return None

    def _create_tab_record(self, session: BrowserSessionRecord, backend_handle: Any, *, created_at: str | None = None) -> BrowserTabRecord:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        tab_id = f"{session.session_id}-tab-{session.next_tab_id}"
        session.next_tab_id += 1
        current_url = str(getattr(backend_handle, "url", "") or "")
        current_title = str(getattr(backend_handle, "title", lambda: "")() if hasattr(backend_handle, "title") else "")
        return BrowserTabRecord(
            tab_id=tab_id,
            created_at=timestamp,
            current_url=current_url,
            current_title=current_title,
            backend_handle=backend_handle,
        )

    def _sync_session_from_active_tab(self, session: BrowserSessionRecord) -> None:
        active_tab = self._active_tab(session)
        if active_tab is None:
            session.current_url = ""
            session.current_title = ""
            session.navigation_history = []
            session.history_index = -1
            session.page_version = 0
            return
        session.current_url = active_tab.current_url
        session.current_title = active_tab.current_title
        session.navigation_history = list(active_tab.navigation_history)
        session.history_index = active_tab.history_index
        session.page_version = active_tab.page_version

    def _tab_snapshots(self, session: BrowserSessionRecord) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for tab_id in session.tab_order:
            tab = session.tabs.get(tab_id)
            if tab is None:
                continue
            snapshots.append(
                {
                    "tab_id": tab.tab_id,
                    "active": tab.tab_id == session.active_tab_id,
                    "url": tab.current_url,
                    "title": tab.current_title,
                    "history_length": len(tab.navigation_history),
                    "scroll_y": tab.scroll_y,
                }
            )
        return snapshots

    def _format_tabs(self, session: BrowserSessionRecord) -> str:
        lines = [f"Browser session: {session.session_id}"]
        snapshots = self._tab_snapshots(session)
        if not snapshots:
            lines.append("No open tabs.")
            return "\n".join(lines)
        lines.append(f"Active tab: {session.active_tab_id}")
        for tab in snapshots:
            marker = "*" if tab["active"] else "-"
            title = str(tab.get("title") or "(untitled)")
            url = str(tab.get("url") or "(no page loaded)")
            lines.append(f"{marker} {tab['tab_id']} | {title} | {url}")
        return "\n".join(lines)

    def _resolve_tab_target(
        self,
        session: BrowserSessionRecord,
        *,
        target: str,
        tab_id: str,
        operation: str,
    ) -> BrowserTabRecord | BrowserEvidence:
        if not session.tab_order:
            return self._failure(operation, session_id=session.session_id, error_category="invalid_session", error_reason="No browser tabs are open.")
        if isinstance(tab_id, str) and tab_id.strip():
            selected = session.tabs.get(tab_id.strip())
            if selected is None:
                return self._failure(operation, session_id=session.session_id, error_category="invalid_arguments", error_reason="Unknown browser tab.")
            return selected
        normalized = str(target or "current").strip().lower()
        if normalized not in ALLOWED_TAB_TARGETS:
            return self._failure(operation, session_id=session.session_id, error_category="invalid_arguments", error_reason="Unsupported browser tab target.")
        current_id = session.active_tab_id or session.tab_order[0]
        current_index = session.tab_order.index(current_id) if current_id in session.tab_order else 0
        if normalized == "current":
            return session.tabs[session.tab_order[current_index]]
        if normalized == "previous":
            if current_index == 0:
                return self._failure(operation, session_id=session.session_id, error_category="invalid_arguments", error_reason="No previous browser tab is available.")
            return session.tabs[session.tab_order[current_index - 1]]
        if normalized == "next":
            if current_index + 1 >= len(session.tab_order):
                return self._failure(operation, session_id=session.session_id, error_category="invalid_arguments", error_reason="No next browser tab is available.")
            return session.tabs[session.tab_order[current_index + 1]]
        if normalized == "first":
            return session.tabs[session.tab_order[0]]
        return session.tabs[session.tab_order[-1]]

    def _successor_tab_id(self, session: BrowserSessionRecord, closing_tab_id: str) -> str:
        if closing_tab_id not in session.tab_order:
            return session.active_tab_id
        index = session.tab_order.index(closing_tab_id)
        remaining = [tab_id for tab_id in session.tab_order if tab_id != closing_tab_id]
        if not remaining:
            return ""
        if index < len(remaining):
            return remaining[index]
        return remaining[-1]

    def _require_live_session(self, session_id: str, operation: str) -> BrowserSessionRecord | BrowserEvidence:
        if not isinstance(session_id, str) or not session_id.strip():
            return self._failure(operation, error_category="invalid_session", error_reason="Browser session ID is required.")
        session = self._get_session(session_id.strip())
        if session is None:
            return self._failure(operation, session_id=session_id, error_category="invalid_session", error_reason="Unknown browser session.")
        if session.status == BrowserSessionStatus.CLOSED:
            return self._failure(operation, session_id=session_id, error_category="closed_session", error_reason="Browser session is closed.")
        return session

    def _run_session_operation(self, session: BrowserSessionRecord, operation: str, callback):
        state = get_browser_state()
        with state.lock:
            state.active_session_id = session.session_id
            state.active_operation = operation
            state.cancellation_requested = False
            state.focused_session_id = session.session_id
            state.last_safe_status = f"Browser {operation} running."
            session.status = BrowserSessionStatus.NAVIGATING if operation in {"open_url", "open_new_tab", "wait_for_page", "reload_page", "click_element", "submit_form"} else BrowserSessionStatus.READY
        record_audit_event("browser_operation_started", message=f"{session.session_id}:{operation}")
        try:
            result = callback()
        except BrowserCancelledError as error:
            session.status = BrowserSessionStatus.READY
            record_audit_event("browser_operation_cancelled", message=f"{session.session_id}:{operation}")
            return self._failure(operation, session_id=session.session_id, error_category="cancelled", error_reason=str(error), cancelled=True)
        except BrowserTimeoutError as error:
            session.status = BrowserSessionStatus.READY
            record_audit_event("browser_operation_timed_out", message=f"{session.session_id}:{operation}")
            return self._failure(operation, session_id=session.session_id, error_category="timeout", error_reason=str(error), timed_out=True)
        except BrowserUnavailableError as error:
            session.status = BrowserSessionStatus.FAILED
            return self._failure(operation, session_id=session.session_id, error_category="unavailable", error_reason=str(error))
        except (BrowserSessionError, BrowserPolicyError, BrowserOperationError) as error:
            session.status = BrowserSessionStatus.FAILED
            record_audit_event("browser_operation_failed", message=f"{session.session_id}:{operation}")
            return self._failure(operation, session_id=session.session_id, error_category="operation_failed", error_reason=str(error))
        finally:
            with state.lock:
                state.active_session_id = ""
                state.active_operation = ""
                state.cancellation_requested = False
        session.status = BrowserSessionStatus.READY
        return result

    def _record_navigation(self, session: BrowserSessionRecord, final_url: str, title: str) -> None:
        active_tab = self._active_tab(session)
        if active_tab is None:
            return
        active_tab.current_url = final_url
        active_tab.current_title = title
        if active_tab.history_index < len(active_tab.navigation_history) - 1:
            del active_tab.navigation_history[active_tab.history_index + 1 :]
        active_tab.navigation_history.append(final_url)
        active_tab.history_index = len(active_tab.navigation_history) - 1
        active_tab.page_version += 1
        self._sync_session_from_active_tab(session)

    def _move_history(self, session: BrowserSessionRecord, *, direction: int, fallback_url: str, title: str) -> None:
        active_tab = self._active_tab(session)
        if active_tab is None:
            return
        next_index = active_tab.history_index + direction
        if 0 <= next_index < len(active_tab.navigation_history):
            active_tab.history_index = next_index
            active_tab.current_url = active_tab.navigation_history[next_index]
        else:
            active_tab.current_url = fallback_url or active_tab.current_url
        active_tab.current_title = title
        active_tab.page_version += 1
        self._sync_session_from_active_tab(session)

    def _validate_wait_until(self, value: str) -> str:
        candidate = str(value or "domcontentloaded").strip().lower()
        if candidate not in ALLOWED_WAIT_UNTIL:
            raise BrowserOperationError("Unsupported wait condition.")
        return candidate

    def _validate_timeout(self, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > 300:
            raise BrowserOperationError("Browser timeout is invalid.")
        return value

    def _validate_text_limit(self, value: int) -> int:
        config = self.effective_config()
        default_limit = int(config.get("browser_extract_text_max_chars", 4000))
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return default_limit
        return min(value, default_limit)

    def _validate_element_types(self, value: list[str]) -> list[str]:
        if not isinstance(value, list) or not value:
            raise BrowserOperationError("At least one element type is required.")
        selected = [str(item).strip().lower() for item in value if str(item).strip()]
        if not selected or any(item not in ALLOWED_ELEMENT_TYPES for item in selected):
            raise BrowserOperationError("Unsupported element type.")
        return selected

    def _validate_element_limit(self, value: int) -> int:
        config = self.effective_config()
        configured = int(config.get("browser_max_elements", 40))
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return configured
        return min(value, configured)

    def _validate_scroll_direction(self, value: str) -> str:
        candidate = str(value or "down").strip().lower()
        if candidate not in ALLOWED_SCROLL_DIRECTIONS:
            raise BrowserOperationError("Unsupported browser scroll direction.")
        return candidate

    def _validate_scroll_amount(self, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return 600
        return min(value, 4000)

    def _validate_scroll_target_type(self, value: str) -> str:
        candidate = str(value or "").strip().lower()
        if candidate not in ALLOWED_SCROLL_TARGET_TYPES:
            raise BrowserOperationError("Unsupported browser scroll target type.")
        return candidate

    def _validate_click_target_type(self, value: str) -> str:
        candidate = str(value or "").strip().lower()
        if candidate not in ALLOWED_CLICK_TARGET_TYPES:
            raise BrowserOperationError("Unsupported browser click target type.")
        return candidate

    def _validate_form_control_type(self, value: str) -> str:
        candidate = str(value or "").strip().lower()
        if candidate not in ALLOWED_FORM_CONTROL_TYPES:
            raise BrowserOperationError("Unsupported browser form control type.")
        return candidate

    def _validate_page_binding(
        self,
        session: BrowserSessionRecord,
        *,
        operation: str,
        tab_id: str,
        page_version: int,
    ) -> BrowserEvidence | None:
        active_tab = self._active_tab(session)
        if active_tab is None:
            return self._failure(operation, session_id=session.session_id, error_category="invalid_session", error_reason="No active browser tab is available.")
        if tab_id and tab_id.strip() != active_tab.tab_id:
            return self._failure(operation, session_id=session.session_id, error_category="stale_reference", error_reason="Browser tab reference is stale.")
        if isinstance(page_version, int) and page_version > 0 and page_version != active_tab.page_version:
            return self._failure(operation, session_id=session.session_id, error_category="stale_reference", error_reason="Browser page reference is stale.")
        return None

    def validate_capture_freshness(
        self,
        *,
        session_id: str,
        tab_id: str,
        url: str,
        origin: str,
        page_version: int,
    ) -> str:
        session = self._get_session(str(session_id or "").strip())
        if session is None or session.status == BrowserSessionStatus.CLOSED:
            return "Browser capture is stale because the browser session is no longer available."
        active_tab = self._active_tab(session)
        if active_tab is None:
            return "Browser capture is stale because the browser tab is no longer available."
        if str(tab_id or "").strip() and active_tab.tab_id != str(tab_id).strip():
            return "Browser capture is stale because the browser tab changed."
        if page_version > 0 and active_tab.page_version != page_version:
            return "Browser capture is stale because the page changed."
        current_url = str(active_tab.current_url or "").strip()
        if current_url != str(url or "").strip():
            return "Browser capture is stale because the page URL changed."
        current_origin = self._origin_for(current_url)
        if current_origin != str(origin or "").strip():
            return "Browser capture is stale because the page origin changed."
        return ""

    def _looks_sensitive_form_target(self, *values: str) -> bool:
        lowered = " ".join(str(value or "") for value in values).lower()
        return any(
            token in lowered
            for token in ("password", "passcode", "token", "otp", "mfa", "verification code", "credit card", "card number", "security code", "cvv", "login", "sign in")
        )

    def _match_form_control_preview(
        self,
        controls: list[dict[str, Any]],
        *,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        form_text_hint: str,
        submit_text_hint: str,
        ordinal: int,
    ) -> dict[str, Any] | BrowserEvidence:
        matches: list[dict[str, Any]] = []
        for control in controls:
            if not isinstance(control, dict):
                continue
            if label_hint and label_hint.lower() not in str(control.get("label") or "").lower():
                continue
            if placeholder_hint and placeholder_hint.lower() not in str(control.get("placeholder") or "").lower():
                continue
            if name_hint and name_hint.lower() not in str(control.get("name") or "").lower():
                continue
            if form_text_hint and form_text_hint.lower() not in str(control.get("form_text") or "").lower():
                continue
            submit_text = str(control.get("submit_text") or "").lower()
            form_text = str(control.get("form_text") or "").lower()
            if submit_text_hint and submit_text_hint.lower() not in submit_text and submit_text_hint.lower() not in form_text:
                continue
            matches.append(control)
        if ordinal > 0:
            if ordinal > len(matches):
                return self._failure("submit_form", error_category="operation_failed", error_reason="Requested form was not found.")
            return matches[ordinal - 1]
        if not matches:
            return self._failure("submit_form", error_category="operation_failed", error_reason="Requested form was not found.")
        if len(matches) > 1:
            return self._failure("submit_form", error_category="operation_failed", error_reason="Requested form target is ambiguous.")
        return matches[0]

    def _validated_form_origin(self, action_url: str, current_url: str) -> str:
        target = action_url or current_url
        if not target:
            return ""
        validated = self.policy.validate_url(target)
        return self._origin_for(validated)

    def _origin_for(self, value: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

    def _is_cancelled(self) -> bool:
        state = get_browser_state()
        with state.lock:
            return state.cancellation_requested or get_agent_runtime_state().emergency_stop_active

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

    def _capture_dom_elements(self, session: BrowserSessionRecord) -> list[dict[str, Any]]:
        try:
            outcome = self.backend.inspect_elements(
                session.backend_handle,
                element_types=["headings", "links", "buttons", "inputs", "images"],
                max_elements=24,
                element_id_prefix=f"{session.session_id}-p{session.page_version}-cv",
            )
        except Exception:
            return []
        if not isinstance(outcome, BrowserElementInspectionResult):
            return []
        return [element.to_dict() for element in outcome.elements[:24]]

    def _failure(
        self,
        operation: str,
        *,
        session_id: str = "",
        requested_url: str = "",
        final_url: str = "",
        error_category: str,
        error_reason: str,
        browser_backend: str = "",
        cancelled: bool = False,
        timed_out: bool = False,
    ) -> BrowserEvidence:
        return BrowserEvidence(
            browser_operation=operation,
            success=False,
            session_id=session_id,
            requested_url=requested_url,
            final_url=final_url,
            error_category=error_category,
            error_reason=error_reason[:200],
            cancelled=cancelled,
            timed_out=timed_out,
            browser_backend=browser_backend or self.status().backend_name,
        )


_CONTROLLER = BrowserController()


def get_browser_controller() -> BrowserController:
    return _CONTROLLER


def reset_browser_controller(*, backend: BrowserBackend | None = None, policy: BrowserUrlPolicy | None = None) -> BrowserController:
    global _CONTROLLER
    _CONTROLLER = BrowserController(backend=backend, policy=policy)
    return _CONTROLLER
