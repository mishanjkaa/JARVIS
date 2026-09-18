from __future__ import annotations

import io
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from app.brain.browser.errors import BrowserCancelledError, BrowserOperationError, BrowserTimeoutError, BrowserUnavailableError
from app.brain.browser.html_utils import inspect_form_controls_from_html
from app.brain.browser.models import (
    BrowserBackendStatus,
    BrowserBoundingBox,
    BrowserFormActionResult,
    BrowserFormControlMetadata,
    BrowserFormInspectionResult,
    BrowserInteractionResult,
    BrowserElementInspectionResult,
    BrowserElementMetadata,
    BrowserNavigationResult,
    BrowserPageInfoResult,
    BrowserScreenshotResult,
    BrowserScrollResult,
    BrowserTabOpenResult,
    BrowserTextResult,
    BrowserViewportCaptureResult,
)


class BrowserBackend(Protocol):
    name: str

    def status(self) -> BrowserBackendStatus:
        ...

    def start_session(self, *, headless: bool) -> Any:
        ...

    def close_session(self, handle: Any) -> None:
        ...

    def cancel_operation(self, handle: Any) -> None:
        ...

    def open_url(
        self,
        handle: Any,
        *,
        url: str,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserNavigationResult:
        ...

    def get_page_info(self, handle: Any) -> BrowserPageInfoResult:
        ...

    def extract_visible_text(self, handle: Any) -> BrowserTextResult:
        ...

    def inspect_elements(
        self,
        handle: Any,
        *,
        element_types: list[str],
        max_elements: int,
        element_id_prefix: str,
    ) -> BrowserElementInspectionResult:
        ...

    def take_screenshot(self, handle: Any, *, target_path: Path, full_page: bool) -> BrowserScreenshotResult:
        ...

    def capture_viewport(self, handle: Any) -> BrowserViewportCaptureResult:
        ...

    def go_back(self, handle: Any) -> BrowserNavigationResult:
        ...

    def go_forward(self, handle: Any) -> BrowserNavigationResult:
        ...

    def wait_for_page(self, handle: Any, *, wait_until: str, timeout_seconds: int, is_cancelled: callable) -> BrowserPageInfoResult:
        ...

    def open_new_tab(
        self,
        handle: Any,
        *,
        url: str,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserTabOpenResult:
        ...

    def switch_tab(self, handle: Any, *, tab_handle: Any) -> BrowserPageInfoResult:
        ...

    def close_tab(self, handle: Any, *, tab_handle: Any) -> None:
        ...

    def reload_page(self, handle: Any, *, wait_until: str, timeout_seconds: int, is_cancelled: callable) -> BrowserNavigationResult:
        ...

    def click_element(
        self,
        handle: Any,
        *,
        target_type: str,
        text_hint: str,
        href_hint: str,
        ordinal: int,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserInteractionResult:
        ...

    def scroll_page(self, handle: Any, *, direction: str, amount: int) -> BrowserScrollResult:
        ...

    def scroll_to_element(
        self,
        handle: Any,
        *,
        target_type: str,
        text_hint: str,
        href_hint: str,
        ordinal: int,
    ) -> BrowserScrollResult:
        ...

    def inspect_form_controls(
        self,
        handle: Any,
        *,
        max_controls: int,
        control_id_prefix: str,
    ) -> BrowserFormInspectionResult:
        ...

    def input_text(
        self,
        handle: Any,
        *,
        control_type: str,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        ordinal: int,
        text: str,
    ) -> BrowserFormActionResult:
        ...

    def clear_input(
        self,
        handle: Any,
        *,
        control_type: str,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        ordinal: int,
    ) -> BrowserFormActionResult:
        ...

    def submit_form(
        self,
        handle: Any,
        *,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        form_text_hint: str,
        submit_text_hint: str,
        ordinal: int,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserFormActionResult:
        ...


@dataclass
class _PlaywrightHandle:
    playwright: Any
    browser: Any
    context: Any
    page: Any


class PlaywrightBrowserBackend:
    name = "playwright"

    def __init__(self) -> None:
        self._status_cache: BrowserBackendStatus | None = None

    def status(self) -> BrowserBackendStatus:
        if self._status_cache is None:
            self._status_cache = self._probe_status()
        return self._status_cache

    def start_session(self, *, headless: bool) -> Any:
        module = self._sync_api()
        playwright = module.sync_playwright().start()
        try:
            browser = playwright.chromium.launch(headless=headless)
        except Exception as error:
            playwright.stop()
            raise BrowserUnavailableError("Browser binaries are not installed.") from error
        context = browser.new_context()
        page = context.new_page()
        return _PlaywrightHandle(playwright=playwright, browser=browser, context=context, page=page)

    def close_session(self, handle: Any) -> None:
        if not isinstance(handle, _PlaywrightHandle):
            return
        try:
            handle.context.close()
        finally:
            try:
                handle.browser.close()
            finally:
                handle.playwright.stop()

    def cancel_operation(self, handle: Any) -> None:
        if not isinstance(handle, _PlaywrightHandle):
            return
        try:
            handle.page.close()
        except Exception:
            pass

    def open_url(self, handle: Any, *, url: str, wait_until: str, timeout_seconds: int, is_cancelled: callable) -> BrowserNavigationResult:
        if is_cancelled():
            raise BrowserCancelledError("Browser navigation cancelled.")
        try:
            response = handle.page.goto(url, wait_until=wait_until, timeout=timeout_seconds * 1000)
            if is_cancelled():
                raise BrowserCancelledError("Browser navigation cancelled.")
            title = handle.page.title()
            final_url = str(handle.page.url or url)
            http_status = response.status if response is not None else None
            redirect_chain = [url] if final_url == url else [url, final_url]
            return BrowserNavigationResult(
                success=True,
                requested_url=url,
                final_url=final_url,
                title=title,
                load_state=wait_until,
                redirect_chain=redirect_chain,
                http_status=http_status,
            )
        except BrowserCancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "Timeout" in message or "timeout" in message:
                raise BrowserTimeoutError("Browser navigation timed out.") from error
            raise BrowserOperationError(message or "Browser navigation failed.") from error

    def get_page_info(self, handle: Any) -> BrowserPageInfoResult:
        try:
            return BrowserPageInfoResult(True, str(handle.page.url or ""), handle.page.title(), "load")
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not read page information.") from error

    def extract_visible_text(self, handle: Any) -> BrowserTextResult:
        try:
            text = handle.page.locator("body").inner_text()
            return BrowserTextResult(True, str(handle.page.url or ""), handle.page.title(), text)
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not extract page text.") from error

    def inspect_elements(self, handle: Any, *, element_types: list[str], max_elements: int, element_id_prefix: str) -> BrowserElementInspectionResult:
        selector_map = {
            "links": "a",
            "buttons": "button",
            "inputs": "input",
            "forms": "form",
            "headings": "h1, h2, h3, h4, h5, h6",
            "images": "img",
        }
        try:
            elements: list[BrowserElementMetadata] = []
            for element_type in element_types:
                selector = selector_map.get(element_type)
                if not selector:
                    continue
                locator = handle.page.locator(selector)
                count = min(locator.count(), max_elements - len(elements))
                for index in range(count):
                    item = locator.nth(index)
                    visible = item.is_visible()
                    if not visible:
                        continue
                    text = (item.inner_text() if element_type not in {"inputs", "images"} else "").strip()
                    aria = (item.get_attribute("aria-label") or item.get_attribute("alt") or item.get_attribute("value") or "").strip()
                    href = (item.get_attribute("href") or "").strip()
                    input_type = (item.get_attribute("type") or "").strip()
                    disabled = bool(item.is_disabled()) if hasattr(item, "is_disabled") else False
                    tag = selector.split(",")[0].strip().replace("h1", item.evaluate("node => node.tagName.toLowerCase()"))
                    role = (item.get_attribute("role") or "").strip()
                    bounding_box = _normalized_browser_bbox(item.bounding_box(), page=handle.page)
                    elements.append(
                        BrowserElementMetadata(
                            element_id=f"{element_id_prefix}{len(elements) + 1}",
                            element_type=element_type.rstrip("s"),
                            tag=str(tag).strip().lower(),
                            role=role,
                            visible_text=text,
                            accessible_name=aria or text,
                            input_type=input_type,
                            href=href,
                            disabled=disabled,
                            visible=True,
                            bounding_box=bounding_box,
                        )
                    )
                    if len(elements) >= max_elements:
                        break
                if len(elements) >= max_elements:
                    break
            return BrowserElementInspectionResult(True, str(handle.page.url or ""), handle.page.title(), elements)
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not inspect page elements.") from error

    def take_screenshot(self, handle: Any, *, target_path: Path, full_page: bool) -> BrowserScreenshotResult:
        try:
            handle.page.screenshot(path=str(target_path), full_page=full_page)
            return BrowserScreenshotResult(True, str(handle.page.url or ""), handle.page.title(), str(target_path), full_page)
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not save screenshot.") from error

    def capture_viewport(self, handle: Any) -> BrowserViewportCaptureResult:
        page = self._current_page(handle)
        try:
            image_bytes = page.screenshot(full_page=False, type="png")
            decoded_width = 0
            decoded_height = 0
            try:
                with Image.open(io.BytesIO(image_bytes)) as decoded:
                    decoded_width, decoded_height = int(decoded.width), int(decoded.height)
            except Exception:
                decoded_width = 0
                decoded_height = 0
            viewport = page.viewport_size or {}
            width = int(viewport.get("width") or 0)
            height = int(viewport.get("height") or 0)
            if width <= 0 or height <= 0:
                width = int(page.evaluate("() => Math.max(1, Math.round(window.innerWidth || document.documentElement.clientWidth || 1))"))
                height = int(page.evaluate("() => Math.max(1, Math.round(window.innerHeight || document.documentElement.clientHeight || 1))"))
            metrics = {}
            try:
                metrics = page.evaluate(
                    """() => {
                        const vv = window.visualViewport;
                        return {
                            visualViewportWidth: vv && Number.isFinite(vv.width) ? vv.width : (window.innerWidth || document.documentElement.clientWidth || 0),
                            visualViewportHeight: vv && Number.isFinite(vv.height) ? vv.height : (window.innerHeight || document.documentElement.clientHeight || 0),
                            visualViewportOffsetLeft: vv && Number.isFinite(vv.offsetLeft) ? vv.offsetLeft : 0,
                            visualViewportOffsetTop: vv && Number.isFinite(vv.offsetTop) ? vv.offsetTop : 0,
                            scrollX: Number.isFinite(window.scrollX) ? window.scrollX : 0,
                            scrollY: Number.isFinite(window.scrollY) ? window.scrollY : 0,
                            devicePixelRatio: Number.isFinite(window.devicePixelRatio) ? window.devicePixelRatio : 1,
                        };
                    }"""
                )
            except Exception:
                metrics = {}
            device_scale_factor = 0.0
            context = getattr(handle, "context", None)
            options = getattr(context, "_options", None)
            if isinstance(options, dict):
                try:
                    device_scale_factor = float(options.get("device_scale_factor") or options.get("deviceScaleFactor") or 0.0)
                except (TypeError, ValueError):
                    device_scale_factor = 0.0
            return BrowserViewportCaptureResult(
                success=True,
                image_bytes=bytes(image_bytes),
                mime_type="image/png",
                width=max(1, width),
                height=max(1, height),
                screenshot_pixel_width=max(1, decoded_width or width),
                screenshot_pixel_height=max(1, decoded_height or height),
                visual_viewport_width=float(metrics.get("visualViewportWidth") or width),
                visual_viewport_height=float(metrics.get("visualViewportHeight") or height),
                visual_viewport_offset_left=float(metrics.get("visualViewportOffsetLeft") or 0.0),
                visual_viewport_offset_top=float(metrics.get("visualViewportOffsetTop") or 0.0),
                scroll_x=float(metrics.get("scrollX") or 0.0),
                scroll_y=float(metrics.get("scrollY") or 0.0),
                device_pixel_ratio=float(metrics.get("devicePixelRatio") or 0.0),
                device_scale_factor=device_scale_factor,
                screenshot_scale="device",
                viewport_only=True,
                url=str(page.url or ""),
                title=page.title(),
            )
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not capture the current browser viewport.") from error

    def go_back(self, handle: Any) -> BrowserNavigationResult:
        try:
            handle.page.go_back(wait_until="load", timeout=30_000)
            return BrowserNavigationResult(True, "", str(handle.page.url or ""), handle.page.title(), "load", [str(handle.page.url or "")])
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not go back.") from error

    def go_forward(self, handle: Any) -> BrowserNavigationResult:
        try:
            handle.page.go_forward(wait_until="load", timeout=30_000)
            return BrowserNavigationResult(True, "", str(handle.page.url or ""), handle.page.title(), "load", [str(handle.page.url or "")])
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not go forward.") from error

    def wait_for_page(self, handle: Any, *, wait_until: str, timeout_seconds: int, is_cancelled: callable) -> BrowserPageInfoResult:
        if is_cancelled():
            raise BrowserCancelledError("Browser wait cancelled.")
        try:
            handle.page.wait_for_load_state(wait_until, timeout=timeout_seconds * 1000)
            if is_cancelled():
                raise BrowserCancelledError("Browser wait cancelled.")
            return BrowserPageInfoResult(True, str(handle.page.url or ""), handle.page.title(), wait_until)
        except BrowserCancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "Timeout" in message or "timeout" in message:
                raise BrowserTimeoutError("Browser wait timed out.") from error
            raise BrowserOperationError(message or "Could not wait for page load.") from error

    def open_new_tab(
        self,
        handle: Any,
        *,
        url: str,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserTabOpenResult:
        if is_cancelled():
            raise BrowserCancelledError("Browser navigation cancelled.")
        try:
            page = handle.context.new_page()
            response = page.goto(url, wait_until=wait_until, timeout=timeout_seconds * 1000)
            if is_cancelled():
                raise BrowserCancelledError("Browser navigation cancelled.")
            page.bring_to_front()
            handle.page = page
            final_url = str(page.url or url)
            redirect_chain = [url] if final_url == url else [url, final_url]
            return BrowserTabOpenResult(
                success=True,
                tab_handle=page,
                requested_url=url,
                final_url=final_url,
                title=page.title(),
                load_state=wait_until,
                redirect_chain=redirect_chain,
                http_status=response.status if response is not None else None,
            )
        except BrowserCancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "Timeout" in message or "timeout" in message:
                raise BrowserTimeoutError("Browser navigation timed out.") from error
            raise BrowserOperationError(message or "Could not open a new tab.") from error

    def switch_tab(self, handle: Any, *, tab_handle: Any) -> BrowserPageInfoResult:
        try:
            tab_handle.bring_to_front()
            handle.page = tab_handle
            return BrowserPageInfoResult(True, str(tab_handle.url or ""), tab_handle.title(), "load")
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not switch tabs.") from error

    def close_tab(self, handle: Any, *, tab_handle: Any) -> None:
        try:
            tab_handle.close()
            if handle.page == tab_handle:
                remaining_pages = [page for page in handle.context.pages if not page.is_closed()]
                handle.page = remaining_pages[-1] if remaining_pages else None
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not close tab.") from error

    def reload_page(self, handle: Any, *, wait_until: str, timeout_seconds: int, is_cancelled: callable) -> BrowserNavigationResult:
        page = self._current_page(handle)
        if is_cancelled():
            raise BrowserCancelledError("Browser reload cancelled.")
        try:
            response = page.reload(wait_until=wait_until, timeout=timeout_seconds * 1000)
            if is_cancelled():
                raise BrowserCancelledError("Browser reload cancelled.")
            final_url = str(page.url or "")
            return BrowserNavigationResult(
                success=True,
                requested_url=final_url,
                final_url=final_url,
                title=page.title(),
                load_state=wait_until,
                redirect_chain=[final_url] if final_url else [],
                http_status=response.status if response is not None else None,
            )
        except BrowserCancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "Timeout" in message or "timeout" in message:
                raise BrowserTimeoutError("Browser reload timed out.") from error
            raise BrowserOperationError(message or "Could not reload the page.") from error

    def click_element(
        self,
        handle: Any,
        *,
        target_type: str,
        text_hint: str,
        href_hint: str,
        ordinal: int,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserInteractionResult:
        page = self._current_page(handle)
        if is_cancelled():
            raise BrowserCancelledError("Browser click cancelled.")
        try:
            target = self._find_target(page, target_type=target_type, text_hint=text_hint, href_hint=href_hint, ordinal=ordinal)
            description = self._target_description(target_type, target)
            before_url = str(page.url or "")
            if target_type == "link":
                with page.expect_navigation(wait_until=wait_until, timeout=timeout_seconds * 1000):
                    target.click()
                if is_cancelled():
                    raise BrowserCancelledError("Browser click cancelled.")
                final_url = str(page.url or before_url)
                return BrowserInteractionResult(
                    success=True,
                    url=final_url,
                    title=page.title(),
                    target_description=description,
                    target_type=target_type,
                    navigated=final_url != before_url,
                )
            target.click(timeout=timeout_seconds * 1000)
            if is_cancelled():
                raise BrowserCancelledError("Browser click cancelled.")
            return BrowserInteractionResult(
                success=True,
                url=str(page.url or before_url),
                title=page.title(),
                target_description=description,
                target_type=target_type,
                navigated=False,
            )
        except BrowserCancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "Timeout" in message or "timeout" in message:
                raise BrowserTimeoutError("Browser click timed out.") from error
            raise BrowserOperationError(message or "Could not click the requested element.") from error

    def scroll_page(self, handle: Any, *, direction: str, amount: int) -> BrowserScrollResult:
        page = self._current_page(handle)
        delta = amount if direction == "down" else -amount
        try:
            page.evaluate("(value) => window.scrollBy(0, value)", delta)
            scroll_y = int(page.evaluate("() => Math.round(window.scrollY || 0)"))
            return BrowserScrollResult(True, str(page.url or ""), page.title(), scroll_y)
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not scroll the page.") from error

    def scroll_to_element(
        self,
        handle: Any,
        *,
        target_type: str,
        text_hint: str,
        href_hint: str,
        ordinal: int,
    ) -> BrowserScrollResult:
        page = self._current_page(handle)
        try:
            target = self._find_target(page, target_type=target_type, text_hint=text_hint, href_hint=href_hint, ordinal=ordinal)
            target.scroll_into_view_if_needed(timeout=15_000)
            scroll_y = int(page.evaluate("() => Math.round(window.scrollY || 0)"))
            return BrowserScrollResult(
                True,
                str(page.url or ""),
                page.title(),
                scroll_y,
                target_description=self._target_description(target_type, target),
            )
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not scroll to the requested element.") from error

    def inspect_form_controls(self, handle: Any, *, max_controls: int, control_id_prefix: str) -> BrowserFormInspectionResult:
        page = self._current_page(handle)
        try:
            elements: list[BrowserFormControlMetadata] = []
            locator = page.locator("input, textarea")
            count = locator.count()
            for index in range(count):
                item = locator.nth(index)
                payload = self._control_metadata(item)
                if payload is None:
                    continue
                elements.append(
                    BrowserFormControlMetadata(
                        control_id=f"{control_id_prefix}{len(elements) + 1}",
                        control_type=str(payload.get("control_type") or ""),
                        tag=str(payload.get("tag") or ""),
                        input_type=str(payload.get("input_type") or ""),
                        label=str(payload.get("label") or ""),
                        placeholder=str(payload.get("placeholder") or ""),
                        name=str(payload.get("name") or ""),
                        disabled=bool(payload.get("disabled")),
                        visible=bool(payload.get("visible")),
                        form_action=str(payload.get("form_action") or ""),
                        form_method=str(payload.get("form_method") or "get"),
                        form_text=str(payload.get("form_text") or ""),
                        submit_text=str(payload.get("submit_text") or ""),
                    )
                )
                if len(elements) >= max_controls:
                    break
            return BrowserFormInspectionResult(True, str(page.url or ""), page.title(), elements)
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not inspect form controls.") from error

    def input_text(
        self,
        handle: Any,
        *,
        control_type: str,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        ordinal: int,
        text: str,
    ) -> BrowserFormActionResult:
        page = self._current_page(handle)
        try:
            target, metadata = self._find_form_control(
                page,
                control_type=control_type,
                label_hint=label_hint,
                placeholder_hint=placeholder_hint,
                name_hint=name_hint,
                ordinal=ordinal,
            )
            target.fill(text, timeout=15_000)
            target_description = self._control_description(metadata)
            return BrowserFormActionResult(
                success=True,
                url=str(page.url or ""),
                title=page.title(),
                target_description=target_description,
                control_type=control_type,
                field_changed=True,
                text_length=len(text),
            )
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not enter text into the requested control.") from error

    def clear_input(
        self,
        handle: Any,
        *,
        control_type: str,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        ordinal: int,
    ) -> BrowserFormActionResult:
        page = self._current_page(handle)
        try:
            target, metadata = self._find_form_control(
                page,
                control_type=control_type,
                label_hint=label_hint,
                placeholder_hint=placeholder_hint,
                name_hint=name_hint,
                ordinal=ordinal,
            )
            target.fill("", timeout=15_000)
            target_description = self._control_description(metadata)
            return BrowserFormActionResult(
                success=True,
                url=str(page.url or ""),
                title=page.title(),
                target_description=target_description,
                control_type=control_type,
                field_changed=True,
                text_length=0,
            )
        except Exception as error:
            raise BrowserOperationError(str(error) or "Could not clear the requested control.") from error

    def submit_form(
        self,
        handle: Any,
        *,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        form_text_hint: str,
        submit_text_hint: str,
        ordinal: int,
        wait_until: str,
        timeout_seconds: int,
        is_cancelled: callable,
    ) -> BrowserFormActionResult:
        page = self._current_page(handle)
        if is_cancelled():
            raise BrowserCancelledError("Browser form submission cancelled.")
        try:
            form_locator, control_metadata = self._find_target_form(
                page,
                label_hint=label_hint,
                placeholder_hint=placeholder_hint,
                name_hint=name_hint,
                form_text_hint=form_text_hint,
                submit_text_hint=submit_text_hint,
                ordinal=ordinal,
            )
            before_url = str(page.url or "")
            with page.expect_navigation(wait_until=wait_until, timeout=timeout_seconds * 1000):
                form_locator.evaluate("(form) => { if (form.requestSubmit) { form.requestSubmit(); } else { form.submit(); } }")
            if is_cancelled():
                raise BrowserCancelledError("Browser form submission cancelled.")
            final_url = str(page.url or before_url)
            confirmation_text = page.locator("body").inner_text().strip()
            return BrowserFormActionResult(
                success=True,
                url=final_url,
                title=page.title(),
                target_description=self._control_description(control_metadata),
                control_type=str(control_metadata.get("control_type") or ""),
                submitted=True,
                form_action=str(control_metadata.get("form_action") or ""),
                form_method=str(control_metadata.get("form_method") or "get"),
                confirmation_text=confirmation_text,
                navigated=final_url != before_url,
            )
        except BrowserCancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "Timeout" in message or "timeout" in message:
                raise BrowserTimeoutError("Browser form submission timed out.") from error
            raise BrowserOperationError(message or "Could not submit the requested form.") from error

    def _probe_status(self) -> BrowserBackendStatus:
        package_available = self._has_sync_api()
        binary_available = False
        if package_available:
            try:
                module = self._sync_api()
                playwright = module.sync_playwright().start()
                try:
                    browser = playwright.chromium.launch(headless=True)
                    browser.close()
                    binary_available = True
                finally:
                    playwright.stop()
            except Exception:
                binary_available = False
        guidance = (
            "Install Python package: pip install playwright\n"
            "Install browser binaries: python -m playwright install chromium"
        )
        return BrowserBackendStatus(
            backend_name=self.name,
            python_package_available=package_available,
            browser_binary_available=binary_available,
            runtime_ready=package_available and binary_available,
            installation_guidance=guidance,
        )

    def _has_sync_api(self) -> bool:
        try:
            return importlib.util.find_spec("playwright.sync_api") is not None
        except ModuleNotFoundError:
            return False

    def _sync_api(self) -> Any:
        if not self._has_sync_api():
            raise BrowserUnavailableError("Playwright is not installed.")
        from playwright import sync_api  # type: ignore

        return sync_api

    def _current_page(self, handle: Any) -> Any:
        page = getattr(handle, "page", None)
        if page is None:
            raise BrowserOperationError("No active browser tab is available.")
        return page

    def _find_target(self, page: Any, *, target_type: str, text_hint: str, href_hint: str, ordinal: int) -> Any:
        selector_map = {
            "link": "a",
            "button": "button",
            "input": "input",
            "form": "form",
            "heading": "h1, h2, h3, h4, h5, h6",
            "image": "img",
        }
        selector = selector_map.get(target_type)
        if not selector:
            raise BrowserOperationError("Requested browser element type is unsupported.")
        locator = page.locator(selector)
        candidates: list[Any] = []
        for index in range(locator.count()):
            item = locator.nth(index)
            if not item.is_visible():
                continue
            text_value = ""
            try:
                text_value = (item.inner_text() or "").strip()
            except Exception:
                text_value = (item.get_attribute("aria-label") or item.get_attribute("alt") or item.get_attribute("value") or "").strip()
            href_value = (item.get_attribute("href") or "").strip() if target_type == "link" else ""
            if text_hint and text_hint.lower() not in text_value.lower():
                continue
            if href_hint and href_hint.lower() not in href_value.lower():
                continue
            candidates.append(item)
        if ordinal > 0:
            if ordinal > len(candidates):
                raise BrowserOperationError("Requested browser element was not found.")
            return candidates[ordinal - 1]
        if not candidates:
            raise BrowserOperationError("Requested browser element was not found.")
        return candidates[0]

    def _find_form_control(
        self,
        page: Any,
        *,
        control_type: str,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        ordinal: int,
    ) -> tuple[Any, dict[str, Any]]:
        selector = "input" if control_type == "text_input" else "textarea"
        locator = page.locator(selector)
        matches: list[tuple[Any, dict[str, Any]]] = []
        for index in range(locator.count()):
            item = locator.nth(index)
            metadata = self._control_metadata(item)
            if metadata is None or metadata.get("control_type") != control_type:
                continue
            if label_hint and label_hint.lower() not in str(metadata.get("label") or "").lower():
                continue
            if placeholder_hint and placeholder_hint.lower() not in str(metadata.get("placeholder") or "").lower():
                continue
            if name_hint and name_hint.lower() not in str(metadata.get("name") or "").lower():
                continue
            matches.append((item, metadata))
        if ordinal > 0:
            if ordinal > len(matches):
                raise BrowserOperationError("Requested form control was not found.")
            return matches[ordinal - 1]
        if not matches:
            raise BrowserOperationError("Requested form control was not found.")
        if len(matches) > 1:
            raise BrowserOperationError("Form control target is ambiguous.")
        return matches[0]

    def _find_target_form(
        self,
        page: Any,
        *,
        label_hint: str,
        placeholder_hint: str,
        name_hint: str,
        form_text_hint: str,
        submit_text_hint: str,
        ordinal: int,
    ) -> tuple[Any, dict[str, Any]]:
        control_matches: list[tuple[Any, dict[str, Any]]] = []
        locator = page.locator("input, textarea")
        for index in range(locator.count()):
            item = locator.nth(index)
            metadata = self._control_metadata(item)
            if metadata is None:
                continue
            if label_hint and label_hint.lower() not in str(metadata.get("label") or "").lower():
                continue
            if placeholder_hint and placeholder_hint.lower() not in str(metadata.get("placeholder") or "").lower():
                continue
            if name_hint and name_hint.lower() not in str(metadata.get("name") or "").lower():
                continue
            if form_text_hint and form_text_hint.lower() not in str(metadata.get("form_text") or "").lower():
                continue
            submit_text = str(metadata.get("submit_text") or "").lower()
            form_text = str(metadata.get("form_text") or "").lower()
            if submit_text_hint and submit_text_hint.lower() not in submit_text and submit_text_hint.lower() not in form_text:
                continue
            control_matches.append((item, metadata))
        if ordinal > 0:
            if ordinal > len(control_matches):
                raise BrowserOperationError("Requested form was not found.")
            control_matches = [control_matches[ordinal - 1]]
        if not control_matches:
            raise BrowserOperationError("Requested form was not found.")
        if len(control_matches) > 1:
            raise BrowserOperationError("Requested form target is ambiguous.")
        item, metadata = control_matches[0]
        form_handle = item.evaluate_handle("(node) => node.form")
        return form_handle.as_element(), metadata

    def _control_metadata(self, item: Any) -> dict[str, Any] | None:
        try:
            payload = item.evaluate(
                """(node) => {
                    const tag = (node.tagName || '').toLowerCase();
                    const inputType = tag === 'input' ? String(node.getAttribute('type') || 'text').toLowerCase() : '';
                    if (tag === 'input' && !['', 'text', 'search', 'email', 'url', 'tel'].includes(inputType)) return null;
                    if (tag === 'input' && inputType === 'password') return null;
                    const style = globalThis.getComputedStyle ? globalThis.getComputedStyle(node) : null;
                    const visible = !!(node.offsetParent || style) && !(style && (style.display === 'none' || style.visibility === 'hidden'));
                    const disabled = !!node.disabled;
                    if (!visible || disabled) return null;
                    const labels = node.labels ? Array.from(node.labels).map((label) => (label.innerText || label.textContent || '').trim()).filter(Boolean) : [];
                    const form = node.form;
                    const submitter = form ? form.querySelector('button[type=\"submit\"], input[type=\"submit\"], button:not([type]), input[type=\"image\"]') : null;
                    return {
                        control_type: tag === 'textarea' ? 'textarea' : 'text_input',
                        tag,
                        input_type: inputType || 'text',
                        label: labels.join(' ').trim() || String(node.getAttribute('aria-label') || ''),
                        placeholder: String(node.getAttribute('placeholder') || ''),
                        name: String(node.getAttribute('name') || ''),
                        disabled,
                        visible,
                        form_action: form ? String(form.action || '') : '',
                        form_method: form ? String(form.method || 'get').toLowerCase() : 'get',
                        form_text: form ? String((form.innerText || form.textContent || '')).trim() : '',
                        submit_text: submitter ? String((submitter.innerText || submitter.value || submitter.textContent || '')).trim() : '',
                    };
                }"""
            )
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _control_description(self, metadata: dict[str, Any]) -> str:
        for key in ("label", "placeholder", "name", "control_type"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return "form control"

    def _target_description(self, target_type: str, locator: Any) -> str:
        text_value = ""
        href_value = ""
        try:
            text_value = (locator.inner_text() or "").strip()
        except Exception:
            text_value = ""
        if target_type == "link":
            try:
                href_value = (locator.get_attribute("href") or "").strip()
            except Exception:
                href_value = ""
        if text_value:
            return text_value
        if href_value:
            return href_value
        return target_type


def _normalized_browser_bbox(raw_box: Any, *, page: Any) -> BrowserBoundingBox | None:
    if not isinstance(raw_box, dict):
        return None
    try:
        x = float(raw_box.get("x"))
        y = float(raw_box.get("y"))
        width = float(raw_box.get("width"))
        height = float(raw_box.get("height"))
    except (TypeError, ValueError):
        return None
    if min(x, y, width, height) < 0 or width <= 0 or height <= 0:
        return None
    viewport = getattr(page, "viewport_size", None) or {}
    viewport_width = int(viewport.get("width") or 0)
    viewport_height = int(viewport.get("height") or 0)
    if viewport_width <= 0 or viewport_height <= 0:
        try:
            viewport_width = int(page.evaluate("() => Math.max(1, Math.round(window.innerWidth || document.documentElement.clientWidth || 1))"))
            viewport_height = int(page.evaluate("() => Math.max(1, Math.round(window.innerHeight || document.documentElement.clientHeight || 1))"))
        except Exception:
            return None
    normalized_x = x / viewport_width
    normalized_y = y / viewport_height
    normalized_width = width / viewport_width
    normalized_height = height / viewport_height
    if min(normalized_x, normalized_y, normalized_width, normalized_height) < 0:
        return None
    if normalized_x + normalized_width > 1.000001 or normalized_y + normalized_height > 1.000001:
        return None
    return BrowserBoundingBox(
        x=normalized_x,
        y=normalized_y,
        width=normalized_width,
        height=normalized_height,
    )
