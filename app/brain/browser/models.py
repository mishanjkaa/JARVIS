from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


ALLOWED_WAIT_UNTIL = ("domcontentloaded", "load", "networkidle")
ALLOWED_ELEMENT_TYPES = ("links", "buttons", "inputs", "forms", "headings", "images")
ALLOWED_SCREENSHOT_EXTENSIONS = (".png", ".jpg", ".jpeg")
ALLOWED_CLICK_TARGET_TYPES = ("link", "button")
ALLOWED_SCROLL_TARGET_TYPES = ("link", "button", "input", "form", "heading", "image")
ALLOWED_SCROLL_DIRECTIONS = ("up", "down")
ALLOWED_TAB_TARGETS = ("current", "previous", "next", "first", "last")
ALLOWED_FORM_CONTROL_TYPES = ("text_input", "textarea")


class BrowserSessionStatus(str, Enum):
    READY = "ready"
    NAVIGATING = "navigating"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class BrowserBackendStatus:
    backend_name: str
    python_package_available: bool
    browser_binary_available: bool
    runtime_ready: bool
    installation_guidance: str = ""


@dataclass
class BrowserSessionRecord:
    session_id: str
    created_at: str
    status: BrowserSessionStatus
    current_url: str = ""
    current_title: str = ""
    navigation_history: list[str] = field(default_factory=list)
    history_index: int = -1
    audit_correlation_id: str = ""
    headless: bool = True
    page_version: int = 0
    active_tab_id: str = ""
    tabs: dict[str, "BrowserTabRecord"] = field(default_factory=dict)
    tab_order: list[str] = field(default_factory=list)
    next_tab_id: int = 1
    backend_handle: Any = None

    def summary(self) -> str:
        current = self.current_url or "(no page loaded)"
        active = self.active_tab_id or "(no active tab)"
        return f"{self.session_id} | {self.status.value} | tabs={len(self.tabs)} | active={active} | {current}"


@dataclass
class BrowserTabRecord:
    tab_id: str
    created_at: str
    current_url: str = ""
    current_title: str = ""
    navigation_history: list[str] = field(default_factory=list)
    history_index: int = -1
    page_version: int = 0
    backend_handle: Any = None
    scroll_y: int = 0

    def summary(self, *, active: bool = False) -> str:
        current = self.current_url or "(no page loaded)"
        marker = "active" if active else "inactive"
        return f"{self.tab_id} | {marker} | {current}"


@dataclass
class BrowserNavigationResult:
    success: bool
    requested_url: str
    final_url: str
    title: str
    load_state: str
    redirect_chain: list[str] = field(default_factory=list)
    http_status: int | None = None
    error_category: str = ""
    error_reason: str = ""
    timed_out: bool = False
    cancelled: bool = False


@dataclass
class BrowserPageInfoResult:
    success: bool
    url: str
    title: str
    load_state: str = "load"
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserTextResult:
    success: bool
    url: str
    title: str
    text: str
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserBoundingBox:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
            "width": round(float(self.width), 6),
            "height": round(float(self.height), 6),
        }


@dataclass
class BrowserElementMetadata:
    element_id: str
    element_type: str
    tag: str
    role: str
    visible_text: str
    accessible_name: str
    input_type: str
    href: str
    disabled: bool
    visible: bool
    bounding_box: BrowserBoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.bounding_box is not None:
            payload["bounding_box"] = self.bounding_box.to_dict()
        return payload


@dataclass
class BrowserElementInspectionResult:
    success: bool
    url: str
    title: str
    elements: list[BrowserElementMetadata] = field(default_factory=list)
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserFormControlMetadata:
    control_id: str
    control_type: str
    tag: str
    input_type: str
    label: str
    placeholder: str
    name: str
    disabled: bool
    visible: bool
    form_action: str = ""
    form_method: str = "get"
    form_text: str = ""
    submit_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BrowserFormInspectionResult:
    success: bool
    url: str
    title: str
    controls: list[BrowserFormControlMetadata] = field(default_factory=list)
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserScreenshotResult:
    success: bool
    url: str
    title: str
    saved_path: str
    full_page: bool
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserViewportCaptureResult:
    success: bool
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    url: str
    title: str
    screenshot_pixel_width: int = 0
    screenshot_pixel_height: int = 0
    visual_viewport_width: float = 0.0
    visual_viewport_height: float = 0.0
    visual_viewport_offset_left: float = 0.0
    visual_viewport_offset_top: float = 0.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    device_pixel_ratio: float = 0.0
    device_scale_factor: float = 0.0
    screenshot_scale: str = ""
    viewport_only: bool = True
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserTabOpenResult:
    success: bool
    tab_handle: Any
    final_url: str
    title: str
    load_state: str
    redirect_chain: list[str] = field(default_factory=list)
    requested_url: str = ""
    http_status: int | None = None
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserInteractionResult:
    success: bool
    url: str
    title: str
    target_description: str = ""
    target_type: str = ""
    navigated: bool = False
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserFormActionResult:
    success: bool
    url: str
    title: str
    target_description: str = ""
    control_type: str = ""
    field_changed: bool = False
    submitted: bool = False
    text_length: int = 0
    page_version: int = 0
    form_action: str = ""
    form_method: str = "get"
    confirmation_text: str = ""
    navigated: bool = False
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserScrollResult:
    success: bool
    url: str
    title: str
    scroll_y: int
    target_description: str = ""
    error_category: str = ""
    error_reason: str = ""


@dataclass
class BrowserEvidence:
    browser_operation: str
    success: bool
    session_id: str = ""
    session_created: bool = False
    capture_id: str = ""
    status: str = ""
    tab_id: str = ""
    active_tab_id: str = ""
    tab_count: int = 0
    tabs: list[dict[str, Any]] = field(default_factory=list)
    page_version: int = 0
    requested_url: str = ""
    final_url: str = ""
    url: str = ""
    title: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    redirect_count: int = 0
    load_state: str = ""
    history_length: int = 0
    headless: bool = True
    text: str = ""
    text_truncated: bool = False
    character_count: int = 0
    elements: list[dict[str, Any]] = field(default_factory=list)
    element_count: int = 0
    controls: list[dict[str, Any]] = field(default_factory=list)
    control_count: int = 0
    target_description: str = ""
    target_type: str = ""
    field_changed: bool = False
    submitted: bool = False
    text_length: int = 0
    form_action: str = ""
    form_method: str = ""
    confirmation_text: str = ""
    confirmation_text_truncated: bool = False
    navigated: bool = False
    scroll_y: int = 0
    screenshot_path: str = ""
    full_page: bool = False
    origin: str = ""
    captured_at: str = ""
    expires_at: str = ""
    viewport_width: int = 0
    viewport_height: int = 0
    screenshot_pixel_width: int = 0
    screenshot_pixel_height: int = 0
    visual_viewport_width: float = 0.0
    visual_viewport_height: float = 0.0
    visual_viewport_offset_left: float = 0.0
    visual_viewport_offset_top: float = 0.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    device_pixel_ratio: float = 0.0
    device_scale_factor: float = 0.0
    screenshot_scale: str = ""
    viewport_only: bool = True
    error_category: str = ""
    error_reason: str = ""
    timed_out: bool = False
    cancelled: bool = False
    browser_backend: str = ""

    def to_reference_fields(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value not in ("", [], None)}
