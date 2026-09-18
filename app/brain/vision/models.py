from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_VISION_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_VISION_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
VISION_DETAIL_LEVELS = ("brief", "normal", "detailed")


@dataclass
class VisionProviderStatus:
    provider_name: str
    model: str
    configured: bool
    base_url_allowed: bool
    provider_reachable: bool
    model_installed: bool
    image_capability_ready: bool
    generation_ready: bool | None
    generation_detail: str = ""


@dataclass
class BrowserCaptureRecord:
    capture_id: str
    owner_request_id: int | None
    owner_agent_task_id: int | None
    source_type: str
    session_id: str
    tab_id: str
    url: str
    origin: str
    page_version: int
    viewport_width: int
    viewport_height: int
    captured_at: str
    expires_at: str
    image_format: str
    source_hash: str
    byte_size: int
    mime_type: str
    safe_display_name: str
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
    screenshot_scale_option: str = ""
    viewport_only: bool = True
    dom_elements: list[dict[str, Any]] = field(default_factory=list, repr=False)
    image_bytes: bytes = field(repr=False, default=b"")

    def public_metadata(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "source_type": self.source_type,
            "session_id": self.session_id,
            "tab_id": self.tab_id,
            "url": self.url,
            "origin": self.origin,
            "page_version": self.page_version,
            "width": self.viewport_width,
            "height": self.viewport_height,
            "screenshot_pixel_width": self.screenshot_pixel_width,
            "screenshot_pixel_height": self.screenshot_pixel_height,
            "captured_at": self.captured_at,
            "expires_at": self.expires_at,
        }


@dataclass
class VisionFrame:
    frame_id: str
    source_type: str
    safe_display_name: str
    source_hash: str
    mime_type: str
    width: int
    height: int
    created_at: str
    expires_at: str
    temporary_copy: bool
    trust_classification: str
    file_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisionBoundingBox:
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
class VisionOcrBlock:
    text: str
    confidence: float
    bounding_box: VisionBoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "confidence": round(float(self.confidence), 6),
        }
        if self.bounding_box is not None:
            payload["bounding_box"] = self.bounding_box.to_dict()
        return payload


@dataclass
class VisionRegion:
    label: str
    confidence: float
    bounding_box: VisionBoundingBox | None = None
    visible_text: str = ""
    verification: str = ""
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
        }
        if self.bounding_box is not None:
            payload["bounding_box"] = self.bounding_box.to_dict()
        if self.visible_text:
            payload["visible_text"] = self.visible_text
        if self.verification:
            payload["verification"] = self.verification
        if self.attributes:
            payload["attributes"] = {str(key)[:40]: str(value)[:120] for key, value in self.attributes.items()}
        return payload


@dataclass
class VisionObservedObject:
    summary: str = ""
    visible_text: str = ""
    dominant_colors: list[str] = field(default_factory=list)
    shapes: list[str] = field(default_factory=list)
    object_categories: list[str] = field(default_factory=list)
    object_fully_visible: bool | None = None
    bounding_box: VisionBoundingBox | None = None
    coordinate_space: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.summary:
            payload["summary"] = self.summary
        if self.visible_text:
            payload["visible_text"] = self.visible_text
        if self.dominant_colors:
            payload["dominant_colors"] = [str(item)[:40] for item in self.dominant_colors[:8] if str(item).strip()]
        if self.shapes:
            payload["shapes"] = [str(item)[:40] for item in self.shapes[:8] if str(item).strip()]
        if self.object_categories:
            payload["object_categories"] = [str(item)[:60] for item in self.object_categories[:8] if str(item).strip()]
        if self.object_fully_visible is not None:
            payload["object_fully_visible"] = bool(self.object_fully_visible)
        if self.bounding_box is not None:
            payload["bounding_box"] = self.bounding_box.to_dict()
        if self.coordinate_space:
            payload["coordinate_space"] = self.coordinate_space
        return payload


@dataclass
class VisionCropObservation:
    summary: str = ""
    visible_text: str = ""
    dominant_colors: list[str] = field(default_factory=list)
    shapes: list[str] = field(default_factory=list)
    object_categories: list[str] = field(default_factory=list)
    object_fully_visible: bool | None = None
    background_only: bool | None = None
    objects: list[VisionObservedObject] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.summary:
            payload["summary"] = self.summary
        if self.visible_text:
            payload["visible_text"] = self.visible_text
        if self.dominant_colors:
            payload["dominant_colors"] = [str(item)[:40] for item in self.dominant_colors[:8] if str(item).strip()]
        if self.shapes:
            payload["shapes"] = [str(item)[:40] for item in self.shapes[:8] if str(item).strip()]
        if self.object_categories:
            payload["object_categories"] = [str(item)[:60] for item in self.object_categories[:8] if str(item).strip()]
        if self.object_fully_visible is not None:
            payload["object_fully_visible"] = bool(self.object_fully_visible)
        if self.background_only is not None:
            payload["background_only"] = bool(self.background_only)
        if self.objects:
            payload["objects"] = [item.to_dict() for item in self.objects[:8]]
        if self.warnings:
            payload["warnings"] = [str(item)[:160] for item in self.warnings[:8] if str(item).strip()]
        return payload


@dataclass
class VisionEvidence:
    evidence_id: str
    frame_id: str
    operation: str
    success: bool
    provider: str
    model: str
    source_hash: str
    grounded: bool
    created_at: str
    expires_at: str
    description: str = ""
    extracted_text: str = ""
    ocr_blocks: list[VisionOcrBlock] = field(default_factory=list)
    visual_regions: list[VisionRegion] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    no_match: bool = False
    match_outcome: str = ""
    error_category: str = ""
    error_reason: str = ""
    safe_display_name: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    source_url: str = ""
    source_origin: str = ""
    instruction_trust: str = ""
    factual_credibility: str = ""
    captcha_suspected: bool = False
    captcha_reason: str = ""
    grounding_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_reference_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vision_operation": self.operation,
            "success": self.success,
            "evidence_id": self.evidence_id,
            "frame_id": self.frame_id,
            "provider": self.provider,
            "model": self.model,
            "source_hash": self.source_hash,
            "grounded": self.grounded,
            "description": self.description,
            "extracted_text": self.extracted_text,
            "ocr_blocks": [block.to_dict() for block in self.ocr_blocks],
            "visual_regions": [region.to_dict() for region in self.visual_regions],
            "confidence": round(float(self.confidence), 6),
            "warnings": [warning[:160] for warning in self.warnings if warning],
            "no_match": self.no_match,
            "match_outcome": self.match_outcome,
            "error_category": self.error_category,
            "error_reason": self.error_reason,
            "safe_display_name": self.safe_display_name,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "source_url": self.source_url,
            "source_origin": self.source_origin,
            "instruction_trust": self.instruction_trust,
            "factual_credibility": self.factual_credibility,
            "captcha_suspected": self.captcha_suspected,
            "captcha_reason": self.captcha_reason,
            "grounding_diagnostics": self.grounding_diagnostics,
        }
        return {key: value for key, value in payload.items() if value not in ("", [], None)}


@dataclass
class LoadedVisionImage:
    frame: VisionFrame
    original_path: str
    safe_display_name: str
    mime_type: str
    width: int
    height: int
    source_hash: str
    file_size_bytes: int
    image_bytes: bytes
    temp_copy_path: str
