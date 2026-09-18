from __future__ import annotations

import hashlib
import io
import math
import struct
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.path_policy import resolve_path
from app.brain.vision.errors import VisionImageError, VisionPolicyError
from app.brain.vision.models import LoadedVisionImage, SUPPORTED_VISION_EXTENSIONS, SUPPORTED_VISION_MIME_TYPES, VisionBoundingBox, VisionFrame


def load_local_image(path: str) -> LoadedVisionImage:
    config = get_effective_runtime_config()
    try:
        resolved = resolve_path(path, prefer_directory=False, allow_missing=False)
    except FilesystemPathError as error:
        raise VisionPolicyError(str(error)) from error
    image_path = resolved.absolute_path
    if not image_path.exists() or not image_path.is_file():
        raise VisionImageError("Image file not found.")
    max_file_size = int(config.get("vision_max_file_size", 4_194_304))
    file_size = int(image_path.stat().st_size)
    if file_size <= 0:
        raise VisionImageError("Image file is empty.")
    if file_size > max_file_size:
        raise VisionImageError("Image file is too large for safe analysis.")
    data = image_path.read_bytes()
    extension = image_path.suffix.lower()
    if extension not in SUPPORTED_VISION_EXTENSIONS:
        raise VisionImageError("Unsupported image format.")
    mime_type, width, height = _identify_image(data)
    expected_mime = SUPPORTED_VISION_MIME_TYPES.get(extension)
    if mime_type != expected_mime:
        raise VisionImageError("Image extension does not match the file signature.")
    max_width = int(config.get("vision_max_width", 4096))
    max_height = int(config.get("vision_max_height", 4096))
    max_pixels = int(config.get("vision_max_pixels", 4_194_304))
    if width <= 0 or height <= 0:
        raise VisionImageError("Image dimensions are invalid.")
    if width > max_width or height > max_height or width * height > max_pixels:
        raise VisionImageError("Image dimensions exceed safe Vision limits.")
    retention_seconds = int(config.get("vision_evidence_retention_seconds", 300))
    now = datetime.now(timezone.utc)
    source_hash = hashlib.sha256(data).hexdigest()
    temp_copy_path = _create_temp_copy(data, suffix=extension)
    frame = VisionFrame(
        frame_id=f"frame-{source_hash[:12]}",
        source_type="file",
        safe_display_name=resolved.relative_path,
        source_hash=source_hash,
        mime_type=mime_type,
        width=width,
        height=height,
        created_at=datetime.fromtimestamp(image_path.stat().st_ctime, timezone.utc).isoformat(),
        expires_at=(now + timedelta(seconds=retention_seconds)).isoformat(),
        temporary_copy=True,
        trust_classification="trusted_root_file",
        file_size_bytes=file_size,
    )
    return LoadedVisionImage(
        frame=frame,
        original_path=str(image_path),
        safe_display_name=resolved.relative_path,
        mime_type=mime_type,
        width=width,
        height=height,
        source_hash=source_hash,
        file_size_bytes=file_size,
        image_bytes=data,
        temp_copy_path=temp_copy_path,
    )


def cleanup_loaded_image(image: LoadedVisionImage | None) -> None:
    if image is None:
        return
    temp_path = Path(image.temp_copy_path) if image.temp_copy_path else None
    if temp_path is not None and temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass


def crop_loaded_image(
    image: LoadedVisionImage,
    bbox: VisionBoundingBox,
    *,
    safe_display_name: str = "browser-crop.png",
) -> LoadedVisionImage:
    if image is None:
        raise VisionImageError("Image crop source is unavailable.")
    left, top, right, bottom = bbox_to_pixel_rect(bbox, width=int(image.width), height=int(image.height))
    if right <= left or bottom <= top:
        raise VisionImageError("Crop bounding box has zero area.")
    try:
        with Image.open(io.BytesIO(image.image_bytes)) as decoded:
            cropped = decoded.crop((left, top, right, bottom))
            output = io.BytesIO()
            cropped.save(output, format="PNG")
            crop_bytes = output.getvalue()
    except Exception as error:
        raise VisionImageError("Image crop could not be created safely.") from error
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    retention_seconds = int(get_effective_runtime_config().get("vision_evidence_retention_seconds", 300))
    now = datetime.now(timezone.utc)
    source_hash = hashlib.sha256(crop_bytes).hexdigest()
    temp_copy_path = _create_temp_copy(crop_bytes, suffix=".png")
    frame = VisionFrame(
        frame_id=f"frame-{source_hash[:12]}",
        source_type=image.frame.source_type,
        safe_display_name=safe_display_name,
        source_hash=source_hash,
        mime_type="image/png",
        width=crop_width,
        height=crop_height,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=retention_seconds)).isoformat(),
        temporary_copy=True,
        trust_classification=image.frame.trust_classification,
        file_size_bytes=len(crop_bytes),
    )
    return LoadedVisionImage(
        frame=frame,
        original_path="",
        safe_display_name=safe_display_name,
        mime_type="image/png",
        width=crop_width,
        height=crop_height,
        source_hash=source_hash,
        file_size_bytes=len(crop_bytes),
        image_bytes=crop_bytes,
        temp_copy_path=temp_copy_path,
    )


def bbox_to_pixel_rect(bbox: VisionBoundingBox, *, width: int, height: int) -> tuple[int, int, int, int]:
    if min(bbox.x, bbox.y, bbox.width, bbox.height) < 0 or bbox.width <= 0 or bbox.height <= 0:
        raise VisionImageError("Crop bounding box is invalid.")
    if bbox.x + bbox.width > 1.000001 or bbox.y + bbox.height > 1.000001:
        raise VisionImageError("Crop bounding box is outside the image bounds.")
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    left = int(math.floor(bbox.x * safe_width))
    top = int(math.floor(bbox.y * safe_height))
    right = int(math.ceil((bbox.x + bbox.width) * safe_width))
    bottom = int(math.ceil((bbox.y + bbox.height) * safe_height))
    return left, top, right, bottom


def _create_temp_copy(data: bytes, *, suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(prefix="jarvis-vision-", suffix=suffix, delete=False)
    try:
        handle.write(data)
        return handle.name
    finally:
        handle.close()


def _identify_image(data: bytes) -> tuple[str, int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ("image/png", *_parse_png_dimensions(data))
    if data.startswith(b"\xff\xd8"):
        return ("image/jpeg", *_parse_jpeg_dimensions(data))
    if data.startswith(b"RIFF") and len(data) >= 16 and data[8:12] == b"WEBP":
        return ("image/webp", *_parse_webp_dimensions(data))
    raise VisionImageError("Unsupported image format.")


def _parse_png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise VisionImageError("PNG image is malformed or truncated.")
    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    if b"IEND" not in data[-32:]:
        raise VisionImageError("PNG image is truncated.")
    return width, height


def _parse_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or not data.endswith(b"\xff\xd9"):
        raise VisionImageError("JPEG image is truncated.")
    index = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if index + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[index:index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            raise VisionImageError("JPEG image is malformed or truncated.")
        if marker in sof_markers:
            if segment_length < 7:
                raise VisionImageError("JPEG image is malformed or truncated.")
            height = struct.unpack(">H", data[index + 3:index + 5])[0]
            width = struct.unpack(">H", data[index + 5:index + 7])[0]
            return width, height
        index += segment_length
    raise VisionImageError("JPEG image is malformed or truncated.")


def _parse_webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30:
        raise VisionImageError("WebP image is malformed or truncated.")
    chunk = data[12:16]
    if chunk == b"VP8X":
        if len(data) < 30:
            raise VisionImageError("WebP image is malformed or truncated.")
        width_minus_one = int.from_bytes(data[24:27], "little")
        height_minus_one = int.from_bytes(data[27:30], "little")
        return width_minus_one + 1, height_minus_one + 1
    if chunk == b"VP8 ":
        if len(data) < 30:
            raise VisionImageError("WebP image is malformed or truncated.")
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        if len(data) < 25:
            raise VisionImageError("WebP image is malformed or truncated.")
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    raise VisionImageError("WebP image is malformed or truncated.")
