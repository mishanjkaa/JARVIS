from __future__ import annotations

import io
import os
from dataclasses import dataclass

from app.brain.vision.errors import VisionCaptureUnsupportedError, VisionImageError

# DWMWA_EXTENDED_FRAME_BOUNDS: the visually accurate window rectangle on Windows 10/11,
# excluding the invisible resize-border padding that GetWindowRect otherwise includes.
_DWMWA_EXTENDED_FRAME_BOUNDS = 9

# RFC-007C is one-shot capture only: at most this many desktop.capture_screen /
# desktop.capture_window steps, combined, may appear in a single agent plan. Enforced by
# app.brain.planner.plan_validator.validate_plan alongside the existing overall plan step
# cap and the memory.recall cap, so "one-shot" is a real, tested limit rather than a
# description that happens to be true today.
MAX_DESKTOP_CAPTURES_PER_PLAN = 1


@dataclass
class WindowInfo:
    window_id: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


@dataclass
class RawCapture:
    image_bytes: bytes
    width: int
    height: int
    mime_type: str = "image/png"


def is_supported() -> bool:
    return os.name == "nt"


def _require_supported() -> None:
    if not is_supported():
        raise VisionCaptureUnsupportedError("Desktop capture is only supported on Windows.")


def _user32():
    import ctypes

    return ctypes.windll.user32


def list_windows() -> list[WindowInfo]:
    """Enumerate currently open, visible, non-minimized top-level windows with titles.

    Uses only ctypes calls into user32.dll (no pywin32 dependency). Returns no image data,
    just names and a stable-for-this-session identifier (the window handle), so a later
    request can target one precisely.
    """
    _require_supported()
    import ctypes
    from ctypes import wintypes

    user32 = _user32()
    results: list[WindowInfo] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _on_window(hwnd, _lparam):
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if not title:
                return True
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            if rect.right <= rect.left or rect.bottom <= rect.top:
                return True
            window_id = int(ctypes.cast(hwnd, ctypes.c_void_p).value or 0)
            results.append(WindowInfo(window_id=window_id, title=title, left=rect.left, top=rect.top, right=rect.right, bottom=rect.bottom))
        except Exception:
            return True
        return True

    user32.EnumWindows(EnumWindowsProc(_on_window), 0)
    return results


def find_window(window_id: int) -> WindowInfo | None:
    """Look up a window by its handle right now, for revalidation at capture time."""
    _require_supported()
    for window in list_windows():
        if window.window_id == int(window_id):
            return window
    return None


def _extended_frame_bounds(window_id: int):
    """Best-effort tighter rectangle via DWM, excluding the invisible resize border."""
    import ctypes
    from ctypes import wintypes

    try:
        dwmapi = ctypes.windll.dwmapi
    except OSError:
        return None
    rect = wintypes.RECT()
    result = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(window_id),
        ctypes.c_uint32(_DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if result != 0:
        return None
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def _encode_png(image) -> RawCapture:
    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")
    rgb_image.save(buffer, format="PNG")
    width, height = rgb_image.size
    return RawCapture(image_bytes=buffer.getvalue(), width=width, height=height)


def capture_full_desktop() -> RawCapture:
    _require_supported()
    from PIL import ImageGrab

    try:
        image = ImageGrab.grab()
    except Exception as error:
        raise VisionImageError("Desktop capture failed.") from error
    return _encode_png(image)


def capture_window(window_id: int) -> RawCapture:
    _require_supported()
    from PIL import ImageGrab

    window = find_window(window_id)
    if window is None:
        raise VisionImageError("The target window is no longer available.")
    bbox = _extended_frame_bounds(window_id) or (window.left, window.top, window.right, window.bottom)
    try:
        image = ImageGrab.grab(bbox=bbox)
    except Exception as error:
        raise VisionImageError("Window capture failed.") from error
    return _encode_png(image)
