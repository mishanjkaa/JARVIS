from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib
from pathlib import Path


WIDTH = 760
HEIGHT = 280
BACKGROUND = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 32, 32)
BLUE = (40, 90, 220)
FONT_SCALE = 8
FIXTURE_TEXT = "VISION TEST 42"
TEXT_X = 36
TEXT_Y = 28
MIN_MARGIN = 24
MIN_OBJECT_GAP = 28
CIRCLE_CENTER_X = 176
CIRCLE_CENTER_Y = 184
CIRCLE_RADIUS = 52
RECTANGLE_X = 508
RECTANGLE_Y = 124
RECTANGLE_WIDTH = 112
RECTANGLE_HEIGHT = 112

_FONT: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "2": ("11111", "00001", "00001", "11111", "10000", "10000", "11111"),
    "4": ("10001", "10001", "10001", "11111", "00001", "00001", "00001"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "V": ("10001", "10001", "10001", "10001", "01010", "01010", "00100"),
}


@dataclass(frozen=True)
class Bounds:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class FixtureLayout:
    width: int
    height: int
    text: str
    text_bounds: Bounds
    circle_bounds: Bounds
    rectangle_bounds: Bounds


def main() -> None:
    output_path = Path(__file__).with_name("vision_sample.png")
    output_path.write_bytes(generate_fixture_png_bytes())


def generate_fixture_png_bytes(*, text: str = FIXTURE_TEXT) -> bytes:
    image = generate_fixture_image(text=text)
    return _encode_png(image)


def generate_fixture_image(*, text: str = FIXTURE_TEXT) -> list[list[tuple[int, int, int]]]:
    layout = build_fixture_layout(text=text)
    validate_fixture_layout(layout)
    image = [[BACKGROUND for _ in range(layout.width)] for _ in range(layout.height)]
    _draw_text(image, layout.text, x=TEXT_X, y=TEXT_Y, color=BLACK, scale=FONT_SCALE)
    _draw_circle(image, center_x=CIRCLE_CENTER_X, center_y=CIRCLE_CENTER_Y, radius=CIRCLE_RADIUS, color=RED)
    _draw_rectangle(
        image,
        x=RECTANGLE_X,
        y=RECTANGLE_Y,
        width=RECTANGLE_WIDTH,
        height=RECTANGLE_HEIGHT,
        color=BLUE,
    )
    return image


def build_fixture_layout(*, text: str = FIXTURE_TEXT) -> FixtureLayout:
    origin_text_bounds = measure_text_bounds(text, scale=FONT_SCALE)
    text_bounds = _translate_bounds(origin_text_bounds, dx=TEXT_X, dy=TEXT_Y)
    return FixtureLayout(
        width=WIDTH,
        height=HEIGHT,
        text=text,
        text_bounds=text_bounds,
        circle_bounds=Bounds(
            left=CIRCLE_CENTER_X - CIRCLE_RADIUS,
            top=CIRCLE_CENTER_Y - CIRCLE_RADIUS,
            right=CIRCLE_CENTER_X + CIRCLE_RADIUS + 1,
            bottom=CIRCLE_CENTER_Y + CIRCLE_RADIUS + 1,
        ),
        rectangle_bounds=Bounds(
            left=RECTANGLE_X,
            top=RECTANGLE_Y,
            right=RECTANGLE_X + RECTANGLE_WIDTH,
            bottom=RECTANGLE_Y + RECTANGLE_HEIGHT,
        ),
    )


def measure_text_bounds(text: str, *, scale: int) -> Bounds:
    cursor_x = 0
    min_x: int | None = None
    min_y: int | None = None
    max_x = 0
    max_y = 0
    for char in text.upper():
        glyph = _FONT.get(char)
        if glyph is None:
            cursor_x += 6 * scale
            continue
        for row_index, row in enumerate(glyph):
            for col_index, pixel in enumerate(row):
                if pixel != "1":
                    continue
                pixel_left = cursor_x + col_index * scale
                pixel_top = row_index * scale
                pixel_right = pixel_left + scale
                pixel_bottom = pixel_top + scale
                min_x = pixel_left if min_x is None else min(min_x, pixel_left)
                min_y = pixel_top if min_y is None else min(min_y, pixel_top)
                max_x = max(max_x, pixel_right)
                max_y = max(max_y, pixel_bottom)
        cursor_x += 6 * scale
    if min_x is None or min_y is None:
        return Bounds(0, 0, 0, 0)
    return Bounds(min_x, min_y, max_x, max_y)


def count_text_lit_pixels(text: str) -> int:
    total = 0
    for char in text.upper():
        glyph = _FONT.get(char)
        if glyph is None:
            continue
        total += sum(1 for row in glyph for pixel in row if pixel == "1")
    return total


def validate_fixture_layout(layout: FixtureLayout) -> None:
    _require_bounds_inside_image(layout.text_bounds, width=layout.width, height=layout.height, label="text")
    _require_bounds_inside_image(layout.circle_bounds, width=layout.width, height=layout.height, label="circle")
    _require_bounds_inside_image(layout.rectangle_bounds, width=layout.width, height=layout.height, label="rectangle")
    _require_margin(layout.text_bounds.left >= MIN_MARGIN, "text left margin is too small")
    _require_margin(layout.text_bounds.top >= MIN_MARGIN, "text top margin is too small")
    _require_margin(layout.width - layout.text_bounds.right >= MIN_MARGIN, "text right margin is too small")
    _require_margin(layout.circle_bounds.left >= MIN_MARGIN, "circle left margin is too small")
    _require_margin(layout.height - layout.circle_bounds.bottom >= MIN_MARGIN, "circle bottom margin is too small")
    _require_margin(layout.width - layout.rectangle_bounds.right >= MIN_MARGIN, "rectangle right margin is too small")
    _require_margin(layout.height - layout.rectangle_bounds.bottom >= MIN_MARGIN, "rectangle bottom margin is too small")
    _require_margin(not _bounds_overlap(layout.text_bounds, layout.circle_bounds, padding=MIN_OBJECT_GAP), "text overlaps the circle layout region")
    _require_margin(not _bounds_overlap(layout.text_bounds, layout.rectangle_bounds, padding=MIN_OBJECT_GAP), "text overlaps the rectangle layout region")
    _require_margin(not _bounds_overlap(layout.circle_bounds, layout.rectangle_bounds, padding=MIN_OBJECT_GAP), "circle overlaps the rectangle layout region")
    _require_margin(layout.circle_bounds.left < layout.rectangle_bounds.left, "red circle must remain to the left of the blue shape")


def _require_bounds_inside_image(bounds: Bounds, *, width: int, height: int, label: str) -> None:
    if bounds.left < 0 or bounds.top < 0 or bounds.right > width or bounds.bottom > height:
        raise ValueError(f"{label} does not fit inside the fixture image")


def _require_margin(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bounds_overlap(first: Bounds, second: Bounds, *, padding: int = 0) -> bool:
    return not (
        first.right + padding <= second.left
        or second.right + padding <= first.left
        or first.bottom + padding <= second.top
        or second.bottom + padding <= first.top
    )


def _translate_bounds(bounds: Bounds, *, dx: int, dy: int) -> Bounds:
    return Bounds(
        left=bounds.left + dx,
        top=bounds.top + dy,
        right=bounds.right + dx,
        bottom=bounds.bottom + dy,
    )


def _draw_text(image: list[list[tuple[int, int, int]]], text: str, *, x: int, y: int, color: tuple[int, int, int], scale: int) -> None:
    cursor_x = x
    for char in text.upper():
        glyph = _FONT.get(char)
        if glyph is None:
            cursor_x += 6 * scale
            continue
        for row_index, row in enumerate(glyph):
            for col_index, pixel in enumerate(row):
                if pixel != "1":
                    continue
                _fill_rect(
                    image,
                    x=cursor_x + col_index * scale,
                    y=y + row_index * scale,
                    width=scale,
                    height=scale,
                    color=color,
                )
        cursor_x += 6 * scale


def _draw_circle(image: list[list[tuple[int, int, int]]], *, center_x: int, center_y: int, radius: int, color: tuple[int, int, int]) -> None:
    radius_sq = radius * radius
    for row in range(max(0, center_y - radius), min(len(image), center_y + radius + 1)):
        for col in range(max(0, center_x - radius), min(len(image[0]), center_x + radius + 1)):
            dx = col - center_x
            dy = row - center_y
            if dx * dx + dy * dy <= radius_sq:
                image[row][col] = color


def _draw_rectangle(
    image: list[list[tuple[int, int, int]]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    _fill_rect(image, x=x, y=y, width=width, height=height, color=color)


def _fill_rect(
    image: list[list[tuple[int, int, int]]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    for row in range(max(0, y), min(len(image), y + height)):
        for col in range(max(0, x), min(len(image[0]), x + width)):
            image[row][col] = color


def _encode_png(image: list[list[tuple[int, int, int]]]) -> bytes:
    if not image or not image[0]:
        raise ValueError("fixture image cannot be empty")
    height = len(image)
    width = len(image[0])
    raw_rows = bytearray()
    for row in image:
        raw_rows.append(0)
        for red, green, blue in row:
            raw_rows.extend((red, green, blue))
    compressed = zlib.compress(bytes(raw_rows), level=9)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


if __name__ == "__main__":
    main()
