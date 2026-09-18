from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


VIEWPORT_WIDTH = 800
VIEWPORT_HEIGHT = 600
CROP_WIDTH = 280
CROP_HEIGHT = 280
BACKGROUND = (255, 255, 255)
BLACK = (24, 24, 24)
GRAY = (90, 90, 90)
RED = (220, 32, 32)

POSITIVE_CIRCLE_BOX = (280, 160, 500, 380)
NEGATIVE_HEADING = "Example Domain"
NEGATIVE_BODY = "No red circle here."
UNRELATED_TEXT = "Example Domain"


def main() -> None:
    fixture_dir = Path(__file__).resolve().parent
    fixture_dir.joinpath("browser_visual_negative.png").write_bytes(_encode(_negative_fixture()))
    fixture_dir.joinpath("browser_visual_positive_red_circle.png").write_bytes(_encode(_positive_fixture()))
    fixture_dir.joinpath("browser_visual_partial_red_patch.png").write_bytes(_encode(_partial_patch_fixture()))
    fixture_dir.joinpath("browser_visual_unrelated_text.png").write_bytes(_encode(_unrelated_text_fixture()))


def _negative_fixture() -> Image.Image:
    image = Image.new("RGB", (VIEWPORT_WIDTH, VIEWPORT_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((120, 120), NEGATIVE_HEADING, fill=BLACK)
    draw.text((120, 180), NEGATIVE_BODY, fill=GRAY)
    return image


def _positive_fixture() -> Image.Image:
    image = _negative_fixture()
    draw = ImageDraw.Draw(image)
    draw.ellipse(POSITIVE_CIRCLE_BOX, fill=RED, outline=RED)
    draw.text((120, 420), "Visible shape fixture", fill=BLACK)
    return image


def _partial_patch_fixture() -> Image.Image:
    image = Image.new("RGB", (CROP_WIDTH, CROP_HEIGHT), RED)
    draw = ImageDraw.Draw(image)
    draw.text((18, 22), "Interior crop", fill=(255, 255, 255))
    return image


def _unrelated_text_fixture() -> Image.Image:
    image = Image.new("RGB", (CROP_WIDTH, CROP_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((32, 110), UNRELATED_TEXT, fill=BLACK)
    return image


def _encode(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
