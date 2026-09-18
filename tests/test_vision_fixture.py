from __future__ import annotations

import unittest
from pathlib import Path

from tests.fixtures import generate_vision_fixture as fixture


class VisionFixtureTests(unittest.TestCase):
    def test_measured_text_width_fits_inside_image(self) -> None:
        layout = fixture.build_fixture_layout()
        self.assertLessEqual(layout.text_bounds.right, layout.width)
        self.assertGreaterEqual(layout.text_bounds.left, fixture.MIN_MARGIN)
        self.assertGreaterEqual(layout.width - layout.text_bounds.right, fixture.MIN_MARGIN)

    def test_all_intended_elements_fit_inside_image_bounds(self) -> None:
        layout = fixture.build_fixture_layout()
        for bounds in (layout.text_bounds, layout.circle_bounds, layout.rectangle_bounds):
            with self.subTest(bounds=bounds):
                self.assertGreaterEqual(bounds.left, 0)
                self.assertGreaterEqual(bounds.top, 0)
                self.assertLessEqual(bounds.right, layout.width)
                self.assertLessEqual(bounds.bottom, layout.height)

    def test_fixture_objects_do_not_overlap_unintentionally(self) -> None:
        layout = fixture.build_fixture_layout()
        self.assertLess(layout.text_bounds.bottom + fixture.MIN_OBJECT_GAP, layout.circle_bounds.top)
        self.assertLess(layout.text_bounds.bottom + fixture.MIN_OBJECT_GAP, layout.rectangle_bounds.top)
        self.assertLess(layout.circle_bounds.right + fixture.MIN_OBJECT_GAP, layout.rectangle_bounds.left)

    def test_committed_png_is_byte_for_byte_reproducible(self) -> None:
        expected = fixture.generate_fixture_png_bytes()
        actual = Path("tests/fixtures/vision_sample.png").read_bytes()
        self.assertEqual(actual, expected)

    def test_complete_ground_truth_text_is_represented_by_glyph_geometry(self) -> None:
        image = fixture.generate_fixture_image()
        black_pixels = sum(1 for row in image for pixel in row if pixel == fixture.BLACK)
        expected_black_pixels = fixture.count_text_lit_pixels(fixture.FIXTURE_TEXT) * fixture.FONT_SCALE * fixture.FONT_SCALE
        self.assertEqual(black_pixels, expected_black_pixels)

    def test_oversized_text_fails_validation_instead_of_clipping(self) -> None:
        with self.assertRaisesRegex(ValueError, "text right margin is too small|text does not fit inside the fixture image"):
            fixture.generate_fixture_png_bytes(text=f"{fixture.FIXTURE_TEXT} {fixture.FIXTURE_TEXT}")


if __name__ == "__main__":
    unittest.main()
