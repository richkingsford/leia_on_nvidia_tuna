import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_detector_yolo as detector


class TestHelperBrickDetectorYoloCyanPalette(unittest.TestCase):
    def test_manual_cyan_hex_palette_matches_expected_values(self):
        self.assertEqual(
            detector.CYAN_SHADE_HEXES,
            (
                "13A561",
                "3ABD8D",
                "0A5839",
                "A8B6AA",
                "449868",
                "2B824A",
                "51A276",
                "3D8D62",
            ),
        )

    def test_every_saturated_manual_cyan_hex_fits_inside_balanced_hsv_range(self):
        lower = detector.CYAN_HSV_BALANCED_LOWER
        upper = detector.CYAN_HSV_BALANCED_UPPER
        washed_highlight_hexes = {"A8B6AA"}
        for hex_code in detector.CYAN_SHADE_HEXES:
            if hex_code in washed_highlight_hexes:
                continue
            hue, sat, val = detector._hex_to_opencv_hsv(hex_code)
            self.assertGreaterEqual(hue, lower[0], hex_code)
            self.assertLessEqual(hue, upper[0], hex_code)
            self.assertGreaterEqual(sat, lower[1], hex_code)
            self.assertLessEqual(sat, upper[1], hex_code)
            self.assertGreaterEqual(val, lower[2], hex_code)
            self.assertLessEqual(val, upper[2], hex_code)

    def test_every_manual_cyan_hex_fits_inside_wide_hsv_range(self):
        lower = detector.CYAN_HSV_WIDE_LOWER
        upper = detector.CYAN_HSV_WIDE_UPPER
        for hex_code in detector.CYAN_SHADE_HEXES:
            hue, sat, val = detector._hex_to_opencv_hsv(hex_code)
            self.assertGreaterEqual(hue, lower[0], hex_code)
            self.assertLessEqual(hue, upper[0], hex_code)
            self.assertGreaterEqual(sat, lower[1], hex_code)
            self.assertLessEqual(sat, upper[1], hex_code)
            self.assertGreaterEqual(val, lower[2], hex_code)
            self.assertLessEqual(val, upper[2], hex_code)

    def test_lights_on_saturated_samples_fit_default_tight_hsv_range(self):
        lower = detector.CYAN_HSV_TIGHT_LOWER
        upper = detector.CYAN_HSV_TIGHT_UPPER
        for hex_code in ("449868", "2B824A", "51A276", "3D8D62"):
            hue, sat, val = detector._hex_to_opencv_hsv(hex_code)
            self.assertGreaterEqual(hue, lower[0], hex_code)
            self.assertLessEqual(hue, upper[0], hex_code)
            self.assertGreaterEqual(sat, lower[1], hex_code)
            self.assertLessEqual(sat, upper[1], hex_code)
            self.assertGreaterEqual(val, lower[2], hex_code)
            self.assertLessEqual(val, upper[2], hex_code)

    def test_cyan_hsv_ranges_are_nested_from_tight_to_wide(self):
        for idx in range(3):
            self.assertLessEqual(detector.CYAN_HSV_WIDE_LOWER[idx], detector.CYAN_HSV_BALANCED_LOWER[idx])
            self.assertLessEqual(detector.CYAN_HSV_BALANCED_LOWER[idx], detector.CYAN_HSV_TIGHT_LOWER[idx])
            self.assertGreaterEqual(detector.CYAN_HSV_WIDE_UPPER[idx], detector.CYAN_HSV_BALANCED_UPPER[idx])
            self.assertGreaterEqual(detector.CYAN_HSV_BALANCED_UPPER[idx], detector.CYAN_HSV_TIGHT_UPPER[idx])

    def test_default_detector_hsv_constants_use_balanced_palette_range(self):
        self.assertEqual(detector.CYAN_HSV_LOWER.tolist(), list(detector.CYAN_HSV_BALANCED_LOWER))
        self.assertEqual(detector.CYAN_HSV_UPPER.tolist(), list(detector.CYAN_HSV_BALANCED_UPPER))


if __name__ == "__main__":
    unittest.main()
