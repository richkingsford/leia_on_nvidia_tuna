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
                "03929C",
                "068A9C",
                "028991",
                "0F949B",
                "018C96",
                "36BAC6",
                "007C81",
                "00888C",
                "068A90",
                "007378",
                "39B7C1",
                "2AB0BA",
                "31B0BB",
            ),
        )

    def test_every_manual_cyan_hex_fits_inside_balanced_hsv_range(self):
        lower = detector.CYAN_HSV_BALANCED_LOWER
        upper = detector.CYAN_HSV_BALANCED_UPPER
        for hex_code in detector.CYAN_SHADE_HEXES:
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
