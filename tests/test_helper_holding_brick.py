import unittest

import cv2
import numpy as np

from helper_holding_brick import HoldingBrickConfig, detect_holding_brick


def _brick_bgr():
    hsv = np.array([[[75, 180, 200]]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class TestHoldingBrickDetector(unittest.TestCase):
    def test_detects_large_green_region_in_upper_prong_roi(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[70:175, 90:550] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertTrue(result["holding"])
        self.assertEqual(result["reason"], "held_brick_detected")
        self.assertTrue(result["checks"]["area"])
        self.assertTrue(result["checks"]["width"])

    def test_ignores_floor_brick_below_prong_roi(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[320:370, 285:355] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertFalse(result["holding"])

    def test_thresholds_reject_tiny_top_speckle(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[70:78, 120:150] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertFalse(result["holding"])
        self.assertIn(result["reason"], {"below_threshold", "no_green_contour"})

    def test_missing_frame_is_false_without_error(self):
        self.assertFalse(detect_holding_brick(None)["holding"])

    def test_config_can_narrow_roi(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[70:175, 90:550] = _brick_bgr()
        cfg = HoldingBrickConfig(roi_x_min_ratio=0.45, roi_x_max_ratio=0.55, min_width_ratio=0.05)

        result = detect_holding_brick(frame, cfg)

        self.assertTrue(result["holding"])


if __name__ == "__main__":
    unittest.main()
