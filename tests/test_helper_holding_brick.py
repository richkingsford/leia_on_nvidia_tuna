import unittest

import cv2
import numpy as np

from helper_holding_brick import (
    HoldingBrickConfig,
    contour_target_result_tuple,
    detect_holding_brick,
    detect_masked_target_brick_contour,
    mask_held_brick_for_target_frame,
)


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

    def test_rejects_upper_target_brick_below_prong_center_band(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[144:270, 236:440] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertFalse(result["holding"])
        self.assertFalse(result["checks"]["center_y"])

    def test_missing_frame_is_false_without_error(self):
        self.assertFalse(detect_holding_brick(None)["holding"])

    def test_config_can_narrow_roi(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[70:175, 90:550] = _brick_bgr()
        cfg = HoldingBrickConfig(roi_x_min_ratio=0.45, roi_x_max_ratio=0.55, min_width_ratio=0.05)

        result = detect_holding_brick(frame, cfg)

        self.assertTrue(result["holding"])

    def test_mask_held_brick_never_extends_below_top_limit(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[20:95, 90:550] = _brick_bgr()
        frame[220:270, 280:365] = _brick_bgr()
        result = detect_holding_brick(frame)

        masked = mask_held_brick_for_target_frame(frame, result)

        self.assertIsNotNone(masked)
        self.assertEqual(int(masked[45, 100].sum()), 0)
        self.assertGreater(int(masked[60, 100].sum()), 0)
        self.assertGreater(int(masked[70, 100].sum()), 0)
        self.assertGreater(int(masked[240, 320].sum()), 0)

    def test_mask_held_brick_lifts_center_bottom_but_keeps_outer_nubs_masked(self):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        result = {"best": {"bbox": (90, 20, 460, 105)}}

        masked = mask_held_brick_for_target_frame(frame, result)

        self.assertIsNotNone(masked)
        self.assertEqual(int(masked[29, 320].sum()), 0)
        self.assertGreater(int(masked[30, 320].sum()), 0)
        self.assertEqual(int(masked[59, 100].sum()), 0)
        self.assertEqual(int(masked[59, 540].sum()), 0)
        self.assertGreater(int(masked[60, 100].sum()), 0)
        self.assertGreater(int(masked[60, 540].sum()), 0)

    def test_masked_target_contour_uses_full_wide_lower_brick(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[40:180, 40:600] = _brick_bgr()
        frame[245:305, 160:500] = _brick_bgr()
        holding = detect_holding_brick(frame)
        masked = mask_held_brick_for_target_frame(frame, holding)

        result = detect_masked_target_brick_contour(masked)

        self.assertTrue(result["found"])
        x, y, w, h = result["bbox"]
        self.assertLessEqual(x, 165)
        self.assertGreaterEqual(x + w, 495)
        self.assertGreaterEqual(w, 330)
        self.assertGreater(float(result["confidence_pct"]), 60.0)

    def test_masked_target_contour_rejects_far_side_blob(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[95:158, 346:404] = _brick_bgr()

        result = detect_masked_target_brick_contour(frame)

        self.assertFalse(result["found"])
        self.assertFalse(result["candidates"][0]["checks"]["center_x"])

    def test_contour_target_result_tuple_matches_detector_contract(self):
        row = {
            "found": True,
            "dist_mm": 42.0,
            "x_mm": -3.0,
            "y_mm": 2.0,
            "confidence_pct": 88.0,
        }

        result = contour_target_result_tuple(row)

        self.assertEqual(result, (True, 0.0, 42.0, -3.0, 88.0, 2.0, False, False))


if __name__ == "__main__":
    unittest.main()
