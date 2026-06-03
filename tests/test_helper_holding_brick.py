import unittest

import cv2
import numpy as np

from helper_holding_brick import (
    HoldingBrickConfig,
    HoldingMaskLock,
    HoldingMaskLockConfig,
    contour_target_result_tuple,
    detect_holding_brick,
    detect_masked_target_brick_contour,
    mask_held_brick_for_target_frame,
)


def _brick_bgr():
    hsv = np.array([[[75, 180, 200]]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class TestHoldingBrickDetector(unittest.TestCase):
    def test_holding_mask_lock_survives_brief_not_held_flicker(self):
        lock = HoldingMaskLock(HoldingMaskLockConfig(release_after_misses=3))
        held = {"holding": True, "reason": "held_brick_top_nubs", "mask_regions": [(10, 0, 20, 12)]}

        first = lock.update(held)
        second = lock.update({"holding": False, "reason": "no_green_contour"})
        third = lock.update({"holding": False, "reason": "no_green_contour"})
        released = lock.update({"holding": False, "reason": "no_green_contour"})

        self.assertTrue(first["holding"])
        self.assertTrue(second["holding"])
        self.assertTrue(second["holding_mask_locked"])
        self.assertEqual(second["mask_regions"], [(10, 0, 20, 12)])
        self.assertEqual(third["holding_mask_lock_misses"], 2)
        self.assertFalse(released["holding"])
        self.assertTrue(released["holding_mask_lock_released"])

    def test_holding_mask_lock_keeps_original_mask_while_holding_jitters(self):
        lock = HoldingMaskLock(HoldingMaskLockConfig(release_after_misses=3))

        first = lock.update({"holding": True, "reason": "held", "mask_regions": [(10, 0, 20, 12)]})
        second = lock.update({"holding": True, "reason": "held", "mask_regions": [(40, 0, 20, 12)]})

        self.assertEqual(first["mask_regions"], [(10, 0, 20, 12)])
        self.assertEqual(second["mask_regions"], [(10, 0, 20, 12)])
        self.assertTrue(second["holding_mask_locked"])

    def test_holding_mask_lock_reset_clears_stale_held_mask(self):
        lock = HoldingMaskLock(HoldingMaskLockConfig(release_after_misses=3))

        lock.update({"holding": True, "reason": "held", "mask_regions": [(10, 0, 20, 12)]})
        lock.reset()
        result = lock.update({"holding": False, "reason": "clear"})

        self.assertFalse(result["holding"])
        self.assertNotIn("holding_mask_locked", result)
        self.assertEqual(result["reason"], "clear")

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

    def test_mask_held_brick_detected_below_top_limit(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[70:175, 90:550] = _brick_bgr()
        frame[220:270, 280:365] = _brick_bgr()
        result = detect_holding_brick(frame)

        masked = mask_held_brick_for_target_frame(frame, result)

        self.assertIsNotNone(masked)
        self.assertEqual(int(masked[90, 100].sum()), 0)
        self.assertGreater(int(masked[160, 320].sum()), 0)
        self.assertGreater(int(masked[240, 320].sum()), 0)

    def test_mask_held_brick_excludes_upper_bbox_without_erasing_lower_target(self):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        result = {"best": {"bbox": (90, 20, 460, 105)}}

        masked = mask_held_brick_for_target_frame(frame, result)

        self.assertIsNotNone(masked)
        self.assertEqual(int(masked[29, 320].sum()), 0)
        self.assertEqual(int(masked[90, 320].sum()), 0)
        self.assertGreater(int(masked[130, 320].sum()), 0)
        self.assertGreater(int(masked[160, 100].sum()), 0)
        self.assertGreater(int(masked[160, 540].sum()), 0)
        self.assertGreater(int(masked[180, 320].sum()), 0)

    def test_top_edge_nubs_detect_holding_and_mask_only_nubs(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        # The cropped held brick now shows as two small hanging nubs at the
        # very top edge, with a faint top bridge that should not become a
        # broad mask over the real target.
        frame[0:11, 95:345] = _brick_bgr()
        frame[7:35, 140:156] = _brick_bgr()
        frame[7:35, 292:310] = _brick_bgr()
        frame[74:100, 210:258] = _brick_bgr()

        result = detect_holding_brick(frame)
        masked = mask_held_brick_for_target_frame(frame, result)

        self.assertTrue(result["holding"])
        self.assertEqual(result["reason"], "held_brick_top_nubs")
        self.assertEqual(len(result["mask_regions"]), 2)
        self.assertEqual(int(masked[18, 148].sum()), 0)
        self.assertEqual(int(masked[18, 300].sum()), 0)
        self.assertGreater(int(masked[86, 234].sum()), 0)

    def test_broad_holding_detection_uses_top_edge_mask_when_available(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[0:38, 0:404] = _brick_bgr()
        frame[74:100, 210:258] = _brick_bgr()

        result = detect_holding_brick(frame)
        masked = mask_held_brick_for_target_frame(frame, result)

        self.assertTrue(result["holding"])
        self.assertIn("mask_regions", result)
        self.assertEqual(int(masked[18, 202].sum()), 0)
        self.assertGreater(int(masked[86, 234].sum()), 0)

    def test_thin_top_edge_sliver_does_not_count_as_holding(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[0:6, 0:320] = _brick_bgr()
        frame[74:100, 210:258] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertFalse(result["holding"])

    def test_short_broad_top_edge_does_not_count_as_holding(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[0:17, 0:404] = _brick_bgr()
        frame[74:100, 210:258] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertFalse(result["holding"])

    def test_single_top_nub_does_not_count_as_holding(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[7:35, 140:156] = _brick_bgr()
        frame[74:100, 210:258] = _brick_bgr()

        result = detect_holding_brick(frame)

        self.assertFalse(result["holding"])

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

    def test_target_contour_finds_high_stack_after_lower_camera_move(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[0:38, 100:320] = _brick_bgr()
        frame[74:104, 150:254] = _brick_bgr()

        result = detect_masked_target_brick_contour(frame)

        self.assertTrue(result["found"])
        _x, y, _w, _h = result["bbox"]
        self.assertLess(y, 90)

    def test_masked_target_contour_rejects_far_side_blob(self):
        frame = np.zeros((300, 404, 3), dtype=np.uint8)
        frame[95:158, 346:404] = _brick_bgr()

        result = detect_masked_target_brick_contour(frame)

        self.assertFalse(result["found"])
        checked = next(row for row in result["candidates"] if isinstance(row, dict) and "checks" in row)
        self.assertFalse(checked["checks"]["center_x"])

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
