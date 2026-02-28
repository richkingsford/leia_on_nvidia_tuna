import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from brick_detector_yolo import BrickDetector


class TestBrickDetectorYoloCenterLock(unittest.TestCase):
    def _detector_stub(self):
        det = BrickDetector.__new__(BrickDetector)
        det.frame_w = 640
        det.frame_h = 480
        det.focal_px = 100.0
        det.camera_center_offset_px = 0.0
        det._smooth_alpha = 1.0
        det._prev_angle = None
        det._prev_dist = None
        det._prev_offset = None
        det._prev_offset_y = None
        det._hsv_enabled = True
        det._center_lock_enabled = True
        det._center_lock_radius_px = 120.0
        det._center_switch_margin_px = 20.0
        det._center_partial_penalty = 2.0
        det._center_axis_weight_x = 1.0
        det._center_axis_weight_y = 1.0
        det._center_lock_prev_center = None
        det.debug = False
        det.last_status = "idle"
        det.last_primary_confidence = 0.0
        return det

    def test_select_center_brick_prefers_closest_xy_center(self):
        det = self._detector_stub()
        bricks = [
            {"center_x": 250.0, "center_y": 240.0, "partial": False},
            {"center_x": 320.0, "center_y": 240.0, "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[1])

    def test_select_center_brick_uses_lock_hysteresis_when_scores_are_close(self):
        det = self._detector_stub()
        det._center_lock_prev_center = (282.0, 240.0)
        # Brick[0] is near previous lock but farther from frame center.
        # Brick[1] is slightly better center score, but not enough to switch.
        bricks = [
            {"center_x": 284.0, "center_y": 240.0, "partial": False},
            {"center_x": 300.0, "center_y": 240.0, "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[0])

    def test_select_center_brick_switches_when_center_is_much_better(self):
        det = self._detector_stub()
        det._center_lock_prev_center = (282.0, 240.0)
        bricks = [
            {"center_x": 284.0, "center_y": 240.0, "partial": False},
            {"center_x": 320.0, "center_y": 240.0, "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[1])

    def test_fallback_path_selects_center_box_not_highest_confidence(self):
        det = self._detector_stub()
        det._hsv_enabled = False
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        det._estimate_distance = lambda _bbox_h: 200.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # First box has higher confidence but is farther from center than second.
        bricks = [
            (100, 200, 140, 240, 0.95),   # far left
            (300, 200, 340, 240, 0.60),   # centered
        ]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        # offset_x should be centered box -> ~0
        self.assertAlmostEqual(result[3], 0.0, places=6)
        self.assertAlmostEqual(result[4], 60.0, places=6)


if __name__ == "__main__":
    unittest.main()
