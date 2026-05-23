import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_brick_detector_yolo import (
    BrickDetector,
    CYAN_HSV_WIDE_LOWER,
    CYAN_HSV_WIDE_UPPER,
)


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
        det._hsv_lower = np.array(CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
        det._hsv_upper = np.array(CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
        det._hsv_erode_iterations = 0
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
        det.last_partial_count = 0
        det.last_partial_labels = []
        det.last_primary_partial_kind = None
        det.last_primary_partial_label = None
        det.log = type(
            "_LogStub",
            (),
            {
                "info": staticmethod(lambda *_args, **_kwargs: None),
                "error": staticmethod(lambda *_args, **_kwargs: None),
            },
        )()
        return det

    def test_select_center_brick_prefers_closest_xy_center(self):
        det = self._detector_stub()
        bricks = [
            {"center_x": 250.0, "center_y": 240.0, "partial": False},
            {"center_x": 320.0, "center_y": 240.0, "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[1])

    def test_select_center_brick_ignores_cutout_anchor_for_face_midpoint(self):
        det = self._detector_stub()
        bricks = [
            {
                "center_x": 320.0,
                "center_y": 340.0,
                "bbox": (300.0, 320.0, 40.0, 40.0),
                "selection_anchor_x": 320.0,
                "selection_anchor_y": 242.0,
                "partial": False,
            },
            {"center_x": 320.0, "center_y": 260.0, "bbox": (300.0, 240.0, 40.0, 40.0), "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[1])

    def test_hsv_telemetry_uses_face_midpoint_not_slot_anchor(self):
        det = self._detector_stub()
        det._conf_gate_pct = 0.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 200.0
        depth_anchor = {}

        def _prefer_depth(fallback, cx, cy, bbox=None):
            depth_anchor["cx"] = cx
            depth_anchor["cy"] = cy
            depth_anchor["bbox"] = bbox
            return fallback

        det._prefer_depth_distance = _prefer_depth
        primary = {
            "center_x": 320.0,
            "center_y": 240.0,
            "bbox": (340.0, 320.0, 40.0, 40.0),
            "contour": None,
            "rect": None,
            "area": 1600.0,
            "partial": False,
            "partial_kind": None,
            "partial_label": None,
            "partial_edges": {},
            "shape_profile": "full",
            "negative_cutout_polygons": [],
            "selection_anchor_x": 320.0,
            "selection_anchor_y": 240.0,
        }
        det._segment_bricks_hsv = lambda *_args, **_kwargs: [primary]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = BrickDetector._process_bricks(det, frame, [(0, 0, 640, 480, 0.9)])

        self.assertTrue(result[0])
        self.assertAlmostEqual(depth_anchor["cx"], 360.0, places=6)
        self.assertAlmostEqual(depth_anchor["cy"], 340.0, places=6)
        self.assertAlmostEqual(result[3], 80.0, places=6)
        self.assertAlmostEqual(result[5], 100.0, places=6)

    def test_select_center_brick_prefers_closest_midpoint_across_rows(self):
        det = self._detector_stub()
        bricks = [
            {"center_x": 318.0, "center_y": 294.0, "partial": False},
            {"center_x": 374.0, "center_y": 246.0, "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[0])

    def test_select_center_brick_keeps_previous_lock_when_midpoints_are_close(self):
        det = self._detector_stub()
        det._center_lock_prev_center = (282.0, 240.0)
        # Brick[0] is near previous lock; Brick[1] is closer to crosshairs,
        # but not enough to switch away from the locked shrink-wrap target.
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

    def test_select_center_brick_prefers_closest_midpoint_over_higher_confidence(self):
        det = self._detector_stub()
        bricks = [
            {"center_x": 220.0, "center_y": 240.0, "source_conf": 0.95, "partial": False},
            {"center_x": 318.0, "center_y": 242.0, "source_conf": 0.60, "partial": False},
        ]
        selected = BrickDetector._select_center_brick(det, bricks, 640, 480)
        self.assertIs(selected, bricks[1])

    def test_center_stack_filter_discards_center_background_singleton(self):
        det = self._detector_stub()
        candidates = [
            {"center_x": 285.0, "center_y": 210.0, "bbox": (260.0, 190.0, 50.0, 40.0), "partial": False},
            {"center_x": 285.0, "center_y": 270.0, "bbox": (260.0, 250.0, 50.0, 40.0), "partial": False},
            {"center_x": 320.0, "center_y": 240.0, "bbox": (310.0, 230.0, 20.0, 20.0), "partial": False},
        ]

        filtered = BrickDetector._filter_candidates_to_center_stack(det, candidates, 640, 480)

        self.assertEqual(filtered, candidates[:2])
        selected = BrickDetector._select_center_brick(det, filtered, 640, 480)
        self.assertIn(selected, candidates[:2])

    def test_center_stack_filter_switches_from_stale_lock_to_center_stack(self):
        det = self._detector_stub()
        det._center_lock_prev_center = (285.0, 210.0)
        candidates = [
            {"center_x": 285.0, "center_y": 210.0, "bbox": (260.0, 190.0, 50.0, 40.0), "partial": False},
            {"center_x": 285.0, "center_y": 270.0, "bbox": (260.0, 250.0, 50.0, 40.0), "partial": False},
            {"center_x": 320.0, "center_y": 220.0, "bbox": (300.0, 200.0, 40.0, 40.0), "partial": False},
            {"center_x": 320.0, "center_y": 270.0, "bbox": (300.0, 250.0, 40.0, 40.0), "partial": False},
        ]

        filtered = BrickDetector._filter_candidates_to_center_stack(det, candidates, 640, 480)

        self.assertEqual(filtered, candidates[2:])

    def test_center_stack_filter_keeps_lock_when_stack_is_still_centered(self):
        det = self._detector_stub()
        det._center_lock_prev_center = (305.0, 220.0)
        candidates = [
            {"center_x": 305.0, "center_y": 220.0, "bbox": (295.0, 200.0, 20.0, 40.0), "partial": False},
            {"center_x": 305.0, "center_y": 270.0, "bbox": (295.0, 250.0, 20.0, 40.0), "partial": False},
            {"center_x": 320.0, "center_y": 220.0, "bbox": (310.0, 200.0, 20.0, 40.0), "partial": False},
            {"center_x": 320.0, "center_y": 270.0, "bbox": (310.0, 250.0, 20.0, 40.0), "partial": False},
        ]

        filtered = BrickDetector._filter_candidates_to_center_stack(det, candidates, 640, 480)

        self.assertEqual(filtered, candidates[:2])

    def test_fallback_path_selects_center_box_not_highest_confidence(self):
        det = self._detector_stub()
        det._hsv_enabled = False
        det._conf_gate_pct = 0.0
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 200.0

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

    def test_hsv_search_stops_after_center_box_finds_candidates(self):
        det = self._detector_stub()
        det._conf_gate_pct = 0.0
        det.last_max_confidence = 1.0
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 200.0
        det._prefer_depth_distance = lambda fallback, *_args, **_kwargs: fallback
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._detect_pink_dot_in_brick = lambda *_args, **_kwargs: (False, None, None)
        det._dist_from_triangle_span = lambda *_args, **_kwargs: None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        calls = []
        centered_candidate = {
            "center_x": 320.0,
            "center_y": 240.0,
            "bbox": (300.0, 220.0, 40.0, 40.0),
            "contour": None,
            "rect": None,
            "area": 1600.0,
            "partial": False,
            "partial_kind": None,
            "partial_label": None,
            "partial_edges": {},
            "shape_profile": "full",
            "shape_match_score": None,
            "negative_cutout_polygons": [],
        }

        def _segment(_frame, x1, y1, x2, y2):
            calls.append((x1, y1, x2, y2))
            if (x1, y1, x2, y2) == (295, 210, 345, 270):
                return [centered_candidate]
            self.fail("HSV search should stop before scanning the background box")

        det._segment_bricks_hsv = _segment

        bricks = [
            (40, 200, 100, 260, 0.95),
            (295, 210, 345, 270, 0.40),
        ]
        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertEqual(calls, [(295, 210, 345, 270)])
        self.assertAlmostEqual(result[3], 0.0, places=6)

    def test_hsv_stack_bbox_shrink_wraps_green_pixels_inside_loose_candidate(self):
        det = self._detector_stub()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        low_sat_green = cv2.cvtColor(
            np.uint8([[[76, 30, 120]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0].tolist()
        frame[70:95, 15:110] = low_sat_green
        cv2.rectangle(frame, (42, 18), (74, 66), (97, 165, 19), thickness=cv2.FILLED)
        candidate = {
            "center_x": 60.0,
            "center_y": 50.0,
            "bbox": (12.0, 8.0, 96.0, 86.0),
            "partial": False,
        }

        bbox = BrickDetector._hsv_stack_tight_bbox(det, frame, [candidate])

        self.assertEqual(bbox, (42, 18, 75, 67))

    def test_partial_info_prioritizes_top_and_bottom_edges(self):
        det = self._detector_stub()

        top_info = BrickDetector._partial_info_for_crop_bbox(det, 0, 0, 30, 20, 60, 60)
        bottom_info = BrickDetector._partial_info_for_crop_bbox(det, 10, 40, 30, 20, 60, 60)
        left_info = BrickDetector._partial_info_for_crop_bbox(det, 0, 10, 20, 20, 60, 60)

        self.assertTrue(bool(top_info.get("partial")))
        self.assertEqual(top_info.get("kind"), "top_half")
        self.assertEqual(top_info.get("label"), "TOP HALF")
        self.assertTrue(bool(bottom_info.get("partial")))
        self.assertEqual(bottom_info.get("kind"), "bottom_half")
        self.assertEqual(bottom_info.get("label"), "LOWER PARTIAL")
        self.assertTrue(bool(left_info.get("partial")))
        self.assertEqual(left_info.get("kind"), "left_partial")
        self.assertEqual(left_info.get("label"), "LEFT PARTIAL")


if __name__ == "__main__":
    unittest.main()
