import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_brick_detector_yolo import BrickDetector


class TestBrickDetectorYoloYAxis(unittest.TestCase):
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
        det.debug = False
        det._hsv_enabled = False
        det.last_status = "idle"
        det.last_primary_confidence = 0.0
        det.last_partial_count = 0
        det.last_partial_labels = []
        det.last_primary_partial_kind = None
        det.last_primary_partial_label = None
        det.current_frame = None
        return det

    def test_estimate_cam_height_matches_centerline_geometry(self):
        det = self._detector_stub()

        center = BrickDetector._estimate_cam_height(det, center_y_px=240.0, dist_mm=200.0)
        below = BrickDetector._estimate_cam_height(det, center_y_px=300.0, dist_mm=200.0)
        above = BrickDetector._estimate_cam_height(det, center_y_px=180.0, dist_mm=200.0)

        self.assertAlmostEqual(center, 0.0, places=6)
        self.assertAlmostEqual(below, 120.0, places=6)
        self.assertAlmostEqual(above, -120.0, places=6)

    def test_process_bricks_fallback_returns_nonzero_cam_height(self):
        det = self._detector_stub()
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        det._estimate_distance = lambda _bbox_h: 200.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(300, 280, 340, 320, 0.95)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertAlmostEqual(result[3], 0.0, places=6)
        self.assertAlmostEqual(result[5], 120.0, places=6)

    def test_process_bricks_hsv_path_uses_primary_center_y(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det._estimate_distance = lambda _bbox_h: 200.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._select_center_brick = lambda hsv_bricks, _w, _h: hsv_bricks[0]
        det._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)

        primary = {
            "center_x": 320,
            "center_y": 200,
            "bbox": (300, 180, 40, 40),
            "contour": np.array([[[300, 180]], [[340, 180]], [[340, 220]], [[300, 220]]], dtype=np.int32),
            "rect": ((320.0, 200.0), (40.0, 40.0), 0.0),
        }
        det._segment_bricks_hsv = lambda *_args, **_kwargs: [primary]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(260, 220, 380, 360, 0.9)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        # center_y=200 against frame center=240 -> negative cam height
        self.assertAlmostEqual(result[5], -80.0, places=6)

    def test_process_bricks_hsv_path_tracks_partial_labels(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det._estimate_distance = lambda _bbox_h: 200.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._select_center_brick = lambda hsv_bricks, _w, _h: hsv_bricks[0]
        det._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)

        primary = {
            "center_x": 320,
            "center_y": 200,
            "bbox": (300, 180, 40, 40),
            "contour": np.array([[[300, 180]], [[340, 180]], [[340, 220]], [[300, 220]]], dtype=np.int32),
            "rect": ((320.0, 200.0), (40.0, 40.0), 0.0),
            "partial": True,
            "partial_kind": "top_half",
            "partial_label": "TOP HALF",
        }
        other = {
            "center_x": 320,
            "center_y": 280,
            "bbox": (300, 260, 40, 40),
            "contour": np.array([[[300, 260]], [[340, 260]], [[340, 300]], [[300, 300]]], dtype=np.int32),
            "rect": ((320.0, 280.0), (40.0, 40.0), 0.0),
            "partial": True,
            "partial_kind": "bottom_half",
            "partial_label": "BOTTOM HALF",
        }
        det._segment_bricks_hsv = lambda *_args, **_kwargs: [primary, other]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(260, 140, 380, 340, 0.9)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertEqual(det.last_partial_count, 2)
        self.assertEqual(det.last_partial_labels, ["TOP HALF", "BOTTOM HALF"])
        self.assertEqual(det.last_primary_partial_kind, "top_half")
        self.assertEqual(det.last_primary_partial_label, "TOP HALF")

    def test_draw_debug_hsv_colors_partial_primary_orange(self):
        det = self._detector_stub()
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        primary = {
            "center_x": 30,
            "center_y": 30,
            "bbox": (10, 10, 40, 40),
            "contour": np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]], dtype=np.int32),
            "rect": ((30.0, 30.0), (40.0, 40.0), 0.0),
            "partial": True,
            "partial_kind": "top_half",
            "partial_label": "TOP HALF",
        }

        BrickDetector._draw_debug_hsv(
            det,
            frame,
            [],
            [primary],
            primary,
            angle=0.0,
            dist=0.0,
            offset_x=0.0,
            conf=0.9,
        )

        self.assertIsNotNone(det.current_frame)
        self.assertTrue(np.array_equal(det.current_frame[10, 30], np.array([0, 165, 255], dtype=np.uint8)))

    def test_process_bricks_fallback_rejects_shape_mismatch_when_gate_enabled(self):
        det = self._detector_stub()
        det._hsv_enabled = False
        det._face_shape_templates = {"full": np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]], dtype=np.float32)}
        det._extract_shape_contour_in_box = lambda *_args, **_kwargs: np.array(
            [[[0.0, 0.0]], [[4.0, 0.0]], [[4.0, 4.0]], [[0.0, 4.0]]], dtype=np.float32
        )
        det._classify_contour_shape = lambda _cnt: (None, 0.9)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(300, 280, 340, 320, 0.95)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertFalse(result[0])
        self.assertEqual(det.last_status, "shape mismatch")
        self.assertEqual(det.last_primary_confidence, 0.0)

    def test_effective_distance_height_scales_top_bottom_partials(self):
        det = self._detector_stub()

        self.assertEqual(BrickDetector._effective_distance_height_px(det, 20, None), 20.0)
        self.assertEqual(BrickDetector._effective_distance_height_px(det, 20, "full"), 20.0)
        self.assertEqual(BrickDetector._effective_distance_height_px(det, 20, "top_half"), 40.0)
        self.assertEqual(BrickDetector._effective_distance_height_px(det, 20, "bottom_half"), 40.0)

    def test_hsv_partial_uses_full_equivalent_height_for_distance(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        called_heights = []
        det._estimate_distance = lambda h: called_heights.append(float(h)) or 200.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._select_center_brick = lambda hsv_bricks, _w, _h: hsv_bricks[0]
        det._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)

        primary = {
            "center_x": 320,
            "center_y": 200,
            "bbox": (300, 180, 40, 40),
            "contour": np.array([[[300, 180]], [[340, 180]], [[340, 220]], [[300, 220]]], dtype=np.int32),
            "rect": ((320.0, 200.0), (40.0, 40.0), 0.0),
            "partial": True,
            "partial_kind": "top_half",
            "partial_label": "TOP HALF",
        }
        det._segment_bricks_hsv = lambda *_args, **_kwargs: [primary]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(260, 220, 380, 360, 0.9)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertEqual(called_heights, [80.0])

    def test_fallback_partial_uses_full_equivalent_height_for_distance(self):
        det = self._detector_stub()
        det._hsv_enabled = False
        det._face_shape_templates = {"full": np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]], dtype=np.float32)}
        det._extract_shape_contour_in_box = lambda *_args, **_kwargs: np.array(
            [[[0.0, 0.0]], [[4.0, 0.0]], [[4.0, 4.0]], [[0.0, 4.0]]], dtype=np.float32
        )
        det._classify_contour_shape = lambda _cnt: ("bottom_half", 0.1)
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        called_heights = []
        det._estimate_distance = lambda h: called_heights.append(float(h)) or 200.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(300, 280, 340, 320, 0.95)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertEqual(called_heights, [80.0])


if __name__ == "__main__":
    unittest.main()
