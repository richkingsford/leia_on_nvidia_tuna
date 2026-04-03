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
        det._hsv_lower = np.array([75, 60, 50])
        det._hsv_upper = np.array([115, 255, 255])
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
        det.current_frame = None
        return det

    def test_estimate_cam_height_matches_centerline_geometry(self):
        det = self._detector_stub()

        center = BrickDetector._estimate_cam_height(det, center_y_px=240.0, dist_signal=200.0)
        below = BrickDetector._estimate_cam_height(det, center_y_px=300.0, dist_signal=200.0)
        above = BrickDetector._estimate_cam_height(det, center_y_px=180.0, dist_signal=200.0)

        self.assertAlmostEqual(center, 0.0, places=6)
        self.assertAlmostEqual(below, 60.0, places=6)
        self.assertAlmostEqual(above, -60.0, places=6)

    def test_estimate_offset_x_mm_matches_camera_center_geometry(self):
        det = self._detector_stub()
        det.focal_px = 580.0

        centered = BrickDetector._estimate_offset_x_mm(det, center_x_px=320.0, dist_mm=100.0)
        right = BrickDetector._estimate_offset_x_mm(det, center_x_px=378.0, dist_mm=100.0)
        left = BrickDetector._estimate_offset_x_mm(det, center_x_px=262.0, dist_mm=100.0)

        self.assertAlmostEqual(centered, 0.0, places=6)
        self.assertAlmostEqual(right, 10.0, places=6)
        self.assertAlmostEqual(left, -10.0, places=6)

    def test_estimate_distance_from_box_prefers_width_signal(self):
        det = self._detector_stub()
        det.focal_px = 580.0

        width_only = BrickDetector._estimate_distance_from_width(det, 290.0)
        height_only = BrickDetector._estimate_distance(det, 120.0)
        blended = BrickDetector._estimate_distance_from_box(det, 290.0, 120.0)

        self.assertAlmostEqual(width_only, 106.0, places=6)
        self.assertGreater(height_only, width_only)
        self.assertGreater(blended, width_only)
        self.assertLess(blended, height_only)
        self.assertLess(abs(blended - width_only), abs(blended - height_only))

    def test_calibrate_focal_blends_width_and_height(self):
        det = self._detector_stub()
        det._detect = lambda _frame: [(100, 120, 390, 320, 0.9)]

        focal = BrickDetector.calibrate_focal(det, 100.0, frame=np.zeros((480, 640, 3), dtype=np.uint8))

        width_focal = (100.0 * 290.0) / 53.0
        height_focal = (100.0 * 200.0) / 35.4
        expected = (0.8 * width_focal) + (0.2 * height_focal)
        self.assertAlmostEqual(focal, expected, places=6)

    def test_process_bricks_fallback_returns_nonzero_cam_height(self):
        det = self._detector_stub()
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 200.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(300, 280, 340, 320, 0.95)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertAlmostEqual(result[3], 0.0, places=6)
        self.assertAlmostEqual(result[5], 60.0, places=6)

    def test_process_bricks_fallback_returns_horizontal_offset_in_mm(self):
        det = self._detector_stub()
        det.focal_px = 580.0
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 100.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(358, 180, 398, 220, 0.95)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertAlmostEqual(result[3], 10.0, places=6)

    def test_process_bricks_hsv_path_uses_primary_center_y(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 200.0
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
        self.assertAlmostEqual(result[5], -40.0, places=6)

    def test_process_bricks_hsv_path_returns_horizontal_offset_in_mm(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det.focal_px = 580.0
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 100.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._select_center_brick = lambda hsv_bricks, _w, _h: hsv_bricks[0]
        det._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)

        primary = {
            "center_x": 378,
            "center_y": 200,
            "bbox": (358, 180, 40, 40),
            "contour": np.array([[[358, 180]], [[398, 180]], [[398, 220]], [[358, 220]]], dtype=np.int32),
            "rect": ((378.0, 200.0), (40.0, 40.0), 0.0),
        }
        det._segment_bricks_hsv = lambda *_args, **_kwargs: [primary]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(320, 160, 420, 240, 0.9)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertAlmostEqual(result[3], 10.0, places=6)

    def test_process_bricks_hsv_closeup_fallback_uses_full_frame_when_yolo_has_no_boxes(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 100.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._select_center_brick = lambda hsv_bricks, _w, _h: hsv_bricks[0]
        det._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)

        primary = {
            "center_x": 320,
            "center_y": 240,
            "bbox": (20, 10, 600, 450),
            "contour": np.array([[[20, 10]], [[620, 10]], [[620, 460]], [[20, 460]]], dtype=np.int32),
            "rect": ((320.0, 235.0), (600.0, 450.0), 0.0),
        }
        segment_calls = []

        def _segment(frame, x1, y1, x2, y2):
            segment_calls.append((x1, y1, x2, y2))
            return [primary]

        det._segment_bricks_hsv = _segment

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = BrickDetector._process_bricks(det, frame, [])

        self.assertTrue(result[0])
        self.assertEqual(segment_calls, [(0, 0, 640, 480)])
        self.assertEqual(det.last_status, "target locked (HSV)")
        self.assertAlmostEqual(result[4], 100.0, places=6)

    def test_process_bricks_hsv_closeup_fallback_uses_full_frame_when_large_yolo_box_fails(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 100.0
        det._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        det._select_center_brick = lambda hsv_bricks, _w, _h: hsv_bricks[0]
        det._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)

        primary = {
            "center_x": 320,
            "center_y": 240,
            "bbox": (20, 10, 600, 450),
            "contour": np.array([[[20, 10]], [[620, 10]], [[620, 460]], [[20, 460]]], dtype=np.int32),
            "rect": ((320.0, 235.0), (600.0, 450.0), 0.0),
        }
        segment_calls = []

        def _segment(frame, x1, y1, x2, y2):
            segment_calls.append((x1, y1, x2, y2))
            if (x1, y1, x2, y2) == (0, 0, 640, 480):
                return [primary]
            return []

        det._segment_bricks_hsv = _segment

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(8, 6, 632, 472, 0.87)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertEqual(segment_calls, [(8, 6, 632, 472), (0, 0, 640, 480)])
        self.assertEqual(det.last_status, "target locked (HSV)")
        self.assertAlmostEqual(result[4], 87.0, places=6)

    def test_process_bricks_hsv_path_tracks_partial_labels(self):
        det = self._detector_stub()
        det._hsv_enabled = True
        det._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 200.0
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

    def test_draw_primary_face_outline_prefers_actual_contour(self):
        det = self._detector_stub()
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        primary = {
            "center_x": 30,
            "center_y": 30,
            "bbox": (10, 10, 40, 40),
            "contour": np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]], dtype=np.int32),
            "rect": ((70.0, 70.0), (80.0, 80.0), 0.0),
        }

        BrickDetector._draw_primary_face_outline(det, frame, primary)

        self.assertTrue(np.array_equal(frame[10, 30], np.array([0, 255, 0], dtype=np.uint8)))

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
        called_dims = []
        det._estimate_distance_from_box = (
            lambda w, h, _partial_kind=None: called_dims.append((float(w), float(h), _partial_kind)) or 200.0
        )
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
        self.assertEqual(called_dims, [(40.0, 40.0, "top_half")])

    def test_fallback_partial_uses_full_equivalent_height_for_distance(self):
        det = self._detector_stub()
        det._hsv_enabled = False
        det._face_shape_templates = {"full": np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]], dtype=np.float32)}
        det._extract_shape_contour_in_box = lambda *_args, **_kwargs: np.array(
            [[[0.0, 0.0]], [[4.0, 0.0]], [[4.0, 4.0]], [[0.0, 4.0]]], dtype=np.float32
        )
        det._classify_contour_shape = lambda _cnt: ("bottom_half", 0.1)
        det._estimate_angle = lambda *_args, **_kwargs: 0.0
        called_dims = []
        det._estimate_distance_from_box = (
            lambda w, h, _partial_kind=None: called_dims.append((float(w), float(h), _partial_kind)) or 200.0
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bricks = [(300, 280, 340, 320, 0.95)]

        result = BrickDetector._process_bricks(det, frame, bricks)

        self.assertTrue(result[0])
        self.assertEqual(called_dims, [(40.0, 40.0, "bottom_half")])


if __name__ == "__main__":
    unittest.main()
