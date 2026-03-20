import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_vision
import helper_vision_aruco
from helper_brick_vision import BrickDetector
from helper_vision_aruco import ArucoBrickVision, BrickPose


class TestHelperStackFlagSemantics(unittest.TestCase):
    def test_aruco_stack_flags_require_separate_verified_marker_below_selected(self):
        vision = ArucoBrickVision.__new__(ArucoBrickVision)
        vision.camera_matrix = np.array(
            [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        vision.BRICK_H = 48.0

        best = BrickPose(
            found=True,
            position=(0.0, 0.0, 100.0),
            orientation=(0.0, 0.0, 0.0),
            confidence=98.0,
            marker_id=11,
            corners=np.array(
                [[310.0, 230.0], [330.0, 230.0], [330.0, 250.0], [310.0, 250.0]],
                dtype=np.float32,
            ),
        )
        below = BrickPose(
            found=True,
            position=(0.0, 48.0, 100.0),
            orientation=(0.0, 0.0, 0.0),
            confidence=97.0,
            marker_id=11,
            corners=np.array(
                [[312.0, 286.0], [332.0, 286.0], [332.0, 306.0], [312.0, 306.0]],
                dtype=np.float32,
            ),
        )

        brick_above, brick_below = vision._stack_flags_from_verified_markers(best, [best, below])

        self.assertFalse(brick_above)
        self.assertTrue(brick_below)

    def test_aruco_process_ignores_debug_possible_marker_hits_for_stack_flags(self):
        vision = ArucoBrickVision.__new__(ArucoBrickVision)
        vision.camera_matrix = np.array(
            [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        vision.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        vision.debug = True
        vision.debug_rejected_candidates = True
        vision.marker_points_3d = np.zeros((4, 3), dtype=np.float32)
        vision.BRICK_H = 48.0
        vision.ALPHA = 1.0
        vision.last_pose = None
        vision.last_found_ts_mono = 0.0
        vision.visibility_hold_seconds = 0.0
        vision.debug_frame = None
        vision.target_marker_id = 11
        vision._detect_markers_with_fallback = lambda _gray: (
            [
                np.array(
                    [[[310.0, 230.0], [330.0, 230.0], [330.0, 250.0], [310.0, 250.0]]],
                    dtype=np.float32,
                )
            ],
            np.array([[11]], dtype=np.int32),
            [],
        )
        vision._estimate_marker_confidence = lambda *_args, **_kwargs: 98.0
        vision._draw_debug_marker_candidates = lambda *_args, **_kwargs: 1

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = np.zeros((480, 640), dtype=np.uint8)

        with patch.object(
            helper_vision_aruco.cv2,
            "solvePnP",
            return_value=(
                True,
                np.zeros((3, 1), dtype=np.float32),
                np.array([[0.0], [0.0], [100.0]], dtype=np.float32),
            ),
        ), patch.object(
            helper_vision_aruco.cv2,
            "Rodrigues",
            return_value=(np.eye(3, dtype=np.float32), None),
        ), patch.object(
            helper_vision_aruco.cv2,
            "RQDecomp3x3",
            return_value=((0.0, 0.0, 0.0), None, None, None, None, None),
        ):
            pose = ArucoBrickVision.process(vision, frame, gray=gray)

        self.assertTrue(pose.found)
        self.assertFalse(pose.brickAbove)
        self.assertFalse(pose.brickBelow)

    def test_blob_stack_flags_require_same_column_candidate(self):
        det = BrickDetector.__new__(BrickDetector)
        selected = {"center": (320.0, 240.0), "bbox": (300, 200, 40, 80)}
        off_axis_below = {"center": (380.0, 340.0), "bbox": (360, 300, 40, 80)}
        same_column_below = {"center": (325.0, 340.0), "bbox": (305, 300, 40, 80)}

        brick_above, brick_below = det._stack_flags_from_candidates(selected, [selected, off_axis_below])
        self.assertFalse(brick_above)
        self.assertFalse(brick_below)

        brick_above, brick_below = det._stack_flags_from_candidates(selected, [selected, same_column_below])
        self.assertFalse(brick_above)
        self.assertTrue(brick_below)

    def test_blob_process_does_not_infer_stack_from_notch_row(self):
        det = BrickDetector.__new__(BrickDetector)
        det.camera_matrix = np.array(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        det.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        det.debug_search_zones = []
        det.clahe = type("ClaheStub", (), {"apply": staticmethod(lambda img: img)})()
        det.hsv_ranges = [
            (
                np.array([0, 0, 0], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )
        ]
        det.camera_center_offset_px = 0.0
        det.speed_optimize = True
        det.stack_history = deque(maxlen=6)
        det.draw_text_with_bg = lambda *_args, **_kwargs: None
        det.get_notch_from_contour = lambda _mask, _cnt: (None, None)
        det.calculate_fingerprint_confidence = lambda _cnt, _img_pts: (0.0, 90.0, [])
        det.stabilize_stack_state = lambda above, below, _found, _conf: (above, below)
        det.detect_notch_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("detect_notch_row should not be used for stack flags")
        )

        contour = np.array([[[1, 1]], [[1, 6]], [[6, 6]], [[6, 1]]], dtype=np.int32)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        with patch.object(helper_brick_vision, "MIN_AREA_THRESHOLD", 1), patch.object(
            helper_brick_vision.cv2,
            "findContours",
            side_effect=[([contour], None), ([contour], None)],
        ):
            result = BrickDetector.process_frame(det, frame)

        self.assertTrue(result[0])
        self.assertFalse(result[7])
        self.assertFalse(result[8])


if __name__ == "__main__":
    unittest.main()
