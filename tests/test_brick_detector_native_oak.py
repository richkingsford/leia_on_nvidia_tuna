import sys
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_brick_detector_native_oak import NativeOakBrickDetector
from helper_brick_detector_yolo import CYAN_HSV_BALANCED_LOWER, CYAN_HSV_BALANCED_UPPER


def _bgr(hex_code: str) -> tuple[int, int, int]:
    text = str(hex_code).strip().lstrip("#")
    return int(text[4:6], 16), int(text[2:4], 16), int(text[0:2], 16)


class TestNativeOakBrickDetector(unittest.TestCase):
    def _detector(self) -> NativeOakBrickDetector:
        detector = NativeOakBrickDetector(debug=False, width=320, height=240)
        detector.set_runtime_tuning(
            hsv_lower=list(CYAN_HSV_BALANCED_LOWER),
            hsv_upper=list(CYAN_HSV_BALANCED_UPPER),
            hsv_erode_iterations=0,
        )
        return detector

    def test_dark_green_palette_sample_locks_as_native_rect(self):
        detector = self._detector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (130, 70), (200, 170), _bgr("27832F"), thickness=cv2.FILLED)

        result = detector.read_frame(frame)

        self.assertTrue(result[0])
        self.assertEqual(detector.last_status, "target locked (native color+rect)")
        self.assertGreaterEqual(float(result[4]), 75.0)
        self.assertGreaterEqual(float(detector.last_bbox_h_px), 95.0)

    def test_center_thin_green_strip_is_discarded_for_rectangular_stack(self):
        detector = self._detector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # A ghosty horizontal strip is closer to center but not brick-stack shaped.
        cv2.rectangle(frame, (110, 108), (220, 132), _bgr("27832F"), thickness=cv2.FILLED)
        cv2.rectangle(frame, (136, 66), (196, 174), _bgr("287D3B"), thickness=cv2.FILLED)

        primary, candidates = detector._detect_color_rectangle_candidate(frame)

        self.assertIsNotNone(primary)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreaterEqual(int(primary["bbox"][3]), 100)
        self.assertLessEqual(float(primary["bbox_aspect"]), 3.25)

    def test_connected_ghost_strip_does_not_expand_stack_bbox(self):
        detector = self._detector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Real brick stack: tall-ish, straight left/right edges, horizontal gaps.
        for y1, y2 in ((72, 96), (108, 132), (144, 168)):
            cv2.rectangle(frame, (122, y1), (188, y2), _bgr("27832F"), thickness=cv2.FILLED)
        # Same-color ghost pixels connected to the stack. A raw contour bbox would
        # include this bridge and jump far to the right.
        cv2.rectangle(frame, (188, 116), (276, 124), _bgr("27832F"), thickness=cv2.FILLED)
        cv2.rectangle(frame, (260, 44), (268, 176), _bgr("27832F"), thickness=cv2.FILLED)

        primary, candidates = detector._detect_color_rectangle_candidate(frame)

        self.assertIsNotNone(primary)
        self.assertGreaterEqual(len(candidates), 1)
        x, y, w, h = primary["bbox"]
        self.assertLessEqual(int(x), 126)
        self.assertGreaterEqual(int(x + w), 184)
        self.assertLessEqual(int(w), 76)
        self.assertGreaterEqual(float(primary["edge_straightness"]), 0.7)
        self.assertGreaterEqual(float(primary["width_coherence"]), 0.7)

    def test_solid_stack_clipped_at_top_remains_visible(self):
        detector = self._detector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # The top of the stack is outside the camera frame. Its visible face is
        # still a dense green rectangle with coherent straight edges.
        cv2.rectangle(frame, (80, 0), (240, 160), _bgr("27832F"), thickness=cv2.FILLED)

        primary, candidates = detector._detect_color_rectangle_candidate(frame)

        self.assertIsNotNone(primary)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(bool(primary["partial"]))
        self.assertTrue(bool(primary["partial_edges"]["top"]))
        self.assertEqual(primary["partial_label"], "CLIPPED SOLID GREEN")

    def test_solid_stack_clipped_at_frame_corner_remains_visible(self):
        detector = self._detector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (180, 100), _bgr("27832F"), thickness=cv2.FILLED)

        primary, _candidates = detector._detect_color_rectangle_candidate(frame)

        self.assertIsNotNone(primary)
        self.assertTrue(bool(primary["partial"]))
        self.assertTrue(bool(primary["partial_edges"]["top"]))
        self.assertTrue(bool(primary["partial_edges"]["left"]))

    def test_camera_frame_timeout_reopens_source_and_returns_non_confident_read(self):
        detector = self._detector()
        detector._open_camera = Mock(side_effect=[True, True])
        detector._start_capture_worker = Mock()

        # No frame arrives after the capture worker starts. Recovery must be
        # contained inside the detector and expose a normal lost-visibility
        # result to the motion controller.
        with patch.object(detector, "_capture_thread", None), patch.object(
            detector._capture_stop, "set"
        ), patch.object(
            __import__("helper_brick_detector_native_oak").time,
            "monotonic",
            side_effect=[0.0, 2.0],
        ):
            result = detector.read()

        self.assertFalse(result[0])
        self.assertEqual(detector.last_status, "camera read failed")
        self.assertEqual(detector._open_camera.call_count, 2)


if __name__ == "__main__":
    unittest.main()
