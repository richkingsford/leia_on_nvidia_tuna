"""Regression checks for the empty-game brick vision model lock."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "helper_brick_detector_native_oak.py"
sys.path.append(str(ROOT))


def _source_between(text: str, start: str, end: str) -> str:
    start_idx = text.index(start)
    end_idx = text.index(end, start_idx)
    return text[start_idx:end_idx]


class TestBrickVisionModelLock(unittest.TestCase):
    def test_read_frame_uses_green_edge_only_as_last_resort_fallback(self):
        text = DETECTOR.read_text(encoding="utf-8")
        read_frame = _source_between(text, "    def read_frame(self, frame):", "    def _detect_color_rectangle_candidate")

        self.assertIn("_green_edge_close_range_result(frame, require_close=False)", read_frame)
        self.assertIn("if primary is None:", read_frame)

    def test_native_rect_result_path_does_not_switch_to_green_edge_distance(self):
        text = DETECTOR.read_text(encoding="utf-8")
        result_prefix = _source_between(text, "    def _result_from_candidate", "        raw_angle =")

        self.assertNotIn("_green_edge_close_range_result", result_prefix)
        self.assertIn("Model lock", result_prefix)

    def test_painted_column_fallback_does_not_calibrate_far_pose_to_close(self):
        try:
            import cv2
            import numpy as np
            from helper_brick_detector_native_oak import NativeOakBrickDetector
        except Exception as exc:  # pragma: no cover - optional local deps
            self.skipTest(f"OpenCV/native detector unavailable: {exc}")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (25, 25, 25)

        def bgr_from_hsv(h: int, s: int, v: int):
            px = np.array([[[h, s, v]]], dtype=np.uint8)
            return tuple(int(x) for x in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])

        stack_green = bgr_from_hsv(70, 210, 220)
        cv2.rectangle(frame, (430, 60), (546, 240), stack_green, -1)

        vision = NativeOakBrickDetector(debug=True)
        result = vision._green_edge_close_range_result(frame, require_close=False)

        self.assertIsNotNone(result)
        self.assertTrue(result[0])
        self.assertIn(
            vision.last_geometry_source,
            {"green_edge_painted_column_width", "green_edge_top_strip_width"},
        )
        self.assertGreater(result[2], 180.0)
        self.assertGreater(vision.last_bbox_w_px, 90)
        self.assertLess(vision.last_bbox_w_px, 150)


if __name__ == "__main__":
    unittest.main()
