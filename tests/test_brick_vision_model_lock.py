"""Regression checks for the empty-game brick vision model lock."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "helper_brick_detector_native_oak.py"


def _source_between(text: str, start: str, end: str) -> str:
    start_idx = text.index(start)
    end_idx = text.index(end, start_idx)
    return text[start_idx:end_idx]


class TestBrickVisionModelLock(unittest.TestCase):
    def test_read_frame_does_not_use_green_edge_primary_or_fallback(self):
        text = DETECTOR.read_text(encoding="utf-8")
        read_frame = _source_between(text, "    def read_frame(self, frame):", "    def _detect_color_rectangle_candidate")

        self.assertNotIn("_green_edge_close_range_result", read_frame)

    def test_native_rect_result_path_does_not_switch_to_green_edge_distance(self):
        text = DETECTOR.read_text(encoding="utf-8")
        result_prefix = _source_between(text, "    def _result_from_candidate", "        raw_angle =")

        self.assertNotIn("_green_edge_close_range_result", result_prefix)
        self.assertIn("Model lock", result_prefix)


if __name__ == "__main__":
    unittest.main()
