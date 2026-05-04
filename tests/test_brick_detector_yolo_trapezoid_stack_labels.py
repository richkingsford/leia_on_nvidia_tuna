import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_detector_yolo as detector


class TestBrickDetectorYoloTrapezoidStackLabels(unittest.TestCase):
    def _build_detector_stub(self):
        vision = detector.BrickDetector.__new__(detector.BrickDetector)
        vision._face_polygon_model = None
        vision._face_cutouts_model = []
        vision._face_lines_model = []
        vision._face_shape_templates = {}
        vision._hsv_lower = np.array(detector.CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
        vision._hsv_upper = np.array(detector.CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
        vision._hsv_erode_iterations = 0
        detector.BrickDetector._load_face_shape_model(vision)
        detector.BrickDetector._init_shape_match_templates(vision)
        return vision

    def _draw_trapezoid_slot(self, frame, y):
        slot = np.asarray(
            [
                [58, y],
                [162, y],
                [150, y + 14],
                [70, y + 14],
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(frame, [slot], (0, 0, 0))

    def test_merged_cyan_stack_splits_into_topmost_numbered_bricks(self):
        vision = self._build_detector_stub()
        frame = np.zeros((180, 220, 3), dtype=np.uint8)
        cyan_bgr = (177, 157, 31)
        cv2.rectangle(frame, (32, 18), (188, 160), cyan_bgr, thickness=cv2.FILLED)
        self._draw_trapezoid_slot(frame, 46)
        self._draw_trapezoid_slot(frame, 108)

        candidates = detector.BrickDetector._segment_bricks_hsv(
            vision,
            frame,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
        )

        self.assertGreaterEqual(len(candidates), 2)
        top_two = sorted(candidates, key=lambda item: item["center_y"])[:2]
        self.assertLess(top_two[0]["center_y"], top_two[1]["center_y"])

        highlighted = frame.copy()
        detector.BrickDetector._draw_brick_id_labels(vision, highlighted, top_two)
        yellow_mask = (
            (highlighted[:, :, 0] == 0)
            & (highlighted[:, :, 1] == 255)
            & (highlighted[:, :, 2] == 255)
        )
        self.assertGreater(int(np.count_nonzero(yellow_mask)), 0)
        outline_band_y = top_two[0]["bbox"][1]
        self.assertFalse(bool(np.any(yellow_mask[outline_band_y, :])))


if __name__ == "__main__":
    unittest.main()
