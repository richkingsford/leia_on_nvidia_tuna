import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main2
from helper_brick_detector_yolo import BRICK_HEIGHT_MM, BRICK_WIDTH_MM


class TestMain2LivePosition(unittest.TestCase):
    def test_candidate_scale_uses_detector_scale_when_present(self):
        self.assertAlmostEqual(
            main2._candidate_scale_px_per_mm({"scale_px_per_mm": 3.25}),
            3.25,
            places=3,
        )

    def test_candidate_scale_falls_back_to_bbox_size(self):
        candidate = {
            "bbox": (10, 20, BRICK_WIDTH_MM * 2.0, BRICK_HEIGHT_MM * 2.0),
        }

        self.assertAlmostEqual(
            main2._candidate_scale_px_per_mm(candidate),
            2.0,
            places=3,
        )

    def test_candidate_center_falls_back_to_bbox_center(self):
        self.assertEqual(
            main2._candidate_center({"bbox": (10, 20, 40, 60)}),
            (30.0, 50.0),
        )

    def test_candidate_metrics_uses_camera_center_and_scale(self):
        metrics = main2._candidate_metrics(
            {
                "center_x": 360,
                "center_y": 220,
                "scale_px_per_mm": 2.0,
            },
            frame_w=640,
            frame_h=480,
            focal_px=580.0,
        )

        self.assertAlmostEqual(metrics["dist_mm"], 290.0, places=3)
        self.assertAlmostEqual(metrics["x_mm"], 20.0, places=3)
        self.assertAlmostEqual(metrics["y_mm"], -10.0, places=3)

    def test_format_metric_lines_reports_three_values(self):
        lines = main2._format_metric_lines(
            {
                "x_mm": 12.4,
                "y_mm": -3.6,
                "dist_mm": 149.8,
            }
        )

        self.assertEqual(lines, ["x +12 mm", "y -4 mm", "dist 150 mm"])

    def test_draw_brick_metric_label_changes_frame(self):
        frame = np.zeros((180, 260, 3), dtype=np.uint8)
        before = frame.copy()

        main2._draw_brick_metric_label(
            frame,
            {"bbox": (30, 40, 60, 40)},
            {"x_mm": 12.0, "y_mm": -4.0, "dist_mm": 150.0},
        )

        self.assertGreater(int(np.count_nonzero(frame != before)), 0)


if __name__ == "__main__":
    unittest.main()
