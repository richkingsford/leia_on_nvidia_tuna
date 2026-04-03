import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_detector_yolo as detector


class TestHelperBrickDetectorYoloCrownShape(unittest.TestCase):
    def _build_detector_stub(self):
        vision = detector.BrickDetector.__new__(detector.BrickDetector)
        vision._face_polygon_model = None
        vision._face_cutouts_model = []
        vision._face_lines_model = []
        vision._face_shape_templates = {}
        vision._hsv_lower = detector.CYAN_HSV_LOWER.copy()
        vision._hsv_upper = detector.CYAN_HSV_UPPER.copy()
        vision._hsv_erode_iterations = detector.HSV_ERODE_ITERATIONS
        detector.BrickDetector._load_face_shape_model(vision)
        detector.BrickDetector._init_shape_match_templates(vision)
        return vision

    def _synthetic_negative_cutout_frame(self, *, missing_right=False):
        vision = self._build_detector_stub()
        frame = np.zeros((140, 200, 3), dtype=np.uint8)
        cyan_bgr = (177, 157, 31)
        cv2.rectangle(frame, (30, 20), (170, 120), cyan_bgr, thickness=cv2.FILLED)

        rect = ((100.0, 70.0), (140.0, 100.0), 0.0)
        rect_points = detector.BrickDetector._ordered_rect_points(vision, rect)
        for idx, cutout_points in enumerate(vision._face_cutouts_model):
            if missing_right and idx == 1:
                continue
            cutout_proj = detector.BrickDetector._project_model_points_to_rect(
                vision,
                cutout_points,
                rect_points,
            )
            cutout_draw = np.round(cutout_proj).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(frame, [cutout_draw], (0, 0, 0))
        return vision, frame

    def test_world_model_shape_loads_cutouts_and_full_profile(self):
        vision = self._build_detector_stub()

        self.assertIsNotNone(vision._face_polygon_model)
        self.assertEqual(len(vision._face_cutouts_model), 2)
        self.assertIn("full", vision._face_shape_templates)
        self.assertEqual(vision._face_shape_gate_mode, "negative_cutouts")

        full_contour = detector.BrickDetector._contour_from_points(
            vision,
            vision._face_polygon_model,
        )
        profile, score = detector.BrickDetector._classify_contour_shape(vision, full_contour)
        self.assertEqual(profile, "full")
        self.assertIsInstance(score, float)

    def test_world_model_partial_profiles_classify_against_self(self):
        vision = self._build_detector_stub()
        poly = vision._face_polygon_model
        y_vals = poly[:, 1]
        y_mid = (float(y_vals.min()) + float(y_vals.max())) * 0.5

        top_poly = detector.BrickDetector._clip_polygon_y(vision, poly, y_mid, keep_above=True)
        top_contour = detector.BrickDetector._contour_from_points(vision, top_poly)
        top_profile, _top_score = detector.BrickDetector._classify_contour_shape(vision, top_contour)

        bottom_poly = detector.BrickDetector._clip_polygon_y(vision, poly, y_mid, keep_above=False)
        bottom_contour = detector.BrickDetector._contour_from_points(vision, bottom_poly)
        bottom_profile, _bottom_score = detector.BrickDetector._classify_contour_shape(vision, bottom_contour)

        self.assertEqual(top_profile, "top_half")
        self.assertEqual(bottom_profile, "bottom_half")

    def test_hsv_segment_accepts_two_negative_cutouts_inside_cyan(self):
        vision, frame = self._synthetic_negative_cutout_frame()

        bricks = detector.BrickDetector._segment_bricks_hsv(
            vision,
            frame,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
        )

        self.assertEqual(len(bricks), 1)
        self.assertEqual(bricks[0].get("shape_profile"), "full")
        self.assertFalse(bool(bricks[0].get("partial")))

    def test_hsv_segment_rejects_candidate_missing_one_negative_cutout(self):
        vision, frame = self._synthetic_negative_cutout_frame(missing_right=True)

        bricks = detector.BrickDetector._segment_bricks_hsv(
            vision,
            frame,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
        )

        self.assertEqual(bricks, [])


if __name__ == "__main__":
    unittest.main()
