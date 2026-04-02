import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_detector_yolo as detector


class TestHelperBrickDetectorYoloCrownShape(unittest.TestCase):
    def _build_detector_stub(self):
        vision = detector.BrickDetector.__new__(detector.BrickDetector)
        vision._face_polygon_model = None
        vision._face_cutouts_model = []
        vision._face_lines_model = []
        vision._face_shape_templates = {}
        detector.BrickDetector._load_face_shape_model(vision)
        detector.BrickDetector._init_shape_match_templates(vision)
        return vision

    def test_crown_world_model_shape_loads_cutout_and_full_profile(self):
        vision = self._build_detector_stub()

        self.assertIsNotNone(vision._face_polygon_model)
        self.assertEqual(len(vision._face_cutouts_model), 1)
        self.assertIn("full", vision._face_shape_templates)

        full_contour = detector.BrickDetector._contour_from_points(
            vision,
            vision._face_polygon_model,
        )
        profile, score = detector.BrickDetector._classify_contour_shape(vision, full_contour)
        self.assertEqual(profile, "full")
        self.assertIsInstance(score, float)

    def test_crown_world_model_partial_profiles_classify_against_self(self):
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


if __name__ == "__main__":
    unittest.main()
