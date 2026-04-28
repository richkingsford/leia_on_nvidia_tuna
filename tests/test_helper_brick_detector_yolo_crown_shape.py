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

    def _synthetic_negative_cutout_frame(self, *, missing_right=False, right_y_shift=0.0):
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
            if idx == 1 and float(right_y_shift) != 0.0:
                cutout_proj = cutout_proj.copy()
                cutout_proj[:, 1] += float(right_y_shift)
            cutout_draw = np.round(cutout_proj).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(frame, [cutout_draw], (0, 0, 0))
        return vision, frame

    def _synthetic_pair_indicator_frame(self, *, right_y_shift=0.0):
        vision = self._build_detector_stub()
        frame = np.zeros((140, 220, 3), dtype=np.uint8)
        cyan_bgr = (177, 157, 31)
        cv2.rectangle(frame, (20, 15), (200, 125), cyan_bgr, thickness=cv2.FILLED)

        cutout_items = detector.BrickDetector._sorted_face_cutout_models(vision)
        model_pair_center = (
            cutout_items[0]["centroid"] + cutout_items[1]["centroid"]
        ) * 0.5
        for idx, item in enumerate(cutout_items[:2]):
            cutout_proj = detector.BrickDetector._transform_model_points_similarity(
                vision,
                item["polygon"],
                model_origin=model_pair_center,
                image_origin=np.asarray([110.0, 70.0], dtype=np.float32),
                scale=2.2,
                cos_theta=1.0,
                sin_theta=0.0,
            )
            if idx == 1 and float(right_y_shift) != 0.0:
                cutout_proj = cutout_proj.copy()
                cutout_proj[:, 1] += float(right_y_shift)
            cutout_draw = np.round(cutout_proj).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(frame, [cutout_draw], (0, 0, 0))
        return vision, frame

    def test_world_model_shape_loads_cutouts_and_full_profile(self):
        vision = self._build_detector_stub()

        self.assertIsNotNone(vision._face_polygon_model)
        self.assertEqual(len(vision._face_cutouts_model), 2)
        self.assertIn("full", vision._face_shape_templates)
        self.assertEqual(vision._face_shape_gate_mode, "negative_cutouts")
        self.assertEqual(float(vision._negative_cutout_pair_x_axis_max_angle_deg), 15.0)
        self.assertEqual(float(vision._negative_cutout_focus_y_min_ratio), 0.35)
        self.assertEqual(float(vision._negative_cutout_focus_y_max_ratio), 0.68)
        self.assertEqual(float(vision._negative_cutout_focus_weight), 1.25)

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

    def test_negative_cutout_focus_penalty_prefers_band_below_midpoint(self):
        vision = self._build_detector_stub()
        near_mid_triangle = np.array(
            [[40.0, 54.0], [60.0, 54.0], [50.0, 74.0]],
            dtype=np.float32,
        )
        low_triangle = np.array(
            [[40.0, 84.0], [60.0, 84.0], [50.0, 104.0]],
            dtype=np.float32,
        )

        near_mid_penalty = detector.BrickDetector._negative_cutout_focus_penalty(
            vision,
            near_mid_triangle,
            (120, 120),
        )
        low_penalty = detector.BrickDetector._negative_cutout_focus_penalty(
            vision,
            low_triangle,
            (120, 120),
        )

        self.assertLess(near_mid_penalty, low_penalty)

    def test_negative_cutout_pair_requires_same_x_axis_plane(self):
        vision = self._build_detector_stub()
        side_by_side = [
            np.array([[40.0, 54.0], [60.0, 54.0], [50.0, 74.0]], dtype=np.float32),
            np.array([[80.0, 54.0], [100.0, 54.0], [90.0, 74.0]], dtype=np.float32),
        ]
        shifted = [
            np.array([[40.0, 54.0], [60.0, 54.0], [50.0, 74.0]], dtype=np.float32),
            np.array([[80.0, 72.0], [100.0, 72.0], [90.0, 92.0]], dtype=np.float32),
        ]

        passed_side_by_side, side_by_side_metrics = (
            detector.BrickDetector._negative_cutout_pair_x_axis_pass(
                vision,
                side_by_side,
            )
        )
        passed_shifted, shifted_metrics = (
            detector.BrickDetector._negative_cutout_pair_x_axis_pass(
                vision,
                shifted,
            )
        )

        self.assertTrue(passed_side_by_side)
        self.assertTrue(bool(side_by_side_metrics.get("passed")))
        self.assertFalse(passed_shifted)
        self.assertFalse(bool(shifted_metrics.get("passed")))
        self.assertGreater(
            float(shifted_metrics.get("angle_from_x_axis_deg")),
            float(shifted_metrics.get("max_angle_deg")),
        )

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
        self.assertIsInstance(bricks[0].get("selection_anchor_x"), float)
        self.assertIsInstance(bricks[0].get("selection_anchor_y"), float)

    def test_hsv_segment_can_recover_from_cutout_pair_only_indicator(self):
        vision, frame = self._synthetic_pair_indicator_frame()
        vision._match_negative_cutout_pair = lambda *_args, **_kwargs: (False, None)

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
        self.assertIsInstance(bricks[0].get("shape_match_score"), float)
        self.assertAlmostEqual(float(bricks[0].get("selection_anchor_x")), 110.0, delta=8.0)

    def test_hsv_segment_pair_only_recovery_rejects_misaligned_triangles(self):
        vision, frame = self._synthetic_pair_indicator_frame(right_y_shift=18.0)
        vision._match_negative_cutout_pair = lambda *_args, **_kwargs: (False, None)

        bricks = detector.BrickDetector._segment_bricks_hsv(
            vision,
            frame,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
        )

        self.assertEqual(bricks, [])

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

    def test_shape_only_detector_highlights_one_side_by_side_cutout_pair(self):
        vision, frame = self._synthetic_pair_indicator_frame()

        primary, candidates = detector.detect_single_negative_cutout_brick(
            vision,
            frame,
        )

        self.assertIsNotNone(primary)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(float(primary["selection_anchor_x"]), 110.0, delta=4.0)

        highlighted = frame.copy()
        detector.draw_single_negative_cutout_brick_highlight(
            vision,
            highlighted,
            primary,
        )
        self.assertGreater(int(np.count_nonzero(cv2.absdiff(frame, highlighted))), 0)


if __name__ == "__main__":
    unittest.main()
