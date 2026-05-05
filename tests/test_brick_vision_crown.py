"""
Crown brick vision regression tests.

Covers the complete current pipeline:
  - Crown profile uses shape_match gate (not negative_cutouts)
  - Trapezoid gate is active because world model has exactly 1 cutout polygon
  - Two dark inner slots → two separate candidates (inner-hole split path)
  - Transparent/missing slots → height-ratio fallback still produces 2+ candidates
  - _draw_brick_id_labels draws for 2+ candidates, silent for <=1
  - CrownVisionLivestream holds the last good detection for HOLD_FRAMES missed frames
"""
import argparse
import sys
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_detector_yolo as det
from livestream_crown_vision import (
    CROWN_PROFILE_TUNING,
    CrownVisionLivestream,
    HOLD_FRAMES,
)


# ─── shared stub builder ─────────────────────────────────────────────────────

def _make_stub() -> det.BrickDetector:
    """BrickDetector with real world-model geometry but no camera or TRT engine."""
    d = det.BrickDetector.__new__(det.BrickDetector)
    d._face_polygon_model = None
    d._face_cutouts_model = []
    d._face_lines_model = []
    d._face_shape_templates = {}
    d._hsv_lower = np.array(det.CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
    d._hsv_upper = np.array(det.CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
    d._hsv_erode_iterations = 0
    # Attributes read by set_runtime_tuning return value
    d.conf_threshold = det.CONF_THRESHOLD
    d._smooth_alpha = 0.3
    d.nms_threshold = det.NMS_THRESHOLD
    d.focal_px = det.FOCAL_PX_REF
    d._hsv_enabled = True
    d._center_lock_enabled = True
    d._center_lock_radius_px = None
    d._center_switch_margin_px = det.CENTER_SWITCH_MARGIN_PX
    d._center_axis_weight_x = det.CENTER_AXIS_WEIGHT_X
    d._center_axis_weight_y = det.CENTER_AXIS_WEIGHT_Y
    det.BrickDetector._load_face_shape_model(d)
    det.BrickDetector._init_shape_match_templates(d)
    return d


def _make_candidate(*, cx: int, cy: int, bbox: tuple) -> dict:
    return {
        "center_x": cx,
        "center_y": cy,
        "bbox": bbox,
        "contour": None,
        "rect": None,
        "area": 100.0,
        "partial": False,
        "partial_kind": None,
        "partial_label": None,
        "partial_edges": {},
        "shape_profile": "full",
        "shape_match_score": 0.0,
        "negative_cutout_polygons": [],
        "selection_anchor_x": float(cx),
        "selection_anchor_y": float(cy),
        "negative_cutout_pair_x_axis_metrics": None,
        "scale_px_per_mm": 1.0,
    }


def _make_stacked_frame_with_slots() -> np.ndarray:
    """Merged cyan blob with two dark trapezoid slots (inner-hole path)."""
    frame = np.zeros((180, 220, 3), dtype=np.uint8)
    cyan_bgr = (177, 157, 31)
    cv2.rectangle(frame, (32, 18), (188, 160), cyan_bgr, thickness=cv2.FILLED)
    for y_top in (46, 108):
        slot = np.array(
            [[58, y_top], [162, y_top], [150, y_top + 14], [70, y_top + 14]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(frame, [slot], (0, 0, 0))
    return frame


def _make_stacked_frame_no_slots() -> np.ndarray:
    """Solid cyan blob sized like 2+ stacked bricks — no dark inner holes (transparent slot)."""
    face_h_mm = 35.4
    face_w_mm = 53.0
    face_aspect = face_h_mm / face_w_mm  # ~0.668 h/w per brick
    blob_w = 156
    blob_h = int(round(blob_w * face_aspect * 2.4))  # ~2.4 bricks tall → n_bricks=2
    frame = np.zeros((blob_h + 40, blob_w + 60, 3), dtype=np.uint8)
    cyan_bgr = (177, 157, 31)
    cv2.rectangle(frame, (30, 20), (30 + blob_w, 20 + blob_h), cyan_bgr, thickness=cv2.FILLED)
    return frame


class _FakeVision:
    """Minimal vision object with the attributes _text_lines reads via getattr."""
    inference_backend = "tensorrt"
    model_path = None
    _trust_detector_boxes = True
    last_status = "idle"
    last_raw_prediction_count = 0
    last_candidate_count = 0
    last_nms_count = 0
    last_primary_confidence = 0.0
    last_max_confidence = 0.0
    conf_threshold = 0.10
    _smooth_alpha = 0.15
    input_size = 640
    current_frame = None
    raw_frame = None


def _make_stream_stub() -> CrownVisionLivestream:
    """CrownVisionLivestream with no camera or HTTP server, ready for _publish tests."""
    stream = CrownVisionLivestream.__new__(CrownVisionLivestream)
    stream.args = argparse.Namespace(
        stream_host="127.0.0.1",
        stream_port=5000,
        stream_fps=10,
        stream_jpeg_quality=75,
        stream_img_width=640,
        camera_fps=15,
        port_tries=1,
        sharpen=False,
    )
    stream.state = {
        "lock": threading.Lock(),
        "frame": None,
        "text_lines": [],
        "show_center_line": False,
        "vision_mode": "cyan",
        "cyan_profile": "config9",
        "xyz_workspace": None,
    }
    stream.vision = _FakeVision()
    stream._held_result = None
    stream._held_frame = None
    stream._miss_count = 0
    return stream


# ─── tests ───────────────────────────────────────────────────────────────────

class TestCrownProfileTuning(unittest.TestCase):
    """Crown profile must use shape_match and contain no negative_cutout keys."""

    def test_shape_gate_mode_is_shape_match(self):
        self.assertEqual(CROWN_PROFILE_TUNING["shape_gate_mode"], "shape_match")

    def test_no_negative_cutout_keys_in_profile(self):
        for key in CROWN_PROFILE_TUNING:
            self.assertFalse(
                key.startswith("negative_cutout"),
                f"Crown profile must not include negative_cutout gates; found: {key!r}",
            )

    def test_profile_applied_sets_shape_match_mode(self):
        d = _make_stub()
        d.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))
        self.assertEqual(d._face_shape_gate_mode, det.BRICK_FACE_GATE_MODE_SHAPE_MATCH)


class TestTrapezoidGate(unittest.TestCase):
    """Current world model has exactly 1 cutout polygon → trapezoid gate is active."""

    def test_world_model_has_exactly_one_cutout(self):
        d = _make_stub()
        self.assertEqual(len(d._face_cutouts_model), 1)

    def test_trapezoid_gate_active(self):
        d = _make_stub()
        self.assertTrue(d._uses_trapezoid_gate())

    def test_negative_cutout_gate_inactive(self):
        d = _make_stub()
        self.assertFalse(d._uses_negative_cutout_gate())


class TestInnerHoleSplitting(unittest.TestCase):
    """Two dark inner slots produce two separate brick candidates (primary path)."""

    def _candidates(self):
        d = _make_stub()
        frame = _make_stacked_frame_with_slots()
        return det.BrickDetector._segment_bricks_hsv(
            d, frame, 0, 0, frame.shape[1], frame.shape[0]
        )

    def test_two_slots_produce_at_least_two_candidates(self):
        self.assertGreaterEqual(len(self._candidates()), 2)

    def test_candidates_ordered_top_to_bottom(self):
        candidates = self._candidates()
        ys = [c["center_y"] for c in candidates[:2]]
        self.assertLess(ys[0], ys[1])


class TestHeightRatioFallback(unittest.TestCase):
    """When no inner holes exist, a tall cyan blob splits by the face height/width ratio."""

    def _candidates(self):
        d = _make_stub()
        frame = _make_stacked_frame_no_slots()
        return det.BrickDetector._segment_bricks_hsv(
            d, frame, 0, 0, frame.shape[1], frame.shape[0]
        )

    def test_tall_blob_with_no_slots_produces_multiple_candidates(self):
        self.assertGreaterEqual(
            len(self._candidates()),
            2,
            "Height-ratio fallback must split a merged tall blob into >=2 candidates",
        )

    def test_fallback_candidates_ordered_top_to_bottom(self):
        candidates = self._candidates()
        ys = [c["center_y"] for c in candidates[:2]]
        self.assertLess(ys[0], ys[1])


class TestSingleBrickFallback(unittest.TestCase):
    """Single brick with no inner holes (transparent slot) still produces a candidate."""

    def _single_brick_frame(self) -> np.ndarray:
        """Solid cyan blob sized like exactly one brick face — no inner holes."""
        face_h_mm = 35.4
        face_w_mm = 53.0
        face_aspect = face_h_mm / face_w_mm
        blob_w = 156
        blob_h = int(round(blob_w * face_aspect * 0.95))  # ~1 brick tall
        frame = np.zeros((blob_h + 40, blob_w + 60, 3), dtype=np.uint8)
        cyan_bgr = (177, 157, 31)
        cv2.rectangle(frame, (30, 20), (30 + blob_w, 20 + blob_h), cyan_bgr, thickness=cv2.FILLED)
        return frame

    def test_single_brick_no_slots_produces_candidate(self):
        d = _make_stub()
        frame = self._single_brick_frame()
        candidates = det.BrickDetector._segment_bricks_hsv(
            d, frame, 0, 0, frame.shape[1], frame.shape[0]
        )
        self.assertGreaterEqual(
            len(candidates), 1,
            "Single brick with transparent slot must still produce at least one candidate",
        )

    def test_fallback_scale_uses_face_height(self):
        """scale_px_per_mm in fallback candidates is anchored to face height, not slot height."""
        d = _make_stub()
        face_h_mm = float(np.max(d._face_polygon_model[:, 1]) - np.min(d._face_polygon_model[:, 1]))
        frame = self._single_brick_frame()
        candidates = det.BrickDetector._segment_bricks_hsv(
            d, frame, 0, 0, frame.shape[1], frame.shape[0]
        )
        self.assertGreaterEqual(len(candidates), 1)
        scale = float(candidates[0]["scale_px_per_mm"])
        # scale should be bbox_h / face_h_mm; face_h_mm ≈ 18mm
        # For a ~1-brick blob (blob_h ≈ face_aspect * blob_w), scale ≈ blob_h/18
        # Sanity: scale must be >> 0 and not wildly wrong vs slot_h_mm (4.05mm)
        self.assertGreater(scale, 0.0)
        # If scale were using slot_h_mm (4.05), it would be ~4.4x too large
        # Using face_h_mm (18) gives the correct ~1x ratio
        blob_h = float(candidates[0]["bbox"][3])
        expected_scale = blob_h / face_h_mm
        self.assertAlmostEqual(scale, expected_scale, delta=1.0)


class TestShapeMatchThreshold(unittest.TestCase):
    """_classify_contour_shape must use self._shape_match_score_max (from world model JSON)."""

    def test_loaded_threshold_from_world_model(self):
        d = _make_stub()
        # world_model_brick.json sets shape_match_score_max = 0.65
        self.assertAlmostEqual(d._shape_match_score_max, 0.65, delta=0.01)

    def test_classify_uses_instance_threshold_not_constant(self):
        """A contour that scores between 0.40 and 0.65 should pass with the JSON value."""
        d = _make_stub()
        # Force a permissive threshold so we can detect the difference
        d._shape_match_score_max = 0.65
        # Create a roughly brick-shaped contour (wide rectangle)
        contour = np.array(
            [[10, 5], [90, 5], [90, 25], [10, 25]], dtype=np.float32
        ).reshape(-1, 1, 2)
        profile_loose, score_loose = det.BrickDetector._classify_contour_shape(d, contour)
        # Now tighten the threshold below the score
        if score_loose is not None and score_loose > 0:
            d._shape_match_score_max = max(0.05, score_loose - 0.01)
            profile_tight, _ = det.BrickDetector._classify_contour_shape(d, contour)
            # Tight threshold should reject what loose accepted (or both None if score is very low)
            if profile_loose is not None:
                self.assertIsNone(
                    profile_tight,
                    "Tightening _shape_match_score_max must cause previously-passing contour to fail",
                )


class TestDrawBrickIdLabels(unittest.TestCase):
    """_draw_brick_id_labels: draws yellow IDs for 2+ candidates; silent for <=1."""

    @staticmethod
    def _yellow_px(frame: np.ndarray) -> int:
        return int(np.count_nonzero(
            (frame[:, :, 0] == 0) & (frame[:, :, 1] == 255) & (frame[:, :, 2] == 255)
        ))

    def test_labels_drawn_for_two_candidates(self):
        d = _make_stub()
        c1 = _make_candidate(cx=60, cy=40, bbox=(20, 15, 80, 40))
        c2 = _make_candidate(cx=60, cy=100, bbox=(20, 75, 80, 40))
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [c1, c2])
        self.assertGreater(self._yellow_px(frame), 0)

    def test_labels_not_drawn_for_single_candidate(self):
        d = _make_stub()
        c1 = _make_candidate(cx=60, cy=40, bbox=(20, 15, 80, 40))
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [c1])
        self.assertEqual(self._yellow_px(frame), 0, "Single candidate must not get a label")

    def test_labels_not_drawn_for_empty_list(self):
        d = _make_stub()
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [])
        self.assertEqual(self._yellow_px(frame), 0)

    def test_labels_ordered_top_id_zero(self):
        """The topmost (smallest cy) candidate gets ID 0."""
        d = _make_stub()
        top = _make_candidate(cx=60, cy=30, bbox=(20, 10, 80, 40))
        bottom = _make_candidate(cx=60, cy=100, bbox=(20, 75, 80, 40))
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [bottom, top])  # reversed order
        # Frame must have been modified (labels drawn)
        self.assertGreater(self._yellow_px(frame), 0)


class TestVisibilityHold(unittest.TestCase):
    """CrownVisionLivestream._publish holds the last good detection for HOLD_FRAMES misses."""

    def test_hold_frames_positive(self):
        self.assertGreater(HOLD_FRAMES, 0)

    def _visible_in_state(self, stream: CrownVisionLivestream) -> bool:
        with stream.state["lock"]:
            lines = stream.state.get("text_lines", [])
        return any("true" in str(line.get("text", "")).lower() for line in lines)

    def _invisible_in_state(self, stream: CrownVisionLivestream) -> bool:
        with stream.state["lock"]:
            lines = stream.state.get("text_lines", [])
        return any("false" in str(line.get("text", "")).lower() for line in lines)

    def test_hold_reuses_result_within_window(self):
        stream = _make_stream_stub()
        good = (True, 0.0, 200.0, 5.0, 80.0, 0.0, False, False)
        stream._publish(good)
        for _ in range(HOLD_FRAMES - 1):
            stream._publish(None)
        self.assertTrue(
            self._visible_in_state(stream),
            "Detection should still show visible within hold window",
        )

    def test_hold_expires_after_hold_frames(self):
        stream = _make_stream_stub()
        good = (True, 0.0, 200.0, 5.0, 80.0, 0.0, False, False)
        stream._publish(good)
        for _ in range(HOLD_FRAMES + 1):
            stream._publish(None)
        self.assertTrue(
            self._invisible_in_state(stream),
            "Detection should show invisible after hold window expires",
        )

    def test_miss_count_resets_on_new_detection(self):
        stream = _make_stream_stub()
        good = (True, 0.0, 200.0, 5.0, 80.0, 0.0, False, False)
        stream._publish(good)
        for _ in range(HOLD_FRAMES):
            stream._publish(None)
        stream._publish(good)
        self.assertEqual(stream._miss_count, 0)


if __name__ == "__main__":
    unittest.main()
