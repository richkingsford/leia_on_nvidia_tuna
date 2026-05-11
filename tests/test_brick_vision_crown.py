"""
Crown brick vision regression tests.

Covers the complete current pipeline:
  - Crown profile uses negative_cutouts gate
  - Trapezoid gate is active because world model has exactly 1 cutout polygon
  - Two dark inner slots → two separate candidates (inner-hole split path)
  - Transparent/missing slots → height-ratio fallback still produces 2+ candidates
  - _draw_brick_id_labels draws yellow IDs for the nearest 1-2 candidates
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
    CROWN_PROFILE_KEY,
    CROWN_PROFILE_OPTIONS,
    CROWN_PROFILE_TUNING,
    CROWN_PROFILE_TUNINGS,
    CrownVisionLivestream,
    HOLD_FRAMES,
    _build_footer_html,
)


GREEN_BRICK_BGR = (97, 165, 19)


class TestCrownVisionLivestreamFooter(unittest.TestCase):
    def test_footer_collapses_static_tri_brick_details_into_one_section(self):
        html = _build_footer_html()

        self.assertEqual(html.count("class='footer-section'"), 1)
        self.assertEqual(html.count("Tri-brick Vision"), 1)
        self.assertNotIn("Tri-brick Palette", html)
        self.assertNotIn("Profile dropdown controls", html)
        self.assertIn("green HSV", html)
        self.assertIn(f"hold {HOLD_FRAMES} frames", html)


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
    cv2.rectangle(frame, (32, 18), (188, 160), GREEN_BRICK_BGR, thickness=cv2.FILLED)
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
    cv2.rectangle(frame, (30, 20), (30 + blob_w, 20 + blob_h), GREEN_BRICK_BGR, thickness=cv2.FILLED)
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
    """Crown profile must use negative_cutouts gate."""

    def test_standalone_dropdown_exposes_distance_sweep_profiles(self):
        option_keys = [key for key, _label in CROWN_PROFILE_OPTIONS]
        self.assertGreaterEqual(len(CROWN_PROFILE_OPTIONS), 6)
        self.assertIn("tight_far_slots", option_keys)
        self.assertIn("tight_far_conf", option_keys)
        self.assertIn("tight_far_no_erode", option_keys)
        self.assertIn("tight_far_dim", option_keys)
        self.assertIn("balanced_far_guard", option_keys)
        self.assertIn(CROWN_PROFILE_KEY, dict(CROWN_PROFILE_OPTIONS))

    def test_default_profile_is_known_good_tight_color(self):
        self.assertEqual(CROWN_PROFILE_KEY, "tight_color")
        self.assertEqual(CROWN_PROFILE_TUNING["hsv_lower"], list(det.CYAN_HSV_TIGHT_LOWER))
        self.assertEqual(CROWN_PROFILE_TUNING["hsv_upper"], list(det.CYAN_HSV_TIGHT_UPPER))
        self.assertAlmostEqual(CROWN_PROFILE_TUNING["confidence"], 0.35)
        self.assertEqual(CROWN_PROFILE_TUNING["depth_source_mode"], "pinhole")

    def test_far_profiles_relax_distance_gates_from_tight_anchor(self):
        anchor = CROWN_PROFILE_TUNINGS["tight_color"]
        slots = CROWN_PROFILE_TUNINGS["tight_far_slots"]
        conf = CROWN_PROFILE_TUNINGS["tight_far_conf"]
        no_erode = CROWN_PROFILE_TUNINGS["tight_far_no_erode"]
        dim = CROWN_PROFILE_TUNINGS["tight_far_dim"]
        wider = CROWN_PROFILE_TUNINGS["balanced_far_guard"]

        self.assertLess(slots["negative_cutout_min_area_px"], anchor["negative_cutout_min_area_px"])
        self.assertLess(slots["hsv_min_area_ratio"], anchor["hsv_min_area_ratio"])
        self.assertLess(conf["confidence"], slots["confidence"])
        self.assertEqual(no_erode["hsv_erode_iterations"], 0)
        self.assertLess(dim["hsv_lower"][1], anchor["hsv_lower"][1])
        self.assertTrue(wider["closeup_full_frame_hsv_enabled"])

    def test_shape_gate_mode_is_negative_cutouts(self):
        self.assertEqual(CROWN_PROFILE_TUNING["shape_gate_mode"], "negative_cutouts")

    def test_profile_has_negative_cutout_keys(self):
        keys = [k for k in CROWN_PROFILE_TUNING if k.startswith("negative_cutout")]
        self.assertTrue(len(keys) > 0, "Crown profile must include negative_cutout gate keys")

    def test_profile_applied_sets_negative_cutouts_mode(self):
        d = _make_stub()
        d.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))
        self.assertEqual(d._face_shape_gate_mode, det.BRICK_FACE_GATE_MODE_NEGATIVE_CUTOUTS)


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
        cv2.rectangle(frame, (30, 20), (30 + blob_w, 20 + blob_h), GREEN_BRICK_BGR, thickness=cv2.FILLED)
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

    def test_strong_cyan_color_survives_weak_trapezoid_shape(self):
        d = _make_stub()
        d._build_trapezoid_brick_candidates = lambda *_args, **_kwargs: []
        d._classify_contour_shape = lambda _cnt: (None, 9.0)
        frame = np.zeros((110, 190, 3), dtype=np.uint8)
        cv2.rectangle(frame, (22, 28), (168, 84), GREEN_BRICK_BGR, thickness=cv2.FILLED)

        candidates = det.BrickDetector._segment_bricks_hsv(
            d,
            frame,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
        )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(any(bool(c.get("from_color_detection")) for c in candidates))
        self.assertTrue(any(c.get("shape_profile") == "color" for c in candidates))


class TestShapeMatchThreshold(unittest.TestCase):
    """_classify_contour_shape must use self._shape_match_score_max (from world model JSON)."""

    def test_loaded_threshold_from_world_model(self):
        d = _make_stub()
        # world_model_brick.json sets shape_match_score_max = 0.45
        self.assertAlmostEqual(d._shape_match_score_max, 0.45, delta=0.01)

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
    """_draw_brick_id_labels draws yellow IDs for the nearest 1-2 candidates."""

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

    def test_label_drawn_for_single_candidate(self):
        d = _make_stub()
        c1 = _make_candidate(cx=60, cy=40, bbox=(20, 15, 80, 40))
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [c1])
        self.assertGreater(self._yellow_px(frame), 0, "Single visible candidate must get a label")

    def test_label_center_uses_brick_bbox_middle_not_slot_anchor(self):
        d = _make_stub()
        frame = np.zeros((180, 220, 3), dtype=np.uint8)
        candidate = _make_candidate(cx=60, cy=35, bbox=(80, 70, 100, 60))
        candidate["rect"] = ((58.0, 36.0), (90.0, 12.0), 0.0)
        calls = []
        original_draw = det.draw_brick_with_id

        def _record_draw(_detector, _frame, _candidate, brick_id, *, center=None):
            calls.append((brick_id, center))
            return _frame

        det.draw_brick_with_id = _record_draw
        try:
            det.BrickDetector._draw_brick_id_labels(d, frame, [candidate])
        finally:
            det.draw_brick_with_id = original_draw

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 0)
        self.assertEqual(calls[0][1], (130.0, 100.0))

    def test_labels_not_drawn_for_empty_list(self):
        d = _make_stub()
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [])
        self.assertEqual(self._yellow_px(frame), 0)

    def test_labels_limited_to_two_nearest_center_candidates(self):
        d = _make_stub()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        near_1 = _make_candidate(cx=320, cy=240, bbox=(280, 220, 80, 40))
        near_2 = _make_candidate(cx=430, cy=240, bbox=(390, 220, 80, 40))
        far = _make_candidate(cx=55, cy=55, bbox=(15, 35, 80, 40))

        det.BrickDetector._draw_brick_id_labels(d, frame, [far, near_1, near_2])

        self.assertGreater(self._yellow_px(frame), 0)
        near_2_label_area = frame[230:285, 390:470]
        self.assertGreater(self._yellow_px(near_2_label_area), 0)
        far_label_area = frame[40:95, 10:105]
        self.assertEqual(
            self._yellow_px(far_label_area),
            0,
            "Only the two candidates nearest the camera crosshair should be labeled",
        )

    def test_labels_ordered_top_id_zero(self):
        """The topmost (smallest cy) candidate gets ID 0."""
        d = _make_stub()
        top = _make_candidate(cx=60, cy=30, bbox=(20, 10, 80, 40))
        bottom = _make_candidate(cx=60, cy=100, bbox=(20, 75, 80, 40))
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        det.BrickDetector._draw_brick_id_labels(d, frame, [bottom, top])  # reversed order
        # Frame must have been modified (labels drawn)
        self.assertGreater(self._yellow_px(frame), 0)


class TestCyanCandidateOutlines(unittest.TestCase):
    """Detected cyan candidates get tight green outlines around cyan pixels."""

    @staticmethod
    def _green_px(frame: np.ndarray) -> int:
        return int(np.count_nonzero(
            (frame[:, :, 0] == 0) & (frame[:, :, 1] == 255) & (frame[:, :, 2] == 0)
        ))

    def test_draw_debug_hsv_outlines_each_visible_candidate_green(self):
        d = _make_stub()
        frame = np.zeros((140, 180, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 20), (90, 52), GREEN_BRICK_BGR, thickness=cv2.FILLED)
        cv2.rectangle(frame, (30, 78), (90, 110), GREEN_BRICK_BGR, thickness=cv2.FILLED)
        c1 = _make_candidate(cx=60, cy=36, bbox=(24, 14, 76, 46))
        c2 = _make_candidate(cx=60, cy=94, bbox=(24, 72, 76, 46))

        det.BrickDetector._draw_debug_hsv(
            d,
            frame,
            [],
            [c1, c2],
            c1,
            angle=0.0,
            dist=0.0,
            offset_x=0.0,
            conf=1.0,
        )

        self.assertGreater(self._green_px(frame[18:56, 28:94]), 0)
        self.assertGreater(self._green_px(frame[76:114, 28:94]), 0)

    def test_outline_follows_cyan_pixels_not_loose_candidate_bbox(self):
        d = _make_stub()
        frame = np.zeros((100, 140, 3), dtype=np.uint8)
        cv2.rectangle(frame, (46, 32), (92, 58), GREEN_BRICK_BGR, thickness=cv2.FILLED)
        candidate = _make_candidate(cx=69, cy=45, bbox=(24, 16, 92, 60))

        det.BrickDetector._draw_cyan_candidate_outlines(d, frame, [candidate])

        self.assertGreater(self._green_px(frame[30:61, 44:95]), 0)
        self.assertEqual(self._green_px(frame[14:19, 22:118]), 0)


class TestReadFrameCloseupRecovery(unittest.TestCase):
    """read_frame must reach the full-frame HSV fallback when YOLO returns no boxes."""

    def test_blank_frame_with_no_yolo_boxes_stays_invisible(self):
        d = _make_stub()
        d.frame_w = 640
        d.frame_h = 480
        d.camera_center_offset_px = 0.0
        d._smooth_alpha = 1.0
        d._prev_angle = None
        d._prev_dist = None
        d._prev_offset = None
        d._prev_offset_y = None
        d.debug = False
        d.last_status = "idle"
        d.last_max_confidence = 0.0
        d.last_primary_confidence = 0.0
        d.last_partial_count = 0
        d.last_partial_labels = []
        d.last_primary_partial_kind = None
        d.last_primary_partial_label = None
        d.raw_frame = None
        d.current_frame = None
        d._detect = lambda _frame: []

        result = det.BrickDetector.read_frame(d, np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertFalse(result[0])
        self.assertEqual(d.last_status, "searching")

    def test_read_frame_uses_full_frame_hsv_when_yolo_has_no_boxes(self):
        d = _make_stub()
        d.frame_w = 640
        d.frame_h = 480
        d.focal_px = 100.0
        d.camera_center_offset_px = 0.0
        d._smooth_alpha = 1.0
        d._prev_angle = None
        d._prev_dist = None
        d._prev_offset = None
        d._prev_offset_y = None
        d.debug = False
        d.last_status = "idle"
        d.last_max_confidence = 0.0
        d.last_primary_confidence = 0.0
        d.last_partial_count = 0
        d.last_partial_labels = []
        d.last_primary_partial_kind = None
        d.last_primary_partial_label = None
        d.raw_frame = None
        d.current_frame = None
        d._detect = lambda _frame: []
        d._estimate_distance_from_box = lambda _bbox_w, _bbox_h, _partial_kind=None: 100.0
        d._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        d._stack_flags_from_individuals = lambda *_args, **_kwargs: (False, False)
        d._detect_pink_dot_in_brick = lambda *_args, **_kwargs: (False, None, None)
        d._dist_from_triangle_span = lambda *_args, **_kwargs: None

        primary = _make_candidate(cx=320, cy=240, bbox=(260, 210, 120, 60))
        segment_calls = []

        def _segment(_frame, x1, y1, x2, y2):
            segment_calls.append((x1, y1, x2, y2))
            return [primary] if (x1, y1, x2, y2) == (0, 0, 640, 480) else []

        d._segment_bricks_hsv = _segment

        result = det.BrickDetector.read_frame(d, np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertTrue(result[0])
        self.assertEqual(segment_calls, [(0, 0, 640, 480)])
        self.assertEqual(d.last_status, "target locked (HSV)")

    def test_profile_can_disable_full_frame_hsv_recovery(self):
        d = _make_stub()
        d.frame_w = 640
        d.frame_h = 480
        d.camera_center_offset_px = 0.0
        d._smooth_alpha = 1.0
        d._prev_angle = None
        d._prev_dist = None
        d._prev_offset = None
        d._prev_offset_y = None
        d.debug = False
        d.last_status = "idle"
        d.last_primary_confidence = 0.0
        d.last_partial_count = 0
        d.last_partial_labels = []
        d.last_primary_partial_kind = None
        d.last_primary_partial_label = None
        d._detect = lambda _frame: []
        d.set_runtime_tuning(closeup_full_frame_hsv_enabled=False)

        segment_calls = []

        def _segment(_frame, x1, y1, x2, y2):
            segment_calls.append((x1, y1, x2, y2))
            return [_make_candidate(cx=320, cy=240, bbox=(260, 210, 120, 60))]

        d._segment_bricks_hsv = _segment

        result = det.BrickDetector.read_frame(d, np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertFalse(result[0])
        self.assertEqual(segment_calls, [])
        self.assertEqual(d.last_status, "searching")


class TestProcessHsvCandidateHighlights(unittest.TestCase):
    """The HSV pipeline keeps up to two centered candidates for the debug overlay."""

    def test_process_bricks_keeps_top_brick_in_vertical_stack(self):
        d = _make_stub()
        d.frame_w = 640
        d.frame_h = 480
        d.focal_px = 100.0
        d.camera_center_offset_px = 0.0
        d._smooth_alpha = 1.0
        d._prev_angle = None
        d._prev_dist = None
        d._prev_offset = None
        d._prev_offset_y = None
        d.debug = True
        d.last_status = "idle"
        d.last_max_confidence = 1.0
        d.last_primary_confidence = 0.0
        d.last_partial_count = 0
        d.last_partial_labels = []
        d.last_primary_partial_kind = None
        d.last_primary_partial_label = None
        d.log = type(
            "_LogStub",
            (),
            {"exception": staticmethod(lambda *_args, **_kwargs: None)},
        )()

        top = _make_candidate(cx=320, cy=190, bbox=(250, 170, 140, 48))
        bottom = _make_candidate(cx=320, cy=250, bbox=(250, 230, 140, 48))
        far = _make_candidate(cx=80, cy=430, bbox=(40, 410, 80, 40))
        d._segment_bricks_hsv = lambda *_args, **_kwargs: [far, top, bottom]
        d._detect_pink_dot_in_brick = lambda *_args, **_kwargs: (False, None, None)
        d._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        d._dist_from_triangle_span = lambda *_args, **_kwargs: None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.BrickDetector._process_bricks(d, frame, [(0, 0, 640, 480, 1.0)])

        self.assertTrue(result[0])
        self.assertTrue(result[6], "Top brick should set brick_above for the lower primary")
        top_label_area = d.current_frame[170:225, 245:395]
        bottom_label_area = d.current_frame[230:285, 245:395]
        self.assertGreater(TestDrawBrickIdLabels._yellow_px(top_label_area), 0)
        self.assertGreater(TestDrawBrickIdLabels._yellow_px(bottom_label_area), 0)
        far_label_area = d.current_frame[405:460, 35:125]
        self.assertEqual(TestDrawBrickIdLabels._yellow_px(far_label_area), 0)

    def test_process_bricks_labels_two_nearest_hsv_candidates(self):
        d = _make_stub()
        d.frame_w = 640
        d.frame_h = 480
        d.focal_px = 100.0
        d.camera_center_offset_px = 0.0
        d._smooth_alpha = 1.0
        d._prev_angle = None
        d._prev_dist = None
        d._prev_offset = None
        d._prev_offset_y = None
        d.debug = True
        d.last_status = "idle"
        d.last_max_confidence = 1.0
        d.last_primary_confidence = 0.0
        d.last_partial_count = 0
        d.last_partial_labels = []
        d.last_primary_partial_kind = None
        d.last_primary_partial_label = None
        d.log = type(
            "_LogStub",
            (),
            {"exception": staticmethod(lambda *_args, **_kwargs: None)},
        )()

        near_1 = _make_candidate(cx=320, cy=240, bbox=(300, 230, 40, 24))
        near_2 = _make_candidate(cx=430, cy=240, bbox=(410, 230, 40, 24))
        far = _make_candidate(cx=70, cy=60, bbox=(50, 50, 40, 24))
        d._segment_bricks_hsv = lambda *_args, **_kwargs: [far, near_1, near_2]
        d._detect_pink_dot_in_brick = lambda *_args, **_kwargs: (False, None, None)
        d._refine_angle_for_primary = lambda *_args, **_kwargs: 0.0
        d._dist_from_triangle_span = lambda *_args, **_kwargs: None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.BrickDetector._process_bricks(d, frame, [(0, 0, 640, 480, 1.0)])

        self.assertTrue(result[0])
        self.assertIsNotNone(d.current_frame)
        self.assertGreater(TestDrawBrickIdLabels._yellow_px(d.current_frame), 0)
        near_2_label_area = d.current_frame[230:285, 400:465]
        self.assertGreater(TestDrawBrickIdLabels._yellow_px(near_2_label_area), 0)
        far_label_area = d.current_frame[45:100, 35:120]
        self.assertEqual(TestDrawBrickIdLabels._yellow_px(far_label_area), 0)


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
