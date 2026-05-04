"""
TensorRT Crown-Brick Detector
=============================

Uses a YOLOv8-nano model trained on real brick footage from Leia's OV2710 camera.
Works WITHOUT ArUco markers, handles cluttered backgrounds, and runs at 5-15 fps
on Jetson Orin Nano via the blessed TensorRT engine.

Same interface as the original BrickDetector:
    read() -> (found, angle, dist, offset_x, confidence, cam_height,
               brick_above, brick_below)

Usage:
    from helper_brick_detector_yolo import BrickDetector

Dependencies:
    pip install opencv-contrib-python numpy

    Runtime inference uses brick_yolo_v4_fast.plan. Training/export artifacts
    do not belong in this runtime path.
"""

import cv2
import numpy as np
import math
import os
import logging
import time
import colorsys
from pathlib import Path

from helper_camera_sources import candidate_camera_sources, existing_camera_nodes, open_opencv_camera_source

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Known brick dimensions (mm) — from world_model_brick.json
BRICK_WIDTH_MM = 53.0
BRICK_HEIGHT_MM = 35.4
BRICK_DEPTH_MM = 20.0

# Camera defaults (overridden at runtime from actual frame size)
DEFAULT_FRAME_W = 640
DEFAULT_FRAME_H = 480

# OV2710 camera parameters (100-degree FOV, 1/2.7" sensor)
# Focal length in pixels, calibrated at 640px frame width.
# Distance estimation now uses brick width as the primary apparent-size signal,
# with height retained as a small stabilizing fallback.
# Calibrated against ArUco gate targets (dist≈98-102mm at scoop position).
FOCAL_PX_REF = 580.0
FOCAL_REF_WIDTH = 640.0

# One permanent brick model: the current TensorRT plan.
TENSORRT_ENGINE_PATH = Path(__file__).resolve().parent / "brick_yolo_v4_fast.plan"
BRICK_MODEL_PATH = Path(__file__).resolve().parent / "world_model_brick.json"

# Detection confidence threshold
CONF_THRESHOLD = 0.15

# NMS IoU threshold
NMS_THRESHOLD = 0.45

# YOLO input size (model was exported at 640x640)
YOLO_INPUT_SIZE = 640

# ---------------------------------------------------------------------------
# HSV segmentation for individual cyan brick detection within YOLO boxes
# ---------------------------------------------------------------------------
CYAN_SHADE_HEXES = (
    "03929C",
    "068A9C",
    "028991",
    "0F949B",
    "018C96",
)


def _hex_to_opencv_hsv(hex_code: str) -> tuple[float, float, float]:
    text = str(hex_code or "").strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6 hex chars, got {hex_code!r}")
    red = int(text[0:2], 16) / 255.0
    green = int(text[2:4], 16) / 255.0
    blue = int(text[4:6], 16) / 255.0
    hue, sat, val = colorsys.rgb_to_hsv(red, green, blue)
    return float(hue * 180.0), float(sat * 255.0), float(val * 255.0)


def _cyan_palette_hsv_range(
    *,
    hue_margin: int,
    sat_margin: int,
    val_margin_lower: int,
    val_ceiling: int = 255,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    hsv_values = [_hex_to_opencv_hsv(hex_code) for hex_code in CYAN_SHADE_HEXES]
    min_h = min(value[0] for value in hsv_values)
    max_h = max(value[0] for value in hsv_values)
    min_s = min(value[1] for value in hsv_values)
    min_v = min(value[2] for value in hsv_values)
    lower = (
        max(0, int(round(float(min_h) - float(hue_margin)))),
        max(0, int(round(float(min_s) - float(sat_margin)))),
        max(0, int(round(float(min_v) - float(val_margin_lower)))),
    )
    upper = (
        min(179, int(round(float(max_h) + float(hue_margin)))),
        255,
        min(255, int(val_ceiling)),
    )
    return lower, upper


CYAN_HSV_TIGHT_LOWER, CYAN_HSV_TIGHT_UPPER = _cyan_palette_hsv_range(
    hue_margin=3,
    sat_margin=60,
    val_margin_lower=25,
    val_ceiling=255,
)
CYAN_HSV_BALANCED_LOWER, CYAN_HSV_BALANCED_UPPER = _cyan_palette_hsv_range(
    hue_margin=6,
    sat_margin=110,
    val_margin_lower=45,
    val_ceiling=255,
)
CYAN_HSV_WIDE_LOWER, CYAN_HSV_WIDE_UPPER = _cyan_palette_hsv_range(
    hue_margin=10,
    sat_margin=150,
    val_margin_lower=65,
    val_ceiling=255,
)

CYAN_HSV_LOWER = np.array(CYAN_HSV_BALANCED_LOWER)
CYAN_HSV_UPPER = np.array(CYAN_HSV_BALANCED_UPPER)

PINK_DOT_HEX_SHADES = ("B8304C", "BE2646", "B8244C")


def _pink_dot_hsv_range(
    *,
    hue_margin: int = 5,
    sat_margin: int = 40,
    val_margin_lower: int = 50,
    val_ceiling: int = 220,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    hsv_values = [_hex_to_opencv_hsv(h) for h in PINK_DOT_HEX_SHADES]
    min_h = min(v[0] for v in hsv_values)
    max_h = max(v[0] for v in hsv_values)
    min_s = min(v[1] for v in hsv_values)
    min_v = min(v[2] for v in hsv_values)
    lower = (
        max(0, int(round(float(min_h) - float(hue_margin)))),
        max(0, int(round(float(min_s) - float(sat_margin)))),
        max(0, int(round(float(min_v) - float(val_margin_lower)))),
    )
    upper = (
        min(179, int(round(float(max_h) + float(hue_margin)))),
        255,
        min(255, int(val_ceiling)),
    )
    return lower, upper


_PINK_DOT_HSV_LOWER_TUPLE, _PINK_DOT_HSV_UPPER_TUPLE = _pink_dot_hsv_range()
PINK_DOT_HSV_LOWER = np.array(_PINK_DOT_HSV_LOWER_TUPLE, dtype=np.uint8)
PINK_DOT_HSV_UPPER = np.array(_PINK_DOT_HSV_UPPER_TUPLE, dtype=np.uint8)
PINK_DOT_COLOR_BGR = (70, 38, 190)   # BE2646 in BGR
CONF_GATE_PCT = 65.0                 # minimum combined confidence to report a brick
PINK_DOT_CONF_BONUS = 25.0           # confidence bonus when pink dot is confirmed

HSV_ERODE_KERNEL = 5
HSV_ERODE_ITERATIONS = 2
HSV_MIN_AREA_RATIO = 0.05       # Min 5% of YOLO bbox area = real brick
HSV_CYAN_COVERAGE_MIN = 0.08    # Need 8% cyan coverage to engage HSV path
STACK_X_OVERLAP_RATIO = 0.6     # Same-column tolerance for above/below
STACK_Y_GAP_RATIO = 0.35        # Vertical gap threshold for above/below
# Reject cyan candidates whose contour shape is nowhere near expected
# brick face/top aspect ratios. Kept broad to allow perspective distortion.
BRICK_SHAPE_REL_ERROR_MAX = 0.75
BRICK_SHAPE_FILL_RATIO_MIN = 0.28
BRICK_FACE_MATCH_MAX_FULL = 0.40
BRICK_FACE_MATCH_MAX_PARTIAL = 0.55
FALLBACK_SHAPE_MIN_AREA_RATIO = 0.06
BRICK_FACE_GATE_MODE_DEFAULT = "negative_cutouts"
BRICK_FACE_GATE_MODE_NEGATIVE_CUTOUTS = "negative_cutouts"
BRICK_FACE_GATE_MODE_SHAPE_MATCH = "shape_match"
NEGATIVE_CUTOUT_CYAN_FILL_MAX = 0.20
NEGATIVE_CUTOUT_RING_CYAN_MIN = 0.52
NEGATIVE_CUTOUT_RING_DILATE_PX = 4
NEGATIVE_CUTOUT_MIN_AREA_PX = 18.0
NEGATIVE_CUTOUT_TRIANGLE_SIDE_RATIO_MAX = 2.0
NEGATIVE_CUTOUT_TRIANGLE_ANGLE_SPREAD_MAX_DEG = 75.0
NEGATIVE_CUTOUT_TRIANGLE_OVERLAP_MIN = 0.50
NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MIN = 0.45
NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MAX = 1.40
NEGATIVE_CUTOUT_PAIR_X_AXIS_MAX_ANGLE_DEG = 15.0
NEGATIVE_CUTOUT_FOCUS_Y_MIN_RATIO = None
NEGATIVE_CUTOUT_FOCUS_Y_MAX_RATIO = None
NEGATIVE_CUTOUT_FOCUS_WEIGHT = 0.0

# Cyan-target experiment: stabilize selection around centerpoint.
CENTER_LOCK_RADIUS_RATIO = 0.18
CENTER_SWITCH_MARGIN_PX = 18.0
CENTER_PARTIAL_PENALTY = 2.0
CENTER_AXIS_WEIGHT_X = 1.0
CENTER_AXIS_WEIGHT_Y = 1.0
STACK_CENTER_Y_ROW_FILTER_ENABLED = True
STACK_CENTER_Y_ROW_MIN_CANDIDATES = 2
STACK_CENTER_Y_ROW_TOLERANCE_PX = 12.0
PARTIAL_EDGE_MARGIN_PX = 1
PARTIAL_COLOR_BGR = (0, 165, 255)
PARTIAL_LABEL_TOP = "TOP HALF"
PARTIAL_LABEL_BOTTOM = "LOWER PARTIAL"
PARTIAL_LABEL_VERTICAL = "TOP/BOTTOM"
PARTIAL_LABEL_LEFT = "LEFT PARTIAL"
PARTIAL_LABEL_RIGHT = "RIGHT PARTIAL"
PARTIAL_LABEL_GENERIC = "PARTIAL"
NEGATIVE_TRIANGLE_COLOR_BGR = (0, 255, 0)


class BrickDetector:
    """
    TensorRT-backed crown-brick detector.

    Provides the same read() interface expected by telemetry_brick.py
    and runtime training/orchestration scripts.

    Uses the committed brick_yolo_v4_fast.plan TensorRT engine. There is no
    operator-facing model override in the runtime path.
    """

    def __init__(self, debug=True, save_folder=None, speed_optimize=False,
                 model_path=None, focal_px=None):
        self.debug = debug
        self.speed_optimize = speed_optimize
        self.log = logging.getLogger("BrickVisionYOLO")

        # Headless detection
        self.headless = True
        if self.debug and (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self.headless = False

        self.save_folder = save_folder
        self.current_frame = None
        self.raw_frame = None
        if self.save_folder and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        if model_path:
            self.log.warning(
                "Ignoring brick model override %s; using required TensorRT engine %s",
                model_path,
                TENSORRT_ENGINE_PATH,
            )
        self.net = None
        self._trt_engine = None
        self.inference_backend = "tensorrt"
        self._init_inference_backend()
        self._trust_detector_boxes = True
        self.conf_threshold = float(CONF_THRESHOLD)
        self.nms_threshold = float(NMS_THRESHOLD)
        self.input_size = int(YOLO_INPUT_SIZE)
        self.last_raw_prediction_count = 0
        self.last_candidate_count = 0
        self.last_nms_count = 0
        self.last_primary_confidence = 0.0
        self.last_max_confidence = 0.0
        self.last_status = "idle"
        self.last_partial_count = 0
        self.last_partial_labels = []
        self.last_primary_partial_kind = None
        self.last_primary_partial_label = None
        # Keep debug view readable when low confidence rescue profiles are active.
        self.debug_show_all_candidates = False
        self.debug_max_boxes = 1

        # Camera setup: prefer Jetson-native GStreamer/NVIDIA conversion pipelines.
        self.cap = None
        self.camera_source = None
        tried_sources = []
        for source in candidate_camera_sources(width=DEFAULT_FRAME_W, height=DEFAULT_FRAME_H):
            tried_sources.append(source)
            self.log.info("Attempting camera %s...", source)
            temp = open_opencv_camera_source(
                source,
                cv2,
                width=DEFAULT_FRAME_W,
                height=DEFAULT_FRAME_H,
            )
            if temp is not None and temp.isOpened():
                self.log.info("Connected to camera %s (%s)", source,
                              temp.getBackendName())
                self.cap = temp
                self.camera_source = source
                break
            if temp is not None:
                temp.release()

        if self.cap is None:
            existing_nodes = existing_camera_nodes()
            self.log.error(
                "No camera found. Tried sources: %s. Detected nodes: %s. Using dummy.",
                tried_sources,
                existing_nodes,
            )
            dummy_source = existing_nodes[0] if existing_nodes else 0
            self.cap = open_opencv_camera_source(
                dummy_source,
                cv2,
                width=DEFAULT_FRAME_W,
                height=DEFAULT_FRAME_H,
            )
            if self.cap is None:
                self.cap = cv2.VideoCapture(dummy_source)

        # Frame dimensions
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or DEFAULT_FRAME_W
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or DEFAULT_FRAME_H

        # Camera parameters — scale focal length with resolution
        if focal_px is not None:
            self.focal_px = float(focal_px)
        else:
            self.focal_px = FOCAL_PX_REF * (self.frame_w / FOCAL_REF_WIDTH)
        self.camera_center_offset_px = 0.0
        self.log.info("focal_px=%.1f (frame %dx%d). Use calibrate_focal() or "
                      "set_runtime_tuning(focal_px=N) to adjust.",
                      self.focal_px, self.frame_w, self.frame_h)

        # HSV segmentation for individual cyan brick splitting
        self._hsv_lower = CYAN_HSV_LOWER.copy()
        self._hsv_upper = CYAN_HSV_UPPER.copy()
        self._hsv_enabled = True
        self._hsv_erode_iterations = HSV_ERODE_ITERATIONS

        # Center-target lock for cyan mode: prefer center-most candidate while
        # avoiding frame-to-frame jitter between nearly equivalent bricks.
        self._center_lock_enabled = True
        self._center_lock_radius_px = None
        self._center_switch_margin_px = float(CENTER_SWITCH_MARGIN_PX)
        self._center_partial_penalty = float(CENTER_PARTIAL_PENALTY)
        self._center_axis_weight_x = float(CENTER_AXIS_WEIGHT_X)
        self._center_axis_weight_y = float(CENTER_AXIS_WEIGHT_Y)
        self._center_lock_prev_center = None

        # Canonical brick face shape (world model) for debug/overlay rendering.
        self._face_polygon_model = None
        self._face_cutouts_model = []
        self._face_lines_model = []
        self._face_shape_templates = {}
        self._face_shape_gate_mode = BRICK_FACE_GATE_MODE_DEFAULT
        self._negative_cutout_cyan_fill_max = float(NEGATIVE_CUTOUT_CYAN_FILL_MAX)
        self._negative_cutout_ring_cyan_min = float(NEGATIVE_CUTOUT_RING_CYAN_MIN)
        self._negative_cutout_ring_dilate_px = int(NEGATIVE_CUTOUT_RING_DILATE_PX)
        self._negative_cutout_min_area_px = float(NEGATIVE_CUTOUT_MIN_AREA_PX)
        self._negative_cutout_triangle_side_ratio_max = float(
            NEGATIVE_CUTOUT_TRIANGLE_SIDE_RATIO_MAX
        )
        self._negative_cutout_triangle_angle_spread_max_deg = float(
            NEGATIVE_CUTOUT_TRIANGLE_ANGLE_SPREAD_MAX_DEG
        )
        self._negative_cutout_triangle_overlap_min = float(
            NEGATIVE_CUTOUT_TRIANGLE_OVERLAP_MIN
        )
        self._negative_cutout_triangle_area_ratio_min = float(
            NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MIN
        )
        self._negative_cutout_triangle_area_ratio_max = float(
            NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MAX
        )
        self._negative_cutout_pair_x_axis_max_angle_deg = float(
            NEGATIVE_CUTOUT_PAIR_X_AXIS_MAX_ANGLE_DEG
        )
        self._negative_cutout_focus_y_min_ratio = NEGATIVE_CUTOUT_FOCUS_Y_MIN_RATIO
        self._negative_cutout_focus_y_max_ratio = NEGATIVE_CUTOUT_FOCUS_Y_MAX_RATIO
        self._negative_cutout_focus_weight = float(NEGATIVE_CUTOUT_FOCUS_WEIGHT)
        self._load_face_shape_model()
        self._init_shape_match_templates()

        # Temporal smoothing
        self._prev_angle = None
        self._prev_dist = None
        self._prev_offset = None
        self._prev_offset_y = None
        self._smooth_alpha = 0.3  # EMA weight for new values (lower = smoother)

        # Camera read failure log throttling to avoid console spam.
        self._camera_read_fail_count = 0
        self._last_camera_read_fail_log_at = 0.0
        self._camera_read_fail_log_interval_s = 2.0

    def set_runtime_tuning(self, confidence=None, smoothing_alpha=None,
                           nms_threshold=None, focal_px=None,
                           hsv_lower=None, hsv_upper=None,
                           hsv_enabled=None, hsv_erode_iterations=None,
                           shape_gate_mode=None,
                           negative_cutout_cyan_fill_max=None,
                           negative_cutout_ring_cyan_min=None,
                           negative_cutout_ring_dilate_px=None,
                           negative_cutout_min_area_px=None,
                           negative_cutout_triangle_side_ratio_max=None,
                           negative_cutout_triangle_angle_spread_max_deg=None,
                           negative_cutout_triangle_overlap_min=None,
                           negative_cutout_triangle_area_ratio_min=None,
                           negative_cutout_triangle_area_ratio_max=None,
                           negative_cutout_pair_x_axis_max_angle_deg=None,
                           negative_cutout_focus_y_min_ratio=None,
                           negative_cutout_focus_y_max_ratio=None,
                           negative_cutout_focus_weight=None,
                           center_lock_enabled=None,
                           center_lock_radius_px=None,
                           center_switch_margin_px=None,
                           center_axis_weight_x=None,
                           center_axis_weight_y=None):
        if confidence is not None:
            try:
                conf_val = float(confidence)
                self.conf_threshold = max(0.0, min(1.0, conf_val))
            except (TypeError, ValueError):
                pass
        if smoothing_alpha is not None:
            try:
                alpha_val = float(smoothing_alpha)
                self._smooth_alpha = max(0.0, min(1.0, alpha_val))
            except (TypeError, ValueError):
                pass
        if nms_threshold is not None:
            try:
                nms_val = float(nms_threshold)
                self.nms_threshold = max(0.0, min(1.0, nms_val))
            except (TypeError, ValueError):
                pass
        if focal_px is not None:
            try:
                self.focal_px = max(1.0, float(focal_px))
            except (TypeError, ValueError):
                pass
        if hsv_lower is not None:
            try:
                self._hsv_lower = np.array(hsv_lower, dtype=np.uint8)
            except (TypeError, ValueError):
                pass
        if hsv_upper is not None:
            try:
                self._hsv_upper = np.array(hsv_upper, dtype=np.uint8)
            except (TypeError, ValueError):
                pass
        if hsv_enabled is not None:
            self._hsv_enabled = bool(hsv_enabled)
        if hsv_erode_iterations is not None:
            try:
                self._hsv_erode_iterations = max(0, int(hsv_erode_iterations))
            except (TypeError, ValueError):
                pass
        if shape_gate_mode is not None:
            mode_text = str(shape_gate_mode or "").strip().lower()
            if mode_text in {
                BRICK_FACE_GATE_MODE_DEFAULT,
                BRICK_FACE_GATE_MODE_NEGATIVE_CUTOUTS,
                BRICK_FACE_GATE_MODE_SHAPE_MATCH,
            }:
                self._face_shape_gate_mode = mode_text
        if negative_cutout_cyan_fill_max is not None:
            try:
                value = float(negative_cutout_cyan_fill_max)
                self._negative_cutout_cyan_fill_max = max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                pass
        if negative_cutout_ring_cyan_min is not None:
            try:
                value = float(negative_cutout_ring_cyan_min)
                self._negative_cutout_ring_cyan_min = max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                pass
        if negative_cutout_ring_dilate_px is not None:
            try:
                self._negative_cutout_ring_dilate_px = max(
                    1,
                    int(negative_cutout_ring_dilate_px),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_min_area_px is not None:
            try:
                self._negative_cutout_min_area_px = max(
                    1.0,
                    float(negative_cutout_min_area_px),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_triangle_side_ratio_max is not None:
            try:
                self._negative_cutout_triangle_side_ratio_max = max(
                    1.0,
                    float(negative_cutout_triangle_side_ratio_max),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_triangle_angle_spread_max_deg is not None:
            try:
                self._negative_cutout_triangle_angle_spread_max_deg = max(
                    1.0,
                    float(negative_cutout_triangle_angle_spread_max_deg),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_triangle_overlap_min is not None:
            try:
                self._negative_cutout_triangle_overlap_min = max(
                    0.0,
                    min(1.0, float(negative_cutout_triangle_overlap_min)),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_triangle_area_ratio_min is not None:
            try:
                self._negative_cutout_triangle_area_ratio_min = max(
                    0.05,
                    float(negative_cutout_triangle_area_ratio_min),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_triangle_area_ratio_max is not None:
            try:
                self._negative_cutout_triangle_area_ratio_max = max(
                    max(0.05, self._negative_cutout_triangle_area_ratio_min),
                    float(negative_cutout_triangle_area_ratio_max),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_pair_x_axis_max_angle_deg is not None:
            try:
                self._negative_cutout_pair_x_axis_max_angle_deg = max(
                    0.0,
                    min(89.0, float(negative_cutout_pair_x_axis_max_angle_deg)),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_focus_y_min_ratio is not None:
            try:
                self._negative_cutout_focus_y_min_ratio = max(
                    0.0,
                    min(1.0, float(negative_cutout_focus_y_min_ratio)),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_focus_y_max_ratio is not None:
            try:
                self._negative_cutout_focus_y_max_ratio = max(
                    0.0,
                    min(1.0, float(negative_cutout_focus_y_max_ratio)),
                )
            except (TypeError, ValueError):
                pass
        if negative_cutout_focus_weight is not None:
            try:
                self._negative_cutout_focus_weight = max(
                    0.0,
                    float(negative_cutout_focus_weight),
                )
            except (TypeError, ValueError):
                pass
        if center_lock_enabled is not None:
            self._center_lock_enabled = bool(center_lock_enabled)
        if center_lock_radius_px is not None:
            try:
                radius_val = float(center_lock_radius_px)
                self._center_lock_radius_px = max(0.0, radius_val)
            except (TypeError, ValueError):
                pass
        if center_switch_margin_px is not None:
            try:
                margin_val = float(center_switch_margin_px)
                self._center_switch_margin_px = max(0.0, margin_val)
            except (TypeError, ValueError):
                pass
        if center_axis_weight_x is not None:
            try:
                weight_x = float(center_axis_weight_x)
                self._center_axis_weight_x = max(0.01, weight_x)
            except (TypeError, ValueError):
                pass
        if center_axis_weight_y is not None:
            try:
                weight_y = float(center_axis_weight_y)
                self._center_axis_weight_y = max(0.01, weight_y)
            except (TypeError, ValueError):
                pass
        return {
            "conf_threshold": float(self.conf_threshold),
            "smooth_alpha": float(self._smooth_alpha),
            "nms_threshold": float(self.nms_threshold),
            "focal_px": float(self.focal_px),
            "hsv_lower": self._hsv_lower.tolist(),
            "hsv_upper": self._hsv_upper.tolist(),
            "hsv_enabled": self._hsv_enabled,
            "hsv_erode_iterations": self._hsv_erode_iterations,
            "shape_gate_mode": str(self._face_shape_gate_mode),
            "negative_cutout_cyan_fill_max": float(self._negative_cutout_cyan_fill_max),
            "negative_cutout_ring_cyan_min": float(self._negative_cutout_ring_cyan_min),
            "negative_cutout_ring_dilate_px": int(self._negative_cutout_ring_dilate_px),
            "negative_cutout_min_area_px": float(self._negative_cutout_min_area_px),
            "negative_cutout_pair_x_axis_max_angle_deg": float(
                self._negative_cutout_pair_x_axis_max_angle_deg
            ),
            "negative_cutout_focus_y_min_ratio": (
                None
                if self._negative_cutout_focus_y_min_ratio is None
                else float(self._negative_cutout_focus_y_min_ratio)
            ),
            "negative_cutout_focus_y_max_ratio": (
                None
                if self._negative_cutout_focus_y_max_ratio is None
                else float(self._negative_cutout_focus_y_max_ratio)
            ),
            "negative_cutout_focus_weight": float(self._negative_cutout_focus_weight),
            "center_lock_enabled": bool(self._center_lock_enabled),
            "center_lock_radius_px": (
                None
                if self._center_lock_radius_px is None
                else float(self._center_lock_radius_px)
            ),
            "center_switch_margin_px": float(self._center_switch_margin_px),
            "center_axis_weight_x": float(self._center_axis_weight_x),
            "center_axis_weight_y": float(self._center_axis_weight_y),
        }

    # ------------------------------------------------------------------
    # YOLO inference helpers
    # ------------------------------------------------------------------

    def _init_inference_backend(self):
        engine_path = TENSORRT_ENGINE_PATH
        if not engine_path.exists():
            raise FileNotFoundError(
                f"Required TensorRT brick model not found at {engine_path}."
            )
        try:
            from helper_tensorrt_yolo import TensorRTYoloEngine

            self._trt_engine = TensorRTYoloEngine(engine_path)
        except Exception as exc:
            raise RuntimeError(
                f"Required TensorRT brick model failed to load from {engine_path}: {exc}"
            ) from exc
        self.inference_backend = "tensorrt"
        self.model_path = str(engine_path)
        self.model_name = engine_path.name
        self.log.info("Loaded TensorRT YOLO engine from %s", engine_path)

    def _forward_yolo(self, blob):
        if self._trt_engine is not None:
            return self._trt_engine.infer(blob)
        raise RuntimeError("TensorRT brick model is not initialized.")

    def _letterbox(self, frame):
        """Resize frame to YOLO_INPUT_SIZE maintaining aspect ratio with padding."""
        h, w = frame.shape[:2]
        scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        pad_top = (YOLO_INPUT_SIZE - new_h) // 2
        pad_bottom = YOLO_INPUT_SIZE - new_h - pad_top
        pad_left = (YOLO_INPUT_SIZE - new_w) // 2
        pad_right = YOLO_INPUT_SIZE - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, scale, pad_left, pad_top

    def _detect(self, frame):
        """
        Run YOLO TensorRT inference on a frame.

        Returns:
            list of (x1, y1, x2, y2, confidence) tuples in original
            frame coordinates.
        """
        h_frame, w_frame = frame.shape[:2]

        # Letterbox to 640x640
        padded, scale, pad_left, pad_top = self._letterbox(frame)

        # Create blob: normalize [0,1], BGR→RGB
        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
            swapRB=True, crop=False
        )

        # Forward pass returns (1, 5, 8400).
        output = self._forward_yolo(blob)

        # Transpose to (8400, 5) — each row: [cx, cy, w, h, conf]
        output = output[0].T
        self.last_raw_prediction_count = int(output.shape[0]) if output.ndim == 2 else 0

        # Filter by confidence
        conf_threshold = float(getattr(self, "conf_threshold", CONF_THRESHOLD))
        nms_threshold = float(getattr(self, "nms_threshold", NMS_THRESHOLD))
        confs = output[:, 4]
        self.last_max_confidence = float(np.max(confs)) if confs.size else 0.0
        mask = confs > conf_threshold
        output = output[mask]
        confs = confs[mask]
        self.last_candidate_count = int(len(output))

        if len(output) == 0:
            self.last_nms_count = 0
            return []

        # Prepare boxes for NMS — cv2.dnn.NMSBoxes expects [x, y, w, h]
        boxes_for_nms = []
        for cx, cy, w, h, _ in output:
            boxes_for_nms.append([
                float(cx - w / 2),
                float(cy - h / 2),
                float(w),
                float(h),
            ])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, confs.tolist(),
            conf_threshold, nms_threshold
        )
        self.last_nms_count = int(len(indices)) if indices is not None else 0

        if len(indices) == 0:
            return []

        bricks = []
        for i in indices:
            idx = int(i) if np.ndim(i) == 0 else int(i[0])
            cx, cy, bw, bh = output[idx, :4]
            conf = float(confs[idx])

            # Undo letterbox: map from 640x640 padded space → original frame
            x1 = (cx - bw / 2 - pad_left) / scale
            y1 = (cy - bh / 2 - pad_top) / scale
            x2 = (cx + bw / 2 - pad_left) / scale
            y2 = (cy + bh / 2 - pad_top) / scale

            # Clamp to frame bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_frame, int(x2))
            y2 = min(h_frame, int(y2))

            if x2 > x1 and y2 > y1:
                bricks.append((x1, y1, x2, y2, conf))

        return bricks

    # ------------------------------------------------------------------
    # Brick geometry estimation (unchanged from original)
    # ------------------------------------------------------------------

    def _estimate_angle(self, frame, x1, y1, x2, y2):
        """Estimate brick rotation angle from the detected bounding box region."""
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return 0.0

        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        w, h = rect[1]
        if w < h:
            angle = angle - 90

        return angle

    def _estimate_distance(self, bbox_height_px):
        """Legacy height-only distance estimate.

        Kept as a compatibility fallback for tests and any old call sites that
        still pass only a height signal.
        """
        if bbox_height_px <= 0:
            return 999.0
        return (BRICK_HEIGHT_MM * self.focal_px) / float(bbox_height_px)

    def _estimate_distance_from_width(self, bbox_width_px):
        """Estimate distance from apparent brick width in the image."""
        if bbox_width_px <= 0:
            return 999.0
        return (BRICK_WIDTH_MM * self.focal_px) / float(bbox_width_px)

    def _estimate_offset_x_mm(self, center_x_px, dist_mm):
        """
        Estimate camera-space horizontal offset in mm from the camera X center
        to the detected brick center.

        Positive means the brick center is to the right in the image.
        """
        frame_center_x = float(self.frame_w) / 2.0 + float(self.camera_center_offset_px)
        offset_px = float(center_x_px) - frame_center_x
        try:
            focal_px = max(1e-6, float(self.focal_px))
            dist_signal = float(dist_mm)
        except (TypeError, ValueError):
            return 0.0
        return (offset_px * dist_signal) / focal_px

    def _effective_distance_height_px(self, bbox_height_px, partial_kind=None):
        """
        Convert observed bbox height into a full-brick equivalent height.

        Top/bottom partial profiles represent roughly half the face height,
        so scale them up before deriving the pixel-size distance proxy.
        """
        h_px = max(1.0, float(bbox_height_px))
        kind = str(partial_kind or "").strip().lower()
        if kind in {"top_half", "bottom_half"}:
            h_px *= 2.0
        return h_px

    def _effective_distance_width_px(self, bbox_width_px, partial_kind=None):
        """
        Convert observed bbox width into a full-brick equivalent width.

        For the current full-face and top/bottom partial profiles, width remains
        the most stable apparent-size cue, so no partial scaling is required.
        """
        _ = partial_kind
        return max(1.0, float(bbox_width_px))

    def _estimate_distance_from_box(self, bbox_width_px, bbox_height_px, partial_kind=None):
        """Estimate distance using width as the primary apparent-size signal.

        Width is more stable for the current brick face than height, which can
        fluctuate with the interlocking top/bottom teeth and close-up framing.
        Height is retained as a secondary stabilizer so partial detections and
        noisy boxes do not swing as hard.
        """
        eff_w = self._effective_distance_width_px(bbox_width_px, partial_kind)
        eff_h = self._effective_distance_height_px(bbox_height_px, partial_kind)
        width_dist = self._estimate_distance_from_width(eff_w) if eff_w > 0 else None
        height_dist = self._estimate_distance(eff_h) if eff_h > 0 else None
        if width_dist is None and height_dist is None:
            return 999.0
        if width_dist is None:
            return float(height_dist)
        if height_dist is None:
            return float(width_dist)
        return float((0.8 * float(width_dist)) + (0.2 * float(height_dist)))

    def _dist_from_triangle_span(self, cutout_polys):
        """Estimate dist from the pixel span between confirmed triangle outer edges.

        The outer left edge of the left triangle and the outer right edge of the
        right triangle together span BRICK_WIDTH_MM in world-model coordinates,
        so applying the pinhole formula gives a stable dist estimate that is
        independent of YOLO bounding-box noise.

        Returns None if fewer than two triangle polygons are available or the
        computed span is degenerate.
        """
        if not isinstance(cutout_polys, list) or len(cutout_polys) < 2:
            return None
        p0 = np.asarray(cutout_polys[0], dtype=np.float32)
        p1 = np.asarray(cutout_polys[1], dtype=np.float32)
        if p0.ndim != 2 or p0.shape[0] < 3 or p1.ndim != 2 or p1.shape[0] < 3:
            return None
        # Sort so p0 is the left triangle (smaller mean x).
        if float(np.mean(p0[:, 0])) > float(np.mean(p1[:, 0])):
            p0, p1 = p1, p0
        span_px = float(np.max(p1[:, 0])) - float(np.min(p0[:, 0]))
        if span_px < 2.0:
            return None
        return (BRICK_WIDTH_MM * self.focal_px) / span_px

    def _smooth(self, new_val, prev_val):
        """Exponential moving average for temporal smoothing."""
        if prev_val is None:
            return new_val
        return self._smooth_alpha * new_val + (1 - self._smooth_alpha) * prev_val

    def _estimate_cam_height(self, center_y_px, dist_signal):
        """
        Estimate camera-space vertical offset from image-space Y center.

        Returns raw pixel offset (not converted to mm):
        - 0 px at image vertical center
        - positive when brick center is below image center
        - negative when brick center is above image center
        """
        frame_center_y = float(self.frame_h) / 2.0
        offset_y_px = float(center_y_px) - frame_center_y
        return offset_y_px

    def _clear_partial_state(self):
        self.last_partial_count = 0
        self.last_partial_labels = []
        self.last_primary_partial_kind = None
        self.last_primary_partial_label = None

    def _partial_info_from_edges(self, *, left_touch, top_touch, right_touch, bottom_touch):
        left_touch = bool(left_touch)
        top_touch = bool(top_touch)
        right_touch = bool(right_touch)
        bottom_touch = bool(bottom_touch)
        partial = bool(left_touch or top_touch or right_touch or bottom_touch)
        kind = None
        label = None
        if partial:
            # Prioritize top/bottom clipping because that is the most important
            # operator signal when tracking stack halves in cyan mode.
            if top_touch and not bottom_touch:
                kind = "top_half"
                label = PARTIAL_LABEL_TOP
            elif bottom_touch and not top_touch:
                kind = "bottom_half"
                label = PARTIAL_LABEL_BOTTOM
            elif top_touch and bottom_touch:
                kind = "vertical_partial"
                label = PARTIAL_LABEL_VERTICAL
            elif left_touch and not right_touch:
                kind = "left_partial"
                label = PARTIAL_LABEL_LEFT
            elif right_touch and not left_touch:
                kind = "right_partial"
                label = PARTIAL_LABEL_RIGHT
            else:
                kind = "partial"
                label = PARTIAL_LABEL_GENERIC
        return {
            "partial": partial,
            "kind": kind,
            "label": label,
            "edges": {
                "left": left_touch,
                "top": top_touch,
                "right": right_touch,
                "bottom": bottom_touch,
            },
        }

    def _partial_info_for_crop_bbox(self, bx, by, bw, bh, crop_w, crop_h):
        margin = int(max(0, PARTIAL_EDGE_MARGIN_PX))
        return self._partial_info_from_edges(
            left_touch=int(bx) <= margin,
            top_touch=int(by) <= margin,
            right_touch=int(bx + bw) >= int(crop_w) - margin,
            bottom_touch=int(by + bh) >= int(crop_h) - margin,
        )

    def _partial_info_for_frame_box(self, x1, y1, x2, y2, frame_w, frame_h):
        margin = int(max(0, PARTIAL_EDGE_MARGIN_PX))
        return self._partial_info_from_edges(
            left_touch=int(x1) <= margin,
            top_touch=int(y1) <= margin,
            right_touch=int(x2) >= int(frame_w) - margin,
            bottom_touch=int(y2) >= int(frame_h) - margin,
        )

    def _set_partial_state(self, partial_items, *, primary_partial_kind=None, primary_partial_label=None):
        labels = []
        if isinstance(partial_items, list):
            for item in partial_items:
                if not isinstance(item, dict) or not bool(item.get("partial")):
                    continue
                label = str(item.get("label") or "").strip()
                if label:
                    labels.append(label)
        self.last_partial_labels = labels
        self.last_partial_count = len(labels)
        self.last_primary_partial_kind = (
            str(primary_partial_kind).strip() if primary_partial_kind else None
        )
        self.last_primary_partial_label = (
            str(primary_partial_label).strip() if primary_partial_label else None
        )

    def _smooth_angle(self, raw_angle, prev_angle):
        """
        Angle-aware EMA with jump clamping.

        Brick orientation is symmetric at 180°, so angles live in [-90, 90].
        A raw jump from e.g. -85° to +85° is really a 10° rotation (through
        the ±90° wrap), not a 170° flip.  This method:
        1. Computes the shortest angular delta (handling ±90° wrap).
        2. Clamps the delta to ±MAX_ANGLE_JUMP_PER_FRAME to reject noise.
        3. Applies EMA smoothing on the clamped result.
        """
        MAX_ANGLE_JUMP = 30.0  # degrees per frame
        if prev_angle is None:
            return raw_angle

        delta = raw_angle - prev_angle
        # Shortest path in [-90, 90] orientation space (period = 180°)
        if delta > 90.0:
            delta -= 180.0
        elif delta < -90.0:
            delta += 180.0

        # Clamp
        if delta > MAX_ANGLE_JUMP:
            delta = MAX_ANGLE_JUMP
        elif delta < -MAX_ANGLE_JUMP:
            delta = -MAX_ANGLE_JUMP

        clamped = prev_angle + delta
        # Normalise back to [-90, 90]
        while clamped > 90.0:
            clamped -= 180.0
        while clamped < -90.0:
            clamped += 180.0

        # EMA on the clamped value
        return self._smooth_alpha * clamped + (1 - self._smooth_alpha) * prev_angle

    # ------------------------------------------------------------------
    # HSV segmentation for individual cyan brick detection
    # ------------------------------------------------------------------

    def _segment_bricks_hsv(self, frame, x1, y1, x2, y2):
        """
        Segment individual cyan bricks within a single YOLO bbox using HSV
        color filtering.

        Returns a list of brick dicts, or empty list if no cyan found
        (uses the detector box path).
        """
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        feature_mask, mask = self._build_hsv_masks(crop)
        if feature_mask is None or mask is None:
            return []

        # Check cyan coverage — if too little, not a cyan brick
        bbox_area = crop.shape[0] * crop.shape[1]
        cyan_pixels = cv2.countNonZero(feature_mask)
        if cyan_pixels < bbox_area * HSV_CYAN_COVERAGE_MIN:
            return []

        if self._uses_trapezoid_gate():
            candidates = self._build_trapezoid_brick_candidates(
                feature_mask,
                crop_x=0,
                crop_y=0,
                crop_w=crop.shape[1],
                crop_h=crop.shape[0],
            )
            if candidates:
                return [
                    self._offset_candidate_to_frame(candidate, x1, y1)
                    for candidate in candidates
                ]

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours is None:
            contours = []

        min_area = bbox_area * HSV_MIN_AREA_RATIO
        bricks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            bx, by, bw, bh = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)

            # Shape gate: ignore cyan blobs that are not brick-like.
            if bw <= 0 or bh <= 0:
                continue
            fill_ratio = float(area) / float(max(1, bw * bh))
            if fill_ratio < BRICK_SHAPE_FILL_RATIO_MIN:
                continue
            rw, rh = rect[1]
            if rw <= 0.0 or rh <= 0.0:
                continue
            aspect = float(max(rw, rh) / max(1e-6, min(rw, rh)))
            expected_aspects = (
                float(BRICK_WIDTH_MM / BRICK_HEIGHT_MM),
                float(BRICK_WIDTH_MM / BRICK_DEPTH_MM),
                float(BRICK_HEIGHT_MM / BRICK_DEPTH_MM),
            )
            rel_err = min(abs(aspect - exp) / max(1e-6, exp) for exp in expected_aspects)
            if rel_err > BRICK_SHAPE_REL_ERROR_MAX:
                continue

            cutout_polygons = []

            if self._uses_negative_cutout_gate():
                gate_rect = rect
                cutout_match, cutout_summary = self._match_negative_cutout_pair(
                    feature_mask,
                    gate_rect,
                )
                if not cutout_match:
                    # If two stacked bricks merge into a single tall contour,
                    # probe top/bottom halves independently for the two-cutout pattern.
                    if bh >= max(12, int(round(float(bw) * 1.15))):
                        half_h = max(8, int(round(float(bh) * 0.5)))
                        split_boxes = [
                            (int(bx), int(by), int(bw), int(half_h)),
                            (int(bx), int(by + bh - half_h), int(bw), int(half_h)),
                        ]
                        for sbx, sby, sbw, sbh in split_boxes:
                            sub_rect = (
                                (float(sbx) + (float(sbw) * 0.5), float(sby) + (float(sbh) * 0.5)),
                                (float(sbw), float(sbh)),
                                0.0,
                            )
                            sub_match, sub_summary = self._match_negative_cutout_pair(
                                feature_mask,
                                sub_rect,
                            )
                            if not sub_match:
                                continue

                            sub_cutouts = (
                                list(sub_summary.get("polygons") or [])
                                if isinstance(sub_summary, dict)
                                else []
                            )
                            sub_cutouts_frame = []
                            for poly in sub_cutouts:
                                poly_arr = np.asarray(poly, dtype=np.float32)
                                if poly_arr.ndim != 2 or poly_arr.shape[0] < 3:
                                    continue
                                poly_arr = poly_arr.copy()
                                poly_arr[:, 0] += float(x1)
                                poly_arr[:, 1] += float(y1)
                                sub_cutouts_frame.append(poly_arr)

                            split_contour = np.asarray(
                                [
                                    [float(sbx), float(sby)],
                                    [float(sbx + sbw), float(sby)],
                                    [float(sbx + sbw), float(sby + sbh)],
                                    [float(sbx), float(sby + sbh)],
                                ],
                                dtype=np.float32,
                            ).reshape(-1, 1, 2)
                            split_contour[:, :, 0] += float(x1)
                            split_contour[:, :, 1] += float(y1)

                            split_rect = (
                                (
                                    float(sbx) + (float(sbw) * 0.5) + float(x1),
                                    float(sby) + (float(sbh) * 0.5) + float(y1),
                                ),
                                (float(sbw), float(sbh)),
                                0.0,
                            )
                            split_partial_info = self._partial_info_for_crop_bbox(
                                sbx,
                                sby,
                                sbw,
                                sbh,
                                crop.shape[1],
                                crop.shape[0],
                            )
                            split_shape_score = (
                                float(sub_summary.get("score"))
                                if isinstance(sub_summary, dict)
                                and isinstance(sub_summary.get("score"), (int, float))
                                else None
                            )
                            split_selection_anchor = self._negative_cutout_selection_anchor(
                                sub_cutouts_frame
                            )

                            bricks.append({
                                "contour": split_contour,
                                "center_x": int(round(float(split_rect[0][0]))),
                                "center_y": int(round(float(split_rect[0][1]))),
                                "rect": split_rect,
                                "bbox": (int(sbx + x1), int(sby + y1), int(sbw), int(sbh)),
                                "area": float(max(1, sbw * sbh)),
                                "partial": False,
                                "partial_kind": None,
                                "partial_label": None,
                                "partial_edges": split_partial_info.get("edges") or {},
                                "shape_profile": "full",
                                "shape_match_score": split_shape_score,
                                "negative_cutout_polygons": sub_cutouts_frame,
                                "selection_anchor_x": (
                                    None
                                    if split_selection_anchor is None
                                    else float(split_selection_anchor[0])
                                ),
                                "selection_anchor_y": (
                                    None
                                    if split_selection_anchor is None
                                    else float(split_selection_anchor[1])
                                ),
                            })
                    continue
                shape_profile = "full"
                shape_match_score = (
                    cutout_summary.get("score")
                    if isinstance(cutout_summary, dict)
                    else None
                )
                cutout_polygons = (
                    list(cutout_summary.get("polygons") or [])
                    if isinstance(cutout_summary, dict)
                    else []
                )
                partial = False
                partial_kind = None
                partial_label = None
            else:
                # Canonical shape gate: accept only full/top/bottom profile matches.
                shape_profile, shape_match_score = self._classify_contour_shape(cnt)
                if shape_profile is None:
                    continue

                partial = False
                partial_kind = None
                partial_label = None
                if shape_profile == "top_half":
                    partial = True
                    partial_kind = "top_half"
                    partial_label = PARTIAL_LABEL_TOP
                elif shape_profile == "bottom_half":
                    partial = True
                    partial_kind = "bottom_half"
                    partial_label = PARTIAL_LABEL_BOTTOM

            partial_info = self._partial_info_for_crop_bbox(
                bx,
                by,
                bw,
                bh,
                crop.shape[1],
                crop.shape[0],
            )

            # Offset from crop-space to frame-space
            cnt_frame = cnt.copy()
            cnt_frame[:, :, 0] += x1
            cnt_frame[:, :, 1] += y1
            rect_frame = (
                (rect[0][0] + x1, rect[0][1] + y1),
                rect[1],
                rect[2],
            )

            cutout_polygons_frame = []
            for poly in cutout_polygons:
                poly_arr = np.asarray(poly, dtype=np.float32)
                if poly_arr.ndim != 2 or poly_arr.shape[0] < 3:
                    continue
                poly_arr = poly_arr.copy()
                poly_arr[:, 0] += float(x1)
                poly_arr[:, 1] += float(y1)
                cutout_polygons_frame.append(poly_arr)
            selection_anchor = self._negative_cutout_selection_anchor(
                cutout_polygons_frame
            )

            bricks.append({
                "contour": cnt_frame,
                "center_x": cx + x1,
                "center_y": cy + y1,
                "rect": rect_frame,
                "bbox": (bx + x1, by + y1, bw, bh),
                "area": area,
                "partial": bool(partial),
                "partial_kind": partial_kind,
                "partial_label": partial_label,
                "partial_edges": partial_info.get("edges") or {},
                "shape_profile": shape_profile,
                "shape_match_score": (
                    float(shape_match_score)
                    if isinstance(shape_match_score, (int, float))
                    else None
                ),
                "negative_cutout_polygons": cutout_polygons_frame,
                "selection_anchor_x": (
                    None
                    if selection_anchor is None
                    else float(selection_anchor[0])
                ),
                "selection_anchor_y": (
                    None
                    if selection_anchor is None
                    else float(selection_anchor[1])
                ),
            })

        if self._uses_negative_cutout_gate():
            pair_candidates = self._build_negative_cutout_pair_candidates(
                feature_mask,
                crop_x=x1,
                crop_y=y1,
                crop_w=crop.shape[1],
                crop_h=crop.shape[0],
            )
            for pair_candidate in pair_candidates:
                pair_anchor = self._candidate_selection_point(pair_candidate)
                matched_idx = None
                for idx, existing in enumerate(bricks):
                    existing_anchor = self._candidate_selection_point(existing)
                    if math.hypot(
                        float(pair_anchor[0]) - float(existing_anchor[0]),
                        float(pair_anchor[1]) - float(existing_anchor[1]),
                    ) <= 14.0:
                        matched_idx = idx
                        break
                if matched_idx is None:
                    bricks.append(pair_candidate)
                    continue

                existing_score = bricks[matched_idx].get("shape_match_score")
                pair_score = pair_candidate.get("shape_match_score")
                if isinstance(pair_score, (int, float)) and (
                    not isinstance(existing_score, (int, float))
                    or float(pair_score) < float(existing_score)
                ):
                    bricks[matched_idx] = pair_candidate

        return bricks

    def _offset_candidate_to_frame(self, candidate, offset_x, offset_y):
        """Convert a crop-local candidate dict into frame coordinates."""
        if not isinstance(candidate, dict):
            return candidate
        shifted = dict(candidate)
        dx = float(offset_x)
        dy = float(offset_y)

        for key, delta in (("center_x", dx), ("selection_anchor_x", dx)):
            value = shifted.get(key)
            if isinstance(value, (int, float)):
                shifted[key] = float(value) + delta
        for key, delta in (("center_y", dy), ("selection_anchor_y", dy)):
            value = shifted.get(key)
            if isinstance(value, (int, float)):
                shifted[key] = float(value) + delta

        bbox = shifted.get("bbox")
        if isinstance(bbox, tuple) and len(bbox) >= 4:
            bx, by, bw, bh = bbox[:4]
            shifted["bbox"] = (
                int(round(float(bx) + dx)),
                int(round(float(by) + dy)),
                int(round(float(bw))),
                int(round(float(bh))),
            )

        rect = shifted.get("rect")
        if isinstance(rect, tuple) and len(rect) >= 3:
            center, size, angle = rect[:3]
            try:
                shifted["rect"] = (
                    (float(center[0]) + dx, float(center[1]) + dy),
                    size,
                    angle,
                )
            except Exception:
                pass

        contour = shifted.get("contour")
        if isinstance(contour, np.ndarray):
            contour_shifted = contour.copy()
            try:
                contour_shifted[:, :, 0] += dx
                contour_shifted[:, :, 1] += dy
                shifted["contour"] = contour_shifted
            except Exception:
                pass

        cutout_polygons = shifted.get("negative_cutout_polygons")
        if isinstance(cutout_polygons, list):
            shifted_polygons = []
            for poly in cutout_polygons:
                poly_arr = np.asarray(poly, dtype=np.float32)
                if poly_arr.ndim != 2 or poly_arr.shape[0] < 3:
                    continue
                poly_arr = poly_arr.copy()
                poly_arr[:, 0] += dx
                poly_arr[:, 1] += dy
                shifted_polygons.append(poly_arr)
            shifted["negative_cutout_polygons"] = shifted_polygons

        return shifted

    def _estimate_angle_from_rect(self, rect):
        """
        Extract angle from cv2.minAreaRect result.
        Same normalization as _estimate_angle (w < h -> angle - 90).
        """
        angle = rect[2]
        w, h = rect[1]
        if w < h:
            angle = angle - 90
        return angle

    def _refine_angle_for_primary(self, frame, primary):
        """
        Re-extract angle from an un-eroded HSV mask of the primary brick.

        Erosion is needed to separate touching bricks, but it distorts
        contour shape — making near-square contours that cause minAreaRect
        to flip between 0 and 90 degrees.  By re-running HSV with only
        morphological OPEN (no erosion) on the primary brick's bbox, we
        get the full color contour whose aspect ratio matches the real
        brick face, giving a stable angle.
        """
        refined_contour = self._refine_primary_contour(frame, primary)
        if refined_contour is None or len(refined_contour) < 5:
            return self._estimate_angle_from_rect(primary["rect"])

        rect = cv2.minAreaRect(refined_contour)
        return self._estimate_angle_from_rect(rect)

    def _refine_primary_contour(self, frame, primary):
        """Return a tighter primary HSV contour in frame coordinates."""
        bx, by, bw, bh = primary["bbox"]

        # Pad the bbox slightly to capture edge pixels
        pad = max(5, int(max(bw, bh) * 0.15))
        rx1 = max(0, bx - pad)
        ry1 = max(0, by - pad)
        rx2 = min(frame.shape[1], bx + bw + pad)
        ry2 = min(frame.shape[0], by + bh + pad)

        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return primary.get("contour")

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Light cleanup only — no erosion.  Erosion was causing contour
        # distortion (near-square shapes) especially at close range where
        # the brick fills the frame.  Since we already isolated this brick's
        # bbox, separation from neighbours is not needed here.
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return primary.get("contour")

        # Pick the contour closest to the primary brick's center (in crop coords)
        pcx = primary["center_x"] - rx1
        pcy = primary["center_y"] - ry1
        best_cnt = None
        best_dist = float("inf")
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            d = (cx - pcx) ** 2 + (cy - pcy) ** 2
            if d < best_dist:
                best_dist = d
                best_cnt = cnt

        if best_cnt is None or len(best_cnt) < 3:
            return primary.get("contour")

        cnt_frame = best_cnt.copy()
        cnt_frame[:, :, 0] += rx1
        cnt_frame[:, :, 1] += ry1
        return cnt_frame

    def _select_center_brick(self, individual_bricks, frame_w, frame_h):
        """
        Pick the brick closest to frame center. Partial bricks get 2x
        distance penalty so we prefer fully-visible bricks.
        """
        if not individual_bricks:
            self._center_lock_prev_center = None
            return None

        candidates = []
        for idx, brick in enumerate(individual_bricks):
            candidates.append(
                {
                    "index": idx,
                    "center_x": float(brick["center_x"]),
                    "center_y": float(brick["center_y"]),
                    "selection_anchor_x": brick.get("selection_anchor_x"),
                    "selection_anchor_y": brick.get("selection_anchor_y"),
                    "partial": bool(brick.get("partial", False)),
                }
            )
        selected_idx = self._select_center_candidate_index(candidates, frame_w, frame_h)
        if selected_idx is None:
            self._center_lock_prev_center = None
            return None
        return individual_bricks[int(selected_idx)]

    def _center_lock_radius(self, frame_w, frame_h):
        radius_px = getattr(self, "_center_lock_radius_px", None)
        if radius_px is not None:
            try:
                return max(0.0, float(radius_px))
            except (TypeError, ValueError):
                pass
        return max(8.0, float(min(frame_w, frame_h)) * float(CENTER_LOCK_RADIUS_RATIO))

    def _candidate_selection_point(self, candidate):
        if not isinstance(candidate, dict):
            return 0.0, 0.0
        anchor_x = candidate.get("selection_anchor_x")
        anchor_y = candidate.get("selection_anchor_y")
        if isinstance(anchor_x, (int, float)) and isinstance(anchor_y, (int, float)):
            return float(anchor_x), float(anchor_y)
        return (
            float(candidate.get("center_x", 0.0)),
            float(candidate.get("center_y", 0.0)),
        )

    def _candidate_center_score(self, candidate, frame_w, frame_h):
        frame_center_x = float(frame_w) / 2.0 + float(getattr(self, "camera_center_offset_px", 0.0) or 0.0)
        frame_center_y = float(frame_h) / 2.0
        selection_x, selection_y = self._candidate_selection_point(candidate)
        dx = float(selection_x) - frame_center_x
        dy = float(selection_y) - frame_center_y
        weight_x = max(0.01, float(getattr(self, "_center_axis_weight_x", CENTER_AXIS_WEIGHT_X) or CENTER_AXIS_WEIGHT_X))
        weight_y = max(0.01, float(getattr(self, "_center_axis_weight_y", CENTER_AXIS_WEIGHT_Y) or CENTER_AXIS_WEIGHT_Y))
        score = math.hypot(dx * weight_x, dy * weight_y)
        if bool(candidate.get("partial", False)):
            partial_penalty = max(1.0, float(getattr(self, "_center_partial_penalty", CENTER_PARTIAL_PENALTY) or CENTER_PARTIAL_PENALTY))
            score *= partial_penalty
        return float(score)

    def _center_y_row_candidate_indices(self, candidates, frame_w, frame_h):
        if not candidates:
            return []

        row_filter_enabled = bool(
            getattr(
                self,
                "_stack_center_y_row_filter_enabled",
                STACK_CENTER_Y_ROW_FILTER_ENABLED,
            )
        )
        min_candidates = max(
            1,
            int(
                getattr(
                    self,
                    "_stack_center_y_row_min_candidates",
                    STACK_CENTER_Y_ROW_MIN_CANDIDATES,
                )
                or STACK_CENTER_Y_ROW_MIN_CANDIDATES
            ),
        )
        if not row_filter_enabled or len(candidates) < min_candidates:
            return list(range(len(candidates)))

        frame_center_x = float(frame_w) / 2.0 + float(
            getattr(self, "camera_center_offset_px", 0.0) or 0.0
        )
        frame_center_y = float(frame_h) / 2.0
        row_tolerance_px = max(
            0.0,
            float(
                getattr(
                    self,
                    "_stack_center_y_row_tolerance_px",
                    STACK_CENTER_Y_ROW_TOLERANCE_PX,
                )
                or STACK_CENTER_Y_ROW_TOLERANCE_PX
            ),
        )

        selection_points = [
            (idx, *self._candidate_selection_point(candidate))
            for idx, candidate in enumerate(candidates)
        ]
        anchor_idx, _anchor_x, anchor_y = min(
            selection_points,
            key=lambda item: (
                abs(float(item[2]) - frame_center_y),
                abs(float(item[1]) - frame_center_x),
                float(item[2]),
            ),
        )
        allowed_indices = [
            int(idx)
            for idx, _sel_x, sel_y in selection_points
            if abs(float(sel_y) - float(anchor_y)) <= row_tolerance_px
        ]
        return allowed_indices or [int(anchor_idx)]

    def _filter_candidates_to_center_y_row(self, candidates, frame_w, frame_h):
        allowed_indices = self._center_y_row_candidate_indices(
            candidates,
            frame_w,
            frame_h,
        )
        if not allowed_indices:
            return []
        return [candidates[int(idx)] for idx in allowed_indices]

    def _select_center_candidate_index(self, candidates, frame_w, frame_h):
        if not candidates:
            self._center_lock_prev_center = None
            return None

        allowed_indices = self._center_y_row_candidate_indices(
            candidates,
            frame_w,
            frame_h,
        )
        if not allowed_indices:
            allowed_indices = list(range(len(candidates)))

        scores = [self._candidate_center_score(cand, frame_w, frame_h) for cand in candidates]
        best_idx = min(allowed_indices, key=lambda i: scores[i])
        chosen_idx = int(best_idx)

        lock_enabled = bool(getattr(self, "_center_lock_enabled", True))
        prev_center = getattr(self, "_center_lock_prev_center", None)
        if lock_enabled and isinstance(prev_center, tuple) and len(prev_center) == 2:
            prev_x = float(prev_center[0])
            prev_y = float(prev_center[1])
            radius = self._center_lock_radius(frame_w, frame_h)
            near_idx = None
            near_dist = float("inf")
            for idx in allowed_indices:
                cand = candidates[idx]
                cand_x, cand_y = self._candidate_selection_point(cand)
                dist = math.hypot(float(cand_x) - prev_x, float(cand_y) - prev_y)
                if dist < near_dist:
                    near_dist = dist
                    near_idx = idx
            if near_idx is not None and near_dist <= radius:
                switch_margin = max(
                    0.0,
                    float(
                        getattr(
                            self,
                            "_center_switch_margin_px",
                            CENTER_SWITCH_MARGIN_PX,
                        ) or CENTER_SWITCH_MARGIN_PX
                    ),
                )
                if (scores[near_idx] - scores[best_idx]) <= switch_margin:
                    chosen_idx = int(near_idx)

        chosen = candidates[chosen_idx]
        chosen_x, chosen_y = self._candidate_selection_point(chosen)
        self._center_lock_prev_center = (
            float(chosen_x),
            float(chosen_y),
        )
        return int(chosen_idx)

    def _select_center_box(self, bricks, frame_w, frame_h):
        if not bricks:
            self._center_lock_prev_center = None
            return None
        candidates = []
        for idx, (x1, y1, x2, y2, _conf) in enumerate(bricks):
            candidates.append(
                {
                    "index": idx,
                    "center_x": float((x1 + x2) / 2.0),
                    "center_y": float((y1 + y2) / 2.0),
                    "partial": False,
                }
            )
        selected_idx = self._select_center_candidate_index(candidates, frame_w, frame_h)
        if selected_idx is None:
            return None
        return bricks[int(selected_idx)]

    def _should_try_full_frame_hsv_fallback(self, bricks, frame_w, frame_h):
        if not isinstance(bricks, list) or not bricks:
            return True
        try:
            frame_area = float(max(1, int(frame_w)) * max(1, int(frame_h)))
        except Exception:
            return False
        edge_margin_px = 24
        for x1, y1, x2, y2, _conf in bricks:
            try:
                bw = max(1, int(x2) - int(x1))
                bh = max(1, int(y2) - int(y1))
            except Exception:
                continue
            box_area_ratio = float(bw * bh) / frame_area
            touches_edge = (
                int(x1) <= edge_margin_px
                or int(y1) <= edge_margin_px
                or int(x2) >= int(frame_w) - edge_margin_px
                or int(y2) >= int(frame_h) - edge_margin_px
            )
            if touches_edge or float(bw) >= float(frame_w) * 0.55 or float(bh) >= float(frame_h) * 0.55:
                return True
            if box_area_ratio >= 0.25:
                return True
        return False

    def _stack_flags_from_individuals(self, primary, all_bricks):
        """
        Determine above/below flags using individual brick positions.
        Uses the legacy-proven center-overlap and vertical-gap heuristic.
        """
        if not primary or len(all_bricks) < 2:
            return False, False

        sel_x = primary["center_x"]
        sel_y = primary["center_y"]
        _, _, bw, bh = primary["bbox"]

        x_tol = max(10.0, bw * STACK_X_OVERLAP_RATIO)
        y_tol = max(10.0, bh * STACK_Y_GAP_RATIO)

        above = False
        below = False
        for b in all_bricks:
            if b is primary:
                continue
            if abs(b["center_x"] - sel_x) > x_tol:
                continue
            if b["center_y"] < sel_y - y_tol:
                above = True
            elif b["center_y"] > sel_y + y_tol:
                below = True
        return above, below

    # ------------------------------------------------------------------
    # Shared processing pipeline
    # ------------------------------------------------------------------

    def _process_bricks(self, frame, bricks):
        """
        Shared pipeline for read() and read_frame().
        Runs HSV segmentation on YOLO boxes, falls back to current
        behavior for non-cyan bricks.

        Returns the standard 8-tuple.
        """
        w_frame = self.frame_w
        h_frame = self.frame_h

        # Sort by confidence (highest first)
        bricks.sort(key=lambda b: b[4], reverse=True)
        fallback_conf = float(bricks[0][4]) if bricks else 1.0

        # Try HSV segmentation on each YOLO box to find individual cyan bricks
        all_hsv_bricks = []
        hsv_used = False
        closeup_hsv_fallback_used = False
        if self._hsv_enabled:
            if bricks:
                for x1, y1, x2, y2, conf in bricks:
                    individuals = self._segment_bricks_hsv(frame, x1, y1, x2, y2)
                    all_hsv_bricks.extend(individuals)
            if not all_hsv_bricks and self._should_try_full_frame_hsv_fallback(bricks, w_frame, h_frame):
                # Very close bricks can fill most of the frame and fail YOLO/NMS.
                # They can also be clipped by the frame so the YOLO box misses
                # the full face geometry. Let the cyan shape gate inspect the
                # full frame directly before declaring failure.
                individuals = self._segment_bricks_hsv(frame, 0, 0, w_frame, h_frame)
                if individuals:
                    all_hsv_bricks.extend(individuals)
                    closeup_hsv_fallback_used = True
                    if not bricks:
                        bricks = [(0, 0, w_frame, h_frame, 1.0)]
            if all_hsv_bricks:
                hsv_used = True

        if hsv_used:
            detected_hsv_bricks = list(all_hsv_bricks)
            selectable_hsv_bricks = list(all_hsv_bricks)
            filtered_hsv_bricks = self._filter_candidates_to_center_y_row(
                all_hsv_bricks,
                w_frame,
                h_frame,
            )
            if filtered_hsv_bricks:
                selectable_hsv_bricks = filtered_hsv_bricks
            # HSV path: center-most brick, contour-based angle, individual distance
            primary = self._select_center_brick(selectable_hsv_bricks, w_frame, h_frame)
            self.last_status = "target locked (HSV)"

            # Use YOLO confidence from the box that contained this brick
            # (find which YOLO box contains the primary brick center)
            yolo_conf = bricks[0][4]
            for x1, y1, x2, y2, conf in bricks:
                if (x1 <= primary["center_x"] <= x2 and
                        y1 <= primary["center_y"] <= y2):
                    yolo_conf = conf
                    break
            if closeup_hsv_fallback_used:
                yolo_conf = max(float(yolo_conf), float(fallback_conf))
            self.last_primary_confidence = float(yolo_conf)
            conf_pct = float(yolo_conf) * 100.0

            # Pink dot detection: confirms the brick and boosts confidence.
            dot_found, dot_cx, dot_cy = self._detect_pink_dot_in_brick(frame, primary)
            # When YOLO found nothing at all (top conf = 0), the only signal that
            # distinguishes a real brick from a background false positive is the pink dot.
            if float(getattr(self, "last_max_confidence", 1.0)) == 0.0 and not dot_found:
                self.last_status = "low confidence"
                self.last_primary_confidence = 0.0
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
            combined_conf = conf_pct + (PINK_DOT_CONF_BONUS if dot_found else 0.0)
            if combined_conf < CONF_GATE_PCT:
                self.last_status = "low confidence"
                self.last_primary_confidence = 0.0
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
            conf_pct = min(100.0, combined_conf)
            primary["_dot_found"] = dot_found
            primary["_dot_cx"] = dot_cx
            primary["_dot_cy"] = dot_cy

            # Angle from contour with jump clamping to prevent 0/90 flips
            raw_angle = self._refine_angle_for_primary(frame, primary)
            angle = self._smooth_angle(raw_angle, self._prev_angle)
            self._prev_angle = angle

            # Distance from apparent brick size.
            # When both triangles are confirmed use their pixel span: the outer
            # left edge to outer right edge spans BRICK_WIDTH_MM in world model,
            # giving a much more stable signal than the YOLO bounding box.
            _, _, ind_bw, ind_bh = primary["bbox"]
            raw_dist = self._estimate_distance_from_box(ind_bw, ind_bh, primary.get("partial_kind"))
            tri_span_dist = self._dist_from_triangle_span(
                primary.get("negative_cutout_polygons")
            )
            if tri_span_dist is not None:
                raw_dist = tri_span_dist
            dist = self._smooth(raw_dist, self._prev_dist)
            self._prev_dist = dist

            # Horizontal and vertical offsets anchored to triangles + dot centroid
            # when all markers are detected; otherwise falls back to contour centroid.
            anchor_cx = float(primary["center_x"])
            anchor_cy = float(primary["center_y"])
            if dot_found and dot_cx is not None:
                cutout_polys = primary.get("negative_cutout_polygons") or []
                tri_cx_list = []
                tri_cy_list = []
                for poly in cutout_polys:
                    arr = np.asarray(poly, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] >= 3:
                        tri_cx_list.append(float(np.mean(arr[:, 0])))
                        tri_cy_list.append(float(np.mean(arr[:, 1])))
                if tri_cx_list:
                    all_cx = tri_cx_list + [float(dot_cx)]
                    all_cy = tri_cy_list + [float(dot_cy)]
                    anchor_cx = sum(all_cx) / len(all_cx)
                    anchor_cy = sum(all_cy) / len(all_cy)

            raw_offset_x = self._estimate_offset_x_mm(anchor_cx, dist)
            offset_x = self._smooth(raw_offset_x, self._prev_offset)
            self._prev_offset = offset_x

            raw_cam_height = self._estimate_cam_height(anchor_cy, dist)
            cam_height = self._smooth(raw_cam_height, self._prev_offset_y)
            self._prev_offset_y = cam_height

            # Stack flags from individual positions
            brick_above, brick_below = self._stack_flags_from_individuals(
                primary, detected_hsv_bricks
            )
            partial_bricks = [
                {
                    "partial": bool(b.get("partial")),
                    "label": b.get("partial_label"),
                }
                for b in selectable_hsv_bricks
                if isinstance(b, dict) and bool(b.get("partial"))
            ]
            self._set_partial_state(
                partial_bricks,
                primary_partial_kind=(primary.get("partial_kind") if isinstance(primary, dict) else None),
                primary_partial_label=(primary.get("partial_label") if isinstance(primary, dict) and bool(primary.get("partial")) else None),
            )

            if self.debug:
                self._draw_debug_hsv(
                    frame, bricks, detected_hsv_bricks, primary,
                    angle, dist, offset_x, yolo_conf
                )

            return (True, angle, dist, offset_x, conf_pct, cam_height,
                    brick_above, brick_below)

        # Detector-box path used when HSV does not split a confident face.
        trust_detector_boxes = bool(getattr(self, "_trust_detector_boxes", False))
        shape_gate_enabled = (
            not trust_detector_boxes
            and (
                bool(getattr(self, "_face_shape_templates", None))
                or self._uses_negative_cutout_gate()
            )
        )
        matched_boxes = []
        matched_partials = []
        if shape_gate_enabled:
            for bx1, by1, bx2, by2, bconf in bricks:
                if self._uses_negative_cutout_gate():
                    cutout_match, _cutout_summary = self._match_negative_cutout_pair_in_box(
                        frame,
                        bx1,
                        by1,
                        bx2,
                        by2,
                    )
                    if not cutout_match:
                        continue
                    shape_profile = "full"
                else:
                    contour = self._extract_shape_contour_in_box(frame, bx1, by1, bx2, by2)
                    shape_profile, _shape_score = self._classify_contour_shape(contour)
                if shape_profile is None:
                    continue
                matched_boxes.append((bx1, by1, bx2, by2, bconf))
                matched_partials.append(self._partial_from_shape_profile(shape_profile))
        else:
            matched_boxes = list(bricks)
            matched_partials = [self._partial_from_shape_profile("full") for _ in matched_boxes]

        if not matched_boxes:
            self._prev_angle = None
            self._prev_dist = None
            self._prev_offset = None
            self._prev_offset_y = None
            self._center_lock_prev_center = None
            self.last_status = self._shape_gate_mismatch_status()
            self.last_primary_confidence = 0.0
            self._clear_partial_state()
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        primary_box = self._select_center_box(matched_boxes, w_frame, h_frame)
        if primary_box is None:
            primary_box = matched_boxes[0]
        x1, y1, x2, y2, conf = primary_box
        primary_idx = 0
        for idx, box in enumerate(matched_boxes):
            if box == primary_box:
                primary_idx = idx
                break
        self.last_status = "target locked"
        self.last_primary_confidence = float(conf)
        conf_pct = float(conf) * 100.0

        fallback_primary_bbox = {"bbox": (x1, y1, x2 - x1, y2 - y1)}
        dot_found_fb, _, _ = self._detect_pink_dot_in_brick(frame, fallback_primary_bbox)
        combined_conf_fb = conf_pct + (PINK_DOT_CONF_BONUS if dot_found_fb else 0.0)
        if combined_conf_fb < CONF_GATE_PCT:
            self.last_status = "low confidence"
            self.last_primary_confidence = 0.0
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
        conf_pct = min(100.0, combined_conf_fb)

        raw_angle = self._estimate_angle(frame, x1, y1, x2, y2)
        angle = self._smooth_angle(raw_angle, self._prev_angle)
        self._prev_angle = angle

        primary_partial_info = (
            matched_partials[primary_idx]
            if 0 <= int(primary_idx) < len(matched_partials)
            else {}
        )

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        raw_dist = self._estimate_distance_from_box(bbox_w, bbox_h, primary_partial_info.get("kind"))
        dist = self._smooth(raw_dist, self._prev_dist)
        self._prev_dist = dist

        brick_center_x = (x1 + x2) / 2.0
        frame_center_x = w_frame / 2.0 + self.camera_center_offset_px
        raw_offset_x = self._estimate_offset_x_mm(brick_center_x, dist)
        offset_x = self._smooth(raw_offset_x, self._prev_offset)
        self._prev_offset = offset_x

        brick_center_y = (y1 + y2) / 2.0
        raw_cam_height = self._estimate_cam_height(brick_center_y, dist)
        cam_height = self._smooth(raw_cam_height, self._prev_offset_y)
        self._prev_offset_y = cam_height

        primary_cy = (y1 + y2) / 2.0
        brick_above = False
        brick_below = False
        for idx, (bx1, by1, bx2, by2, bconf) in enumerate(matched_boxes):
            if idx == primary_idx:
                continue
            other_cy = (by1 + by2) / 2.0
            if other_cy < primary_cy - bbox_h * 0.5:
                brick_above = True
            elif other_cy > primary_cy + bbox_h * 0.5:
                brick_below = True
        box_partial_infos = list(matched_partials)
        self._set_partial_state(
            box_partial_infos,
            primary_partial_kind=(primary_partial_info.get("kind") if isinstance(primary_partial_info, dict) and bool(primary_partial_info.get("partial")) else None),
            primary_partial_label=(primary_partial_info.get("label") if isinstance(primary_partial_info, dict) and bool(primary_partial_info.get("partial")) else None),
        )

        if self.debug:
            ordered = [primary_box] + [box for idx, box in enumerate(matched_boxes) if idx != primary_idx]
            ordered_partial_infos = [box_partial_infos[primary_idx]] + [
                box_partial_infos[idx]
                for idx in range(len(matched_boxes))
                if idx != primary_idx
            ]
            self._draw_debug(frame, ordered, angle, dist, offset_x, conf, partial_infos=ordered_partial_infos)

        return (True, angle, dist, offset_x, conf_pct, cam_height,
                brick_above, brick_below)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def read(self):
        """
        Capture a frame and detect bricks.

        Returns:
            tuple: (found, angle, dist, offset_x, confidence,
                    cam_height, brick_above, brick_below)
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._camera_read_fail_count += 1
            now_s = time.monotonic()
            if (
                self._camera_read_fail_count == 1
                or (now_s - self._last_camera_read_fail_log_at) >= self._camera_read_fail_log_interval_s
            ):
                self.log.warning(
                    "Frame capture failed (consecutive=%d)",
                    self._camera_read_fail_count,
                )
                self._last_camera_read_fail_log_at = now_s
            self.last_status = "camera read failed"
            self.last_primary_confidence = 0.0
            self._clear_partial_state()
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        if self._camera_read_fail_count > 0:
            self.log.info(
                "Frame capture recovered after %d consecutive failures",
                self._camera_read_fail_count,
            )
            self._camera_read_fail_count = 0

        self.raw_frame = frame.copy()
        self.current_frame = frame.copy()

        h_frame, w_frame = frame.shape[:2]
        self.frame_w = w_frame
        self.frame_h = h_frame

        bricks = self._detect(frame)

        if not bricks:
            self._prev_angle = None
            self._prev_dist = None
            self._prev_offset = None
            self._prev_offset_y = None
            self._center_lock_prev_center = None
            self.last_status = "searching"
            self.last_primary_confidence = 0.0
            self._clear_partial_state()
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        try:
            return self._process_bricks(frame, bricks)
        except Exception as exc:
            # Keep livestream camera output alive even if post-processing fails.
            self.log.exception("Brick post-processing failed in read(): %s", exc)
            self.current_frame = frame.copy()
            self.last_status = "processing error"
            self.last_primary_confidence = 0.0
            self._clear_partial_state()
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

    def read_frame(self, frame):
        """
        Detect bricks in a provided frame (no camera capture).
        Same return signature as read().
        """
        self.raw_frame = frame.copy()
        self.current_frame = frame.copy()

        h_frame, w_frame = frame.shape[:2]
        self.frame_w = w_frame
        self.frame_h = h_frame

        bricks = self._detect(frame)

        if not bricks:
            self._prev_angle = None
            self._prev_dist = None
            self._prev_offset = None
            self._prev_offset_y = None
            self._center_lock_prev_center = None
            self.last_status = "searching"
            self.last_primary_confidence = 0.0
            self._clear_partial_state()
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        # Use a copy so debug drawing doesn't mutate the caller's frame
        try:
            return self._process_bricks(frame.copy(), bricks)
        except Exception as exc:
            # Keep caller preview available during debug/simulation failures.
            self.log.exception("Brick post-processing failed in read_frame(): %s", exc)
            self.current_frame = frame.copy()
            self.last_status = "processing error"
            self.last_primary_confidence = 0.0
            self._clear_partial_state()
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

    def _draw_debug(self, frame, bricks, angle, dist, offset_x, conf, partial_infos=None):
        """Draw detection visualization on frame."""
        if self.debug_show_all_candidates:
            draw_boxes = list(bricks)
        else:
            max_boxes = max(1, int(getattr(self, "debug_max_boxes", 1) or 1))
            draw_boxes = list(bricks[:max_boxes])

        for i, (x1, y1, x2, y2, c) in enumerate(draw_boxes):
            partial_info = (
                partial_infos[i]
                if isinstance(partial_infos, list) and i < len(partial_infos)
                else {}
            )
            is_partial = bool((partial_info or {}).get("partial"))
            if self._uses_negative_cutout_gate():
                if i == 0:
                    rect_points = np.asarray(
                        [
                            [float(x1), float(y1)],
                            [float(x2), float(y1)],
                            [float(x2), float(y2)],
                            [float(x1), float(y2)],
                        ],
                        dtype=np.float32,
                    )
                    self._draw_negative_cutout_polygons(frame, rect_points)
            else:
                color = PARTIAL_COLOR_BGR if is_partial else ((0, 255, 0) if i == 0 else (0, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = str((partial_info or {}).get("label") or "").strip()
            if label:
                text_y = max(18, int(y1) - 8)
                cv2.putText(
                    frame,
                    label,
                    (max(0, int(x1)), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    PARTIAL_COLOR_BGR,
                    2,
                )

            # Angle indicator
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = min(x2 - x1, y2 - y1) // 2
            measured_angle = (
                angle if i == 0 else self._estimate_angle(frame, x1, y1, x2, y2)
            )
            # UI convention: 0° points straight down; negative leans left.
            rad = math.radians(measured_angle - 90.0)
            ex = int(cx + length * math.cos(rad))
            ey = int(cy - length * math.sin(rad))
            cv2.line(frame, (cx, cy), (ex, ey), (0, 0, 255), 2)

        self.current_frame = frame

    def _ordered_rect_points(self, rect):
        pts = cv2.boxPoints(rect)
        pts = np.asarray(pts, dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered

    def _rect_segment_name(self, p0, p1, center):
        midpoint = (np.asarray(p0, dtype=np.float32) + np.asarray(p1, dtype=np.float32)) * 0.5
        dx = float(midpoint[0] - center[0])
        dy = float(midpoint[1] - center[1])
        if abs(dx) >= abs(dy):
            return "right" if dx >= 0.0 else "left"
        return "bottom" if dy >= 0.0 else "top"

    def _load_face_shape_model(self):
        self._face_polygon_model = None
        self._face_cutouts_model = []
        self._face_lines_model = []
        self._face_shape_gate_mode = BRICK_FACE_GATE_MODE_DEFAULT
        self._shape_match_gap_bridge_px = 0
        self._shape_match_score_max = float(BRICK_FACE_MATCH_MAX_FULL)
        self._negative_cutout_cyan_fill_max = float(NEGATIVE_CUTOUT_CYAN_FILL_MAX)
        self._negative_cutout_ring_cyan_min = float(NEGATIVE_CUTOUT_RING_CYAN_MIN)
        self._negative_cutout_ring_dilate_px = int(NEGATIVE_CUTOUT_RING_DILATE_PX)
        self._negative_cutout_min_area_px = float(NEGATIVE_CUTOUT_MIN_AREA_PX)
        self._negative_cutout_triangle_side_ratio_max = float(
            NEGATIVE_CUTOUT_TRIANGLE_SIDE_RATIO_MAX
        )
        self._negative_cutout_triangle_angle_spread_max_deg = float(
            NEGATIVE_CUTOUT_TRIANGLE_ANGLE_SPREAD_MAX_DEG
        )
        self._negative_cutout_triangle_overlap_min = float(
            NEGATIVE_CUTOUT_TRIANGLE_OVERLAP_MIN
        )
        self._negative_cutout_triangle_area_ratio_min = float(
            NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MIN
        )
        self._negative_cutout_triangle_area_ratio_max = float(
            NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MAX
        )
        self._negative_cutout_pair_x_axis_max_angle_deg = float(
            NEGATIVE_CUTOUT_PAIR_X_AXIS_MAX_ANGLE_DEG
        )
        self._negative_cutout_focus_y_min_ratio = NEGATIVE_CUTOUT_FOCUS_Y_MIN_RATIO
        self._negative_cutout_focus_y_max_ratio = NEGATIVE_CUTOUT_FOCUS_Y_MAX_RATIO
        self._negative_cutout_focus_weight = float(NEGATIVE_CUTOUT_FOCUS_WEIGHT)
        try:
            import json

            data = json.loads(BRICK_MODEL_PATH.read_text())
        except Exception:
            return

        brick = data.get("brick") if isinstance(data, dict) else None
        if not isinstance(brick, dict):
            return

        raw_poly = brick.get("facePolygon")
        points = []
        if isinstance(raw_poly, list):
            for row in raw_poly:
                if not isinstance(row, dict):
                    continue
                try:
                    points.append([float(row.get("x")), float(row.get("y"))])
                except (TypeError, ValueError):
                    continue
        if len(points) >= 3:
            self._face_polygon_model = np.asarray(points, dtype=np.float32)

        raw_cutouts = brick.get("faceCutouts")
        if isinstance(raw_cutouts, list):
            for raw_cutout in raw_cutouts:
                if not isinstance(raw_cutout, list):
                    continue
                cutout_points = []
                for row in raw_cutout:
                    if not isinstance(row, dict):
                        continue
                    try:
                        cutout_points.append([float(row.get("x")), float(row.get("y"))])
                    except (TypeError, ValueError):
                        continue
                if len(cutout_points) >= 3:
                    self._face_cutouts_model.append(np.asarray(cutout_points, dtype=np.float32))

        raw_lines = brick.get("faceLines")
        if isinstance(raw_lines, list):
            for row in raw_lines:
                if not isinstance(row, dict):
                    continue
                p1 = row.get("p1") if isinstance(row.get("p1"), dict) else None
                p2 = row.get("p2") if isinstance(row.get("p2"), dict) else None
                if p1 is None or p2 is None:
                    continue
                try:
                    self._face_lines_model.append(
                        (
                            (float(p1.get("x")), float(p1.get("y"))),
                            (float(p2.get("x")), float(p2.get("y"))),
                        )
                    )
                except (TypeError, ValueError):
                    continue

        shape_gate = brick.get("shapeGate")
        if isinstance(shape_gate, dict):
            mode_text = str(shape_gate.get("mode") or "").strip().lower()
            if mode_text in {
                BRICK_FACE_GATE_MODE_DEFAULT,
                BRICK_FACE_GATE_MODE_NEGATIVE_CUTOUTS,
                BRICK_FACE_GATE_MODE_SHAPE_MATCH,
            }:
                self._face_shape_gate_mode = mode_text
            try:
                self._shape_match_gap_bridge_px = max(
                    0, int(shape_gate.get("shape_match_gap_bridge_px"))
                )
            except (TypeError, ValueError):
                pass
            try:
                self._shape_match_score_max = max(
                    0.05, float(shape_gate.get("shape_match_score_max"))
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_cyan_fill_max = float(
                    shape_gate.get("cutout_cyan_fill_max")
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_ring_cyan_min = float(
                    shape_gate.get("cutout_ring_cyan_min")
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_ring_dilate_px = max(
                    1, int(shape_gate.get("cutout_ring_dilate_px"))
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_min_area_px = max(
                    1.0, float(shape_gate.get("cutout_min_area_px"))
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_triangle_side_ratio_max = max(
                    1.0,
                    float(shape_gate.get("cutout_triangle_side_ratio_max")),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_triangle_angle_spread_max_deg = max(
                    1.0,
                    float(shape_gate.get("cutout_triangle_angle_spread_max_deg")),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_triangle_overlap_min = max(
                    0.0,
                    min(1.0, float(shape_gate.get("cutout_triangle_overlap_min"))),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_triangle_area_ratio_min = max(
                    0.05,
                    float(shape_gate.get("cutout_triangle_area_ratio_min")),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_triangle_area_ratio_max = max(
                    max(0.05, self._negative_cutout_triangle_area_ratio_min),
                    float(shape_gate.get("cutout_triangle_area_ratio_max")),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_pair_x_axis_max_angle_deg = max(
                    0.0,
                    min(
                        89.0,
                        float(shape_gate.get("cutout_pair_x_axis_max_angle_deg")),
                    ),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_focus_y_min_ratio = max(
                    0.0,
                    min(1.0, float(shape_gate.get("cutout_focus_y_min_ratio"))),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_focus_y_max_ratio = max(
                    0.0,
                    min(1.0, float(shape_gate.get("cutout_focus_y_max_ratio"))),
                )
            except (TypeError, ValueError):
                pass
            try:
                self._negative_cutout_focus_weight = max(
                    0.0,
                    float(shape_gate.get("cutout_focus_weight")),
                )
            except (TypeError, ValueError):
                pass

        pink_dot = brick.get("pinkDot")
        if isinstance(pink_dot, dict):
            try:
                self._pink_dot_model_xy = (
                    float(pink_dot.get("x", 0.0)),
                    float(pink_dot.get("y", 9.0)),
                )
            except (TypeError, ValueError):
                pass

    def _contour_from_points(self, points):
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            return None
        return points.reshape(-1, 1, 2).astype(np.float32)

    def _shape_match_contour(self, points_or_contour):
        contour = self._contour_from_points(points_or_contour)
        if contour is None:
            contour = np.asarray(points_or_contour, dtype=np.float32)
            if contour.ndim == 2 and contour.shape[1] == 2:
                contour = contour.reshape(-1, 1, 2)
        if contour is None or contour.ndim != 3 or contour.shape[0] < 3:
            return None
        contour = contour.astype(np.float32)
        try:
            perimeter = float(cv2.arcLength(contour, True))
        except Exception:
            return contour
        epsilon = max(0.4, perimeter * 0.01)
        try:
            approx = cv2.approxPolyDP(contour, epsilon, True)
        except Exception:
            return contour
        if approx is None or approx.shape[0] < 3:
            return contour
        return approx.astype(np.float32)

    def _clip_polygon_y(self, points, y_cut, keep_above):
        if not isinstance(points, np.ndarray) or points.ndim != 2 or len(points) < 3:
            return np.empty((0, 2), dtype=np.float32)

        def inside(pt):
            if keep_above:
                return float(pt[1]) >= float(y_cut)
            return float(pt[1]) <= float(y_cut)

        clipped = []
        prev = points[-1]
        prev_inside = inside(prev)
        for cur in points:
            cur_inside = inside(cur)
            if cur_inside != prev_inside:
                dy = float(cur[1] - prev[1])
                if abs(dy) > 1e-6:
                    t = (float(y_cut) - float(prev[1])) / dy
                    ix = float(prev[0]) + t * float(cur[0] - prev[0])
                else:
                    ix = float(cur[0])
                clipped.append([ix, float(y_cut)])
            if cur_inside:
                clipped.append([float(cur[0]), float(cur[1])])
            prev = cur
            prev_inside = cur_inside

        if len(clipped) < 3:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(clipped, dtype=np.float32)

    def _init_shape_match_templates(self):
        self._face_shape_templates = {}
        poly = self._face_polygon_model
        if not isinstance(poly, np.ndarray) or poly.shape[0] < 3:
            return

        full_contour = self._contour_from_points(poly)
        if full_contour is None:
            return

        y_min = float(np.min(poly[:, 1]))
        y_max = float(np.max(poly[:, 1]))
        y_mid = (y_min + y_max) * 0.5
        top_poly = self._clip_polygon_y(poly, y_mid, keep_above=True)
        bottom_poly = self._clip_polygon_y(poly, y_mid, keep_above=False)

        top_contour = self._contour_from_points(top_poly)
        bottom_contour = self._contour_from_points(bottom_poly)

        full_template = self._shape_match_contour(full_contour)
        if full_template is not None:
            self._face_shape_templates["full"] = full_template
        if top_contour is not None:
            top_template = self._shape_match_contour(top_contour)
            if top_template is not None:
                self._face_shape_templates["top_half"] = top_template
        if bottom_contour is not None:
            bottom_template = self._shape_match_contour(bottom_contour)
            if bottom_template is not None:
                self._face_shape_templates["bottom_half"] = bottom_template

    def _classify_contour_shape(self, contour):
        templates = getattr(self, "_face_shape_templates", None)
        if not isinstance(templates, dict) or not templates:
            # Keep test stubs and non-initialized instances functional.
            return "full", 0.0
        if contour is None:
            return None, None

        contour_arr = np.asarray(contour, dtype=np.float32)
        if contour_arr.ndim != 3 or contour_arr.shape[0] < 3:
            return None, None

        candidate_shape = self._shape_match_contour(contour_arr)
        if candidate_shape is None:
            return None, None

        best_profile = None
        best_score = None
        for profile, templ in templates.items():
            try:
                score = float(
                    cv2.matchShapes(
                        candidate_shape,
                        templ,
                        cv2.CONTOURS_MATCH_I1,
                        0.0,
                    )
                )
            except Exception:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_profile = str(profile)

        if best_profile is None or best_score is None:
            return None, None

        threshold = (
            float(BRICK_FACE_MATCH_MAX_FULL)
            if best_profile == "full"
            else float(BRICK_FACE_MATCH_MAX_PARTIAL)
        )
        if float(best_score) > threshold:
            return None, float(best_score)
        return best_profile, float(best_score)

    def _uses_negative_cutout_gate(self):
        face_cutouts_model = getattr(self, "_face_cutouts_model", None)
        return isinstance(face_cutouts_model, list) and len(face_cutouts_model) >= 2

    def _uses_trapezoid_gate(self):
        face_cutouts_model = getattr(self, "_face_cutouts_model", None)
        return isinstance(face_cutouts_model, list) and len(face_cutouts_model) == 1

    def _uses_shape_match_gate(self):
        return (
            str(getattr(self, "_face_shape_gate_mode", "")) == BRICK_FACE_GATE_MODE_SHAPE_MATCH
            or (not self._uses_negative_cutout_gate() and not self._uses_trapezoid_gate())
        )

    def _shape_gate_mismatch_status(self):
        if self._uses_negative_cutout_gate():
            return "inner triangles mismatch"
        return "shape mismatch"

    def _build_hsv_masks(self, crop):
        if crop is None or getattr(crop, "size", 0) == 0:
            return None, None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        feature_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kern_open)

        contour_mask = feature_mask.copy()
        kern_erode = cv2.getStructuringElement(
            cv2.MORPH_RECT, (HSV_ERODE_KERNEL, HSV_ERODE_KERNEL)
        )
        contour_mask = cv2.erode(
            contour_mask,
            kern_erode,
            iterations=self._hsv_erode_iterations,
        )
        kern_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        contour_mask = cv2.dilate(contour_mask, kern_dilate, iterations=1)
        return feature_mask, contour_mask

    def _mask_fill_ratio(self, mask, polygon_points):
        if mask is None:
            return None, None, 0.0
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.ndim != 2:
            return None, None, 0.0
        polygon = np.asarray(polygon_points, dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] < 3:
            return None, None, 0.0

        region_mask = np.zeros(mask_arr.shape[:2], dtype=np.uint8)
        region_draw = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        try:
            cv2.fillPoly(region_mask, [region_draw], 255)
        except Exception:
            return None, None, 0.0

        area = float(cv2.countNonZero(region_mask))
        if area <= 0.0:
            return None, None, 0.0
        active = cv2.bitwise_and(mask_arr, mask_arr, mask=region_mask)
        return float(cv2.countNonZero(active)) / area, region_mask, area

    def _polygon_centroid(self, polygon_points):
        pts = np.asarray(polygon_points, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            return None
        contour = pts.reshape(-1, 1, 2).astype(np.float32)
        try:
            moments = cv2.moments(contour)
        except Exception:
            moments = None
        if moments and abs(float(moments.get("m00", 0.0))) > 1e-6:
            area = float(moments["m00"])
            return (
                float(moments["m10"]) / area,
                float(moments["m01"]) / area,
            )
        return (
            float(np.mean(pts[:, 0])),
            float(np.mean(pts[:, 1])),
        )

    def _triangle_from_contour(self, contour):
        if contour is None:
            return None
        try:
            peri = float(cv2.arcLength(contour, True))
        except Exception:
            peri = 0.0
        if peri > 0.0:
            try:
                approx = cv2.approxPolyDP(contour, max(1.0, peri * 0.06), True)
            except Exception:
                approx = None
            if approx is not None and approx.shape[0] == 3:
                return approx.reshape(-1, 2).astype(np.float32)

        try:
            hull = cv2.convexHull(contour)
        except Exception:
            hull = None
        if hull is not None and len(hull) >= 3:
            try:
                peri_h = float(cv2.arcLength(hull, True))
            except Exception:
                peri_h = 0.0
            if peri_h > 0.0:
                try:
                    approx_h = cv2.approxPolyDP(hull, max(1.0, peri_h * 0.10), True)
                except Exception:
                    approx_h = None
                if approx_h is not None and approx_h.shape[0] == 3:
                    return approx_h.reshape(-1, 2).astype(np.float32)

        try:
            _tri_area_fit, tri_fit = cv2.minEnclosingTriangle(contour)
        except Exception:
            tri_fit = None
        if tri_fit is None:
            return None
        try:
            tri = np.asarray(tri_fit, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if tri.shape != (3, 2):
            return None
        return tri

    def _triangle_metrics_pass(self, triangle_points):
        metrics = self._triangle_metrics(triangle_points)
        if not isinstance(metrics, dict):
            return None

        side_ratio_max = float(
            getattr(
                self,
                "_negative_cutout_triangle_side_ratio_max",
                NEGATIVE_CUTOUT_TRIANGLE_SIDE_RATIO_MAX,
            )
        )
        angle_spread_max_deg = float(
            getattr(
                self,
                "_negative_cutout_triangle_angle_spread_max_deg",
                NEGATIVE_CUTOUT_TRIANGLE_ANGLE_SPREAD_MAX_DEG,
            )
        )
        if float(metrics["side_ratio"]) > side_ratio_max:
            return None
        if float(metrics["angle_spread_deg"]) > angle_spread_max_deg:
            return None
        return metrics

    def _negative_cutout_triangle_pass_profiles(self):
        overlap_min = float(
            getattr(
                self,
                "_negative_cutout_triangle_overlap_min",
                NEGATIVE_CUTOUT_TRIANGLE_OVERLAP_MIN,
            )
        )
        area_ratio_min = float(
            getattr(
                self,
                "_negative_cutout_triangle_area_ratio_min",
                NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MIN,
            )
        )
        area_ratio_max = float(
            getattr(
                self,
                "_negative_cutout_triangle_area_ratio_max",
                NEGATIVE_CUTOUT_TRIANGLE_AREA_RATIO_MAX,
            )
        )
        return [
            {
                "area_ratio_min": float(area_ratio_min),
                "area_ratio_max": float(area_ratio_max),
                "overlap_proj_min": float(overlap_min),
                "overlap_tri_min": 0.65,
            },
            {
                "area_ratio_min": max(0.18, float(area_ratio_min) * 0.5),
                "area_ratio_max": max(float(area_ratio_max), 2.2),
                "overlap_proj_min": max(0.22, float(overlap_min) * 0.55),
                "overlap_tri_min": 0.45,
            },
        ]

    def _sorted_face_cutout_models(self):
        face_cutouts_model = getattr(self, "_face_cutouts_model", None)
        if not isinstance(face_cutouts_model, list):
            return []

        items = []
        for cutout_points in face_cutouts_model:
            cutout_arr = np.asarray(cutout_points, dtype=np.float32)
            if cutout_arr.ndim != 2 or cutout_arr.shape[0] < 3:
                continue
            centroid = self._polygon_centroid(cutout_arr)
            if centroid is None:
                continue
            items.append(
                {
                    "polygon": cutout_arr,
                    "centroid": np.asarray(centroid, dtype=np.float32),
                }
            )
        items.sort(key=lambda item: float(item["centroid"][0]))
        return items

    def _transform_model_points_similarity(
        self,
        model_points,
        *,
        model_origin,
        image_origin,
        scale,
        cos_theta,
        sin_theta,
    ):
        pts = np.asarray(model_points, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 1:
            return None
        model_origin_arr = np.asarray(model_origin, dtype=np.float32).reshape(2)
        image_origin_arr = np.asarray(image_origin, dtype=np.float32).reshape(2)
        centered = pts - model_origin_arr
        transformed = np.empty_like(centered)
        transformed[:, 0] = (centered[:, 0] * float(cos_theta)) - (
            centered[:, 1] * float(sin_theta)
        )
        transformed[:, 1] = (centered[:, 0] * float(sin_theta)) + (
            centered[:, 1] * float(cos_theta)
        )
        transformed *= float(scale)
        transformed += image_origin_arr
        return transformed.astype(np.float32)

    def _negative_cutout_focus_band(self):
        weight = float(
            getattr(
                self,
                "_negative_cutout_focus_weight",
                NEGATIVE_CUTOUT_FOCUS_WEIGHT,
            )
            or 0.0
        )
        if weight <= 0.0:
            return None
        min_ratio = getattr(
            self,
            "_negative_cutout_focus_y_min_ratio",
            NEGATIVE_CUTOUT_FOCUS_Y_MIN_RATIO,
        )
        max_ratio = getattr(
            self,
            "_negative_cutout_focus_y_max_ratio",
            NEGATIVE_CUTOUT_FOCUS_Y_MAX_RATIO,
        )
        if min_ratio is None or max_ratio is None:
            return None
        try:
            band_min = max(0.0, min(1.0, float(min_ratio)))
            band_max = max(0.0, min(1.0, float(max_ratio)))
        except (TypeError, ValueError):
            return None
        if band_max < band_min:
            band_min, band_max = band_max, band_min
        return band_min, band_max, weight

    def _negative_cutout_focus_penalty(self, polygon_points, mask_shape):
        band = self._negative_cutout_focus_band()
        if band is None:
            return 0.0
        centroid = self._polygon_centroid(polygon_points)
        if centroid is None:
            return 0.0
        try:
            mask_h = int(mask_shape[0])
        except Exception:
            return 0.0
        if mask_h <= 1:
            return 0.0

        y_ratio = float(centroid[1]) / float(mask_h - 1)
        y_ratio = max(0.0, min(1.0, y_ratio))
        band_min, band_max, weight = band
        if y_ratio < band_min:
            distance = band_min - y_ratio
        elif y_ratio > band_max:
            distance = y_ratio - band_max
        else:
            band_center = (band_min + band_max) * 0.5
            distance = abs(y_ratio - band_center) * 0.35
        return float(distance) * float(weight)

    def _negative_cutout_selection_anchor(self, polygons):
        if not isinstance(polygons, list) or not polygons:
            return None
        centroids = []
        for poly in polygons:
            centroid = self._polygon_centroid(poly)
            if centroid is None:
                continue
            centroids.append(centroid)
        if not centroids:
            return None
        centroid_arr = np.asarray(centroids, dtype=np.float32)
        return (
            float(np.mean(centroid_arr[:, 0])),
            float(np.mean(centroid_arr[:, 1])),
        )

    def _negative_cutout_pair_x_axis_metrics(self, polygons):
        if not isinstance(polygons, list) or len(polygons) < 2:
            return None

        items = []
        for poly in polygons:
            centroid = self._polygon_centroid(poly)
            if centroid is None:
                continue
            items.append(
                {
                    "polygon": np.asarray(poly, dtype=np.float32),
                    "centroid": np.asarray(centroid, dtype=np.float32),
                }
            )
        if len(items) < 2:
            return None

        items.sort(key=lambda item: float(item["centroid"][0]))
        left = items[0]["centroid"]
        right = items[1]["centroid"]
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        pair_dist = math.hypot(dx, dy)
        if pair_dist <= 1e-6:
            return None

        angle_from_x_axis_deg = abs(
            math.degrees(math.atan2(abs(dy), max(1e-6, abs(dx))))
        )
        return {
            "dx": float(dx),
            "dy": float(dy),
            "pair_dist": float(pair_dist),
            "angle_from_x_axis_deg": float(angle_from_x_axis_deg),
        }

    def _negative_cutout_pair_x_axis_pass(self, polygons):
        metrics = self._negative_cutout_pair_x_axis_metrics(polygons)
        if not isinstance(metrics, dict):
            return False, None
        max_angle = float(
            getattr(
                self,
                "_negative_cutout_pair_x_axis_max_angle_deg",
                NEGATIVE_CUTOUT_PAIR_X_AXIS_MAX_ANGLE_DEG,
            )
        )
        passed = float(metrics["angle_from_x_axis_deg"]) <= max_angle
        metrics = dict(metrics)
        metrics["max_angle_deg"] = float(max_angle)
        metrics["passed"] = bool(passed)
        return bool(passed), metrics

    def _build_shape_match_candidates(
        self,
        feature_mask,
        *,
        crop_x=0,
        crop_y=0,
        crop_w=None,
        crop_h=None,
    ):
        """Find brick candidates by matching HSV blobs against the face polygon via matchShapes."""
        if feature_mask is None:
            return []
        mask_arr = np.asarray(feature_mask, dtype=np.uint8)
        if mask_arr.ndim != 2 or mask_arr.size == 0:
            return []

        h_mask, w_mask = mask_arr.shape[:2]
        ex, ey = int(crop_x), int(crop_y)
        ew = int(crop_w) if crop_w is not None else w_mask - ex
        eh = int(crop_h) if crop_h is not None else h_mask - ey
        roi = mask_arr[ey:ey + eh, ex:ex + ew]

        bridge_px = max(0, int(getattr(self, "_shape_match_gap_bridge_px", 0)))
        if bridge_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, bridge_px))
            roi = cv2.dilate(roi, kernel, iterations=1)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_area = max(1000.0, float(ew * eh) * 0.02)
        score_max = float(getattr(self, "_shape_match_score_max", BRICK_FACE_MATCH_MAX_FULL))
        templates = getattr(self, "_face_shape_templates", None)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            fill_ratio = float(area) / float(max(1, bw * bh))
            if fill_ratio < BRICK_SHAPE_FILL_RATIO_MIN:
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw <= 0.0 or rh <= 0.0:
                continue
            aspect = float(max(rw, rh) / max(1e-6, min(rw, rh)))
            expected_aspects = (
                float(BRICK_WIDTH_MM / BRICK_HEIGHT_MM),
                float(BRICK_WIDTH_MM / BRICK_DEPTH_MM),
                float(BRICK_HEIGHT_MM / BRICK_DEPTH_MM),
            )
            rel_err = min(abs(aspect - exp) / max(1e-6, exp) for exp in expected_aspects)
            if rel_err > BRICK_SHAPE_REL_ERROR_MAX:
                continue

            shape_profile = "full"
            shape_match_score = 0.0
            if isinstance(templates, dict) and templates:
                candidate_shape = self._shape_match_contour(cnt)
                if candidate_shape is None:
                    continue
                best_score = None
                best_profile = "full"
                for prof, templ in templates.items():
                    try:
                        s = float(cv2.matchShapes(candidate_shape, templ, cv2.CONTOURS_MATCH_I1, 0.0))
                        if best_score is None or s < best_score:
                            best_score = s
                            best_profile = prof
                    except Exception:
                        pass
                if best_score is None or best_score > score_max:
                    continue
                shape_profile = best_profile
                shape_match_score = best_score

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx_frame = float(M["m10"] / M["m00"]) + float(ex)
            cy_frame = float(M["m01"] / M["m00"]) + float(ey)

            candidates.append({
                "contour": cnt,
                "center_x": int(round(cx_frame)),
                "center_y": int(round(cy_frame)),
                "rect": rect,
                "bbox": (bx + ex, by + ey, bw, bh),
                "area": float(area),
                "partial": False,
                "partial_kind": None,
                "partial_label": None,
                "partial_edges": {},
                "shape_profile": shape_profile,
                "shape_match_score": float(shape_match_score) if isinstance(shape_match_score, (int, float)) else 0.0,
                "negative_cutout_polygons": [],
                "selection_anchor_x": cx_frame,
                "selection_anchor_y": cy_frame,
                "negative_cutout_pair_x_axis_metrics": None,
            })

        candidates.sort(key=lambda c: float(c.get("shape_match_score") or 999.0))
        return candidates

    def _build_trapezoid_brick_candidates(
        self,
        feature_mask,
        *,
        crop_x=0,
        crop_y=0,
        crop_w=None,
        crop_h=None,
    ):
        """Find bricks by detecting dark horizontal slot inner contours inside cyan blobs.

        Each physical brick has one wide dark slot. We identify each slot as a dark inner
        contour (hole) within a cyan outer region, then infer the brick bbox from the
        slot's position and the model's slot-to-face size ratios. This lets two vertically
        stacked bricks produce two separate candidates even when their cyan regions merge.
        """
        if feature_mask is None:
            return []
        mask_arr = np.asarray(feature_mask, dtype=np.uint8)
        if mask_arr.ndim != 2 or mask_arr.size == 0:
            return []

        face_cutouts_model = getattr(self, "_face_cutouts_model", None)
        if not isinstance(face_cutouts_model, list) or len(face_cutouts_model) < 1:
            return []

        # Compute model geometry ratios for bbox estimation from a detected slot.
        cutout_pts = face_cutouts_model[0]  # shape (N, 2) in mm
        slot_ys = cutout_pts[:, 1]
        slot_y_min_mm = float(np.min(slot_ys))
        slot_y_max_mm = float(np.max(slot_ys))
        slot_h_mm = max(1e-3, slot_y_max_mm - slot_y_min_mm)

        face_poly = getattr(self, "_face_polygon_model", None)
        if face_poly is not None and isinstance(face_poly, np.ndarray) and face_poly.shape[0] >= 3:
            face_y_min_mm = float(np.min(face_poly[:, 1]))
            face_y_max_mm = float(np.max(face_poly[:, 1]))
            face_h_mm = max(1e-3, face_y_max_mm - face_y_min_mm)
            face_w_mm = max(1e-3, float(np.max(face_poly[:, 0])) - float(np.min(face_poly[:, 0])))
        else:
            face_h_mm = BRICK_HEIGHT_MM
            face_w_mm = BRICK_WIDTH_MM

        # Ratio of face height to slot height (used to scale bbox from detected slot size)
        face_to_slot_h = face_h_mm / slot_h_mm
        # Fraction of face height above the slot top
        slot_top_ratio = (slot_y_min_mm - (face_y_min_mm if face_poly is not None else 0.0)) / face_h_mm

        h_mask, w_mask = mask_arr.shape[:2]
        ex, ey = int(crop_x), int(crop_y)
        ew = int(crop_w) if crop_w is not None else w_mask - ex
        eh = int(crop_h) if crop_h is not None else h_mask - ey
        roi = mask_arr[ey:ey + eh, ex:ex + ew]

        # Horizontal close fills within-brick pixel gaps without merging separate bricks vertically.
        close_w = max(3, min(15, ew // 40))
        kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 3))
        roi_closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kern_close)

        contours, hierarchy = cv2.findContours(roi_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            return []
        hierarchy_arr = np.asarray(hierarchy)
        if hierarchy_arr.ndim != 3 or hierarchy_arr.shape[0] < 1:
            return []

        # Minimum slot area: must be big enough to be a real slot, not noise.
        min_slot_area = max(50.0, float(getattr(self, "_negative_cutout_min_area_px", NEGATIVE_CUTOUT_MIN_AREA_PX)) * 5.0)

        candidates = []
        for idx, cnt in enumerate(contours):
            # Only inner contours (dark holes inside a cyan blob).
            parent_idx = int(hierarchy_arr[0, idx, 3])
            if parent_idx < 0:
                continue

            area = float(cv2.contourArea(cnt))
            if area < min_slot_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue

            # The slot must be horizontally elongated (wide and thin).
            slot_aspect = float(bw) / float(bh)
            if slot_aspect < 2.0 or slot_aspect > 25.0:
                continue

            # The slot must not dominate the parent (avoids huge false-positive holes).
            parent_cnt = contours[parent_idx]
            parent_area = float(cv2.contourArea(parent_cnt))
            if parent_area > 0 and area / parent_area > 0.70:
                continue

            # The slot must be within the middle 80% of the parent's y-extent.
            pbx, pby, pbw, pbh = cv2.boundingRect(parent_cnt)
            if pbh > 0:
                slot_rel_y = float(by - pby) / float(pbh)
                if slot_rel_y < 0.05 or slot_rel_y > 0.90:
                    continue

            # Estimate the brick bbox from slot position + model ratios.
            # Scale: slot_h_px / slot_h_mm gives px-per-mm; then face_h_px = scale * face_h_mm
            scale_px_per_mm = float(bh) / slot_h_mm
            face_h_px = int(round(face_h_mm * scale_px_per_mm))
            face_w_px = int(round(face_w_mm * scale_px_per_mm))
            # Slot top is slot_top_ratio down from the face top.
            face_top_px = (by + ey) - int(round(slot_top_ratio * face_h_px))
            face_left_px = (bx + ex) + bw // 2 - face_w_px // 2

            # Centroid from slot position (center of slot = good per-brick anchor)
            cx_frame = float(bx + bw / 2.0) + float(ex)
            cy_frame = float(by + bh / 2.0) + float(ey)

            candidates.append({
                "contour": parent_cnt,
                "center_x": int(round(cx_frame)),
                "center_y": int(round(cy_frame)),
                "rect": cv2.minAreaRect(cnt),
                "bbox": (face_left_px, face_top_px, face_w_px, face_h_px),
                "area": area,
                "partial": False,
                "partial_kind": None,
                "partial_label": None,
                "partial_edges": {},
                "shape_profile": "full",
                "shape_match_score": 0.0,
                "negative_cutout_polygons": [],
                "selection_anchor_x": cx_frame,
                "selection_anchor_y": cy_frame,
                "negative_cutout_pair_x_axis_metrics": None,
                "scale_px_per_mm": scale_px_per_mm,
            })

        candidates.sort(key=lambda c: float(c.get("center_y", 0)))
        return candidates

    def _build_negative_cutout_pair_candidates(
        self,
        feature_mask,
        *,
        crop_x=0,
        crop_y=0,
        crop_w=None,
        crop_h=None,
    ):
        if feature_mask is None or not self._uses_negative_cutout_gate():
            return []

        mask_arr = np.asarray(feature_mask, dtype=np.uint8)
        if mask_arr.ndim != 2 or mask_arr.size == 0:
            return []

        model_cutouts = self._sorted_face_cutout_models()
        if len(model_cutouts) < 2:
            return []

        model_left = model_cutouts[0]
        model_right = model_cutouts[1]
        model_pair_center = (model_left["centroid"] + model_right["centroid"]) * 0.5
        model_pair_vec = model_right["centroid"] - model_left["centroid"]
        model_pair_dist = float(np.linalg.norm(model_pair_vec))
        if model_pair_dist <= 1e-6:
            return []

        try:
            contours, hierarchy = cv2.findContours(
                mask_arr,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        except Exception:
            contours, hierarchy = [], None
        if not contours or hierarchy is None:
            return []

        hierarchy_arr = np.asarray(hierarchy)
        if hierarchy_arr.ndim != 3 or hierarchy_arr.shape[0] < 1:
            return []

        ring_radius = max(
            1,
            int(
                getattr(
                    self,
                    "_negative_cutout_ring_dilate_px",
                    NEGATIVE_CUTOUT_RING_DILATE_PX,
                )
            ),
        )
        ring_kernel_size = max(3, (ring_radius * 2) + 1)
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ring_kernel_size, ring_kernel_size),
        )
        cyan_fill_max = float(
            getattr(
                self,
                "_negative_cutout_cyan_fill_max",
                NEGATIVE_CUTOUT_CYAN_FILL_MAX,
            )
        )
        ring_cyan_min = float(
            getattr(
                self,
                "_negative_cutout_ring_cyan_min",
                NEGATIVE_CUTOUT_RING_CYAN_MIN,
            )
        )
        min_area = float(
            getattr(self, "_negative_cutout_min_area_px", NEGATIVE_CUTOUT_MIN_AREA_PX)
        )
        pass_profiles = self._negative_cutout_triangle_pass_profiles()

        triangle_items = []
        for idx, contour in enumerate(contours):
            parent_idx = int(hierarchy_arr[0, idx, 3])
            if parent_idx < 0:
                continue

            tri = self._triangle_from_contour(contour)
            if tri is None:
                continue
            metrics = self._triangle_metrics_pass(tri)
            if metrics is None:
                continue

            cyan_fill_ratio, tri_mask, tri_area = self._mask_fill_ratio(mask_arr, tri)
            if cyan_fill_ratio is None or tri_mask is None:
                continue
            if tri_area < max(4.0, min_area * 0.2):
                continue
            if float(cyan_fill_ratio) > cyan_fill_max:
                continue

            ring_mask = cv2.dilate(tri_mask, ring_kernel, iterations=1)
            ring_mask = cv2.subtract(ring_mask, tri_mask)
            ring_area = float(cv2.countNonZero(ring_mask))
            if ring_area < max(4.0, min_area * 0.35):
                continue
            ring_active = cv2.bitwise_and(mask_arr, mask_arr, mask=ring_mask)
            ring_cyan_ratio = float(cv2.countNonZero(ring_active)) / ring_area
            if float(ring_cyan_ratio) < ring_cyan_min:
                continue

            centroid = self._polygon_centroid(tri)
            if centroid is None:
                continue

            triangle_items.append(
                {
                    "parent_idx": parent_idx,
                    "polygon": tri.astype(np.float32),
                    "centroid": np.asarray(centroid, dtype=np.float32),
                    "mask": tri_mask,
                    "area": float(tri_area),
                    "metrics": metrics,
                    "ring_cyan_ratio": float(ring_cyan_ratio),
                    "cyan_fill_ratio": float(cyan_fill_ratio),
                }
            )

        if len(triangle_items) < 2:
            return []

        crop_w_val = int(mask_arr.shape[1] if crop_w is None else crop_w)
        crop_h_val = int(mask_arr.shape[0] if crop_h is None else crop_h)
        face_polygon_model = getattr(self, "_face_polygon_model", None)
        candidates = []
        for left_idx in range(len(triangle_items)):
            for right_idx in range(left_idx + 1, len(triangle_items)):
                first = triangle_items[left_idx]
                second = triangle_items[right_idx]
                if int(first["parent_idx"]) != int(second["parent_idx"]):
                    continue

                ordered = sorted(
                    [first, second],
                    key=lambda item: float(item["centroid"][0]),
                )
                actual_left, actual_right = ordered
                pair_x_axis_pass, pair_x_axis_metrics = (
                    self._negative_cutout_pair_x_axis_pass(
                        [actual_left["polygon"], actual_right["polygon"]]
                    )
                )
                if not pair_x_axis_pass:
                    continue
                actual_pair_center = (
                    actual_left["centroid"] + actual_right["centroid"]
                ) * 0.5
                actual_pair_vec = actual_right["centroid"] - actual_left["centroid"]
                actual_pair_dist = float(np.linalg.norm(actual_pair_vec))
                if actual_pair_dist < 4.0:
                    continue

                scale = float(actual_pair_dist) / float(model_pair_dist)
                model_angle = math.atan2(
                    float(model_pair_vec[1]),
                    float(model_pair_vec[0]),
                )
                actual_angle = math.atan2(
                    float(actual_pair_vec[1]),
                    float(actual_pair_vec[0]),
                )
                theta = float(actual_angle - model_angle)
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)

                projected_cutouts_local = []
                for model_item in (model_left, model_right):
                    projected = self._transform_model_points_similarity(
                        model_item["polygon"],
                        model_origin=model_pair_center,
                        image_origin=actual_pair_center,
                        scale=scale,
                        cos_theta=cos_theta,
                        sin_theta=sin_theta,
                    )
                    if projected is None or projected.shape[0] < 3:
                        projected_cutouts_local = []
                        break
                    projected_cutouts_local.append(projected.astype(np.float32))
                if len(projected_cutouts_local) != 2:
                    continue

                match_score = None
                for pass_idx, profile in enumerate(pass_profiles):
                    pair_score = 0.0
                    pair_pass = True
                    for actual_item, projected_poly in zip(
                        (actual_left, actual_right),
                        projected_cutouts_local,
                    ):
                        proj_fill_ratio, proj_mask, proj_area = self._mask_fill_ratio(
                            mask_arr,
                            projected_poly,
                        )
                        if proj_fill_ratio is None or proj_mask is None or proj_area < min_area:
                            pair_pass = False
                            break

                        ring_mask = cv2.dilate(proj_mask, ring_kernel, iterations=1)
                        ring_mask = cv2.subtract(ring_mask, proj_mask)
                        ring_area = float(cv2.countNonZero(ring_mask))
                        if ring_area < max(4.0, min_area * 0.35):
                            pair_pass = False
                            break
                        ring_active = cv2.bitwise_and(mask_arr, mask_arr, mask=ring_mask)
                        ring_cyan_ratio = float(cv2.countNonZero(ring_active)) / ring_area
                        if (
                            float(proj_fill_ratio) > cyan_fill_max
                            or float(ring_cyan_ratio) < ring_cyan_min
                        ):
                            pair_pass = False
                            break

                        actual_area = float(actual_item["area"])
                        if actual_area <= 0.0:
                            pair_pass = False
                            break
                        area_ratio = actual_area / proj_area
                        if (
                            area_ratio < float(profile["area_ratio_min"])
                            or area_ratio > float(profile["area_ratio_max"])
                        ):
                            pair_pass = False
                            break

                        overlap_mask = cv2.bitwise_and(actual_item["mask"], proj_mask)
                        overlap_area = float(cv2.countNonZero(overlap_mask))
                        overlap_proj_ratio = overlap_area / proj_area
                        if overlap_proj_ratio < float(profile["overlap_proj_min"]):
                            pair_pass = False
                            break
                        overlap_tri_ratio = overlap_area / actual_area
                        if overlap_tri_ratio < float(profile["overlap_tri_min"]):
                            pair_pass = False
                            break

                        cutout_score = max(
                            float(proj_fill_ratio),
                            max(0.0, 1.0 - float(ring_cyan_ratio)),
                        )
                        cutout_score += abs(float(area_ratio) - 1.0)
                        cutout_score += max(0.0, 1.0 - float(overlap_proj_ratio))
                        cutout_score += max(0.0, 1.0 - float(overlap_tri_ratio))
                        cutout_score += self._negative_cutout_focus_penalty(
                            actual_item["polygon"],
                            mask_arr.shape,
                        )
                        pair_score = max(pair_score, float(cutout_score))

                    if not pair_pass:
                        continue
                    if pass_idx > 0:
                        pair_score += 0.15
                    match_score = float(pair_score)
                    break

                if match_score is None:
                    continue

                projected_face_local = None
                if isinstance(face_polygon_model, np.ndarray) and face_polygon_model.shape[0] >= 3:
                    projected_face_local = self._transform_model_points_similarity(
                        face_polygon_model,
                        model_origin=model_pair_center,
                        image_origin=actual_pair_center,
                        scale=scale,
                        cos_theta=cos_theta,
                        sin_theta=sin_theta,
                    )
                if projected_face_local is None or projected_face_local.shape[0] < 3:
                    projected_face_local = np.vstack(projected_cutouts_local).astype(
                        np.float32
                    )

                try:
                    face_rect_local = cv2.minAreaRect(
                        projected_face_local.reshape(-1, 1, 2).astype(np.float32)
                    )
                except Exception:
                    continue

                raw_x1 = float(np.min(projected_face_local[:, 0]))
                raw_y1 = float(np.min(projected_face_local[:, 1]))
                raw_x2 = float(np.max(projected_face_local[:, 0]))
                raw_y2 = float(np.max(projected_face_local[:, 1]))
                raw_w = max(1.0, raw_x2 - raw_x1)
                raw_h = max(1.0, raw_y2 - raw_y1)
                partial_info = self._partial_info_from_edges(
                    left_touch=raw_x1 <= 0.0,
                    top_touch=raw_y1 <= 0.0,
                    right_touch=raw_x2 >= float(crop_w_val),
                    bottom_touch=raw_y2 >= float(crop_h_val),
                )

                projected_face_frame = projected_face_local.copy()
                projected_face_frame[:, 0] += float(crop_x)
                projected_face_frame[:, 1] += float(crop_y)
                face_rect_frame = (
                    (
                        float(face_rect_local[0][0]) + float(crop_x),
                        float(face_rect_local[0][1]) + float(crop_y),
                    ),
                    face_rect_local[1],
                    face_rect_local[2],
                )
                actual_cutouts_frame = []
                for actual_item in (actual_left, actual_right):
                    poly_frame = actual_item["polygon"].copy()
                    poly_frame[:, 0] += float(crop_x)
                    poly_frame[:, 1] += float(crop_y)
                    actual_cutouts_frame.append(poly_frame.astype(np.float32))

                actual_pair_center_frame = (
                    float(actual_pair_center[0]) + float(crop_x),
                    float(actual_pair_center[1]) + float(crop_y),
                )
                projected_face_contour = projected_face_frame.reshape(-1, 1, 2).astype(
                    np.float32
                )
                try:
                    projected_area = float(
                        abs(cv2.contourArea(projected_face_contour))
                    )
                except Exception:
                    projected_area = float(raw_w * raw_h)

                candidates.append(
                    {
                        "contour": projected_face_contour,
                        "center_x": int(round(actual_pair_center_frame[0])),
                        "center_y": int(round(actual_pair_center_frame[1])),
                        "rect": face_rect_frame,
                        "bbox": (
                            int(round(raw_x1 + float(crop_x))),
                            int(round(raw_y1 + float(crop_y))),
                            int(round(raw_w)),
                            int(round(raw_h)),
                        ),
                        "area": float(max(1.0, projected_area)),
                        "partial": bool(partial_info.get("partial")),
                        "partial_kind": partial_info.get("kind"),
                        "partial_label": partial_info.get("label"),
                        "partial_edges": partial_info.get("edges") or {},
                        "shape_profile": "full",
                        "shape_match_score": float(match_score),
                        "negative_cutout_polygons": actual_cutouts_frame,
                        "selection_anchor_x": float(actual_pair_center_frame[0]),
                        "selection_anchor_y": float(actual_pair_center_frame[1]),
                        "negative_cutout_pair_x_axis_metrics": pair_x_axis_metrics,
                    }
                )

        candidates.sort(
            key=lambda item: float(item.get("shape_match_score") or 999.0)
        )
        return candidates

    def _triangle_metrics(self, triangle_points):
        pts = np.asarray(triangle_points, dtype=np.float32).reshape(-1, 2)
        if pts.shape != (3, 2):
            return None

        sides = []
        for i in range(3):
            j = (i + 1) % 3
            sides.append(float(np.linalg.norm(pts[j] - pts[i])))
        min_side = min(sides)
        max_side = max(sides)
        if min_side <= 1e-6:
            return None
        side_ratio = max_side / min_side

        angles = []
        for i in range(3):
            prev_pt = pts[(i - 1) % 3]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % 3]
            v1 = prev_pt - curr_pt
            v2 = next_pt - curr_pt
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom <= 1e-6:
                return None
            cos_val = float(np.dot(v1, v2) / denom)
            cos_val = max(-1.0, min(1.0, cos_val))
            angles.append(math.degrees(math.acos(cos_val)))
        angle_spread = max(angles) - min(angles)

        return {
            "side_ratio": float(side_ratio),
            "angle_spread_deg": float(angle_spread),
            "angles_deg": [float(a) for a in angles],
        }

    def _extract_regular_dark_triangle(self, cyan_mask, cutout_mask, cutout_proj, min_area):
        if cyan_mask is None or cutout_mask is None:
            return None

        dark_in_cutout = cv2.bitwise_and(cv2.bitwise_not(cyan_mask), cutout_mask)
        dark_cnts, _ = cv2.findContours(
            dark_in_cutout, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not dark_cnts:
            return None

        proj_mask = np.zeros(cyan_mask.shape[:2], dtype=np.uint8)
        proj_draw = np.round(np.asarray(cutout_proj, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(proj_mask, [proj_draw], 255)
        proj_area = float(cv2.countNonZero(proj_mask))
        if proj_area <= 0.0:
            return None

        pass_profiles = self._negative_cutout_triangle_pass_profiles()

        best_poly = None
        best_score = None
        for pass_idx, profile in enumerate(pass_profiles):
            for cnt in dark_cnts:
                area = float(cv2.contourArea(cnt))
                if area < max(4.0, min_area * 0.2):
                    continue

                tri = self._triangle_from_contour(cnt)
                if tri is None:
                    continue

                metrics = self._triangle_metrics_pass(tri)
                if metrics is None:
                    continue

                tri_mask = np.zeros(cyan_mask.shape[:2], dtype=np.uint8)
                tri_draw = np.round(tri).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(tri_mask, [tri_draw], 255)
                tri_area = float(cv2.countNonZero(tri_mask))
                if tri_area <= 0.0:
                    continue
                area_ratio = tri_area / proj_area
                if (
                    area_ratio < float(profile["area_ratio_min"])
                    or area_ratio > float(profile["area_ratio_max"])
                ):
                    continue
                overlap_mask = cv2.bitwise_and(tri_mask, proj_mask)
                overlap_area = float(cv2.countNonZero(overlap_mask))
                overlap_proj_ratio = overlap_area / proj_area
                if overlap_proj_ratio < float(profile["overlap_proj_min"]):
                    continue
                overlap_tri_ratio = overlap_area / tri_area
                if overlap_tri_ratio < float(profile["overlap_tri_min"]):
                    continue

                # Prefer regular, correctly-sized triangles and strict-pass matches.
                score = float(metrics["side_ratio"] - 1.0) + (
                    float(metrics["angle_spread_deg"]) / 180.0
                ) + abs(area_ratio - 1.0)
                score += self._negative_cutout_focus_penalty(tri, cyan_mask.shape)
                if pass_idx > 0:
                    score += 0.15
                if best_score is None or score < best_score:
                    best_score = score
                    best_poly = tri
            if best_poly is not None:
                break

        return best_poly

    def _match_negative_cutout_pair(self, feature_mask, rect, contour=None):
        if not self._uses_negative_cutout_gate():
            return False, None
        if feature_mask is None:
            return False, None

        try:
            rect_points = self._ordered_rect_points(rect)
        except Exception:
            return False, None
        if not isinstance(rect_points, np.ndarray) or rect_points.shape != (4, 2):
            return False, None

        mask_arr = np.asarray(feature_mask, dtype=np.uint8)
        if mask_arr.ndim != 2:
            return False, None

        body_mask = np.zeros(mask_arr.shape[:2], dtype=np.uint8)
        rect_draw = np.round(rect_points).astype(np.int32).reshape(-1, 1, 2)
        try:
            cv2.fillPoly(body_mask, [rect_draw], 255)
        except Exception:
            return False, None

        ring_radius = max(
            1,
            int(
                getattr(
                    self,
                    "_negative_cutout_ring_dilate_px",
                    NEGATIVE_CUTOUT_RING_DILATE_PX,
                )
            ),
        )
        ring_kernel_size = max(3, (ring_radius * 2) + 1)
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ring_kernel_size, ring_kernel_size),
        )
        cyan_fill_max = float(
            getattr(
                self,
                "_negative_cutout_cyan_fill_max",
                NEGATIVE_CUTOUT_CYAN_FILL_MAX,
            )
        )
        ring_cyan_min = float(
            getattr(
                self,
                "_negative_cutout_ring_cyan_min",
                NEGATIVE_CUTOUT_RING_CYAN_MIN,
            )
        )
        min_area = float(
            getattr(self, "_negative_cutout_min_area_px", NEGATIVE_CUTOUT_MIN_AREA_PX)
        )

        cutout_scores = []
        cutout_polygons = []
        passed = 0
        for cutout_points in self._face_cutouts_model:
            cutout_proj = self._project_model_points_to_rect(cutout_points, rect_points)
            if not isinstance(cutout_proj, np.ndarray) or cutout_proj.shape[0] < 3:
                return False, None

            cyan_fill_ratio, cutout_mask, cutout_area = self._mask_fill_ratio(
                mask_arr, cutout_proj
            )
            if cyan_fill_ratio is None or cutout_mask is None or cutout_area < min_area:
                return False, None

            ring_mask = cv2.dilate(cutout_mask, ring_kernel, iterations=1)
            ring_mask = cv2.subtract(ring_mask, cutout_mask)
            ring_mask = cv2.bitwise_and(ring_mask, body_mask)
            ring_area = float(cv2.countNonZero(ring_mask))
            if ring_area < min_area:
                return False, None
            ring_active = cv2.bitwise_and(mask_arr, mask_arr, mask=ring_mask)
            ring_cyan_ratio = float(cv2.countNonZero(ring_active)) / ring_area

            passed_cutout = (
                float(cyan_fill_ratio) <= cyan_fill_max
                and float(ring_cyan_ratio) >= ring_cyan_min
            )
            actual_poly = None
            if passed_cutout:
                actual_poly = self._extract_regular_dark_triangle(
                    mask_arr,
                    cutout_mask,
                    cutout_proj,
                    min_area,
                )
                passed_cutout = actual_poly is not None
            if passed_cutout:
                passed += 1
                cutout_polygons.append(actual_poly.astype(np.float32))
            else:
                cutout_polygons.append(cutout_proj.astype(np.float32))
            cutout_scores.append(
                {
                    "cyan_fill_ratio": float(cyan_fill_ratio),
                    "ring_cyan_ratio": float(ring_cyan_ratio),
                    "passed": bool(passed_cutout),
                }
            )

        if passed < 2:
            return False, {
                "score": 1.0,
                "cutouts": cutout_scores,
                "passed": passed,
                "polygons": cutout_polygons,
                "selection_anchor": self._negative_cutout_selection_anchor(
                    cutout_polygons
                ),
            }

        pair_x_axis_pass, pair_x_axis_metrics = self._negative_cutout_pair_x_axis_pass(
            cutout_polygons
        )
        if not pair_x_axis_pass:
            return False, {
                "score": 1.0,
                "cutouts": cutout_scores,
                "passed": passed,
                "polygons": cutout_polygons,
                "selection_anchor": self._negative_cutout_selection_anchor(
                    cutout_polygons
                ),
                "pair_x_axis": pair_x_axis_metrics,
            }

        score = 0.0
        for item in cutout_scores:
            score = max(
                score,
                float(item["cyan_fill_ratio"]),
                max(0.0, 1.0 - float(item["ring_cyan_ratio"])),
            )
        return True, {
            "score": float(score),
            "cutouts": cutout_scores,
            "passed": passed,
            "polygons": cutout_polygons,
            "selection_anchor": self._negative_cutout_selection_anchor(
                cutout_polygons
            ),
            "pair_x_axis": pair_x_axis_metrics,
        }

    def _match_negative_cutout_pair_in_box(self, frame, x1, y1, x2, y2):
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return False, None

        feature_mask, contour_mask = self._build_hsv_masks(crop)
        if feature_mask is None or contour_mask is None:
            return False, None

        contours, _ = cv2.findContours(
            contour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return False, None

        best_summary = None
        best_score = None
        best_passed = -1
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area <= 0.0:
                continue
            gate_rect = cv2.minAreaRect(cnt)
            matched, summary = self._match_negative_cutout_pair(feature_mask, gate_rect)
            if matched:
                return True, summary
            if not isinstance(summary, dict):
                continue
            passed = int(summary.get("passed") or 0)
            score = float(summary.get("score") or 1.0)
            if (
                passed > best_passed
                or (passed == best_passed and (best_score is None or score < best_score))
            ):
                best_passed = passed
                best_score = score
                best_summary = summary
        return False, best_summary

    def _extract_shape_contour_in_box(self, frame, x1, y1, x2, y2):
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        crop_area = float(crop.shape[0] * crop.shape[1])
        min_area = max(8.0, crop_area * float(FALLBACK_SHAPE_MIN_AREA_RATIO))
        best = None
        best_area = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            if area > best_area:
                best = cnt
                best_area = area
        return best

    def _partial_from_shape_profile(self, profile):
        if profile == "top_half":
            return {
                "partial": True,
                "kind": "top_half",
                "label": PARTIAL_LABEL_TOP,
            }
        if profile == "bottom_half":
            return {
                "partial": True,
                "kind": "bottom_half",
                "label": PARTIAL_LABEL_BOTTOM,
            }
        return {
            "partial": False,
            "kind": None,
            "label": None,
        }

    def _project_model_points_to_rect(self, model_points, rect_points):
        if not isinstance(model_points, np.ndarray) or model_points.size == 0:
            return None
        if not isinstance(rect_points, np.ndarray) or rect_points.shape != (4, 2):
            return None

        half_width = float(BRICK_WIDTH_MM) * 0.5
        model_height = float(BRICK_HEIGHT_MM)

        # World model uses Y-positive-up. Map model corners into image TL/TR/BR/BL.
        src = np.asarray(
            [
                [-half_width, model_height],  # top-left in model coords
                [half_width, model_height],   # top-right
                [half_width, 0.0],            # bottom-right
                [-half_width, 0.0],           # bottom-left
            ],
            dtype=np.float32,
        )
        
        # Preserve brick's aspect ratio (64mm wide × 48mm tall = 4:3)
        rect_pts = rect_points.astype(np.float32)
        
        # Calculate center of bounding box
        center = np.mean(rect_pts, axis=0)
        
        # Calculate width/height vectors of the detected rectangle
        tl, tr, br, bl = rect_pts
        width_vector = tr - tl
        height_vector = bl - tl
        rect_width = np.linalg.norm(width_vector)
        rect_height = np.linalg.norm(height_vector)
        
        # Normalize vectors
        norm_width = width_vector / rect_width if rect_width > 0 else np.array([1.0, 0.0])
        norm_height = height_vector / rect_height if rect_height > 0 else np.array([0.0, 1.0])
        
        # Brick is BRICK_WIDTH_MM × BRICK_HEIGHT_MM in the world model.
        # Scale to fit within the detected box while preserving aspect ratio
        target_aspect = float(BRICK_WIDTH_MM) / max(1e-6, float(BRICK_HEIGHT_MM))
        current_aspect = rect_width / rect_height if rect_height > 0 else target_aspect
        
        if current_aspect > target_aspect:
            # Detected box is too wide; scale height up
            actual_width = rect_height * target_aspect
            actual_height = rect_height
        else:
            # Detected box is too tall; scale width up
            actual_width = rect_width
            actual_height = rect_width / target_aspect
        
        # Create adjusted destination rectangle centered at the same center
        half_width = norm_width * actual_width * 0.5
        half_height = norm_height * actual_height * 0.5
        
        new_tl = center - half_width - half_height
        new_tr = center + half_width - half_height
        new_br = center + half_width + half_height
        new_bl = center - half_width + half_height
        
        dst = np.asarray([new_tl, new_tr, new_br, new_bl], dtype=np.float32)
        
        try:
            H = cv2.getPerspectiveTransform(src, dst)
        except Exception:
            return None
        pts_in = np.asarray(model_points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(pts_in, H)
        except Exception:
            return None
        if projected is None:
            return None
        return projected.reshape(-1, 2)

    def _project_model_points_to_rect_direct(self, model_points, rect_points):
        if not isinstance(model_points, np.ndarray) or model_points.size == 0:
            return None
        if not isinstance(rect_points, np.ndarray) or rect_points.shape != (4, 2):
            return None

        half_width = float(BRICK_WIDTH_MM) * 0.5
        model_height = float(BRICK_HEIGHT_MM)
        src = np.asarray(
            [
                [-half_width, model_height],
                [half_width, model_height],
                [half_width, 0.0],
                [-half_width, 0.0],
            ],
            dtype=np.float32,
        )
        dst = np.asarray(rect_points, dtype=np.float32)
        try:
            transform = cv2.getPerspectiveTransform(src, dst)
        except Exception:
            return None
        pts_in = np.asarray(model_points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(pts_in, transform)
        except Exception:
            return None
        if projected is None:
            return None
        return projected.reshape(-1, 2)

    def _draw_primary_face_outline(self, frame, primary):
        if not isinstance(primary, dict):
            return
        contour = primary.get("debug_outline_contour")
        if contour is None:
            contour = primary.get("contour")
        contour_arr = None
        if contour is not None:
            try:
                contour_arr = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
            except Exception:
                contour_arr = None
        rect = primary.get("rect")
        if contour_arr is None and not rect:
            return

        is_partial = bool(primary.get("partial"))
        outline_color = PARTIAL_COLOR_BGR if is_partial else (0, 255, 0)
        outline_thickness = 3 if is_partial else 2

        if self._uses_negative_cutout_gate():
            rect_points = None
            if rect is not None:
                try:
                    rect_points = self._ordered_rect_points(rect)
                except Exception:
                    rect_points = None
            cutout_polygons = primary.get("negative_cutout_polygons")
            if not isinstance(cutout_polygons, list) or not cutout_polygons:
                if isinstance(rect_points, np.ndarray) and rect_points.shape == (4, 2):
                    cutout_polygons = self._project_negative_cutout_polygons(rect_points)

            if contour_arr is not None and contour_arr.shape[0] >= 3:
                cv2.polylines(
                    frame,
                    [contour_arr],
                    True,
                    outline_color,
                    outline_thickness,
                    cv2.LINE_AA,
                )
            elif isinstance(rect_points, np.ndarray) and rect_points.shape == (4, 2):
                rect_draw = np.round(rect_points).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    frame,
                    [rect_draw],
                    True,
                    outline_color,
                    outline_thickness,
                    cv2.LINE_AA,
                )
            if isinstance(cutout_polygons, list) and cutout_polygons:
                for poly in cutout_polygons:
                    poly_arr = np.asarray(poly, dtype=np.float32)
                    if poly_arr.ndim != 2 or poly_arr.shape[0] < 3:
                        continue
                    poly_draw = np.round(poly_arr).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        frame,
                        [poly_draw],
                        True,
                        NEGATIVE_TRIANGLE_COLOR_BGR,
                        2,
                        cv2.LINE_AA,
                    )
            return

        # Prioritize model-constrained shape when rect is available.
        # Only fall back to raw contour if rect is unavailable.
        if rect is None:
            if contour_arr is not None and contour_arr.shape[0] >= 3:
                cv2.polylines(frame, [contour_arr], True, outline_color, outline_thickness, cv2.LINE_AA)
            return

        points = self._ordered_rect_points(rect)
        center = np.asarray(rect[0], dtype=np.float32)
        edges = primary.get("partial_edges") or {}

        shape_profile = str(primary.get("shape_profile") or "").strip().lower()
        if not shape_profile:
            if is_partial:
                partial_kind = str(primary.get("partial_kind") or "").strip().lower()
                if partial_kind in {"top_half", "bottom_half"}:
                    shape_profile = partial_kind
            if not shape_profile:
                shape_profile = "full"

        # Draw canonical world-model face shape for full/top/bottom profiles.
        model_poly = None
        face_polygon_model = getattr(self, "_face_polygon_model", None)
        if isinstance(face_polygon_model, np.ndarray):
            if shape_profile == "top_half":
                y_vals = face_polygon_model[:, 1]
                y_mid = (float(np.min(y_vals)) + float(np.max(y_vals))) * 0.5
                model_poly = self._clip_polygon_y(face_polygon_model, y_mid, keep_above=True)
            elif shape_profile == "bottom_half":
                y_vals = face_polygon_model[:, 1]
                y_mid = (float(np.min(y_vals)) + float(np.max(y_vals))) * 0.5
                model_poly = self._clip_polygon_y(face_polygon_model, y_mid, keep_above=False)
            else:
                model_poly = face_polygon_model

        if isinstance(model_poly, np.ndarray) and model_poly.size >= 6:
            poly_proj = self._project_model_points_to_rect(model_poly, points)
            if isinstance(poly_proj, np.ndarray) and len(poly_proj) >= 3:
                poly_draw = np.round(poly_proj).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [poly_draw], True, outline_color, 2, cv2.LINE_AA)

                return

        if contour_arr is not None and contour_arr.shape[0] >= 3:
            cv2.polylines(frame, [contour_arr], True, outline_color, outline_thickness, cv2.LINE_AA)
            return

        visible_segments = []
        for idx in range(4):
            p0 = points[idx]
            p1 = points[(idx + 1) % 4]
            segment_name = self._rect_segment_name(p0, p1, center)
            if is_partial and bool(edges.get(segment_name)):
                continue
            visible_segments.append((p0, p1))

        if not visible_segments:
            visible_segments = [
                (points[idx], points[(idx + 1) % 4])
                for idx in range(4)
            ]

        for p0, p1 in visible_segments:
            start = tuple(np.intp(np.round(p0)))
            end = tuple(np.intp(np.round(p1)))
            cv2.line(frame, start, end, outline_color, outline_thickness, cv2.LINE_AA)

    def _detect_pink_dot_in_brick(self, frame, primary):
        """Search for the pink dot marker in the lower portion of the brick bbox.

        Returns (found, center_x_px, center_y_px).  The dot sits below the
        dark triangle cutouts, so we search the lower ~45% of the bounding box.
        """
        bx, by, bw, bh = primary["bbox"]
        if bw <= 0 or bh <= 0:
            return False, None, None
        search_top = by + int(bh * 0.55)
        search_bot = min(frame.shape[0], by + bh)
        search_left = max(0, bx - int(bw * 0.05))
        search_right = min(frame.shape[1], bx + bw + int(bw * 0.05))
        if search_top >= search_bot or search_left >= search_right:
            return False, None, None
        crop = frame[search_top:search_bot, search_left:search_right]
        if crop.size == 0:
            return False, None, None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, PINK_DOT_HSV_LOWER, PINK_DOT_HSV_UPPER)
        pink_px = int(cv2.countNonZero(mask))
        min_px = max(4, int(bw * bh * 0.003))
        if pink_px < min_px:
            return False, None, None
        M = cv2.moments(mask)
        if M["m00"] <= 0:
            return False, None, None
        cx = int(round(M["m10"] / M["m00"])) + search_left
        cy = int(round(M["m01"] / M["m00"])) + search_top
        return True, cx, cy

    def _draw_pink_dot_on_brick(self, frame, primary):
        """Draw the pink dot on the overlay — filled if detected, hollow if projected."""
        if not isinstance(primary, dict):
            return
        dot_found = bool(primary.get("_dot_found"))
        dot_cx = primary.get("_dot_cx")
        dot_cy = primary.get("_dot_cy")
        if dot_found and dot_cx is not None and dot_cy is not None:
            cv2.circle(frame, (int(dot_cx), int(dot_cy)), 6, PINK_DOT_COLOR_BGR, -1)
            cv2.circle(frame, (int(dot_cx), int(dot_cy)), 6, (240, 200, 220), 1)
        else:
            rect = primary.get("rect")
            if rect is None:
                return
            try:
                rect_points = self._ordered_rect_points(rect)
                dot_xy = getattr(self, "_pink_dot_model_xy", None) or (0.0, 9.0)
                dot_model = np.asarray([list(dot_xy)], dtype=np.float32)
                dot_proj = self._project_model_points_to_rect(dot_model, rect_points)
                if dot_proj is not None and len(dot_proj) >= 1:
                    dx = int(round(float(dot_proj[0, 0])))
                    dy = int(round(float(dot_proj[0, 1])))
                    cv2.circle(frame, (dx, dy), 6, PINK_DOT_COLOR_BGR, 1)
            except Exception:
                pass

    def _draw_debug_hsv(self, frame, yolo_bricks, hsv_bricks, primary,
                        angle, dist, offset_x, conf):
        """Draw enhanced debug visualization for HSV-segmented bricks."""
        self._draw_brick_id_labels(frame, hsv_bricks)
        if isinstance(hsv_bricks, list) and len(hsv_bricks) > 1:
            self.current_frame = frame
            return

        # Only highlight the selected brick nearest the crosshairs.
        if primary:
            refined_outline = self._refine_primary_contour(frame, primary)
            if refined_outline is not None:
                primary = dict(primary)
                primary["debug_outline_contour"] = refined_outline
            self._draw_primary_face_outline(frame, primary)
            self._draw_pink_dot_on_brick(frame, primary)
            pcx = int(round(float(primary["center_x"])))
            pcy = int(round(float(primary["center_y"])))
            primary_color = PARTIAL_COLOR_BGR if bool(primary.get("partial")) else (0, 255, 0)
            cv2.drawMarker(frame, (pcx, pcy), primary_color,
                           cv2.MARKER_CROSS, 15, 2)

            label = str(primary.get("partial_label") or "").strip()
            if label:
                bx, by, _bw, _bh = primary["bbox"]
                text_y = max(18, int(by) - 8)
                cv2.putText(
                    frame,
                    label,
                    (max(0, int(bx)), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    PARTIAL_COLOR_BGR,
                    2,
                )

            # Red angle line from primary brick center
            _, _, bw, bh = primary["bbox"]
            length = min(bw, bh) // 2
            rad = math.radians(angle - 90.0)
            ex = int(pcx + length * math.cos(rad))
            ey = int(pcy - length * math.sin(rad))
            cv2.line(frame, (pcx, pcy), (ex, ey), (0, 0, 255), 2)

        self.current_frame = frame

    def _draw_brick_id_labels(self, frame, candidates):
        if frame is None or not isinstance(candidates, list) or len(candidates) <= 1:
            return
        def _label_center(candidate):
            bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
            if isinstance(bbox, tuple) and len(bbox) >= 4:
                try:
                    bx, by, bw, bh = [float(v) for v in bbox[:4]]
                    if bw > 0.0 and bh > 0.0:
                        return bx + (bw * 0.5), by + (bh * 0.5)
                except Exception:
                    pass
            return (
                float(candidate.get("center_x", 0.0)),
                float(candidate.get("center_y", 0.0)),
            )

        ordered = sorted(
            [candidate for candidate in candidates if isinstance(candidate, dict)],
            key=lambda candidate: (
                _label_center(candidate)[1],
                _label_center(candidate)[0],
            ),
        )
        for brick_id, candidate in enumerate(ordered):
            label_x, label_y = _label_center(candidate)
            draw_brick_with_id(self, frame, candidate, brick_id, center=(label_x, label_y))

    def _project_negative_cutout_polygons(self, rect_points):
        if not isinstance(rect_points, np.ndarray) or rect_points.shape != (4, 2):
            return []
        face_cutouts_model = getattr(self, "_face_cutouts_model", None)
        if not isinstance(face_cutouts_model, list):
            return []
        polygons = []
        for cutout_points in face_cutouts_model:
            cutout_proj = self._project_model_points_to_rect_direct(cutout_points, rect_points)
            if isinstance(cutout_proj, np.ndarray) and cutout_proj.shape[0] >= 3:
                polygons.append(cutout_proj.astype(np.float32))
        return polygons

    def _draw_negative_cutout_polygons(self, frame, rect_points):
        cutout_polygons = self._project_negative_cutout_polygons(rect_points)
        for poly in cutout_polygons:
            poly_arr = np.asarray(poly, dtype=np.float32)
            if poly_arr.ndim != 2 or poly_arr.shape[0] < 3:
                continue
            poly_draw = np.round(poly_arr).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                frame,
                [poly_draw],
                True,
                NEGATIVE_TRIANGLE_COLOR_BGR,
                2,
                cv2.LINE_AA,
            )

    def calibrate_focal(self, known_distance_mm, frame=None):
        """
        Calibrate focal_px by detecting a brick at a known distance.

        Place a brick at a measured distance (e.g. ArUco-verified), then call:
            detector.calibrate_focal(102.0)

        Args:
            known_distance_mm: True distance to the brick in mm.
            frame: Optional frame to use. If None, captures from camera.

        Returns:
            float: The computed focal_px, or None if detection failed.
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.log.error("calibrate_focal: frame capture failed")
                return None

        bricks = self._detect(frame)
        if not bricks:
            self.log.error("calibrate_focal: no brick detected")
            return None

        bricks.sort(key=lambda b: b[4], reverse=True)
        x1, y1, x2, y2, conf = bricks[0]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        if bbox_w <= 0 and bbox_h <= 0:
            return None

        focal_candidates = []
        if bbox_w > 0:
            focal_candidates.append((known_distance_mm * bbox_w) / BRICK_WIDTH_MM)
        if bbox_h > 0:
            focal_candidates.append((known_distance_mm * bbox_h) / BRICK_HEIGHT_MM)
        if len(focal_candidates) == 1:
            new_focal = float(focal_candidates[0])
        else:
            new_focal = float((0.8 * float(focal_candidates[0])) + (0.2 * float(focal_candidates[-1])))
        if new_focal < 1.0:
            self.log.error(
                "calibrate_focal: computed focal_px=%.1f is too small "
                "(dist=%.1f, bbox_w=%d, bbox_h=%d)",
                new_focal,
                known_distance_mm,
                bbox_w,
                bbox_h,
            )
            return None
        self.focal_px = new_focal
        self.log.info(
            "calibrate_focal: bbox_w=%d bbox_h=%d dist=%.1fmm -> focal_px=%.1f",
            bbox_w,
            bbox_h,
            known_distance_mm,
            self.focal_px,
        )
        return self.focal_px

    def release(self):
        """Release camera resources."""
        if getattr(self, "cap", None):
            self.cap.release()
        trt_engine = getattr(self, "_trt_engine", None)
        if trt_engine is not None:
            trt_engine.close()
            self._trt_engine = None

    def close(self):
        """Release camera resources."""
        self.release()

    def __del__(self):
        self.release()


def build_negative_cutout_shape_detector(
    *,
    hsv_lower=None,
    hsv_upper=None,
    hsv_erode_iterations=0,
    center_lock_enabled=True,
):
    """Build a camera-free detector for the cyan face + dark-triangle shape gate."""
    detector = BrickDetector.__new__(BrickDetector)
    detector.debug = False
    detector.log = logging.getLogger("BrickVisionShapeOnly")
    detector.current_frame = None
    detector.raw_frame = None
    detector.frame_w = DEFAULT_FRAME_W
    detector.frame_h = DEFAULT_FRAME_H
    detector.camera_center_offset_px = 0.0
    detector._hsv_lower = np.array(
        hsv_lower if hsv_lower is not None else CYAN_HSV_WIDE_LOWER,
        dtype=np.uint8,
    )
    detector._hsv_upper = np.array(
        hsv_upper if hsv_upper is not None else CYAN_HSV_WIDE_UPPER,
        dtype=np.uint8,
    )
    detector._hsv_enabled = True
    detector._hsv_erode_iterations = max(0, int(hsv_erode_iterations or 0))
    detector._face_polygon_model = None
    detector._face_cutouts_model = []
    detector._face_lines_model = []
    detector._face_shape_templates = {}
    detector._load_face_shape_model()
    detector._init_shape_match_templates()
    detector._center_lock_enabled = bool(center_lock_enabled)
    detector._center_lock_radius_px = None
    detector._center_switch_margin_px = float(CENTER_SWITCH_MARGIN_PX)
    detector._center_partial_penalty = float(CENTER_PARTIAL_PENALTY)
    detector._center_axis_weight_x = float(CENTER_AXIS_WEIGHT_X)
    detector._center_axis_weight_y = float(CENTER_AXIS_WEIGHT_Y)
    detector._center_lock_prev_center = None
    detector._stack_center_y_row_filter_enabled = STACK_CENTER_Y_ROW_FILTER_ENABLED
    detector._stack_center_y_row_min_candidates = STACK_CENTER_Y_ROW_MIN_CANDIDATES
    detector._stack_center_y_row_tolerance_px = STACK_CENTER_Y_ROW_TOLERANCE_PX
    detector.last_status = "idle"
    detector.last_primary_confidence = 0.0
    detector.last_max_confidence = 0.0
    return detector


def detect_single_negative_cutout_brick(detector, frame):
    """Return the one selected brick candidate and the full candidate list."""
    if detector is None or frame is None or getattr(frame, "size", 0) == 0:
        return None, []

    frame_h, frame_w = frame.shape[:2]
    detector.frame_w = int(frame_w)
    detector.frame_h = int(frame_h)

    feature_mask, _contour_mask = detector._build_hsv_masks(frame)

    if detector._uses_trapezoid_gate():
        candidates = detector._build_trapezoid_brick_candidates(
            feature_mask,
            crop_x=0,
            crop_y=0,
            crop_w=frame_w,
            crop_h=frame_h,
        )
        no_match_status = "trapezoid mismatch"
        if not candidates:
            detector._center_lock_prev_center = None
            detector.last_status = no_match_status
            detector.last_primary_confidence = 0.0
            return None, []
        primary = detector._select_center_brick(candidates, frame_w, frame_h)
    elif detector._uses_shape_match_gate():
        candidates = detector._build_shape_match_candidates(
            feature_mask,
            crop_x=0,
            crop_y=0,
            crop_w=frame_w,
            crop_h=frame_h,
        )
        no_match_status = "shape mismatch"
        if not candidates:
            detector._center_lock_prev_center = None
            detector.last_status = no_match_status
            detector.last_primary_confidence = 0.0
            return None, []
        filtered = detector._filter_candidates_to_center_y_row(candidates, frame_w, frame_h)
        if filtered:
            candidates = filtered
        primary = detector._select_center_brick(candidates, frame_w, frame_h)
    else:
        candidates = detector._build_negative_cutout_pair_candidates(
            feature_mask,
            crop_x=0,
            crop_y=0,
            crop_w=frame_w,
            crop_h=frame_h,
        )
        no_match_status = "inner triangles mismatch"
        if not candidates:
            detector._center_lock_prev_center = None
            detector.last_status = no_match_status
            detector.last_primary_confidence = 0.0
            return None, []
        filtered = detector._filter_candidates_to_center_y_row(candidates, frame_w, frame_h)
        if filtered:
            candidates = filtered
        primary = detector._select_center_brick(candidates, frame_w, frame_h)
    if primary is None:
        detector.last_status = no_match_status
        detector.last_primary_confidence = 0.0
        return None, candidates

    detector.last_status = "target locked (shape)"
    detector.last_primary_confidence = 1.0
    return primary, candidates


def draw_single_negative_cutout_brick_highlight(detector, frame, candidate):
    """Draw one selected brick outline."""
    if detector is None or frame is None or not isinstance(candidate, dict):
        return frame

    bbox = candidate.get("bbox")
    if isinstance(bbox, tuple) and len(bbox) >= 4:
        x, y, w, h = [int(round(float(v))) for v in bbox[:4]]
        if w > 0 and h > 0:
            cv2.rectangle(
                frame,
                (max(0, x), max(0, y)),
                (max(0, x + w), max(0, y + h)),
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
    detector._draw_primary_face_outline(frame, candidate)
    anchor_x, anchor_y = detector._candidate_selection_point(candidate)
    cv2.drawMarker(
        frame,
        (int(round(anchor_x)), int(round(anchor_y))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_brick_with_id(detector, frame, candidate, brick_id, *, center=None):
    """Stamp the numeric ID (0 = topmost) onto each detected brick."""
    if center is not None:
        cx, cy = center
    else:
        cx = candidate.get("center_x", 0)
        cy = candidate.get("center_y", 0)
    label = str(brick_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    lx = max(0, int(round(float(cx))) - tw // 2)
    ly = max(th + 4, int(round(float(cy))) + th // 2)
    cv2.putText(frame, label, (lx, ly), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, (lx, ly), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return frame
