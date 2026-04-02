"""
YOLO-Based Brick Detector - Drop-in replacement for helper_brick_vision.py
==========================================================================

Uses a YOLOv8-nano model trained on real brick footage from Leia's OV2710 camera.
Works WITHOUT ArUco markers, handles cluttered backgrounds, and runs at 5-15 fps
on Jetson Orin Nano.

Same interface as the original BrickDetector:
    read() -> (found, angle, dist, offset_x, confidence, cam_height,
               brick_above, brick_below)

Usage:
    # In your runtime script, replace:
    #   from helper_brick_vision import BrickDetector
    # with:
    #   from helper_brick_detector_yolo import BrickDetector

Dependencies:
    pip install opencv-contrib-python numpy

    Note: ultralytics + PyTorch are only needed for training/export,
    not for runtime inference. This module uses OpenCV DNN with an
    ONNX-exported model, so no heavy ML frameworks are required.
"""

import cv2
import numpy as np
import math
import os
import logging
import time
import colorsys
from pathlib import Path

from helper_camera_sources import candidate_camera_sources, existing_camera_nodes

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
# ArUco detector uses focal=frame_width; YOLO bbox height maps to
# BRICK_HEIGHT_MM with ~10% bbox padding, giving ~0.9*frame_width.
# Calibrated against ArUco gate targets (dist≈98-102mm at scoop position).
FOCAL_PX_REF = 580.0
FOCAL_REF_WIDTH = 640.0

# YOLO model path (relative to this file) — scenario-2 runtime uses v4 only
MODEL_PATH = Path(__file__).resolve().parent / "brick_yolo_v4.onnx"
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
    "018F97",
    "005A5F",
    "007179",
    "049397",
    "00898F",
    "017E87",
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

# Cyan-target experiment: stabilize selection around centerpoint.
CENTER_LOCK_RADIUS_RATIO = 0.18
CENTER_SWITCH_MARGIN_PX = 18.0
CENTER_PARTIAL_PENALTY = 2.0
CENTER_AXIS_WEIGHT_X = 1.0
CENTER_AXIS_WEIGHT_Y = 1.0
PARTIAL_EDGE_MARGIN_PX = 1
PARTIAL_COLOR_BGR = (0, 165, 255)
PARTIAL_LABEL_TOP = "TOP HALF"
PARTIAL_LABEL_BOTTOM = "BOTTOM HALF"
PARTIAL_LABEL_VERTICAL = "TOP/BOTTOM"
PARTIAL_LABEL_LEFT = "LEFT PARTIAL"
PARTIAL_LABEL_RIGHT = "RIGHT PARTIAL"
PARTIAL_LABEL_GENERIC = "PARTIAL"


class BrickDetector:
    """
    YOLO-based brick detector. Drop-in replacement for the original
    BrickDetector in helper_brick_vision.py.

    Provides the same read() interface expected by telemetry_brick.py
    and runtime training/orchestration scripts.

    Uses OpenCV DNN with an ONNX-exported YOLOv8-nano model — no
    PyTorch or ultralytics required at runtime.
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

        # Load ONNX model via OpenCV DNN
        mpath = Path(model_path) if model_path else MODEL_PATH
        if not mpath.exists():
            raise FileNotFoundError(
                f"YOLO ONNX model not found at {mpath}. "
                f"Place brick_yolo_v4.onnx next to this script."
            )
        self.net = cv2.dnn.readNetFromONNX(str(mpath))
        self.log.info("Loaded YOLO ONNX model from %s", mpath)
        self.model_path = str(mpath)
        self.model_name = mpath.name
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

        # Camera setup — probe explicit overrides and real /dev/video* nodes first.
        self.cap = None
        self.camera_source = None
        tried_sources = []
        for source in candidate_camera_sources():
            tried_sources.append(source)
            self.log.info("Attempting camera %s...", source)
            temp = cv2.VideoCapture(source)
            if temp.isOpened():
                self.log.info("Connected to camera %s (%s)", source,
                              temp.getBackendName())
                self.cap = temp
                self.camera_source = source
                break
            temp.release()

        if self.cap is None:
            existing_nodes = existing_camera_nodes()
            self.log.error(
                "No camera found. Tried sources: %s. Detected nodes: %s. Using dummy.",
                tried_sources,
                existing_nodes,
            )
            dummy_source = existing_nodes[0] if existing_nodes else 0
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
    # ONNX inference helpers
    # ------------------------------------------------------------------

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
        Run YOLO ONNX inference on a frame.

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

        # Forward pass
        self.net.setInput(blob)
        output = self.net.forward()  # shape: (1, 5, 8400)

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
        """Estimate distance from apparent brick size in the image.

        Uses inverse size relation: larger bbox height means closer brick,
        yielding a smaller distance value.
        """
        if bbox_height_px <= 0:
            return 999.0
        return (BRICK_HEIGHT_MM * self.focal_px) / float(bbox_height_px)

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
        (triggers fallback to current YOLO-only pipeline).
        """
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Check cyan coverage — if too little, not a cyan brick
        bbox_area = crop.shape[0] * crop.shape[1]
        cyan_pixels = cv2.countNonZero(mask)
        if cyan_pixels < bbox_area * HSV_CYAN_COVERAGE_MIN:
            return []

        # Morphological cleanup: open to remove noise, erode to separate
        # touching bricks, dilate to restore edges
        kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_open)

        kern_erode = cv2.getStructuringElement(
            cv2.MORPH_RECT, (HSV_ERODE_KERNEL, HSV_ERODE_KERNEL)
        )
        mask = cv2.erode(mask, kern_erode, iterations=self._hsv_erode_iterations)

        kern_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kern_dilate, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

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

            rect = cv2.minAreaRect(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)

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
            })

        return bricks

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

    def _candidate_center_score(self, candidate, frame_w, frame_h):
        frame_center_x = float(frame_w) / 2.0 + float(getattr(self, "camera_center_offset_px", 0.0) or 0.0)
        frame_center_y = float(frame_h) / 2.0
        dx = float(candidate.get("center_x", 0.0)) - frame_center_x
        dy = float(candidate.get("center_y", 0.0)) - frame_center_y
        weight_x = max(0.01, float(getattr(self, "_center_axis_weight_x", CENTER_AXIS_WEIGHT_X) or CENTER_AXIS_WEIGHT_X))
        weight_y = max(0.01, float(getattr(self, "_center_axis_weight_y", CENTER_AXIS_WEIGHT_Y) or CENTER_AXIS_WEIGHT_Y))
        score = math.hypot(dx * weight_x, dy * weight_y)
        if bool(candidate.get("partial", False)):
            partial_penalty = max(1.0, float(getattr(self, "_center_partial_penalty", CENTER_PARTIAL_PENALTY) or CENTER_PARTIAL_PENALTY))
            score *= partial_penalty
        return float(score)

    def _select_center_candidate_index(self, candidates, frame_w, frame_h):
        if not candidates:
            self._center_lock_prev_center = None
            return None

        scores = [self._candidate_center_score(cand, frame_w, frame_h) for cand in candidates]
        best_idx = min(range(len(candidates)), key=lambda i: scores[i])
        chosen_idx = int(best_idx)

        lock_enabled = bool(getattr(self, "_center_lock_enabled", True))
        prev_center = getattr(self, "_center_lock_prev_center", None)
        if lock_enabled and isinstance(prev_center, tuple) and len(prev_center) == 2:
            prev_x = float(prev_center[0])
            prev_y = float(prev_center[1])
            radius = self._center_lock_radius(frame_w, frame_h)
            near_idx = None
            near_dist = float("inf")
            for idx, cand in enumerate(candidates):
                dist = math.hypot(float(cand.get("center_x", 0.0)) - prev_x, float(cand.get("center_y", 0.0)) - prev_y)
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
        self._center_lock_prev_center = (
            float(chosen.get("center_x", 0.0)),
            float(chosen.get("center_y", 0.0)),
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

        # Try HSV segmentation on each YOLO box to find individual cyan bricks
        all_hsv_bricks = []
        hsv_used = False
        if self._hsv_enabled:
            for x1, y1, x2, y2, conf in bricks:
                individuals = self._segment_bricks_hsv(frame, x1, y1, x2, y2)
                all_hsv_bricks.extend(individuals)
            if all_hsv_bricks:
                hsv_used = True

        if hsv_used:
            # HSV path: center-most brick, contour-based angle, individual distance
            primary = self._select_center_brick(all_hsv_bricks, w_frame, h_frame)
            self.last_status = "target locked (HSV)"

            # Use YOLO confidence from the box that contained this brick
            # (find which YOLO box contains the primary brick center)
            yolo_conf = bricks[0][4]
            for x1, y1, x2, y2, conf in bricks:
                if (x1 <= primary["center_x"] <= x2 and
                        y1 <= primary["center_y"] <= y2):
                    yolo_conf = conf
                    break
            self.last_primary_confidence = float(yolo_conf)
            conf_pct = float(yolo_conf) * 100.0

            # Angle from contour with jump clamping to prevent 0/90 flips
            raw_angle = self._refine_angle_for_primary(frame, primary)
            angle = self._smooth_angle(raw_angle, self._prev_angle)
            self._prev_angle = angle

            # Distance from individual brick bbox height
            _, _, _, ind_bh = primary["bbox"]
            eff_h = self._effective_distance_height_px(ind_bh, primary.get("partial_kind"))
            raw_dist = self._estimate_distance(eff_h)
            dist = self._smooth(raw_dist, self._prev_dist)
            self._prev_dist = dist

            # Horizontal offset from camera center to brick center in mm.
            raw_offset_x = self._estimate_offset_x_mm(primary["center_x"], dist)
            offset_x = self._smooth(raw_offset_x, self._prev_offset)
            self._prev_offset = offset_x

            raw_cam_height = self._estimate_cam_height(primary["center_y"], dist)
            cam_height = self._smooth(raw_cam_height, self._prev_offset_y)
            self._prev_offset_y = cam_height

            # Stack flags from individual positions
            brick_above, brick_below = self._stack_flags_from_individuals(
                primary, all_hsv_bricks
            )
            partial_bricks = [
                {
                    "partial": bool(b.get("partial")),
                    "label": b.get("partial_label"),
                }
                for b in all_hsv_bricks
                if isinstance(b, dict) and bool(b.get("partial"))
            ]
            self._set_partial_state(
                partial_bricks,
                primary_partial_kind=(primary.get("partial_kind") if isinstance(primary, dict) else None),
                primary_partial_label=(primary.get("partial_label") if isinstance(primary, dict) and bool(primary.get("partial")) else None),
            )

            if self.debug:
                self._draw_debug_hsv(
                    frame, bricks, all_hsv_bricks, primary,
                    angle, dist, offset_x, yolo_conf
                )

            return (True, angle, dist, offset_x, conf_pct, cam_height,
                    brick_above, brick_below)

        # Fallback path: require canonical full/top/bottom shape match.
        shape_gate_enabled = bool(getattr(self, "_face_shape_templates", None))
        matched_boxes = []
        matched_partials = []
        if shape_gate_enabled:
            for bx1, by1, bx2, by2, bconf in bricks:
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
            self.last_status = "shape mismatch"
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

        raw_angle = self._estimate_angle(frame, x1, y1, x2, y2)
        angle = self._smooth_angle(raw_angle, self._prev_angle)
        self._prev_angle = angle

        primary_partial_info = (
            matched_partials[primary_idx]
            if 0 <= int(primary_idx) < len(matched_partials)
            else {}
        )

        bbox_h = y2 - y1
        eff_h = self._effective_distance_height_px(bbox_h, primary_partial_info.get("kind"))
        raw_dist = self._estimate_distance(eff_h)
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

        return self._process_bricks(frame, bricks)

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
        return self._process_bricks(frame.copy(), bricks)

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

                if shape_profile == "full":
                    face_cutouts_model = getattr(self, "_face_cutouts_model", None)
                    if isinstance(face_cutouts_model, list):
                        for cutout_points in face_cutouts_model:
                            cutout_proj = self._project_model_points_to_rect(cutout_points, points)
                            if isinstance(cutout_proj, np.ndarray) and len(cutout_proj) >= 3:
                                cutout_draw = np.round(cutout_proj).astype(np.int32).reshape(-1, 1, 2)
                                cv2.polylines(frame, [cutout_draw], True, (220, 245, 255), 1, cv2.LINE_AA)
                if shape_profile == "full" and isinstance(self._face_lines_model, list) and self._face_lines_model:
                    for p1, p2 in self._face_lines_model:
                        line_points = np.asarray([p1, p2], dtype=np.float32)
                        line_proj = self._project_model_points_to_rect(line_points, points)
                        if isinstance(line_proj, np.ndarray) and len(line_proj) == 2:
                            a = tuple(np.intp(np.round(line_proj[0])))
                            b = tuple(np.intp(np.round(line_proj[1])))
                            cv2.line(frame, a, b, (220, 245, 255), 1, cv2.LINE_AA)
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

    def _draw_debug_hsv(self, frame, yolo_bricks, hsv_bricks, primary,
                        angle, dist, offset_x, conf):
        """Draw enhanced debug visualization for HSV-segmented bricks."""
        # Only highlight the selected brick nearest the crosshairs.
        if primary:
            refined_outline = self._refine_primary_contour(frame, primary)
            if refined_outline is not None:
                primary = dict(primary)
                primary["debug_outline_contour"] = refined_outline
            self._draw_primary_face_outline(frame, primary)
            pcx, pcy = primary["center_x"], primary["center_y"]
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
        bbox_h = y2 - y1
        if bbox_h <= 0:
            return None

        new_focal = (known_distance_mm * bbox_h) / BRICK_HEIGHT_MM
        if new_focal < 1.0:
            self.log.error("calibrate_focal: computed focal_px=%.1f is too small "
                           "(dist=%.1f, bbox_h=%d)", new_focal, known_distance_mm, bbox_h)
            return None
        self.focal_px = new_focal
        self.log.info("calibrate_focal: bbox_h=%d, dist=%.1fmm -> focal_px=%.1f",
                      bbox_h, known_distance_mm, self.focal_px)
        return self.focal_px

    def release(self):
        """Release camera resources."""
        if getattr(self, "cap", None):
            self.cap.release()

    def close(self):
        """Release camera resources."""
        self.release()

    def __del__(self):
        self.release()
