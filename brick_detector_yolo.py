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
    # In autobuild.py, replace:
    #   from helper_brick_vision import BrickDetector
    # with:
    #   from brick_detector_yolo import BrickDetector

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
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Known brick dimensions (mm) — from world_model_brick.json
BRICK_WIDTH_MM = 64.0
BRICK_HEIGHT_MM = 48.0
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

# YOLO model path (relative to this file) — prefers newest available
MODEL_PATH_V4 = Path(__file__).resolve().parent / "brick_yolo_v4.onnx"
MODEL_PATH_V3 = Path(__file__).resolve().parent / "brick_yolo_v3.onnx"
MODEL_PATH_V2 = Path(__file__).resolve().parent / "brick_yolo_v2.onnx"
MODEL_PATH = (MODEL_PATH_V4 if MODEL_PATH_V4.exists() else
              MODEL_PATH_V3 if MODEL_PATH_V3.exists() else MODEL_PATH_V2)

# Detection confidence threshold
CONF_THRESHOLD = 0.15

# NMS IoU threshold
NMS_THRESHOLD = 0.45

# YOLO input size (model was exported at 640x640)
YOLO_INPUT_SIZE = 640

# ---------------------------------------------------------------------------
# HSV segmentation for individual cyan brick detection within YOLO boxes
# ---------------------------------------------------------------------------
CYAN_HSV_LOWER = np.array([75, 60, 50])
CYAN_HSV_UPPER = np.array([115, 255, 255])
HSV_ERODE_KERNEL = 5
HSV_ERODE_ITERATIONS = 2
HSV_MIN_AREA_RATIO = 0.05       # Min 5% of YOLO bbox area = real brick
HSV_CYAN_COVERAGE_MIN = 0.08    # Need 8% cyan coverage to engage HSV path
STACK_X_OVERLAP_RATIO = 0.6     # Same-column tolerance for above/below
STACK_Y_GAP_RATIO = 0.35        # Vertical gap threshold for above/below


class BrickDetector:
    """
    YOLO-based brick detector. Drop-in replacement for the original
    BrickDetector in helper_brick_vision.py.

    Provides the same read() interface expected by telemetry_brick.py
    and autobuild.py.

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
                f"Place brick_yolo_v3.onnx (or brick_yolo_v2.onnx) next to this script."
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
        # Keep debug view readable when low confidence rescue profiles are active.
        self.debug_show_all_candidates = False
        self.debug_max_boxes = 1

        # Camera setup — try indices 0-3
        self.cap = None
        for idx in range(4):
            self.log.info("Attempting camera %d...", idx)
            temp = cv2.VideoCapture(idx)
            if temp.isOpened():
                self.log.info("Connected to camera %d (%s)", idx,
                              temp.getBackendName())
                self.cap = temp
                break
            temp.release()

        if self.cap is None:
            self.log.error("No camera found (0-3). Using dummy.")
            self.cap = cv2.VideoCapture(0)

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

        # Temporal smoothing
        self._prev_angle = None
        self._prev_dist = None
        self._prev_offset = None
        self._smooth_alpha = 0.3  # EMA weight for new values (lower = smoother)

    def set_runtime_tuning(self, confidence=None, smoothing_alpha=None,
                           nms_threshold=None, focal_px=None,
                           hsv_lower=None, hsv_upper=None,
                           hsv_enabled=None, hsv_erode_iterations=None):
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
        return {
            "conf_threshold": float(self.conf_threshold),
            "smooth_alpha": float(self._smooth_alpha),
            "nms_threshold": float(self.nms_threshold),
            "focal_px": float(self.focal_px),
            "hsv_lower": self._hsv_lower.tolist(),
            "hsv_upper": self._hsv_upper.tolist(),
            "hsv_enabled": self._hsv_enabled,
            "hsv_erode_iterations": self._hsv_erode_iterations,
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
        """Estimate distance to brick using pinhole camera model."""
        if bbox_height_px <= 0:
            return 999.0
        dist_mm = (BRICK_HEIGHT_MM * self.focal_px) / bbox_height_px
        return dist_mm

    def _smooth(self, new_val, prev_val):
        """Exponential moving average for temporal smoothing."""
        if prev_val is None:
            return new_val
        return self._smooth_alpha * new_val + (1 - self._smooth_alpha) * prev_val

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

            # Check if contour touches crop edge (partial brick)
            partial = (bx <= 1 or by <= 1 or
                       bx + bw >= crop.shape[1] - 1 or
                       by + bh >= crop.shape[0] - 1)

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
                "partial": partial,
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
        bx, by, bw, bh = primary["bbox"]

        # Pad the bbox slightly to capture edge pixels
        pad = max(5, int(max(bw, bh) * 0.15))
        rx1 = max(0, bx - pad)
        ry1 = max(0, by - pad)
        rx2 = min(frame.shape[1], bx + bw + pad)
        ry2 = min(frame.shape[0], by + bh + pad)

        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return self._estimate_angle_from_rect(primary["rect"])

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Light cleanup: open to remove noise, then gentle erosion (3x3, 1 iter)
        # to separate touching bricks without distorting shape.
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.erode(mask, kern, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return self._estimate_angle_from_rect(primary["rect"])

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

        if best_cnt is None or len(best_cnt) < 5:
            return self._estimate_angle_from_rect(primary["rect"])

        rect = cv2.minAreaRect(best_cnt)
        return self._estimate_angle_from_rect(rect)

    def _select_center_brick(self, individual_bricks, frame_w, frame_h):
        """
        Pick the brick closest to frame center. Partial bricks get 2x
        distance penalty so we prefer fully-visible bricks.
        """
        if not individual_bricks:
            return None

        center_x = frame_w / 2.0 + self.camera_center_offset_px
        center_y = frame_h / 2.0

        best = None
        best_score = float("inf")
        for b in individual_bricks:
            dx = b["center_x"] - center_x
            dy = b["center_y"] - center_y
            score = math.sqrt(dx * dx + dy * dy)
            if b["partial"]:
                score *= 2.0
            if score < best_score:
                best_score = score
                best = b
        return best

    def _stack_flags_from_individuals(self, primary, all_bricks):
        """
        Determine above/below flags using individual brick positions.
        Reuses the proven pattern from OLD/vision_brick_color.py:252-271.
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

            # Angle from un-eroded contour (erosion distorts near-square shapes)
            raw_angle = self._refine_angle_for_primary(frame, primary)
            angle = self._smooth(raw_angle, self._prev_angle)
            self._prev_angle = angle

            # Distance from individual brick bbox height
            _, _, _, ind_bh = primary["bbox"]
            raw_dist = self._estimate_distance(ind_bh)
            dist = self._smooth(raw_dist, self._prev_dist)
            self._prev_dist = dist

            # Horizontal offset
            frame_center_x = w_frame / 2.0 + self.camera_center_offset_px
            offset_px = primary["center_x"] - frame_center_x
            raw_offset_x = (offset_px / self.focal_px) * dist
            offset_x = self._smooth(raw_offset_x, self._prev_offset)
            self._prev_offset = offset_x

            # Stack flags from individual positions
            brick_above, brick_below = self._stack_flags_from_individuals(
                primary, all_hsv_bricks
            )

            cam_height = 0.0

            if self.debug:
                self._draw_debug_hsv(
                    frame, bricks, all_hsv_bricks, primary,
                    angle, dist, offset_x, yolo_conf
                )

            return (True, angle, dist, offset_x, conf_pct, cam_height,
                    brick_above, brick_below)

        # Fallback path: current YOLO-only pipeline (gray bricks, HSV disabled, etc.)
        x1, y1, x2, y2, conf = bricks[0]
        self.last_status = "target locked"
        self.last_primary_confidence = float(conf)
        conf_pct = float(conf) * 100.0

        raw_angle = self._estimate_angle(frame, x1, y1, x2, y2)
        angle = self._smooth(raw_angle, self._prev_angle)
        self._prev_angle = angle

        bbox_h = y2 - y1
        raw_dist = self._estimate_distance(bbox_h)
        dist = self._smooth(raw_dist, self._prev_dist)
        self._prev_dist = dist

        brick_center_x = (x1 + x2) / 2.0
        frame_center_x = w_frame / 2.0 + self.camera_center_offset_px
        offset_px = brick_center_x - frame_center_x
        raw_offset_x = (offset_px / self.focal_px) * dist
        offset_x = self._smooth(raw_offset_x, self._prev_offset)
        self._prev_offset = offset_x

        primary_cy = (y1 + y2) / 2.0
        brick_above = False
        brick_below = False
        for bx1, by1, bx2, by2, bconf in bricks[1:]:
            other_cy = (by1 + by2) / 2.0
            if other_cy < primary_cy - bbox_h * 0.5:
                brick_above = True
            elif other_cy > primary_cy + bbox_h * 0.5:
                brick_below = True

        cam_height = 0.0

        if self.debug:
            self._draw_debug(frame, bricks, angle, dist, offset_x, conf)

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
            self.log.warning("Frame capture failed")
            self.last_status = "camera read failed"
            self.last_primary_confidence = 0.0
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

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
            self.last_status = "searching"
            self.last_primary_confidence = 0.0
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
            self.last_status = "searching"
            self.last_primary_confidence = 0.0
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        # Use a copy so debug drawing doesn't mutate the caller's frame
        return self._process_bricks(frame.copy(), bricks)

    def _draw_debug(self, frame, bricks, angle, dist, offset_x, conf):
        """Draw detection visualization on frame."""
        if self.debug_show_all_candidates:
            draw_boxes = list(bricks)
        else:
            max_boxes = max(1, int(getattr(self, "debug_max_boxes", 1) or 1))
            draw_boxes = list(bricks[:max_boxes])

        for i, (x1, y1, x2, y2, c) in enumerate(draw_boxes):
            color = (0, 255, 0) if i == 0 else (0, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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

    def _draw_debug_hsv(self, frame, yolo_bricks, hsv_bricks, primary,
                        angle, dist, offset_x, conf):
        """Draw enhanced debug visualization for HSV-segmented bricks."""
        # Gray thin-line YOLO boxes (reference)
        for x1, y1, x2, y2, c in yolo_bricks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Yellow contours for all HSV-segmented bricks
        for b in hsv_bricks:
            cv2.drawContours(frame, [b["contour"]], -1, (0, 255, 255), 1)
            # Magenta rotated rectangle (minAreaRect)
            box_pts = cv2.boxPoints(b["rect"])
            box_pts = np.intp(box_pts)
            cv2.drawContours(frame, [box_pts], 0, (255, 0, 255), 1)

        # Green thick contour + crosshair on selected primary brick
        if primary:
            cv2.drawContours(frame, [primary["contour"]], -1, (0, 255, 0), 2)
            pcx, pcy = primary["center_x"], primary["center_y"]
            cv2.drawMarker(frame, (pcx, pcy), (0, 255, 0),
                           cv2.MARKER_CROSS, 15, 2)

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
