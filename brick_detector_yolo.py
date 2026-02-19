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

# YOLO model path (relative to this file)
MODEL_PATH_V3 = Path(__file__).resolve().parent / "brick_yolo_v3.onnx"
MODEL_PATH_V2 = Path(__file__).resolve().parent / "brick_yolo_v2.onnx"
MODEL_PATH = MODEL_PATH_V3 if MODEL_PATH_V3.exists() else MODEL_PATH_V2

# Detection confidence threshold
CONF_THRESHOLD = 0.15

# NMS IoU threshold
NMS_THRESHOLD = 0.45

# YOLO input size (model was exported at 640x640)
YOLO_INPUT_SIZE = 640


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

        # Temporal smoothing
        self._prev_angle = None
        self._prev_dist = None
        self._prev_offset = None
        self._smooth_alpha = 0.3  # EMA weight for new values (lower = smoother)

    def set_runtime_tuning(self, confidence=None, smoothing_alpha=None,
                           nms_threshold=None, focal_px=None):
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
        return {
            "conf_threshold": float(self.conf_threshold),
            "smooth_alpha": float(self._smooth_alpha),
            "nms_threshold": float(self.nms_threshold),
            "focal_px": float(self.focal_px),
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
    # Main interface
    # ------------------------------------------------------------------

    def read(self):
        """
        Capture a frame and detect bricks.

        Returns:
            tuple: (found, angle, dist, offset_x, confidence,
                    cam_height, brick_above, brick_below)

            found (bool): True if a brick was detected
            angle (float): Brick rotation angle in degrees
            dist (float): Estimated distance to brick in mm
            offset_x (float): Horizontal offset from camera center in mm
            confidence (float): Detection confidence (0-100)
            cam_height (float): Estimated camera height (0 if unknown)
            brick_above (bool): Whether a brick is detected above current
            brick_below (bool): Whether a brick is detected below current
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

        # Run YOLO ONNX inference
        bricks = self._detect(frame)

        if not bricks:
            self._prev_angle = None
            self._prev_dist = None
            self._prev_offset = None
            self.last_status = "searching"
            self.last_primary_confidence = 0.0
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        # Pick the highest-confidence brick as primary target
        bricks.sort(key=lambda b: b[4], reverse=True)
        x1, y1, x2, y2, conf = bricks[0]
        self.last_status = "target locked"
        self.last_primary_confidence = float(conf)
        conf_pct = float(conf) * 100.0

        # Angle estimation
        raw_angle = self._estimate_angle(frame, x1, y1, x2, y2)
        angle = self._smooth(raw_angle, self._prev_angle)
        self._prev_angle = angle

        # Distance estimation
        bbox_h = y2 - y1
        raw_dist = self._estimate_distance(bbox_h)
        dist = self._smooth(raw_dist, self._prev_dist)
        self._prev_dist = dist

        # Horizontal offset from camera center (in mm)
        brick_center_x = (x1 + x2) / 2.0
        frame_center_x = w_frame / 2.0 + self.camera_center_offset_px
        offset_px = brick_center_x - frame_center_x
        raw_offset_x = (offset_px / self.focal_px) * dist
        offset_x = self._smooth(raw_offset_x, self._prev_offset)
        self._prev_offset = offset_x

        # Check for bricks above/below the primary target
        primary_cy = (y1 + y2) / 2.0
        brick_above = False
        brick_below = False
        for bx1, by1, bx2, by2, bconf in bricks[1:]:
            other_cy = (by1 + by2) / 2.0
            if other_cy < primary_cy - bbox_h * 0.5:
                brick_above = True
            elif other_cy > primary_cy + bbox_h * 0.5:
                brick_below = True

        # Camera height estimate (0 = unknown)
        cam_height = 0.0

        # Debug visualization
        if self.debug:
            self._draw_debug(frame, bricks, angle, dist, offset_x, conf)

        return (True, angle, dist, offset_x, conf_pct, cam_height,
                brick_above, brick_below)

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

        bricks.sort(key=lambda b: b[4], reverse=True)
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
            debug_frame = frame.copy()
            self._draw_debug(debug_frame, bricks, angle, dist, offset_x, conf)

        return (True, angle, dist, offset_x, conf_pct, cam_height,
                brick_above, brick_below)

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
            rad = math.radians(angle if i == 0 else
                               self._estimate_angle(frame, x1, y1, x2, y2))
            ex = int(cx + length * math.cos(rad))
            ey = int(cy - length * math.sin(rad))
            cv2.line(frame, (cx, cy), (ex, ey), (0, 0, 255), 2)

            label = f"brick {c:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD
        cv2.putText(frame, f"Angle: {angle:.1f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Dist: {dist:.0f} mm", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Offset: {offset_x:.1f} mm", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {conf:.0%}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
