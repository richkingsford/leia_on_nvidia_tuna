"""Native OAK-D brick detector using green color and brick shape gates.

This module intentionally does not load TensorRT/YOLO.  It wraps the existing
camera-free color/shape detector with the same runtime interface used by
follow-the-brick:

    read() -> (found, angle, dist, offset_x, confidence, cam_height,
               brick_above, brick_below)
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from typing import Optional

import cv2
import numpy as np

from helper_camera_sources import (
    CameraSource,
    DEPTHAI_CAMERA_SOURCE,
    candidate_camera_sources,
    existing_camera_nodes,
    open_opencv_camera_source,
)
from helper_brick_detector_yolo import (
    BRICK_HEIGHT_MM,
    BRICK_WIDTH_MM,
    COLOR_ONLY_CONF_PCT,
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
    CYAN_HSV_WIDE_LOWER,
    CYAN_HSV_WIDE_UPPER,
    DEFAULT_FRAME_H,
    DEFAULT_FRAME_W,
    FOCAL_PX_REF,
    FOCAL_REF_WIDTH,
    NMS_THRESHOLD,
    YOLO_INPUT_SIZE,
    build_negative_cutout_shape_detector,
    detect_single_negative_cutout_brick,
)


NATIVE_RECT_MIN_FILL_RATIO = 0.22
NATIVE_RECT_MIN_SOLIDITY = 0.30
NATIVE_RECT_MAX_BBOX_ASPECT = 3.25
NATIVE_RECT_MAX_BBOX_WIDTH_RATIO = 0.55
NATIVE_RECT_MAX_MIN_AREA_ASPECT = 4.25
NATIVE_RECT_STRIP_ASPECT = 2.55
NATIVE_RECT_STRIP_MAX_HEIGHT_RATIO = 0.12
NATIVE_RECT_COLUMN_MIN_WIDTH_PX = 6
NATIVE_RECT_COLUMN_MAX_COUNT_RATIO = 0.28
NATIVE_RECT_ROW_MAX_COUNT_RATIO = 0.12
NATIVE_RECT_MIN_EDGE_ROWS = 5
NATIVE_RECT_MIN_ACTIVE_ROW_RATIO = 0.14
NATIVE_RECT_MIN_EDGE_STRAIGHTNESS = 0.0
NATIVE_RECT_MIN_WIDTH_COHERENCE = 0.0
NATIVE_RECT_MAX_DIST_STEP_MM = 8.0
# Close-range stack-width distance: at close range the raw width detection jitters
# more than real per-frame motion, so the median/outlier filter must SMOOTH the
# jitter rather than reset on it. Longer history + wider outlier band + far less
# trigger-happy resets give a stable published distance for Step 2 / picking.
NATIVE_RECT_DIST_HISTORY_LEN = 12
NATIVE_RECT_DIST_MIN_INLIERS = 3
NATIVE_RECT_DIST_STABLE_SPREAD_MM = 9.0
NATIVE_RECT_DIST_OUTLIER_BAND_MM = 18.0
NATIVE_RECT_DIST_MAX_PUBLISH_STEP_MM = 7.0
NATIVE_RECT_DIST_RESET_CENTER_STEP_PX = 130.0
NATIVE_RECT_DIST_RESET_WIDTH_RATIO = 0.55
# Temporal gate on the RAW width-distance before it enters the median filter.
# Close-range stack-width segmentation flickers between interpretations while the
# brick is stationary; reject raw readings that jump beyond GATE_MM from the locked
# estimate (hold last good), but re-lock if RELOCK_FRAMES consecutive readings agree
# within RELOCK_SPREAD (a genuine move/approach), not random multimodal noise.
NATIVE_RECT_DIST_GATE_MM = 35.0
NATIVE_RECT_DIST_RELOCK_FRAMES = 4
NATIVE_RECT_DIST_RELOCK_SPREAD_MM = 22.0
# Confidence gate (RELATIVE): the mis-segmentation mode reads notably lower than the
# correct reading *at the same pose* (e.g. 85 vs 99). A fixed absolute threshold breaks
# at poses where the brick is legitimately seen at lower confidence (e.g. ~88 when low
# in frame) — it would freeze the published distance. Instead reject frames whose
# confidence is CONF_MARGIN below the recent-max confidence, so it rejects the bad mode
# without freezing a consistent lower-confidence pose. Accept anyway after MAX_CONF_HOLD.
NATIVE_RECT_DIST_CONF_MARGIN_PCT = 7.0
NATIVE_RECT_DIST_CONF_WINDOW = 12
NATIVE_RECT_DIST_MAX_CONF_HOLD = 8
# Close-range green-edge fallback: at pick range the stack fills/overflows the frame
# (often cut off at top), so the full-rectangle gate fails — but the LEFT/RIGHT green
# edges stay crisp and stable (±1.5px). When the normal path finds no rectangle, derive
# distance from the green stack's horizontal extent. Only trusted when both side edges
# sit inside the frame (not cut off), coverage is real, and width implies close range.
GREEN_EDGE_CLOSE_RANGE_ENABLED = True
GREEN_EDGE_MIN_WIDTH_PX = 60
GREEN_EDGE_FRAME_MARGIN_PX = 4
GREEN_EDGE_MIN_COVERAGE = 0.05
GREEN_EDGE_COL_ACTIVE_FRAC = 0.15
GREEN_EDGE_CONF_PCT = 95.0
GREEN_EDGE_MAX_DIST_MM = 220.0
GREEN_EDGE_TOP_STRIP_ENABLED = True
GREEN_EDGE_TOP_STRIP_MAX_Y_RATIO = 0.18
GREEN_EDGE_TOP_STRIP_MIN_HEIGHT_PX = 12
GREEN_EDGE_TOP_STRIP_MIN_WIDTH_PX = 120
GREEN_EDGE_TOP_STRIP_CONF_PCT = 88.0
GREEN_EDGE_TOP_STRIP_ROW_ACTIVE_FRAC = 0.08
GREEN_EDGE_TOP_STRIP_COL_ACTIVE_FRAC = 0.35
GREEN_EDGE_PAINTED_COLUMN_ENABLED = True
GREEN_EDGE_PAINTED_COLUMN_MIN_HEIGHT_FRAC = 0.30
GREEN_EDGE_PAINTED_COLUMN_MIN_COVERAGE = 0.035
GREEN_EDGE_PAINTED_COLUMN_COL_ACTIVE_FRAC = 0.18
GREEN_EDGE_PAINTED_COLUMN_CONF_PCT = 92.0
GREEN_EDGE_PAINTED_COLUMN_MAX_DIST_MM = 320.0
GREEN_EDGE_PAINTED_COLUMN_MAX_RAW_CALIBRATION_MM = 130.0
GREEN_EDGE_PAINTED_CLOSE_CALIBRATION_ENABLED = True
GREEN_EDGE_PAINTED_CLOSE_CALIBRATION_POINTS = (
    (69.0, 90.0),
    (89.0, 100.0),
    (116.0, 150.0),
)
# Green-edge becomes the PRIMARY distance source (overriding the rectangle path) once
# the stack is wide enough to mean close range. The rectangle path returns unreliable
# (often garbage) distances at close range without failing cleanly, so we can't wait for
# it to return None. ~180px width corresponds to ~148mm; Step 1 mid-range (~227mm) is
# only ~117px wide, so it stays on the verified rectangle path.
GREEN_EDGE_PRIMARY_MIN_WIDTH_PX = 180
NATIVE_RECT_MAX_CENTER_STEP_PX = 140.0
NATIVE_RECT_MAX_WIDTH_RATIO_JUMP = 0.42
NATIVE_RECT_PREFERRED_LOCK_DIST_MM = 170.0
NATIVE_RECT_MAX_INITIAL_ABS_Y_MM = 130.0
NATIVE_RECT_MAX_INITIAL_ABS_X_MM = 420.0


def _calibrate_painted_close_green_edge_dist(raw_dist_mm: float) -> float:
    if not bool(GREEN_EDGE_PAINTED_CLOSE_CALIBRATION_ENABLED):
        return float(raw_dist_mm)
    points = sorted((float(x), float(y)) for x, y in GREEN_EDGE_PAINTED_CLOSE_CALIBRATION_POINTS)
    if len(points) < 2:
        return float(raw_dist_mm)
    raw = float(raw_dist_mm)
    if raw <= points[0][0]:
        left, right = points[0], points[1]
    elif raw >= points[-1][0]:
        left, right = points[-2], points[-1]
    else:
        left, right = points[0], points[1]
        for idx in range(len(points) - 1):
            a, b = points[idx], points[idx + 1]
            if raw <= b[0]:
                left, right = a, b
                break
    dx = float(right[0]) - float(left[0])
    if abs(dx) < 1e-6:
        return float(raw)
    fraction = (raw - float(left[0])) / dx
    return float(left[1]) + fraction * (float(right[1]) - float(left[1]))


class NativeOakBrickDetector:
    """OAK-D/OpenCV detector that confirms bricks by color and shape."""

    def __init__(
        self,
        debug: bool = True,
        *,
        width: int = DEFAULT_FRAME_W,
        height: int = DEFAULT_FRAME_H,
        camera_index: Optional[int] = None,
        focal_px: float | None = None,
    ) -> None:
        self.debug = bool(debug)
        self.log = logging.getLogger("BrickVisionNativeOAK")
        self.cap = None
        self.camera_index = None
        self.current_frame = None
        self.raw_frame = None
        self.frame_w = int(width)
        self.frame_h = int(height)
        self.camera_center_offset_px = 0.0
        self.inference_backend = "native_oak_color_shape"
        self.model_path = "native_oak_color_shape"
        self.model_name = "native_oak_color_shape"
        self.input_size = 0
        self.conf_threshold = 0.0
        self.nms_threshold = float(NMS_THRESHOLD)

        self._detector = build_negative_cutout_shape_detector(center_lock_enabled=True)
        self._detector.debug = bool(debug)
        self._detector.log = self.log
        self._detector._trust_detector_boxes = False
        self._detector._require_cyan_shape = True
        self._detector._center_lock_radius_px = 36.0
        self._detector._center_switch_margin_px = 12.0
        self._detector.conf_threshold = float(self.conf_threshold)
        self._detector.nms_threshold = float(self.nms_threshold)
        self._detector.input_size = int(YOLO_INPUT_SIZE)
        self._detector.focal_px = float(
            focal_px
            if focal_px is not None
            else FOCAL_PX_REF * (float(self.frame_w) / float(FOCAL_REF_WIDTH))
        )
        self._detector._smooth_alpha = 0.15
        self._detector._native_rect_max_dist_step_mm = float(NATIVE_RECT_MAX_DIST_STEP_MM)
        self._detector._depth_source_mode = "pinhole"
        self._detector._stereo_config_mode = "standard"
        self._detector.cap = None
        self._native_last_good_dist = None
        self._native_last_good_center = None
        self._native_last_good_width_px = None
        self._native_dist_history = deque(maxlen=int(NATIVE_RECT_DIST_HISTORY_LEN))
        self._native_last_stable_dist = None
        self._native_gate_anchor = None
        self._native_gate_pending = []
        self._native_conf_hold_count = 0
        self._native_conf_window = deque(maxlen=int(NATIVE_RECT_DIST_CONF_WINDOW))
        self._native_miss_count = 0
        self._reset_detector_tracking()
        self._clear_detection_metadata()
        self._camera_index_preference = camera_index

    def __getattr__(self, name):
        detector = self.__dict__.get("_detector")
        if detector is not None and hasattr(detector, name):
            return getattr(detector, name)
        raise AttributeError(name)

    def _reset_detector_tracking(self) -> None:
        self._detector._prev_angle = None
        self._detector._prev_dist = None
        self._detector._prev_offset = None
        self._detector._prev_offset_y = None
        self._detector._center_lock_prev_center = None
        self._reset_native_dist_filter()

    def _reset_native_dist_filter(self) -> None:
        history = getattr(self, "_native_dist_history", None)
        if history is not None:
            history.clear()
        self._native_last_stable_dist = None
        self._native_gate_anchor = None
        self._native_gate_pending = []
        self._native_conf_hold_count = 0
        window = getattr(self, "_native_conf_window", None)
        if window is not None:
            window.clear()

    def _clear_detection_metadata(self) -> None:
        detector = self._detector
        detector.last_raw_prediction_count = 0
        detector.last_candidate_count = 0
        detector.last_nms_count = 0
        detector.last_primary_confidence = 0.0
        detector.last_max_confidence = 0.0
        detector.last_status = "idle"
        self.last_status = detector.last_status
        detector.last_geometry_source = "pinhole_size"
        detector.last_suspected_far_brick = False
        detector.last_distance_display_text = None
        detector.last_bbox_w_px = None
        detector.last_bbox_h_px = None
        detector.last_bbox_eff_w_px = None
        detector.last_bbox_eff_h_px = None
        detector.last_bbox_width_dist = None
        detector.last_bbox_height_dist = None
        detector.last_bbox_calibrated_height_dist = None
        detector.last_bbox_dist = None
        detector.last_bbox_distance_source = None
        detector.last_raw_dist = None
        detector.last_final_dist = None
        detector.last_stable_dist_sample_count = 0
        detector.last_stable_dist_inlier_count = 0
        detector.last_stable_dist_spread_mm = None
        detector.last_stable_dist_unlimited_mm = None
        detector.last_stable_dist_published_mm = None
        detector.last_stable_dist_publish_limited = False
        detector.last_pre_depth_dist = None
        detector.last_depth_dist = None
        detector.last_depth_stats = {}
        detector.last_tri_span_px = None
        detector.last_tri_span_dist = None
        detector.last_partial_count = 0
        detector.last_partial_labels = []
        detector.last_primary_partial_kind = None
        detector.last_primary_partial_label = None

    def set_runtime_tuning(self, **kwargs):
        tuning = dict(kwargs)
        # Native color+shape detection has no YOLO confidence threshold.
        tuning.pop("confidence", None)
        self.conf_threshold = 0.0
        self._detector.conf_threshold = 0.0
        self._detector.set_runtime_tuning(**tuning)
        self._detector._trust_detector_boxes = False
        self._detector._require_cyan_shape = True
        self._sync_public_geometry()
        return self._detector.set_runtime_tuning()

    def _sync_public_geometry(self) -> None:
        self.frame_w = int(getattr(self._detector, "frame_w", self.frame_w))
        self.frame_h = int(getattr(self._detector, "frame_h", self.frame_h))
        self.camera_center_offset_px = float(
            getattr(self._detector, "camera_center_offset_px", self.camera_center_offset_px)
        )

    def _candidate_camera_sources(self, preferred_index: Optional[int]) -> list[CameraSource]:
        sources: list[CameraSource] = [DEPTHAI_CAMERA_SOURCE]
        for source in candidate_camera_sources(
                preferred_index,
                width=int(self.frame_w),
                height=int(self.frame_h),
                include_nvidia_pipelines=False,
        ):
            if source not in sources:
                sources.append(source)
        return sources

    def _open_camera(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        tried_sources: list[CameraSource] = []
        for source in self._candidate_camera_sources(self._camera_index_preference):
            tried_sources.append(source)
            cap = open_opencv_camera_source(
                source,
                cv2,
                width=int(self.frame_w),
                height=int(self.frame_h),
            )
            if cap is None or not cap.isOpened():
                continue
            self.cap = cap
            self.camera_index = source
            self._detector.cap = cap
            self._configure_camera_geometry(cap)
            self.log.info("Connected to camera %s (%s)", source, self._backend_name(cap))
            return True

        self.cap = None
        self.camera_index = None
        self._detector.cap = None
        self._log_camera_open_failure(tried_sources)
        return False

    def _backend_name(self, cap) -> str:
        try:
            return str(cap.getBackendName())
        except Exception:
            return "-"

    def _configure_camera_geometry(self, cap) -> None:
        try:
            width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
            height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            if width > 0:
                self.frame_w = width
                self._detector.frame_w = width
            if height > 0:
                self.frame_h = height
                self._detector.frame_h = height
        except Exception:
            pass

        intrinsics = getattr(cap, "camera_intrinsics", None)
        if isinstance(intrinsics, dict):
            try:
                self._detector._camera_fx_px = max(1.0, float(intrinsics.get("fx")))
                self._detector._camera_fy_px = max(1.0, float(intrinsics.get("fy")))
                self._detector._camera_cx_px = float(intrinsics.get("cx"))
                self._detector._camera_cy_px = float(intrinsics.get("cy"))
                self._detector.focal_px = self._detector._camera_fx_px
                return
            except (TypeError, ValueError):
                pass
        self._detector.focal_px = FOCAL_PX_REF * (
            float(max(1, self.frame_w)) / float(FOCAL_REF_WIDTH)
        )

    def _log_camera_open_failure(self, tried_sources: list[CameraSource]) -> None:
        self.log.error("Unable to open camera. Tried sources: %s", tried_sources)
        existing_nodes = existing_camera_nodes()
        if existing_nodes:
            self.log.error("Detected camera nodes: %s", existing_nodes)

    def read(self):
        if not self._open_camera():
            self._mark_not_found("camera unavailable")
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.release()
            self._mark_not_found("camera read failed")
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
        return self.read_frame(frame)

    def read_frame(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            self._mark_not_found("empty frame")
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
        self.raw_frame = frame.copy()
        self._detector.raw_frame = self.raw_frame
        self._detector.current_frame = frame
        self._sync_frame_shape(frame)

        primary, candidates = self._detect_color_rectangle_candidate(frame)
        self._detector.last_raw_prediction_count = int(len(candidates))
        self._detector.last_candidate_count = int(len(candidates))
        self._detector.last_nms_count = int(len(candidates))

        if primary is None:
            green_edge_result = self._green_edge_close_range_result(frame, require_close=False)
            if isinstance(green_edge_result, tuple) and len(green_edge_result) >= 1 and bool(green_edge_result[0]):
                return green_edge_result
            self._mark_not_found(str(getattr(self._detector, "last_status", "shape mismatch")))
            self.current_frame = frame.copy()
            self._detector.current_frame = self.current_frame
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        result = self._result_from_candidate(frame, primary, candidates)
        self._draw_debug_frame(frame, primary, candidates, result)
        return result

    def _detect_color_rectangle_candidate(self, frame):
        detector = self._detector
        loose_feature_mask, loose_contour_mask = detector._build_hsv_masks(frame)
        feature_mask, contour_mask = self._build_native_brick_masks(
            frame,
            loose_feature_mask,
            loose_contour_mask,
        )
        if feature_mask is None or contour_mask is None:
            return None, []

        contours, _ = cv2.findContours(
            contour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours is None:
            contours = []

        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1.0, float(frame_w * frame_h))
        min_area = max(180.0, frame_area * 0.0015)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue

            shrink = self._shrinkwrap_native_rect_bbox(
                feature_mask,
                bbox=(int(x), int(y), int(w), int(h)),
                frame_w=int(frame_w),
                frame_h=int(frame_h),
            )
            if shrink is None:
                continue
            x, y, w, h = shrink["bbox"]
            edge_metrics = shrink["edge_metrics"]
            if w < 8 or h < 8:
                continue

            roi_points = cv2.findNonZero(feature_mask[y:y + h, x:x + w])
            if roi_points is None:
                continue
            roi_contour = roi_points.reshape(-1, 1, 2)
            roi_contour[:, :, 0] += int(x)
            roi_contour[:, :, 1] += int(y)
            contour = roi_contour
            area = float(cv2.countNonZero(feature_mask[y:y + h, x:x + w]))
            if area < min_area:
                continue

            bbox_area = float(max(1, w * h))
            fill_ratio = area / bbox_area
            rect = cv2.minAreaRect(contour)
            rw, rh = float(rect[1][0]), float(rect[1][1])
            if rw <= 0.0 or rh <= 0.0:
                continue
            aspect = float(max(rw, rh)) / float(max(1e-6, min(rw, rh)))
            ok, geometry = self._native_rect_geometry_ok(
                contour,
                bbox=(int(x), int(y), int(w), int(h)),
                frame_w=int(frame_w),
                frame_h=int(frame_h),
                area=float(area),
                fill_ratio=float(fill_ratio),
                min_area_aspect=float(aspect),
                edge_metrics=edge_metrics,
            )
            if not ok:
                continue

            roi = feature_mask[y:y + h, x:x + w]
            coverage = float(cv2.countNonZero(roi)) / bbox_area if roi.size else 0.0
            coverage_floor = max(
                0.06,
                float(getattr(detector, "_hsv_cyan_coverage_min", 0.08)) * 0.45,
            )
            if coverage < coverage_floor:
                continue

            cx = float(x) + (float(w) * 0.5)
            cy = float(y) + (float(h) * 0.5)
            try:
                proxy_dist = detector._estimate_distance_from_width(max(1.0, float(w)))
            except Exception:
                proxy_dist = None
            proxy_x = proxy_y = None
            if proxy_dist is not None:
                try:
                    proxy_x = detector._estimate_offset_x_mm(cx, proxy_dist)
                    proxy_y = detector._estimate_cam_height(cy, proxy_dist)
                except Exception:
                    proxy_x = proxy_y = None
            try:
                if proxy_y is not None and abs(float(proxy_y)) > float(NATIVE_RECT_MAX_INITIAL_ABS_Y_MM):
                    continue
                if proxy_x is not None and abs(float(proxy_x)) > float(NATIVE_RECT_MAX_INITIAL_ABS_X_MM):
                    continue
            except (TypeError, ValueError):
                pass
            area_score = min(10.0, area / 1800.0)
            edge_score = float(edge_metrics.get("edge_score", 0.0))
            width_coherence = float(edge_metrics.get("width_coherence", 0.0))
            confidence = max(
                75.0,
                min(
                    99.0,
                    56.0
                    + (coverage * 24.0)
                    + (fill_ratio * 12.0)
                    + (edge_score * 18.0)
                    + (width_coherence * 8.0)
                    + area_score,
                ),
            )
            candidates.append(
                {
                    "center_x": cx,
                    "center_y": cy,
                    "contour": contour,
                    "rect": rect,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "area": float(area),
                    "partial": False,
                    "partial_kind": None,
                    "partial_label": None,
                    "partial_edges": {},
                    "shape_profile": "native_rect",
                    "shape_match_score": None,
                    "negative_cutout_polygons": [],
                    "selection_anchor_x": cx,
                    "selection_anchor_y": cy,
                    "from_color_detection": True,
                    "native_rect_confidence_pct": float(confidence),
                    "confidence_pct": float(confidence),
                    "cyan_coverage": float(coverage),
                    "fill_ratio": float(fill_ratio),
                    "bbox_aspect": float(geometry["bbox_aspect"]),
                    "solidity": float(geometry["solidity"]),
                    "edge_straightness": float(edge_metrics["edge_straightness"]),
                    "width_coherence": float(width_coherence),
                    "active_row_ratio": float(edge_metrics["active_row_ratio"]),
                    "native_width_dist_mm": None if proxy_dist is None else float(proxy_dist),
                    "native_proxy_x_mm": None if proxy_x is None else float(proxy_x),
                    "native_proxy_y_mm": None if proxy_y is None else float(proxy_y),
                }
            )

        if not candidates:
            detector.last_status = "native color rectangle mismatch"
            self.last_status = detector.last_status
            return None, []

        candidates = detector._enforce_non_overlapping_candidate_bboxes(candidates)
        selected = self._select_native_rect_candidate(candidates, frame_w, frame_h)
        if selected is None:
            detector.last_status = "native color rectangle mismatch"
            self.last_status = detector.last_status
            return None, candidates
        detector.last_status = "target locked (native color+rect)"
        self.last_status = detector.last_status
        return selected, candidates

    def _build_native_brick_masks(self, frame, loose_feature_mask, loose_contour_mask):
        """Build strict native masks from saturated brick-green pixels.

        The live "max reach" profile may use a permissive HSV range to avoid
        false negatives at distance.  Candidate discovery should still start
        from the measured saturated brick colors so low-saturation wall/table
        pixels cannot become ghost rectangles.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        strict_mask = cv2.inRange(
            hsv,
            np.array(CYAN_HSV_BALANCED_LOWER, dtype=np.uint8),
            np.array(CYAN_HSV_BALANCED_UPPER, dtype=np.uint8),
        )
        if loose_feature_mask is not None:
            strict_mask = cv2.bitwise_and(strict_mask, np.asarray(loose_feature_mask, dtype=np.uint8))

        kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        feature_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kern_open)

        close_w = 7
        close_h = max(5, int(round(float(frame.shape[0]) * 0.035)))
        if close_h % 2 == 0:
            close_h += 1
        close_h = min(17, close_h)
        kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
        contour_mask = cv2.morphologyEx(feature_mask, cv2.MORPH_CLOSE, kern_close)
        kern_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        contour_mask = cv2.dilate(contour_mask, kern_dilate, iterations=1)

        # Bricks may carry non-green surface markings (slots, stickers, dots)
        # that punch interior holes in the green mask and fragment the outline.
        # Fill each external contour solid so the brick reads as one block and
        # the bounding box wraps the full green outline (incl. its true width),
        # instead of the rectangle gates rejecting a holey/rounded blob.
        contour_mask = self._fill_mask_interior_holes(contour_mask)
        feature_mask = self._fill_mask_interior_holes(feature_mask)

        if cv2.countNonZero(contour_mask) <= 0 and loose_contour_mask is not None:
            return feature_mask, np.asarray(loose_contour_mask, dtype=np.uint8)
        return feature_mask, contour_mask

    @staticmethod
    def _fill_mask_interior_holes(mask):
        """Return a copy of `mask` with interior holes of each external contour
        filled solid. Exterior background is untouched, so this only closes gaps
        that lie inside a green outline (surface markings), never merges separate
        blobs."""
        if mask is None:
            return mask
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.ndim != 2 or arr.size == 0 or cv2.countNonZero(arr) <= 0:
            return mask
        contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        filled = arr.copy()
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    def _dominant_green_component_bbox(self, saturated_mask, *, frame_h: int, frame_w: int):
        """Tight bounding box around the whole brick's saturated-green outline.

        Returns (x, y, w, h) or None. Robust to brick shape and to full-width
        surface markings (e.g. a slot/band) that split the green into stacked
        pieces: the helper unions every green blob that horizontally overlaps the
        largest one, so a cap-over-body brick reads as a single rectangle. A
        faint floor reflection is a separate, off-to-the-side and usually
        desaturated blob, so it does not get unioned in."""
        if saturated_mask is None:
            return None
        filled = self._fill_mask_interior_holes(np.asarray(saturated_mask, dtype=np.uint8))
        if filled is None or cv2.countNonZero(filled) <= 0:
            return None
        try:
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(filled, 8)
        except Exception:
            return None
        min_area = max(150.0, float(frame_h * frame_w) * 0.003)
        comps = []
        for idx in range(1, int(count)):
            x0, y0, cw, ch, area = (float(stats[idx][k]) for k in range(5))
            if area < min_area:
                continue
            comps.append((int(x0), int(y0), int(cw), int(ch), float(area)))
        if not comps:
            return None
        main = max(comps, key=lambda c: c[4])
        mx1, mx2 = main[0], main[0] + main[2]
        main_w = max(1, main[2])
        selected = []
        for c in comps:
            cx1, cx2 = c[0], c[0] + c[2]
            overlap = min(mx2, cx2) - max(mx1, cx1)
            # Same brick column: meaningful horizontal overlap with the main blob.
            if c is main or overlap >= 0.35 * min(main_w, max(1, c[2])):
                selected.append(c)
        x1 = min(c[0] for c in selected)
        y1 = min(c[1] for c in selected)
        x2 = max(c[0] + c[2] for c in selected)
        y2 = max(c[1] + c[3] for c in selected)
        return (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1)))

    def _shrinkwrap_native_rect_bbox(
        self,
        feature_mask,
        *,
        bbox: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ):
        """Return a tight stack bbox from dense, straight-edged green pixels."""
        if feature_mask is None:
            return None
        x, y, w, h = [int(v) for v in bbox[:4]]
        x1 = max(0, min(int(frame_w), x))
        y1 = max(0, min(int(frame_h), y))
        x2 = max(0, min(int(frame_w), x + w))
        y2 = max(0, min(int(frame_h), y + h))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = np.asarray(feature_mask[y1:y2, x1:x2], dtype=np.uint8)
        if roi.ndim != 2 or roi.size == 0 or cv2.countNonZero(roi) < 4:
            return None

        close_w = max(3, min(13, int(round(float(roi.shape[1]) * 0.08))))
        close_h = max(3, min(11, int(round(float(roi.shape[0]) * 0.08))))
        if close_w % 2 == 0:
            close_w += 1
        if close_h % 2 == 0:
            close_h += 1
        work = cv2.morphologyEx(
            roi,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        )

        col_counts = np.count_nonzero(work, axis=0).astype(np.float32)
        if col_counts.size == 0 or float(np.max(col_counts)) <= 0.0:
            return None
        col_floor = max(
            3.0,
            float(np.max(col_counts)) * float(NATIVE_RECT_COLUMN_MAX_COUNT_RATIO),
        )
        active_cols = np.where(col_counts >= col_floor)[0]
        x_run = self._best_projection_run(
            active_cols,
            col_counts,
            min_width=max(
                int(NATIVE_RECT_COLUMN_MIN_WIDTH_PX),
                int(round(float(frame_w) * 0.018)),
            ),
        )
        if x_run is None:
            return None
        lx1, lx2 = x_run

        column_slice = work[:, int(lx1):int(lx2)]
        row_counts = np.count_nonzero(column_slice, axis=1).astype(np.float32)
        if row_counts.size == 0 or float(np.max(row_counts)) <= 0.0:
            return None
        row_floor = max(
            2.0,
            float(np.max(row_counts)) * float(NATIVE_RECT_ROW_MAX_COUNT_RATIO),
        )
        active_rows = np.where(row_counts >= row_floor)[0]
        if active_rows.size <= 0:
            return None
        ly1 = int(active_rows[0])
        ly2 = int(active_rows[-1]) + 1
        if lx2 <= lx1 or ly2 <= ly1:
            return None

        edge_metrics = self._native_straight_edge_metrics(
            work[int(ly1):int(ly2), int(lx1):int(lx2)]
        )
        if edge_metrics is None:
            return None

        nx1 = int(x1 + lx1)
        ny1 = int(y1 + ly1)
        nx2 = int(x1 + lx2)
        ny2 = int(y1 + ly2)
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return {
            "bbox": (nx1, ny1, int(nx2 - nx1), int(ny2 - ny1)),
            "edge_metrics": edge_metrics,
        }

    def _best_projection_run(self, active_indices, counts, *, min_width: int):
        if active_indices is None or len(active_indices) <= 0:
            return None
        idx = [int(v) for v in list(active_indices)]
        runs = []
        start = idx[0]
        prev = idx[0]
        for value in idx[1:]:
            if int(value) == int(prev) + 1:
                prev = int(value)
                continue
            runs.append((int(start), int(prev) + 1))
            start = int(value)
            prev = int(value)
        runs.append((int(start), int(prev) + 1))

        viable = [
            run for run in runs if int(run[1]) - int(run[0]) >= max(1, int(min_width))
        ]
        if not viable:
            return None

        def _score(run):
            start_idx, end_idx = run
            width = int(end_idx) - int(start_idx)
            density = float(np.sum(counts[int(start_idx):int(end_idx)]))
            return density + (float(width) * max(1.0, float(np.max(counts)) * 0.15))

        return max(viable, key=_score)

    def _native_straight_edge_metrics(self, mask_roi):
        arr = np.asarray(mask_roi, dtype=np.uint8)
        if arr.ndim != 2 or arr.size == 0:
            return None
        height, width = arr.shape[:2]
        row_counts = np.count_nonzero(arr, axis=1)
        active_floor = max(2, int(round(float(width) * 0.12)))
        active_rows = np.where(row_counts >= active_floor)[0]
        if int(active_rows.size) < int(NATIVE_RECT_MIN_EDGE_ROWS):
            return None

        left_edges = []
        right_edges = []
        widths = []
        for row_idx in active_rows:
            cols = np.flatnonzero(arr[int(row_idx), :])
            if cols.size <= 0:
                continue
            left = int(cols[0])
            right = int(cols[-1])
            left_edges.append(float(left))
            right_edges.append(float(right))
            widths.append(float(right - left + 1))
        if len(widths) < int(NATIVE_RECT_MIN_EDGE_ROWS):
            return None

        width_ref = max(1.0, float(width))
        left_std = float(np.std(np.asarray(left_edges, dtype=np.float32))) / width_ref
        right_std = float(np.std(np.asarray(right_edges, dtype=np.float32))) / width_ref
        mean_width = max(1.0, float(np.mean(np.asarray(widths, dtype=np.float32))))
        width_cv = float(np.std(np.asarray(widths, dtype=np.float32))) / mean_width
        edge_straightness = max(0.0, 1.0 - ((left_std + right_std) / 0.42))
        width_coherence = max(0.0, 1.0 - (width_cv / 0.55))
        active_row_ratio = float(len(widths)) / max(1.0, float(height))
        edge_score = (edge_straightness * 0.62) + (width_coherence * 0.38)
        return {
            "edge_straightness": float(min(1.0, edge_straightness)),
            "width_coherence": float(min(1.0, width_coherence)),
            "active_row_ratio": float(min(1.0, active_row_ratio)),
            "edge_score": float(min(1.0, edge_score)),
        }

    def _native_rect_geometry_ok(
        self,
        contour,
        *,
        bbox: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        area: float,
        fill_ratio: float,
        min_area_aspect: float,
        edge_metrics: dict | None = None,
    ) -> tuple[bool, dict]:
        _x, _y, w, h = bbox
        bbox_aspect = float(max(w, h)) / float(max(1, min(w, h)))
        bbox_width_ratio = float(w) / float(max(1, frame_w))
        hull_area = 0.0
        try:
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
        except Exception:
            hull_area = 0.0
        solidity = float(area) / float(hull_area) if hull_area > 0.0 else float(fill_ratio)
        metrics = {
            "bbox_aspect": float(bbox_aspect),
            "bbox_width_ratio": float(bbox_width_ratio),
            "solidity": float(solidity),
            "reason": "ok",
        }
        edge_metrics = edge_metrics if isinstance(edge_metrics, dict) else {}
        metrics.update(edge_metrics)
        if float(fill_ratio) < float(NATIVE_RECT_MIN_FILL_RATIO):
            metrics["reason"] = "low_fill"
            return False, metrics
        if float(solidity) < float(NATIVE_RECT_MIN_SOLIDITY):
            metrics["reason"] = "low_solidity"
            return False, metrics
        if float(bbox_aspect) > float(NATIVE_RECT_MAX_BBOX_ASPECT):
            metrics["reason"] = "implausible_bbox_aspect"
            return False, metrics
        if float(bbox_width_ratio) > float(NATIVE_RECT_MAX_BBOX_WIDTH_RATIO):
            metrics["reason"] = "implausible_bbox_width_ratio"
            return False, metrics
        if float(min_area_aspect) > float(NATIVE_RECT_MAX_MIN_AREA_ASPECT):
            metrics["reason"] = "implausible_rect_aspect"
            return False, metrics
        horizontal_aspect = float(w) / float(max(1, h))
        height_ratio = float(h) / float(max(1, frame_h))
        if (
            horizontal_aspect > float(NATIVE_RECT_STRIP_ASPECT)
            and height_ratio < float(NATIVE_RECT_STRIP_MAX_HEIGHT_RATIO)
        ):
            metrics["reason"] = "thin_horizontal_strip"
            return False, metrics
        if float(edge_metrics.get("active_row_ratio", 0.0)) < float(NATIVE_RECT_MIN_ACTIVE_ROW_RATIO):
            metrics["reason"] = "too_few_straight_edge_rows"
            return False, metrics
        if float(edge_metrics.get("edge_straightness", 0.0)) < float(NATIVE_RECT_MIN_EDGE_STRAIGHTNESS):
            metrics["reason"] = "unstable_vertical_edges"
            return False, metrics
        if float(edge_metrics.get("width_coherence", 0.0)) < float(NATIVE_RECT_MIN_WIDTH_COHERENCE):
            metrics["reason"] = "unstable_width"
            return False, metrics
        return True, metrics

    def _sync_frame_shape(self, frame) -> None:
        height, width = frame.shape[:2]
        self.frame_w = int(width)
        self.frame_h = int(height)
        self._detector.frame_w = int(width)
        self._detector.frame_h = int(height)

    def _mark_not_found(self, status: str) -> None:
        self._native_miss_count = min(12, int(getattr(self, "_native_miss_count", 0) or 0) + 1)
        if int(self._native_miss_count) > 3:
            self._reset_detector_tracking()
        self._detector.last_status = str(status or "shape mismatch")
        self.last_status = self._detector.last_status
        self._detector.last_primary_confidence = 0.0
        self._detector.last_max_confidence = 0.0
        self._detector._clear_partial_state()

    def _green_edge_close_range_result(self, frame, *, require_close: bool = False):
        """Close-range distance from the green stack's left/right edges. The side
        edges stay crisp even when the top overflows the frame at pick range.
        When require_close is True, only returns a result if the stack is wide
        enough to be unambiguously close range (used to OVERRIDE the unreliable
        rectangle path); otherwise it acts as a last-resort fallback."""
        if not bool(GREEN_EDGE_CLOSE_RANGE_ENABLED) or frame is None:
            return None
        detector = self._detector
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            balanced_mask = cv2.inRange(
                hsv,
                np.array(CYAN_HSV_BALANCED_LOWER, dtype=np.uint8),
                np.array(CYAN_HSV_BALANCED_UPPER, dtype=np.uint8),
            )
            wide_mask = cv2.inRange(
                hsv,
                np.array(CYAN_HSV_WIDE_LOWER, dtype=np.uint8),
                np.array(CYAN_HSV_WIDE_UPPER, dtype=np.uint8),
            )
            mask = cv2.bitwise_or(balanced_mask, wide_mask)
        except Exception:
            return None
        h, w = mask.shape[:2]
        if h <= 0 or w <= 0:
            return None
        top_strip = False
        painted_column = False
        painted_column_raw_dist = None
        painted_column_bbox = None
        painted_column_active_rows = None
        active = np.asarray([], dtype=np.int64)
        if bool(GREEN_EDGE_TOP_STRIP_ENABLED):
            row_counts = np.count_nonzero(mask, axis=1)
            row_thr = max(8, int(round(float(w) * float(GREEN_EDGE_TOP_STRIP_ROW_ACTIVE_FRAC))))
            active_rows = np.where(row_counts > row_thr)[0]
            if active_rows.size > 0:
                top_y = int(active_rows[0])
                top_limit = int(round(float(h) * float(GREEN_EDGE_TOP_STRIP_MAX_Y_RATIO)))
                contiguous = [int(top_y)]
                for row_idx in active_rows[1:]:
                    if int(row_idx) > top_limit:
                        break
                    if int(row_idx) > int(contiguous[-1]) + 2:
                        break
                    contiguous.append(int(row_idx))
                bottom_y = int(contiguous[-1]) + 1
                strip_h = int(bottom_y - top_y)
                if (
                    top_y <= top_limit
                    and strip_h >= int(GREEN_EDGE_TOP_STRIP_MIN_HEIGHT_PX)
                ):
                    strip = mask[int(top_y):int(bottom_y), :]
                    strip_counts = np.count_nonzero(strip, axis=0)
                    strip_thr = max(3, int(round(float(strip_h) * float(GREEN_EDGE_TOP_STRIP_COL_ACTIVE_FRAC))))
                    strip_active = np.where(strip_counts >= strip_thr)[0]
                    if strip_active.size >= 6:
                        strip_left = int(np.percentile(strip_active, 2))
                        strip_right = int(np.percentile(strip_active, 98))
                        strip_width = int(strip_right - strip_left)
                        margin = int(GREEN_EDGE_FRAME_MARGIN_PX)
                        edges_inside = (
                            int(strip_left) > int(margin)
                            and int(strip_right) < int(w - margin)
                        )
                        not_full_frame_artifact = float(strip_width) <= float(w) * 0.78
                        if (
                            strip_width >= int(GREEN_EDGE_TOP_STRIP_MIN_WIDTH_PX)
                            and bool(edges_inside)
                            and bool(not_full_frame_artifact)
                        ):
                            active = strip_active
                            top_strip = True
        if not bool(top_strip):
            coverage = float(np.count_nonzero(mask)) / float(h * w)
            coverage_floor = float(GREEN_EDGE_MIN_COVERAGE)
            if bool(GREEN_EDGE_PAINTED_COLUMN_ENABLED):
                coverage_floor = min(coverage_floor, float(GREEN_EDGE_PAINTED_COLUMN_MIN_COVERAGE))
            if coverage < coverage_floor:
                return None
            col_counts = np.count_nonzero(mask, axis=0)
            col_active_frac = float(GREEN_EDGE_COL_ACTIVE_FRAC)
            if bool(GREEN_EDGE_PAINTED_COLUMN_ENABLED):
                col_active_frac = min(col_active_frac, float(GREEN_EDGE_PAINTED_COLUMN_COL_ACTIVE_FRAC))
            col_thr = max(8, int(col_active_frac * float(h)))
            active = np.where(col_counts > col_thr)[0]
            if bool(GREEN_EDGE_PAINTED_COLUMN_ENABLED):
                # The permissive wide mask is excellent for true close top-strip
                # reads, but at reset distance it can glue the painted stack to
                # wall/table artifacts and publish a fake ~90mm distance.  For
                # the painted-column fallback, measure the actual saturated
                # green component instead.
                balanced_labels = None
                try:
                    balanced_labels = cv2.connectedComponentsWithStats(balanced_mask, 8)
                except Exception:
                    balanced_labels = None
                best_component = None
                if balanced_labels is not None:
                    comp_count, _labels, stats, _centroids = balanced_labels
                    min_height = float(h) * float(GREEN_EDGE_PAINTED_COLUMN_MIN_HEIGHT_FRAC)
                    min_area = max(120.0, float(h * w) * 0.002)
                    for comp_idx in range(1, int(comp_count)):
                        x0, y0, cw, ch, area = stats[comp_idx]
                        if float(area) < float(min_area):
                            continue
                        if float(ch) < float(min_height):
                            continue
                        if int(cw) < int(GREEN_EDGE_MIN_WIDTH_PX):
                            continue
                        raw_component_dist = detector._estimate_distance_from_width(float(cw))
                        if raw_component_dist is None or float(raw_component_dist) <= 0.0:
                            continue
                        # Prefer the tall green column/stack component over
                        # floor reflections.  Rows high in the image are the
                        # supply stack; bottom-only blobs are usually reflection.
                        top_weight = max(0.0, 1.0 - (float(y0) / max(1.0, float(h))))
                        score = (float(area) * 0.02) + float(ch) + (float(cw) * 0.5) + (top_weight * 120.0)
                        if best_component is None or score > float(best_component["score"]):
                            best_component = {
                                "score": float(score),
                                "bbox": (int(x0), int(y0), int(cw), int(ch)),
                                "raw_dist": float(raw_component_dist),
                            }
                if best_component is not None:
                    bx, by, bw, bh = best_component["bbox"]
                    painted_column = True
                    painted_column_bbox = (int(bx), int(by), int(bw), int(bh))
                    painted_column_raw_dist = float(best_component["raw_dist"])
                    painted_column_active_rows = np.arange(int(by), int(by + bh), dtype=np.int64)
                    active = np.arange(int(bx), int(bx + bw), dtype=np.int64)
                elif active.size >= 6:
                    left_probe = int(np.percentile(active, 2))
                    right_probe = int(np.percentile(active, 98))
                    band = mask[:, max(0, left_probe):min(w, right_probe + 1)]
                    active_rows = np.where(np.count_nonzero(band, axis=1) > 0)[0]
                    if active_rows.size > 0:
                        run_height = int(active_rows[-1]) - int(active_rows[0]) + 1
                        painted_column = bool(float(run_height) >= (float(h) * float(GREEN_EDGE_PAINTED_COLUMN_MIN_HEIGHT_FRAC)))
        if active.size < 6:
            return None
        left = int(np.percentile(active, 2))
        right = int(np.percentile(active, 98))
        width_px = right - left
        if width_px < int(GREEN_EDGE_MIN_WIDTH_PX):
            return None
        if bool(top_strip) and width_px < int(GREEN_EDGE_TOP_STRIP_MIN_WIDTH_PX):
            return None
        # When used to override the rectangle path, require the stack to be wide
        # enough to be unambiguously close range.
        if bool(require_close) and width_px < int(GREEN_EDGE_PRIMARY_MIN_WIDTH_PX):
            return None
        margin = int(GREEN_EDGE_FRAME_MARGIN_PX)
        # Both side edges must sit inside the frame, else the width is cut off.
        if left <= margin or right >= (w - margin):
            return None
        raw_dist = (
            float(painted_column_raw_dist)
            if bool(painted_column) and painted_column_raw_dist is not None
            else detector._estimate_distance_from_width(float(width_px))
        )
        max_green_edge_dist = (
            float(GREEN_EDGE_PAINTED_COLUMN_MAX_DIST_MM)
            if bool(painted_column)
            else float(GREEN_EDGE_MAX_DIST_MM)
        )
        if raw_dist is None or not (0.0 < float(raw_dist) <= float(max_green_edge_dist)):
            return None
        calibrated_raw_dist = float(raw_dist)
        if (
            bool(painted_column)
            and float(raw_dist) <= float(GREEN_EDGE_PAINTED_COLUMN_MAX_RAW_CALIBRATION_MM)
        ):
            calibrated_raw_dist = _calibrate_painted_close_green_edge_dist(float(raw_dist))
        gate_primary = {
            "shape_profile": "native_rect",
            "native_rect_confidence_pct": float(GREEN_EDGE_CONF_PCT),
        }
        prev_for_dist = detector._prev_dist
        if prev_for_dist is None:
            prev_for_dist = self._native_last_good_dist
        detector.last_raw_dist = float(raw_dist)
        detector.last_bbox_dist = float(calibrated_raw_dist)
        detector.last_bbox_distance_source = "green_edge_close_range"
        dist = float(calibrated_raw_dist)
        detector._prev_dist = dist
        detector.last_final_dist = dist
        self._native_last_good_dist = float(dist)
        self._native_miss_count = 0
        cx = float(left + right) / 2.0
        if bool(painted_column) and painted_column_active_rows is not None:
            rows = painted_column_active_rows
        else:
            band = mask[:, int(left):int(right)]
            green_rows = np.where(np.count_nonzero(band, axis=1) > 0)[0]
            # The brick is one contiguous block of green rows from the top. A
            # glossy floor can mirror it as faint green near the bottom of the
            # frame, separated by a gap of non-green wood. Walk down from the
            # topmost green row, tolerating small gaps (interior markings such
            # as a slot or stickers) but stopping at the first large gap, so the
            # box wraps only the brick and not its reflection.
            if green_rows.size > 0:
                row_gap_tol = max(4, int(round(float(h) * 0.05)))
                top_row = int(green_rows[0])
                bottom_row = top_row
                for row_idx in green_rows[1:]:
                    if int(row_idx) - bottom_row > row_gap_tol:
                        break
                    bottom_row = int(row_idx)
                rows = green_rows[(green_rows >= top_row) & (green_rows <= bottom_row)]
            else:
                rows = green_rows
        cy = float(rows.mean()) if rows.size > 0 else float(h) / 2.0
        if bool(painted_column) and painted_column_bbox is not None:
            box_x, box_y, box_w, box_h = [int(v) for v in painted_column_bbox]
        else:
            # Prefer a tight box around the whole saturated-green blob so the
            # rectangle wraps the full brick (width included), independent of the
            # strip-column percentiles, which can overshoot on stray columns.
            comp_bbox = self._dominant_green_component_bbox(
                balanced_mask, frame_h=int(h), frame_w=int(w)
            )
            if comp_bbox is not None:
                box_x, box_y, box_w, box_h = comp_bbox
            else:
                box_x = int(left)
                box_y = int(rows[0]) if rows.size > 0 else 0
                box_w = max(1, int(right) - int(left))
                box_h = max(1, (int(rows[-1]) - int(rows[0]) + 1) if rows.size > 0 else int(h))
        detector.last_bbox_w_px = int(box_w)
        detector.last_bbox_h_px = int(box_h)
        detector.last_bbox_eff_w_px = int(box_w)
        detector.last_bbox_eff_h_px = int(box_h)
        detector.last_bbox_width_dist = float(raw_dist)
        detector.last_bbox_height_dist = None
        detector.last_bbox_calibrated_height_dist = None
        detector.last_distance_display_text = f"{float(dist):.0f}mm"
        raw_offset_x = detector._estimate_offset_x_mm(cx, dist)
        offset_x = detector._smooth(raw_offset_x, detector._prev_offset)
        detector._prev_offset = offset_x
        raw_cam_height = detector._estimate_cam_height(cy, dist)
        cam_height = detector._smooth(raw_cam_height, detector._prev_offset_y)
        detector._prev_offset_y = cam_height
        if bool(top_strip):
            conf_pct = float(GREEN_EDGE_TOP_STRIP_CONF_PCT)
        elif bool(painted_column):
            conf_pct = float(GREEN_EDGE_PAINTED_COLUMN_CONF_PCT)
        else:
            conf_pct = float(GREEN_EDGE_CONF_PCT)
        detector.last_primary_confidence = conf_pct / 100.0
        detector.last_max_confidence = conf_pct / 100.0
        if bool(top_strip):
            detector.last_geometry_source = "green_edge_top_strip_width"
            detector.last_status = "target locked (green-edge top strip width)"
        elif bool(painted_column):
            detector.last_geometry_source = "green_edge_painted_column_width"
            detector.last_status = "target locked (painted green column width)"
        else:
            detector.last_geometry_source = "green_edge_close_range_width"
            detector.last_status = "target locked (green-edge close range)"
        self.last_status = detector.last_status
        detector.last_candidate_count = 1
        detector.last_raw_prediction_count = 1
        detector.last_nms_count = 1
        debug_frame = frame.copy()
        if bool(self.debug):
            pt1 = (max(0, int(box_x)), max(0, int(box_y)))
            pt2 = (
                min(int(w) - 1, int(box_x + box_w)),
                min(int(h) - 1, int(box_y + box_h)),
            )
            cv2.rectangle(debug_frame, pt1, pt2, (0, 255, 0), 2)
            center = (int(round(cx)), int(round(cy)))
            cv2.circle(debug_frame, center, 4, (0, 255, 255), 2)
            cv2.putText(
                debug_frame,
                f"{detector.last_geometry_source} dist={float(dist):.0f} x={float(offset_x):+.1f}",
                (max(4, pt1[0]), max(18, pt1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        self.current_frame = debug_frame
        detector.current_frame = debug_frame
        return (
            True,
            0.0,
            float(dist),
            float(offset_x),
            float(conf_pct),
            float(cam_height),
            False,
            False,
        )

    def _result_from_candidate(self, frame, primary: dict, candidates: list[dict]):
        detector = self._detector
        if str(primary.get("shape_profile") or "") == "native_rect":
            # Model lock: do not swap to the green-edge close/top-strip/painted-column
            # distance model mid-run. That switch caused 250mm poses to jump to
            # about 90mm and made the follow game dishonest.
            primary = dict(primary)
            primary["native_rect_distance_fallback_only"] = True
        raw_angle = detector._refine_angle_for_primary(frame, primary)
        angle = detector._smooth_angle(raw_angle, detector._prev_angle)
        detector._prev_angle = angle

        anchor_cx, anchor_cy = detector._candidate_face_midpoint(primary)
        if self._native_rect_center_jump_rejected(primary, anchor_cx, anchor_cy):
            self._mark_not_found("native center jump rejected")
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
        bbox = primary.get("bbox", (0, 0, 1, 1))
        if str(primary.get("shape_profile") or "") == "native_rect":
            _bx, _by, bbox_w, bbox_h = bbox[:4]
        else:
            face_size = detector._candidate_face_size_px(primary)
            if face_size is not None:
                bbox_w, bbox_h = face_size
            else:
                _bx, _by, bbox_w, bbox_h = bbox[:4]
        components = detector._distance_components_from_box(
            bbox_w,
            bbox_h,
            primary.get("partial_kind"),
        )
        detector._record_bbox_distance_components(components)
        if str(primary.get("shape_profile") or "") == "native_rect":
            raw_dist = components.get("width_dist_mm")
            detector.last_bbox_dist = raw_dist
            detector.last_bbox_distance_source = "native_rect_width"
            detector.last_pre_depth_dist = raw_dist
            detector.last_depth_dist = None
            detector.last_depth_stats = {}
            detector.last_geometry_source = "native_rect_width_fallback_only"
        else:
            raw_dist = detector._estimate_distance_from_box(
                bbox_w,
                bbox_h,
                primary.get("partial_kind"),
            )
            tri_span_dist = detector._dist_from_triangle_span(primary.get("negative_cutout_polygons"))
            if tri_span_dist is not None:
                raw_dist = tri_span_dist
            raw_dist = detector._prefer_depth_distance(
                raw_dist,
                anchor_cx,
                anchor_cy,
                bbox=primary.get("bbox"),
            )
        detector.last_raw_dist = raw_dist
        if str(primary.get("shape_profile") or "") == "native_rect":
            self._maybe_reset_dist_filter_for_native_candidate(primary)
        prev_for_dist = detector._prev_dist
        if prev_for_dist is None:
            prev_for_dist = self._native_last_good_dist
        if str(primary.get("shape_profile") or "") == "native_rect":
            dist = float(raw_dist)
        else:
            dist = self._native_rect_distance_from_raw(raw_dist, prev_for_dist, primary)
        detector._prev_dist = dist
        detector.last_final_dist = dist
        if (
            str(primary.get("shape_profile") or "") == "native_rect"
            and not str(getattr(detector, "last_geometry_source", "") or "").startswith("native_rect_width_stable_avg")
        ):
            detector.last_status = (
                f"target locked (native robust median warming "
                f"({int(getattr(detector, 'last_stable_dist_inlier_count', 0) or 0)} "
                f"samples, spread={float(getattr(detector, 'last_stable_dist_spread_mm', 0.0) or 0.0):.1f}mm))"
            )
            self.last_status = detector.last_status
        if str(primary.get("shape_profile") or "") == "native_rect":
            self._native_last_good_dist = float(dist)
            self._native_last_good_center = (float(anchor_cx), float(anchor_cy))
            self._native_last_good_width_px = float(bbox_w)
            self._native_miss_count = 0

        raw_offset_x = detector._estimate_offset_x_mm(anchor_cx, dist)
        offset_x = detector._smooth(raw_offset_x, detector._prev_offset)
        detector._prev_offset = offset_x

        raw_cam_height = detector._estimate_cam_height(anchor_cy, dist)
        cam_height = detector._smooth(raw_cam_height, detector._prev_offset_y)
        detector._prev_offset_y = cam_height

        detected_bricks = [primary]
        brick_above, brick_below = detector._stack_flags_from_individuals(
            primary,
            candidates if candidates else detected_bricks,
        )
        partial_bricks = [
            {"partial": bool(c.get("partial")), "label": c.get("partial_label")}
            for c in candidates
            if isinstance(c, dict) and bool(c.get("partial"))
        ]
        detector._set_partial_state(
            partial_bricks,
            primary_partial_kind=primary.get("partial_kind"),
            primary_partial_label=(
                primary.get("partial_label") if bool(primary.get("partial")) else None
            ),
        )

        conf_pct = float(primary.get("native_rect_confidence_pct") or 100.0)
        if bool(primary.get("from_color_detection")):
            conf_pct = max(conf_pct, float(COLOR_ONLY_CONF_PCT))
        detector.last_primary_confidence = conf_pct / 100.0
        detector.last_max_confidence = conf_pct / 100.0
        if not (
            str(primary.get("shape_profile") or "") == "native_rect"
            and "native robust median warming" in str(getattr(detector, "last_status", ""))
        ):
            detector.last_status = (
                "target locked (native color+rect)"
                if str(primary.get("shape_profile") or "") == "native_rect"
                else "target locked (native color+shape)"
            )
        self.last_status = detector.last_status
        detector.last_suspected_far_brick = False
        detector.last_distance_display_text = None
        return (
            True,
            float(angle),
            float(dist),
            float(offset_x),
            float(conf_pct),
            float(cam_height),
            bool(brick_above),
            bool(brick_below),
        )

    def _select_native_rect_candidate(self, candidates, frame_w: int, frame_h: int):
        candidate_list = [
            candidate for candidate in list(candidates or []) if isinstance(candidate, dict)
        ]
        if not candidate_list:
            return None
        frame_cx = float(frame_w) * 0.5 + float(getattr(self, "camera_center_offset_px", 0.0) or 0.0)
        frame_cy = float(frame_h) * 0.5
        previous_center = self._native_last_good_center
        previous_width = self._native_last_good_width_px
        has_previous = isinstance(previous_center, tuple) and len(previous_center) == 2
        miss_count = int(getattr(self, "_native_miss_count", 0) or 0)

        def _candidate_values(candidate):
            cx, cy = self._detector._candidate_face_midpoint(candidate)
            bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
            if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                try:
                    _x, _y, bw, bh = [float(v) for v in bbox[:4]]
                except (TypeError, ValueError):
                    bw, bh = 1.0, 1.0
            else:
                bw, bh = 1.0, 1.0
            return float(cx), float(cy), max(1.0, float(bw)), max(1.0, float(bh))

        scored = []
        for idx, candidate in enumerate(candidate_list):
            cx, cy, bw, bh = _candidate_values(candidate)
            edge_score = float(candidate.get("edge_score", 0.0) or 0.0)
            width_coherence = float(candidate.get("width_coherence", 0.0) or 0.0)
            conf = float(candidate.get("native_rect_confidence_pct", candidate.get("confidence_pct", 0.0)) or 0.0)
            proxy_dist = candidate.get("native_width_dist_mm")
            try:
                dist_preference = abs(float(proxy_dist) - float(NATIVE_RECT_PREFERRED_LOCK_DIST_MM))
            except (TypeError, ValueError):
                dist_preference = 0.0
            screen_dist = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5)
            continuity_dist = screen_dist
            width_ratio = 0.0
            if has_previous:
                px, py = float(previous_center[0]), float(previous_center[1])
                continuity_dist = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
                try:
                    prev_w = max(1.0, float(previous_width))
                    width_ratio = abs(float(bw) - prev_w) / prev_w
                except (TypeError, ValueError):
                    width_ratio = 0.0
            score = (
                # Strong continuity weight = sticky tracking: once locked, stay on the
                # brick nearest last frame's lock unless it clearly disappears. This
                # stops the lock hopping between similar-width candidates (the stack /
                # adjacent bricks / ghosts) which made x and distance jump frame-to-frame.
                (continuity_dist * (2.5 if has_previous else 0.35))
                + (screen_dist * (0.12 if has_previous else 1.0))
                + (width_ratio * 90.0)
                + (dist_preference * (0.08 if has_previous else 0.65))
                - (edge_score * 12.0)
                - (width_coherence * 10.0)
                - (conf * 0.03)
            )
            scored.append((float(score), int(idx), continuity_dist, width_ratio, screen_dist))

        scored.sort(key=lambda row: (row[0], row[4]))
        best_score, best_idx, continuity_dist, width_ratio, _screen_dist = scored[0]
        if has_previous:
            max_center_step = float(NATIVE_RECT_MAX_CENTER_STEP_PX) + (float(miss_count) * 12.0)
            max_width_ratio = float(NATIVE_RECT_MAX_WIDTH_RATIO_JUMP) + (float(miss_count) * 0.04)
            # Reject if the best candidate jumped too far in position OR changed width
            # too much (was AND, which let a same-width candidate at a very different
            # position pass and hop the lock). A single rejected frame holds the last
            # good reading via the visibility bridge rather than publishing a jump.
            # The miss_count terms still relax both thresholds so a genuinely lost lock
            # can re-acquire.
            if float(continuity_dist) > max_center_step or float(width_ratio) > max_width_ratio:
                self.last_status = "native color rectangle continuity reject"
                self._detector.last_status = self.last_status
                return None
        selected = candidate_list[int(best_idx)]
        try:
            selected["native_tracker_score"] = float(best_score)
            selected["native_tracker_center_delta_px"] = float(continuity_dist)
            selected["native_tracker_width_ratio_delta"] = float(width_ratio)
            selected["native_tracker_has_previous"] = bool(has_previous)
        except Exception:
            pass
        return selected

    def _native_rect_center_jump_rejected(self, primary: dict | None, center_x, center_y) -> bool:
        if str((primary or {}).get("shape_profile") or "") != "native_rect":
            return False
        previous = self._native_last_good_center
        if not isinstance(previous, tuple) or len(previous) != 2:
            return False
        try:
            px = float(previous[0])
            py = float(previous[1])
            cx = float(center_x)
            cy = float(center_y)
        except (TypeError, ValueError):
            return False
        max_step = float(NATIVE_RECT_MAX_CENTER_STEP_PX)
        center_step = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
        if center_step <= max_step:
            return False
        frame_cx = float(self.frame_w) * 0.5 + float(getattr(self, "camera_center_offset_px", 0.0) or 0.0)
        frame_cy = float(self.frame_h) * 0.5
        previous_screen_dist = float(((px - frame_cx) ** 2 + (py - frame_cy) ** 2) ** 0.5)
        current_screen_dist = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5)
        if current_screen_dist + 35.0 < previous_screen_dist:
            return False
        bbox = (primary or {}).get("bbox")
        try:
            _x, _y, width_px, _h = [float(v) for v in bbox[:4]]
            prev_w = max(1.0, float(self._native_last_good_width_px))
            width_ratio = abs(float(width_px) - prev_w) / prev_w
        except (TypeError, ValueError):
            width_ratio = 0.0
        return bool(width_ratio > float(NATIVE_RECT_MAX_WIDTH_RATIO_JUMP))

    def _maybe_reset_dist_filter_for_native_candidate(self, primary: dict | None) -> None:
        if str((primary or {}).get("shape_profile") or "") != "native_rect":
            return
        if not bool((primary or {}).get("native_tracker_has_previous")):
            return
        try:
            center_delta = float((primary or {}).get("native_tracker_center_delta_px") or 0.0)
        except (TypeError, ValueError):
            center_delta = 0.0
        try:
            width_ratio = float((primary or {}).get("native_tracker_width_ratio_delta") or 0.0)
        except (TypeError, ValueError):
            width_ratio = 0.0
        if (
            center_delta > float(NATIVE_RECT_DIST_RESET_CENTER_STEP_PX)
            or width_ratio > float(NATIVE_RECT_DIST_RESET_WIDTH_RATIO)
        ):
            self._reset_native_dist_filter()

    @staticmethod
    def _trimmed_mean(values: list[float]) -> float:
        cleaned = sorted(float(v) for v in values)
        if len(cleaned) >= 5:
            cleaned = cleaned[1:-1]
        if not cleaned:
            return 0.0
        return float(statistics.mean(cleaned))

    def _native_rect_distance_from_raw(self, raw_dist, prev_dist, primary: dict | None) -> float:
        try:
            raw_val = float(raw_dist)
        except (TypeError, ValueError):
            raw_val = 999.0
        if str((primary or {}).get("shape_profile") or "") != "native_rect":
            return float(self._detector._smooth(raw_val, prev_dist))
        history = getattr(self, "_native_dist_history", None)
        if history is None:
            self._native_dist_history = deque(maxlen=int(NATIVE_RECT_DIST_HISTORY_LEN))
            history = self._native_dist_history

        # NOTE: An earlier experiment added a temporal "distance gate" and a
        # confidence-hold gate here that, on a jump beyond GATE_MM from the last
        # locked value, held the last good distance until several consecutive raw
        # readings agreed. That assumes a STATIONARY brick: when the robot itself
        # moves, the real distance legitimately changes every frame, so the gate
        # treated genuine approach as "flicker" and FROZE the published distance
        # (then snapped to the new level on re-lock) — locking failed while moving.
        # The median window + outlier band + max-publish-step below already reject
        # single-frame segmentation flicker without freezing, so the raw reading is
        # fed straight in and the published distance tracks real motion.
        history.append(float(raw_val))
        values = [float(value) for value in list(history)]
        median = float(statistics.median(values))
        inliers = [
            float(value)
            for value in values
            if abs(float(value) - float(median)) <= float(NATIVE_RECT_DIST_OUTLIER_BAND_MM)
        ]
        if not inliers:
            inliers = [float(median)]
        spread = float(max(inliers) - min(inliers)) if len(inliers) > 1 else 0.0
        sample_count = len(values)
        inlier_count = len(inliers)
        self._detector.last_stable_dist_sample_count = int(sample_count)
        self._detector.last_stable_dist_inlier_count = int(inlier_count)
        self._detector.last_stable_dist_spread_mm = float(spread)
        self._detector.last_stable_dist_unlimited_mm = None
        self._detector.last_stable_dist_published_mm = None
        self._detector.last_stable_dist_publish_limited = False

        # Disabled for now: this stable-average publisher can disagree with the
        # livestream/top-strip distance model at close range and make follow
        # trials act on a different distance than the operator sees.

        # While the window is warming or contains a rejected transition, use a
        # robust median. This avoids the old artificial 8mm/frame staircase.
        robust = float(statistics.median(inliers))
        self._detector.last_geometry_source = "native_rect_width_robust_median"
        return float(robust)

    def _draw_debug_frame(self, frame, primary, candidates, result) -> None:
        debug_frame = frame.copy()
        if self.debug:
            _found, angle, dist, offset_x, conf_pct, _cam_height, _above, _below = result
            self._detector._draw_debug_hsv(
                debug_frame,
                [],
                candidates if candidates else [primary],
                primary,
                angle,
                dist,
                offset_x,
                conf_pct / 100.0,
            )
        self.current_frame = debug_frame
        self._detector.current_frame = debug_frame

    def release(self) -> None:
        cap = self.cap
        self.cap = None
        self._detector.cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def close(self) -> None:
        self.release()


BrickDetector = NativeOakBrickDetector


__all__ = [
    "BRICK_HEIGHT_MM",
    "BRICK_WIDTH_MM",
    "BrickDetector",
    "NativeOakBrickDetector",
]
