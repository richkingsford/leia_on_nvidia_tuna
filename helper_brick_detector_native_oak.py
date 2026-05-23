"""Native OAK-D brick detector using green color and brick shape gates.

This module intentionally does not load TensorRT/YOLO.  It wraps the existing
camera-free color/shape detector with the same runtime interface used by
follow-the-brick:

    read() -> (found, angle, dist, offset_x, confidence, cam_height,
               brick_above, brick_below)
"""

from __future__ import annotations

import logging
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
    DEFAULT_FRAME_H,
    DEFAULT_FRAME_W,
    FOCAL_PX_REF,
    FOCAL_REF_WIDTH,
    NMS_THRESHOLD,
    YOLO_INPUT_SIZE,
    build_negative_cutout_shape_detector,
    detect_single_negative_cutout_brick,
)


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
        self._detector.conf_threshold = float(self.conf_threshold)
        self._detector.nms_threshold = float(self.nms_threshold)
        self._detector.input_size = int(YOLO_INPUT_SIZE)
        self._detector.focal_px = float(
            focal_px
            if focal_px is not None
            else FOCAL_PX_REF * (float(self.frame_w) / float(FOCAL_REF_WIDTH))
        )
        self._detector._smooth_alpha = 0.15
        self._detector._depth_source_mode = "pinhole"
        self._detector._stereo_config_mode = "standard"
        self._detector.cap = None
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
        if primary is None:
            primary, candidates = detect_single_negative_cutout_brick(self._detector, frame)
        self._detector.last_raw_prediction_count = int(len(candidates))
        self._detector.last_candidate_count = int(len(candidates))
        self._detector.last_nms_count = int(len(candidates))

        if primary is None:
            self._mark_not_found(str(getattr(self._detector, "last_status", "shape mismatch")))
            self.current_frame = frame.copy()
            self._detector.current_frame = self.current_frame
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

        result = self._result_from_candidate(frame, primary, candidates)
        self._draw_debug_frame(frame, primary, candidates, result)
        return result

    def _detect_color_rectangle_candidate(self, frame):
        detector = self._detector
        feature_mask, contour_mask = detector._build_hsv_masks(frame)
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
            bbox_area = float(max(1, w * h))
            fill_ratio = area / bbox_area
            if fill_ratio < 0.16:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = float(rect[1][0]), float(rect[1][1])
            if rw <= 0.0 or rh <= 0.0:
                continue
            aspect = float(max(rw, rh)) / float(max(1e-6, min(rw, rh)))
            if aspect > 6.0:
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
            area_score = min(10.0, area / 1800.0)
            confidence = max(
                75.0,
                min(98.0, 62.0 + (coverage * 30.0) + (fill_ratio * 15.0) + area_score),
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
                    "cyan_coverage": float(coverage),
                    "fill_ratio": float(fill_ratio),
                }
            )

        if not candidates:
            detector.last_status = "native color rectangle mismatch"
            self.last_status = detector.last_status
            return None, []

        candidates = detector._enforce_non_overlapping_candidate_bboxes(candidates)
        selected = detector._select_center_brick(candidates, frame_w, frame_h)
        if selected is None:
            detector.last_status = "native color rectangle mismatch"
            self.last_status = detector.last_status
            return None, candidates
        detector.last_status = "target locked (native color+rect)"
        self.last_status = detector.last_status
        return selected, candidates

    def _sync_frame_shape(self, frame) -> None:
        height, width = frame.shape[:2]
        self.frame_w = int(width)
        self.frame_h = int(height)
        self._detector.frame_w = int(width)
        self._detector.frame_h = int(height)

    def _mark_not_found(self, status: str) -> None:
        self._reset_detector_tracking()
        self._detector.last_status = str(status or "shape mismatch")
        self.last_status = self._detector.last_status
        self._detector.last_primary_confidence = 0.0
        self._detector.last_max_confidence = 0.0
        self._detector._clear_partial_state()

    def _result_from_candidate(self, frame, primary: dict, candidates: list[dict]):
        detector = self._detector
        raw_angle = detector._refine_angle_for_primary(frame, primary)
        angle = detector._smooth_angle(raw_angle, detector._prev_angle)
        detector._prev_angle = angle

        anchor_cx, anchor_cy = detector._candidate_face_midpoint(primary)
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
        dist = detector._smooth(raw_dist, detector._prev_dist)
        detector._prev_dist = dist
        detector.last_final_dist = dist

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
