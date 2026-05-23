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
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
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
NATIVE_RECT_MAX_MIN_AREA_ASPECT = 4.25
NATIVE_RECT_STRIP_ASPECT = 2.55
NATIVE_RECT_STRIP_MAX_HEIGHT_RATIO = 0.12
NATIVE_RECT_COLUMN_MIN_WIDTH_PX = 6
NATIVE_RECT_COLUMN_MAX_COUNT_RATIO = 0.28
NATIVE_RECT_ROW_MAX_COUNT_RATIO = 0.12
NATIVE_RECT_MIN_EDGE_ROWS = 5
NATIVE_RECT_MIN_ACTIVE_ROW_RATIO = 0.14
NATIVE_RECT_MIN_EDGE_STRAIGHTNESS = 0.22
NATIVE_RECT_MIN_WIDTH_COHERENCE = 0.25


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
                }
            )

        if not candidates:
            detector.last_status = "native color rectangle mismatch"
            self.last_status = detector.last_status
            return None, []

        candidates = detector._enforce_non_overlapping_candidate_bboxes(candidates)
        candidates = detector._filter_candidates_to_center_stack(candidates, frame_w, frame_h)
        selected = detector._select_center_brick(candidates, frame_w, frame_h)
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

        if cv2.countNonZero(contour_mask) <= 0 and loose_contour_mask is not None:
            return feature_mask, np.asarray(loose_contour_mask, dtype=np.uint8)
        return feature_mask, contour_mask

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
        hull_area = 0.0
        try:
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
        except Exception:
            hull_area = 0.0
        solidity = float(area) / float(hull_area) if hull_area > 0.0 else float(fill_ratio)
        metrics = {
            "bbox_aspect": float(bbox_aspect),
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
