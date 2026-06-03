"""Sidecar held-brick detector.

This helper intentionally does not participate in target brick selection.  It
only inspects a fixed upper/prong ROI in an already-captured BGR frame and
reports whether a large green/cyan region is present.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from helper_brick_detector_yolo import (
    BRICK_WIDTH_MM,
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
    CYAN_HSV_WIDE_LOWER,
    CYAN_HSV_WIDE_UPPER,
)


@dataclass(frozen=True)
class HoldingBrickConfig:
    roi_y_min_ratio: float = 0.06
    roi_y_max_ratio: float = 0.46
    roi_x_min_ratio: float = 0.08
    roi_x_max_ratio: float = 0.92
    min_area_px: int = 1800
    min_width_ratio: float = 0.45
    min_height_px: int = 24
    min_coverage_ratio: float = 0.035
    max_center_y_ratio: float = 0.34
    top_nub_strip_ratio: float = 0.13
    top_nub_strip_min_px: int = 24
    top_nub_strip_max_px: int = 44
    top_nub_bridge_cut_px: int = 6
    top_nub_min_area_px: int = 35
    top_nub_min_width_px: int = 5
    top_nub_min_height_px: int = 7
    top_nub_min_pair_gap_ratio: float = 0.12
    top_nub_max_pair_center_y_ratio: float = 0.11
    top_edge_min_width_ratio: float = 0.18
    top_edge_min_area_px: int = 180
    top_edge_min_height_px: int = 18


DEFAULT_HOLDING_BRICK_CONFIG = HoldingBrickConfig()

HELD_BRICK_MASK_BOTTOM_FRACTION = 0.72


@dataclass(frozen=True)
class HoldingMaskLockConfig:
    release_after_misses: int = 8


DEFAULT_HOLDING_MASK_LOCK_CONFIG = HoldingMaskLockConfig()


class HoldingMaskLock:
    """Keep a held-brick mask stable through brief detection flicker."""

    def __init__(self, config: HoldingMaskLockConfig | None = None) -> None:
        self.config = config or DEFAULT_HOLDING_MASK_LOCK_CONFIG
        self._locked_result: dict[str, Any] | None = None
        self._misses = 0
        self._age = 0

    def reset(self) -> None:
        self._locked_result = None
        self._misses = 0
        self._age = 0

    def update(self, result: dict[str, Any] | None) -> dict[str, Any]:
        raw = dict(result) if isinstance(result, dict) else {"holding": False, "reason": "invalid_holding_result"}
        if bool(raw.get("holding")):
            self._misses = 0
            if self._locked_result is None:
                self._locked_result = deepcopy(raw)
                self._age = 0
            elif not self._locked_result.get("mask_regions") and raw.get("mask_regions"):
                self._locked_result = deepcopy(raw)
                self._age = 0
            else:
                self._age += 1
            return self._locked_copy(raw_reason=raw.get("reason"))

        if self._locked_result is None:
            return raw

        self._misses += 1
        if self._misses >= max(1, int(self.config.release_after_misses)):
            released = dict(raw)
            released["holding_mask_lock_released"] = True
            released["holding_mask_lock_misses"] = self._misses
            self.reset()
            return released

        self._age += 1
        return self._locked_copy(raw_reason=raw.get("reason"), missed=True)

    def _locked_copy(self, *, raw_reason: Any = None, missed: bool = False) -> dict[str, Any]:
        locked = deepcopy(self._locked_result) if self._locked_result is not None else {}
        locked["holding"] = True
        locked["holding_mask_locked"] = True
        locked["holding_mask_lock_age"] = self._age
        locked["holding_mask_lock_misses"] = self._misses
        if raw_reason is not None:
            locked["holding_mask_lock_raw_reason"] = raw_reason
        if missed:
            locked["reason"] = "held_brick_mask_lock"
        return locked


@dataclass(frozen=True)
class TargetContourConfig:
    min_area_px: int = 900
    min_width_ratio: float = 0.16
    min_height_px: int = 12
    min_y_ratio: float = 0.20
    max_y_ratio: float = 0.96
    max_center_x_offset_ratio: float = 0.36
    close_kernel_w_ratio: float = 0.055
    close_kernel_h: int = 7
    span_min_width_ratio: float = 0.24
    span_min_area_px: int = 1600
    span_min_coverage_ratio: float = 0.035
    span_max_dist_mm: float = 240.0


DEFAULT_TARGET_CONTOUR_CONFIG = TargetContourConfig()


def _clamp_ratio(value: float, fallback: float) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = float(fallback)
    return min(1.0, max(0.0, raw))


def _roi_bounds(frame_shape: tuple[int, ...], config: HoldingBrickConfig) -> tuple[int, int, int, int]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1 = int(round(width * _clamp_ratio(config.roi_x_min_ratio, DEFAULT_HOLDING_BRICK_CONFIG.roi_x_min_ratio)))
    x2 = int(round(width * _clamp_ratio(config.roi_x_max_ratio, DEFAULT_HOLDING_BRICK_CONFIG.roi_x_max_ratio)))
    y1 = int(round(height * _clamp_ratio(config.roi_y_min_ratio, DEFAULT_HOLDING_BRICK_CONFIG.roi_y_min_ratio)))
    y2 = int(round(height * _clamp_ratio(config.roi_y_max_ratio, DEFAULT_HOLDING_BRICK_CONFIG.roi_y_max_ratio)))
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    if x2 <= x1:
        x1, x2 = 0, width
    if y2 <= y1:
        y1, y2 = 0, height
    return x1, y1, x2, y2


def _top_edge_nub_holding(frame_bgr: np.ndarray, config: HoldingBrickConfig) -> dict[str, Any] | None:
    """Detect the two cropped held-brick nubs in the top edge of the frame."""
    frame_h, frame_w = frame_bgr.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return None

    strip_h = int(round(float(frame_h) * float(config.top_nub_strip_ratio)))
    strip_h = max(int(config.top_nub_strip_min_px), min(int(config.top_nub_strip_max_px), strip_h))
    strip_h = max(1, min(frame_h, strip_h))
    cut_y = max(0, min(strip_h - 1, int(config.top_nub_bridge_cut_px)))
    if strip_h - cut_y < 6:
        return None

    hsv = cv2.cvtColor(frame_bgr[:strip_h, :], cv2.COLOR_BGR2HSV)
    lower = np.array(CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
    upper = np.array(CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    drop = mask[cut_y:strip_h, :]
    col_counts = np.count_nonzero(drop, axis=0)
    min_col_count = max(3, int(round(float(drop.shape[0]) * 0.26)))
    active_cols = np.flatnonzero(col_counts >= min_col_count)
    if len(active_cols) <= 0:
        return None

    groups: list[tuple[int, int]] = []
    start = int(active_cols[0])
    prev = int(active_cols[0])
    for raw_col in active_cols[1:]:
        col = int(raw_col)
        if col > prev + 1:
            groups.append((start, prev))
            start = col
        prev = col
    groups.append((start, prev))

    components: list[dict[str, Any]] = []
    for gx1, gx2 in groups:
        span_w = int(gx2 - gx1 + 1)
        if span_w < int(config.top_nub_min_width_px):
            continue
        ys, xs = np.nonzero(drop[:, gx1:gx2 + 1])
        if len(xs) <= 0:
            continue
        x = int(gx1 + xs.min())
        y = int(cut_y + ys.min())
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area_px = int(len(xs))
        if area_px < int(config.top_nub_min_area_px):
            continue
        if w < int(config.top_nub_min_width_px) or h < int(config.top_nub_min_height_px):
            continue
        components.append(
            {
                "area_px": float(area_px),
                "bbox": (x, y, w, h),
                "center_x": float(x) + (float(w) / 2.0),
                "center_y": float(y) + (float(h) / 2.0),
            }
        )

    if len(components) < 2:
        return None

    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_score = -1.0
    min_gap_px = float(frame_w) * float(config.top_nub_min_pair_gap_ratio)
    for idx, left in enumerate(components):
        for right in components[idx + 1:]:
            gap = abs(float(right["center_x"]) - float(left["center_x"]))
            if gap < min_gap_px:
                continue
            score = gap + float(left["area_px"]) * 0.02 + float(right["area_px"]) * 0.02
            if score > best_score:
                best_score = score
                best_pair = (left, right)
    if best_pair is None:
        return None

    pair = sorted(best_pair, key=lambda row: float(row["center_x"]))
    x_min = min(int(row["bbox"][0]) for row in pair)
    y_min = min(int(row["bbox"][1]) for row in pair)
    x_max = max(int(row["bbox"][0]) + int(row["bbox"][2]) for row in pair)
    y_max = max(int(row["bbox"][1]) + int(row["bbox"][3]) for row in pair)
    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    center_y_ratio = (float(y_min) + float(bbox[3]) / 2.0) / float(max(1, frame_h))
    gap_ratio = abs(float(pair[1]["center_x"]) - float(pair[0]["center_x"])) / float(max(1, frame_w))
    checks = {
        "top_nub_pair": True,
        "top_nub_gap": gap_ratio >= float(config.top_nub_min_pair_gap_ratio),
        "top_nub_center_y": center_y_ratio <= float(config.top_nub_max_pair_center_y_ratio),
    }
    if not all(checks.values()):
        return None

    area_px = float(sum(float(row["area_px"]) for row in pair))
    return {
        "holding": True,
        "reason": "held_brick_top_nubs",
        "roi": (0, 0, frame_w, strip_h),
        "green_px": int(cv2.countNonZero(mask)),
        "coverage_ratio": float(cv2.countNonZero(mask)) / float(max(1, strip_h * frame_w)),
        "best": {
            "area_px": area_px,
            "bbox": bbox,
            "width_ratio": float(bbox[2]) / float(max(1, frame_w)),
            "height_px": int(bbox[3]),
            "center_y_ratio": center_y_ratio,
            "top_edge_nubs": True,
            "components": pair,
        },
        "checks": checks,
        "mask_regions": [row["bbox"] for row in pair],
        "candidates": components[:6],
    }


def _top_edge_green_mask_regions(frame_bgr: np.ndarray, config: HoldingBrickConfig) -> list[tuple[int, int, int, int]]:
    """Return small top-edge green regions to suppress while holding."""
    frame_h, frame_w = frame_bgr.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return []
    strip_h = int(round(float(frame_h) * float(config.top_nub_strip_ratio)))
    strip_h = max(int(config.top_nub_strip_min_px), min(int(config.top_nub_strip_max_px), strip_h))
    strip_h = max(1, min(frame_h, strip_h))

    hsv = cv2.cvtColor(frame_bgr[:strip_h, :], cv2.COLOR_BGR2HSV)
    lower = np.array(CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
    upper = np.array(CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < int(config.top_nub_min_width_px) or h < 4:
            continue
        green_px = int(cv2.countNonZero(mask[y:y + h, x:x + w]))
        if green_px < int(config.top_nub_min_area_px):
            continue
        regions.append((int(x), int(y), int(w), int(h)))
    regions.sort(key=lambda row: int(row[2]) * int(row[3]), reverse=True)
    return regions[:4]


def _top_edge_holding_from_regions(
    frame_bgr: np.ndarray,
    config: HoldingBrickConfig,
    regions: list[tuple[int, int, int, int]],
) -> dict[str, Any] | None:
    """Classify a cropped top-edge held brick even when the nubs merge."""
    if not regions:
        return None
    frame_h, frame_w = frame_bgr.shape[:2]
    top_touching = [row for row in regions if int(row[1]) <= 2]
    if not top_touching:
        return None

    x_min = min(int(row[0]) for row in top_touching)
    y_min = min(int(row[1]) for row in top_touching)
    x_max = max(int(row[0]) + int(row[2]) for row in top_touching)
    y_max = max(int(row[1]) + int(row[3]) for row in top_touching)
    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    area_px = float(sum(int(row[2]) * int(row[3]) for row in top_touching))
    width_ratio = float(bbox[2]) / float(max(1, frame_w))
    center_y_ratio = (float(bbox[1]) + float(bbox[3]) / 2.0) / float(max(1, frame_h))
    checks = {
        "top_edge_area": area_px >= float(config.top_edge_min_area_px),
        "top_edge_width": width_ratio >= float(config.top_edge_min_width_ratio),
        "top_edge_height": int(bbox[3]) >= int(config.top_edge_min_height_px),
        "top_edge_center_y": center_y_ratio <= float(config.top_nub_max_pair_center_y_ratio),
    }
    if not all(checks.values()):
        return None

    strip_h = max(int(row[1]) + int(row[3]) for row in top_touching)
    return {
        "holding": True,
        "reason": "held_brick_top_edge",
        "roi": (0, 0, frame_w, int(strip_h)),
        "green_px": int(area_px),
        "coverage_ratio": area_px / float(max(1, int(strip_h) * frame_w)),
        "best": {
            "area_px": area_px,
            "bbox": bbox,
            "width_ratio": width_ratio,
            "height_px": int(bbox[3]),
            "center_y_ratio": center_y_ratio,
            "top_edge": True,
        },
        "checks": checks,
        "mask_regions": top_touching,
        "mask_strategy": "top_edge_green_regions",
        "candidates": [{"bbox": row, "area_px": float(int(row[2]) * int(row[3]))} for row in regions],
    }


def detect_holding_brick(frame_bgr: np.ndarray | None, config: HoldingBrickConfig | None = None) -> dict[str, Any]:
    """Return held-brick sidecar classification and diagnostics."""
    cfg = config or DEFAULT_HOLDING_BRICK_CONFIG
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        return {"holding": False, "reason": "missing_frame"}

    top_mask_regions = _top_edge_green_mask_regions(frame_bgr, cfg)
    top_nub_result = _top_edge_nub_holding(frame_bgr, cfg)
    if top_nub_result is not None:
        if not top_nub_result.get("mask_regions") and top_mask_regions:
            top_nub_result["mask_regions"] = top_mask_regions
        return top_nub_result
    top_edge_result = _top_edge_holding_from_regions(frame_bgr, cfg, top_mask_regions)
    if top_edge_result is not None:
        return top_edge_result

    x1, y1, x2, y2 = _roi_bounds(frame_bgr.shape, cfg)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"holding": False, "reason": "empty_roi", "roi": (x1, y1, x2, y2)}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array(CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
    upper = np.array(CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    green_px = int(cv2.countNonZero(mask))
    roi_area = int(mask.shape[0] * mask.shape[1])
    coverage = float(green_px) / float(max(1, roi_area))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        candidates.append(
            {
                "area_px": float(area),
                "bbox": (int(x1 + bx), int(y1 + by), int(bw), int(bh)),
                "width_ratio": float(bw) / float(max(1, frame_bgr.shape[1])),
                "height_px": int(bh),
                "center_y_ratio": float(y1 + by + (bh / 2.0)) / float(max(1, frame_bgr.shape[0])),
            }
        )
    candidates.sort(key=lambda row: float(row["area_px"]), reverse=True)
    best = candidates[0] if candidates else None

    if best is None:
        return {
            "holding": False,
            "reason": "no_green_contour",
            "roi": (x1, y1, x2, y2),
            "green_px": green_px,
            "coverage_ratio": coverage,
            "top_edge": {"holding": False, "reason": "no_top_nub_pair"},
            "candidates": [],
        }

    checks = {
        "area": float(best["area_px"]) >= float(cfg.min_area_px),
        "width": float(best["width_ratio"]) >= float(cfg.min_width_ratio),
        "height": int(best["height_px"]) >= int(cfg.min_height_px),
        "coverage": coverage >= float(cfg.min_coverage_ratio),
        "center_y": float(best.get("center_y_ratio", 1.0)) <= float(cfg.max_center_y_ratio),
    }
    holding = all(checks.values())
    result = {
        "holding": bool(holding),
        "reason": "held_brick_detected" if holding else "below_threshold",
        "roi": (x1, y1, x2, y2),
        "green_px": green_px,
        "coverage_ratio": coverage,
        "best": best,
        "checks": checks,
        "top_edge": {"holding": False, "reason": "no_top_nub_pair"},
        "candidates": candidates[:5],
    }
    if holding and top_mask_regions:
        result["mask_regions"] = top_mask_regions
        result["mask_strategy"] = "top_edge_green_regions"
    return result


def mask_held_brick_for_target_frame(frame_bgr: np.ndarray | None, result: dict[str, Any] | None) -> np.ndarray | None:
    """Mask the held brick as a hard exclusion zone for target detection."""
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        return None
    masked = frame_bgr.copy()
    height, width = masked.shape[:2]
    mask_regions = result.get("mask_regions") if isinstance(result, dict) else None
    if isinstance(mask_regions, list) and mask_regions:
        for region in mask_regions:
            if not (isinstance(region, (tuple, list)) and len(region) == 4):
                continue
            x, y, w, h = [int(v) for v in region]
            pad_x = max(5, int(round(float(w) * 0.35)))
            pad_y = max(4, int(round(float(h) * 0.25)))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            if x2 > x1 and y2 > y1:
                masked[y1:y2, x1:x2] = 0
        return masked

    best = result.get("best") if isinstance(result, dict) else None
    bbox = best.get("bbox") if isinstance(best, dict) else None
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        return frame_bgr.copy()

    x, y, w, h = [int(v) for v in bbox]
    pad_x = max(16, int(round(float(w) * 0.05)))
    pad_y = max(14, int(round(float(h) * 0.10)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    # In the holding game the held-brick bbox often overlaps the target stack.
    # Masking the full bbox erases the lower green brick we actually need to
    # measure, so suppress only the upper held-brick area.
    y2 = min(height, y + max(1, int(round(float(h) * HELD_BRICK_MASK_BOTTOM_FRACTION))))
    if x2 <= x1 or y2 <= y1:
        return masked

    masked[y1:y2, x1:x2] = 0
    return masked


def _camera_center_x(detector: Any, frame_w: int) -> float:
    value = getattr(detector, "_camera_cx_px", None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    offset = getattr(detector, "camera_center_offset_px", 0.0)
    try:
        offset = float(offset)
    except (TypeError, ValueError):
        offset = 0.0
    return float(frame_w) / 2.0 + offset


def _camera_center_y(detector: Any, frame_h: int) -> float:
    value = getattr(detector, "_camera_cy_px", None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return float(frame_h) / 2.0


def _camera_focal_x(detector: Any) -> float:
    for name in ("_camera_fx_px", "focal_px"):
        value = getattr(detector, name, None)
        if value is None:
            continue
        try:
            return max(1e-6, float(value))
        except (TypeError, ValueError):
            continue
    return 500.0


def _camera_focal_y(detector: Any) -> float:
    for name in ("_camera_fy_px", "focal_px"):
        value = getattr(detector, name, None)
        if value is None:
            continue
        try:
            return max(1e-6, float(value))
        except (TypeError, ValueError):
            continue
    return 500.0


def _as_int_odd(value: float, minimum: int = 3) -> int:
    raw = max(int(minimum), int(round(value)))
    return raw + 1 if raw % 2 == 0 else raw


def _target_contours_from_mask(mask: np.ndarray, min_y: int, max_y: int) -> list[tuple[float, tuple[int, int, int, int], Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows: list[tuple[float, tuple[int, int, int, int], Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cy = y + h / 2.0
        if cy < min_y or cy > max_y:
            continue
        rows.append((area, (int(x), int(y), int(w), int(h)), contour))
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows


def _target_span_from_mask(
    mask: np.ndarray,
    min_y: int,
    max_y: int,
    detector: Any | None,
    config: TargetContourConfig,
) -> dict[str, Any] | None:
    """Estimate target distance from the full green horizontal span.

    The held-brick mask can split the lower brick into disconnected pieces.
    Contour area alone may then latch onto a tiny patch. For the holding game,
    the useful distance cue is the full visible width of the target stack.
    """
    frame_h, frame_w = mask.shape[:2]
    y1 = max(0, int(min_y))
    y2 = min(frame_h, int(max_y))
    if y2 <= y1:
        return None
    region = mask[y1:y2, :]
    ys, xs = np.nonzero(region)
    if len(xs) <= 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min()) + y1
    y_max = int(ys.max()) + y1
    w = int(x_max - x_min + 1)
    h = int(y_max - y_min + 1)
    if w <= 0 or h <= 0:
        return None
    area = int(len(xs))

    width_ratio = float(w) / float(max(1, frame_w))
    coverage = float(area) / float(max(1, w * h))
    camera_center_x = _camera_center_x(detector, frame_w)
    center_x = float(x_min) + float(w) / 2.0
    max_center_offset_px = frame_w * _clamp_ratio(
        config.max_center_x_offset_ratio,
        DEFAULT_TARGET_CONTOUR_CONFIG.max_center_x_offset_ratio,
    )
    focal_x = _camera_focal_x(detector)
    focal_y = _camera_focal_y(detector)
    dist_mm = (float(BRICK_WIDTH_MM) * focal_x) / float(max(1, w))
    checks = {
        "area": float(area) >= float(config.span_min_area_px),
        "width": width_ratio >= float(config.span_min_width_ratio),
        "height": int(h) >= int(config.min_height_px),
        "coverage": coverage >= float(config.span_min_coverage_ratio),
        "center_x": abs(float(center_x) - float(camera_center_x)) <= float(max_center_offset_px),
        "dist": float(dist_mm) <= float(config.span_max_dist_mm),
    }
    if not all(checks.values()):
        return {
            "found": False,
            "reason": "span_rejected",
            "bbox": (x_min, y_min, w, h),
            "checks": checks,
            "area_px": float(area),
            "coverage_ratio": float(coverage),
            "width_ratio": float(width_ratio),
            "dist_mm": float(dist_mm),
        }

    center_y = float(y_min) + float(h) / 2.0
    offset_x_mm = ((center_x - _camera_center_x(detector, frame_w)) * dist_mm) / focal_x
    y_mm = ((center_y - _camera_center_y(detector, frame_h)) * dist_mm) / focal_y
    confidence = max(60.0, min(98.0, 70.0 + (coverage * 35.0) + min(8.0, area / 3500.0)))
    return {
        "found": True,
        "reason": "masked_target_width_span",
        "bbox": (x_min, y_min, w, h),
        "draw_bbox": _shrinkwrap_bbox_from_mask(mask, x_min, y_min, x_max, y_max),
        "area_px": float(area),
        "coverage_ratio": float(coverage),
        "width_ratio": float(width_ratio),
        "dist_mm": float(dist_mm),
        "x_mm": float(offset_x_mm),
        "y_mm": float(y_mm),
        "confidence_pct": float(confidence),
        "checks": checks,
    }


def _dense_axis_bounds(
    counts: np.ndarray,
    *,
    min_count: int,
    peak_fraction: float,
) -> tuple[int, int] | None:
    if counts is None or len(counts) <= 0:
        return None
    peak = int(np.max(counts))
    if peak <= 0:
        return None
    threshold = max(int(min_count), int(round(float(peak) * float(peak_fraction))))
    indices = np.flatnonzero(counts >= threshold)
    if len(indices) <= 0:
        return None
    return int(indices[0]), int(indices[-1])


def _shrinkwrap_bbox_from_mask(
    mask: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> tuple[int, int, int, int]:
    """Return a display box around dense green pixels, not sparse floor noise."""
    frame_h, frame_w = mask.shape[:2]
    x1 = max(0, min(int(x_min), frame_w - 1))
    x2 = max(0, min(int(x_max), frame_w - 1))
    y1 = max(0, min(int(y_min), frame_h - 1))
    y2 = max(0, min(int(y_max), frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1))

    roi = mask[y1:y2 + 1, x1:x2 + 1]
    row_counts = np.count_nonzero(roi, axis=1)
    col_counts = np.count_nonzero(roi, axis=0)
    row_bounds = _dense_axis_bounds(row_counts, min_count=12, peak_fraction=0.18)
    col_bounds = _dense_axis_bounds(col_counts, min_count=10, peak_fraction=0.14)
    if row_bounds is None or col_bounds is None:
        return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    tight_x1 = x1 + int(col_bounds[0])
    tight_x2 = x1 + int(col_bounds[1])
    tight_y1 = y1 + int(row_bounds[0])
    tight_y2 = y1 + int(row_bounds[1])
    pad_x = 2
    pad_y = 2
    tight_x1 = max(0, tight_x1 - pad_x)
    tight_x2 = min(frame_w - 1, tight_x2 + pad_x)
    tight_y1 = max(0, tight_y1 - pad_y)
    tight_y2 = min(frame_h - 1, tight_y2 + pad_y)
    return (
        int(tight_x1),
        int(tight_y1),
        max(1, int(tight_x2 - tight_x1 + 1)),
        max(1, int(tight_y2 - tight_y1 + 1)),
    )


def _draw_regions_from_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None,
) -> list[dict[str, Any]]:
    """Return tight display regions for actual green components."""
    if mask is None or not hasattr(mask, "shape") or len(mask.shape) < 2:
        return []
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        return []
    frame_h, frame_w = mask.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, min(x, frame_w - 1))
    y1 = max(0, min(y, frame_h - 1))
    x2 = max(0, min(x + max(1, w), frame_w))
    y2 = max(0, min(y + max(1, h), frame_h))
    if x2 <= x1 or y2 <= y1:
        return []
    roi = mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 80.0:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 5 or bh < 5:
            continue
        contour = contour + np.array([[[x1, y1]]], dtype=contour.dtype)
        tight_bbox = (int(x1 + bx), int(y1 + by), int(bw), int(bh))
        regions.append(
            {
                "area_px": float(area),
                "bbox": _square_bbox(tight_bbox, frame_w, frame_h),
                "tight_bbox": tight_bbox,
                "contour": contour,
            }
        )
    regions.sort(key=lambda row: float(row.get("area_px", 0.0)), reverse=True)
    return regions[:8]


def _square_bbox(
    bbox: tuple[int, int, int, int] | list[int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in bbox]
    side = max(1, int(max(w, h)))
    side = min(side, int(frame_w), int(frame_h))
    cx = float(x) + float(w) / 2.0
    cy = float(y) + float(h) / 2.0
    x1 = int(round(cx - float(side) / 2.0))
    y1 = int(round(cy - float(side) / 2.0))
    x1 = max(0, min(x1, int(frame_w) - side))
    y1 = max(0, min(y1, int(frame_h) - side))
    return (int(x1), int(y1), int(side), int(side))


def detect_masked_target_brick_contour(
    frame_bgr: np.ndarray | None,
    detector: Any | None = None,
    config: TargetContourConfig | None = None,
) -> dict[str, Any]:
    """Find the full lower target brick as a broad green rectangle.

    This is intentionally color/shape based and is used only after the held
    brick has been masked. It favors the whole visible brick face over a small
    central model box so close-up held-brick games keep the true left/right
    edges.
    """
    cfg = config or DEFAULT_TARGET_CONTOUR_CONFIG
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        return {"found": False, "reason": "missing_frame"}

    frame_h, frame_w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(CYAN_HSV_WIDE_LOWER, dtype=np.uint8)
    upper = np.array(CYAN_HSV_WIDE_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    draw_lower = np.array(CYAN_HSV_BALANCED_LOWER, dtype=np.uint8)
    draw_upper = np.array(CYAN_HSV_BALANCED_UPPER, dtype=np.uint8)
    draw_mask = cv2.inRange(hsv, draw_lower, draw_upper)

    min_y = int(round(frame_h * _clamp_ratio(cfg.min_y_ratio, DEFAULT_TARGET_CONTOUR_CONFIG.min_y_ratio)))
    max_y = int(round(frame_h * _clamp_ratio(cfg.max_y_ratio, DEFAULT_TARGET_CONTOUR_CONFIG.max_y_ratio)))
    focus = np.zeros_like(mask)
    focus[max(0, min_y):min(frame_h, max_y), :] = mask[max(0, min_y):min(frame_h, max_y), :]
    draw_focus = np.zeros_like(draw_mask)
    draw_focus[max(0, min_y):min(frame_h, max_y), :] = draw_mask[max(0, min_y):min(frame_h, max_y), :]

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones(
        (
            max(1, int(cfg.close_kernel_h)),
            _as_int_odd(float(frame_w) * float(cfg.close_kernel_w_ratio), minimum=9),
        ),
        np.uint8,
    )
    focus = cv2.morphologyEx(focus, cv2.MORPH_OPEN, open_kernel)
    draw_focus = cv2.morphologyEx(draw_focus, cv2.MORPH_OPEN, open_kernel)
    focus = cv2.morphologyEx(focus, cv2.MORPH_CLOSE, close_kernel)

    span_result = _target_span_from_mask(focus, min_y, max_y, detector, cfg)
    if isinstance(span_result, dict) and bool(span_result.get("found")):
        span_result = dict(span_result)
        bbox = span_result.get("bbox")
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            span_result["draw_bbox"] = _shrinkwrap_bbox_from_mask(
                draw_focus,
                x,
                y,
                x + max(1, w) - 1,
                y + max(1, h) - 1,
            )
            span_result["draw_regions"] = _draw_regions_from_mask(draw_focus, bbox)
        span_result["mask"] = focus
        span_result["draw_mask"] = draw_focus
        span_result["candidates"] = [dict(span_result)]
        return span_result

    candidates = _target_contours_from_mask(focus, min_y, max_y)
    accepted: list[dict[str, Any]] = []
    camera_center_x = _camera_center_x(detector, frame_w)
    max_center_offset_px = frame_w * _clamp_ratio(
        cfg.max_center_x_offset_ratio,
        DEFAULT_TARGET_CONTOUR_CONFIG.max_center_x_offset_ratio,
    )
    for area, bbox, _contour in candidates:
        x, y, w, h = bbox
        center_x = float(x) + float(w) / 2.0
        checks = {
            "area": area >= float(cfg.min_area_px),
            "width": (float(w) / float(max(1, frame_w))) >= float(cfg.min_width_ratio),
            "height": int(h) >= int(cfg.min_height_px),
            "center_x": abs(float(center_x) - float(camera_center_x)) <= float(max_center_offset_px),
        }
        accepted.append({"area_px": area, "bbox": bbox, "checks": checks})
        if not all(checks.values()):
            continue

        focal_x = _camera_focal_x(detector)
        focal_y = _camera_focal_y(detector)
        dist_mm = (float(BRICK_WIDTH_MM) * focal_x) / float(max(1, w))
        center_y = float(y) + float(h) / 2.0
        offset_x_mm = ((center_x - _camera_center_x(detector, frame_w)) * dist_mm) / focal_x
        y_mm = ((center_y - _camera_center_y(detector, frame_h)) * dist_mm) / focal_y
        coverage = float(cv2.countNonZero(focus[y:y + h, x:x + w])) / float(max(1, w * h))
        confidence = max(55.0, min(98.0, 65.0 + (coverage * 25.0) + min(8.0, area / 2500.0)))
        return {
            "found": True,
            "reason": "masked_target_contour",
            "bbox": bbox,
            "area_px": area,
            "coverage_ratio": coverage,
            "dist_mm": float(dist_mm),
            "x_mm": float(offset_x_mm),
            "y_mm": float(y_mm),
            "confidence_pct": float(confidence),
            "mask": focus,
            "candidates": accepted[:5],
        }

    if isinstance(span_result, dict):
        accepted.insert(0, {"span": span_result})
    return {"found": False, "reason": "no_plausible_target_contour", "candidates": accepted[:5]}


def contour_target_result_tuple(contour_result: dict[str, Any]) -> tuple:
    """Return the standard BrickDetector result tuple for a contour result."""
    if not bool(contour_result.get("found")):
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
    return (
        True,
        0.0,
        float(contour_result.get("dist_mm", 0.0)),
        float(contour_result.get("x_mm", 0.0)),
        float(contour_result.get("confidence_pct", 0.0)),
        float(contour_result.get("y_mm", 0.0)),
        False,
        False,
    )


def draw_masked_target_contour(frame_bgr: np.ndarray, contour_result: dict[str, Any]) -> np.ndarray:
    out = frame_bgr.copy()
    regions = contour_result.get("draw_regions")
    if isinstance(regions, list) and regions:
        label_anchor = None
        for region in regions:
            if not isinstance(region, dict):
                continue
            contour = region.get("contour")
            bbox = region.get("bbox")
            if contour is not None:
                try:
                    cv2.drawContours(out, [contour], -1, (255, 0, 255), 2)
                except Exception:
                    pass
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 1)
                if label_anchor is None:
                    label_anchor = (max(8, x), max(22, y - 8))
        if label_anchor is not None:
            label = (
                f"held-target dist={float(contour_result.get('dist_mm', 0.0)):.1f} "
                f"x={float(contour_result.get('x_mm', 0.0)):+.1f} "
                f"y={float(contour_result.get('y_mm', 0.0)):+.1f}"
            )
            cv2.putText(out, label, label_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
            return out

    bbox = contour_result.get("draw_bbox") or contour_result.get("bbox")
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, w, h = [int(v) for v in _square_bbox(bbox, out.shape[1], out.shape[0])]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
        label = (
            f"held-target dist={float(contour_result.get('dist_mm', 0.0)):.1f} "
            f"x={float(contour_result.get('x_mm', 0.0)):+.1f} "
            f"y={float(contour_result.get('y_mm', 0.0)):+.1f}"
        )
        cv2.putText(out, label, (max(8, x), max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    return out


def draw_holding_debug(frame_bgr: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    out = frame_bgr.copy()
    roi = result.get("roi")
    if isinstance(roi, (tuple, list)) and len(roi) == 4:
        x1, y1, x2, y2 = [int(v) for v in roi]
        color = (0, 255, 0) if bool(result.get("holding")) else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    best = result.get("best")
    if isinstance(best, dict) and isinstance(best.get("bbox"), (tuple, list)):
        x, y, w, h = [int(v) for v in best["bbox"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 180, 0), 2)
    mask_regions = result.get("mask_regions")
    if isinstance(mask_regions, list):
        for region in mask_regions:
            if isinstance(region, (tuple, list)) and len(region) == 4:
                x, y, w, h = [int(v) for v in region]
                cv2.rectangle(out, (x, y), (x + w, y + h), (255, 180, 0), 1)
    label = f"holding={bool(result.get('holding'))} {result.get('reason', '')}"
    cv2.putText(out, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out
