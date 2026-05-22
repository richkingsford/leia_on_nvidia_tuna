"""Sidecar held-brick detector.

This helper intentionally does not participate in target brick selection.  It
only inspects a fixed upper/prong ROI in an already-captured BGR frame and
reports whether a large green/cyan region is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from helper_brick_detector_yolo import BRICK_WIDTH_MM, CYAN_HSV_WIDE_LOWER, CYAN_HSV_WIDE_UPPER


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


DEFAULT_HOLDING_BRICK_CONFIG = HoldingBrickConfig()

HELD_BRICK_MASK_CENTER_LIFT_PX = 30
HELD_BRICK_MASK_CENTER_WIDTH_RATIO = 0.80
HELD_BRICK_MASK_BOTTOM_LIMIT_PX = 60


@dataclass(frozen=True)
class TargetContourConfig:
    min_area_px: int = 900
    min_width_ratio: float = 0.10
    min_height_px: int = 12
    min_y_ratio: float = 0.30
    max_y_ratio: float = 0.96
    max_center_x_offset_ratio: float = 0.36
    close_kernel_w_ratio: float = 0.055
    close_kernel_h: int = 7


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


def detect_holding_brick(frame_bgr: np.ndarray | None, config: HoldingBrickConfig | None = None) -> dict[str, Any]:
    """Return held-brick sidecar classification and diagnostics."""
    cfg = config or DEFAULT_HOLDING_BRICK_CONFIG
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        return {"holding": False, "reason": "missing_frame"}

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
    return {
        "holding": bool(holding),
        "reason": "held_brick_detected" if holding else "below_threshold",
        "roi": (x1, y1, x2, y2),
        "green_px": green_px,
        "coverage_ratio": coverage,
        "best": best,
        "checks": checks,
        "candidates": candidates[:5],
    }


def mask_held_brick_for_target_frame(frame_bgr: np.ndarray | None, result: dict[str, Any] | None) -> np.ndarray | None:
    """Mask the held brick as a hard exclusion zone for target detection."""
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        return None
    best = result.get("best") if isinstance(result, dict) else None
    bbox = best.get("bbox") if isinstance(best, dict) else None
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        return frame_bgr.copy()

    masked = frame_bgr.copy()
    height, width = masked.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    pad_x = 12
    pad_y = 10
    bottom_extra_px = 45
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, int(HELD_BRICK_MASK_BOTTOM_LIMIT_PX), y + h + pad_y + bottom_extra_px)
    if x2 <= x1 or y2 <= y1:
        return masked

    mask_w = max(1, x2 - x1)
    side_ratio = max(0.0, min(0.5, (1.0 - float(HELD_BRICK_MASK_CENTER_WIDTH_RATIO)) / 2.0))
    inner_x1 = min(x2, max(x1, x1 + int(round(mask_w * side_ratio))))
    inner_x2 = max(x1, min(x2, x2 - int(round(mask_w * side_ratio))))
    lifted_y2 = max(y1, y2 - int(HELD_BRICK_MASK_CENTER_LIFT_PX))

    masked[y1:lifted_y2, x1:x2] = 0
    if lifted_y2 < y2:
        masked[lifted_y2:y2, x1:inner_x1] = 0
        masked[lifted_y2:y2, inner_x2:x2] = 0
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

    min_y = int(round(frame_h * _clamp_ratio(cfg.min_y_ratio, DEFAULT_TARGET_CONTOUR_CONFIG.min_y_ratio)))
    max_y = int(round(frame_h * _clamp_ratio(cfg.max_y_ratio, DEFAULT_TARGET_CONTOUR_CONFIG.max_y_ratio)))
    focus = np.zeros_like(mask)
    focus[max(0, min_y):min(frame_h, max_y), :] = mask[max(0, min_y):min(frame_h, max_y), :]

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones(
        (
            max(1, int(cfg.close_kernel_h)),
            _as_int_odd(float(frame_w) * float(cfg.close_kernel_w_ratio), minimum=9),
        ),
        np.uint8,
    )
    focus = cv2.morphologyEx(focus, cv2.MORPH_OPEN, open_kernel)
    focus = cv2.morphologyEx(focus, cv2.MORPH_CLOSE, close_kernel)

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
    bbox = contour_result.get("bbox")
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, w, h = [int(v) for v in bbox]
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
    label = f"holding={bool(result.get('holding'))} {result.get('reason', '')}"
    cv2.putText(out, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out
