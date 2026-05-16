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

from helper_brick_detector_yolo import CYAN_HSV_WIDE_LOWER, CYAN_HSV_WIDE_UPPER


@dataclass(frozen=True)
class HoldingBrickConfig:
    roi_y_min_ratio: float = 0.06
    roi_y_max_ratio: float = 0.46
    roi_x_min_ratio: float = 0.08
    roi_x_max_ratio: float = 0.92
    min_area_px: int = 1800
    min_width_ratio: float = 0.24
    min_height_px: int = 24
    min_coverage_ratio: float = 0.035


DEFAULT_HOLDING_BRICK_CONFIG = HoldingBrickConfig()


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
