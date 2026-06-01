#!/usr/bin/env python3
"""Holding-game distance calibration for the masked brick-stack target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROBOT_MODEL_FILE = Path(__file__).resolve().parent / "world_model_robot.json"
CONFIG_KEY = "holding_target_distance_calibration"


def _coerce_float(value: Any, fallback: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None if fallback is None else float(fallback)


def _coerce_point(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, dict):
        return None
    reported = _coerce_float(raw.get("reported_mm"))
    true = _coerce_float(raw.get("true_mm"))
    if reported is None or true is None:
        return None
    return float(reported), float(true)


def load_holding_distance_calibration_config(path: Path | None = None) -> dict:
    """Load holding-only stack-width distance calibration from world_model_robot.json."""
    model_path = path if isinstance(path, Path) else ROBOT_MODEL_FILE
    try:
        payload = json.loads(model_path.read_text())
    except Exception:
        return {"enabled": False, "points": []}
    follow = payload.get("follow_the_brick") if isinstance(payload, dict) else {}
    raw = follow.get(CONFIG_KEY) if isinstance(follow, dict) else {}
    if not isinstance(raw, dict):
        return {"enabled": False, "points": []}
    points = []
    for item in raw.get("points", []):
        point = _coerce_point(item)
        if point is not None:
            points.append(point)
    deduped = {}
    for reported, true in points:
        deduped[float(reported)] = float(true)
    points = sorted(deduped.items())
    return {
        "enabled": bool(raw.get("enabled", False)) and len(points) >= 2,
        "points": points,
    }


def calibrate_holding_distance_mm(
    reported_mm: Any,
    config: dict | None = None,
) -> tuple[float | None, bool]:
    """Return calibrated real-world millimeters using piecewise linear interpolation."""
    reported = _coerce_float(reported_mm)
    if reported is None:
        return None, False
    cfg = config if isinstance(config, dict) else load_holding_distance_calibration_config()
    points = cfg.get("points") if isinstance(cfg.get("points"), list) else []
    if not bool(cfg.get("enabled")) or len(points) < 2:
        return float(reported), False
    clean = []
    for point in points:
        if isinstance(point, dict):
            x = _coerce_float(point.get("reported_mm"))
            y = _coerce_float(point.get("true_mm"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x = _coerce_float(point[0])
            y = _coerce_float(point[1])
        else:
            continue
        if x is not None and y is not None:
            clean.append((float(x), float(y)))
    clean = sorted(clean)
    if len(clean) < 2:
        return float(reported), False
    left, right = clean[0], clean[1]
    if reported >= clean[-1][0]:
        left, right = clean[-2], clean[-1]
    else:
        for idx in range(len(clean) - 1):
            a, b = clean[idx], clean[idx + 1]
            if reported <= b[0]:
                left, right = a, b
                break
    dx = float(right[0]) - float(left[0])
    if abs(dx) < 1e-6:
        return float(reported), False
    fraction = (float(reported) - float(left[0])) / dx
    calibrated = float(left[1]) + fraction * (float(right[1]) - float(left[1]))
    return float(calibrated), True


def apply_holding_distance_calibration_to_reading(
    reading: dict,
    config: dict | None = None,
) -> dict:
    """Return a reading with dist_mm calibrated, preserving the uncalibrated value."""
    if not isinstance(reading, dict):
        return reading
    if reading.get("dist_mm") is None:
        return reading
    calibrated, used = calibrate_holding_distance_mm(reading.get("dist_mm"), config=config)
    if calibrated is None or not used:
        return reading
    out = dict(reading)
    raw_dist = out.get("dist_mm")
    out["uncalibrated_dist_mm"] = raw_dist
    out["dist_mm"] = float(calibrated)
    out["holding_distance_calibrated"] = True
    result = list(out.get("result") or [])
    if len(result) >= 3:
        result[2] = float(calibrated)
        out["result"] = result
    return out


def apply_holding_distance_calibration_to_result(
    result,
    config: dict | None = None,
) -> tuple[object, float | None, float | None, bool]:
    """Return a detector result tuple with calibrated distance in slot 2."""
    if not isinstance(result, tuple) or len(result) < 3 or not bool(result[0]):
        return result, None, None, False
    raw_dist = _coerce_float(result[2])
    calibrated, used = calibrate_holding_distance_mm(raw_dist, config=config)
    if raw_dist is None or calibrated is None or not used:
        return result, raw_dist, calibrated, False
    out = list(result)
    out[2] = float(calibrated)
    return tuple(out), float(raw_dist), float(calibrated), True
