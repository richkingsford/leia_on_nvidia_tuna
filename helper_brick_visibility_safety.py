#!/usr/bin/env python3
"""Shared hard guard: never send motion unless a brick is confidently visible."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


log = logging.getLogger("brick_visibility_safety")

ROBOT_MODEL_FILE = Path(__file__).resolve().parent / "world_model_robot.json"
CONFIG_KEY = "brick_visibility_motion_safety"
DEFAULT_MIN_CONFIDENCE_PCT = 75.0


def _coerce_float(value: Any, fallback: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = float(fallback)
    if minimum is not None:
        coerced = max(float(minimum), float(coerced))
    if maximum is not None:
        coerced = min(float(maximum), float(coerced))
    return float(coerced)


def load_brick_visibility_motion_safety_config(path: Path | None = None) -> dict:
    """Load the confidence threshold for motion gating from world_model_robot.json."""
    model_path = path if isinstance(path, Path) else ROBOT_MODEL_FILE
    cfg = {"min_confidence_pct": float(DEFAULT_MIN_CONFIDENCE_PCT)}
    try:
        payload = json.loads(model_path.read_text())
    except Exception:
        return cfg
    raw = payload.get(CONFIG_KEY) if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return cfg
    cfg["min_confidence_pct"] = _coerce_float(
        raw.get("min_confidence_pct"),
        DEFAULT_MIN_CONFIDENCE_PCT,
        minimum=0.0,
        maximum=100.0,
    )
    return cfg


def _conf_from_mapping(reading: dict) -> float | None:
    for key in ("conf", "confidence", "confidence_pct"):
        if key not in reading:
            continue
        try:
            return float(reading.get(key))
        except (TypeError, ValueError):
            return None
    return None


def brick_motion_measurement_from_result(result, *, min_confidence_pct: float | None = None) -> dict:
    """Normalize a detector result into the motion-safety visibility semantics."""
    threshold = (
        _coerce_float(min_confidence_pct, DEFAULT_MIN_CONFIDENCE_PCT, minimum=0.0, maximum=100.0)
        if min_confidence_pct is not None
        else float(load_brick_visibility_motion_safety_config().get("min_confidence_pct", DEFAULT_MIN_CONFIDENCE_PCT))
    )
    measurement = {
        "visible": False,
        "confident": False,
        "conf": None,
        "min_confidence_pct": float(threshold),
        "reason": "not_visible",
        "result": result,
    }

    if isinstance(result, dict):
        visible = bool(result.get("visible"))
        conf = _conf_from_mapping(result)
        measurement.update(
            {
                "visible": bool(visible),
                "conf": conf,
                "dist_mm": result.get("dist_mm", result.get("dist")),
                "x_mm": result.get("x_mm", result.get("x_axis", result.get("offset_x"))),
                "y_mm": result.get("y_mm", result.get("y_axis", result.get("offset_y"))),
                "brick_above": result.get("brick_above"),
                "brick_below": result.get("brick_below"),
            }
        )
    elif isinstance(result, tuple) and len(result) >= 5 and bool(result[0]):
        try:
            conf = float(result[4])
            measurement.update(
                {
                    "visible": True,
                    "dist_mm": float(result[2]),
                    "x_mm": float(result[3]),
                    "y_mm": float(result[5]) if len(result) >= 6 else None,
                    "conf": conf,
                    "brick_above": bool(result[6]) if len(result) >= 7 else None,
                    "brick_below": bool(result[7]) if len(result) >= 8 else None,
                }
            )
        except (TypeError, ValueError):
            measurement["reason"] = "invalid_measurement"
            return measurement
    else:
        return measurement

    if not bool(measurement.get("visible")):
        measurement["reason"] = "not_visible"
        return measurement
    conf_val = measurement.get("conf")
    if conf_val is None:
        measurement["reason"] = "confidence_missing"
        return measurement
    if float(conf_val) < float(threshold):
        measurement["reason"] = "low_confidence"
        return measurement

    measurement["confident"] = True
    measurement["reason"] = "confident_visible"
    return measurement


def brick_motion_allowed(reading: dict | None, *, min_confidence_pct: float | None = None) -> bool:
    if not isinstance(reading, dict):
        return False
    threshold = (
        _coerce_float(min_confidence_pct, DEFAULT_MIN_CONFIDENCE_PCT, minimum=0.0, maximum=100.0)
        if min_confidence_pct is not None
        else _coerce_float(
            reading.get("min_confidence_pct"),
            load_brick_visibility_motion_safety_config().get("min_confidence_pct", DEFAULT_MIN_CONFIDENCE_PCT),
            minimum=0.0,
            maximum=100.0,
        )
    )
    conf = _conf_from_mapping(reading)
    return bool(reading.get("visible")) and conf is not None and float(conf) >= float(threshold)


def _stop_robot(robot) -> None:
    stop = getattr(robot, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            pass


def _motion_cmd(cmd: str | None, pwm: Any = None) -> bool:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key in {"", "s", "stop"}:
        return False
    if pwm is not None:
        try:
            return int(round(float(pwm))) > 0
        except (TypeError, ValueError):
            return True
    return True


def _custom_actions_move(action_specs) -> bool:
    for row in action_specs or []:
        if not isinstance(row, dict):
            continue
        if _motion_cmd(row.get("action"), row.get("pwm")):
            return True
    return False


def block_reason(reading: dict | None) -> str:
    if not isinstance(reading, dict):
        return "missing_visibility_reading"
    reason = str(reading.get("reason") or "").strip()
    return reason or "not_confidently_visible"


def require_confident_brick_for_motion(robot, reading: dict | None, *, context: str = "") -> bool:
    """Return true only when the current reading allows non-stop motion."""
    if brick_motion_allowed(reading):
        return True
    _stop_robot(robot)
    context_text = f" context={context}" if context else ""
    log.warning("[BRICK SAFETY] Motion blocked:%s reason=%s", context_text, block_reason(reading))
    return False


def guarded_send_command_pwm(robot, cmd, pwm, *, duration_ms=None, reading: dict | None, context: str = ""):
    if _motion_cmd(cmd, pwm) and not require_confident_brick_for_motion(robot, reading, context=context):
        return {"blocked": True, "reason": block_reason(reading), "cmd": str(cmd or "").strip().lower()}
    return robot.send_command_pwm(cmd, pwm, duration_ms=duration_ms)


def guarded_send_custom_actions_pwm(robot, cmd, action_specs, *, duration_ms=None, reading: dict | None, context: str = ""):
    if _custom_actions_move(action_specs) and not require_confident_brick_for_motion(robot, reading, context=context):
        return {"blocked": True, "reason": block_reason(reading), "cmd": str(cmd or "").strip().lower()}
    return robot.send_custom_actions_pwm(cmd, action_specs, duration_ms=duration_ms)
