#!/usr/bin/env python3
"""Follow-the-brick: repeatedly tracks a brick to the happy place.

The happy place is the ALIGN_BRICK x/dist target. After a win, reset is one
backward-turn act in a random direction, then a measured pause before following
continues.

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from helper_brick_detector_yolo import (
    BrickDetector,
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
)
from helper_brick_visibility_safety import (
    brick_motion_measurement_from_result,
    guarded_send_command_pwm,
    guarded_send_custom_actions_pwm,
)
from helper_robot_control import Robot
import telemetry_robot as _telemetry_robot

# ── tuning ────────────────────────────────────────────────────────────────────
# Success-gate values from world_model_process.json → steps → ALIGN_BRICK → success_gates
TARGET_DIST_MM = 63.882  # dist.target
DIST_TOL_MM    = 20.0    # dist.tol
X_TOL_MM       = 9.0     # xAxis_offset_abs.tol  (centred at x=0)
Y_TARGET_MM    = -4.25129472 # signed y target from confident brick reading
Y_TOL_MM       = 9.0     # yAxis_offset_abs.tol from ALIGN_BRICK

SPEED_SCORE        = 1    # slowest motor speed score
PULSE_MS       = 200     # motor pulse duration — long enough for slow motor to engage
LOOP_S         = 0.05    # control loop interval (20 Hz)
WARMUP_READS   = 16      # reads to warm the camera pipeline before capture
PREGAME_VISIBILITY_TIMEOUT_S = 4.0
PREGAME_VISIBILITY_SAMPLE_S = 0.12

RESET_REVERSE_TURN_PULSE_MS = 150    # Slower movement (was 50ms)
RESET_REVERSE_TURN_TIMEOUT_S = 1.5
RESET_REVERSE_TURN_SETTLE_S = 0.06
RESET_POST_PAUSE_S = 2.0
RESET_X_OFFSET_MIN_MM = 12.0
RESET_X_OFFSET_MAX_MM = 18.0
RESET_TARGET_ABS_X_MM = 15.0
RESET_X_OFFSET_CONFIRM_FRAMES = 2
RESET_DIST_TARGET_MM = TARGET_DIST_MM * 1.75
RESET_DIST_TOL_MM = 9.0
RESET_Y_TARGET_MM = -5.0
RESET_SHARP_FINISH_MS = 300
RESET_MAST_UP_MIN_MS = 400
RESET_MAST_UP_MAX_MS = 700
RESET_MAST_UP_PWM = 255
RESET_MAST_UP_SETTLE_S = 0.1
DEFAULT_RESET_ARC_ALGORITHM_POINTS = (
    {"x_gap_mm": 0.0, "slower_pwm": 103, "faster_pwm": 111},
    {"x_gap_mm": 20.0, "slower_pwm": 104, "faster_pwm": 125},
    {"x_gap_mm": 41.28, "slower_pwm": 106, "faster_pwm": 146},
    {"x_gap_mm": 60.0, "slower_pwm": 108, "faster_pwm": 162},
)
DEFAULT_MOTION_POWER_SCALE = 1.0
DEFAULT_NORMAL_SPEED_SCORE = 1
DEFAULT_TURN_CURVE_INNER_PWM = 104
DEFAULT_TURN_CURVE_OUTER_PWMS = {
    "gentle": 155,
    "medium": 181,
    "strong": 209,
}
DEFAULT_STRONG_CURVE_ABS_X_ERR_MM = 18.0
DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM = 10.0
DEFAULT_MAX_ACT_MS = 400
DEFAULT_FOLLOW_COMBINED_GAP_POLICY = {
    "straight_x_outside_max_mm": 0.0,
    "straight_dist_outside_min_mm": 8.0,
    "micro_x_outside_max_mm": 2.0,
    "gentle_x_outside_max_mm": 6.0,
    "medium_x_outside_max_mm": 12.0,
}
DEFAULT_DIST_APPROACH_POLICY = {
    "closure_shots": 1.0,
    "settle_after_act_s": 0.04,
    "require_y_ok_before_dist": True,
    "min_forward_pulse_ms": 200,
    "max_forward_pulse_ms": 400,
    "full_forward_gap_mm": 75.0,
    "near_target_forward_veto_mm": 12.0,
}
DEFAULT_X_PRIORITY_POLICY = {
    "polish_abs_x_mm": 6.0,
    "huge_dist_gap_mm": 200.0,
    "huge_dist_tiny_abs_x_mm": 5.0,
    "x_first_turn_strength": "adaptive",
    "adaptive_outer_pwm_scale": 0.75,
}
DEFAULT_X_DIST_CURVE_POLICY = {
    "large_dist_gap_mm": 60.0,
    "small_x_gap_mm": 6.0,
    "large_dist_small_x_strength": "gentle",
    "near_dist_gap_mm": 12.0,
    "wide_x_gap_mm": 9.0,
    "near_wide_x_strength": "strong",
    "too_close_wide_x_drive_mode": "backward",
}
DEFAULT_X_ONLY_TURN_POLICY = {
    "drive_mode": "backward",
    "far_drive_mode": "forward",
    "forward_min_dist_err_mm": 60.0,
}
DEFAULT_TOO_CLOSE_ESCAPE_POLICY = {
    "pwm": 104,
    "pulse_ms": 400,
    "min_pulse_ms": 200,
    "full_escape_gap_mm": 25.0,
    "attach_mast": True,
}
DEFAULT_WIN_CONFIRMATION_CONFIG = {
    "settle_s": 0.25,
    "confirm_frames": 2,
    "min_axis_closeness_pct": 0.0,
}
DEFAULT_FOLLOW_Y_AXIS_CONFIG = {
    "enabled": True,
    "win_target_mm": Y_TARGET_MM,
    "win_tol_mm": Y_TOL_MM,
    "reset_target_mm": RESET_Y_TARGET_MM,
    "reset_tol_mm": Y_TOL_MM,
    "approach_high_factor": 1.3,
    "endgame_dist_tol_mm": DIST_TOL_MM,
    "endgame_x_tol_mm": X_TOL_MM,
    "finish_y_only_dist_deadband_mm": DIST_TOL_MM,
    "finish_y_only_too_far_deadband_mm": 15.0,
    "finish_y_only_too_close_deadband_mm": DIST_TOL_MM,
    "finish_y_only_x_deadband_mm": X_TOL_MM,
    "protect_below_y_mm": 11.0,
    "priority_abs_err_mm": 14.0,
    "lock_on_enabled": True,
    "lock_on_dist_mm": 90.0,
    "lock_on_dist_window_mm": 10.0,
    "lock_on_mast_pwm": 255,
    "lock_on_pulse_ms": 400,
    "mast_pwm": 40,
    "mast_pulse_ms": 220,
    "finish_mast_pwm": 120,
    "finish_mast_pulse_ms": 300,
}
DEFAULT_TURN_BIAS_CURVES = {
    "micro": {"inner_pwm": 103, "outer_pwm": 106},
    "gentle": {"inner_pwm": 103, "outer_pwm": 117},
    "medium": {"inner_pwm": 103, "outer_pwm": 133},
    "strong": {"inner_pwm": 104, "outer_pwm": 155},
}
TURN_BIAS_STRENGTHS = ("micro", "gentle", "medium", "strong")
DEFAULT_VISION_MIN_LFB_MB = 16.0

CROWN_PROFILE_TUNING = {
    "confidence": 0.08,
    "smoothing_alpha": 0.15,
    "hsv_enabled": True,
    "hsv_erode_iterations": 1,
    "hsv_lower": list(CYAN_HSV_BALANCED_LOWER),
    "hsv_upper": list(CYAN_HSV_BALANCED_UPPER),
    "hsv_cyan_coverage_min": 0.08,
    "hsv_min_area_ratio": 0.04,
    "shape_gate_mode": "shape_match",
    "conf_gate_pct": 40.0,
    "trust_detector_boxes": True,
    "require_cyan_shape": False,
    "closeup_full_frame_hsv_enabled": True,
    "depth_source_mode": "pinhole",
}
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("follow_the_brick")
ROBOT_MODEL_FILE = Path(__file__).resolve().parent / "world_model_robot.json"
FOLLOW_MOTION_CONFIG_KEY = "follow_the_brick"
RESET_MOTION_CONFIG_KEY = "follow_the_brick_reset"


def _stop_robot(robot: Robot) -> None:
    try:
        robot.stop()
    except Exception:
        pass


def _emergency_stop_robot() -> None:
    robot = None
    try:
        robot = Robot(exit_on_failure=False)
        robot.stop()
        print("[FOLLOW] Recovery stop sent.", flush=True)
    except Exception as exc:
        print(f"[FOLLOW] Recovery stop skipped: {exc}", flush=True)
    finally:
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass


def _parse_tegra_lfb_mb(text: str | None) -> float | None:
    if not text:
        return None
    values = []
    for count_text, block_text in re.findall(r"lfb\s+(\d+)x(\d+)MB", str(text)):
        try:
            _count = int(count_text)
            block_mb = float(block_text)
        except (TypeError, ValueError):
            continue
        if block_mb > 0.0:
            values.append(float(block_mb))
    if not values:
        return None
    return max(values)


def _vision_memory_preflight(*, min_lfb_mb: float = DEFAULT_VISION_MIN_LFB_MB) -> tuple[bool, str]:
    try:
        min_lfb = max(0.0, float(min_lfb_mb))
    except (TypeError, ValueError):
        min_lfb = float(DEFAULT_VISION_MIN_LFB_MB)
    if min_lfb <= 0.0:
        return True, "disabled"
    try:
        result = subprocess.run(
            ["timeout", "2s", "tegrastats"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3.0,
        )
    except Exception as exc:
        return True, f"tegrastats unavailable ({exc})"
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    lfb_mb = _parse_tegra_lfb_mb(text)
    if lfb_mb is None:
        return True, "tegrastats lfb unavailable"
    if float(lfb_mb) < float(min_lfb):
        return (
            False,
            f"largest free block {float(lfb_mb):.0f}MB < required {float(min_lfb):.0f}MB",
        )
    return True, f"largest free block {float(lfb_mb):.0f}MB"


def _coerce_int(value, fallback: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        coerced = int(round(float(value)))
    except (TypeError, ValueError):
        coerced = int(fallback)
    if minimum is not None:
        coerced = max(int(minimum), int(coerced))
    if maximum is not None:
        coerced = min(int(maximum), int(coerced))
    return int(coerced)


def _coerce_float(
    value,
    fallback: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = float(fallback)
    if minimum is not None:
        coerced = max(float(minimum), float(coerced))
    if maximum is not None:
        coerced = min(float(maximum), float(coerced))
    return float(coerced)


def _coerce_curve_strength(value, fallback: str = "gentle") -> str:
    strength = str(value or "").strip().lower()
    if strength == "adaptive":
        return "adaptive"
    if strength in {"gentle", "medium", "strong"}:
        return strength
    fallback_strength = str(fallback or "").strip().lower()
    if fallback_strength == "adaptive":
        return "adaptive"
    if fallback_strength in {"gentle", "medium", "strong"}:
        return fallback_strength
    return "gentle"


def _coerce_discrete_curve_strength(value, fallback: str = "gentle") -> str:
    strength = str(value or "").strip().lower()
    if strength in {"gentle", "medium", "strong"}:
        return strength
    fallback_strength = str(fallback or "").strip().lower()
    if fallback_strength in {"gentle", "medium", "strong"}:
        return fallback_strength
    return "gentle"


def _load_follow_motion_config(path: Path | None = None) -> dict:
    model_path = path if isinstance(path, Path) else ROBOT_MODEL_FILE
    cfg = {
        "motion_power_scale": float(DEFAULT_MOTION_POWER_SCALE),
        "normal_speed_score": int(DEFAULT_NORMAL_SPEED_SCORE),
        "turn_curves": {
            "inner_pwm": int(DEFAULT_TURN_CURVE_INNER_PWM),
            "forward": {
                name: {"outer_pwm": int(pwm)}
                for name, pwm in DEFAULT_TURN_CURVE_OUTER_PWMS.items()
            },
            "backward": {
                name: {"outer_pwm": int(pwm)}
                for name, pwm in DEFAULT_TURN_CURVE_OUTER_PWMS.items()
            },
        },
        "curve_strength_abs_x_err_mm": {
            "medium": float(DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM),
            "strong": float(DEFAULT_STRONG_CURVE_ABS_X_ERR_MM),
        },
        "max_act_ms": int(DEFAULT_MAX_ACT_MS),
        "combined_gap_policy": dict(DEFAULT_FOLLOW_COMBINED_GAP_POLICY),
        "dist_approach_policy": dict(DEFAULT_DIST_APPROACH_POLICY),
        "x_priority_policy": dict(DEFAULT_X_PRIORITY_POLICY),
        "x_dist_curve_policy": dict(DEFAULT_X_DIST_CURVE_POLICY),
        "x_only_turn": dict(DEFAULT_X_ONLY_TURN_POLICY),
        "too_close_escape": dict(DEFAULT_TOO_CLOSE_ESCAPE_POLICY),
        "win_confirmation": dict(DEFAULT_WIN_CONFIRMATION_CONFIG),
        "y_axis": dict(DEFAULT_FOLLOW_Y_AXIS_CONFIG),
        "turn_bias_curves": {
            drive_mode: {
                name: dict(curve)
                for name, curve in DEFAULT_TURN_BIAS_CURVES.items()
            }
            for drive_mode in ("forward", "backward")
        },
    }
    try:
        payload = json.loads(model_path.read_text())
    except Exception:
        return cfg
    raw = payload.get(FOLLOW_MOTION_CONFIG_KEY) if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return cfg
    cfg["motion_power_scale"] = _coerce_float(
        raw.get("motion_power_scale"),
        DEFAULT_MOTION_POWER_SCALE,
        minimum=0.01,
        maximum=2.0,
    )
    cfg["normal_speed_score"] = _coerce_int(
        raw.get("normal_speed_score"),
        DEFAULT_NORMAL_SPEED_SCORE,
        minimum=1,
        maximum=100,
    )
    cfg["max_act_ms"] = _coerce_int(
        raw.get("max_act_ms"),
        DEFAULT_MAX_ACT_MS,
        minimum=1,
        maximum=400,
    )
    turn_curves = raw.get("turn_curves") if isinstance(raw.get("turn_curves"), dict) else {}
    cfg["turn_curves"]["inner_pwm"] = _coerce_int(
        turn_curves.get("inner_pwm"),
        DEFAULT_TURN_CURVE_INNER_PWM,
        minimum=0,
        maximum=255,
    )
    for drive_mode in ("forward", "backward"):
        raw_drive_curves = turn_curves.get(drive_mode) if isinstance(turn_curves.get(drive_mode), dict) else {}
        for strength, fallback_pwm in DEFAULT_TURN_CURVE_OUTER_PWMS.items():
            raw_curve = raw_drive_curves.get(strength) if isinstance(raw_drive_curves.get(strength), dict) else {}
            cfg["turn_curves"][drive_mode][strength]["outer_pwm"] = _coerce_int(
                raw_curve.get("outer_pwm"),
                fallback_pwm,
                minimum=1,
                maximum=255,
            )
    thresholds = (
        raw.get("curve_strength_abs_x_err_mm")
        if isinstance(raw.get("curve_strength_abs_x_err_mm"), dict)
        else {}
    )
    cfg["curve_strength_abs_x_err_mm"]["medium"] = _coerce_float(
        thresholds.get("medium"),
        DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    cfg["curve_strength_abs_x_err_mm"]["strong"] = _coerce_float(
        thresholds.get("strong"),
        DEFAULT_STRONG_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    raw_policy = raw.get("combined_gap_policy") if isinstance(raw.get("combined_gap_policy"), dict) else {}
    for key, fallback in DEFAULT_FOLLOW_COMBINED_GAP_POLICY.items():
        cfg["combined_gap_policy"][key] = _coerce_float(
            raw_policy.get(key),
            fallback,
            minimum=0.0,
        )
    raw_dist_approach = (
        raw.get("dist_approach_policy")
        if isinstance(raw.get("dist_approach_policy"), dict)
        else {}
    )
    cfg["dist_approach_policy"]["closure_shots"] = _coerce_float(
        raw_dist_approach.get("closure_shots"),
        DEFAULT_DIST_APPROACH_POLICY["closure_shots"],
        minimum=1.0,
    )
    cfg["dist_approach_policy"]["settle_after_act_s"] = _coerce_float(
        raw_dist_approach.get("settle_after_act_s"),
        DEFAULT_DIST_APPROACH_POLICY["settle_after_act_s"],
        minimum=0.0,
        maximum=2.0,
    )
    cfg["dist_approach_policy"]["require_y_ok_before_dist"] = bool(
        raw_dist_approach.get(
            "require_y_ok_before_dist",
            DEFAULT_DIST_APPROACH_POLICY["require_y_ok_before_dist"],
        )
    )
    for key in ("min_forward_pulse_ms", "max_forward_pulse_ms"):
        cfg["dist_approach_policy"][key] = _coerce_int(
            raw_dist_approach.get(key),
            DEFAULT_DIST_APPROACH_POLICY[key],
            minimum=1,
            maximum=400,
        )
    cfg["dist_approach_policy"]["full_forward_gap_mm"] = _coerce_float(
        raw_dist_approach.get("full_forward_gap_mm"),
        DEFAULT_DIST_APPROACH_POLICY["full_forward_gap_mm"],
        minimum=0.1,
    )
    cfg["dist_approach_policy"]["near_target_forward_veto_mm"] = _coerce_float(
        raw_dist_approach.get("near_target_forward_veto_mm"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_forward_veto_mm"],
        minimum=0.0,
    )
    raw_x_priority = raw.get("x_priority_policy") if isinstance(raw.get("x_priority_policy"), dict) else {}
    for key in ("polish_abs_x_mm", "huge_dist_gap_mm", "huge_dist_tiny_abs_x_mm"):
        cfg["x_priority_policy"][key] = _coerce_float(
            raw_x_priority.get(key),
            DEFAULT_X_PRIORITY_POLICY[key],
            minimum=0.0,
        )
    cfg["x_priority_policy"]["x_first_turn_strength"] = _coerce_curve_strength(
        raw_x_priority.get("x_first_turn_strength"),
        DEFAULT_X_PRIORITY_POLICY["x_first_turn_strength"],
    )
    cfg["x_priority_policy"]["adaptive_outer_pwm_scale"] = _coerce_float(
        raw_x_priority.get("adaptive_outer_pwm_scale"),
        DEFAULT_X_PRIORITY_POLICY["adaptive_outer_pwm_scale"],
        minimum=0.5,
        maximum=2.0,
    )
    raw_x_dist = raw.get("x_dist_curve_policy") if isinstance(raw.get("x_dist_curve_policy"), dict) else {}
    for key in ("large_dist_gap_mm", "small_x_gap_mm", "near_dist_gap_mm", "wide_x_gap_mm"):
        cfg["x_dist_curve_policy"][key] = _coerce_float(
            raw_x_dist.get(key),
            DEFAULT_X_DIST_CURVE_POLICY[key],
            minimum=0.0,
        )
    cfg["x_dist_curve_policy"]["large_dist_small_x_strength"] = _coerce_curve_strength(
        raw_x_dist.get("large_dist_small_x_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["large_dist_small_x_strength"],
    )
    cfg["x_dist_curve_policy"]["near_wide_x_strength"] = _coerce_curve_strength(
        raw_x_dist.get("near_wide_x_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["near_wide_x_strength"],
    )
    if cfg["x_dist_curve_policy"]["near_wide_x_strength"] == "adaptive":
        cfg["x_dist_curve_policy"]["near_wide_x_strength"] = "strong"
    if cfg["x_dist_curve_policy"]["large_dist_small_x_strength"] == "adaptive":
        cfg["x_dist_curve_policy"]["large_dist_small_x_strength"] = "gentle"
    drive_mode = str(
        raw_x_dist.get(
            "too_close_wide_x_drive_mode",
            DEFAULT_X_DIST_CURVE_POLICY["too_close_wide_x_drive_mode"],
        )
    ).strip().lower()
    cfg["x_dist_curve_policy"]["too_close_wide_x_drive_mode"] = (
        drive_mode if drive_mode in {"forward", "backward"} else "backward"
    )
    raw_x_only = raw.get("x_only_turn") if isinstance(raw.get("x_only_turn"), dict) else {}
    for key in ("drive_mode", "far_drive_mode"):
        mode = str(raw_x_only.get(key, DEFAULT_X_ONLY_TURN_POLICY[key])).strip().lower()
        cfg["x_only_turn"][key] = mode if mode in {"forward", "backward"} else DEFAULT_X_ONLY_TURN_POLICY[key]
    cfg["x_only_turn"]["forward_min_dist_err_mm"] = _coerce_float(
        raw_x_only.get("forward_min_dist_err_mm"),
        DEFAULT_X_ONLY_TURN_POLICY["forward_min_dist_err_mm"],
        minimum=0.0,
    )
    raw_escape = raw.get("too_close_escape") if isinstance(raw.get("too_close_escape"), dict) else {}
    cfg["too_close_escape"]["pwm"] = _coerce_int(
        raw_escape.get("pwm"),
        DEFAULT_TOO_CLOSE_ESCAPE_POLICY["pwm"],
        minimum=1,
        maximum=255,
    )
    cfg["too_close_escape"]["pulse_ms"] = _coerce_int(
        raw_escape.get("pulse_ms"),
        DEFAULT_TOO_CLOSE_ESCAPE_POLICY["pulse_ms"],
        minimum=1,
        maximum=400,
    )
    cfg["too_close_escape"]["min_pulse_ms"] = _coerce_int(
        raw_escape.get("min_pulse_ms"),
        DEFAULT_TOO_CLOSE_ESCAPE_POLICY["min_pulse_ms"],
        minimum=1,
        maximum=400,
    )
    cfg["too_close_escape"]["full_escape_gap_mm"] = _coerce_float(
        raw_escape.get("full_escape_gap_mm"),
        DEFAULT_TOO_CLOSE_ESCAPE_POLICY["full_escape_gap_mm"],
        minimum=0.1,
    )
    cfg["too_close_escape"]["attach_mast"] = bool(
        raw_escape.get("attach_mast", DEFAULT_TOO_CLOSE_ESCAPE_POLICY["attach_mast"])
    )
    raw_win_confirmation = (
        raw.get("win_confirmation") if isinstance(raw.get("win_confirmation"), dict) else {}
    )
    cfg["win_confirmation"]["settle_s"] = _coerce_float(
        raw_win_confirmation.get("settle_s"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    cfg["win_confirmation"]["confirm_frames"] = _coerce_int(
        raw_win_confirmation.get("confirm_frames"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["confirm_frames"],
        minimum=1,
        maximum=10,
    )
    cfg["win_confirmation"]["min_axis_closeness_pct"] = _coerce_float(
        raw_win_confirmation.get("min_axis_closeness_pct"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["min_axis_closeness_pct"],
        minimum=0.0,
        maximum=100.0,
    )
    raw_y_axis = raw.get("y_axis") if isinstance(raw.get("y_axis"), dict) else {}
    cfg["y_axis"]["enabled"] = bool(raw_y_axis.get("enabled", DEFAULT_FOLLOW_Y_AXIS_CONFIG["enabled"]))
    for key, fallback in DEFAULT_FOLLOW_Y_AXIS_CONFIG.items():
        if key == "enabled":
            continue
        minimum = 0.0 if key not in {"win_target_mm", "reset_target_mm"} else None
        cfg["y_axis"][key] = _coerce_float(raw_y_axis.get(key), fallback, minimum=minimum)
    raw_bias_curves = raw.get("turn_bias_curves") if isinstance(raw.get("turn_bias_curves"), dict) else {}
    for drive_mode in ("forward", "backward"):
        raw_drive = raw_bias_curves.get(drive_mode) if isinstance(raw_bias_curves.get(drive_mode), dict) else {}
        for strength in TURN_BIAS_STRENGTHS:
            raw_curve = raw_drive.get(strength) if isinstance(raw_drive.get(strength), dict) else {}
            fallback_curve = DEFAULT_TURN_BIAS_CURVES[strength]
            cfg["turn_bias_curves"][drive_mode][strength]["inner_pwm"] = _coerce_int(
                raw_curve.get("inner_pwm"),
                fallback_curve["inner_pwm"],
                minimum=0,
                maximum=255,
            )
            cfg["turn_bias_curves"][drive_mode][strength]["outer_pwm"] = _coerce_int(
                raw_curve.get("outer_pwm"),
                fallback_curve["outer_pwm"],
                minimum=1,
                maximum=255,
            )
    return cfg


def _load_reset_motion_config(path: Path | None = None) -> dict:
    model_path = path if isinstance(path, Path) else ROBOT_MODEL_FILE
    cfg = {
        "reverse_turn": {
            "pulse_ms": int(RESET_REVERSE_TURN_PULSE_MS),
            "timeout_s": float(RESET_REVERSE_TURN_TIMEOUT_S),
            "settle_s": float(RESET_REVERSE_TURN_SETTLE_S),
            "post_pause_s": float(RESET_POST_PAUSE_S),
            "x_offset_min_mm": float(RESET_X_OFFSET_MIN_MM),
            "x_offset_max_mm": float(RESET_X_OFFSET_MAX_MM),
            "target_abs_x_mm": float(RESET_TARGET_ABS_X_MM),
            "y_target_mm": float(RESET_Y_TARGET_MM),
            "y_tol_mm": float(DEFAULT_FOLLOW_Y_AXIS_CONFIG["reset_tol_mm"]),
            "confirm_frames": int(RESET_X_OFFSET_CONFIRM_FRAMES),
            "arc_algorithm": {"points": [dict(point) for point in DEFAULT_RESET_ARC_ALGORITHM_POINTS]},
            "sharp_finish": {
                "enabled": True,
                "duration_ms": int(RESET_SHARP_FINISH_MS),
                "mode": "faster_wheel_only",
            },
        },
        "mast_up": {
            "enabled": True,
            "min_duration_ms": int(RESET_MAST_UP_MIN_MS),
            "max_duration_ms": int(RESET_MAST_UP_MAX_MS),
            "pwm": int(RESET_MAST_UP_PWM),
            "settle_s": float(RESET_MAST_UP_SETTLE_S),
        },
    }
    try:
        payload = json.loads(model_path.read_text())
    except Exception:
        return cfg
    raw = payload.get(RESET_MOTION_CONFIG_KEY) if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return cfg

    reverse_turn = raw.get("reverse_turn") if isinstance(raw.get("reverse_turn"), dict) else {}
    cfg["reverse_turn"]["pulse_ms"] = _coerce_int(
        reverse_turn.get("pulse_ms"),
        RESET_REVERSE_TURN_PULSE_MS,
        minimum=1,
    )
    cfg["reverse_turn"]["timeout_s"] = _coerce_float(
        reverse_turn.get("timeout_s"),
        RESET_REVERSE_TURN_TIMEOUT_S,
        minimum=0.1,
    )
    cfg["reverse_turn"]["settle_s"] = _coerce_float(
        reverse_turn.get("settle_s"),
        RESET_REVERSE_TURN_SETTLE_S,
        minimum=0.0,
    )
    cfg["reverse_turn"]["post_pause_s"] = _coerce_float(
        reverse_turn.get("post_pause_s"),
        RESET_POST_PAUSE_S,
        minimum=0.0,
    )
    x_min = _coerce_float(
        reverse_turn.get("x_offset_min_mm"),
        RESET_X_OFFSET_MIN_MM,
        minimum=0.0,
    )
    x_max = _coerce_float(
        reverse_turn.get("x_offset_max_mm"),
        RESET_X_OFFSET_MAX_MM,
        minimum=0.0,
    )
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    cfg["reverse_turn"]["x_offset_min_mm"] = float(x_min)
    cfg["reverse_turn"]["x_offset_max_mm"] = float(x_max)
    cfg["reverse_turn"]["target_abs_x_mm"] = _coerce_float(
        reverse_turn.get("target_abs_x_mm"),
        RESET_TARGET_ABS_X_MM,
        minimum=0.0,
    )
    cfg["reverse_turn"]["confirm_frames"] = _coerce_int(
        reverse_turn.get("confirm_frames"),
        RESET_X_OFFSET_CONFIRM_FRAMES,
        minimum=1,
    )
    cfg["reverse_turn"]["dist_target_mm"] = _coerce_float(
        reverse_turn.get("dist_target_mm"),
        RESET_DIST_TARGET_MM,
        minimum=0.0,
    )
    cfg["reverse_turn"]["dist_tol_mm"] = _coerce_float(
        reverse_turn.get("dist_tol_mm"),
        RESET_DIST_TOL_MM,
        minimum=0.0,
    )
    cfg["reverse_turn"]["y_target_mm"] = _coerce_float(
        reverse_turn.get("y_target_mm"),
        RESET_Y_TARGET_MM,
    )
    cfg["reverse_turn"]["y_tol_mm"] = _coerce_float(
        reverse_turn.get("y_tol_mm"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["reset_tol_mm"],
        minimum=0.0,
    )
    raw_algorithm = reverse_turn.get("arc_algorithm") if isinstance(reverse_turn.get("arc_algorithm"), dict) else {}
    raw_points = raw_algorithm.get("points") if isinstance(raw_algorithm.get("points"), list) else []
    points = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        points.append(
            {
                "x_gap_mm": _coerce_float(raw_point.get("x_gap_mm"), 0.0, minimum=0.0),
                "slower_pwm": _coerce_int(raw_point.get("slower_pwm"), 103, minimum=1, maximum=255),
                "faster_pwm": _coerce_int(raw_point.get("faster_pwm"), 112, minimum=1, maximum=255),
            }
        )
    if not points:
        points = [dict(point) for point in DEFAULT_RESET_ARC_ALGORITHM_POINTS]
    points = sorted(points, key=lambda point: float(point["x_gap_mm"]))
    cfg["reverse_turn"]["arc_algorithm"] = {
        "gap_metric": str(raw_algorithm.get("gap_metric") or "target_abs_x_minus_current_abs_x"),
        "points": points,
    }
    raw_sharp_finish = (
        reverse_turn.get("sharp_finish")
        if isinstance(reverse_turn.get("sharp_finish"), dict)
        else {}
    )
    cfg["reverse_turn"]["sharp_finish"] = {
        "enabled": bool(raw_sharp_finish.get("enabled", True)),
        "duration_ms": _coerce_int(
            raw_sharp_finish.get("duration_ms"),
            RESET_SHARP_FINISH_MS,
            minimum=0,
            maximum=max(0, cfg["reverse_turn"]["pulse_ms"]),
        ),
        "mode": str(raw_sharp_finish.get("mode") or "faster_wheel_only").strip().lower(),
    }
    mast_up = raw.get("mast_up") if isinstance(raw.get("mast_up"), dict) else {}
    cfg["mast_up"]["enabled"] = bool(mast_up.get("enabled", True))
    cfg["mast_up"]["min_duration_ms"] = _coerce_int(
        mast_up.get("min_duration_ms"),
        RESET_MAST_UP_MIN_MS,
        minimum=1,
    )
    cfg["mast_up"]["max_duration_ms"] = _coerce_int(
        mast_up.get("max_duration_ms"),
        RESET_MAST_UP_MAX_MS,
        minimum=1,
    )
    if cfg["mast_up"]["min_duration_ms"] > cfg["mast_up"]["max_duration_ms"]:
        cfg["mast_up"]["min_duration_ms"], cfg["mast_up"]["max_duration_ms"] = (
            cfg["mast_up"]["max_duration_ms"],
            cfg["mast_up"]["min_duration_ms"],
        )
    cfg["mast_up"]["pwm"] = _coerce_int(
        mast_up.get("pwm"),
        RESET_MAST_UP_PWM,
        minimum=1,
        maximum=255,
    )
    cfg["mast_up"]["settle_s"] = _coerce_float(
        mast_up.get("settle_s"),
        RESET_MAST_UP_SETTLE_S,
        minimum=0.0,
    )
    return cfg


def _reset_motion_config() -> dict:
    cfg = getattr(_reset_motion_config, "_cache", None)
    if not isinstance(cfg, dict):
        cfg = _load_reset_motion_config()
        setattr(_reset_motion_config, "_cache", cfg)
    return cfg


def _reset_post_pause_s() -> float:
    cfg = _reset_motion_config().get("reverse_turn")
    reset_cfg = cfg if isinstance(cfg, dict) else {}
    return _coerce_float(
        reset_cfg.get("post_pause_s"),
        RESET_POST_PAUSE_S,
        minimum=0.0,
    )


def _follow_motion_config() -> dict:
    cfg = getattr(_follow_motion_config, "_cache", None)
    if not isinstance(cfg, dict):
        cfg = _load_follow_motion_config()
        setattr(_follow_motion_config, "_cache", cfg)
    return cfg


def _motion_power_scale() -> float:
    cfg = _follow_motion_config()
    return _coerce_float(
        cfg.get("motion_power_scale"),
        DEFAULT_MOTION_POWER_SCALE,
        minimum=0.01,
        maximum=2.0,
    )


def _normal_speed_score() -> int:
    cfg = _follow_motion_config()
    return _coerce_int(
        cfg.get("normal_speed_score"),
        DEFAULT_NORMAL_SPEED_SCORE,
        minimum=1,
        maximum=100,
    )


def _x_only_turn_drive_mode() -> str:
    cfg = _follow_motion_config()
    raw = cfg.get("x_only_turn") if isinstance(cfg.get("x_only_turn"), dict) else {}
    drive_mode = str(raw.get("drive_mode") or DEFAULT_X_ONLY_TURN_POLICY["drive_mode"]).strip().lower()
    return drive_mode if drive_mode in {"forward", "backward"} else DEFAULT_X_ONLY_TURN_POLICY["drive_mode"]


def _x_only_turn_drive_mode_for_dist(dist_err: float) -> str:
    cfg = _follow_motion_config()
    raw = cfg.get("x_only_turn") if isinstance(cfg.get("x_only_turn"), dict) else {}
    near_mode = _x_only_turn_drive_mode()
    far_mode = str(raw.get("far_drive_mode") or DEFAULT_X_ONLY_TURN_POLICY["far_drive_mode"]).strip().lower()
    if far_mode not in {"forward", "backward"}:
        far_mode = DEFAULT_X_ONLY_TURN_POLICY["far_drive_mode"]
    forward_min_dist = _coerce_float(
        raw.get("forward_min_dist_err_mm"),
        DEFAULT_X_ONLY_TURN_POLICY["forward_min_dist_err_mm"],
        minimum=0.0,
    )
    try:
        dist_val = float(dist_err)
    except (TypeError, ValueError):
        dist_val = 0.0
    if dist_val >= float(forward_min_dist):
        return far_mode
    return near_mode


def _max_act_ms() -> int:
    cfg = _follow_motion_config()
    return _coerce_int(
        cfg.get("max_act_ms"),
        DEFAULT_MAX_ACT_MS,
        minimum=1,
        maximum=400,
    )


def _bounded_act_duration_ms(duration_ms: int | float | None) -> int:
    return min(
        int(_max_act_ms()),
        _coerce_int(duration_ms, PULSE_MS, minimum=1),
    )


def _reset_act_duration_ms(reset_cfg: dict | None = None) -> int:
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    timeout_s = _coerce_float(
        cfg.get("timeout_s"),
        RESET_REVERSE_TURN_TIMEOUT_S,
        minimum=0.001,
    )
    return _coerce_int(
        cfg.get("pulse_ms"),
        RESET_REVERSE_TURN_PULSE_MS,
        minimum=1,
        maximum=max(1, int(round(float(timeout_s) * 1000.0))),
    )


def _scaled_pwm(pwm: int | float | None) -> int:
    try:
        pwm_val = int(round(float(pwm)))
    except (TypeError, ValueError):
        pwm_val = 0
    if pwm_val <= 0:
        return 0
    return int(_telemetry_robot.clamp_pwm(int(round(float(pwm_val) * _motion_power_scale()))))


def _pwm_floor_for_cmd(cmd: str | None) -> int:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"f", "b", "l", "r"}:
        return 0
    try:
        floor = int(_telemetry_robot.baseline_pwm_floor_for_cmd(cmd_key))
    except Exception:
        floor = 0
    if cmd_key in {"l", "r"}:
        try:
            floor = max(int(floor), int(_telemetry_robot.turn_pwm_floor()))
        except Exception:
            pass
    return int(_telemetry_robot.clamp_pwm(max(0, int(floor))) or 0)


def _scaled_pwm_for_cmd(cmd: str | None, pwm: int | float | None) -> int:
    scaled = _scaled_pwm(pwm)
    if int(scaled) <= 0:
        return 0
    return int(max(int(scaled), int(_pwm_floor_for_cmd(cmd))))


def _approved_straight_drive_pwm(cmd: str | None) -> int:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"f", "b"}:
        return 0
    raw_pwm = int(_speed_pwm(cmd_key, _normal_speed_score()))
    # The Uno API serializes PWM as integer percent. Use the lowest percent
    # that preserves the approved score-1 drive floor, then treat that
    # wire-effective PWM as the hard top speed for straight drive.
    percent = int((float(raw_pwm) * 100.0 + 254.0) // 255.0)
    percent = max(0, min(100, int(percent)))
    return int((int(percent) * 255) / 100)


def _clamp_to_approved_straight_drive_pwm(cmd: str | None, pwm: int | float | None) -> int:
    cmd_key = str(cmd or "").strip().lower()
    pwm_val = int(_scaled_pwm_for_cmd(cmd_key, pwm))
    ceiling = int(_approved_straight_drive_pwm(cmd_key))
    if ceiling <= 0 or pwm_val <= 0:
        return int(pwm_val)
    return int(min(int(pwm_val), int(ceiling)))


def _scaled_actions(action_specs) -> list[dict]:
    scaled = []
    for action in action_specs or []:
        if not isinstance(action, dict):
            continue
        row = dict(action)
        row["pwm"] = _scaled_pwm_for_cmd(row.get("action"), row.get("pwm"))
        scaled.append(row)
    return scaled


def _normal_drive_pwm(direction: str = "f") -> int:
    return _speed_pwm(direction, _normal_speed_score())


def _curve_strength_for_abs_x_err(abs_x_err: float) -> str:
    cfg = _follow_motion_config()
    thresholds = (
        cfg.get("curve_strength_abs_x_err_mm")
        if isinstance(cfg.get("curve_strength_abs_x_err_mm"), dict)
        else {}
    )
    medium_threshold = _coerce_float(
        thresholds.get("medium"),
        DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    strong_threshold = _coerce_float(
        thresholds.get("strong"),
        DEFAULT_STRONG_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    if strong_threshold > 0.0 and abs_x_err >= strong_threshold:
        return "strong"
    if medium_threshold > 0.0 and abs_x_err >= medium_threshold:
        return "medium"
    return "gentle"


def _curve_strength_for_reading(reading: dict | None) -> str:
    """Determine curve strength based on x-axis error only.

    Uses x-offset magnitude to select gentle/medium/strong curves.
    This keeps the robot more stable during distance corrections.
    """
    try:
        x_err = abs(float((reading or {}).get("x_mm", 0.0)))
    except (TypeError, ValueError):
        x_err = 0.0

    return _curve_strength_for_abs_x_err(float(x_err))


def _turn_cmd_to_close_x_gap(x_mm: float) -> str | None:
    if float(x_mm) > 0.0:
        return "r"
    if float(x_mm) < 0.0:
        return "l"
    return None


def _opposite_turn_cmd(cmd: str | None) -> str | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "l":
        return "r"
    if cmd_key == "r":
        return "l"
    return None


def _turn_cmd_to_open_x_gap(x_mm: float, fallback_direction: str) -> str:
    close_cmd = _turn_cmd_to_close_x_gap(float(x_mm))
    open_cmd = _opposite_turn_cmd(close_cmd)
    if open_cmd in {"l", "r"}:
        return open_cmd
    fallback_cmd = str(fallback_direction or "").strip().lower()
    return fallback_cmd if fallback_cmd in {"l", "r"} else "l"


def _turn_curve_for_drive_mode(drive_mode: str, strength: str) -> dict:
    cfg = _follow_motion_config()
    turn_curves = cfg.get("turn_curves") if isinstance(cfg.get("turn_curves"), dict) else {}
    drive_key = str(drive_mode or "").strip().lower()
    if drive_key not in {"forward", "backward"}:
        drive_key = "forward"
    strength_key = _coerce_discrete_curve_strength(strength, "gentle")
    raw_drive = turn_curves.get(drive_key) if isinstance(turn_curves.get(drive_key), dict) else {}
    raw_curve = raw_drive.get(strength_key) if isinstance(raw_drive.get(strength_key), dict) else {}
    return {
        "inner_pwm": _coerce_int(
            turn_curves.get("inner_pwm"),
            DEFAULT_TURN_CURVE_INNER_PWM,
            minimum=0,
            maximum=255,
        ),
        "outer_pwm": _coerce_int(
            raw_curve.get("outer_pwm"),
            DEFAULT_TURN_CURVE_OUTER_PWMS[strength_key],
            minimum=1,
            maximum=255,
        ),
        "strength": strength_key,
        "drive_mode": drive_key,
    }


def _turn_bias_curve_for_drive_mode(drive_mode: str, strength: str) -> dict:
    cfg = _follow_motion_config()
    bias_curves = cfg.get("turn_bias_curves") if isinstance(cfg.get("turn_bias_curves"), dict) else {}
    drive_key = str(drive_mode or "").strip().lower()
    if drive_key not in {"forward", "backward"}:
        drive_key = "forward"
    strength_key = str(strength or "").strip().lower()
    if strength_key not in TURN_BIAS_STRENGTHS:
        strength_key = "micro"
    raw_drive = bias_curves.get(drive_key) if isinstance(bias_curves.get(drive_key), dict) else {}
    raw_curve = raw_drive.get(strength_key) if isinstance(raw_drive.get(strength_key), dict) else {}
    fallback_curve = DEFAULT_TURN_BIAS_CURVES[strength_key]
    return {
        "inner_pwm": _coerce_int(
            raw_curve.get("inner_pwm"),
            fallback_curve["inner_pwm"],
            minimum=0,
            maximum=255,
        ),
        "outer_pwm": _coerce_int(
            raw_curve.get("outer_pwm"),
            fallback_curve["outer_pwm"],
            minimum=1,
            maximum=255,
        ),
        "strength": strength_key,
        "drive_mode": drive_key,
    }


def _lerp_pwm(low_pwm: int, high_pwm: int, ratio: float) -> int:
    ratio_val = max(0.0, min(1.0, float(ratio)))
    return int(round(float(low_pwm) + ((float(high_pwm) - float(low_pwm)) * ratio_val)))


def _interp_between_curve_points(points: list[tuple[float, dict]], x_abs_mm: float) -> dict:
    ordered = sorted(points, key=lambda row: float(row[0]))
    if not ordered:
        return {}
    x_val = max(0.0, float(x_abs_mm))
    if x_val <= float(ordered[0][0]):
        out = dict(ordered[0][1])
        out["x_curve_gap_mm"] = float(x_val)
        return out
    for (left_x, left_curve), (right_x, right_curve) in zip(ordered, ordered[1:]):
        if x_val > float(right_x):
            continue
        span = max(1e-6, float(right_x) - float(left_x))
        ratio = (x_val - float(left_x)) / span
        out = dict(left_curve)
        out["inner_pwm"] = _lerp_pwm(int(left_curve["inner_pwm"]), int(right_curve["inner_pwm"]), ratio)
        out["outer_pwm"] = _lerp_pwm(int(left_curve["outer_pwm"]), int(right_curve["outer_pwm"]), ratio)
        out["strength"] = f"adaptive_{x_val:.1f}mm"
        out["x_curve_gap_mm"] = float(x_val)
        out["x_curve_ratio"] = float(max(0.0, min(1.0, ratio)))
        return out
    out = dict(ordered[-1][1])
    out["strength"] = f"adaptive_{x_val:.1f}mm"
    out["x_curve_gap_mm"] = float(x_val)
    out["x_curve_ratio"] = 1.0
    return out


def _adaptive_turn_curve_for_drive_mode(drive_mode: str, x_abs_mm: float) -> dict:
    medium_at = _coerce_float(
        _follow_motion_config().get("curve_strength_abs_x_err_mm", {}).get("medium"),
        DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    strong_at = _coerce_float(
        _follow_motion_config().get("curve_strength_abs_x_err_mm", {}).get("strong"),
        DEFAULT_STRONG_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    if strong_at < medium_at:
        strong_at = medium_at
    points = [
        (0.0, _turn_curve_for_drive_mode(drive_mode, "gentle")),
        (medium_at, _turn_curve_for_drive_mode(drive_mode, "medium")),
        (strong_at, _turn_curve_for_drive_mode(drive_mode, "strong")),
    ]
    out = _interp_between_curve_points(points, x_abs_mm)
    out["drive_mode"] = str(drive_mode or out.get("drive_mode") or "forward")
    scale = float(_follow_x_priority_policy().get("adaptive_outer_pwm_scale", 1.0))
    out["outer_pwm"] = _telemetry_robot.clamp_pwm(int(round(float(out["outer_pwm"]) * scale)))
    out["adaptive_outer_pwm_scale"] = float(scale)
    return out


def _adaptive_turn_bias_curve_for_drive_mode(drive_mode: str, x_abs_mm: float) -> dict:
    medium_at = _coerce_float(
        _follow_motion_config().get("curve_strength_abs_x_err_mm", {}).get("medium"),
        DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    strong_at = _coerce_float(
        _follow_motion_config().get("curve_strength_abs_x_err_mm", {}).get("strong"),
        DEFAULT_STRONG_CURVE_ABS_X_ERR_MM,
        minimum=0.0,
    )
    if strong_at < medium_at:
        strong_at = medium_at
    points = [
        (0.0, _turn_bias_curve_for_drive_mode(drive_mode, "gentle")),
        (medium_at, _turn_bias_curve_for_drive_mode(drive_mode, "medium")),
        (strong_at, _turn_bias_curve_for_drive_mode(drive_mode, "strong")),
    ]
    out = _interp_between_curve_points(points, x_abs_mm)
    out["drive_mode"] = str(drive_mode or out.get("drive_mode") or "forward")
    scale = float(_follow_x_priority_policy().get("adaptive_outer_pwm_scale", 1.0))
    out["outer_pwm"] = _telemetry_robot.clamp_pwm(int(round(float(out["outer_pwm"]) * scale)))
    out["adaptive_outer_pwm_scale"] = float(scale)
    return out


def _turn_curve_actions(*, drive_mode: str, cmd: str, curve: dict) -> list[dict]:
    turn_cmd = str(cmd or "").strip().lower()
    drive_key = str(drive_mode or "").strip().lower()
    if turn_cmd not in {"l", "r"} or drive_key not in {"forward", "backward"}:
        return []
    try:
        inner_pwm = int(curve.get("inner_pwm"))
    except (TypeError, ValueError):
        inner_pwm = int(DEFAULT_TURN_CURVE_INNER_PWM)
    try:
        outer_pwm = int(curve.get("outer_pwm"))
    except (TypeError, ValueError):
        outer_pwm = int(DEFAULT_TURN_CURVE_OUTER_PWMS["gentle"])
    drive_actions = (
        {"l": "b", "r": "f"}
        if drive_key == "forward"
        else {"l": "f", "r": "b"}
    )
    if turn_cmd == "r":
        by_target = {
            "l": {"target": "l", "action": drive_actions["l"], "pwm": int(outer_pwm)},
            "r": {"target": "r", "action": drive_actions["r"], "pwm": int(inner_pwm)},
        }
    else:
        by_target = {
            "l": {"target": "l", "action": drive_actions["l"], "pwm": int(inner_pwm)},
            "r": {"target": "r", "action": drive_actions["r"], "pwm": int(outer_pwm)},
        }
    return [dict(by_target[target]) for target in ("l", "r")]


def _turn_bias_actions(*, drive_mode: str, turn_cmd: str, curve: dict) -> list[dict]:
    return _turn_curve_actions(drive_mode=drive_mode, cmd=turn_cmd, curve=curve)


def _interp_reset_arc_point(points: list[dict], x_gap_mm: float) -> dict:
    if not points:
        points = [dict(point) for point in DEFAULT_RESET_ARC_ALGORITHM_POINTS]
    ordered = sorted(points, key=lambda point: float(point.get("x_gap_mm", 0.0)))
    gap = max(0.0, float(x_gap_mm))
    if gap <= float(ordered[0].get("x_gap_mm", 0.0)):
        return dict(ordered[0])
    for left, right in zip(ordered, ordered[1:]):
        left_gap = float(left.get("x_gap_mm", 0.0))
        right_gap = float(right.get("x_gap_mm", left_gap))
        if gap > right_gap:
            continue
        span = max(1e-6, right_gap - left_gap)
        frac = max(0.0, min(1.0, (gap - left_gap) / span))
        return {
            "x_gap_mm": float(gap),
            "slower_pwm": int(round(float(left.get("slower_pwm", 103)) + (float(right.get("slower_pwm", 103)) - float(left.get("slower_pwm", 103))) * frac)),
            "faster_pwm": int(round(float(left.get("faster_pwm", 112)) + (float(right.get("faster_pwm", 112)) - float(left.get("faster_pwm", 112))) * frac)),
        }
    return dict(ordered[-1])


def _reset_arc_curve_for_reading(reading: dict, reset_cfg: dict) -> dict:
    try:
        current_abs_x = abs(float((reading or {}).get("x_mm")))
    except (TypeError, ValueError):
        current_abs_x = 0.0
    x_min = _coerce_float(reset_cfg.get("x_offset_min_mm"), RESET_X_OFFSET_MIN_MM, minimum=0.0)
    x_max = _coerce_float(reset_cfg.get("x_offset_max_mm"), RESET_X_OFFSET_MAX_MM, minimum=0.0)
    target_abs_x = _coerce_float(
        reset_cfg.get("target_abs_x_mm"),
        (float(x_min) + float(x_max)) / 2.0,
        minimum=0.0,
    )
    if target_abs_x < min(x_min, x_max) or target_abs_x > max(x_min, x_max):
        target_abs_x = (float(x_min) + float(x_max)) / 2.0
    x_gap = max(0.0, float(target_abs_x) - float(current_abs_x))
    algorithm = reset_cfg.get("arc_algorithm") if isinstance(reset_cfg.get("arc_algorithm"), dict) else {}
    points = algorithm.get("points") if isinstance(algorithm.get("points"), list) else []
    point = _interp_reset_arc_point(points, x_gap)
    slower_pwm = _coerce_int(point.get("slower_pwm"), 103, minimum=1, maximum=255)
    faster_pwm = _coerce_int(point.get("faster_pwm"), 112, minimum=1, maximum=255)
    if faster_pwm < slower_pwm:
        faster_pwm, slower_pwm = slower_pwm, faster_pwm
    return {
        "inner_pwm": int(slower_pwm),
        "outer_pwm": int(faster_pwm),
        "strength": f"gap_{x_gap:.1f}mm",
        "drive_mode": "backward",
        "x_gap_mm": float(x_gap),
        "target_abs_x_mm": float(target_abs_x),
        "current_abs_x_mm": float(current_abs_x),
        "slower_pwm": int(slower_pwm),
        "faster_pwm": int(faster_pwm),
    }


def _send_turn_curve(
    robot: Robot,
    *,
    cmd: str,
    drive_mode: str,
    strength: str,
    duration_ms: int,
    reading: dict,
    context: str,
    mast_cmd: str | None = None,
) -> dict | None:
    try:
        x_abs = abs(float((reading or {}).get("x_mm", 0.0)))
    except (TypeError, ValueError):
        x_abs = 0.0
    curve = (
        _adaptive_turn_curve_for_drive_mode(drive_mode, x_abs)
        if str(strength or "").strip().lower() == "adaptive"
        else _turn_curve_for_drive_mode(drive_mode, strength)
    )
    actions = _turn_curve_actions(drive_mode=drive_mode, cmd=cmd, curve=curve)
    if not actions:
        return None
    scaled_actions = _actions_with_mast(_scaled_actions(actions), mast_cmd)
    send_result = guarded_send_custom_actions_pwm(
        robot,
        str(cmd),
        scaled_actions,
        duration_ms=_bounded_act_duration_ms(duration_ms),
        reading=reading,
        context=f"{context}_{curve['drive_mode']}_{curve['strength']}",
    )
    if send_result is None:
        return {
            "cmd_sent": str(cmd),
            "actions": scaled_actions,
            "duration_ms": _bounded_act_duration_ms(duration_ms),
            "x_curve": dict(curve),
        }
    if isinstance(send_result, dict):
        send_result["x_curve"] = dict(curve)
    return send_result


def _send_drive_bias(
    robot: Robot,
    *,
    turn_cmd: str,
    drive_mode: str,
    strength: str,
    duration_ms: int,
    reading: dict,
    context: str,
    mast_cmd: str | None = None,
) -> dict | None:
    drive_key = str(drive_mode or "").strip().lower()
    logical_cmd = "b" if drive_key == "backward" else "f"
    try:
        x_abs = abs(float((reading or {}).get("x_mm", 0.0)))
    except (TypeError, ValueError):
        x_abs = 0.0
    curve = (
        _adaptive_turn_bias_curve_for_drive_mode(drive_key, x_abs)
        if str(strength or "").strip().lower() == "adaptive"
        else _turn_bias_curve_for_drive_mode(drive_key, strength)
    )
    actions = _turn_bias_actions(drive_mode=drive_key, turn_cmd=turn_cmd, curve=curve)
    if not actions:
        return None
    scaled_actions = _actions_with_mast(_scaled_actions(actions), mast_cmd)
    send_result = guarded_send_custom_actions_pwm(
        robot,
        logical_cmd,
        scaled_actions,
        duration_ms=_bounded_act_duration_ms(duration_ms),
        reading=reading,
        context=f"{context}_{curve['drive_mode']}_{curve['strength']}_{turn_cmd}",
    )
    if send_result is None:
        return {
            "cmd_sent": logical_cmd,
            "turn_cmd": str(turn_cmd),
            "actions": scaled_actions,
            "duration_ms": _bounded_act_duration_ms(duration_ms),
            "x_curve": dict(curve),
        }
    if isinstance(send_result, dict):
        send_result["x_curve"] = dict(curve)
    return send_result


def _speed_pwm(cmd: str, score: int) -> int:
    pwm, _duration_ms = _speed_pwm_duration(cmd, score)
    return int(pwm)


def _speed_pwm_duration(cmd: str, score: int) -> tuple[int, int]:
    _, pwm, _, duration_ms = _telemetry_robot.speed_power_pwm_for_cmd(cmd, int(score))
    try:
        duration = max(1, int(round(float(duration_ms))))
    except (TypeError, ValueError):
        duration = PULSE_MS
    return int(pwm), int(duration)


def _curve_forward(robot: Robot, cmd: str, reading: dict) -> None:
    """Send one configured forward turn-curve pulse.

    The physical tread directions come from the explicit world-model curve pair,
    with sharpness selected from the current x-axis error.
    """
    return _send_turn_curve(
        robot,
        cmd=cmd,
        drive_mode="forward",
        strength=_curve_strength_for_reading(reading),
        duration_ms=_bounded_act_duration_ms(PULSE_MS),
        reading=reading,
        context=f"follow_curve_{cmd}",
    )


def _mast_action_spec(direction: str | None) -> dict | None:
    cmd = str(direction or "").strip().lower()
    if cmd not in {"u", "d"}:
        return None
    y_cfg = _follow_y_axis_config()
    # Custom action specs are already at the Uno target/action layer. Keep the
    # operator-facing plan logical, but serialize through the same mast polarity
    # used by Robot.send_command_pwm: logical up -> m.d, logical down -> m.u.
    wire_action = "d" if cmd == "u" else "u"
    return {
        "target": "m",
        "action": wire_action,
        "pwm": _scaled_pwm_for_cmd(cmd, y_cfg.get("mast_pwm")),
    }


def _actions_with_mast(actions, mast_cmd: str | None):
    out = [dict(action) for action in (actions or []) if isinstance(action, dict)]
    mast_action = _mast_action_spec(mast_cmd)
    if mast_action is not None:
        out.append(mast_action)
    return out


def _straight_drive_actions(direction: str, pwm: int) -> list[dict]:
    cmd = str(direction or "").strip().lower()
    left_action, right_action = ("f", "b") if cmd == "b" else ("b", "f")
    return [
        {"target": "l", "action": left_action, "pwm": int(pwm)},
        {"target": "r", "action": right_action, "pwm": int(pwm)},
    ]


def _drive(
    robot: Robot,
    direction: str,
    reading: dict,
    *,
    mast_cmd: str | None = None,
    pwm: int | float | None = None,
    duration_ms: int | float | None = None,
) -> None:
    """Straight drive: 'f' forward, 'b' backward."""
    cmd = str(direction or "").strip().lower()
    requested_pwm = _normal_drive_pwm(cmd) if pwm is None else pwm
    drive_pwm = _clamp_to_approved_straight_drive_pwm(cmd, requested_pwm)
    act_ms = _bounded_act_duration_ms(PULSE_MS if duration_ms is None else duration_ms)
    if str(mast_cmd or "").strip().lower() in {"u", "d"}:
        actions = _actions_with_mast(_straight_drive_actions(cmd, drive_pwm), mast_cmd)
        return guarded_send_custom_actions_pwm(
            robot,
            cmd,
            actions,
            duration_ms=act_ms,
            reading=reading,
            context=f"follow_drive_{cmd}_with_mast_{mast_cmd}",
        )
    return guarded_send_command_pwm(
        robot,
        cmd,
        drive_pwm,
        duration_ms=act_ms,
        reading=reading,
        context=f"follow_drive_{cmd}",
    )


def _mast(
    robot: Robot,
    direction: str,
    reading: dict,
    *,
    pwm: int | float | None = None,
    duration_ms: int | float | None = None,
) -> dict | None:
    cmd = str(direction or "").strip().lower()
    if cmd not in {"u", "d"}:
        return None
    y_cfg = _follow_y_axis_config()
    mast_pwm = y_cfg.get("mast_pwm") if pwm is None else pwm
    mast_duration_ms = y_cfg.get("mast_pulse_ms") if duration_ms is None else duration_ms
    pwm = _scaled_pwm_for_cmd(cmd, mast_pwm)
    duration_ms = _bounded_act_duration_ms(mast_duration_ms)
    return guarded_send_command_pwm(
        robot,
        cmd,
        pwm,
        duration_ms=duration_ms,
        reading=reading,
        context=f"follow_mast_{cmd}",
    )


def _lock_on_mast_down(robot: Robot, reading: dict) -> dict | None:
    y_cfg = _follow_y_axis_config()
    pwm = _scaled_pwm_for_cmd("d", y_cfg.get("lock_on_mast_pwm", 255))
    duration_ms = _bounded_act_duration_ms(y_cfg.get("lock_on_pulse_ms"))
    return guarded_send_command_pwm(
        robot,
        "d",
        pwm,
        duration_ms=duration_ms,
        reading=reading,
        context="follow_y_lock_on_mast_down",
    )


def _should_run_y_lock_on(stats: dict, reading: dict) -> bool:
    y_cfg = _follow_y_axis_config()
    if not bool(y_cfg.get("lock_on_enabled", True)):
        return False
    if not bool(stats.get("y_lock_on_armed", True)):
        return False
    try:
        dist_mm = float((reading or {}).get("dist_mm"))
    except (TypeError, ValueError):
        return False
    target = _coerce_float(y_cfg.get("lock_on_dist_mm"), 90.0, minimum=0.0)
    window = _coerce_float(y_cfg.get("lock_on_dist_window_mm"), 10.0, minimum=0.0)
    if abs(float(dist_mm) - float(target)) > float(window):
        return False
    try:
        y_err = _y_err_for_reading(
            reading,
            target=float(y_cfg.get("win_target_mm", Y_TARGET_MM)),
        )
        tol = float(y_cfg.get("win_tol_mm", Y_TOL_MM))
    except (TypeError, ValueError):
        return False
    return y_err is not None and float(y_err) > float(tol)


def _turn_in_place(robot: Robot, cmd: str, reading: dict) -> None:
    """Short in-place x correction without spending a distance act."""
    return guarded_send_command_pwm(
        robot,
        cmd,
        _scaled_pwm_for_cmd(cmd, _speed_pwm(cmd, _normal_speed_score())),
        duration_ms=_bounded_act_duration_ms(PULSE_MS),
        reading=reading,
        context=f"follow_turn_{cmd}",
    )


def _reset_sharp_finish_ms(reset_cfg: dict, total_duration_ms: int) -> int:
    sharp_cfg = reset_cfg.get("sharp_finish") if isinstance(reset_cfg.get("sharp_finish"), dict) else {}
    if not bool(sharp_cfg.get("enabled", True)):
        return 0
    return _coerce_int(
        sharp_cfg.get("duration_ms"),
        RESET_SHARP_FINISH_MS,
        minimum=0,
        maximum=max(0, int(total_duration_ms)),
    )


def _reset_segmented_turn_actions(actions: list[dict], reset_cfg: dict, curve: dict, duration_ms: int) -> tuple[list[dict], int]:
    """Build one reset packet: gentle arc first, then faster wheel finishes sharp."""
    total_ms = _coerce_int(duration_ms, RESET_REVERSE_TURN_PULSE_MS, minimum=1)
    sharp_ms = _reset_sharp_finish_ms(reset_cfg, total_ms)
    gentle_ms = max(1, int(total_ms) - int(sharp_ms))
    try:
        faster_pwm = int(curve.get("faster_pwm"))
    except (TypeError, ValueError):
        faster_pwm = max((int(action.get("pwm") or 0) for action in actions or []), default=0)
    segmented = []
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        row = dict(action)
        try:
            raw_pwm = int(round(float(row.get("pwm"))))
        except (TypeError, ValueError):
            raw_pwm = 0
        row["duration_ms"] = int(total_ms if raw_pwm >= int(faster_pwm) else gentle_ms)
        segmented.append(row)
    return _scaled_actions(segmented), int(sharp_ms)


def _reset_reverse_turn(
    robot: Robot,
    direction: str,
    reading: dict,
    *,
    rng=None,
) -> dict | None:
    """Send one combined reset act: backward wheel arc plus mast-up."""
    turn_cmd = str(direction or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        return None
    cfg = _reset_motion_config().get("reverse_turn")
    reset_cfg = cfg if isinstance(cfg, dict) else {}
    curve = _reset_arc_curve_for_reading(reading, reset_cfg)
    actions = _turn_curve_actions(drive_mode="backward", cmd=turn_cmd, curve=curve)
    if not actions:
        return None
    duration_ms = _reset_act_duration_ms(reset_cfg)
    scaled_actions, sharp_finish_ms = _reset_segmented_turn_actions(actions, reset_cfg, curve, duration_ms)
    mast_action, mast_up_ms, mast_settle_s = _reset_mast_up_action_spec(rng=rng)
    if mast_action is not None:
        scaled_actions.append(mast_action)
    send_result = guarded_send_custom_actions_pwm(
        robot,
        turn_cmd,
        scaled_actions,
        duration_ms=duration_ms,
        reading=reading,
        context=f"reset_backward_arc_{turn_cmd}_gap_{curve['x_gap_mm']:.1f}mm",
    )
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        return None
    return {
        "wheel_ms": int(duration_ms),
        "gentle_ms": int(max(1, int(duration_ms) - int(sharp_finish_ms))),
        "sharp_finish_ms": int(sharp_finish_ms),
        "mast_up_ms": int(mast_up_ms),
        "mast_settle_s": float(mast_settle_s),
        "duration_ms": int(max(int(duration_ms), int(mast_up_ms))),
        "actions": [dict(action) for action in scaled_actions],
        "curve": dict(curve),
    }


def _reset_mast_up_action_spec(*, rng=None) -> tuple[dict | None, int, float]:
    """Build the reset mast-up custom action for the combined reset packet."""
    reset_cfg = _reset_motion_config()
    cfg = reset_cfg.get("mast_up") if isinstance(reset_cfg.get("mast_up"), dict) else {}
    settle_s = _coerce_float(cfg.get("settle_s"), RESET_MAST_UP_SETTLE_S, minimum=0.0)
    if not bool(cfg.get("enabled", True)):
        return None, 0, float(settle_s)
    min_ms = _coerce_int(cfg.get("min_duration_ms"), RESET_MAST_UP_MIN_MS, minimum=1)
    max_ms = _coerce_int(cfg.get("max_duration_ms"), RESET_MAST_UP_MAX_MS, minimum=1)
    if min_ms > max_ms:
        min_ms, max_ms = max_ms, min_ms
    random_source = rng if rng is not None else random
    try:
        duration_ms = int(round(float(random_source.uniform(float(min_ms), float(max_ms)))))
    except AttributeError:
        duration_ms = int(round((float(min_ms) + float(max_ms)) / 2.0))
    pwm = _coerce_int(cfg.get("pwm"), RESET_MAST_UP_PWM, minimum=1, maximum=255)
    return {
        "target": "m",
        # Custom actions are wire-level: logical mast-up maps to m.d.
        "action": "d",
        "pwm": _scaled_pwm_for_cmd("u", pwm),
        "duration_ms": int(duration_ms),
    }, int(duration_ms), float(settle_s)


def _reset_mast_up_enabled() -> bool:
    reset_cfg = _reset_motion_config()
    cfg = reset_cfg.get("mast_up") if isinstance(reset_cfg.get("mast_up"), dict) else {}
    return bool(cfg.get("enabled", True))


def _warmup(vision: BrickDetector) -> None:
    log.info("Warming up camera pipeline (%d reads)...", WARMUP_READS)
    for _ in range(WARMUP_READS):
        try:
            vision.read()
        except Exception:
            pass
        time.sleep(0.06)
    log.info("Camera ready.")


def _wait_for_confident_brick(
    vision: BrickDetector,
    *,
    timeout_s: float = PREGAME_VISIBILITY_TIMEOUT_S,
    sample_s: float = PREGAME_VISIBILITY_SAMPLE_S,
) -> dict:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last_reading = brick_motion_measurement_from_result(None)
    first_sample = True
    while first_sample or time.monotonic() <= deadline:
        first_sample = False
        last_reading = _read_brick_measurement(vision)
        if bool(last_reading.get("confident")):
            try:
                dist_mm = float(last_reading.get("dist_mm"))
                x_mm = float(last_reading.get("x_mm"))
                y_mm = float(last_reading.get("y_mm"))
                conf = float(last_reading.get("conf"))
                print(
                    f"[FOLLOW] Pregame visibility ok: dist={dist_mm:.1f}mm "
                    f"x={x_mm:+.1f}mm y={y_mm:+.1f}mm conf={conf:.0f}%",
                    flush=True,
                )
            except (TypeError, ValueError):
                print("[FOLLOW] Pregame visibility ok.", flush=True)
            return last_reading
        time.sleep(max(0.01, float(sample_s)))
    print(
        "[FOLLOW] Pregame visibility blocked: brick was not confident; "
        "no robot motion was started.",
        flush=True,
    )
    return last_reading


def _read_brick_measurement(vision: BrickDetector) -> dict:
    """Return a fresh brick reading without any held-frame fallback."""
    try:
        result = vision.read()
    except Exception as exc:
        log.warning("Vision read error: %s", exc)
        return brick_motion_measurement_from_result(None)
    return brick_motion_measurement_from_result(result)


def _follow_combined_gap_policy() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("combined_gap_policy") if isinstance(cfg.get("combined_gap_policy"), dict) else {}
    policy = {}
    for key, fallback in DEFAULT_FOLLOW_COMBINED_GAP_POLICY.items():
        policy[key] = _coerce_float(raw.get(key), fallback, minimum=0.0)
    return policy


def _follow_dist_approach_policy() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("dist_approach_policy") if isinstance(cfg.get("dist_approach_policy"), dict) else {}
    out = {
        "closure_shots": _coerce_float(
            raw.get("closure_shots"),
            DEFAULT_DIST_APPROACH_POLICY["closure_shots"],
            minimum=1.0,
        ),
        "settle_after_act_s": _coerce_float(
            raw.get("settle_after_act_s"),
            DEFAULT_DIST_APPROACH_POLICY["settle_after_act_s"],
            minimum=0.0,
            maximum=2.0,
        ),
        "require_y_ok_before_dist": bool(
            raw.get(
                "require_y_ok_before_dist",
                DEFAULT_DIST_APPROACH_POLICY["require_y_ok_before_dist"],
            )
        ),
        "min_forward_pulse_ms": _coerce_int(
            raw.get("min_forward_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["min_forward_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "max_forward_pulse_ms": _coerce_int(
            raw.get("max_forward_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["max_forward_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "full_forward_gap_mm": _coerce_float(
            raw.get("full_forward_gap_mm"),
            DEFAULT_DIST_APPROACH_POLICY["full_forward_gap_mm"],
            minimum=0.1,
        ),
        "near_target_forward_veto_mm": _coerce_float(
            raw.get("near_target_forward_veto_mm"),
            DEFAULT_DIST_APPROACH_POLICY["near_target_forward_veto_mm"],
            minimum=0.0,
        ),
    }
    if int(out["min_forward_pulse_ms"]) > int(out["max_forward_pulse_ms"]):
        out["min_forward_pulse_ms"], out["max_forward_pulse_ms"] = (
            out["max_forward_pulse_ms"],
            out["min_forward_pulse_ms"],
        )
    return out


def _follow_x_priority_policy() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("x_priority_policy") if isinstance(cfg.get("x_priority_policy"), dict) else {}
    return {
        "polish_abs_x_mm": _coerce_float(
            raw.get("polish_abs_x_mm"),
            DEFAULT_X_PRIORITY_POLICY["polish_abs_x_mm"],
            minimum=0.0,
        ),
        "huge_dist_gap_mm": _coerce_float(
            raw.get("huge_dist_gap_mm"),
            DEFAULT_X_PRIORITY_POLICY["huge_dist_gap_mm"],
            minimum=0.0,
        ),
        "huge_dist_tiny_abs_x_mm": _coerce_float(
            raw.get("huge_dist_tiny_abs_x_mm"),
            DEFAULT_X_PRIORITY_POLICY["huge_dist_tiny_abs_x_mm"],
            minimum=0.0,
        ),
        "x_first_turn_strength": _coerce_curve_strength(
            raw.get("x_first_turn_strength"),
            DEFAULT_X_PRIORITY_POLICY["x_first_turn_strength"],
        ),
        "adaptive_outer_pwm_scale": _coerce_float(
            raw.get("adaptive_outer_pwm_scale"),
            DEFAULT_X_PRIORITY_POLICY["adaptive_outer_pwm_scale"],
            minimum=0.5,
            maximum=2.0,
        ),
    }


def _follow_x_dist_curve_policy() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("x_dist_curve_policy") if isinstance(cfg.get("x_dist_curve_policy"), dict) else {}
    out = {}
    for key in ("large_dist_gap_mm", "small_x_gap_mm", "near_dist_gap_mm", "wide_x_gap_mm"):
        out[key] = _coerce_float(
            raw.get(key),
            DEFAULT_X_DIST_CURVE_POLICY[key],
            minimum=0.0,
        )
    out["large_dist_small_x_strength"] = _coerce_curve_strength(
        raw.get("large_dist_small_x_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["large_dist_small_x_strength"],
    )
    out["near_wide_x_strength"] = _coerce_curve_strength(
        raw.get("near_wide_x_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["near_wide_x_strength"],
    )
    if out["large_dist_small_x_strength"] == "adaptive":
        out["large_dist_small_x_strength"] = "gentle"
    if out["near_wide_x_strength"] == "adaptive":
        out["near_wide_x_strength"] = "strong"
    drive_mode = str(
        raw.get(
            "too_close_wide_x_drive_mode",
            DEFAULT_X_DIST_CURVE_POLICY["too_close_wide_x_drive_mode"],
        )
    ).strip().lower()
    out["too_close_wide_x_drive_mode"] = drive_mode if drive_mode in {"forward", "backward"} else "backward"
    return out


def _too_close_escape_policy() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("too_close_escape") if isinstance(cfg.get("too_close_escape"), dict) else {}
    approved_pwm = int(_approved_straight_drive_pwm("b"))
    out = {
        "pwm": _coerce_int(
            raw.get("pwm"),
            approved_pwm or DEFAULT_TOO_CLOSE_ESCAPE_POLICY["pwm"],
            minimum=1,
            maximum=max(1, approved_pwm or DEFAULT_TOO_CLOSE_ESCAPE_POLICY["pwm"]),
        ),
        "pulse_ms": _coerce_int(
            raw.get("pulse_ms"),
            DEFAULT_TOO_CLOSE_ESCAPE_POLICY["pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "min_pulse_ms": _coerce_int(
            raw.get("min_pulse_ms"),
            DEFAULT_TOO_CLOSE_ESCAPE_POLICY["min_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "full_escape_gap_mm": _coerce_float(
            raw.get("full_escape_gap_mm"),
            DEFAULT_TOO_CLOSE_ESCAPE_POLICY["full_escape_gap_mm"],
            minimum=0.1,
        ),
        "attach_mast": bool(raw.get("attach_mast", DEFAULT_TOO_CLOSE_ESCAPE_POLICY["attach_mast"])),
    }
    if int(out["min_pulse_ms"]) > int(out["pulse_ms"]):
        out["min_pulse_ms"], out["pulse_ms"] = int(out["pulse_ms"]), int(out["min_pulse_ms"])
    return out


def _win_confirmation_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("win_confirmation") if isinstance(cfg.get("win_confirmation"), dict) else {}
    return {
        "settle_s": _coerce_float(
            raw.get("settle_s"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["settle_s"],
            minimum=0.0,
            maximum=2.0,
        ),
        "confirm_frames": _coerce_int(
            raw.get("confirm_frames"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["confirm_frames"],
            minimum=1,
            maximum=10,
        ),
        "min_axis_closeness_pct": _coerce_float(
            raw.get("min_axis_closeness_pct"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["min_axis_closeness_pct"],
            minimum=0.0,
            maximum=100.0,
        ),
    }


def _follow_y_axis_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    out = dict(DEFAULT_FOLLOW_Y_AXIS_CONFIG)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    for key, fallback in DEFAULT_FOLLOW_Y_AXIS_CONFIG.items():
        if key == "enabled":
            continue
        minimum = 0.0 if key not in {"win_target_mm", "reset_target_mm"} else None
        out[key] = _coerce_float(raw.get(key), fallback, minimum=minimum)
    return out


def _x_outside_gate_mm(x_err: float) -> float:
    return max(0.0, abs(float(x_err)) - float(X_TOL_MM))


def _dist_outside_gate_mm(dist_err: float) -> float:
    return max(0.0, abs(float(dist_err)) - float(DIST_TOL_MM))


def _y_err_for_reading(reading: dict, *, target: float | None = None) -> float | None:
    try:
        y_mm = float((reading or {}).get("y_mm"))
    except (TypeError, ValueError):
        return None
    target_val = float(Y_TARGET_MM if target is None else target)
    return float(y_mm - target_val)


def _target_closeness_pct(error: float, tolerance: float) -> float:
    try:
        abs_error = abs(float(error))
        tol = float(tolerance)
    except (TypeError, ValueError):
        return 0.0
    if tol <= 0.0:
        return 100.0 if abs_error <= 0.0 else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - (abs_error / tol))))


def _win_min_axis_closeness_pct() -> float:
    cfg = _win_confirmation_config()
    return _coerce_float(
        cfg.get("min_axis_closeness_pct"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["min_axis_closeness_pct"],
        minimum=0.0,
        maximum=100.0,
    )


def _win_axis_ok(error: float, tolerance: float) -> bool:
    try:
        abs_error = abs(float(error))
    except (TypeError, ValueError):
        return False
    return float(abs_error) <= float(_win_effective_tolerance(tolerance))


def _win_effective_tolerance(tolerance: float) -> float:
    try:
        tol = float(tolerance)
    except (TypeError, ValueError):
        return 0.0
    min_close = _win_min_axis_closeness_pct()
    return max(0.0, float(tol) * (1.0 - (float(min_close) / 100.0)))


def _band_target_closeness_pct(value: float, *, target: float, minimum: float, maximum: float) -> float:
    try:
        val = float(value)
        target_val = float(target)
        min_val = float(minimum)
        max_val = float(maximum)
    except (TypeError, ValueError):
        return 0.0
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    tolerance = target_val - min_val if val <= target_val else max_val - target_val
    return _target_closeness_pct(val - target_val, tolerance)


def _avg(values) -> float | None:
    cleaned = []
    for value in values or []:
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return None
    return float(sum(cleaned) / float(len(cleaned)))


def _stddev(values) -> float | None:
    cleaned = []
    for value in values or []:
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    if len(cleaned) < 2:
        return 0.0 if cleaned else None
    avg = sum(cleaned) / float(len(cleaned))
    variance = sum((float(value) - avg) ** 2 for value in cleaned) / float(len(cleaned))
    return float(variance ** 0.5)


def _pct_text(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.0f}%"


def _fmt_mm(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.1f}mm"


def _pct_avg_std_text(values) -> str:
    avg = _avg(values)
    std = _stddev(values)
    if avg is None:
        return "N/A"
    return f"{float(avg):.0f}%±{float(std or 0.0):.0f}%"


def _bias_strength_for_x_outside(x_outside_mm: float) -> str:
    policy = _follow_combined_gap_policy()
    x_gap = max(0.0, float(x_outside_mm))
    if x_gap <= float(policy.get("micro_x_outside_max_mm", 0.0)):
        return "micro"
    if x_gap <= float(policy.get("gentle_x_outside_max_mm", 0.0)):
        return "gentle"
    if x_gap <= float(policy.get("medium_x_outside_max_mm", 0.0)):
        return "medium"
    return "strong"


def _bias_strength_for_dist_x(*, dist_err: float, x_err: float, x_outside_mm: float) -> str:
    policy = _follow_x_dist_curve_policy()
    if (
        float(dist_err) >= float(policy.get("large_dist_gap_mm", DEFAULT_X_DIST_CURVE_POLICY["large_dist_gap_mm"]))
        and abs(float(x_err)) <= float(policy.get("small_x_gap_mm", DEFAULT_X_DIST_CURVE_POLICY["small_x_gap_mm"]))
    ):
        return str(policy.get("large_dist_small_x_strength", "gentle"))
    return _bias_strength_for_x_outside(x_outside_mm)


def _near_wide_x_turn_strength(abs_x_err: float) -> str:
    policy = _follow_x_dist_curve_policy()
    if abs(float(abs_x_err)) >= float(policy.get("wide_x_gap_mm", DEFAULT_X_DIST_CURVE_POLICY["wide_x_gap_mm"])):
        return str(policy.get("near_wide_x_strength", "strong"))
    return _curve_strength_for_abs_x_err(abs(float(abs_x_err)))


def _should_back_turn_for_too_close_wide_x(*, dist_err: float, x_err: float) -> bool:
    policy = _follow_x_dist_curve_policy()
    return (
        float(dist_err) <= float(policy.get("near_dist_gap_mm", DEFAULT_X_DIST_CURVE_POLICY["near_dist_gap_mm"]))
        and abs(float(x_err)) >= float(policy.get("wide_x_gap_mm", DEFAULT_X_DIST_CURVE_POLICY["wide_x_gap_mm"]))
    )


def _proportional_duration_ms(*, gap_mm: float, min_ms: int, max_ms: int, full_gap_mm: float) -> int:
    low = _coerce_int(min_ms, PULSE_MS, minimum=1, maximum=_max_act_ms())
    high = _coerce_int(max_ms, low, minimum=1, maximum=_max_act_ms())
    if low > high:
        low, high = high, low
    span = max(0, int(high) - int(low))
    if span <= 0:
        return _bounded_act_duration_ms(low)
    full_gap = max(0.1, float(full_gap_mm))
    ratio = max(0.0, min(1.0, float(gap_mm) / float(full_gap)))
    return _bounded_act_duration_ms(int(round(float(low) + (float(span) * ratio))))


def _distance_creep_duration_ms(dist_err: float) -> int:
    policy = _follow_dist_approach_policy()
    gap = max(0.0, float(dist_err) - _win_effective_tolerance(DIST_TOL_MM))
    return _proportional_duration_ms(
        gap_mm=gap,
        min_ms=int(policy.get("min_forward_pulse_ms", PULSE_MS)),
        max_ms=int(policy.get("max_forward_pulse_ms", PULSE_MS)),
        full_gap_mm=float(policy.get("full_forward_gap_mm", DEFAULT_DIST_APPROACH_POLICY["full_forward_gap_mm"])),
    )


def _too_close_escape_duration_ms(dist_err: float, escape_policy: dict | None = None) -> int:
    policy = escape_policy if isinstance(escape_policy, dict) else _too_close_escape_policy()
    gap = max(0.0, abs(float(dist_err)) - _win_effective_tolerance(DIST_TOL_MM))
    return _proportional_duration_ms(
        gap_mm=gap,
        min_ms=int(policy.get("min_pulse_ms", PULSE_MS)),
        max_ms=int(policy.get("pulse_ms", DEFAULT_TOO_CLOSE_ESCAPE_POLICY["pulse_ms"])),
        full_gap_mm=float(policy.get("full_escape_gap_mm", DEFAULT_TOO_CLOSE_ESCAPE_POLICY["full_escape_gap_mm"])),
    )


def _should_close_x_before_distance(*, abs_x_err: float, dist_err: float) -> bool:
    policy = _follow_x_priority_policy()
    abs_x = abs(float(abs_x_err))
    if abs_x <= float(policy.get("polish_abs_x_mm", DEFAULT_X_PRIORITY_POLICY["polish_abs_x_mm"])):
        return False
    if (
        float(dist_err) >= float(policy.get("huge_dist_gap_mm", DEFAULT_X_PRIORITY_POLICY["huge_dist_gap_mm"]))
        and abs_x <= float(policy.get("huge_dist_tiny_abs_x_mm", DEFAULT_X_PRIORITY_POLICY["huge_dist_tiny_abs_x_mm"]))
    ):
        return False
    return True


def _near_target_forward_veto_active(*, dist_err: float, x_ok: bool, y_ok: bool) -> bool:
    policy = _follow_dist_approach_policy()
    veto_mm = float(policy.get("near_target_forward_veto_mm", 0.0))
    if veto_mm <= 0.0:
        return False
    if bool(x_ok) and bool(y_ok):
        return False
    return _win_effective_tolerance(DIST_TOL_MM) < float(dist_err) <= float(veto_mm)


def _finish_y_dist_ok(dist_err: float, y_cfg: dict | None = None) -> bool:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    dist_fallback = _coerce_float(
        cfg.get("finish_y_only_dist_deadband_mm"),
        DIST_TOL_MM,
        minimum=0.0,
    )
    too_far_deadband = _coerce_float(
        cfg.get("finish_y_only_too_far_deadband_mm"),
        dist_fallback,
        minimum=0.0,
    )
    too_close_deadband = _coerce_float(
        cfg.get("finish_y_only_too_close_deadband_mm"),
        dist_fallback,
        minimum=0.0,
    )
    dist_val = float(dist_err)
    if dist_val < 0.0:
        return abs(dist_val) <= float(too_close_deadband)
    return dist_val <= float(too_far_deadband)


def _should_finish_y_before_wheels(y_plan: dict | None, *, dist_err: float, x_err: float) -> bool:
    if not isinstance(y_plan, dict):
        return False
    y_cfg = _follow_y_axis_config()
    x_deadband = _coerce_float(
        y_cfg.get("finish_y_only_x_deadband_mm"),
        X_TOL_MM,
        minimum=0.0,
    )
    return _finish_y_dist_ok(dist_err, y_cfg) and abs(float(x_err)) <= float(x_deadband)


def _y_axis_action_plan(reading: dict, *, dist_err: float, x_err: float) -> dict | None:
    y_cfg = _follow_y_axis_config()
    if not bool(y_cfg.get("enabled")):
        return None
    try:
        y_mm = float((reading or {}).get("y_mm"))
    except (TypeError, ValueError):
        return None
    target = float(y_cfg.get("win_target_mm", Y_TARGET_MM))
    tol = float(y_cfg.get("win_tol_mm", Y_TOL_MM))
    high_target = target * float(y_cfg.get("approach_high_factor", 1.3))
    protect_below = float(y_cfg.get("protect_below_y_mm", max(high_target + tol, target + (3.0 * tol))))
    near_end = (
        abs(float(dist_err)) <= float(y_cfg.get("endgame_dist_tol_mm", DIST_TOL_MM))
        and abs(float(x_err)) <= float(y_cfg.get("endgame_x_tol_mm", X_TOL_MM))
    ) or (
        _finish_y_dist_ok(dist_err, y_cfg)
        and abs(float(x_err)) <= float(y_cfg.get("finish_y_only_x_deadband_mm", X_TOL_MM))
    )
    if bool((reading or {}).get("brick_below")) or y_mm > protect_below:
        return {
            "kind": "mast",
            "cmd": "d",
            "action": "MAST_D_PROTECT",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": float(y_mm - target),
            "y_mm": float(y_mm),
            "y_target_mm": float(target),
            "reason": "protect_lower_edge",
        }
    active_target = target if near_end else high_target
    active_tol = _win_effective_tolerance(tol) if near_end else tol
    y_err = y_mm - active_target
    if abs(y_err) <= active_tol:
        return None
    cmd = "d" if y_err > 0.0 else "u"
    pwm_key = "finish_mast_pwm" if near_end else "mast_pwm"
    pulse_key = "finish_mast_pulse_ms" if near_end else "mast_pulse_ms"
    return {
        "kind": "mast",
        "cmd": cmd,
        "action": f"MAST_{cmd.upper()}",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "y_err": float(y_err),
        "y_mm": float(y_mm),
        "y_target_mm": float(active_target),
        "pwm": int(_coerce_int(y_cfg.get(pwm_key), y_cfg.get("mast_pwm", 40), minimum=1, maximum=255)),
        "duration_ms": int(_bounded_act_duration_ms(y_cfg.get(pulse_key, y_cfg.get("mast_pulse_ms", 220)))),
        "reason": "final_y" if near_end else "approach_high_y",
    }


def _attach_mast_to_plan(plan: dict, y_plan: dict | None) -> dict:
    if not isinstance(plan, dict) or not isinstance(y_plan, dict):
        return plan
    cmd = str(y_plan.get("cmd") or "").strip().lower()
    if cmd not in {"u", "d"}:
        return plan
    out = dict(plan)
    out["mast_cmd"] = cmd
    out["mast_reason"] = y_plan.get("reason")
    out["mast_y_err"] = y_plan.get("y_err")
    out["mast_y_target_mm"] = y_plan.get("y_target_mm")
    action = str(out.get("action") or "").strip()
    if action:
        out["action"] = f"{action}_MAST_{cmd.upper()}"
    return out


def _follow_action_plan(reading: dict) -> dict:
    dist_mm = float(reading["dist_mm"])
    x_mm = float(reading["x_mm"])
    dist_err = dist_mm - TARGET_DIST_MM
    x_err = x_mm
    x_ok = _win_axis_ok(x_err, X_TOL_MM)
    dist_ok = _win_axis_ok(dist_err, DIST_TOL_MM)
    dist_happy_tol = _win_effective_tolerance(DIST_TOL_MM)
    y_cfg = _follow_y_axis_config()
    y_plan = _y_axis_action_plan(reading, dist_err=dist_err, x_err=x_err)
    y_err = _y_err_for_reading(reading, target=float(y_cfg.get("win_target_mm", Y_TARGET_MM)))
    y_ok = (
        True
        if y_err is None or not bool(y_cfg.get("enabled"))
        else _win_axis_ok(float(y_err), float(y_cfg.get("win_tol_mm", Y_TOL_MM)))
    )

    if x_ok and dist_ok and y_ok:
        return {"kind": "hold", "action": "HAPPY", "dist_err": dist_err, "x_err": x_err, "y_err": y_err}
    if _should_finish_y_before_wheels(y_plan, dist_err=dist_err, x_err=x_err):
        return y_plan
    if dist_err < -dist_happy_tol:
        if _should_back_turn_for_too_close_wide_x(dist_err=dist_err, x_err=x_err):
            escape = _too_close_escape_policy()
            turn_cmd = _turn_cmd_to_close_x_gap(x_err) or "r"
            policy = _follow_x_dist_curve_policy()
            drive_mode = str(policy.get("too_close_wide_x_drive_mode", "backward"))
            strength = str(policy.get("near_wide_x_strength", "strong"))
            return _attach_mast_to_plan({
                "kind": "drive_bias",
                "cmd": "b" if drive_mode == "backward" else "f",
                "turn_cmd": turn_cmd,
                "drive_mode": drive_mode,
                "strength": strength,
                "action": f"BCK_BIAS_{turn_cmd.upper()}_{strength.upper()}" if drive_mode == "backward" else f"BIAS_{turn_cmd.upper()}_{strength.upper()}",
                "dist_err": dist_err,
                "x_err": x_err,
                "x_outside_mm": float(_x_outside_gate_mm(x_err)),
                "dist_outside_mm": float(_dist_outside_gate_mm(dist_err)),
                "duration_ms": _too_close_escape_duration_ms(dist_err, escape),
                "reason": "too_close_wide_x_back_turn",
            }, y_plan)
        escape = _too_close_escape_policy()
        duration_ms = _too_close_escape_duration_ms(dist_err, escape)
        plan = {
            "kind": "drive",
            "cmd": "b",
            "action": "BCK_ESCAPE",
            "dist_err": dist_err,
            "x_err": x_err,
            "pwm": int(escape["pwm"]),
            "duration_ms": int(duration_ms),
            "reason": "too_close_escape",
        }
        if bool(escape.get("attach_mast")):
            return _attach_mast_to_plan(plan, y_plan)
        return plan
    dist_approach = _follow_dist_approach_policy()
    if bool(dist_approach.get("require_y_ok_before_dist")) and y_plan is not None and x_ok:
        return y_plan
    if y_plan is not None and x_ok:
        try:
            priority_y_gap = abs(float(y_plan.get("y_err")))
            priority_threshold = float(y_cfg.get("priority_abs_err_mm", 14.0))
        except (TypeError, ValueError):
            priority_y_gap = 0.0
            priority_threshold = 0.0
        if priority_threshold > 0.0 and priority_y_gap >= priority_threshold:
            return y_plan
    if y_plan is not None and x_ok and dist_ok:
        return y_plan

    if not x_ok:
        turn_cmd = _turn_cmd_to_close_x_gap(x_err) or "r"
        x_outside = _x_outside_gate_mm(x_err)
        dist_outside = _dist_outside_gate_mm(dist_err)
        if _near_target_forward_veto_active(dist_err=dist_err, x_ok=x_ok, y_ok=y_ok):
            return _attach_mast_to_plan({
                "kind": "turn",
                "cmd": turn_cmd,
                "drive_mode": _x_only_turn_drive_mode_for_dist(dist_err),
                "strength": _near_wide_x_turn_strength(abs(x_err)),
                "action": f"TURN_{turn_cmd.upper()}",
                "dist_err": dist_err,
                "x_err": x_err,
                "x_outside_mm": float(x_outside),
                "dist_outside_mm": float(dist_outside),
                "reason": "near_target_x_before_forward",
            }, y_plan)
        if _should_close_x_before_distance(abs_x_err=abs(x_err), dist_err=dist_err):
            return _attach_mast_to_plan({
                "kind": "turn",
                "cmd": turn_cmd,
                "drive_mode": _x_only_turn_drive_mode_for_dist(dist_err),
                "strength": str(_follow_x_priority_policy().get("x_first_turn_strength", "strong")),
                "action": f"TURN_{turn_cmd.upper()}",
                "dist_err": dist_err,
                "x_err": x_err,
                "x_outside_mm": float(x_outside),
                "dist_outside_mm": float(dist_outside),
                "reason": "x_first_before_dist",
            }, y_plan)
        if dist_err > dist_happy_tol:
            policy = _follow_combined_gap_policy()
            if (
                abs(float(x_err)) <= _win_effective_tolerance(X_TOL_MM)
                and x_outside <= float(policy.get("straight_x_outside_max_mm", 0.0))
                and dist_outside >= float(policy.get("straight_dist_outside_min_mm", 0.0))
            ):
                return _attach_mast_to_plan({
                    "kind": "drive",
                    "cmd": "f",
                    "action": "FWD",
                    "dist_err": dist_err,
                    "x_err": x_err,
                    "x_outside_mm": float(x_outside),
                    "dist_outside_mm": float(dist_outside),
                    "duration_ms": _distance_creep_duration_ms(dist_err),
                    "distance_creep": True,
                    "reason": "dist_dominant_tiny_x_gap",
                }, y_plan)
            strength = _bias_strength_for_dist_x(dist_err=dist_err, x_err=x_err, x_outside_mm=x_outside)
            return _attach_mast_to_plan({
                "kind": "drive_bias",
                "cmd": "f",
                "turn_cmd": turn_cmd,
                "drive_mode": "forward",
                "strength": strength,
                "action": f"BIAS_{turn_cmd.upper()}_{strength.upper()}",
                "dist_err": dist_err,
                "x_err": x_err,
                "x_outside_mm": float(x_outside),
                "dist_outside_mm": float(dist_outside),
                "duration_ms": _distance_creep_duration_ms(dist_err),
                "distance_creep": True,
                "reason": "x_polish_while_creeping_dist",
            }, y_plan)
        return _attach_mast_to_plan({
            "kind": "turn",
            "cmd": turn_cmd,
            "drive_mode": _x_only_turn_drive_mode_for_dist(dist_err),
            "strength": str(_follow_x_priority_policy().get("x_first_turn_strength", "strong")),
            "action": f"TURN_{turn_cmd.upper()}",
            "dist_err": dist_err,
            "x_err": x_err,
            "x_outside_mm": float(x_outside),
            "dist_outside_mm": float(dist_outside),
            "reason": "x_first_before_dist",
        }, y_plan)

    return _attach_mast_to_plan({
        "kind": "drive",
        "cmd": "f",
        "action": "FWD",
        "dist_err": dist_err,
        "x_err": x_err,
        "duration_ms": _distance_creep_duration_ms(dist_err),
        "distance_creep": True,
        "reason": "dist_only_creep",
    }, y_plan)


def _execute_follow_action(robot: Robot, plan: dict, reading: dict) -> None:
    kind = str((plan or {}).get("kind") or "").strip().lower()
    if kind == "drive":
        return _drive(
            robot,
            str(plan.get("cmd") or "f"),
            reading,
            mast_cmd=plan.get("mast_cmd"),
            pwm=plan.get("pwm"),
            duration_ms=plan.get("duration_ms"),
        )
    elif kind == "drive_bias":
        return _send_drive_bias(
            robot,
            turn_cmd=str(plan.get("turn_cmd") or "l"),
            drive_mode=str(plan.get("drive_mode") or "forward"),
            strength=str(plan.get("strength") or "micro"),
            duration_ms=_bounded_act_duration_ms(plan.get("duration_ms", PULSE_MS)),
            reading=reading,
            context="follow_drive_bias",
            mast_cmd=plan.get("mast_cmd"),
        )
    elif kind == "turn":
        return _send_turn_curve(
            robot,
            cmd=str(plan.get("cmd") or "l"),
            drive_mode=str(plan.get("drive_mode") or _x_only_turn_drive_mode()),
            strength=str(plan.get("strength") or _curve_strength_for_reading(reading)),
            duration_ms=_bounded_act_duration_ms(PULSE_MS),
            reading=reading,
            context="follow_x_only_turn_curve",
            mast_cmd=plan.get("mast_cmd"),
        )
    elif kind == "mast":
        return _mast(
            robot,
            str(plan.get("cmd") or "u"),
            reading,
            pwm=plan.get("pwm"),
            duration_ms=plan.get("duration_ms"),
        )
    else:
        _stop_robot(robot)
        return {"cmd_sent": "s", "pwm": 0, "power": 0.0, "duration_ms": 0}


def _x_curve_for_plan(plan: dict, reading: dict) -> dict | None:
    if not isinstance(plan, dict):
        return None
    kind = str(plan.get("kind") or "").strip().lower()
    if kind not in {"turn", "drive_bias"}:
        return None
    try:
        x_abs = abs(float((reading or {}).get("x_mm", plan.get("x_err", 0.0))))
    except (TypeError, ValueError):
        x_abs = 0.0
    drive_mode = str(plan.get("drive_mode") or _x_only_turn_drive_mode()).strip().lower()
    strength = str(plan.get("strength") or "").strip().lower()
    if kind == "drive_bias":
        return (
            _adaptive_turn_bias_curve_for_drive_mode(drive_mode, x_abs)
            if strength == "adaptive"
            else _turn_bias_curve_for_drive_mode(drive_mode, strength)
        )
    return (
        _adaptive_turn_curve_for_drive_mode(drive_mode, x_abs)
        if strength == "adaptive"
        else _turn_curve_for_drive_mode(drive_mode, strength)
    )


def _post_action_wait_s(plan: dict, send_result) -> float:
    if not isinstance(plan, dict) or not bool(plan.get("distance_creep")):
        return float(LOOP_S)
    policy = _follow_dist_approach_policy()
    try:
        duration_ms = float((send_result or {}).get("duration_ms"))
    except (AttributeError, TypeError, ValueError):
        try:
            duration_ms = float(plan.get("duration_ms"))
        except (TypeError, ValueError):
            duration_ms = float(PULSE_MS)
    shots = float(policy.get("closure_shots", DEFAULT_DIST_APPROACH_POLICY["closure_shots"]))
    settle_s = float(policy.get("settle_after_act_s", DEFAULT_DIST_APPROACH_POLICY["settle_after_act_s"]))
    readback_spacing_s = max(0.0, float(shots) - 1.0) * float(LOOP_S)
    return max(
        float(LOOP_S),
        (max(1.0, duration_ms) / 1000.0) + max(0.0, settle_s) + readback_spacing_s,
    )


def _reset_x_offset_ready(
    x_mm: float,
    dist_mm: float | None = None,
    reset_cfg: dict | None = None,
    y_mm: float | None = None,
) -> bool:
    """Check if both x offset and distance are within ready range."""
    if isinstance(dist_mm, dict) and reset_cfg is None:
        reset_cfg = dist_mm
        dist_mm = None
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}

    # Check x offset
    x_min = float(cfg.get("x_offset_min_mm", RESET_X_OFFSET_MIN_MM))
    x_max = float(cfg.get("x_offset_max_mm", RESET_X_OFFSET_MAX_MM))
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    abs_x = abs(float(x_mm))
    x_ok = float(x_min) <= float(abs_x) <= float(x_max)
    if dist_mm is None:
        return bool(x_ok)

    # Check distance
    dist_target = float(cfg.get("dist_target_mm", RESET_DIST_TARGET_MM))
    dist_tol = float(cfg.get("dist_tol_mm", RESET_DIST_TOL_MM))
    dist_ok = abs(float(dist_mm) - dist_target) <= dist_tol
    try:
        y_val = float(y_mm)
    except (TypeError, ValueError):
        y_val = None
    if y_val is not None and cfg.get("y_target_mm") is not None and cfg.get("y_tol_mm") is not None:
        y_ok = abs(y_val - float(cfg.get("y_target_mm"))) <= float(cfg.get("y_tol_mm"))
        return x_ok and dist_ok and y_ok

    return x_ok and dist_ok


def _reverse_turn_until_x_offset(
    vision: BrickDetector,
    robot: Robot,
    *,
    direction: str,
    rng=None,
) -> tuple[bool, str, dict | None]:
    """Reset sequence: one backward-turn pulse, then observe the result."""
    turn_cmd = str(direction or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        return False, "invalid_turn_direction", None

    cfg = _reset_motion_config().get("reverse_turn")
    reset_cfg = cfg if isinstance(cfg, dict) else {}
    x_min = float(reset_cfg.get("x_offset_min_mm", RESET_X_OFFSET_MIN_MM))
    x_max = float(reset_cfg.get("x_offset_max_mm", RESET_X_OFFSET_MAX_MM))
    target_abs_x = _coerce_float(
        reset_cfg.get("target_abs_x_mm"),
        (float(x_min) + float(x_max)) / 2.0,
        minimum=0.0,
    )
    if target_abs_x < x_min or target_abs_x > x_max:
        target_abs_x = (float(x_min) + float(x_max)) / 2.0
    dist_target = float(reset_cfg.get("dist_target_mm", RESET_DIST_TARGET_MM))
    dist_tol = float(reset_cfg.get("dist_tol_mm", RESET_DIST_TOL_MM))
    y_target = float(reset_cfg.get("y_target_mm", RESET_Y_TARGET_MM))
    y_tol = float(reset_cfg.get("y_tol_mm", Y_TOL_MM))
    settle_s = float(reset_cfg.get("settle_s", RESET_REVERSE_TURN_SETTLE_S))

    print(
        f"[RESET] One-act reset: BACK_TURN_{turn_cmd.upper()} "
        f"target dist={dist_target:.0f}±{dist_tol:.0f}mm, "
        f"|x|~{target_abs_x:.0f}mm ({x_min:.0f}-{x_max:.0f}mm), "
        f"y={y_target:+.0f}±{y_tol:.0f}mm",
        flush=True,
    )

    before_reading = _read_brick_measurement(vision)
    if not bool(before_reading.get("confident")):
        _stop_robot(robot)
        return False, "brick_not_confident_before_reset_motion", before_reading

    try:
        before_dist = float(before_reading["dist_mm"])
        before_x = float(before_reading["x_mm"])
    except (TypeError, ValueError):
        _stop_robot(robot)
        return False, "invalid_reset_start_reading", before_reading
    try:
        before_y_text = f"{float(before_reading.get('y_mm')):+.1f}mm"
    except (TypeError, ValueError):
        before_y_text = "N/A"

    reset_curve = _reset_arc_curve_for_reading(before_reading, reset_cfg)
    reset_motion = _reset_reverse_turn(robot, turn_cmd, before_reading, rng=rng)
    if reset_motion is None:
        _stop_robot(robot)
        return False, "reverse_turn_unavailable", before_reading
    pulse_ms = int(reset_motion.get("wheel_ms", reset_motion.get("duration_ms", 0)) or 0)
    gentle_ms = int(reset_motion.get("gentle_ms", 0) or 0)
    sharp_finish_ms = int(reset_motion.get("sharp_finish_ms", 0) or 0)
    mast_up_ms = int(reset_motion.get("mast_up_ms", 0) or 0)
    mast_settle_s = float(reset_motion.get("mast_settle_s", 0.0) or 0.0)
    mast_text = f" mast_up={mast_up_ms}ms" if mast_up_ms > 0 else " mast_up=off"
    finish_text = (
        f" gentle={gentle_ms}ms sharp_finish={sharp_finish_ms}ms"
        if sharp_finish_ms > 0
        else ""
    )

    print(
        f"[RESET] BACK_TURN_{turn_cmd.upper()} sent: "
        f"need_x_gap={reset_curve['x_gap_mm']:.1f}mm "
        f"faster_pwm={int(reset_curve['faster_pwm'])} slower_pwm={int(reset_curve['slower_pwm'])} "
        f"before dist={before_dist:.1f}mm x={before_x:+.1f}mm y={before_y_text} "
        f"wheel_pulse={int(pulse_ms)}ms{finish_text}{mast_text}",
        flush=True,
    )
    wheel_wait_s = (float(pulse_ms) / 1000.0) + float(settle_s)
    mast_wait_s = ((float(mast_up_ms) / 1000.0) + float(mast_settle_s)) if mast_up_ms > 0 else 0.0
    time.sleep(max(LOOP_S, float(wheel_wait_s), float(mast_wait_s)))
    _stop_robot(robot)
    pause_s = _reset_post_pause_s()
    if pause_s > 0.0:
        print(f"[RESET] Pause {pause_s:.1f}s before measuring reset result.", flush=True)
        time.sleep(pause_s)

    after_reading = _read_brick_measurement(vision)
    if not bool(after_reading.get("confident")):
        return False, "lost_confident_brick_after_reset", after_reading

    try:
        after_dist = float(after_reading["dist_mm"])
        after_x = float(after_reading["x_mm"])
    except (TypeError, ValueError):
        return False, "invalid_reset_after_reading", after_reading
    try:
        after_y_text = f"{float(after_reading.get('y_mm')):+.1f}mm"
    except (TypeError, ValueError):
        after_y_text = "N/A"

    try:
        after_y_for_gate = float(after_reading.get("y_mm"))
    except (TypeError, ValueError):
        after_y_for_gate = None
    target_met = _reset_x_offset_ready(after_x, after_dist, reset_cfg, y_mm=after_y_for_gate)
    closeness = _reset_closeness_from_reading(after_reading, reset_cfg)
    close_text = ""
    if closeness is not None:
        dist_close, x_close, y_close, combined_close = closeness
        y_text = "" if y_close is None else f" y={y_close:.0f}%"
        close_text = (
            f" close={combined_close:.0f}% "
            f"(dist={dist_close:.0f}% x={x_close:.0f}%{y_text})"
        )
    print(
        f"[RESET] One act complete: dist={after_dist:.1f}mm x={after_x:+.1f}mm y={after_y_text} "
        f"target={'hit' if target_met else 'miss'}{close_text}",
        flush=True,
    )
    return True, "target_hit" if target_met else "one_act_complete", after_reading


def _run_reset_sequence(
    vision: BrickDetector,
    robot: Robot,
    *,
    rng=None,
) -> dict:
    random_source = rng if rng is not None else random
    turn_cmd = random_source.choice(("l", "r"))
    offset_ok, offset_reason, offset_reading = _reverse_turn_until_x_offset(
        vision,
        robot,
        direction=turn_cmd,
        rng=random_source,
    )
    target_met = False
    if isinstance(offset_reading, dict):
        try:
            target_met = bool(
                _reset_x_offset_ready(
                    float(offset_reading.get("x_mm")),
                    float(offset_reading.get("dist_mm")),
                    y_mm=offset_reading.get("y_mm"),
                )
            )
        except (TypeError, ValueError):
            target_met = False
    result = {
        "success": bool(offset_ok),
        "phase": "reverse_turn",
        "reason": offset_reason,
        "turn_cmd": turn_cmd,
        "mast_up_sent": bool(offset_ok and _reset_mast_up_enabled()),
        "reading": offset_reading,
        "target_met": bool(target_met),
    }
    return result


def _new_game_stats() -> dict:
    return {
        "sample_count": 0,
        "confident_sample_count": 0,
        "not_confident_count": 0,
        "follow_attempt_count": 0,
        "act_counts": {},
        "sent_act_counts": {},
        "blocked_act_counts": {},
        "observed_after_act_counts": {},
        "no_observed_after_act_counts": {},
        "x_curve_samples": [],
        "miss_reasons": {},
        "non_win_dist_target_closeness_pct": [],
        "non_win_x_target_closeness_pct": [],
        "non_win_target_closeness_pct": [],
        "closest_non_win": None,
        "last_non_win": None,
        "pending_observation": None,
        "win_count": 0,
        "win_dist_target_closeness_pct": [],
        "win_x_target_closeness_pct": [],
        "win_y_target_closeness_pct": [],
        "win_target_closeness_pct": [],
        "reset_attempt_count": 0,
        "reset_count": 0,
        "reset_target_met_count": 0,
        "reset_x_after_mm": [],
        "reset_abs_x_after_mm": [],
        "reset_dist_after_mm": [],
        "reset_y_after_mm": [],
        "reset_dist_target_closeness_pct": [],
        "reset_x_target_closeness_pct": [],
        "reset_y_target_closeness_pct": [],
        "reset_target_closeness_pct": [],
        "last_reset_reason": None,
        "y_lock_on_armed": True,
    }


def _avg_reset_abs_x_after_mm(stats: dict) -> float | None:
    values = stats.get("reset_abs_x_after_mm") if isinstance(stats, dict) else None
    return _avg(values)


def _avg_reset_dist_after_mm(stats: dict) -> float | None:
    values = stats.get("reset_dist_after_mm") if isinstance(stats, dict) else None
    return _avg(values)


def _bump_count(mapping: dict, key: str, amount: int = 1) -> None:
    if not isinstance(mapping, dict):
        return
    key_text = str(key or "").strip() or "unknown"
    mapping[key_text] = int(mapping.get(key_text, 0)) + int(amount)


def _bump_stat_count(stats: dict, bucket: str, key: str, amount: int = 1) -> None:
    if not isinstance(stats, dict):
        return
    mapping = stats.setdefault(bucket, {})
    if isinstance(mapping, dict):
        _bump_count(mapping, key, amount)


def _avg_stat(stats: dict, key: str) -> float | None:
    values = stats.get(key) if isinstance(stats, dict) else None
    return _avg(values)


def _bar_pct(value: float | None, *, width: int = 10) -> str:
    if value is None:
        return "[" + ("?" * int(width)) + "]"
    pct = max(0.0, min(100.0, float(value)))
    filled = int(round((pct / 100.0) * int(width)))
    filled = max(0, min(int(width), int(filled)))
    return "[" + ("#" * filled) + ("-" * (int(width) - filled)) + "]"


def _record_win_stats(stats: dict, reading: dict, plan: dict) -> None:
    if not isinstance(stats, dict):
        return
    try:
        dist_err = float((plan or {}).get("dist_err"))
        x_err = float((plan or {}).get("x_err"))
    except (TypeError, ValueError):
        try:
            dist_err = float((reading or {}).get("dist_mm")) - float(TARGET_DIST_MM)
            x_err = float((reading or {}).get("x_mm"))
        except (TypeError, ValueError):
            return
    dist_closeness = _target_closeness_pct(dist_err, DIST_TOL_MM)
    x_closeness = _target_closeness_pct(x_err, X_TOL_MM)
    y_cfg = _follow_y_axis_config()
    y_closeness = None
    y_err = _y_err_for_reading(reading, target=float(y_cfg.get("win_target_mm", Y_TARGET_MM)))
    if y_err is not None and bool(y_cfg.get("enabled")):
        y_closeness = _target_closeness_pct(y_err, float(y_cfg.get("win_tol_mm", Y_TOL_MM)))
    stats.setdefault("win_dist_target_closeness_pct", []).append(float(dist_closeness))
    stats.setdefault("win_x_target_closeness_pct", []).append(float(x_closeness))
    if y_closeness is not None:
        stats.setdefault("win_y_target_closeness_pct", []).append(float(y_closeness))
        stats.setdefault("win_target_closeness_pct", []).append(float((dist_closeness + x_closeness + y_closeness) / 3.0))
    else:
        stats.setdefault("win_target_closeness_pct", []).append(float((dist_closeness + x_closeness) / 2.0))


def _miss_reason_for_plan(plan: dict) -> str:
    try:
        dist_err = float((plan or {}).get("dist_err"))
        x_err = float((plan or {}).get("x_err"))
    except (TypeError, ValueError):
        return "invalid_reading"
    dist_outside = abs(dist_err) > float(DIST_TOL_MM)
    x_outside = abs(x_err) > float(X_TOL_MM)
    if dist_outside and x_outside:
        return "dist_and_x_outside"
    if x_outside:
        return "x_outside"
    if dist_outside:
        return "too_far" if dist_err > 0.0 else "too_close"
    return "inside_target"


def _record_non_win_stats(
    stats: dict,
    reading: dict,
    plan: dict,
    *,
    action: str | None = None,
    reason: str | None = None,
) -> None:
    if not isinstance(stats, dict):
        return
    action_text = str(action or (plan or {}).get("action") or "UNKNOWN")
    reason_text = str(reason or _miss_reason_for_plan(plan))
    try:
        dist_err = float((plan or {}).get("dist_err"))
        x_err = float((plan or {}).get("x_err"))
        dist_mm = float((reading or {}).get("dist_mm"))
        x_mm = float((reading or {}).get("x_mm"))
    except (TypeError, ValueError):
        return
    dist_closeness = _target_closeness_pct(dist_err, DIST_TOL_MM)
    x_closeness = _target_closeness_pct(x_err, X_TOL_MM)
    combined_closeness = float((dist_closeness + x_closeness) / 2.0)
    stats.setdefault("non_win_dist_target_closeness_pct", []).append(float(dist_closeness))
    stats.setdefault("non_win_x_target_closeness_pct", []).append(float(x_closeness))
    stats.setdefault("non_win_target_closeness_pct", []).append(float(combined_closeness))
    snapshot = {
        "action": action_text,
        "reason": reason_text,
        "dist_mm": float(dist_mm),
        "x_mm": float(x_mm),
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "dist_closeness_pct": float(dist_closeness),
        "x_closeness_pct": float(x_closeness),
        "closeness_pct": float(combined_closeness),
    }
    stats["last_non_win"] = dict(snapshot)
    closest = stats.get("closest_non_win")
    if not isinstance(closest, dict) or combined_closeness > float(closest.get("closeness_pct", -1.0)):
        stats["closest_non_win"] = dict(snapshot)


def _record_follow_attempt_stats(stats: dict, reading: dict, plan: dict) -> None:
    if not isinstance(stats, dict):
        return
    stats["follow_attempt_count"] = int(stats.get("follow_attempt_count", 0)) + 1
    action = str((plan or {}).get("action") or "UNKNOWN")
    _bump_stat_count(stats, "act_counts", action)
    reason = _miss_reason_for_plan(plan)
    _bump_stat_count(stats, "miss_reasons", reason)
    _record_non_win_stats(stats, reading, plan, action=action, reason=reason)


def _record_send_result(stats: dict, action: str, send_result) -> None:
    if not isinstance(stats, dict):
        return
    action_key = str(action or "").strip() or "UNKNOWN"
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        _bump_stat_count(stats, "blocked_act_counts", action_key)
        _bump_stat_count(stats, "miss_reasons", f"send_blocked_{send_result.get('reason', 'unknown')}")
        return
    _bump_stat_count(stats, "sent_act_counts", action_key)


def _reset_motion_was_sent(reset_result: dict) -> bool:
    reason = str((reset_result or {}).get("reason") or "").strip()
    return reason not in {
        "invalid_turn_direction",
        "brick_not_confident_before_reset_motion",
        "invalid_reset_start_reading",
        "reverse_turn_unavailable",
    }


def _record_observed_after_pending_act(stats: dict, reading: dict) -> None:
    if not isinstance(stats, dict):
        return
    pending = stats.get("pending_observation")
    if not isinstance(pending, dict):
        return
    stats["pending_observation"] = None
    try:
        prev_dist = float(pending.get("dist_mm"))
        prev_x = float(pending.get("x_mm"))
        dist_mm = float((reading or {}).get("dist_mm"))
        x_mm = float((reading or {}).get("x_mm"))
    except (TypeError, ValueError):
        return
    delta_y = 0.0
    try:
        delta_y = abs(float((reading or {}).get("y_mm")) - float(pending.get("y_mm")))
    except (TypeError, ValueError):
        delta_y = 0.0
    delta_dist = abs(float(dist_mm) - float(prev_dist))
    delta_x = abs(float(x_mm) - float(prev_x))
    action = str(pending.get("action") or "UNKNOWN")
    if delta_dist >= 1.0 or delta_x >= 1.0 or delta_y >= 1.0:
        _bump_stat_count(stats, "observed_after_act_counts", action)
    else:
        _bump_stat_count(stats, "no_observed_after_act_counts", action)
        _bump_stat_count(stats, "miss_reasons", "no_observed_change_after_act")
    x_curve = pending.get("x_curve")
    if isinstance(x_curve, dict):
        before_abs = abs(float(prev_x))
        after_abs = abs(float(x_mm))
        sample = {
            "action": action,
            "x_before_mm": float(prev_x),
            "x_after_mm": float(x_mm),
            "abs_x_before_mm": float(before_abs),
            "abs_x_after_mm": float(after_abs),
            "x_reduction_mm": float(before_abs - after_abs),
            "x_overshot": bool(prev_x and x_mm and (prev_x > 0.0) != (x_mm > 0.0)),
            "drive_mode": str(x_curve.get("drive_mode") or ""),
            "strength": str(x_curve.get("strength") or ""),
            "inner_pwm": int(x_curve.get("inner_pwm", 0) or 0),
            "outer_pwm": int(x_curve.get("outer_pwm", 0) or 0),
            "duration_ms": int(pending.get("duration_ms", 0) or 0),
        }
        stats.setdefault("x_curve_samples", []).append(sample)


def _reset_closeness_from_reading(reading: dict, reset_cfg: dict | None = None) -> tuple[float, float, float | None, float] | None:
    if not isinstance(reading, dict):
        return None
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    try:
        dist_mm = float(reading.get("dist_mm"))
        abs_x = abs(float(reading.get("x_mm")))
    except (TypeError, ValueError):
        return None
    dist_target = _coerce_float(cfg.get("dist_target_mm"), RESET_DIST_TARGET_MM, minimum=0.0)
    dist_tol = _coerce_float(cfg.get("dist_tol_mm"), RESET_DIST_TOL_MM, minimum=0.0)
    x_min = _coerce_float(cfg.get("x_offset_min_mm"), RESET_X_OFFSET_MIN_MM, minimum=0.0)
    x_max = _coerce_float(cfg.get("x_offset_max_mm"), RESET_X_OFFSET_MAX_MM, minimum=0.0)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    target_abs_x = _coerce_float(
        cfg.get("target_abs_x_mm"),
        (float(x_min) + float(x_max)) / 2.0,
        minimum=0.0,
    )
    if target_abs_x < x_min or target_abs_x > x_max:
        target_abs_x = (float(x_min) + float(x_max)) / 2.0
    dist_closeness = _target_closeness_pct(dist_mm - dist_target, dist_tol)
    x_closeness = _band_target_closeness_pct(
        abs_x,
        target=target_abs_x,
        minimum=x_min,
        maximum=x_max,
    )
    y_closeness = None
    try:
        y_mm = float(reading.get("y_mm"))
        y_target = _coerce_float(cfg.get("y_target_mm"), _follow_y_axis_config().get("reset_target_mm"))
        y_tol = _coerce_float(cfg.get("y_tol_mm"), _follow_y_axis_config().get("reset_tol_mm"), minimum=0.0)
        y_closeness = _target_closeness_pct(y_mm - y_target, y_tol)
    except (TypeError, ValueError):
        y_closeness = None
    values = [float(dist_closeness), float(x_closeness)]
    if y_closeness is not None:
        values.append(float(y_closeness))
    return float(dist_closeness), float(x_closeness), None if y_closeness is None else float(y_closeness), float(sum(values) / float(len(values)))


def _record_reset_stats(stats: dict, reset_result: dict) -> None:
    if not isinstance(stats, dict) or not isinstance(reset_result, dict):
        return
    stats["reset_attempt_count"] = int(stats.get("reset_attempt_count", 0)) + 1
    stats["last_reset_reason"] = reset_result.get("reason")
    turn_cmd = str(reset_result.get("turn_cmd") or "").strip().upper()
    if turn_cmd in {"L", "R"}:
        action_key = f"RESET_BACK_TURN_{turn_cmd}"
        _bump_stat_count(stats, "act_counts", action_key)
        if _reset_motion_was_sent(reset_result):
            _bump_stat_count(stats, "sent_act_counts", action_key)
    if bool(reset_result.get("mast_up_sent")):
        _bump_stat_count(stats, "act_counts", "RESET_MAST_U")
        _bump_stat_count(stats, "sent_act_counts", "RESET_MAST_U")
    reading = reset_result.get("reading")
    if not isinstance(reading, dict):
        return
    try:
        x_after = float(reading.get("x_mm"))
    except (TypeError, ValueError):
        return
    stats["reset_count"] = int(stats.get("reset_count", 0)) + 1
    if bool(reset_result.get("target_met")):
        stats["reset_target_met_count"] = int(stats.get("reset_target_met_count", 0)) + 1
    stats.setdefault("reset_x_after_mm", []).append(float(x_after))
    stats.setdefault("reset_abs_x_after_mm", []).append(abs(float(x_after)))
    try:
        stats.setdefault("reset_dist_after_mm", []).append(float(reading.get("dist_mm")))
    except (TypeError, ValueError):
        pass
    try:
        stats.setdefault("reset_y_after_mm", []).append(float(reading.get("y_mm")))
    except (TypeError, ValueError):
        pass
    closeness = _reset_closeness_from_reading(reading)
    if closeness is not None:
        dist_closeness, x_closeness, y_closeness, combined_closeness = closeness
        stats.setdefault("reset_dist_target_closeness_pct", []).append(float(dist_closeness))
        stats.setdefault("reset_x_target_closeness_pct", []).append(float(x_closeness))
        if y_closeness is not None:
            stats.setdefault("reset_y_target_closeness_pct", []).append(float(y_closeness))
        stats.setdefault("reset_target_closeness_pct", []).append(float(combined_closeness))


def _print_reset_stats(stats: dict) -> None:
    reset_values = stats.get("reset_x_after_mm") if isinstance(stats, dict) else None
    reset_dist_values = stats.get("reset_dist_after_mm") if isinstance(stats, dict) else None
    reset_y_values = stats.get("reset_y_after_mm") if isinstance(stats, dict) else None
    last_x = float(reset_values[-1]) if reset_values else None
    last_dist = float(reset_dist_values[-1]) if reset_dist_values else None
    last_y = float(reset_y_values[-1]) if reset_y_values else None
    avg_x = _avg_reset_abs_x_after_mm(stats)
    avg_dist = _avg_reset_dist_after_mm(stats)
    avg_y = _avg(reset_y_values)
    last_text = "N/A" if last_x is None else f"{last_x:+.1f}mm"
    last_dist_text = "N/A" if last_dist is None else f"{last_dist:.1f}mm"
    last_y_text = "N/A" if last_y is None else f"{last_y:+.1f}mm"
    avg_x_text = "N/A" if avg_x is None else f"{avg_x:.1f}mm"
    avg_dist_text = "N/A" if avg_dist is None else f"{avg_dist:.1f}mm"
    avg_y_text = "N/A" if avg_y is None else f"{avg_y:.1f}mm"
    win_close_val = _avg_stat(stats, "win_target_closeness_pct")
    reset_close_val = _avg_stat(stats, "reset_target_closeness_pct")
    win_close = _pct_text(win_close_val)
    reset_close = _pct_text(reset_close_val)
    win_dist_close = _pct_text(_avg_stat(stats, "win_dist_target_closeness_pct"))
    win_x_close = _pct_text(_avg_stat(stats, "win_x_target_closeness_pct"))
    win_y_close = _pct_text(_avg_stat(stats, "win_y_target_closeness_pct"))
    reset_dist_close = _pct_text(_avg_stat(stats, "reset_dist_target_closeness_pct"))
    reset_x_close = _pct_text(_avg_stat(stats, "reset_x_target_closeness_pct"))
    reset_y_close = _pct_text(_avg_stat(stats, "reset_y_target_closeness_pct"))
    print(
        f"[STATS] win_target wins={int(stats.get('win_count', 0))} "
        f"avg={win_close}{_bar_pct(win_close_val)} dist/x/y={win_dist_close}/{win_x_close}/{win_y_close} | "
        f"reset_target samples={int(stats.get('reset_count', 0))}/{int(stats.get('reset_attempt_count', 0))} "
        f"hits={int(stats.get('reset_target_met_count', 0))}/{int(stats.get('reset_count', 0))} "
        f"avg={reset_close}{_bar_pct(reset_close_val)} dist/x/y={reset_dist_close}/{reset_x_close}/{reset_y_close} "
        f"last=dist {last_dist_text}, x {last_text}, y {last_y_text} "
        f"avg_after=dist {avg_dist_text}, |x| {avg_x_text}, y {avg_y_text}",
        flush=True,
    )


def _format_count_items(mapping: dict | None, *, empty: str = "none") -> str:
    if not isinstance(mapping, dict) or not mapping:
        return empty
    items = sorted(mapping.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{key}={int(value)}" for key, value in items)


def _format_snapshot(snapshot: dict | None) -> str:
    if not isinstance(snapshot, dict):
        return "N/A"
    try:
        close = float(snapshot.get("closeness_pct"))
        dist_close = float(snapshot.get("dist_closeness_pct"))
        x_close = float(snapshot.get("x_closeness_pct"))
        dist_err = float(snapshot.get("dist_err"))
        x_err = float(snapshot.get("x_err"))
    except (TypeError, ValueError):
        return "N/A"
    return (
        f"{close:.0f}% (dist={dist_close:.0f}% x={x_close:.0f}%, "
        f"dist_err={dist_err:+.1f}mm x_err={x_err:+.1f}mm, "
        f"{snapshot.get('action', 'UNKNOWN')})"
    )


def _format_x_curve_learning(samples: list | None) -> list[str]:
    rows = [
        "",
        "| X Curve Learning | Value |",
        "|---|---:|",
    ]
    if not isinstance(samples, list) or not samples:
        rows.append("| Samples | 0 |")
        return rows
    reductions = [float(s.get("x_reduction_mm", 0.0)) for s in samples if isinstance(s, dict)]
    before_vals = [float(s.get("abs_x_before_mm", 0.0)) for s in samples if isinstance(s, dict)]
    after_vals = [float(s.get("abs_x_after_mm", 0.0)) for s in samples if isinstance(s, dict)]
    overshoots = sum(1 for s in samples if isinstance(s, dict) and bool(s.get("x_overshot")))
    improved = sum(1 for value in reductions if value > 0.0)
    rows.extend(
        [
            f"| Samples | {len(reductions)} |",
            f"| Improved x | {improved}/{len(reductions)} |",
            f"| Overshot x sign | {overshoots}/{len(reductions)} |",
            f"| Avg |x| before | {_fmt_mm(_avg(before_vals))} |",
            f"| Avg |x| after | {_fmt_mm(_avg(after_vals))} |",
            f"| Avg x reduction | {_fmt_mm(_avg(reductions))} |",
        ]
    )
    latest = next((s for s in reversed(samples) if isinstance(s, dict)), None)
    if latest:
        rows.append(
            "| Last curve | "
            f"{latest.get('action')} {latest.get('drive_mode')} {latest.get('strength')} "
            f"{latest.get('inner_pwm')}/{latest.get('outer_pwm')} pwm, "
            f"x {float(latest.get('x_before_mm', 0.0)):+.1f}->{float(latest.get('x_after_mm', 0.0)):+.1f}mm |"
        )
    return rows


def _format_game_results_table(stats: dict) -> str:
    avg_x = _avg_reset_abs_x_after_mm(stats)
    avg_dist = _avg_reset_dist_after_mm(stats)
    avg_y = _avg(stats.get("reset_y_after_mm") if isinstance(stats, dict) else None)
    avg_dist_after_text = "N/A" if avg_dist is None else f"{avg_dist:.1f}mm"
    avg_x_after_text = "N/A" if avg_x is None else f"{avg_x:.1f}mm"
    avg_y_after_text = "N/A" if avg_y is None else f"{avg_y:.1f}mm"
    avg_after_text = (
        "N/A"
        if avg_x is None and avg_dist is None and avg_y is None
        else f"dist {avg_dist_after_text}, |x| {avg_x_after_text}, y {avg_y_after_text}"
    )
    avg_win_close_val = _avg_stat(stats, "win_target_closeness_pct")
    avg_reset_close_val = _avg_stat(stats, "reset_target_closeness_pct")
    win_close_values = stats.get("win_target_closeness_pct")
    win_dist_values = stats.get("win_dist_target_closeness_pct")
    win_x_values = stats.get("win_x_target_closeness_pct")
    win_y_values = stats.get("win_y_target_closeness_pct")
    reset_close_values = stats.get("reset_target_closeness_pct")
    reset_dist_values = stats.get("reset_dist_target_closeness_pct")
    reset_x_values = stats.get("reset_x_target_closeness_pct")
    reset_y_values = stats.get("reset_y_target_closeness_pct")
    attempts = int(stats.get("follow_attempt_count", 0))
    confident = int(stats.get("confident_sample_count", 0))
    sample_count = int(stats.get("sample_count", 0))
    not_confident = int(stats.get("not_confident_count", 0))
    wins = int(stats.get("win_count", 0))
    reset_attempts = int(stats.get("reset_attempt_count", 0))
    reset_hits = int(stats.get("reset_target_met_count", 0))
    act_counts = _format_count_items(stats.get("act_counts"))
    sent_act_counts = _format_count_items(stats.get("sent_act_counts"))
    blocked_act_counts = _format_count_items(stats.get("blocked_act_counts"))
    observed_counts = _format_count_items(stats.get("observed_after_act_counts"))
    no_observed_counts = _format_count_items(stats.get("no_observed_after_act_counts"))
    miss_reasons = _format_count_items(stats.get("miss_reasons"))
    closest_non_win = _format_snapshot(stats.get("closest_non_win"))
    last_non_win = _format_snapshot(stats.get("last_non_win"))
    rows = [
            "| Target | Samples | Hits | Close avg±sd | Dist avg±sd | X avg±sd | Y avg±sd | Avg after |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            f"| Win | {int(stats.get('win_count', 0))} | {int(stats.get('win_count', 0))} | "
            f"{_pct_avg_std_text(win_close_values)} {_bar_pct(avg_win_close_val)} | "
            f"{_pct_avg_std_text(win_dist_values)} | {_pct_avg_std_text(win_x_values)} | {_pct_avg_std_text(win_y_values)} | "
            f"target dist {TARGET_DIST_MM:.1f}mm, x 0.0mm, y {Y_TARGET_MM:.1f}mm |",
            f"| Reset | {int(stats.get('reset_count', 0))}/{int(stats.get('reset_attempt_count', 0))} | "
            f"{int(stats.get('reset_target_met_count', 0))}/{int(stats.get('reset_count', 0))} | "
            f"{_pct_avg_std_text(reset_close_values)} {_bar_pct(avg_reset_close_val)} | "
            f"{_pct_avg_std_text(reset_dist_values)} | {_pct_avg_std_text(reset_x_values)} | {_pct_avg_std_text(reset_y_values)} | {avg_after_text} |",
            "",
            "| Attempts | Count |",
            "|---|---:|",
            f"| Samples | {sample_count} |",
            f"| Confident samples | {confident} |",
            f"| Not confident samples | {not_confident} |",
            f"| Movement attempts | {attempts} |",
            f"| Wins | {wins} |",
            f"| Reset attempts | {reset_attempts} |",
            f"| Reset target hits | {reset_hits} |",
            "",
            "| Movement acts | Count |",
            "|---|---:|",
            f"| Planned: {act_counts} | {sum(int(v) for v in (stats.get('act_counts') or {}).values())} |",
            f"| Sent: {sent_act_counts} | {sum(int(v) for v in (stats.get('sent_act_counts') or {}).values())} |",
            f"| Blocked: {blocked_act_counts} | {sum(int(v) for v in (stats.get('blocked_act_counts') or {}).values())} |",
            f"| Observed change after act: {observed_counts} | {sum(int(v) for v in (stats.get('observed_after_act_counts') or {}).values())} |",
            f"| No observed change after act: {no_observed_counts} | {sum(int(v) for v in (stats.get('no_observed_after_act_counts') or {}).values())} |",
            "",
            "| Why not more wins? | Evidence |",
            "|---|---|",
            f"| Miss reasons | {miss_reasons} |",
            f"| Closest non-win | {closest_non_win} |",
            f"| Last non-win | {last_non_win} |",
        ]
    rows.extend(_format_x_curve_learning(stats.get("x_curve_samples")))
    return "\n".join(rows)


INVISIBLE_STOP_FRAMES = 1   # stop motors on the first not-visible frame


def _confirm_stopped_happy(
    vision: BrickDetector,
    robot: Robot,
    stats: dict,
) -> tuple[bool, str, dict | None, dict | None]:
    """Stop, let motion settle, then require fresh happy readings from rest."""
    cfg = _win_confirmation_config()
    _stop_robot(robot)
    settle_s = float(cfg.get("settle_s", DEFAULT_WIN_CONFIRMATION_CONFIG["settle_s"]))
    if settle_s > 0.0:
        time.sleep(settle_s)

    confirm_frames = int(cfg.get("confirm_frames", DEFAULT_WIN_CONFIRMATION_CONFIG["confirm_frames"]))
    last_reading = None
    last_plan = None
    for frame_idx in range(max(1, confirm_frames)):
        reading = _read_brick_measurement(vision)
        last_reading = reading
        stats["sample_count"] = int(stats.get("sample_count", 0)) + 1
        if not bool(reading.get("confident")):
            stats["not_confident_count"] = int(stats.get("not_confident_count", 0)) + 1
            _bump_stat_count(stats, "miss_reasons", "brick_not_confident_after_stop")
            return False, "brick_not_confident_after_stop", reading, None
        stats["confident_sample_count"] = int(stats.get("confident_sample_count", 0)) + 1
        plan = _follow_action_plan(reading)
        last_plan = plan
        if str(plan.get("kind")) != "hold":
            _bump_stat_count(stats, "miss_reasons", "happy_not_stopped")
            _record_non_win_stats(
                stats,
                reading,
                plan,
                action="HAPPY_REJECT",
                reason="happy_not_stopped",
            )
            return False, "happy_not_stopped", reading, plan
        if frame_idx < confirm_frames - 1:
            time.sleep(LOOP_S)
    return True, "stopped_happy_confirmed", last_reading, last_plan


def _follow_loop(
    vision: BrickDetector,
    robot: Robot,
    duration_s: float = 40.0,
    *,
    reset_after_win: bool = True,
    stop_after_win: bool = False,
) -> dict:
    last_action = ""
    print_ticker = 0
    miss_count = 0
    deadline = time.monotonic() + duration_s
    stats = _new_game_stats()

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        loop_wait_s = float(LOOP_S)

        reading = _read_brick_measurement(vision)
        stats["sample_count"] = int(stats.get("sample_count", 0)) + 1
        found = bool(reading.get("confident"))

        if not found:
            stats["not_confident_count"] = int(stats.get("not_confident_count", 0)) + 1
            _bump_stat_count(stats, "miss_reasons", "brick_not_confident")
            miss_count += 1
            _stop_robot(robot)
            if last_action != "NO_VIS" or miss_count == INVISIBLE_STOP_FRAMES:
                print("[FOLLOW] BRICK NOT CONFIDENT — stopped", flush=True)
            last_action = "NO_VIS"
            elapsed = time.monotonic() - loop_start
            if (remaining := LOOP_S - elapsed) > 0:
                time.sleep(remaining)
            continue

        stats["confident_sample_count"] = int(stats.get("confident_sample_count", 0)) + 1
        _record_observed_after_pending_act(stats, reading)
        miss_count = 0
        dist_mm = float(reading["dist_mm"])
        x_mm = float(reading["x_mm"])
        y_mm = reading.get("y_mm")
        y_text = ""
        try:
            y_text = f" y_err={float(y_mm) - Y_TARGET_MM:+.1f}mm"
        except (TypeError, ValueError):
            y_text = ""
        conf = float(reading["conf"])

        if _should_run_y_lock_on(stats, reading):
            action = "Y_LOCK_MAST_D"
            send_result = _lock_on_mast_down(robot, reading)
            stats["follow_attempt_count"] = int(stats.get("follow_attempt_count", 0)) + 1
            _bump_stat_count(stats, "act_counts", action)
            _bump_stat_count(stats, "miss_reasons", "y_lock_on")
            _record_send_result(stats, action, send_result)
            stats["y_lock_on_armed"] = False
            if not (isinstance(send_result, dict) and bool(send_result.get("blocked"))):
                stats["pending_observation"] = {
                    "action": action,
                    "dist_mm": dist_mm,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                }
            print(
                f"[FOLLOW] {action:<10} dist={dist_mm:.1f}mm x={x_mm:+.1f}mm {y_text} conf={conf:.0f}%",
                flush=True,
            )
            elapsed = time.monotonic() - loop_start
            if (remaining := LOOP_S - elapsed) > 0:
                time.sleep(remaining)
            continue

        plan = _follow_action_plan(reading)
        dist_err = float(plan["dist_err"])
        x_err = float(plan["x_err"])
        action = str(plan.get("action") or "HOLD")

        if str(plan.get("kind")) == "hold":
            action = "HAPPY"
            confirmed, confirm_reason, confirmed_reading, confirmed_plan = _confirm_stopped_happy(
                vision,
                robot,
                stats,
            )
            if not confirmed:
                rejected_dist = dist_err
                rejected_x = x_err
                if isinstance(confirmed_plan, dict):
                    try:
                        rejected_dist = float(confirmed_plan.get("dist_err"))
                        rejected_x = float(confirmed_plan.get("x_err"))
                    except (TypeError, ValueError):
                        pass
                rejected_conf = conf
                rejected_y_text = y_text
                if isinstance(confirmed_reading, dict):
                    try:
                        rejected_conf = float(confirmed_reading.get("conf", rejected_conf))
                    except (TypeError, ValueError):
                        pass
                    try:
                        rejected_y_text = f" y_err={float(confirmed_reading.get('y_mm')) - Y_TARGET_MM:+.1f}mm"
                    except (TypeError, ValueError):
                        rejected_y_text = ""
                print(
                    f"[FOLLOW] HAPPY_REJECT stopped_check={confirm_reason} "
                    f"dist_err={rejected_dist:+.1f}mm x_err={rejected_x:+.1f}mm "
                    f"{rejected_y_text} conf={rejected_conf:.0f}%",
                    flush=True,
                )
                last_action = "HAPPY_REJECT"
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue
            reading = confirmed_reading if isinstance(confirmed_reading, dict) else reading
            plan = confirmed_plan if isinstance(confirmed_plan, dict) else plan
            dist_err = float(plan["dist_err"])
            x_err = float(plan["x_err"])
            y_mm = reading.get("y_mm")
            try:
                y_text = f" y_err={float(y_mm) - Y_TARGET_MM:+.1f}mm"
            except (TypeError, ValueError):
                y_text = ""
            try:
                conf = float(reading["conf"])
            except (TypeError, ValueError):
                conf = 0.0
            stats["win_count"] = int(stats.get("win_count", 0)) + 1
            _record_win_stats(stats, reading, plan)
            win_dist_close = _target_closeness_pct(dist_err, DIST_TOL_MM)
            win_x_close = _target_closeness_pct(x_err, X_TOL_MM)
            print(
                f"[FOLLOW] WIN #{int(stats['win_count'])}: "
                f"dist_err={dist_err:+.1f}mm x_err={x_err:+.1f}mm "
                f"{y_text} close={win_dist_close:.0f}%/{win_x_close:.0f}% conf={conf:.0f}%",
                flush=True,
            )
            if not bool(reset_after_win):
                last_action = "HAPPY"
                if bool(stop_after_win):
                    break
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue
            reset_result = _run_reset_sequence(vision, robot)
            _record_reset_stats(stats, reset_result)
            _print_reset_stats(stats)
            if not bool(reset_result.get("success")):
                print(
                    f"[RESET] Failed during {reset_result.get('phase')}: "
                    f"{reset_result.get('reason')}",
                    flush=True,
                )
                break
            last_action = "RESET"
            stats["y_lock_on_armed"] = True
            print_ticker = 0
            elapsed = time.monotonic() - loop_start
            if (remaining := LOOP_S - elapsed) > 0:
                time.sleep(remaining)
            continue
        else:
            _record_follow_attempt_stats(stats, reading, plan)
            send_result = _execute_follow_action(robot, plan, reading)
            _record_send_result(stats, action, send_result)
            if not (isinstance(send_result, dict) and bool(send_result.get("blocked"))):
                loop_wait_s = _post_action_wait_s(plan, send_result)
            if not (isinstance(send_result, dict) and bool(send_result.get("blocked"))):
                pending = {
                    "action": action,
                    "dist_mm": dist_mm,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                }
                if isinstance(send_result, dict) and isinstance(send_result.get("x_curve"), dict):
                    pending["x_curve"] = dict(send_result["x_curve"])
                    pending["duration_ms"] = int(send_result.get("duration_ms", plan.get("duration_ms", PULSE_MS)) or 0)
                else:
                    plan_curve = _x_curve_for_plan(plan, reading)
                    if isinstance(plan_curve, dict):
                        pending["x_curve"] = dict(plan_curve)
                        try:
                            duration_val = (send_result or {}).get("duration_ms") if isinstance(send_result, dict) else None
                            pending["duration_ms"] = int(duration_val if duration_val is not None else plan.get("duration_ms", PULSE_MS))
                        except (TypeError, ValueError):
                            pending["duration_ms"] = int(PULSE_MS)
                stats["pending_observation"] = pending

        # Print on state change or every 20 ticks (~1 s) to avoid flooding
        print_ticker += 1
        if action != last_action or print_ticker >= 20:
            print(
                f"[FOLLOW] {action:<10} dist_err={dist_err:+.1f}mm  "
                f"x_err={x_err:+.1f}mm {y_text}  conf={conf:.0f}%",
                flush=True,
            )
            print_ticker = 0
        last_action = action
        elapsed = time.monotonic() - loop_start
        remaining = float(loop_wait_s) - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return stats


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Run only the reset: one random backward-turn act, then observe the result.",
    )
    parser.add_argument(
        "--park-happy",
        action="store_true",
        help="Move to the happy target, stop, and do not run reset.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Follow-loop duration when not using --reset-only.",
    )
    parser.add_argument(
        "--skip-vision-preflight",
        action="store_true",
        help="Skip Jetson memory-fragmentation preflight before TensorRT vision startup.",
    )
    parser.add_argument(
        "--min-lfb-mb",
        type=float,
        default=DEFAULT_VISION_MIN_LFB_MB,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _worker_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--duration-s",
        str(float(args.duration_s)),
        "--min-lfb-mb",
        str(float(args.min_lfb_mb)),
    ]
    if bool(args.reset_only):
        argv.append("--reset-only")
    if bool(args.park_happy):
        argv.append("--park-happy")
    if bool(args.skip_vision_preflight):
        argv.append("--skip-vision-preflight")
    return argv


def _supervise_run(args: argparse.Namespace) -> int:
    print("[FOLLOW] Starting supervised follow worker.", flush=True)
    worker = None
    try:
        worker = subprocess.Popen(_worker_argv(args), cwd=str(Path(__file__).resolve().parent))
        returncode = int(worker.wait())
    except KeyboardInterrupt:
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
                worker.wait(timeout=5.0)
            except Exception:
                try:
                    worker.kill()
                except Exception:
                    pass
        _emergency_stop_robot()
        print("\n[FOLLOW] Stopped.", flush=True)
        return 130

    if returncode == 0:
        return 0

    _emergency_stop_robot()
    if returncode < 0:
        print(
            f"[FOLLOW] Worker failed by signal {-returncode}; recovered with stop.",
            flush=True,
        )
    else:
        print(
            f"[FOLLOW] Worker exited with code {returncode}; recovered with stop.",
            flush=True,
        )
    return 1


def _run_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    if not bool(args.skip_vision_preflight):
        ok, reason = _vision_memory_preflight(min_lfb_mb=float(args.min_lfb_mb))
        if not bool(ok):
            print(
                f"[FOLLOW] Vision startup blocked: {reason}. "
                "Recovery: no robot motion was started.",
                flush=True,
            )
            return 1
        log.info("Vision preflight ok: %s", reason)

    vision = None
    robot = None
    print(
        f"[FOLLOW] Target: dist={TARGET_DIST_MM:.0f}mm ±{DIST_TOL_MM:.0f}mm  "
        f"x=0 ±{X_TOL_MM:.0f}mm  |  pulse: {PULSE_MS}ms  |  loop: {int(1/LOOP_S)}Hz  "
        f"|  max act: {_max_act_ms()}ms  |  normal score: {_normal_speed_score()}  "
        f"|  power scale: {_motion_power_scale():.2f}x",
        flush=True,
    )
    print("[FOLLOW] Press Ctrl-C to stop.", flush=True)

    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))
        _warmup(vision)
        pregame_reading = _wait_for_confident_brick(vision)
        if not bool(pregame_reading.get("confident")):
            return 1
        robot = Robot()
        if bool(args.reset_only):
            reset_result = _run_reset_sequence(vision, robot)
            if bool(reset_result.get("success")):
                print(
                    f"[RESET] Done: turn={str(reset_result.get('turn_cmd')).upper()} "
                    f"reason={reset_result.get('reason')}",
                    flush=True,
                )
            else:
                print(
                    f"[RESET] Failed during {reset_result.get('phase')}: "
                    f"{reset_result.get('reason')}",
                    flush=True,
                )
                return 1
        else:
            stats = _follow_loop(
                vision,
                robot,
                duration_s=float(args.duration_s),
                reset_after_win=not bool(args.park_happy),
                stop_after_win=bool(args.park_happy),
            )
            print(f"[FOLLOW] {float(args.duration_s):.0f} s elapsed — done.", flush=True)
            print("[RESULTS]", flush=True)
            print(_format_game_results_table(stats), flush=True)
    except KeyboardInterrupt:
        print("\n[FOLLOW] Stopped.", flush=True)
    except Exception as exc:
        print(f"[FOLLOW] Failed gracefully: {exc}", flush=True)
        return 1
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
            try:
                robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass

    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not bool(args.worker):
        return _supervise_run(args)
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
