#!/usr/bin/env python3
"""Follow-the-brick: repeatedly tracks a brick to the happy place.

The happy place is this runner's Step 1 HAPPY success gate. After a win, reset
is one backward-turn act in a random direction, then a measured pause before
following continues.

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from helper_brick_detector_native_oak import BrickDetector
from helper_brick_detector_yolo import (
    CYAN_HSV_BALANCED_LOWER,
    CYAN_HSV_BALANCED_UPPER,
    CYAN_HSV_WIDE_LOWER,
    CYAN_HSV_WIDE_UPPER,
)
from helper_brick_visibility_safety import (
    brick_motion_measurement_from_result,
    guarded_send_command_pwm,
    guarded_send_custom_actions_pwm,
    load_brick_visibility_motion_safety_config,
)
from helper_holding_brick import (
    contour_target_result_tuple,
    detect_holding_brick,
    detect_masked_target_brick_contour,
    mask_held_brick_for_target_frame,
)
from helper_holding_distance_calibration import apply_holding_distance_calibration_to_reading
from helper_mast_direction_guard import classify_mast_y_effect, mast_effect_is_reversal
from helper_robot_control import Robot
import telemetry_robot as _telemetry_robot

# ── tuning ────────────────────────────────────────────────────────────────────
# Runtime success-gate defaults for this runner. a_follow_the_brick.py owns
# gate evaluation; world_model_process.json gates are not consulted here.
TARGET_DIST_MM = 143.23551791647205  # Step 1 HAPPY dist target
DIST_TOL_MM    = 8.0     # Step 1 HAPPY dist tolerance
X_TARGET_MM    = -0.8462156212848165 # Step 1 HAPPY signed x target
X_TOL_MM       = 3.0     # Step 1 HAPPY signed x tolerance default
Y_TARGET_MM    = -11.952850590581479 # Step 1 HAPPY signed y target
Y_TOL_MM       = 0.6     # Step 1 HAPPY signed y tolerance default
# Operator rule: do not correct y (mast) until distance is at least this % closed to
# target. Closing y while dist is still far wastes mast acts, and the wheel moves
# needed to finish dist then disturb y again (wheels overpower the mast on y).
Y_GATE_MIN_DIST_CLOSENESS_PCT = 80.0

SPEED_SCORE        = 1    # slowest motor speed score
PULSE_MS       = 200     # motor pulse duration — long enough for slow motor to engage
LOOP_S         = 0.05    # control loop interval (20 Hz)
WARMUP_READS   = 16      # reads to warm the camera pipeline before capture
PREGAME_VISIBILITY_TIMEOUT_S = 12.0
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
RESET_SHARP_FINISH_MS = 135
RESET_LOW_X_EXTRA_TURN_THRESHOLD_MM = 35.0
RESET_LOW_X_EXTRA_TURN_DURATION_MS = 200
RESET_LOW_X_EXTRA_TURN_SLOWER_PWM = 103
RESET_LOW_X_EXTRA_TURN_FASTER_PWM = 162
RESET_MAST_UP_MIN_MS = 1000
RESET_MAST_UP_MAX_MS = 1000
RESET_MAST_UP_PWM = 255
RESET_MAST_UP_CMD = "u"
RESET_MAST_UP_SETTLE_S = 0.1
DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG = {
    "enabled": True,
    "duration_ms": 1000,
    "pwm": 103,
    "mast_up_delay_fraction": 0.0,
}
DEFAULT_RESET_X_GOAL_CURVE_CONFIG = {
    "enabled": True,
    "target_fraction": 0.8,
    "min_duration_ms": 0,
    "max_duration_ms": 500,
    "chunk_ms": 100,
    "settle_s": 0.12,
    "inner_pwm": 0,
    "outer_pwm": 162,
}
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
DEFAULT_MAX_ACT_MS = 2000
MAX_MAST_ACT_MS = 1000
HOLDING_S1_TRANSITION_COMMIT_MS = 800
HOLDING_S1_TRANSITION_MAX_DIST_ERR_MM = 45.0
HOLDING_S1_TRANSITION_MAX_X_OUTSIDE_MM = 0.0
HOLDING_S1_RETRY_BACKOFF_MS = 1500
RESET_FINAL_X_POLISH_MARGIN_MM = 2.0
RESET_FINAL_X_POLISH_MIN_MS = 170
RESET_FINAL_X_POLISH_MAX_MS = 250
GAP_REGRESSION_EPSILON_MM = 0.25
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
    "max_forward_pulse_ms": 2000,
    "full_forward_gap_mm": 100.0,
    "near_target_creep_band_mm": 30.0,
    "near_target_min_pulse_ms": 200,
    "near_target_max_pulse_ms": 300,
    "close_target_creep_band_mm": 20.0,
    "close_target_max_pulse_ms": 200,
    "very_close_target_creep_band_mm": 10.0,
    "very_close_target_max_pulse_ms": 200,
    "micro_nudge_abs_dist_err_mm": 16.0,
    "micro_nudge_pulse_ms": 200,
    "micro_nudge_min_effective_pulse_ms": 240,
    "micro_nudge_pwm": 103,
    "min_effective_drive_pulse_ms": 300,
    "near_target_forward_veto_mm": 0.0,
    "pingpong_confirm_abs_dist_err_mm": 20.0,
}
DEFAULT_X_PRIORITY_POLICY = {
    "polish_abs_x_mm": 6.0,
    "huge_dist_gap_mm": 200.0,
    "huge_dist_tiny_abs_x_mm": 5.0,
    "simultaneous_dist_outside_min_mm": 8.0,
    "simultaneous_x_abs_max_mm": 80.0,
    "x_first_turn_strength": "adaptive",
    "adaptive_outer_pwm_scale": 0.75,
    "turn_settle_s": 0.0,
    "attach_y_to_turns": False,
    "avoid_forward_left_bias": True,
    "avoid_forward_left_bias_min_x_outside_mm": 0.0,
    "avoid_forward_left_bias_min_dist_outside_mm": 0.0,
    "tiny_x_outside_no_turn_mm": 2.0,
    "tiny_x_only_backoff_enabled": True,
}
DEFAULT_X_DIST_CURVE_POLICY = {
    "large_dist_gap_mm": 60.0,
    "small_x_gap_mm": 6.0,
    "large_dist_small_x_strength": "gentle",
    "combined_x_dist_strength": "adaptive",
    "combined_bias_max_pulse_ms": 0,
    "near_dist_gap_mm": 12.0,
    "wide_x_gap_mm": 9.0,
    "near_wide_x_strength": "strong",
    "too_close_wide_x_drive_mode": "backward",
}
DEFAULT_X_ONLY_TURN_POLICY = {
    "drive_mode": "backward",
    "far_drive_mode": "forward",
    "forward_min_dist_err_mm": 60.0,
    "sharp_max_abs_dist_err_mm": 20.0,
}
DEFAULT_FOLLOW_X_AXIS_CONFIG = {
    "win_target_mm": X_TARGET_MM,
    "win_tol_mm": X_TOL_MM,
    "positive_error_turn_cmd": "r",
}
DEFAULT_FOLLOW_DIST_AXIS_CONFIG = {
    "win_target_mm": TARGET_DIST_MM,
    "win_tol_mm": DIST_TOL_MM,
    "positive_error_cmd": "f",
}
DEFAULT_VISIBILITY_RECOVERY_CONFIG = {
    "wait_s": 3.0,
    "poll_s": 0.15,
    "mast_down_duration_ms": 1000,
    "mast_down_pwm": 255,
}
DEFAULT_PICKUP_SUSPECT_CONFIG = {
    "enabled": True,
    "min_dist_mm": 260.0,
    "max_y_mm": -80.0,
    "confirm_frames": 2,
    "confirm_poll_s": 0.12,
}
MAST_RAISE_CEILING_ABOVE_TARGET_MM = 5.0
DEFAULT_HOLDING_TARGET_VISION_CONFIG = {
    "min_confidence_pct": 40.0,
    "confidence": 0.12,
    "conf_gate_pct": 45.0,
    "hsv_cyan_coverage_min": 0.08,
    "hsv_min_area_ratio": 0.04,
    "trust_detector_boxes": False,
    "require_cyan_shape": True,
    "far_suspect_enabled": False,
    "use_unmasked_stack_xz": False,
}
DEFAULT_HOLDING_TARGET_DISTANCE_CALIBRATION_CONFIG = {
    "enabled": False,
    "points": [],
}
DEFAULT_ACT_STALL_GUARD_CONFIG = {
    "enabled": True,
    "max_no_change_tries": 6,
    "min_axis_delta_mm": 1.0,
    "close_trap_enabled": True,
    "close_trap_min_abs_dist_err_mm": 25.0,
    "close_trap_max_no_change_tries": 1,
    "y_max_no_change_tries": 59,
    "y_max_no_change_duration_ms": 7800,
    "recovery_boost_enabled": True,
    "recovery_boost_step_pct": 10.0,
    "recovery_boost_max_scale": 1.6,
}
DEFAULT_STEP2_CONFIG = {
    "blind_mast_only": False,
    "seat_mast_cmd": "d",
    "seat_mast_pwm": 255,
    "seat_mast_duration_ms": 600,
    "seat_drive_cmd": "f",
    "seat_drive_pwm": 103,
    "seat_drive_duration_ms": 0,
    "post_seat_pause_s": 0.25,
    "step_timeout_s": 15.0,
    "recovery_creep_enabled": True,
    "recovery_creep_pulse_ms": 200,
    "recovery_creep_max_attempts": 6,
    "recovery_creep_settle_s": 0.15,
    "visibility_recovery_creep_enabled": True,
    "visibility_recovery_wait_s": 3.0,
    "visibility_recovery_wait_poll_s": 0.15,
    "visibility_recovery_creep_pulse_ms": 200,
    "visibility_recovery_creep_max_attempts": 3,
    "visibility_recovery_creep_settle_s": 0.12,
    "precision_settle_enabled": True,
    "precision_max_attempts": 7,
    "post_precision_recovery_cycles": 0,
    "precision_drive_min_pulse_ms": 80,
    "precision_drive_max_pulse_ms": 200,
    "precision_dist_positive_cmd": "f",
    "precision_mast_pulse_ms": 250,
    "precision_mast_small_gap_max_mm": 5.0,
    "precision_mast_small_gap_pwm_scale": 1.0,
    "precision_mast_small_gap_min_pulse_ms": 150,
    "precision_mast_small_gap_max_pulse_ms": 250,
    "precision_settle_s": 0.12,
    "freeze_xz_after_xz_target": False,
    "semi_happy_targets": {
        "dist_mm": 81.0,
        "dist_tol_mm": 5.0,
        "x_mm": 0.3,
        "x_tol_mm": 9.0,
        "y_mm": -5.3,
        "y_tol_mm": 1.0,
    },
    "targets": {
        "dist_mm": None,
        "dist_tol_mm": None,
        "x_mm": 0.0,
        "x_tol_mm": X_TOL_MM,
        "y_mm": -7.093439,
        "y_tol_mm": 0.5,
    },
}
DEFAULT_STEP3_CONFIG = {
    "lift_mast_cmd": "u",
    "lift_mast_pwm": 255,
    "lift_pulse_ms": 250,
    "fixed_lift_only": False,
    "fixed_lift_duration_ms": 1700,
    "lift_settle_s": 0.12,
    "max_lift_attempts": 12,
    "step_timeout_s": 15.0,
    "initial_lift_before_gate": False,
    "no_visibility_fallback_enabled": True,
    "no_visibility_fallback_mast_cmd": "d",
    "no_visibility_fallback_mast_pwm": 255,
    "no_visibility_fallback_duration_ms": 1000,
    "no_visibility_fallback_settle_s": 0.15,
    "targets": {
        "y_mm": -3.5,
        "y_tol_mm": 1.0,
    },
}
DEFAULT_STEP3_RETREAT_CONFIG = {
    "kind": "seat",
    "drive_cmd": "b",
    "drive_pwm": 103,
    "max_duration_ms": 1500,
    "chunk_ms": 250,
    "settle_s": 0.08,
    "target_dist_mm": None,
    "target_dist_profile": "empty",
    "target_dist_tol_mm": 0.0,
    "stop_when_dist_at_or_beyond": True,
}
CURRENT_GAME_PROFILE = "empty"
HOLDING_TARGET_MASK_Y_SHIFT_PX = 100
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
    "single_win_allowance_s": 15.0,
    "active_attempt_budget_s": None,
    "timeout_y_correction_grace_acts": 2,
    "min_axis_closeness_pct": 0.0,
    "min_confidence_pct": 75.0,
    "accept_live_happy_after_stop": False,
}
DEFAULT_CAUTIOUS_VISIBILITY_CONFIG = {
    "motion_min_confidence_pct": 50.0,
    "pregame_timeout_s": PREGAME_VISIBILITY_TIMEOUT_S,
    "pregame_sample_s": PREGAME_VISIBILITY_SAMPLE_S,
    "pregame_sample_frames": 8,
    "pregame_required_frames": 3,
}
DEFAULT_VISION_JUMP_GUARD_CONFIG = {
    "enabled": True,
    "accept_confirmed_jumps": True,
    "confirm_frames": 2,
    "reacquire_frames_after_motion": 0,
    "post_act_stabilize_s": 1.0,
    "hard_reject_dist_jump_mm": 75.0,
    "hard_reject_observe_s": 2.0,
    "hard_reject_observe_poll_s": 0.15,
    "hard_reject_max_recoveries_per_step": 1,
    "max_abs_dist_mm": 320.0,
    "edge_recovery_min_abs_x_mm": 200.0,
    "edge_recovery_max_abs_dist_mm": 500.0,
    "edge_recovery_min_confidence_pct": 90.0,
    "max_dist_jump_mm": 22.0,
    "max_x_jump_mm": 18.0,
    "max_y_jump_mm": 8.0,
    "max_vector_jump_mm": 26.0,
    "confirm_window_mm": 10.0,
    "reacquire_window_mm": 12.0,
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
    "hard_floor_y_mm": None,
    "protect_requires_brick_below": True,
    "protect_disabled_within_dist_mm": 177.0,
    "priority_abs_err_mm": 14.0,
    "lock_on_enabled": True,
    "lock_on_dist_mm": 90.0,
    "lock_on_dist_window_mm": 10.0,
    "lock_on_mast_pwm": 255,
    "lock_on_pulse_ms": 400,
    "mast_pwm": 255,
    "mast_up_pwm": 255,
    "mast_down_pwm": 255,
    "mast_pulse_ms": 130,
    "mast_min_pulse_ms": 80,
    "mast_max_pulse_ms": 130,
    "finish_mast_pwm": 255,
    "finish_mast_up_pwm": 255,
    "finish_mast_down_pwm": 255,
    "finish_mast_pulse_ms": 130,
    "finish_mast_min_pulse_ms": 80,
    "finish_mast_max_pulse_ms": 130,
    "mast_mm_per_100ms": 3.0,
    "mast_up_mm_per_100ms": 3.0,
    "mast_down_mm_per_100ms": 3.0,
    "mast_up_duration_curve": [],
    "mast_down_duration_curve": [],
    "max_step1_mast_up_ms": 0,
    "mast_correction_fraction": 0.9,
    "mast_duration_uses_tolerance_gap": False,
    "mast_coast_settle_s": 0.45,
    "mast_min_effect_mm": 3.0,
    "mast_overshoot_guard_margin_mm": 4.0,
    "tiny_y_no_observed_wait_s": 0.8,
    "tiny_y_no_observed_abs_err_mm": 16.0,
    "spool_reversal_short_act_count": 4,
    "spool_reversal_mast_max_ms": 110,
    "attach_y_min_abs_err_mm": 12.0,
    "attach_y_max_abs_dist_err_mm": 0.0,
}
DEFAULT_TURN_BIAS_CURVES = {
    "micro": {"inner_pwm": 103, "outer_pwm": 106},
    "gentle": {"inner_pwm": 103, "outer_pwm": 117},
    "medium": {"inner_pwm": 103, "outer_pwm": 133},
    "strong": {"inner_pwm": 104, "outer_pwm": 155},
}
TURN_BIAS_STRENGTHS = ("micro", "gentle", "medium", "strong")
DEFAULT_VISION_MIN_LFB_MB = 16.0
VISION_PREFLIGHT_BLOCK_EXIT = 42
PREGAME_VISIBILITY_BLOCK_EXIT = 43
VISION_RECOVERY_RETRIES = 2
PREGAME_VISIBILITY_RECOVERY_RETRIES = 3

CROWN_PROFILE_TUNING = {
    "confidence": 0.08,
    "smoothing_alpha": 0.15,
    "hsv_enabled": True,
    "hsv_erode_iterations": 1,
    "hsv_lower": list(CYAN_HSV_WIDE_LOWER),
    "hsv_upper": list(CYAN_HSV_WIDE_UPPER),
    "hsv_cyan_coverage_min": 0.05,
    "full_frame_hsv_cyan_coverage_min": 0.03,
    "hsv_min_area_ratio": 0.03,
    "full_frame_hsv_min_area_ratio": 0.02,
    "shape_gate_mode": "shape_match",
    "conf_gate_pct": 50.0,
    "trust_detector_boxes": False,
    "require_cyan_shape": True,
    "far_suspect_enabled": False,
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


def _max_mast_act_ms() -> int:
    return int(MAX_MAST_ACT_MS)


def _cap_mast_duration_ms(
    cmd: str | None,
    duration_ms,
    fallback: int | float,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    duration = _coerce_int(duration_ms, fallback, minimum=minimum)
    if str(cmd or "").strip().lower() in {"u", "d"}:
        cap_ms = _max_mast_act_ms() if maximum is None else int(maximum)
        duration = min(int(duration), int(cap_ms))
    return int(duration)


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


def _coerce_optional_float(
    value,
    fallback: float | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None if fallback is None else _coerce_float(fallback, fallback, minimum=minimum, maximum=maximum)
    return _coerce_float(value, fallback if fallback is not None else 0.0, minimum=minimum, maximum=maximum)


def _sanitize_y_duration_curve(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    points: list[dict] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        duration_ms = _coerce_int(row.get("duration_ms"), 0, minimum=0, maximum=_max_mast_act_ms())
        closes_mm = _coerce_float(row.get("closes_mm"), 0.0, minimum=0.0, maximum=100.0)
        if duration_ms <= 0 or closes_mm <= 0.0:
            continue
        points.append({"duration_ms": int(duration_ms), "closes_mm": float(closes_mm)})
    points.sort(key=lambda p: (float(p["closes_mm"]), int(p["duration_ms"])))
    deduped: list[dict] = []
    for point in points:
        if deduped and abs(float(deduped[-1]["closes_mm"]) - float(point["closes_mm"])) < 0.001:
            deduped[-1] = point
        else:
            deduped.append(point)
    return deduped


def _default_step2_like_config() -> dict:
    return {
        **{key: value for key, value in DEFAULT_STEP2_CONFIG.items() if key != "targets"},
        "semi_happy_targets": dict(DEFAULT_STEP2_CONFIG["semi_happy_targets"]),
        "targets": dict(DEFAULT_STEP2_CONFIG["targets"]),
    }


def _apply_step2_like_config(raw_cfg: dict | None, step_cfg: dict) -> dict:
    raw = raw_cfg if isinstance(raw_cfg, dict) else {}
    if "kind" in raw:
        kind = str(raw.get("kind") or "").strip().lower()
        if kind in {"seat", "retreat"}:
            step_cfg["kind"] = kind
    for key in ("seat_mast_cmd", "seat_drive_cmd"):
        if key not in raw:
            continue
        mode = str(raw.get(key) or "").strip().lower()
        if mode in {"u", "d"} and key == "seat_mast_cmd":
            step_cfg[key] = mode
        if mode in {"f", "b"} and key == "seat_drive_cmd":
            step_cfg[key] = mode
    if "drive_cmd" in raw:
        mode = str(raw.get("drive_cmd") or "").strip().lower()
        if mode in {"f", "b"}:
            step_cfg["drive_cmd"] = mode
    for key in ("seat_mast_pwm", "seat_drive_pwm"):
        if key in raw:
            step_cfg[key] = _coerce_int(raw.get(key), step_cfg.get(key), minimum=1, maximum=255)
    if "drive_pwm" in raw:
        step_cfg["drive_pwm"] = _coerce_int(raw.get("drive_pwm"), step_cfg.get("drive_pwm", 103), minimum=1, maximum=255)
    if "seat_mast_duration_ms" in raw:
        step_cfg["seat_mast_duration_ms"] = _cap_mast_duration_ms(
            step_cfg.get("seat_mast_cmd"),
            raw.get("seat_mast_duration_ms"),
            step_cfg.get("seat_mast_duration_ms", DEFAULT_STEP2_CONFIG["seat_mast_duration_ms"]),
            minimum=0,
        )
    if "seat_drive_duration_ms" in raw:
        step_cfg["seat_drive_duration_ms"] = _coerce_int(
            raw.get("seat_drive_duration_ms"),
            step_cfg.get("seat_drive_duration_ms"),
            minimum=0,
            maximum=5000,
        )
    for key, maximum in (("max_duration_ms", 30000), ("chunk_ms", 1000)):
        if key in raw:
            step_cfg[key] = _coerce_int(
                raw.get(key),
                step_cfg.get(key, DEFAULT_STEP3_RETREAT_CONFIG[key]),
                minimum=1,
                maximum=maximum,
            )
    if "settle_s" in raw:
        step_cfg["settle_s"] = _coerce_float(
            raw.get("settle_s"),
            step_cfg.get("settle_s", DEFAULT_STEP3_RETREAT_CONFIG["settle_s"]),
            minimum=0.0,
            maximum=2.0,
        )
    if "target_dist_mm" in raw:
        step_cfg["target_dist_mm"] = _coerce_optional_float(
            raw.get("target_dist_mm"),
            step_cfg.get("target_dist_mm", DEFAULT_STEP3_RETREAT_CONFIG["target_dist_mm"]),
            minimum=0.0,
        )
    if "target_dist_profile" in raw:
        profile = str(raw.get("target_dist_profile") or "").strip().lower()
        if profile in {"empty", "holding"}:
            step_cfg["target_dist_profile"] = profile
    if "target_dist_tol_mm" in raw:
        step_cfg["target_dist_tol_mm"] = _coerce_float(
            raw.get("target_dist_tol_mm"),
            step_cfg.get("target_dist_tol_mm", DEFAULT_STEP3_RETREAT_CONFIG["target_dist_tol_mm"]),
            minimum=0.0,
            maximum=100.0,
        )
    if "stop_when_dist_at_or_beyond" in raw:
        step_cfg["stop_when_dist_at_or_beyond"] = bool(raw.get("stop_when_dist_at_or_beyond"))
    if "post_seat_pause_s" in raw:
        step_cfg["post_seat_pause_s"] = _coerce_float(
            raw.get("post_seat_pause_s"),
            step_cfg.get("post_seat_pause_s", DEFAULT_STEP2_CONFIG["post_seat_pause_s"]),
            minimum=0.0,
            maximum=3.0,
        )
    if "step_timeout_s" in raw:
        step_cfg["step_timeout_s"] = _coerce_float(
            raw.get("step_timeout_s"),
            step_cfg.get("step_timeout_s", DEFAULT_STEP2_CONFIG["step_timeout_s"]),
            minimum=0.0,
            maximum=120.0,
        )
    for key in (
        "blind_mast_only",
        "recovery_creep_enabled",
        "visibility_recovery_creep_enabled",
        "precision_settle_enabled",
        "freeze_xz_after_xz_target",
    ):
        if key in raw:
            step_cfg[key] = bool(raw.get(key))
    for key, maximum in (
        ("recovery_creep_pulse_ms", 400),
        ("recovery_creep_max_attempts", 20),
        ("visibility_recovery_creep_pulse_ms", 400),
        ("visibility_recovery_creep_max_attempts", 10),
        ("precision_max_attempts", 40),
        ("post_precision_recovery_cycles", 20),
        ("precision_drive_min_pulse_ms", 1000),
        ("precision_drive_max_pulse_ms", 1000),
        ("precision_mast_pulse_ms", _max_mast_act_ms()),
        ("precision_mast_small_gap_min_pulse_ms", _max_mast_act_ms()),
        ("precision_mast_small_gap_max_pulse_ms", _max_mast_act_ms()),
    ):
        if key in raw:
            step_cfg[key] = _coerce_int(raw.get(key), step_cfg.get(key, DEFAULT_STEP2_CONFIG.get(key, 0)), minimum=0 if "attempts" in key or key == "post_precision_recovery_cycles" else 1, maximum=maximum)
    for key in ("precision_mast_small_gap_max_mm", "precision_mast_small_gap_pwm_scale"):
        if key in raw:
            step_cfg[key] = _coerce_float(
                raw.get(key),
                step_cfg.get(key, DEFAULT_STEP2_CONFIG[key]),
                minimum=0.0,
                maximum=1.0 if key.endswith("_scale") else 100.0,
            )
    for key in ("recovery_creep_settle_s", "visibility_recovery_wait_s", "visibility_recovery_wait_poll_s", "visibility_recovery_creep_settle_s", "precision_settle_s"):
        if key in raw:
            maximum = 10.0 if key == "visibility_recovery_wait_s" else 2.0
            minimum = 0.01 if key == "visibility_recovery_wait_poll_s" else 0.0
            step_cfg[key] = _coerce_float(
                raw.get(key),
                step_cfg.get(key, DEFAULT_STEP2_CONFIG.get(key, 0.0)),
                minimum=minimum,
                maximum=maximum,
            )
    if "precision_dist_positive_cmd" in raw:
        positive_cmd = str(raw.get("precision_dist_positive_cmd") or "").strip().lower()
        if positive_cmd in {"f", "b"}:
            step_cfg["precision_dist_positive_cmd"] = positive_cmd
    raw_semi = raw.get("semi_happy_targets") if isinstance(raw.get("semi_happy_targets"), dict) else {}
    semi_cfg = step_cfg.get("semi_happy_targets") if isinstance(step_cfg.get("semi_happy_targets"), dict) else {}
    for key in ("dist_mm", "x_mm", "y_mm"):
        if key in raw_semi:
            semi_cfg[key] = None if raw_semi.get(key) is None else _coerce_optional_float(raw_semi.get(key), semi_cfg.get(key))
    for key in ("dist_tol_mm", "x_tol_mm", "y_tol_mm"):
        if key in raw_semi:
            semi_cfg[key] = None if raw_semi.get(key) is None else _coerce_optional_float(raw_semi.get(key), semi_cfg.get(key), minimum=0.0)
    step_cfg["semi_happy_targets"] = semi_cfg
    raw_targets = raw.get("targets") if isinstance(raw.get("targets"), dict) else {}
    target_cfg = step_cfg.get("targets") if isinstance(step_cfg.get("targets"), dict) else {}
    for key in ("dist_mm", "x_mm", "y_mm"):
        if key in raw_targets:
            target_cfg[key] = None if raw_targets.get(key) is None else _coerce_optional_float(raw_targets.get(key), target_cfg.get(key))
    for key in ("dist_tol_mm", "x_tol_mm", "y_tol_mm"):
        if key in raw_targets:
            target_cfg[key] = None if raw_targets.get(key) is None else _coerce_optional_float(raw_targets.get(key), target_cfg.get(key), minimum=0.0)
    step_cfg["targets"] = target_cfg
    if "nickname" in raw:
        step_cfg["nickname"] = str(raw.get("nickname") or "").strip()
    return step_cfg


def _active_game_profile() -> str:
    profile = str(globals().get("CURRENT_GAME_PROFILE", "empty") or "empty").strip().lower()
    return profile if profile in {"empty", "holding"} else "empty"


def _set_game_profile(profile: str) -> None:
    global CURRENT_GAME_PROFILE
    name = str(profile or "empty").strip().lower()
    CURRENT_GAME_PROFILE = name if name in {"empty", "holding"} else "empty"
    if hasattr(_follow_motion_config, "_cache"):
        delattr(_follow_motion_config, "_cache")
    if hasattr(_reset_motion_config, "_cache"):
        delattr(_reset_motion_config, "_cache")


def _auto_select_game_profile(vision: BrickDetector, *, samples: int = 3, sample_s: float = 0.08) -> tuple[str, dict]:
    votes = []
    for _idx in range(max(1, int(samples))):
        try:
            vision.read()
            frame = getattr(vision, "raw_frame", None)
            result = detect_holding_brick(frame)
        except Exception as exc:
            result = {"holding": False, "reason": f"holding_detector_error:{exc}"}
        votes.append(result)
        if sample_s > 0:
            time.sleep(float(sample_s))
    holding_count = sum(1 for row in votes if bool(row.get("holding")))
    selected = "holding" if holding_count > (len(votes) / 2.0) else "empty"
    return selected, {
        "samples": len(votes),
        "holding_count": int(holding_count),
        "empty_count": int(len(votes) - holding_count),
        "votes": votes,
    }


def _hard_motion_min_confidence_pct() -> float:
    cfg = load_brick_visibility_motion_safety_config()
    return _coerce_float(cfg.get("min_confidence_pct"), 75.0, minimum=0.0, maximum=100.0)


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


def _merge_profile_turning_overrides(cfg: dict, raw_profile: dict) -> None:
    """Merge game-profile turn tuning over the shared follow motion config."""
    if not isinstance(cfg, dict) or not isinstance(raw_profile, dict):
        return
    raw_turn_curves = raw_profile.get("turn_curves") if isinstance(raw_profile.get("turn_curves"), dict) else {}
    if raw_turn_curves:
        global_inner = _coerce_int(
            raw_turn_curves.get("inner_pwm"),
            cfg.get("turn_curves", {}).get("inner_pwm", DEFAULT_TURN_CURVE_INNER_PWM),
            minimum=0,
            maximum=255,
        )
        cfg["turn_curves"]["inner_pwm"] = int(global_inner)
        for drive_mode in ("forward", "backward"):
            raw_drive = raw_turn_curves.get(drive_mode) if isinstance(raw_turn_curves.get(drive_mode), dict) else {}
            for strength in ("gentle", "medium", "strong"):
                raw_curve = raw_drive.get(strength) if isinstance(raw_drive.get(strength), dict) else {}
                curve = cfg["turn_curves"][drive_mode][strength]
                curve["inner_pwm"] = _coerce_int(
                    raw_curve.get("inner_pwm"),
                    curve.get("inner_pwm", global_inner),
                    minimum=0,
                    maximum=255,
                )
                curve["outer_pwm"] = _coerce_int(
                    raw_curve.get("outer_pwm"),
                    curve.get("outer_pwm", DEFAULT_TURN_CURVE_OUTER_PWMS[strength]),
                    minimum=1,
                    maximum=255,
                )

    raw_bias_curves = (
        raw_profile.get("turn_bias_curves")
        if isinstance(raw_profile.get("turn_bias_curves"), dict)
        else {}
    )
    if raw_bias_curves:
        for drive_mode in ("forward", "backward"):
            raw_drive = raw_bias_curves.get(drive_mode) if isinstance(raw_bias_curves.get(drive_mode), dict) else {}
            for strength in TURN_BIAS_STRENGTHS:
                raw_curve = raw_drive.get(strength) if isinstance(raw_drive.get(strength), dict) else {}
                curve = cfg["turn_bias_curves"][drive_mode][strength]
                curve["inner_pwm"] = _coerce_int(
                    raw_curve.get("inner_pwm"),
                    curve.get("inner_pwm", DEFAULT_TURN_BIAS_CURVES[strength]["inner_pwm"]),
                    minimum=0,
                    maximum=255,
                )
                curve["outer_pwm"] = _coerce_int(
                    raw_curve.get("outer_pwm"),
                    curve.get("outer_pwm", DEFAULT_TURN_BIAS_CURVES[strength]["outer_pwm"]),
                    minimum=1,
                    maximum=255,
                )

    raw_thresholds = (
        raw_profile.get("curve_strength_abs_x_err_mm")
        if isinstance(raw_profile.get("curve_strength_abs_x_err_mm"), dict)
        else {}
    )
    for key, fallback in (
        ("medium", DEFAULT_MEDIUM_CURVE_ABS_X_ERR_MM),
        ("strong", DEFAULT_STRONG_CURVE_ABS_X_ERR_MM),
    ):
        if key in raw_thresholds:
            cfg["curve_strength_abs_x_err_mm"][key] = _coerce_float(
                raw_thresholds.get(key),
                cfg["curve_strength_abs_x_err_mm"].get(key, fallback),
                minimum=0.0,
            )

    raw_x_priority = (
        raw_profile.get("x_priority_policy")
        if isinstance(raw_profile.get("x_priority_policy"), dict)
        else {}
    )
    for key in (
        "polish_abs_x_mm",
        "huge_dist_gap_mm",
        "huge_dist_tiny_abs_x_mm",
        "simultaneous_dist_outside_min_mm",
        "simultaneous_x_abs_max_mm",
        "turn_settle_s",
        "avoid_forward_left_bias_min_x_outside_mm",
        "avoid_forward_left_bias_min_dist_outside_mm",
        "tiny_x_outside_no_turn_mm",
    ):
        if key in raw_x_priority:
            cfg["x_priority_policy"][key] = _coerce_float(
                raw_x_priority.get(key),
                cfg["x_priority_policy"].get(key, DEFAULT_X_PRIORITY_POLICY.get(key, 0.0)),
                minimum=0.0,
            )
    if "x_first_turn_strength" in raw_x_priority:
        cfg["x_priority_policy"]["x_first_turn_strength"] = _coerce_curve_strength(
            raw_x_priority.get("x_first_turn_strength"),
            cfg["x_priority_policy"].get("x_first_turn_strength", DEFAULT_X_PRIORITY_POLICY["x_first_turn_strength"]),
        )
    if "adaptive_outer_pwm_scale" in raw_x_priority:
        cfg["x_priority_policy"]["adaptive_outer_pwm_scale"] = _coerce_float(
            raw_x_priority.get("adaptive_outer_pwm_scale"),
            cfg["x_priority_policy"].get(
                "adaptive_outer_pwm_scale",
                DEFAULT_X_PRIORITY_POLICY["adaptive_outer_pwm_scale"],
            ),
            minimum=0.5,
            maximum=2.0,
        )
    if "attach_y_to_turns" in raw_x_priority:
        cfg["x_priority_policy"]["attach_y_to_turns"] = bool(raw_x_priority.get("attach_y_to_turns"))
    if "avoid_forward_left_bias" in raw_x_priority:
        cfg["x_priority_policy"]["avoid_forward_left_bias"] = bool(
            raw_x_priority.get("avoid_forward_left_bias")
        )
    if "tiny_x_only_backoff_enabled" in raw_x_priority:
        cfg["x_priority_policy"]["tiny_x_only_backoff_enabled"] = bool(
            raw_x_priority.get("tiny_x_only_backoff_enabled")
        )

    raw_x_dist = (
        raw_profile.get("x_dist_curve_policy")
        if isinstance(raw_profile.get("x_dist_curve_policy"), dict)
        else {}
    )
    for key in ("large_dist_gap_mm", "small_x_gap_mm", "near_dist_gap_mm", "wide_x_gap_mm"):
        if key in raw_x_dist:
            cfg["x_dist_curve_policy"][key] = _coerce_float(
                raw_x_dist.get(key),
                cfg["x_dist_curve_policy"].get(key, DEFAULT_X_DIST_CURVE_POLICY[key]),
                minimum=0.0,
            )
    if "combined_bias_max_pulse_ms" in raw_x_dist:
        cfg["x_dist_curve_policy"]["combined_bias_max_pulse_ms"] = _coerce_int(
            raw_x_dist.get("combined_bias_max_pulse_ms"),
            cfg["x_dist_curve_policy"].get(
                "combined_bias_max_pulse_ms",
                DEFAULT_X_DIST_CURVE_POLICY["combined_bias_max_pulse_ms"],
            ),
            minimum=0,
            maximum=_max_act_ms(),
        )
    for key in ("large_dist_small_x_strength", "combined_x_dist_strength", "near_wide_x_strength"):
        if key in raw_x_dist:
            cfg["x_dist_curve_policy"][key] = _coerce_curve_strength(
                raw_x_dist.get(key),
                cfg["x_dist_curve_policy"].get(key, DEFAULT_X_DIST_CURVE_POLICY[key]),
            )
    if cfg["x_dist_curve_policy"].get("near_wide_x_strength") == "adaptive":
        cfg["x_dist_curve_policy"]["near_wide_x_strength"] = "strong"
    if cfg["x_dist_curve_policy"].get("large_dist_small_x_strength") == "adaptive":
        cfg["x_dist_curve_policy"]["large_dist_small_x_strength"] = "gentle"
    if "too_close_wide_x_drive_mode" in raw_x_dist:
        drive_mode = str(raw_x_dist.get("too_close_wide_x_drive_mode") or "").strip().lower()
        if drive_mode in {"forward", "backward"}:
            cfg["x_dist_curve_policy"]["too_close_wide_x_drive_mode"] = drive_mode
    raw_x_only = (
        raw_profile.get("x_only_turn")
        if isinstance(raw_profile.get("x_only_turn"), dict)
        else {}
    )
    for key in ("drive_mode", "far_drive_mode"):
        if key in raw_x_only:
            mode = str(raw_x_only.get(key) or "").strip().lower()
            if mode in {"forward", "backward"}:
                cfg["x_only_turn"][key] = mode
    if "forward_min_dist_err_mm" in raw_x_only:
        cfg["x_only_turn"]["forward_min_dist_err_mm"] = _coerce_float(
            raw_x_only.get("forward_min_dist_err_mm"),
            cfg["x_only_turn"].get("forward_min_dist_err_mm", DEFAULT_X_ONLY_TURN_POLICY["forward_min_dist_err_mm"]),
            minimum=0.0,
        )
    if "sharp_max_abs_dist_err_mm" in raw_x_only:
        cfg["x_only_turn"]["sharp_max_abs_dist_err_mm"] = _coerce_float(
            raw_x_only.get("sharp_max_abs_dist_err_mm"),
            cfg["x_only_turn"].get(
                "sharp_max_abs_dist_err_mm",
                DEFAULT_X_ONLY_TURN_POLICY["sharp_max_abs_dist_err_mm"],
            ),
            minimum=0.0,
        )


def _load_follow_motion_config(path: Path | None = None) -> dict:
    model_path = path if isinstance(path, Path) else ROBOT_MODEL_FILE
    cfg = {
        "motion_power_scale": float(DEFAULT_MOTION_POWER_SCALE),
        "normal_speed_score": int(DEFAULT_NORMAL_SPEED_SCORE),
        "turn_curves": {
            "inner_pwm": int(DEFAULT_TURN_CURVE_INNER_PWM),
            "forward": {
                name: {
                    "inner_pwm": int(DEFAULT_TURN_CURVE_INNER_PWM),
                    "outer_pwm": int(pwm),
                }
                for name, pwm in DEFAULT_TURN_CURVE_OUTER_PWMS.items()
            },
            "backward": {
                name: {
                    "inner_pwm": int(DEFAULT_TURN_CURVE_INNER_PWM),
                    "outer_pwm": int(pwm),
                }
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
        "cautious_visibility": dict(DEFAULT_CAUTIOUS_VISIBILITY_CONFIG),
        "vision_jump_guard": dict(DEFAULT_VISION_JUMP_GUARD_CONFIG),
        "visibility_recovery": dict(DEFAULT_VISIBILITY_RECOVERY_CONFIG),
        "pickup_suspect": dict(DEFAULT_PICKUP_SUSPECT_CONFIG),
        "holding_target_vision": dict(DEFAULT_HOLDING_TARGET_VISION_CONFIG),
        "holding_target_distance_calibration": dict(DEFAULT_HOLDING_TARGET_DISTANCE_CALIBRATION_CONFIG),
        "act_stall_guard": dict(DEFAULT_ACT_STALL_GUARD_CONFIG),
        "win_confirmation": dict(DEFAULT_WIN_CONFIRMATION_CONFIG),
        "dist_axis": dict(DEFAULT_FOLLOW_DIST_AXIS_CONFIG),
        "x_axis": dict(DEFAULT_FOLLOW_X_AXIS_CONFIG),
        "y_axis": dict(DEFAULT_FOLLOW_Y_AXIS_CONFIG),
        "step2": _default_step2_like_config(),
        "step3": _default_step2_like_config(),
        "step4": {
            **{key: value for key, value in DEFAULT_STEP3_CONFIG.items() if key != "targets"},
            "targets": dict(DEFAULT_STEP3_CONFIG["targets"]),
        },
        "step2_suspended": False,
        "complete_after_step2": False,
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
        maximum=2000,
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
            cfg["turn_curves"][drive_mode][strength]["inner_pwm"] = _coerce_int(
                raw_curve.get("inner_pwm"),
                cfg["turn_curves"]["inner_pwm"],
                minimum=0,
                maximum=255,
            )
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
    raw_visibility_recovery = (
        raw.get("visibility_recovery") if isinstance(raw.get("visibility_recovery"), dict) else {}
    )
    cfg["visibility_recovery"]["wait_s"] = _coerce_float(
        raw_visibility_recovery.get("wait_s"),
        DEFAULT_VISIBILITY_RECOVERY_CONFIG["wait_s"],
        minimum=0.0,
        maximum=10.0,
    )
    cfg["visibility_recovery"]["poll_s"] = _coerce_float(
        raw_visibility_recovery.get("poll_s"),
        DEFAULT_VISIBILITY_RECOVERY_CONFIG["poll_s"],
        minimum=0.01,
        maximum=2.0,
    )
    cfg["visibility_recovery"]["mast_down_duration_ms"] = _coerce_int(
        raw_visibility_recovery.get("mast_down_duration_ms"),
        DEFAULT_VISIBILITY_RECOVERY_CONFIG["mast_down_duration_ms"],
        minimum=0,
        maximum=_max_mast_act_ms(),
    )
    cfg["visibility_recovery"]["mast_down_pwm"] = _coerce_int(
        raw_visibility_recovery.get("mast_down_pwm"),
        DEFAULT_VISIBILITY_RECOVERY_CONFIG["mast_down_pwm"],
        minimum=1,
        maximum=255,
    )
    raw_pickup_suspect = raw.get("pickup_suspect") if isinstance(raw.get("pickup_suspect"), dict) else {}
    cfg["pickup_suspect"]["enabled"] = bool(
        raw_pickup_suspect.get("enabled", DEFAULT_PICKUP_SUSPECT_CONFIG["enabled"])
    )
    cfg["pickup_suspect"]["min_dist_mm"] = _coerce_float(
        raw_pickup_suspect.get("min_dist_mm"),
        DEFAULT_PICKUP_SUSPECT_CONFIG["min_dist_mm"],
        minimum=0.0,
        maximum=1000.0,
    )
    cfg["pickup_suspect"]["max_y_mm"] = _coerce_float(
        raw_pickup_suspect.get("max_y_mm"),
        DEFAULT_PICKUP_SUSPECT_CONFIG["max_y_mm"],
        minimum=-1000.0,
        maximum=1000.0,
    )
    cfg["pickup_suspect"]["confirm_frames"] = _coerce_int(
        raw_pickup_suspect.get("confirm_frames"),
        DEFAULT_PICKUP_SUSPECT_CONFIG["confirm_frames"],
        minimum=1,
        maximum=5,
    )
    cfg["pickup_suspect"]["confirm_poll_s"] = _coerce_float(
        raw_pickup_suspect.get("confirm_poll_s"),
        DEFAULT_PICKUP_SUSPECT_CONFIG["confirm_poll_s"],
        minimum=0.0,
        maximum=1.0,
    )
    raw_holding_target = (
        raw.get("holding_target_vision") if isinstance(raw.get("holding_target_vision"), dict) else {}
    )
    for key, fallback in DEFAULT_HOLDING_TARGET_VISION_CONFIG.items():
        if isinstance(fallback, bool):
            cfg["holding_target_vision"][key] = bool(raw_holding_target.get(key, fallback))
            continue
        maximum = 100.0 if key in {"min_confidence_pct", "conf_gate_pct"} else 1.0
        cfg["holding_target_vision"][key] = _coerce_float(
            raw_holding_target.get(key),
            fallback,
            minimum=0.0,
            maximum=maximum,
        )
    raw_holding_distance = (
        raw.get("holding_target_distance_calibration")
        if isinstance(raw.get("holding_target_distance_calibration"), dict)
        else {}
    )
    holding_distance_points = []
    for item in raw_holding_distance.get("points", []):
        if not isinstance(item, dict):
            continue
        try:
            holding_distance_points.append(
                {
                    "reported_mm": float(item.get("reported_mm")),
                    "true_mm": float(item.get("true_mm")),
                }
            )
        except (TypeError, ValueError):
            continue
    cfg["holding_target_distance_calibration"] = {
        "enabled": bool(raw_holding_distance.get("enabled", False)) and len(holding_distance_points) >= 2,
        "points": holding_distance_points,
    }
    raw_stall_guard = raw.get("act_stall_guard") if isinstance(raw.get("act_stall_guard"), dict) else {}
    cfg["act_stall_guard"]["enabled"] = bool(
        raw_stall_guard.get("enabled", DEFAULT_ACT_STALL_GUARD_CONFIG["enabled"])
    )
    cfg["act_stall_guard"]["max_no_change_tries"] = _coerce_int(
        raw_stall_guard.get("max_no_change_tries"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["max_no_change_tries"],
        minimum=1,
        maximum=50,
    )
    cfg["act_stall_guard"]["min_axis_delta_mm"] = _coerce_float(
        raw_stall_guard.get("min_axis_delta_mm"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["min_axis_delta_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    cfg["act_stall_guard"]["close_trap_enabled"] = bool(
        raw_stall_guard.get("close_trap_enabled", DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_enabled"])
    )
    cfg["act_stall_guard"]["close_trap_min_abs_dist_err_mm"] = _coerce_float(
        raw_stall_guard.get("close_trap_min_abs_dist_err_mm"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_min_abs_dist_err_mm"],
        minimum=0.0,
        maximum=500.0,
    )
    cfg["act_stall_guard"]["close_trap_max_no_change_tries"] = _coerce_int(
        raw_stall_guard.get("close_trap_max_no_change_tries"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_max_no_change_tries"],
        minimum=1,
        maximum=50,
    )
    cfg["act_stall_guard"]["y_max_no_change_tries"] = _coerce_int(
        raw_stall_guard.get("y_max_no_change_tries"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["y_max_no_change_tries"],
        minimum=1,
        maximum=100,
    )
    cfg["act_stall_guard"]["y_max_no_change_duration_ms"] = _coerce_int(
        raw_stall_guard.get("y_max_no_change_duration_ms"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["y_max_no_change_duration_ms"],
        minimum=1,
        maximum=60000,
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
            maximum=2000,
        )
    cfg["dist_approach_policy"]["full_forward_gap_mm"] = _coerce_float(
        raw_dist_approach.get("full_forward_gap_mm"),
        DEFAULT_DIST_APPROACH_POLICY["full_forward_gap_mm"],
        minimum=0.1,
    )
    cfg["dist_approach_policy"]["near_target_creep_band_mm"] = _coerce_float(
        raw_dist_approach.get("near_target_creep_band_mm"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"],
        minimum=0.0,
    )
    for key in ("near_target_min_pulse_ms", "near_target_max_pulse_ms"):
        cfg["dist_approach_policy"][key] = _coerce_int(
            raw_dist_approach.get(key),
            DEFAULT_DIST_APPROACH_POLICY[key],
            minimum=1,
            maximum=2000,
        )
    for key in (
        "close_target_creep_band_mm",
        "very_close_target_creep_band_mm",
        "pingpong_confirm_abs_dist_err_mm",
        "micro_nudge_abs_dist_err_mm",
    ):
        cfg["dist_approach_policy"][key] = _coerce_float(
            raw_dist_approach.get(key),
            DEFAULT_DIST_APPROACH_POLICY[key],
            minimum=0.0,
        )
    for key in (
        "close_target_max_pulse_ms",
        "very_close_target_max_pulse_ms",
        "micro_nudge_pulse_ms",
        "micro_nudge_min_effective_pulse_ms",
        "min_effective_drive_pulse_ms",
    ):
        cfg["dist_approach_policy"][key] = _coerce_int(
            raw_dist_approach.get(key),
            DEFAULT_DIST_APPROACH_POLICY[key],
            minimum=1,
            maximum=2000,
        )
    cfg["dist_approach_policy"]["micro_nudge_pwm"] = _coerce_int(
        raw_dist_approach.get("micro_nudge_pwm"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pwm"],
        minimum=1,
        maximum=255,
    )
    cfg["dist_approach_policy"]["near_target_forward_veto_mm"] = _coerce_float(
        raw_dist_approach.get("near_target_forward_veto_mm"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_forward_veto_mm"],
        minimum=0.0,
    )
    raw_x_priority = raw.get("x_priority_policy") if isinstance(raw.get("x_priority_policy"), dict) else {}
    for key in (
        "polish_abs_x_mm",
        "huge_dist_gap_mm",
        "huge_dist_tiny_abs_x_mm",
        "simultaneous_dist_outside_min_mm",
        "simultaneous_x_abs_max_mm",
        "turn_settle_s",
        "avoid_forward_left_bias_min_x_outside_mm",
        "avoid_forward_left_bias_min_dist_outside_mm",
    ):
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
    cfg["x_priority_policy"]["attach_y_to_turns"] = bool(
        raw_x_priority.get(
            "attach_y_to_turns",
            DEFAULT_X_PRIORITY_POLICY["attach_y_to_turns"],
        )
    )
    cfg["x_priority_policy"]["avoid_forward_left_bias"] = bool(
        raw_x_priority.get(
            "avoid_forward_left_bias",
            DEFAULT_X_PRIORITY_POLICY["avoid_forward_left_bias"],
        )
    )
    raw_x_dist = raw.get("x_dist_curve_policy") if isinstance(raw.get("x_dist_curve_policy"), dict) else {}
    for key in ("large_dist_gap_mm", "small_x_gap_mm", "near_dist_gap_mm", "wide_x_gap_mm"):
        cfg["x_dist_curve_policy"][key] = _coerce_float(
            raw_x_dist.get(key),
            DEFAULT_X_DIST_CURVE_POLICY[key],
            minimum=0.0,
        )
    cfg["x_dist_curve_policy"]["combined_bias_max_pulse_ms"] = _coerce_int(
        raw_x_dist.get("combined_bias_max_pulse_ms"),
        DEFAULT_X_DIST_CURVE_POLICY["combined_bias_max_pulse_ms"],
        minimum=0,
        maximum=_max_act_ms(),
    )
    cfg["x_dist_curve_policy"]["large_dist_small_x_strength"] = _coerce_curve_strength(
        raw_x_dist.get("large_dist_small_x_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["large_dist_small_x_strength"],
    )
    cfg["x_dist_curve_policy"]["combined_x_dist_strength"] = _coerce_curve_strength(
        raw_x_dist.get("combined_x_dist_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["combined_x_dist_strength"],
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
    cfg["x_only_turn"]["sharp_max_abs_dist_err_mm"] = _coerce_float(
        raw_x_only.get("sharp_max_abs_dist_err_mm"),
        DEFAULT_X_ONLY_TURN_POLICY["sharp_max_abs_dist_err_mm"],
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
    cfg["win_confirmation"]["single_win_allowance_s"] = _coerce_float(
        raw_win_confirmation.get("single_win_allowance_s"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["single_win_allowance_s"],
        minimum=0.0,
        maximum=30.0,
    )
    if "active_attempt_budget_s" in raw_win_confirmation:
        cfg["win_confirmation"]["active_attempt_budget_s"] = _coerce_float(
            raw_win_confirmation.get("active_attempt_budget_s"),
            0.0,
            minimum=0.0,
            maximum=120.0,
        )
    cfg["win_confirmation"]["timeout_y_correction_grace_acts"] = _coerce_int(
        raw_win_confirmation.get("timeout_y_correction_grace_acts"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["timeout_y_correction_grace_acts"],
        minimum=0,
        maximum=10,
    )
    cfg["win_confirmation"]["min_axis_closeness_pct"] = _coerce_float(
        raw_win_confirmation.get("min_axis_closeness_pct"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["min_axis_closeness_pct"],
        minimum=0.0,
        maximum=100.0,
    )
    cfg["win_confirmation"]["min_confidence_pct"] = _coerce_float(
        raw_win_confirmation.get("min_confidence_pct"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["min_confidence_pct"],
        minimum=0.0,
        maximum=100.0,
    )
    raw_cautious_visibility = (
        raw.get("cautious_visibility") if isinstance(raw.get("cautious_visibility"), dict) else {}
    )
    cfg["cautious_visibility"]["motion_min_confidence_pct"] = _coerce_float(
        raw_cautious_visibility.get("motion_min_confidence_pct"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["motion_min_confidence_pct"],
        minimum=0.0,
        maximum=100.0,
    )
    cfg["cautious_visibility"]["pregame_timeout_s"] = _coerce_float(
        raw_cautious_visibility.get("pregame_timeout_s"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_timeout_s"],
        minimum=0.0,
        maximum=30.0,
    )
    cfg["cautious_visibility"]["pregame_sample_s"] = _coerce_float(
        raw_cautious_visibility.get("pregame_sample_s"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_sample_s"],
        minimum=0.01,
        maximum=2.0,
    )
    cfg["cautious_visibility"]["pregame_sample_frames"] = _coerce_int(
        raw_cautious_visibility.get("pregame_sample_frames"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_sample_frames"],
        minimum=1,
        maximum=60,
    )
    cfg["cautious_visibility"]["pregame_required_frames"] = _coerce_int(
        raw_cautious_visibility.get("pregame_required_frames"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_required_frames"],
        minimum=1,
        maximum=60,
    )
    raw_jump_guard = raw.get("vision_jump_guard") if isinstance(raw.get("vision_jump_guard"), dict) else {}
    cfg["vision_jump_guard"]["enabled"] = bool(raw_jump_guard.get("enabled", DEFAULT_VISION_JUMP_GUARD_CONFIG["enabled"]))
    cfg["vision_jump_guard"]["accept_confirmed_jumps"] = bool(
        raw_jump_guard.get(
            "accept_confirmed_jumps",
            DEFAULT_VISION_JUMP_GUARD_CONFIG["accept_confirmed_jumps"],
        )
    )
    cfg["vision_jump_guard"]["confirm_frames"] = _coerce_int(
        raw_jump_guard.get("confirm_frames"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["confirm_frames"],
        minimum=1,
        maximum=10,
    )
    cfg["vision_jump_guard"]["reacquire_frames_after_motion"] = _coerce_int(
        raw_jump_guard.get("reacquire_frames_after_motion"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["reacquire_frames_after_motion"],
        minimum=0,
        maximum=10,
    )
    for key in ("hard_reject_dist_jump_mm", "hard_reject_observe_s", "hard_reject_observe_poll_s", "post_act_stabilize_s"):
        cfg["vision_jump_guard"][key] = _coerce_float(
            raw_jump_guard.get(key),
            DEFAULT_VISION_JUMP_GUARD_CONFIG[key],
            minimum=0.0,
            maximum=500.0 if key.endswith("_mm") else 10.0,
        )
    cfg["vision_jump_guard"]["hard_reject_max_recoveries_per_step"] = _coerce_int(
        raw_jump_guard.get("hard_reject_max_recoveries_per_step"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["hard_reject_max_recoveries_per_step"],
        minimum=0,
        maximum=10,
    )
    for key in (
        "max_dist_jump_mm",
        "max_x_jump_mm",
        "max_y_jump_mm",
        "max_vector_jump_mm",
        "max_abs_dist_mm",
        "edge_recovery_min_abs_x_mm",
        "edge_recovery_max_abs_dist_mm",
        "edge_recovery_min_confidence_pct",
        "confirm_window_mm",
    ):
        cfg["vision_jump_guard"][key] = _coerce_float(
            raw_jump_guard.get(key),
            DEFAULT_VISION_JUMP_GUARD_CONFIG[key],
            minimum=0.0,
            maximum=500.0,
        )
    raw_dist_axis = raw.get("dist_axis") if isinstance(raw.get("dist_axis"), dict) else {}
    cfg["dist_axis"]["win_target_mm"] = _coerce_float(
        raw_dist_axis.get("win_target_mm"),
        DEFAULT_FOLLOW_DIST_AXIS_CONFIG["win_target_mm"],
        minimum=0.0,
    )
    cfg["dist_axis"]["win_tol_mm"] = _coerce_float(
        raw_dist_axis.get("win_tol_mm"),
        DEFAULT_FOLLOW_DIST_AXIS_CONFIG["win_tol_mm"],
        minimum=0.0,
    )
    positive_dist_cmd = str(
        raw_dist_axis.get(
            "positive_error_cmd",
            DEFAULT_FOLLOW_DIST_AXIS_CONFIG["positive_error_cmd"],
        )
    ).strip().lower()
    cfg["dist_axis"]["positive_error_cmd"] = (
        positive_dist_cmd if positive_dist_cmd in {"f", "b"} else DEFAULT_FOLLOW_DIST_AXIS_CONFIG["positive_error_cmd"]
    )
    raw_x_axis = raw.get("x_axis") if isinstance(raw.get("x_axis"), dict) else {}
    cfg["x_axis"]["win_target_mm"] = _coerce_float(
        raw_x_axis.get("win_target_mm"),
        DEFAULT_FOLLOW_X_AXIS_CONFIG["win_target_mm"],
    )
    cfg["x_axis"]["win_tol_mm"] = _coerce_float(
        raw_x_axis.get("win_tol_mm"),
        DEFAULT_FOLLOW_X_AXIS_CONFIG["win_tol_mm"],
        minimum=0.0,
    )
    positive_turn_cmd = str(
        raw_x_axis.get(
            "positive_error_turn_cmd",
            DEFAULT_FOLLOW_X_AXIS_CONFIG["positive_error_turn_cmd"],
        )
    ).strip().lower()
    cfg["x_axis"]["positive_error_turn_cmd"] = (
        positive_turn_cmd if positive_turn_cmd in {"l", "r"} else DEFAULT_FOLLOW_X_AXIS_CONFIG["positive_error_turn_cmd"]
    )
    raw_y_axis = raw.get("y_axis") if isinstance(raw.get("y_axis"), dict) else {}
    cfg["y_axis"]["enabled"] = bool(raw_y_axis.get("enabled", DEFAULT_FOLLOW_Y_AXIS_CONFIG["enabled"]))
    y_duration_keys = {
        "lock_on_pulse_ms",
        "mast_pulse_ms",
        "mast_min_pulse_ms",
        "mast_max_pulse_ms",
        "finish_mast_pulse_ms",
        "finish_mast_min_pulse_ms",
        "finish_mast_max_pulse_ms",
        "max_step1_mast_up_ms",
        "spool_reversal_mast_max_ms",
    }
    for key, fallback in DEFAULT_FOLLOW_Y_AXIS_CONFIG.items():
        if key == "enabled":
            continue
        if key in {"protect_requires_brick_below", "lock_on_enabled", "mast_duration_uses_tolerance_gap"}:
            cfg["y_axis"][key] = bool(raw_y_axis.get(key, fallback))
            continue
        if key in {"mast_up_duration_curve", "mast_down_duration_curve"}:
            cfg["y_axis"][key] = _sanitize_y_duration_curve(raw_y_axis.get(key, fallback))
            continue
        if key == "hard_floor_y_mm":
            cfg["y_axis"][key] = _coerce_optional_float(raw_y_axis.get(key), fallback)
            continue
        if key in y_duration_keys:
            cfg["y_axis"][key] = _coerce_int(
                raw_y_axis.get(key),
                fallback,
                minimum=0 if key == "max_step1_mast_up_ms" else 1,
                maximum=_max_mast_act_ms(),
            )
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
    raw_step2 = raw.get("step2") if isinstance(raw.get("step2"), dict) else {}
    step2 = cfg["step2"]
    mast_cmd = str(raw_step2.get("seat_mast_cmd", DEFAULT_STEP2_CONFIG["seat_mast_cmd"])).strip().lower()
    drive_cmd = str(raw_step2.get("seat_drive_cmd", DEFAULT_STEP2_CONFIG["seat_drive_cmd"])).strip().lower()
    step2["seat_mast_cmd"] = mast_cmd if mast_cmd in {"u", "d"} else DEFAULT_STEP2_CONFIG["seat_mast_cmd"]
    step2["seat_drive_cmd"] = drive_cmd if drive_cmd in {"f", "b"} else DEFAULT_STEP2_CONFIG["seat_drive_cmd"]
    step2["seat_mast_pwm"] = _coerce_int(
        raw_step2.get("seat_mast_pwm"),
        DEFAULT_STEP2_CONFIG["seat_mast_pwm"],
        minimum=1,
        maximum=255,
    )
    step2["seat_drive_pwm"] = _coerce_int(
        raw_step2.get("seat_drive_pwm"),
        DEFAULT_STEP2_CONFIG["seat_drive_pwm"],
        minimum=1,
        maximum=255,
    )
    step2["seat_mast_duration_ms"] = _coerce_int(
        raw_step2.get("seat_mast_duration_ms"),
        DEFAULT_STEP2_CONFIG["seat_mast_duration_ms"],
        minimum=0,
        maximum=_max_mast_act_ms(),
    )
    step2["seat_drive_duration_ms"] = _coerce_int(
        raw_step2.get("seat_drive_duration_ms"),
        DEFAULT_STEP2_CONFIG["seat_drive_duration_ms"],
        minimum=0,
        maximum=5000,
    )
    step2["post_seat_pause_s"] = _coerce_float(
        raw_step2.get("post_seat_pause_s"),
        DEFAULT_STEP2_CONFIG["post_seat_pause_s"],
        minimum=0.0,
        maximum=3.0,
    )
    step2["step_timeout_s"] = _coerce_float(
        raw_step2.get("step_timeout_s"),
        DEFAULT_STEP2_CONFIG["step_timeout_s"],
        minimum=0.0,
        maximum=120.0,
    )
    step2["recovery_creep_enabled"] = bool(
        raw_step2.get("recovery_creep_enabled", DEFAULT_STEP2_CONFIG["recovery_creep_enabled"])
    )
    step2["recovery_creep_pulse_ms"] = _coerce_int(
        raw_step2.get("recovery_creep_pulse_ms"),
        DEFAULT_STEP2_CONFIG["recovery_creep_pulse_ms"],
        minimum=1,
        maximum=400,
    )
    step2["recovery_creep_max_attempts"] = _coerce_int(
        raw_step2.get("recovery_creep_max_attempts"),
        DEFAULT_STEP2_CONFIG["recovery_creep_max_attempts"],
        minimum=0,
        maximum=20,
    )
    step2["recovery_creep_settle_s"] = _coerce_float(
        raw_step2.get("recovery_creep_settle_s"),
        DEFAULT_STEP2_CONFIG["recovery_creep_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    step2["visibility_recovery_creep_enabled"] = bool(
        raw_step2.get(
            "visibility_recovery_creep_enabled",
            DEFAULT_STEP2_CONFIG["visibility_recovery_creep_enabled"],
        )
    )
    step2["visibility_recovery_wait_s"] = _coerce_float(
        raw_step2.get("visibility_recovery_wait_s"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_wait_s"],
        minimum=0.0,
        maximum=10.0,
    )
    step2["visibility_recovery_wait_poll_s"] = _coerce_float(
        raw_step2.get("visibility_recovery_wait_poll_s"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_wait_poll_s"],
        minimum=0.01,
        maximum=2.0,
    )
    step2["visibility_recovery_creep_pulse_ms"] = _coerce_int(
        raw_step2.get("visibility_recovery_creep_pulse_ms"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_creep_pulse_ms"],
        minimum=1,
        maximum=400,
    )
    step2["visibility_recovery_creep_max_attempts"] = _coerce_int(
        raw_step2.get("visibility_recovery_creep_max_attempts"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_creep_max_attempts"],
        minimum=0,
        maximum=10,
    )
    step2["visibility_recovery_creep_settle_s"] = _coerce_float(
        raw_step2.get("visibility_recovery_creep_settle_s"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_creep_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    step2["precision_settle_enabled"] = bool(
        raw_step2.get("precision_settle_enabled", DEFAULT_STEP2_CONFIG["precision_settle_enabled"])
    )
    step2["precision_max_attempts"] = _coerce_int(
        raw_step2.get("precision_max_attempts"),
        DEFAULT_STEP2_CONFIG["precision_max_attempts"],
        minimum=1,
        maximum=40,
    )
    step2["post_precision_recovery_cycles"] = _coerce_int(
        raw_step2.get("post_precision_recovery_cycles"),
        DEFAULT_STEP2_CONFIG["post_precision_recovery_cycles"],
        minimum=0,
        maximum=3,
    )
    for key in (
        "precision_drive_min_pulse_ms",
        "precision_drive_max_pulse_ms",
        "precision_mast_pulse_ms",
        "precision_mast_small_gap_min_pulse_ms",
        "precision_mast_small_gap_max_pulse_ms",
    ):
        step2[key] = _coerce_int(
            raw_step2.get(key),
            DEFAULT_STEP2_CONFIG[key],
            minimum=1,
            maximum=1000,
        )
    for key in ("precision_mast_small_gap_max_mm", "precision_mast_small_gap_pwm_scale"):
        step2[key] = _coerce_float(
            raw_step2.get(key),
            DEFAULT_STEP2_CONFIG[key],
            minimum=0.0,
            maximum=1.0 if key.endswith("_scale") else 100.0,
        )
    positive_cmd = str(raw_step2.get("precision_dist_positive_cmd", DEFAULT_STEP2_CONFIG["precision_dist_positive_cmd"])).strip().lower()
    step2["precision_dist_positive_cmd"] = positive_cmd if positive_cmd in {"f", "b"} else DEFAULT_STEP2_CONFIG["precision_dist_positive_cmd"]
    step2["precision_settle_s"] = _coerce_float(
        raw_step2.get("precision_settle_s"),
        DEFAULT_STEP2_CONFIG["precision_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    step2["freeze_xz_after_xz_target"] = bool(
        raw_step2.get(
            "freeze_xz_after_xz_target",
            DEFAULT_STEP2_CONFIG["freeze_xz_after_xz_target"],
        )
    )
    raw_targets = raw_step2.get("targets") if isinstance(raw_step2.get("targets"), dict) else {}
    target_cfg = step2["targets"]
    for key in ("dist_mm", "x_mm", "y_mm"):
        target_cfg[key] = _coerce_optional_float(raw_targets.get(key), DEFAULT_STEP2_CONFIG["targets"][key])
    for key in ("dist_tol_mm", "x_tol_mm", "y_tol_mm"):
        target_cfg[key] = _coerce_optional_float(
            raw_targets.get(key),
            DEFAULT_STEP2_CONFIG["targets"][key],
            minimum=0.0,
        )
    raw_semi = raw_step2.get("semi_happy_targets") if isinstance(raw_step2.get("semi_happy_targets"), dict) else {}
    semi_cfg = dict(DEFAULT_STEP2_CONFIG["semi_happy_targets"])
    for key in ("dist_mm", "x_mm", "y_mm"):
        semi_cfg[key] = _coerce_optional_float(raw_semi.get(key), DEFAULT_STEP2_CONFIG["semi_happy_targets"][key])
    for key in ("dist_tol_mm", "x_tol_mm", "y_tol_mm"):
        semi_cfg[key] = _coerce_optional_float(
            raw_semi.get(key),
            DEFAULT_STEP2_CONFIG["semi_happy_targets"][key],
            minimum=0.0,
        )
    step2["semi_happy_targets"] = semi_cfg
    raw_step3 = raw.get("step3") if isinstance(raw.get("step3"), dict) else {}
    _apply_step2_like_config(raw_step3, cfg["step3"])
    raw_step4 = raw.get("step4") if isinstance(raw.get("step4"), dict) else {}
    raw_step4_lift = raw_step4 if raw_step4 else raw_step3
    step4 = cfg["step4"]
    lift_cmd = str(raw_step4_lift.get("lift_mast_cmd", DEFAULT_STEP3_CONFIG["lift_mast_cmd"])).strip().lower()
    step4["lift_mast_cmd"] = lift_cmd if lift_cmd in {"u", "d"} else DEFAULT_STEP3_CONFIG["lift_mast_cmd"]
    step4["lift_mast_pwm"] = _coerce_int(
        raw_step4_lift.get("lift_mast_pwm"),
        DEFAULT_STEP3_CONFIG["lift_mast_pwm"],
        minimum=1,
        maximum=255,
    )
    step4["lift_pulse_ms"] = _coerce_int(
        raw_step4_lift.get("lift_pulse_ms"),
        DEFAULT_STEP3_CONFIG["lift_pulse_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    step4["lift_settle_s"] = _coerce_float(
        raw_step4_lift.get("lift_settle_s"),
        DEFAULT_STEP3_CONFIG["lift_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    step4["max_lift_attempts"] = _coerce_int(
        raw_step4_lift.get("max_lift_attempts"),
        DEFAULT_STEP3_CONFIG["max_lift_attempts"],
        minimum=1,
        maximum=50,
    )
    step4["step_timeout_s"] = _coerce_float(
        raw_step4_lift.get("step_timeout_s"),
        DEFAULT_STEP3_CONFIG["step_timeout_s"],
        minimum=0.0,
        maximum=120.0,
    )
    step4["initial_lift_before_gate"] = bool(
        raw_step4_lift.get(
            "initial_lift_before_gate",
            DEFAULT_STEP3_CONFIG["initial_lift_before_gate"],
        )
    )
    step4["no_visibility_fallback_enabled"] = bool(
        raw_step4_lift.get(
            "no_visibility_fallback_enabled",
            DEFAULT_STEP3_CONFIG["no_visibility_fallback_enabled"],
        )
    )
    fallback_cmd = str(
        raw_step4_lift.get(
            "no_visibility_fallback_mast_cmd",
            DEFAULT_STEP3_CONFIG["no_visibility_fallback_mast_cmd"],
        )
    ).strip().lower()
    step4["no_visibility_fallback_mast_cmd"] = (
        fallback_cmd if fallback_cmd in {"u", "d"} else DEFAULT_STEP3_CONFIG["no_visibility_fallback_mast_cmd"]
    )
    step4["no_visibility_fallback_mast_pwm"] = _coerce_int(
        raw_step4_lift.get("no_visibility_fallback_mast_pwm"),
        DEFAULT_STEP3_CONFIG["no_visibility_fallback_mast_pwm"],
        minimum=1,
        maximum=255,
    )
    step4["no_visibility_fallback_duration_ms"] = _coerce_int(
        raw_step4_lift.get("no_visibility_fallback_duration_ms"),
        DEFAULT_STEP3_CONFIG["no_visibility_fallback_duration_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    step4["no_visibility_fallback_settle_s"] = _coerce_float(
        raw_step4_lift.get("no_visibility_fallback_settle_s"),
        DEFAULT_STEP3_CONFIG["no_visibility_fallback_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    raw_step4_targets = raw_step4_lift.get("targets") if isinstance(raw_step4_lift.get("targets"), dict) else {}
    step4_targets = step4["targets"]
    step4_targets["y_mm"] = _coerce_optional_float(
        raw_step4_targets.get("y_mm"),
        DEFAULT_STEP3_CONFIG["targets"]["y_mm"],
    )
    step4_targets["y_tol_mm"] = _coerce_optional_float(
        raw_step4_targets.get("y_tol_mm"),
        DEFAULT_STEP3_CONFIG["targets"]["y_tol_mm"],
        minimum=0.0,
    )
    raw_profiles = raw.get("game_profiles") if isinstance(raw.get("game_profiles"), dict) else {}
    raw_profile = raw_profiles.get(_active_game_profile()) if isinstance(raw_profiles.get(_active_game_profile()), dict) else {}
    cfg["step2_suspended"] = bool(raw_profile.get("step2_suspended", cfg.get("step2_suspended", False)))
    cfg["complete_after_step2"] = bool(raw_profile.get("complete_after_step2", cfg.get("complete_after_step2", False)))
    _merge_profile_turning_overrides(cfg, raw_profile)
    raw_profile_dist_axis = raw_profile.get("dist_axis") if isinstance(raw_profile.get("dist_axis"), dict) else {}
    for key, fallback in DEFAULT_FOLLOW_DIST_AXIS_CONFIG.items():
        if key in raw_profile_dist_axis:
            if key == "positive_error_cmd":
                profile_dist_cmd = str(raw_profile_dist_axis.get(key) or "").strip().lower()
                if profile_dist_cmd in {"f", "b"}:
                    cfg["dist_axis"][key] = profile_dist_cmd
                continue
            cfg["dist_axis"][key] = _coerce_float(
                raw_profile_dist_axis.get(key),
                cfg["dist_axis"].get(key, fallback),
                minimum=0.0,
            )
    raw_profile_x_axis = raw_profile.get("x_axis") if isinstance(raw_profile.get("x_axis"), dict) else {}
    for key, fallback in DEFAULT_FOLLOW_X_AXIS_CONFIG.items():
        if key in raw_profile_x_axis:
            if key == "positive_error_turn_cmd":
                profile_turn_cmd = str(raw_profile_x_axis.get(key) or "").strip().lower()
                if profile_turn_cmd in {"l", "r"}:
                    cfg["x_axis"][key] = profile_turn_cmd
                continue
            cfg["x_axis"][key] = _coerce_float(
                raw_profile_x_axis.get(key),
                cfg["x_axis"].get(key, fallback),
                minimum=0.0 if key == "win_tol_mm" else None,
            )
    raw_profile_y_axis = raw_profile.get("y_axis") if isinstance(raw_profile.get("y_axis"), dict) else {}
    for key in DEFAULT_FOLLOW_Y_AXIS_CONFIG:
        if key in raw_profile_y_axis:
            if key in {"enabled", "protect_requires_brick_below", "lock_on_enabled", "mast_duration_uses_tolerance_gap"}:
                cfg["y_axis"][key] = bool(raw_profile_y_axis.get(key))
                continue
            if key in {"mast_up_duration_curve", "mast_down_duration_curve"}:
                cfg["y_axis"][key] = _sanitize_y_duration_curve(raw_profile_y_axis.get(key))
                continue
            if key == "hard_floor_y_mm":
                cfg["y_axis"][key] = _coerce_optional_float(
                    raw_profile_y_axis.get(key),
                    cfg["y_axis"].get(key),
                )
                continue
            if key in y_duration_keys:
                cfg["y_axis"][key] = _coerce_int(
                    raw_profile_y_axis.get(key),
                    cfg["y_axis"].get(key),
                    minimum=0 if key == "max_step1_mast_up_ms" else 1,
                    maximum=_max_mast_act_ms(),
                )
                continue
            minimum = 0.0 if key not in {"win_target_mm", "reset_target_mm"} else None
            cfg["y_axis"][key] = _coerce_float(
                raw_profile_y_axis.get(key),
                cfg["y_axis"].get(key),
                minimum=minimum,
            )
    raw_profile_step2 = raw_profile.get("step2") if isinstance(raw_profile.get("step2"), dict) else {}
    if "nickname" in raw_profile_step2:
        step2["nickname"] = str(raw_profile_step2.get("nickname") or "").strip()
    for key in ("seat_mast_cmd", "seat_drive_cmd"):
        if key in raw_profile_step2:
            mode = str(raw_profile_step2.get(key) or "").strip().lower()
            if mode in {"u", "d"} and key == "seat_mast_cmd":
                step2[key] = mode
            if mode in {"f", "b"} and key == "seat_drive_cmd":
                step2[key] = mode
    for key in ("seat_mast_pwm", "seat_drive_pwm"):
        if key in raw_profile_step2:
            step2[key] = _coerce_int(raw_profile_step2.get(key), step2.get(key), minimum=1, maximum=255)
    if "seat_mast_duration_ms" in raw_profile_step2:
        step2["seat_mast_duration_ms"] = _cap_mast_duration_ms(
            step2.get("seat_mast_cmd"),
            raw_profile_step2.get("seat_mast_duration_ms"),
            step2.get("seat_mast_duration_ms"),
            minimum=0,
            maximum=5000 if bool(raw_profile_step2.get("blind_mast_only", step2.get("blind_mast_only", False))) else None,
        )
    if "seat_drive_duration_ms" in raw_profile_step2:
        step2["seat_drive_duration_ms"] = _coerce_int(
            raw_profile_step2.get("seat_drive_duration_ms"),
            step2.get("seat_drive_duration_ms"),
            minimum=0,
            maximum=5000,
        )
    if "post_seat_pause_s" in raw_profile_step2:
        step2["post_seat_pause_s"] = _coerce_float(
            raw_profile_step2.get("post_seat_pause_s"),
            step2.get("post_seat_pause_s"),
            minimum=0.0,
            maximum=3.0,
        )
    if "recovery_creep_enabled" in raw_profile_step2:
        step2["recovery_creep_enabled"] = bool(raw_profile_step2.get("recovery_creep_enabled"))
    if "recovery_creep_pulse_ms" in raw_profile_step2:
        step2["recovery_creep_pulse_ms"] = _coerce_int(
            raw_profile_step2.get("recovery_creep_pulse_ms"),
            step2.get("recovery_creep_pulse_ms"),
            minimum=1,
            maximum=400,
        )
    if "recovery_creep_max_attempts" in raw_profile_step2:
        step2["recovery_creep_max_attempts"] = _coerce_int(
            raw_profile_step2.get("recovery_creep_max_attempts"),
            step2.get("recovery_creep_max_attempts"),
            minimum=0,
            maximum=20,
        )
    if "recovery_creep_settle_s" in raw_profile_step2:
        step2["recovery_creep_settle_s"] = _coerce_float(
            raw_profile_step2.get("recovery_creep_settle_s"),
            step2.get("recovery_creep_settle_s"),
            minimum=0.0,
            maximum=2.0,
        )
    if "visibility_recovery_creep_enabled" in raw_profile_step2:
        step2["visibility_recovery_creep_enabled"] = bool(raw_profile_step2.get("visibility_recovery_creep_enabled"))
    if "visibility_recovery_wait_s" in raw_profile_step2:
        step2["visibility_recovery_wait_s"] = _coerce_float(
            raw_profile_step2.get("visibility_recovery_wait_s"),
            step2.get("visibility_recovery_wait_s"),
            minimum=0.0,
            maximum=10.0,
        )
    if "visibility_recovery_wait_poll_s" in raw_profile_step2:
        step2["visibility_recovery_wait_poll_s"] = _coerce_float(
            raw_profile_step2.get("visibility_recovery_wait_poll_s"),
            step2.get("visibility_recovery_wait_poll_s"),
            minimum=0.01,
            maximum=2.0,
        )
    if "visibility_recovery_creep_pulse_ms" in raw_profile_step2:
        step2["visibility_recovery_creep_pulse_ms"] = _coerce_int(
            raw_profile_step2.get("visibility_recovery_creep_pulse_ms"),
            step2.get("visibility_recovery_creep_pulse_ms"),
            minimum=1,
            maximum=400,
        )
    if "visibility_recovery_creep_max_attempts" in raw_profile_step2:
        step2["visibility_recovery_creep_max_attempts"] = _coerce_int(
            raw_profile_step2.get("visibility_recovery_creep_max_attempts"),
            step2.get("visibility_recovery_creep_max_attempts"),
            minimum=0,
            maximum=10,
        )
    if "visibility_recovery_creep_settle_s" in raw_profile_step2:
        step2["visibility_recovery_creep_settle_s"] = _coerce_float(
            raw_profile_step2.get("visibility_recovery_creep_settle_s"),
            step2.get("visibility_recovery_creep_settle_s"),
            minimum=0.0,
            maximum=2.0,
        )
    if "precision_settle_enabled" in raw_profile_step2:
        step2["precision_settle_enabled"] = bool(raw_profile_step2.get("precision_settle_enabled"))
    if "blind_mast_only" in raw_profile_step2:
        step2["blind_mast_only"] = bool(raw_profile_step2.get("blind_mast_only"))
    for key in (
        "precision_max_attempts",
        "post_precision_recovery_cycles",
        "precision_drive_min_pulse_ms",
        "precision_drive_max_pulse_ms",
        "precision_mast_pulse_ms",
        "precision_mast_small_gap_min_pulse_ms",
        "precision_mast_small_gap_max_pulse_ms",
    ):
        if key in raw_profile_step2:
            step2[key] = _coerce_int(
                raw_profile_step2.get(key),
                step2.get(key, DEFAULT_STEP2_CONFIG.get(key)),
                minimum=1,
                maximum=3 if key == "post_precision_recovery_cycles" else (1000 if key != "precision_max_attempts" else 40),
            )
    for key in ("precision_mast_small_gap_max_mm", "precision_mast_small_gap_pwm_scale"):
        if key in raw_profile_step2:
            step2[key] = _coerce_float(
                raw_profile_step2.get(key),
                step2.get(key, DEFAULT_STEP2_CONFIG[key]),
                minimum=0.0,
                maximum=1.0 if key.endswith("_scale") else 100.0,
            )
    if "precision_dist_positive_cmd" in raw_profile_step2:
        profile_positive_cmd = str(raw_profile_step2.get("precision_dist_positive_cmd") or "").strip().lower()
        if profile_positive_cmd in {"f", "b"}:
            step2["precision_dist_positive_cmd"] = profile_positive_cmd
    if "precision_settle_s" in raw_profile_step2:
        step2["precision_settle_s"] = _coerce_float(
            raw_profile_step2.get("precision_settle_s"),
            step2.get("precision_settle_s", DEFAULT_STEP2_CONFIG["precision_settle_s"]),
            minimum=0.0,
            maximum=2.0,
        )
    if "step_timeout_s" in raw_profile_step2:
        step2["step_timeout_s"] = _coerce_float(
            raw_profile_step2.get("step_timeout_s"),
            step2.get("step_timeout_s", DEFAULT_STEP2_CONFIG["step_timeout_s"]),
            minimum=0.0,
            maximum=120.0,
        )
    if "freeze_xz_after_xz_target" in raw_profile_step2:
        step2["freeze_xz_after_xz_target"] = bool(raw_profile_step2.get("freeze_xz_after_xz_target"))
    raw_profile_step2_semi = (
        raw_profile_step2.get("semi_happy_targets")
        if isinstance(raw_profile_step2.get("semi_happy_targets"), dict)
        else {}
    )
    for key in ("dist_mm", "x_mm", "y_mm"):
        if key in raw_profile_step2_semi:
            semi_cfg[key] = _coerce_optional_float(raw_profile_step2_semi.get(key), semi_cfg.get(key))
    for key in ("dist_tol_mm", "x_tol_mm", "y_tol_mm"):
        if key in raw_profile_step2_semi:
            semi_cfg[key] = _coerce_optional_float(raw_profile_step2_semi.get(key), semi_cfg.get(key), minimum=0.0)
    step2["semi_happy_targets"] = semi_cfg
    raw_profile_step2_targets = (
        raw_profile_step2.get("targets") if isinstance(raw_profile_step2.get("targets"), dict) else {}
    )
    for key in ("dist_mm", "x_mm", "y_mm"):
        if key in raw_profile_step2_targets:
            target_cfg[key] = (
                None
                if raw_profile_step2_targets.get(key) is None
                else _coerce_optional_float(raw_profile_step2_targets.get(key), target_cfg.get(key))
            )
    for key in ("dist_tol_mm", "x_tol_mm", "y_tol_mm"):
        if key in raw_profile_step2_targets:
            target_cfg[key] = (
                None
                if raw_profile_step2_targets.get(key) is None
                else _coerce_optional_float(raw_profile_step2_targets.get(key), target_cfg.get(key), minimum=0.0)
            )
    raw_profile_step3 = raw_profile.get("step3") if isinstance(raw_profile.get("step3"), dict) else {}
    _apply_step2_like_config(raw_profile_step3, cfg["step3"])
    raw_profile_step4 = raw_profile.get("step4") if isinstance(raw_profile.get("step4"), dict) else {}
    # Backward compatibility: older profiles used step3 as the lift step.
    raw_profile_step4_lift = raw_profile_step4
    if not raw_profile_step4_lift and any(key in raw_profile_step3 for key in ("lift_mast_cmd", "lift_mast_pwm", "lift_pulse_ms", "max_lift_attempts")):
        raw_profile_step4_lift = raw_profile_step3
    if "lift_mast_cmd" in raw_profile_step4_lift:
        profile_lift_cmd = str(raw_profile_step4_lift.get("lift_mast_cmd") or "").strip().lower()
        if profile_lift_cmd in {"u", "d"}:
            step4["lift_mast_cmd"] = profile_lift_cmd
    for key in ("lift_mast_pwm", "lift_pulse_ms", "max_lift_attempts", "no_visibility_fallback_mast_pwm", "no_visibility_fallback_duration_ms"):
        if key in raw_profile_step4_lift:
            maximum = _max_mast_act_ms() if key in {"lift_pulse_ms", "no_visibility_fallback_duration_ms"} else (
                50 if key == "max_lift_attempts" else 5000
            )
            step4[key] = _coerce_int(
                raw_profile_step4_lift.get(key),
                step4.get(key, DEFAULT_STEP3_CONFIG.get(key)),
                minimum=1,
                maximum=maximum,
            )
    for key in ("lift_settle_s", "no_visibility_fallback_settle_s", "step_timeout_s"):
        if key in raw_profile_step4_lift:
            maximum = 120.0 if key == "step_timeout_s" else 2.0
            step4[key] = _coerce_float(
                raw_profile_step4_lift.get(key),
                step4.get(key, DEFAULT_STEP3_CONFIG.get(key)),
                minimum=0.0,
                maximum=maximum,
            )
    if "no_visibility_fallback_enabled" in raw_profile_step4_lift:
        step4["no_visibility_fallback_enabled"] = bool(raw_profile_step4_lift.get("no_visibility_fallback_enabled"))
    if "initial_lift_before_gate" in raw_profile_step4_lift:
        step4["initial_lift_before_gate"] = bool(raw_profile_step4_lift.get("initial_lift_before_gate"))
    if "no_visibility_fallback_mast_cmd" in raw_profile_step4_lift:
        fallback_cmd = str(raw_profile_step4_lift.get("no_visibility_fallback_mast_cmd") or "").strip().lower()
        if fallback_cmd in {"u", "d"}:
            step4["no_visibility_fallback_mast_cmd"] = fallback_cmd
    raw_profile_step4_targets = (
        raw_profile_step4_lift.get("targets") if isinstance(raw_profile_step4_lift.get("targets"), dict) else {}
    )
    step4_targets = step4.get("targets") if isinstance(step4.get("targets"), dict) else {}
    if "y_mm" in raw_profile_step4_targets:
        step4_targets["y_mm"] = _coerce_optional_float(raw_profile_step4_targets.get("y_mm"), step4_targets.get("y_mm"))
    if "y_tol_mm" in raw_profile_step4_targets:
        step4_targets["y_tol_mm"] = _coerce_optional_float(
            raw_profile_step4_targets.get("y_tol_mm"),
            step4_targets.get("y_tol_mm"),
            minimum=0.0,
        )
    step4["targets"] = step4_targets
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
            "holding_pulse_scale": 1.0,
            "arc_algorithm": {"points": [dict(point) for point in DEFAULT_RESET_ARC_ALGORITHM_POINTS]},
            "sharp_finish": {
                "enabled": True,
                "duration_ms": int(RESET_SHARP_FINISH_MS),
                "mode": "faster_wheel_only",
            },
            "low_x_extra_sharp_turn": {
                "enabled": True,
                "threshold_abs_x_mm": float(RESET_LOW_X_EXTRA_TURN_THRESHOLD_MM),
                "duration_ms": int(RESET_LOW_X_EXTRA_TURN_DURATION_MS),
                "slower_pwm": int(RESET_LOW_X_EXTRA_TURN_SLOWER_PWM),
                "faster_pwm": int(RESET_LOW_X_EXTRA_TURN_FASTER_PWM),
            },
            "straight_back_first": dict(DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG),
            "x_goal_curve": dict(DEFAULT_RESET_X_GOAL_CURVE_CONFIG),
            "adjustment": {
                "enabled": True,
                "max_attempts": 6,
                "pulse_min_ms": 60,
                "pulse_max_ms": 260,
                "settle_s": 0.12,
            },
        },
        "mast_up": {
            "enabled": True,
            "min_duration_ms": int(RESET_MAST_UP_MIN_MS),
            "max_duration_ms": int(RESET_MAST_UP_MAX_MS),
            "pwm": int(RESET_MAST_UP_PWM),
            "cmd": RESET_MAST_UP_CMD,
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

    reverse_turn = dict(raw.get("reverse_turn")) if isinstance(raw.get("reverse_turn"), dict) else {}
    raw_profiles = raw.get("game_profiles") if isinstance(raw.get("game_profiles"), dict) else {}
    raw_profile = raw_profiles.get(_active_game_profile()) if isinstance(raw_profiles.get(_active_game_profile()), dict) else {}
    profile_reverse_turn = (
        raw_profile.get("reverse_turn")
        if isinstance(raw_profile.get("reverse_turn"), dict)
        else {}
    )
    if profile_reverse_turn:
        for key, value in profile_reverse_turn.items():
            if isinstance(value, dict) and isinstance(reverse_turn.get(key), dict):
                merged = dict(reverse_turn[key])
                merged.update(value)
                reverse_turn[key] = merged
            else:
                reverse_turn[key] = value
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
    cfg["reverse_turn"]["holding_pulse_scale"] = _coerce_float(
        reverse_turn.get("holding_pulse_scale"),
        1.0,
        minimum=0.1,
        maximum=1.0,
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
    raw_low_x_extra = (
        reverse_turn.get("low_x_extra_sharp_turn")
        if isinstance(reverse_turn.get("low_x_extra_sharp_turn"), dict)
        else {}
    )
    cfg["reverse_turn"]["low_x_extra_sharp_turn"] = {
        "enabled": bool(raw_low_x_extra.get("enabled", True)),
        "threshold_abs_x_mm": _coerce_float(
            raw_low_x_extra.get("threshold_abs_x_mm"),
            RESET_LOW_X_EXTRA_TURN_THRESHOLD_MM,
            minimum=0.0,
        ),
        "duration_ms": _coerce_int(
            raw_low_x_extra.get("duration_ms"),
            RESET_LOW_X_EXTRA_TURN_DURATION_MS,
            minimum=0,
            maximum=1000,
        ),
        "slower_pwm": _coerce_int(
            raw_low_x_extra.get("slower_pwm"),
            RESET_LOW_X_EXTRA_TURN_SLOWER_PWM,
            minimum=1,
            maximum=255,
        ),
        "faster_pwm": _coerce_int(
            raw_low_x_extra.get("faster_pwm"),
            RESET_LOW_X_EXTRA_TURN_FASTER_PWM,
            minimum=1,
            maximum=255,
        ),
    }
    raw_straight_back = (
        reverse_turn.get("straight_back_first")
        if isinstance(reverse_turn.get("straight_back_first"), dict)
        else {}
    )
    cfg["reverse_turn"]["straight_back_first"] = {
        "enabled": bool(raw_straight_back.get("enabled", DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["enabled"])),
        "duration_ms": _coerce_int(
            raw_straight_back.get("duration_ms"),
            DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["duration_ms"],
            minimum=1,
            maximum=5000,
        ),
        "duration_min_ms": _coerce_int(
            raw_straight_back.get("duration_min_ms"),
            0,
            minimum=0,
            maximum=5000,
        ),
        "duration_max_ms": _coerce_int(
            raw_straight_back.get("duration_max_ms"),
            0,
            minimum=0,
            maximum=5000,
        ),
        "pwm": _coerce_int(
            raw_straight_back.get("pwm"),
            DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["pwm"],
            minimum=1,
            maximum=255,
        ),
        "mast_up_delay_fraction": _coerce_float(
            raw_straight_back.get("mast_up_delay_fraction"),
            DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["mast_up_delay_fraction"],
            minimum=0.0,
            maximum=0.95,
        ),
    }
    raw_x_goal_curve = (
        reverse_turn.get("x_goal_curve")
        if isinstance(reverse_turn.get("x_goal_curve"), dict)
        else {}
    )
    cfg["reverse_turn"]["x_goal_curve"] = {
        "enabled": bool(raw_x_goal_curve.get("enabled", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["enabled"])),
        "target_fraction": _coerce_float(
            raw_x_goal_curve.get("target_fraction"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["target_fraction"],
            minimum=0.0,
            maximum=1.0,
        ),
        "max_duration_ms": _coerce_int(
            raw_x_goal_curve.get("max_duration_ms"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["max_duration_ms"],
            minimum=1,
            maximum=3000,
        ),
        "min_duration_ms": _coerce_int(
            raw_x_goal_curve.get("min_duration_ms"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["min_duration_ms"],
            minimum=0,
            maximum=3000,
        ),
        "chunk_ms": _coerce_int(
            raw_x_goal_curve.get("chunk_ms"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["chunk_ms"],
            minimum=50,
            maximum=1000,
        ),
        "settle_s": _coerce_float(
            raw_x_goal_curve.get("settle_s"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["settle_s"],
            minimum=0.0,
            maximum=1.0,
        ),
        "inner_pwm": _coerce_int(
            raw_x_goal_curve.get("inner_pwm"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["inner_pwm"],
            minimum=0,
            maximum=255,
        ),
        "outer_pwm": _coerce_int(
            raw_x_goal_curve.get("outer_pwm"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["outer_pwm"],
            minimum=1,
            maximum=255,
        ),
    }
    raw_adjustment = (
        reverse_turn.get("adjustment")
        if isinstance(reverse_turn.get("adjustment"), dict)
        else {}
    )
    cfg["reverse_turn"]["adjustment"] = {
        "enabled": bool(raw_adjustment.get("enabled", True)),
        "max_attempts": _coerce_int(raw_adjustment.get("max_attempts"), 6, minimum=0, maximum=20),
        "pulse_min_ms": _coerce_int(raw_adjustment.get("pulse_min_ms"), 60, minimum=1, maximum=400),
        "pulse_max_ms": _coerce_int(raw_adjustment.get("pulse_max_ms"), 260, minimum=1, maximum=500),
        "settle_s": _coerce_float(raw_adjustment.get("settle_s"), 0.12, minimum=0.0, maximum=2.0),
    }
    mast_up = raw.get("mast_up") if isinstance(raw.get("mast_up"), dict) else {}
    cfg["mast_up"]["enabled"] = bool(mast_up.get("enabled", True))
    cfg["mast_up"]["min_duration_ms"] = _coerce_int(
        mast_up.get("min_duration_ms"),
        RESET_MAST_UP_MIN_MS,
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    cfg["mast_up"]["max_duration_ms"] = _coerce_int(
        mast_up.get("max_duration_ms"),
        RESET_MAST_UP_MAX_MS,
        minimum=1,
        maximum=_max_mast_act_ms(),
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
    mast_cmd = str(mast_up.get("cmd", cfg["mast_up"].get("cmd", RESET_MAST_UP_CMD)) or "").strip().lower()
    cfg["mast_up"]["cmd"] = mast_cmd if mast_cmd in {"u", "d"} else RESET_MAST_UP_CMD
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


def _visibility_recovery_config() -> dict:
    raw = _follow_motion_config().get("visibility_recovery")
    cfg = raw if isinstance(raw, dict) else {}
    return {
        "wait_s": _coerce_float(
            cfg.get("wait_s"),
            DEFAULT_VISIBILITY_RECOVERY_CONFIG["wait_s"],
            minimum=0.0,
            maximum=10.0,
        ),
        "poll_s": _coerce_float(
            cfg.get("poll_s"),
            DEFAULT_VISIBILITY_RECOVERY_CONFIG["poll_s"],
            minimum=0.01,
            maximum=2.0,
        ),
        "mast_down_duration_ms": _coerce_int(
            cfg.get("mast_down_duration_ms"),
            DEFAULT_VISIBILITY_RECOVERY_CONFIG["mast_down_duration_ms"],
            minimum=0,
            maximum=_max_mast_act_ms(),
        ),
        "mast_down_pwm": _coerce_int(
            cfg.get("mast_down_pwm"),
            DEFAULT_VISIBILITY_RECOVERY_CONFIG["mast_down_pwm"],
            minimum=1,
            maximum=255,
        ),
    }


def _holding_target_vision_config() -> dict:
    raw = _follow_motion_config().get("holding_target_vision")
    cfg = raw if isinstance(raw, dict) else {}
    result = {}
    for key, fallback in DEFAULT_HOLDING_TARGET_VISION_CONFIG.items():
        if isinstance(fallback, bool):
            result[key] = bool(cfg.get(key, fallback))
            continue
        maximum = 100.0 if key in {"min_confidence_pct", "conf_gate_pct"} else 1.0
        result[key] = _coerce_float(cfg.get(key), fallback, minimum=0.0, maximum=maximum)
    return result


def _holding_target_runtime_tuning() -> dict:
    cfg = _holding_target_vision_config()
    return {
        "confidence": cfg["confidence"],
        "conf_gate_pct": cfg["conf_gate_pct"],
        "hsv_cyan_coverage_min": cfg["hsv_cyan_coverage_min"],
        "hsv_min_area_ratio": cfg["hsv_min_area_ratio"],
        "trust_detector_boxes": cfg["trust_detector_boxes"],
        "require_cyan_shape": cfg["require_cyan_shape"],
        "far_suspect_enabled": cfg["far_suspect_enabled"],
    }


def _holding_target_distance_calibration_config() -> dict:
    raw = _follow_motion_config().get("holding_target_distance_calibration")
    cfg = raw if isinstance(raw, dict) else {}
    points = cfg.get("points") if isinstance(cfg.get("points"), list) else []
    return {
        "enabled": bool(cfg.get("enabled", False)) and len(points) >= 2,
        "points": points,
    }


def _apply_holding_target_distance_calibration(reading: dict) -> dict:
    if not isinstance(reading, dict):
        return reading
    if reading.get("holding_xz_source") == "unmasked_stack":
        return reading
    if not bool(reading.get("target_masked_for_holding")):
        return reading
    return apply_holding_distance_calibration_to_reading(
        reading,
        config=_holding_target_distance_calibration_config(),
    )


def _apply_unmasked_stack_xz_if_configured(target_reading: dict, stack_reading: dict) -> dict:
    if not bool(_holding_target_vision_config().get("use_unmasked_stack_xz", False)):
        return target_reading
    if not isinstance(target_reading, dict) or not isinstance(stack_reading, dict):
        return target_reading
    if not bool(stack_reading.get("confident")):
        return target_reading
    out = dict(target_reading)
    for key in ("dist_mm", "x_mm"):
        if stack_reading.get(key) is not None:
            out[key] = stack_reading.get(key)
    result = list(out.get("result") or [])
    if len(result) >= 4:
        result[2] = out.get("dist_mm", result[2])
        result[3] = out.get("x_mm", result[3])
        out["result"] = result
    out["holding_xz_source"] = "unmasked_stack"
    return out


def _act_stall_guard_config() -> dict:
    raw = _follow_motion_config().get("act_stall_guard")
    cfg = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(cfg.get("enabled", DEFAULT_ACT_STALL_GUARD_CONFIG["enabled"])),
        "max_no_change_tries": _coerce_int(
            cfg.get("max_no_change_tries"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["max_no_change_tries"],
            minimum=1,
            maximum=50,
        ),
        "min_axis_delta_mm": _coerce_float(
            cfg.get("min_axis_delta_mm"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["min_axis_delta_mm"],
            minimum=0.0,
            maximum=100.0,
        ),
        "close_trap_enabled": bool(
            cfg.get("close_trap_enabled", DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_enabled"])
        ),
        "close_trap_min_abs_dist_err_mm": _coerce_float(
            cfg.get("close_trap_min_abs_dist_err_mm"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_min_abs_dist_err_mm"],
            minimum=0.0,
            maximum=300.0,
        ),
        "close_trap_max_no_change_tries": _coerce_int(
            cfg.get("close_trap_max_no_change_tries"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_max_no_change_tries"],
            minimum=1,
            maximum=10,
        ),
        "y_max_no_change_tries": _coerce_int(
            cfg.get("y_max_no_change_tries"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["y_max_no_change_tries"],
            minimum=1,
            maximum=100,
        ),
        "y_max_no_change_duration_ms": _coerce_int(
            cfg.get("y_max_no_change_duration_ms"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["y_max_no_change_duration_ms"],
            minimum=1,
            maximum=60000,
        ),
        "recovery_boost_enabled": bool(
            cfg.get("recovery_boost_enabled", DEFAULT_ACT_STALL_GUARD_CONFIG["recovery_boost_enabled"])
        ),
        "recovery_boost_step_pct": _coerce_float(
            cfg.get("recovery_boost_step_pct"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["recovery_boost_step_pct"],
            minimum=0.0,
            maximum=100.0,
        ),
        "recovery_boost_max_scale": _coerce_float(
            cfg.get("recovery_boost_max_scale"),
            DEFAULT_ACT_STALL_GUARD_CONFIG["recovery_boost_max_scale"],
            minimum=1.0,
            maximum=3.0,
        ),
    }


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


def _sharp_x_only_max_abs_dist_err_mm() -> float:
    cfg = _follow_motion_config()
    raw = cfg.get("x_only_turn") if isinstance(cfg.get("x_only_turn"), dict) else {}
    return _coerce_float(
        raw.get("sharp_max_abs_dist_err_mm"),
        DEFAULT_X_ONLY_TURN_POLICY["sharp_max_abs_dist_err_mm"],
        minimum=0.0,
    )


def _sharp_x_only_turn_allowed(dist_err: float) -> bool:
    try:
        abs_dist_err = abs(float(dist_err))
    except (TypeError, ValueError):
        abs_dist_err = 0.0
    return abs_dist_err <= float(_sharp_x_only_max_abs_dist_err_mm())


def _max_act_ms() -> int:
    cfg = _follow_motion_config()
    return _coerce_int(
        cfg.get("max_act_ms"),
        DEFAULT_MAX_ACT_MS,
        minimum=1,
        maximum=2000,
    )


def _bounded_act_duration_ms(duration_ms: int | float | None) -> int:
    return min(
        int(_max_act_ms()),
        _coerce_int(duration_ms, PULSE_MS, minimum=1),
    )


def _wheel_act_duration_ms(duration_ms: int | float | None, *, allow_long_duration: bool = False) -> int:
    if bool(allow_long_duration):
        return _coerce_int(duration_ms, PULSE_MS, minimum=1, maximum=2000)
    return _bounded_act_duration_ms(duration_ms)


def _reset_act_duration_ms(reset_cfg: dict | None = None) -> int:
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    timeout_s = _coerce_float(
        cfg.get("timeout_s"),
        RESET_REVERSE_TURN_TIMEOUT_S,
        minimum=0.001,
    )
    duration_ms = _coerce_int(
        cfg.get("pulse_ms"),
        RESET_REVERSE_TURN_PULSE_MS,
        minimum=1,
        maximum=max(1, int(round(float(timeout_s) * 1000.0))),
    )
    if _active_game_profile() == "holding":
        scale = _coerce_float(cfg.get("holding_pulse_scale"), 1.0, minimum=0.1, maximum=1.0)
        duration_ms = max(1, int(round(float(duration_ms) * float(scale))))
    return int(duration_ms)


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


def _scaled_action_pwm_for_boost(pwm: int | float | None, scale: float) -> int:
    try:
        pwm_val = int(round(float(pwm)))
    except (TypeError, ValueError):
        pwm_val = 0
    if pwm_val <= 0:
        return 0
    try:
        scale_val = float(scale)
    except (TypeError, ValueError):
        scale_val = 1.0
    scale_val = max(1.0, float(scale_val))
    return int(_telemetry_robot.clamp_pwm(int(round(float(pwm_val) * scale_val))))


def _boost_curve_pwms(curve: dict, scale: float | None) -> dict:
    if not isinstance(curve, dict):
        return {}
    try:
        scale_val = float(scale)
    except (TypeError, ValueError):
        scale_val = 1.0
    if scale_val <= 1.0:
        return dict(curve)
    out = dict(curve)
    for key in ("inner_pwm", "outer_pwm"):
        out[key] = _scaled_action_pwm_for_boost(out.get(key), scale_val)
    out["recovery_boost_scale"] = float(scale_val)
    return out


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
        x_err = abs(_x_err_for_reading(reading) or 0.0)
    except (TypeError, ValueError):
        x_err = 0.0

    return _curve_strength_for_abs_x_err(float(x_err))


def _turn_cmd_to_close_x_gap(x_mm: float) -> str | None:
    positive_cmd = str(_follow_x_axis_config().get("positive_error_turn_cmd", "r")).strip().lower()
    if positive_cmd not in {"l", "r"}:
        positive_cmd = "r"
    if float(x_mm) > 0.0:
        return positive_cmd
    if float(x_mm) < 0.0:
        return "l" if positive_cmd == "r" else "r"
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
    global_inner_pwm = _coerce_int(
        turn_curves.get("inner_pwm"),
        DEFAULT_TURN_CURVE_INNER_PWM,
        minimum=0,
        maximum=255,
    )
    return {
        "inner_pwm": _coerce_int(
            raw_curve.get("inner_pwm"),
            global_inner_pwm,
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
    floor_pwm = max(int(_pwm_floor_for_cmd("f")), int(_pwm_floor_for_cmd("b")))
    out["inner_pwm"] = _telemetry_robot.clamp_pwm(max(int(floor_pwm), int(out.get("inner_pwm", 0) or 0)))
    out["outer_pwm"] = _telemetry_robot.clamp_pwm(
        max(int(floor_pwm), int(round(float(out["outer_pwm"]) * scale)))
    )
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
    floor_pwm = max(int(_pwm_floor_for_cmd("f")), int(_pwm_floor_for_cmd("b")))
    out["inner_pwm"] = _telemetry_robot.clamp_pwm(max(int(floor_pwm), int(out.get("inner_pwm", 0) or 0)))
    out["outer_pwm"] = _telemetry_robot.clamp_pwm(
        max(int(floor_pwm), int(round(float(out["outer_pwm"]) * scale)))
    )
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


def _production_turn_curve_for_reading(
    *,
    cmd: str,
    drive_mode: str,
    reading: dict,
    x_err_mm: float,
) -> tuple[dict, int | None] | None:
    try:
        from helper_close_gaps import production_turn_drive_curve_plan
    except Exception:
        return None
    try:
        current_dist = float((reading or {}).get("dist_mm"))
    except (TypeError, ValueError):
        current_dist = 0.0
    plan = production_turn_drive_curve_plan(
        cmd=str(cmd or "").strip().lower(),
        drive_mode=str(drive_mode or "").strip().lower(),
        current_dist_mm=current_dist,
        x_err_mm=float(x_err_mm),
    )
    if not isinstance(plan, dict):
        return None
    profile = plan.get("profile_override") if isinstance(plan.get("profile_override"), dict) else {}
    try:
        base_pwm = int(plan.get("pwm_override"))
    except (TypeError, ValueError):
        base_pwm = 0
    if base_pwm <= 0:
        return None
    outer_ratio = _coerce_float(profile.get("outer_ratio"), 1.0, minimum=0.01, maximum=1.0)
    inner_ratio = _coerce_float(profile.get("inner_ratio"), 0.0, minimum=0.0, maximum=1.0)
    outer_pwm = _telemetry_robot.clamp_pwm(int(round(float(base_pwm) * float(outer_ratio))))
    inner_pwm = _telemetry_robot.clamp_pwm(int(round(float(base_pwm) * float(inner_ratio))))
    curve = {
        "inner_pwm": int(inner_pwm),
        "outer_pwm": int(outer_pwm),
        "strength": "production_sharp",
        "drive_mode": str(drive_mode or profile.get("drive_mode") or "backward").strip().lower(),
        "source": str(plan.get("source") or "production_turn_drive_trials"),
        "curve_name": str(plan.get("curve_name") or ""),
        "curve_value_mm": plan.get("curve_value_mm"),
        "production": True,
    }
    duration_override = plan.get("duration_override_ms")
    try:
        duration_ms = int(round(float(duration_override))) if duration_override is not None else None
    except (TypeError, ValueError):
        duration_ms = None
    return curve, duration_ms


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
    mast_pwm: int | float | None = None,
    mast_duration_ms: int | float | None = None,
    use_production_curve: bool = False,
    recovery_boost_scale: float | None = None,
) -> dict | None:
    try:
        x_err = _x_err_for_reading(reading)
        x_abs = abs(float(x_err if x_err is not None else (reading or {}).get("x_mm", 0.0)))
    except (TypeError, ValueError):
        x_abs = 0.0
        x_err = 0.0
    production_curve = (
        _production_turn_curve_for_reading(
            cmd=cmd,
            drive_mode=drive_mode,
            reading=reading,
            x_err_mm=float(x_err if x_err is not None else x_abs),
        )
        if bool(use_production_curve)
        else None
    )
    if production_curve is not None:
        curve, production_duration_ms = production_curve
        if production_duration_ms is not None:
            duration_ms = int(max(int(duration_ms), int(production_duration_ms), int(_min_motion_duration_ms(cmd))))
    else:
        curve = (
            _adaptive_turn_curve_for_drive_mode(drive_mode, x_abs)
            if str(strength or "").strip().lower() == "adaptive"
            else _turn_curve_for_drive_mode(drive_mode, strength)
        )
    curve = _boost_curve_pwms(curve, recovery_boost_scale)
    actions = _turn_curve_actions(drive_mode=drive_mode, cmd=cmd, curve=curve)
    if not actions:
        return None
    scaled_actions = _actions_with_mast(
        _scaled_actions(actions),
        mast_cmd,
        mast_pwm=mast_pwm,
        mast_duration_ms=mast_duration_ms,
    )
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
    mast_pwm: int | float | None = None,
    mast_duration_ms: int | float | None = None,
    recovery_boost_scale: float | None = None,
    allow_long_duration: bool = False,
) -> dict | None:
    drive_key = str(drive_mode or "").strip().lower()
    logical_cmd = "b" if drive_key == "backward" else "f"
    try:
        x_err = _x_err_for_reading(reading)
        x_abs = abs(float(x_err if x_err is not None else (reading or {}).get("x_mm", 0.0)))
    except (TypeError, ValueError):
        x_abs = 0.0
    curve = (
        _adaptive_turn_bias_curve_for_drive_mode(drive_key, x_abs)
        if str(strength or "").strip().lower() == "adaptive"
        else _turn_bias_curve_for_drive_mode(drive_key, strength)
    )
    curve = _boost_curve_pwms(curve, recovery_boost_scale)
    actions = _turn_bias_actions(drive_mode=drive_key, turn_cmd=turn_cmd, curve=curve)
    if not actions:
        return None
    scaled_actions = _actions_with_mast(
        _scaled_actions(actions),
        mast_cmd,
        mast_pwm=mast_pwm,
        mast_duration_ms=mast_duration_ms,
    )
    send_result = guarded_send_custom_actions_pwm(
        robot,
        logical_cmd,
        scaled_actions,
        duration_ms=_wheel_act_duration_ms(duration_ms, allow_long_duration=allow_long_duration),
        reading=reading,
        context=f"{context}_{curve['drive_mode']}_{curve['strength']}_{turn_cmd}",
    )
    if send_result is None:
        return {
            "cmd_sent": logical_cmd,
            "turn_cmd": str(turn_cmd),
            "actions": scaled_actions,
            "duration_ms": _wheel_act_duration_ms(duration_ms, allow_long_duration=allow_long_duration),
            "x_curve": dict(curve),
        }
    if isinstance(send_result, dict):
        send_result["x_curve"] = dict(curve)
    return send_result


def _send_drive_nudge(
    robot: Robot,
    *,
    cmd: str,
    turn_cmd: str,
    pwm: int,
    duration_ms: int,
    reading: dict,
    mast_cmd: str | None = None,
    mast_pwm: int | float | None = None,
    mast_duration_ms: int | float | None = None,
    recovery_boost_scale: float | None = None,
) -> dict | None:
    drive_key = "backward" if str(cmd or "").strip().lower() == "b" else "forward"
    turn_key = str(turn_cmd or "").strip().lower()
    if turn_key not in {"l", "r"}:
        turn_key = "l"
    curve = {
        "inner_pwm": 0,
        "outer_pwm": _scaled_action_pwm_for_boost(pwm, recovery_boost_scale or 1.0),
        "strength": "micro_nudge",
        "drive_mode": drive_key,
    }
    if recovery_boost_scale is not None:
        try:
            curve["recovery_boost_scale"] = float(recovery_boost_scale)
        except (TypeError, ValueError):
            pass
    actions = _turn_curve_actions(drive_mode=drive_key, cmd=turn_key, curve=curve)
    if not actions:
        return None
    scaled_actions = _actions_with_mast(
        _scaled_actions(actions),
        mast_cmd,
        mast_pwm=mast_pwm,
        mast_duration_ms=mast_duration_ms,
    )
    send_result = guarded_send_custom_actions_pwm(
        robot,
        "b" if drive_key == "backward" else "f",
        scaled_actions,
        duration_ms=_bounded_act_duration_ms(duration_ms),
        reading=reading,
        context=f"follow_dist_micro_nudge_{drive_key}_{turn_key}",
    )
    if send_result is None:
        return {
            "cmd_sent": "b" if drive_key == "backward" else "f",
            "turn_cmd": turn_key,
            "actions": scaled_actions,
            "duration_ms": _bounded_act_duration_ms(duration_ms),
            "x_curve": dict(curve),
        }
    if isinstance(send_result, dict):
        send_result["turn_cmd"] = turn_key
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


def _mast_action_spec(
    direction: str | None,
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
    # Custom action specs are already at the Uno target/action layer. Keep the
    # operator-facing plan logical: U raises the mast, D lowers it.
    wire_action = cmd
    return {
        "target": "m",
        "action": wire_action,
        "pwm": _scaled_pwm_for_cmd(cmd, mast_pwm),
        "duration_ms": _cap_mast_duration_ms(
            cmd,
            mast_duration_ms,
            y_cfg.get("mast_pulse_ms", PULSE_MS),
            minimum=1,
        ),
    }


def _actions_with_mast(
    actions,
    mast_cmd: str | None,
    *,
    mast_pwm: int | float | None = None,
    mast_duration_ms: int | float | None = None,
):
    out = [dict(action) for action in (actions or []) if isinstance(action, dict)]
    mast_action = _mast_action_spec(mast_cmd, pwm=mast_pwm, duration_ms=mast_duration_ms)
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


def _reset_wheel_only_actions(actions) -> list[dict]:
    """Reset is always wheel-only; mast is locked until explicit game exceptions."""
    out = []
    stripped = 0
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        if str(action.get("target") or "").strip().lower() == "m":
            stripped += 1
            continue
        out.append(dict(action))
    if stripped > 0:
        print(f"[RESET] Mast locked: stripped {int(stripped)} mast action(s) from reset packet.", flush=True)
    return out


def _send_reset_custom_actions_pwm(
    robot: Robot,
    cmd: str,
    action_specs,
    *,
    duration_ms: int,
    reading: dict,
    context: str,
):
    return guarded_send_custom_actions_pwm(
        robot,
        cmd,
        _reset_wheel_only_actions(action_specs),
        duration_ms=duration_ms,
        reading=reading,
        context=context,
    )


def _drive(
    robot: Robot,
    direction: str,
    reading: dict,
    *,
    mast_cmd: str | None = None,
    mast_pwm: int | float | None = None,
    mast_duration_ms: int | float | None = None,
    pwm: int | float | None = None,
    duration_ms: int | float | None = None,
    recovery_boost_scale: float | None = None,
    allow_long_duration: bool = False,
) -> None:
    """Straight drive: 'f' forward, 'b' backward."""
    cmd = str(direction or "").strip().lower()
    requested_pwm = _normal_drive_pwm(cmd) if pwm is None else pwm
    try:
        boost_scale = float(recovery_boost_scale)
    except (TypeError, ValueError):
        boost_scale = 1.0
    if boost_scale > 1.0:
        requested_pwm = _scaled_action_pwm_for_boost(requested_pwm, boost_scale)
        drive_pwm = _scaled_pwm_for_cmd(cmd, requested_pwm)
    else:
        drive_pwm = _clamp_to_approved_straight_drive_pwm(cmd, requested_pwm)
    act_ms = _wheel_act_duration_ms(
        PULSE_MS if duration_ms is None else duration_ms,
        allow_long_duration=allow_long_duration,
    )
    if str(mast_cmd or "").strip().lower() in {"u", "d"}:
        actions = _actions_with_mast(
            _straight_drive_actions(cmd, drive_pwm),
            mast_cmd,
            mast_pwm=mast_pwm,
            mast_duration_ms=mast_duration_ms,
        )
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


def _follow_step2_config() -> dict:
    cfg = _follow_motion_config()
    step2 = cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {}
    return step2 if isinstance(step2, dict) else {}


def _game_complete_after_step2() -> bool:
    return bool(_follow_motion_config().get("complete_after_step2", False))


def _follow_step3_config() -> dict:
    cfg = _follow_motion_config()
    step3 = cfg.get("step3") if isinstance(cfg.get("step3"), dict) else {}
    return step3 if isinstance(step3, dict) else {}


def _step3_kind(step3_cfg: dict | None = None) -> str:
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step3_config()
    kind = str((cfg or {}).get("kind") or DEFAULT_STEP3_RETREAT_CONFIG["kind"]).strip().lower()
    return kind if kind in {"seat", "retreat"} else "seat"


def _profile_step1_dist_target_mm(profile: str | None = None) -> float:
    profile_key = str(profile or _active_game_profile()).strip().lower()
    if profile_key not in {"empty", "holding"}:
        profile_key = "empty"
    try:
        payload = json.loads(ROBOT_MODEL_FILE.read_text())
        raw = payload.get(FOLLOW_MOTION_CONFIG_KEY) if isinstance(payload, dict) else {}
        profiles = raw.get("game_profiles") if isinstance(raw, dict) and isinstance(raw.get("game_profiles"), dict) else {}
        profile_cfg = profiles.get(profile_key) if isinstance(profiles.get(profile_key), dict) else {}
        dist_axis = profile_cfg.get("dist_axis") if isinstance(profile_cfg.get("dist_axis"), dict) else {}
        return _coerce_float(
            dist_axis.get("win_target_mm"),
            TARGET_DIST_MM,
            minimum=0.0,
        )
    except Exception:
        return float(TARGET_DIST_MM)


def _profile_step1_dist_tol_mm(profile: str | None = None) -> float:
    profile_key = str(profile or _active_game_profile()).strip().lower()
    if profile_key not in {"empty", "holding"}:
        profile_key = "empty"
    try:
        payload = json.loads(ROBOT_MODEL_FILE.read_text())
        raw = payload.get(FOLLOW_MOTION_CONFIG_KEY) if isinstance(payload, dict) else {}
        profiles = raw.get("game_profiles") if isinstance(raw, dict) and isinstance(raw.get("game_profiles"), dict) else {}
        profile_cfg = profiles.get(profile_key) if isinstance(profiles.get(profile_key), dict) else {}
        dist_axis = profile_cfg.get("dist_axis") if isinstance(profile_cfg.get("dist_axis"), dict) else {}
        return _coerce_float(
            dist_axis.get("win_tol_mm"),
            DIST_TOL_MM,
            minimum=0.0,
        )
    except Exception:
        return float(DIST_TOL_MM)


def _reset_min_attempt_dist_mm(reset_cfg: dict | None = None) -> float:
    raw = reset_cfg if isinstance(reset_cfg, dict) else {}
    configured = raw.get("min_attempt_dist_mm")
    if configured is not None:
        return _coerce_float(configured, 0.0, minimum=0.0)
    return max(0.0, _profile_step1_dist_target_mm("empty") - _profile_step1_dist_tol_mm("empty"))


def _follow_step4_config() -> dict:
    cfg = _follow_motion_config()
    step4 = cfg.get("step4") if isinstance(cfg.get("step4"), dict) else {}
    return step4 if isinstance(step4, dict) else {}


def _step3_missing_target_keys(step3_cfg: dict | None = None) -> list[str]:
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step4_config()
    targets = cfg.get("targets") if isinstance(cfg.get("targets"), dict) else {}
    missing = []
    for key in ("y_mm", "y_tol_mm"):
        if targets.get(key) is None:
            missing.append(key)
    return missing


def _step3_targets_ready(reading: dict, step3_cfg: dict | None = None) -> tuple[bool, str]:
    missing = _step3_missing_target_keys(step3_cfg)
    if missing:
        return False, "step4_targets_pending:" + ",".join(missing)
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step4_config()
    targets = cfg.get("targets") if isinstance(cfg.get("targets"), dict) else {}
    try:
        y_mm = float((reading or {}).get("y_mm"))
        y_target = float(targets.get("y_mm"))
        y_tol = float(targets.get("y_tol_mm"))
    except (TypeError, ValueError):
        return False, "invalid_step4_reading"
    return abs(y_mm - y_target) <= y_tol, "step4_targets_scored"


def _run_step3_no_visibility_fallback(
    vision: BrickDetector,
    robot: Robot,
    step3_cfg: dict,
    *,
    trigger_reason: str,
    reading: dict | None = None,
    attempts: int = 0,
    send_result: dict | None = None,
) -> dict:
    if not bool(step3_cfg.get("no_visibility_fallback_enabled", True)):
        return {
            "success": False,
            "target_met": False,
            "fallback_reset_ok": False,
            "holding": False,
            "reason": trigger_reason,
            "reading": reading or {},
            "attempts": attempts,
            "send_result": send_result,
        }
    cmd = str(
        step3_cfg.get("no_visibility_fallback_mast_cmd")
        or DEFAULT_STEP3_CONFIG["no_visibility_fallback_mast_cmd"]
    ).strip().lower()
    if cmd not in {"u", "d"}:
        cmd = DEFAULT_STEP3_CONFIG["no_visibility_fallback_mast_cmd"]
    pwm = _scaled_pwm_for_cmd(
        cmd,
        step3_cfg.get(
            "no_visibility_fallback_mast_pwm",
            DEFAULT_STEP3_CONFIG["no_visibility_fallback_mast_pwm"],
        ),
    )
    duration_ms = _coerce_int(
        step3_cfg.get("no_visibility_fallback_duration_ms"),
        DEFAULT_STEP3_CONFIG["no_visibility_fallback_duration_ms"],
        minimum=1,
        maximum=10000,
    )
    settle_s = _coerce_float(
        step3_cfg.get("no_visibility_fallback_settle_s"),
        DEFAULT_STEP3_CONFIG["no_visibility_fallback_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    duration_ms = _cap_mast_duration_ms(
        cmd,
        duration_ms,
        DEFAULT_STEP3_CONFIG["no_visibility_fallback_duration_ms"],
        minimum=1,
    )
    fallback_send = robot.send_command_pwm(cmd, pwm, duration_ms=duration_ms)
    time.sleep(float(duration_ms) / 1000.0 + float(settle_s))
    _stop_robot(robot)
    _reset_follow_reading_history(vision)
    after = _read_brick_measurement(vision)
    return {
        "success": True,
        "target_met": False,
        "fallback_reset_ok": True,
        "holding": bool(after.get("holding")),
        "reason": "step4_no_visibility_fallback_lift",
        "trigger_reason": trigger_reason,
        "reading": after,
        "fallback_cmd": cmd,
        "fallback_pwm": pwm,
        "fallback_duration_ms": duration_ms,
        "fallback_send_result": fallback_send,
        "attempts": attempts,
        "send_result": send_result,
    }


def _deadline_from_timeout_s(timeout_s: float | int | None) -> float | None:
    try:
        timeout_val = float(timeout_s)
    except (TypeError, ValueError):
        return None
    if timeout_val <= 0.0:
        return None
    return time.monotonic() + float(timeout_val)


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= float(deadline)


def _run_step3_lift_sequence(vision: BrickDetector, robot: Robot) -> dict:
    step3 = _follow_step4_config()
    if bool(step3.get("fixed_lift_only", DEFAULT_STEP3_CONFIG["fixed_lift_only"])):
        lift_cmd = str(step3.get("lift_mast_cmd") or DEFAULT_STEP3_CONFIG["lift_mast_cmd"]).strip().lower()
        if lift_cmd not in {"u", "d"}:
            lift_cmd = DEFAULT_STEP3_CONFIG["lift_mast_cmd"]
        lift_pwm = _scaled_pwm_for_cmd(lift_cmd, step3.get("lift_mast_pwm"))
        pulse_ms = _coerce_int(
            step3.get("fixed_lift_duration_ms"),
            DEFAULT_STEP3_CONFIG["fixed_lift_duration_ms"],
            minimum=1,
            maximum=10000,
        )
        settle_s = _coerce_float(
            step3.get("lift_settle_s"),
            DEFAULT_STEP3_CONFIG["lift_settle_s"],
            minimum=0.0,
            maximum=2.0,
        )
        send_result = robot.send_command_pwm(lift_cmd, lift_pwm, duration_ms=pulse_ms)
        time.sleep(float(pulse_ms) / 1000.0 + float(settle_s))
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        try:
            after = _read_brick_measurement(vision)
        except Exception:
            after = {"confident": False, "reason": "step4_fixed_lift_read_failed"}
        return {
            "success": True,
            "target_met": True,
            "holding": True,
            "reason": "step4_fixed_lift_only",
            "reading": after,
            "attempts": 1,
            "send_result": send_result,
            "duration_ms": int(pulse_ms),
            "mast_duration_ms": int(pulse_ms),
        }
    targets = step3.get("targets") if isinstance(step3.get("targets"), dict) else {}
    missing = _step3_missing_target_keys(step3)
    before = _read_brick_measurement(vision)
    if not bool(before.get("confident")):
        before = _wait_for_visibility_recovery(
            vision,
            robot,
            before,
            context="step4_start",
        )
    if not bool(before.get("confident")):
        return _run_step3_no_visibility_fallback(
            vision,
            robot,
            step3,
            trigger_reason="brick_not_confident_before_step4",
            reading=before,
        )
    if missing:
        return {
            "success": False,
            "target_met": False,
            "holding": False,
            "reason": "step4_targets_pending:" + ",".join(missing),
            "reading": before,
        }
    lift_cmd = str(step3.get("lift_mast_cmd") or DEFAULT_STEP3_CONFIG["lift_mast_cmd"]).strip().lower()
    lift_pwm = _scaled_pwm_for_cmd(lift_cmd, step3.get("lift_mast_pwm"))
    pulse_ms = _cap_mast_duration_ms(
        lift_cmd,
        step3.get("lift_pulse_ms"),
        DEFAULT_STEP3_CONFIG["lift_pulse_ms"],
        minimum=1,
    )
    max_attempts = _coerce_int(
        step3.get("max_lift_attempts"),
        DEFAULT_STEP3_CONFIG["max_lift_attempts"],
        minimum=1,
        maximum=50,
    )
    settle_s = _coerce_float(
        step3.get("lift_settle_s"),
        DEFAULT_STEP3_CONFIG["lift_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    current = before
    attempts = 0
    last_send = None
    step_timeout_s = _coerce_float(
        step3.get("step_timeout_s"),
        DEFAULT_STEP3_CONFIG["step_timeout_s"],
        minimum=0.0,
        maximum=120.0,
    )
    deadline = _deadline_from_timeout_s(step_timeout_s)
    try:
        y_target = float(targets.get("y_mm"))
        y_tol = float(targets.get("y_tol_mm"))
    except (TypeError, ValueError):
        return {
            "success": False,
            "target_met": False,
            "holding": False,
            "reason": "invalid_step4_targets",
            "reading": before,
        }
    initial_lift_before_gate = bool(step3.get("initial_lift_before_gate", False))
    while attempts <= int(max_attempts):
        if _deadline_expired(deadline):
            _stop_robot(robot)
            return {
                "success": True,
                "target_met": False,
                "holding": False,
                "reason": "step4_timeout",
                "reading": current,
                "attempts": attempts,
                "send_result": last_send,
                "step_timeout_s": float(step_timeout_s),
            }
        if not bool(current.get("confident")):
            current = _wait_for_visibility_recovery(
                vision,
                robot,
                current,
                context="step4_lift",
            )
        if not bool(current.get("confident")):
            return _run_step3_no_visibility_fallback(
                vision,
                robot,
                step3,
                trigger_reason="lost_confident_brick_during_step4",
                reading=current,
                attempts=attempts,
                send_result=last_send,
            )
        try:
            y_mm = float(current.get("y_mm"))
        except (TypeError, ValueError):
            return {
                "success": False,
                "target_met": False,
                "holding": False,
                "reason": "invalid_step4_reading",
                "reading": current,
                "attempts": attempts,
                "send_result": last_send,
            }
        should_score_before_first_lift = not (bool(initial_lift_before_gate) and int(attempts) == 0)
        if bool(should_score_before_first_lift) and abs(y_mm - y_target) <= y_tol:
            return {
                "success": True,
                "target_met": True,
                "holding": True,
                "reason": "step4_targets_scored",
                "reading": current,
                "attempts": attempts,
                "send_result": last_send,
            }
        if bool(should_score_before_first_lift) and lift_cmd == "u" and y_mm < y_target - y_tol:
            return {
                "success": True,
                "target_met": False,
                "holding": True,
                "reason": "step4_already_past_y_target",
                "reading": current,
                "attempts": attempts,
                "send_result": last_send,
            }
        if bool(should_score_before_first_lift) and lift_cmd == "d" and y_mm > y_target + y_tol:
            return {
                "success": True,
                "target_met": False,
                "holding": True,
                "reason": "step4_already_past_y_target",
                "reading": current,
                "attempts": attempts,
                "send_result": last_send,
            }
        if attempts >= int(max_attempts):
            break
        last_send = guarded_send_command_pwm(
            robot,
            lift_cmd,
            lift_pwm,
            duration_ms=pulse_ms,
            reading=current,
            context="follow_step4_lift_to_y_target",
        )
        if isinstance(last_send, dict) and bool(last_send.get("blocked")):
            return {
                "success": False,
                "target_met": False,
                "holding": False,
                "reason": f"step4_lift_blocked:{last_send.get('reason')}",
                "reading": current,
                "attempts": attempts,
                "send_result": last_send,
            }
        attempts += 1
        time.sleep(float(pulse_ms) / 1000.0 + float(settle_s))
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        current = _read_brick_measurement(vision)
    return {
        "success": False,
        "target_met": False,
        "holding": False,
        "reason": "step4_y_target_not_reached",
        "reading": current,
        "attempts": attempts,
        "send_result": last_send,
    }


def _configured_step2_targets(step2_cfg: dict | None = None) -> dict:
    cfg = step2_cfg if isinstance(step2_cfg, dict) else _follow_step2_config()
    targets = cfg.get("targets") if isinstance(cfg.get("targets"), dict) else {}
    return targets if isinstance(targets, dict) else {}


STEP2_AXIS_SPECS = (
    ("dist", "dist_mm", "dist_tol_mm"),
    ("x", "x_mm", "x_tol_mm"),
    ("y", "y_mm", "y_tol_mm"),
)


def _configured_step2_target_axes(step2_cfg: dict | None = None) -> list[tuple[str, str, str]]:
    targets = _configured_step2_targets(step2_cfg)
    axes = []
    for label, value_key, tol_key in STEP2_AXIS_SPECS:
        if targets.get(value_key) is not None and targets.get(tol_key) is not None:
            axes.append((label, value_key, tol_key))
    return axes


def _step2_missing_target_keys(step2_cfg: dict | None = None) -> list[str]:
    targets = _configured_step2_targets(step2_cfg)
    missing = []
    configured_axes = 0
    for _label, value_key, tol_key in STEP2_AXIS_SPECS:
        has_value = targets.get(value_key) is not None
        has_tol = targets.get(tol_key) is not None
        if has_value and has_tol:
            configured_axes += 1
            continue
        if has_value and not has_tol:
            missing.append(tol_key)
        elif has_tol and not has_value:
            missing.append(value_key)
    if configured_axes <= 0 and not missing:
        missing.append("targets")
    return missing


def _step2_target_closeness_from_reading(reading: dict, step2_cfg: dict | None = None) -> dict | None:
    if not isinstance(reading, dict):
        return None
    targets = _configured_step2_targets(step2_cfg)
    closeness = {}
    values = []
    for label, value_key, tol_key in STEP2_AXIS_SPECS:
        target = targets.get(value_key)
        tol = targets.get(tol_key)
        if target is None or tol is None:
            closeness[f"{label}_target_closeness_pct"] = None
            continue
        try:
            value = float(reading.get(value_key))
        except (TypeError, ValueError):
            return None
        close = _target_closeness_pct(value - float(target), float(tol))
        closeness[f"{label}_target_closeness_pct"] = float(close)
        values.append(float(close))
    closeness["target_closeness_pct"] = None if not values else float(sum(values) / float(len(values)))
    return closeness


def _step2_targets_ready(reading: dict, step2_cfg: dict | None = None) -> tuple[bool, str, dict | None]:
    missing = _step2_missing_target_keys(step2_cfg)
    if missing:
        return False, "step2_targets_pending:" + ",".join(missing), None
    closeness = _step2_target_closeness_from_reading(reading, step2_cfg)
    if not isinstance(closeness, dict):
        return False, "invalid_step2_reading", None
    if _pickup_suspected_reading(reading):
        return False, "pickup_suspected_far_low", closeness
    targets = _configured_step2_targets(step2_cfg)
    if bool((reading or {}).get("xz_frozen")) and targets.get("dist_mm") is not None and targets.get("dist_tol_mm") is not None:
        try:
            raw_dist = float((reading or {}).get("raw_dist_mm"))
            dist_target = float(targets.get("dist_mm"))
            dist_tol = float(targets.get("dist_tol_mm"))
        except (TypeError, ValueError):
            raw_dist = None
        if raw_dist is not None and float(raw_dist) < float(dist_target) - float(dist_tol):
            return False, "step2_raw_dist_too_close", closeness
    axis_ok = []
    for _label, value_key, tol_key in _configured_step2_target_axes(step2_cfg):
        try:
            value = float((reading or {}).get(value_key))
            target = float(targets.get(value_key))
            tol = float(targets.get(tol_key))
        except (TypeError, ValueError):
            return False, "invalid_step2_reading", closeness
        axis_ok.append(abs(value - target) <= tol)
    return bool(axis_ok and all(axis_ok)), "step2_targets_scored", closeness


def _step2_target_hit_settle_s(step2_cfg: dict | None = None) -> float:
    settle_s = _coerce_float(
        (step2_cfg or {}).get("precision_settle_s"),
        DEFAULT_STEP2_CONFIG["precision_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    targets = _configured_step2_targets(step2_cfg)
    if targets.get("y_mm") is not None and targets.get("y_tol_mm") is not None:
        settle_s = max(float(settle_s), float(_y_motion_coast_settle_s()))
    return float(settle_s)


def _step2_stop_and_confirm_target_hit(
    vision: BrickDetector,
    robot: Robot,
    reading: dict,
    step2_cfg: dict,
    *,
    label: str = "step2",
    xz_lock_reading: dict | None = None,
) -> tuple[dict, bool, str, dict | None]:
    _stop_robot(robot)
    settle_s = _step2_target_hit_settle_s(step2_cfg)
    if settle_s > 0.0:
        time.sleep(float(settle_s))
    _reset_follow_reading_history(vision)
    settled = _read_brick_measurement(vision)
    if xz_lock_reading is not None:
        settled = _step2_freeze_xz_reading(settled, xz_lock_reading)
    target_met, target_reason, closeness = _step2_targets_ready(settled, step2_cfg)
    if bool(target_met):
        return settled, True, f"{label}_target_hit_confirmed_after_stop", closeness
    if not bool(settled.get("confident")):
        return settled, False, f"{label}_target_hit_unconfirmed_after_stop", None
    return settled, False, f"{label}_target_drifted_after_stop:{target_reason}", closeness


def _step2_xz_missing_target_keys(step2_cfg: dict | None = None) -> list[str]:
    targets = _configured_step2_targets(step2_cfg)
    missing = []
    for _label, value_key, tol_key in STEP2_AXIS_SPECS[:2]:
        has_value = targets.get(value_key) is not None
        has_tol = targets.get(tol_key) is not None
        if has_value and not has_tol:
            missing.append(tol_key)
        elif has_tol and not has_value:
            missing.append(value_key)
    return missing


def _step2_xz_targets_ready(reading: dict, step2_cfg: dict | None = None) -> tuple[bool, str]:
    missing = _step2_xz_missing_target_keys(step2_cfg)
    if missing:
        return False, "step2_xz_targets_pending:" + ",".join(missing)
    targets = _configured_step2_targets(step2_cfg)
    if not all(targets.get(key) is not None for key in ("dist_mm", "dist_tol_mm", "x_mm", "x_tol_mm")):
        return False, "step2_xz_targets_not_configured"
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return False, "invalid_step2_xz_reading"
    try:
        dist_ok = abs(float(reading.get("dist_mm")) - float(targets.get("dist_mm"))) <= float(targets.get("dist_tol_mm"))
        x_ok = abs(float(reading.get("x_mm")) - float(targets.get("x_mm"))) <= float(targets.get("x_tol_mm"))
    except (TypeError, ValueError):
        return False, "invalid_step2_xz_reading"
    return bool(dist_ok and x_ok), "step2_xz_targets_scored"


def _step2_xz_freeze_enabled(step2_cfg: dict | None = None) -> bool:
    cfg = step2_cfg if isinstance(step2_cfg, dict) else _follow_step2_config()
    return bool(cfg.get("freeze_xz_after_xz_target", DEFAULT_STEP2_CONFIG["freeze_xz_after_xz_target"]))


def _step2_freeze_xz_reading(reading: dict, xz_lock_reading: dict | None) -> dict:
    if not isinstance(reading, dict) or not isinstance(xz_lock_reading, dict):
        return reading
    out = dict(reading)
    for axis in ("dist", "x"):
        key = f"{axis}_mm"
        try:
            lock_value = float(xz_lock_reading.get(key))
        except (TypeError, ValueError):
            continue
        raw_key = f"raw_{key}"
        if raw_key not in out:
            try:
                out[raw_key] = float(out.get(key))
            except (TypeError, ValueError):
                pass
        out[key] = lock_value
        out[f"xz_lock_{key}"] = lock_value
    out["xz_frozen"] = True
    out["xz_freeze_reason"] = "step2_xz_targets_scored"
    return out


def _step2_result_freezes_xz(step2_result: dict | None) -> bool:
    if not isinstance(step2_result, dict):
        return False
    reading = step2_result.get("reading")
    if isinstance(reading, dict) and bool(reading.get("xz_frozen")):
        return True
    precision_counts = step2_result.get("precision_counts")
    if isinstance(precision_counts, dict):
        try:
            return int(precision_counts.get("xz_freeze", 0) or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def _step2_should_creep_forward(reading: dict, step2_cfg: dict | None = None) -> bool:
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return False
    if _pickup_suspected_reading(reading):
        return False
    if bool(reading.get("xz_frozen")):
        return False
    targets = _configured_step2_targets(step2_cfg)
    try:
        dist_mm = float(reading.get("dist_mm"))
        dist_target = float(targets.get("dist_mm"))
        dist_tol = float(targets.get("dist_tol_mm"))
    except (TypeError, ValueError):
        return False
    return float(dist_mm) > float(dist_target) + float(dist_tol)


def _step2_precision_drive_duration_ms(dist_gap_mm: float, step2_cfg: dict) -> int:
    min_ms = _coerce_int(
        step2_cfg.get("precision_drive_min_pulse_ms"),
        DEFAULT_STEP2_CONFIG["precision_drive_min_pulse_ms"],
        minimum=1,
        maximum=1000,
    )
    max_ms = _coerce_int(
        step2_cfg.get("precision_drive_max_pulse_ms"),
        DEFAULT_STEP2_CONFIG["precision_drive_max_pulse_ms"],
        minimum=1,
        maximum=1000,
    )
    return _proportional_duration_ms(
        gap_mm=max(0.0, float(dist_gap_mm)),
        min_ms=min_ms,
        max_ms=max_ms,
        full_gap_mm=30.0,
    )


def _step2_precision_mast_cmd(y_err: float) -> str:
    # Positive y error means the observed stack is above the target; D is the
    # only allowed recovery direction from that high-mast/visibility-risk side.
    return "d" if float(y_err) > 0.0 else "u"


def _step2_precision_mast_pwm(cmd: str, y_gap_mm: float, step2_cfg: dict) -> int:
    base_pwm = _scaled_pwm_for_cmd(cmd, step2_cfg.get("seat_mast_pwm", DEFAULT_STEP2_CONFIG["seat_mast_pwm"]))
    small_gap_max = _coerce_float(
        step2_cfg.get("precision_mast_small_gap_max_mm"),
        DEFAULT_STEP2_CONFIG["precision_mast_small_gap_max_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    if small_gap_max <= 0.0 or not (0.0 < float(y_gap_mm) <= float(small_gap_max)):
        return int(base_pwm)
    scale = _coerce_float(
        step2_cfg.get("precision_mast_small_gap_pwm_scale"),
        DEFAULT_STEP2_CONFIG["precision_mast_small_gap_pwm_scale"],
        minimum=0.0,
        maximum=1.0,
    )
    return int(max(1, round(float(base_pwm) * float(scale))))


def _step2_precision_mast_duration_ms(y_gap_mm: float, step2_cfg: dict) -> int:
    base_ms = _coerce_int(
        step2_cfg.get("precision_mast_pulse_ms"),
        DEFAULT_STEP2_CONFIG["precision_mast_pulse_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    small_gap_max = _coerce_float(
        step2_cfg.get("precision_mast_small_gap_max_mm"),
        DEFAULT_STEP2_CONFIG["precision_mast_small_gap_max_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    if small_gap_max <= 0.0 or not (0.0 < float(y_gap_mm) <= float(small_gap_max)):
        return int(base_ms)
    min_ms = _coerce_int(
        step2_cfg.get("precision_mast_small_gap_min_pulse_ms"),
        DEFAULT_STEP2_CONFIG["precision_mast_small_gap_min_pulse_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    max_ms = _coerce_int(
        step2_cfg.get("precision_mast_small_gap_max_pulse_ms"),
        DEFAULT_STEP2_CONFIG["precision_mast_small_gap_max_pulse_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    if min_ms > max_ms:
        min_ms, max_ms = max_ms, min_ms
    return min(
        _max_mast_act_ms(),
        _proportional_duration_ms(
        gap_mm=max(0.0, float(y_gap_mm)),
        min_ms=int(min_ms),
        max_ms=int(max_ms),
        full_gap_mm=float(small_gap_max),
        ),
    )


def _step2_precision_dist_cmd(dist_err: float, step2_cfg: dict) -> str:
    positive_cmd = str(
        step2_cfg.get("precision_dist_positive_cmd", DEFAULT_STEP2_CONFIG["precision_dist_positive_cmd"])
    ).strip().lower()
    if positive_cmd not in {"f", "b"}:
        positive_cmd = DEFAULT_STEP2_CONFIG["precision_dist_positive_cmd"]
    if float(dist_err) > 0.0:
        return positive_cmd
    return "b" if positive_cmd == "f" else "f"


def _step2_precision_settle_to_targets(
    vision: BrickDetector,
    robot: Robot,
    reading: dict,
    step2_cfg: dict,
    *,
    deadline: float | None = None,
) -> tuple[dict, dict]:
    current = reading if isinstance(reading, dict) else {}
    counts = {
        "fwd": 0,
        "bck": 0,
        "turn_l": 0,
        "turn_r": 0,
        "mast_u": 0,
        "mast_d": 0,
        "blocked": 0,
        "gap_closure_samples": [],
    }
    if not bool(step2_cfg.get("precision_settle_enabled", True)):
        return current, counts
    missing_targets = _step2_missing_target_keys(step2_cfg)
    if missing_targets:
        return current, counts
    max_no_progress_attempts = _coerce_int(
        step2_cfg.get("precision_max_attempts"),
        DEFAULT_STEP2_CONFIG["precision_max_attempts"],
        minimum=1,
        maximum=40,
    )
    hard_max_attempts = _coerce_int(
        step2_cfg.get("precision_hard_max_attempts"),
        max(40, int(max_no_progress_attempts) * 20),
        minimum=int(max_no_progress_attempts),
        maximum=400,
    )
    settle_s = _coerce_float(
        step2_cfg.get("precision_settle_s"),
        DEFAULT_STEP2_CONFIG["precision_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    targets = _configured_step2_targets(step2_cfg)
    dist_axis_configured = targets.get("dist_mm") is not None and targets.get("dist_tol_mm") is not None
    x_axis_configured = targets.get("x_mm") is not None and targets.get("x_tol_mm") is not None
    y_axis_configured = targets.get("y_mm") is not None and targets.get("y_tol_mm") is not None
    prev_dist_err = None
    xz_freeze_enabled = _step2_xz_freeze_enabled(step2_cfg)
    xz_lock_reading: dict | None = None
    no_progress_attempts = 0
    total_attempts = 0
    progress_min_mm = max(
        0.25,
        float(_act_stall_guard_config().get("min_axis_delta_mm", DEFAULT_ACT_STALL_GUARD_CONFIG["min_axis_delta_mm"])) * 0.5,
    )
    while (
        no_progress_attempts < int(max_no_progress_attempts)
        and total_attempts < int(hard_max_attempts)
        and not _deadline_expired(deadline)
    ):
        if not bool(current.get("confident")):
            current = _wait_for_visibility_recovery(
                vision,
                robot,
                current,
                context="step2_precision_settle",
            )
        if not bool(current.get("confident")):
            if xz_lock_reading is not None:
                current = _step2_freeze_xz_reading(current, xz_lock_reading)
            break
        if _pickup_suspected_reading(current):
            _stop_robot(robot)
            counts["pickup_suspected_stop"] = int(counts.get("pickup_suspected_stop", 0)) + 1
            counts["target_hit_reason"] = "pickup_suspected_far_low"
            break
        if bool(xz_freeze_enabled):
            if xz_lock_reading is None:
                xz_ready, _xz_reason = _step2_xz_targets_ready(current, step2_cfg)
                if bool(xz_ready):
                    xz_lock_reading = dict(current)
                    counts["xz_freeze"] = int(counts.get("xz_freeze", 0)) + 1
                    try:
                        print(
                            "[STEP2] x/z locked; freezing wheel motion "
                            f"at dist={float(current.get('dist_mm')):.1f}mm "
                            f"x={float(current.get('x_mm')):+.1f}mm.",
                            flush=True,
                        )
                    except (TypeError, ValueError):
                        print("[STEP2] x/z locked; freezing wheel motion.", flush=True)
            if xz_lock_reading is not None:
                current = _step2_freeze_xz_reading(current, xz_lock_reading)
        target_met, _target_reason, _closeness = _step2_targets_ready(current, step2_cfg)
        if bool(target_met):
            current, confirmed, confirm_reason, _confirm_closeness = _step2_stop_and_confirm_target_hit(
                vision,
                robot,
                current,
                step2_cfg,
                label="step2_precision",
                xz_lock_reading=xz_lock_reading,
            )
            counts["target_hit_stop"] = int(counts.get("target_hit_stop", 0)) + 1
            if bool(confirmed):
                counts["target_hit_confirmed"] = int(counts.get("target_hit_confirmed", 0)) + 1
            else:
                counts["target_hit_drifted_after_stop"] = int(
                    counts.get("target_hit_drifted_after_stop", 0)
                ) + 1
                counts["target_hit_confirm_failed"] = int(counts.get("target_hit_confirm_failed", 0)) + 1
            counts["target_hit_reason"] = str(confirm_reason)
            break
        dist_err = x_err = y_err = 0.0
        dist_gap = x_gap = y_gap = 0.0
        if dist_axis_configured:
            try:
                dist_err = float(current.get("dist_mm")) - float(targets.get("dist_mm"))
                dist_tol = float(targets.get("dist_tol_mm"))
            except (TypeError, ValueError):
                break
            dist_gap = max(0.0, abs(float(dist_err)) - float(dist_tol))
        if x_axis_configured:
            try:
                x_err = float(current.get("x_mm")) - float(targets.get("x_mm"))
                x_tol = float(targets.get("x_tol_mm"))
            except (TypeError, ValueError):
                break
            x_gap = max(0.0, abs(float(x_err)) - float(x_tol))
        if y_axis_configured:
            try:
                y_err = float(current.get("y_mm")) - float(targets.get("y_mm"))
                y_tol = float(targets.get("y_tol_mm"))
            except (TypeError, ValueError):
                break
            y_gap = max(0.0, abs(float(y_err)) - float(y_tol))
            if y_gap > 0.0 and float(y_err) < 0.0 and not bool(step2_cfg.get("precision_mast_up_enabled", False)):
                counts["mast_up_locked"] = int(counts.get("mast_up_locked", 0)) + 1
                y_gap = 0.0
        if dist_gap <= 0.0 and x_gap <= 0.0 and y_gap <= 0.0:
            break
        x_turn_plan = None
        attached_mast_cmd = None
        attached_mast_pwm = None
        attached_mast_duration_ms = None
        if y_axis_configured and y_gap > 0.0:
            cmd = _step2_precision_mast_cmd(y_err)
            pwm = _step2_precision_mast_pwm(cmd, y_gap, step2_cfg)
            duration_ms = _step2_precision_mast_duration_ms(y_gap, step2_cfg)
            if cmd == "u":
                blocked, cap_ms, y_now, ceiling = _mast_up_ceiling_status(
                    current,
                    target_mm=float(targets.get("y_mm")),
                )
                if blocked:
                    counts["mast_up_ceiling_blocked"] = int(counts.get("mast_up_ceiling_blocked", 0)) + 1
                    try:
                        print(
                            "[STEP2] Mast-up blocked by y ceiling: "
                            f"y={float(y_now):+.1f}mm ceiling={float(ceiling):+.1f}mm.",
                            flush=True,
                        )
                    except (TypeError, ValueError):
                        print("[STEP2] Mast-up blocked by y ceiling.", flush=True)
                    break
                if cap_ms is not None and int(duration_ms) > int(cap_ms):
                    if int(cap_ms) <= 0:
                        counts["mast_up_ceiling_blocked"] = int(counts.get("mast_up_ceiling_blocked", 0)) + 1
                        break
                    counts["mast_up_ceiling_capped"] = int(counts.get("mast_up_ceiling_capped", 0)) + 1
                    duration_ms = int(cap_ms)
            action_key = "mast_d" if cmd == "d" else "mast_u"
            display_action = "STEP2_PRECISION_MAST_D" if cmd == "d" else "STEP2_PRECISION_MAST_U"
            gap_before = float(y_gap)
            progress_axis = "y"
            before_err_for_sample = float(y_err)
            tol_for_sample = float(targets.get("y_tol_mm"))
        elif dist_axis_configured and dist_gap > 0.0 and dist_gap >= float(x_gap):
            cmd = _step2_precision_dist_cmd(dist_err, step2_cfg)
            pwm = _clamp_to_approved_straight_drive_pwm(
                cmd,
                step2_cfg.get("seat_drive_pwm", DEFAULT_STEP2_CONFIG["seat_drive_pwm"]),
            )
            duration_ms = _step2_precision_drive_duration_ms(dist_gap, step2_cfg)
            if prev_dist_err is not None and (float(prev_dist_err) * float(dist_err)) < 0.0:
                duration_ms = max(40, int(duration_ms * 0.5))
            prev_dist_err = float(dist_err)
            if y_axis_configured and cmd == "f" and float(y_err) > 0.0 and float(y_gap) > 0.0:
                y_cmd = _step2_precision_mast_cmd(y_err)
                if y_cmd == "d":
                    attached_mast_cmd = y_cmd
                    attached_mast_pwm = _step2_precision_mast_pwm(y_cmd, y_gap, step2_cfg)
                    attached_mast_duration_ms = min(
                        int(duration_ms),
                        _step2_precision_mast_duration_ms(y_gap, step2_cfg),
                    )
            action_key = "fwd" if cmd == "f" else "bck"
            display_action = "STEP2_PRECISION_FWD" if cmd == "f" else "STEP2_PRECISION_BCK"
            if attached_mast_cmd:
                display_action = f"{display_action}_MAST_{str(attached_mast_cmd).upper()}"
            gap_before = float(dist_gap)
            progress_axis = "dist"
            before_err_for_sample = float(dist_err)
            tol_for_sample = float(targets.get("dist_tol_mm"))
        elif x_axis_configured and x_gap > 0.0:
            cmd = _turn_cmd_to_close_x_gap(float(x_err))
            if cmd not in {"l", "r"}:
                break
            x_turn_plan = _x_only_turn_plan(
                reading=current,
                turn_cmd=cmd,
                drive_mode=_x_only_turn_drive_mode_for_dist(float(dist_err)),
                strength="micro",
                dist_err=float(dist_err),
                x_err=float(x_err),
                x_outside=float(x_gap),
                dist_outside=float(dist_gap),
                y_plan=None,
                reason="step2_precision_x_polish",
                use_production_curve=True,
            )
            pwm = 0
            duration_ms = int(x_turn_plan.get("duration_ms", PULSE_MS))
            action_key = "turn_l" if cmd == "l" else "turn_r"
            display_action = "STEP2_PRECISION_TURN_L" if cmd == "l" else "STEP2_PRECISION_TURN_R"
            gap_before = float(x_gap)
            progress_axis = "x"
            before_err_for_sample = float(x_err)
            tol_for_sample = float(targets.get("x_tol_mm"))
        else:
            break
        if x_turn_plan is not None:
            send_result = _execute_follow_action(robot, x_turn_plan, current)
        elif attached_mast_cmd:
            send_result = _drive(
                robot,
                cmd,
                current,
                mast_cmd=attached_mast_cmd,
                mast_pwm=attached_mast_pwm,
                mast_duration_ms=attached_mast_duration_ms,
                pwm=pwm,
                duration_ms=duration_ms,
            )
        else:
            send_result = guarded_send_command_pwm(
                robot,
                cmd,
                pwm,
                duration_ms=duration_ms,
                reading=current,
                context="follow_step2_precision_settle",
            )
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            counts["blocked"] = int(counts.get("blocked", 0)) + 1
            break
        counts[action_key] = int(counts.get(action_key, 0)) + 1
        if attached_mast_cmd:
            attached_key = "mast_d" if str(attached_mast_cmd).lower() == "d" else "mast_u"
            counts[attached_key] = int(counts.get(attached_key, 0)) + 1
        total_attempts += 1
        time.sleep((float(duration_ms) / 1000.0) + float(settle_s))
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        current = _read_brick_measurement(vision)
        if _pickup_suspected_reading(current):
            _stop_robot(robot)
            counts["pickup_suspected_stop"] = int(counts.get("pickup_suspected_stop", 0)) + 1
            counts["target_hit_reason"] = "pickup_suspected_far_low"
            break
        gap_after = None
        if bool(current.get("confident")):
            try:
                if progress_axis == "dist":
                    after_err_for_sample = float(current.get("dist_mm")) - float(targets.get("dist_mm"))
                    gap_after = max(
                        0.0,
                        abs(float(after_err_for_sample))
                        - float(tol_for_sample),
                    )
                elif progress_axis == "y":
                    after_err_for_sample = float(current.get("y_mm")) - float(targets.get("y_mm"))
                    gap_after = max(
                        0.0,
                        abs(float(after_err_for_sample))
                        - float(tol_for_sample),
                )
                elif progress_axis == "x":
                    after_err_for_sample = float(current.get("x_mm")) - float(targets.get("x_mm"))
                    gap_after = max(
                        0.0,
                        abs(float(after_err_for_sample))
                        - float(tol_for_sample),
                    )
                before_abs = abs(float(before_err_for_sample))
                after_abs = abs(float(after_err_for_sample))
                regression = float(after_abs - before_abs)
                overshot = bool(
                    before_err_for_sample
                    and after_err_for_sample
                    and (float(before_err_for_sample) > 0.0) != (float(after_err_for_sample) > 0.0)
                )
                sample = {
                    "axis": str(progress_axis),
                    "action": str(display_action),
                    "before_err": float(before_err_for_sample),
                    "after_err": float(after_err_for_sample),
                    "before_abs": float(before_abs),
                    "after_abs": float(after_abs),
                    "reduction": float(before_abs - after_abs),
                    "regressed": bool(regression > GAP_REGRESSION_EPSILON_MM),
                    "regression_mm": float(max(0.0, regression)),
                    "overshot": bool(overshot),
                    "overshot_within_tolerance": bool(overshot and after_abs <= float(tol_for_sample)),
                    "tol": float(tol_for_sample),
                    "duration_ms": int(duration_ms),
                }
                counts.setdefault("gap_closure_samples", []).append(sample)
            except (TypeError, ValueError):
                gap_after = None
        if gap_after is not None and float(gap_after) <= max(0.0, float(gap_before) - float(progress_min_mm)):
            no_progress_attempts = 0
            counts["progress_reset"] = int(counts.get("progress_reset", 0)) + 1
        else:
            no_progress_attempts += 1
            counts["no_progress"] = int(counts.get("no_progress", 0)) + 1
        target_met_after, _target_reason_after, _closeness_after = _step2_targets_ready(current, step2_cfg)
        if bool(target_met_after):
            current, confirmed, confirm_reason, _confirm_closeness = _step2_stop_and_confirm_target_hit(
                vision,
                robot,
                current,
                step2_cfg,
                label="step2_precision",
                xz_lock_reading=xz_lock_reading,
            )
            counts["target_hit_stop"] = int(counts.get("target_hit_stop", 0)) + 1
            if bool(confirmed):
                counts["target_hit_confirmed"] = int(counts.get("target_hit_confirmed", 0)) + 1
            else:
                counts["target_hit_drifted_after_stop"] = int(
                    counts.get("target_hit_drifted_after_stop", 0)
                ) + 1
                counts["target_hit_confirm_failed"] = int(counts.get("target_hit_confirm_failed", 0)) + 1
            counts["target_hit_reason"] = str(confirm_reason)
            break
        if dist_axis_configured and bool(current.get("confident")):
            try:
                safety_dist_err = float(current.get("dist_mm")) - float(targets.get("dist_mm"))
                safety_dist_tol = float(targets.get("dist_tol_mm"))
            except (TypeError, ValueError):
                safety_dist_err = None
                safety_dist_tol = 0.0
            if safety_dist_err is not None and float(safety_dist_err) < -float(safety_dist_tol):
                counts["dist_too_close_safety_stop"] = int(counts.get("dist_too_close_safety_stop", 0)) + 1
                break
        if progress_axis == "dist" and bool(current.get("confident")):
            try:
                after_dist_err = float(current.get("dist_mm")) - float(targets.get("dist_mm"))
                dist_tol = float(targets.get("dist_tol_mm"))
            except (TypeError, ValueError):
                after_dist_err = None
                dist_tol = 0.0
            if after_dist_err is not None and abs(float(after_dist_err)) <= float(dist_tol):
                counts["target_hit_stop"] = int(counts.get("target_hit_stop", 0)) + 1
                break
            if after_dist_err is not None and (float(dist_err) * float(after_dist_err)) < 0.0:
                counts["dist_crossed_target_stop"] = int(counts.get("dist_crossed_target_stop", 0)) + 1
                break
            if after_dist_err is not None and abs(float(after_dist_err)) > abs(float(dist_err)) + float(progress_min_mm):
                counts["dist_wrong_way_stop"] = int(counts.get("dist_wrong_way_stop", 0)) + 1
                break
        if progress_axis == "x" and bool(current.get("confident")):
            try:
                after_x_err = float(current.get("x_mm")) - float(targets.get("x_mm"))
            except (TypeError, ValueError):
                after_x_err = None
            if after_x_err is not None and abs(float(after_x_err)) > abs(float(x_err)) + float(progress_min_mm):
                counts["x_wrong_way_stop"] = int(counts.get("x_wrong_way_stop", 0)) + 1
                break
    counts["precision_total_attempts"] = int(total_attempts)
    counts["precision_no_progress_attempts"] = int(no_progress_attempts)
    if _deadline_expired(deadline):
        counts["step_timeout"] = int(counts.get("step_timeout", 0)) + 1
    if xz_lock_reading is not None:
        current = _step2_freeze_xz_reading(current, xz_lock_reading)
    return current, counts


def _merge_precision_counts(base: dict, update: dict | None) -> None:
    if not isinstance(base, dict) or not isinstance(update, dict):
        return
    for key, value in update.items():
        if key == "gap_closure_samples":
            if isinstance(value, list):
                base.setdefault("gap_closure_samples", []).extend(
                    dict(sample) for sample in value if isinstance(sample, dict)
                )
            continue
        try:
            base[key] = int(base.get(key, 0) or 0) + int(value or 0)
        except (TypeError, ValueError):
            continue


def _step2_creep_forward_if_short(vision: BrickDetector, robot: Robot, reading: dict, step2_cfg: dict) -> tuple[dict, int]:
    current = reading if isinstance(reading, dict) else {}
    if not bool(step2_cfg.get("recovery_creep_enabled", True)):
        return current, 0
    drive_cmd = str(step2_cfg.get("seat_drive_cmd") or DEFAULT_STEP2_CONFIG["seat_drive_cmd"]).strip().lower()
    if drive_cmd != "f":
        return current, 0
    drive_pwm = _clamp_to_approved_straight_drive_pwm(
        "f",
        step2_cfg.get("seat_drive_pwm", DEFAULT_STEP2_CONFIG["seat_drive_pwm"]),
    )
    pulse_ms = _coerce_int(
        step2_cfg.get("recovery_creep_pulse_ms"),
        DEFAULT_STEP2_CONFIG["recovery_creep_pulse_ms"],
        minimum=1,
        maximum=400,
    )
    max_attempts = _coerce_int(
        step2_cfg.get("recovery_creep_max_attempts"),
        DEFAULT_STEP2_CONFIG["recovery_creep_max_attempts"],
        minimum=0,
        maximum=20,
    )
    settle_s = _coerce_float(
        step2_cfg.get("recovery_creep_settle_s"),
        DEFAULT_STEP2_CONFIG["recovery_creep_settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    attempts = 0
    while attempts < int(max_attempts) and _step2_should_creep_forward(current, step2_cfg):
        send_result = guarded_send_command_pwm(
            robot,
            "f",
            drive_pwm,
            duration_ms=pulse_ms,
            reading=current,
            context="follow_step2_recovery_creep_forward",
        )
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            break
        attempts += 1
        time.sleep((float(pulse_ms) / 1000.0) + float(settle_s))
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        current = _read_brick_measurement(vision)
    return current, int(attempts)


def _step2_recover_visibility_with_forward_creeps(
    vision: BrickDetector,
    robot: Robot,
    reading: dict,
    step2_cfg: dict,
) -> tuple[dict, int]:
    current = reading if isinstance(reading, dict) else {}
    if bool(current.get("confident")):
        return current, 0
    if bool(current.get("xz_frozen")):
        return current, 0
    wait_s = _coerce_float(
        step2_cfg.get("visibility_recovery_wait_s"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_wait_s"],
        minimum=0.0,
        maximum=10.0,
    )
    poll_s = _coerce_float(
        step2_cfg.get("visibility_recovery_wait_poll_s"),
        DEFAULT_STEP2_CONFIG["visibility_recovery_wait_poll_s"],
        minimum=0.01,
        maximum=2.0,
    )
    if wait_s > 0.0:
        current = _wait_for_visibility_recovery(
            vision,
            robot,
            current,
            timeout_s=wait_s,
            sample_s=poll_s,
            context="step2_visibility",
        )
    return current, 0


def _run_step2_seat_sequence(
    vision: BrickDetector,
    robot: Robot,
    *,
    probe_before_forward: bool = False,
    step_cfg: dict | None = None,
    step_key: str = "step2",
) -> dict:
    step2 = step_cfg if isinstance(step_cfg, dict) else _follow_step2_config()
    label = str(step_key or "step2").strip().lower()
    step_timeout_s = _coerce_float(
        step2.get("step_timeout_s"),
        DEFAULT_STEP2_CONFIG["step_timeout_s"],
        minimum=0.0,
        maximum=120.0,
    )
    deadline = _deadline_from_timeout_s(step_timeout_s)
    before = _read_brick_measurement(vision)
    if not bool(before.get("confident")):
        before = _wait_for_visibility_recovery(
            vision,
            robot,
            before,
            context=f"{label}_start",
        )
    if not bool(before.get("confident")):
        return {
            "success": False,
            "target_met": False,
            "reason": f"brick_not_confident_before_{label}",
            "before": before,
            "reading": before,
        }
    if _pickup_suspected_reading(before):
        _stop_robot(robot)
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_pickup_suspected_far_low",
            "send_result": {"skipped": True, "reason": "pickup_suspected_far_low"},
            "mast_result": {"skipped": True, "reason": "pickup_suspected_far_low"},
            "before": before,
            "reading": before,
            "duration_ms": 0,
            "mast_duration_ms": 0,
            "drive_duration_ms": 0,
            "creep_attempts": 0,
            "visibility_recovery_creeps": 0,
            "precision_counts": {"pickup_suspected_stop": 1},
            "closeness": _step2_target_closeness_from_reading(before, step2),
        }
    if _deadline_expired(deadline):
        _stop_robot(robot)
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_step_timeout",
            "before": before,
            "reading": before,
            "step_timeout_s": float(step_timeout_s),
        }
    already_target_met, already_reason, already_closeness = _step2_targets_ready(before, step2)
    if bool(already_target_met):
        settled, confirmed, confirm_reason, confirmed_closeness = _step2_stop_and_confirm_target_hit(
            vision,
            robot,
            before,
            step2,
            label=label,
        )
        return {
            "success": True,
            "target_met": bool(confirmed),
            "reason": already_reason if bool(confirmed) else confirm_reason,
            "send_result": {"skipped": True, "reason": f"{label}_already_at_target"},
            "mast_result": {"skipped": True, "reason": f"{label}_already_at_target"},
            "before": before,
            "after_mast": settled,
            "reading": settled,
            "duration_ms": 0,
            "mast_duration_ms": 0,
            "drive_duration_ms": 0,
            "creep_attempts": 0,
            "visibility_recovery_creeps": 0,
            "precision_counts": {
                "target_hit_stop": 1,
                "target_hit_confirmed": 1 if bool(confirmed) else 0,
                "target_hit_drifted_after_stop": 0 if bool(confirmed) else 1,
            },
            "closeness": confirmed_closeness if confirmed_closeness is not None else already_closeness,
        }
    initial_xz_lock_reading: dict | None = None
    if _step2_xz_freeze_enabled(step2):
        xz_ready, _xz_reason = _step2_xz_targets_ready(before, step2)
        if bool(xz_ready):
            initial_xz_lock_reading = dict(before)
    mast_cmd = str(step2.get("seat_mast_cmd") or DEFAULT_STEP2_CONFIG["seat_mast_cmd"]).strip().lower()
    mast_duration_ms = _cap_mast_duration_ms(
        mast_cmd,
        step2.get("seat_mast_duration_ms"),
        DEFAULT_STEP2_CONFIG["seat_mast_duration_ms"],
        minimum=0,
        maximum=5000 if bool(step2.get("blind_mast_only", False)) else None,
    )
    mast_pwm = _scaled_pwm_for_cmd(mast_cmd, step2.get("seat_mast_pwm"))
    mast_result = {"skipped": True, "reason": f"{label}_mast_duration_zero"}
    if mast_duration_ms > 0:
        mast_result = guarded_send_command_pwm(
            robot,
            mast_cmd,
            mast_pwm,
            duration_ms=mast_duration_ms,
            reading=before,
            context=f"follow_{label}_seat_mast",
        )
    if isinstance(mast_result, dict) and bool(mast_result.get("blocked")):
        return {
            "success": False,
            "target_met": False,
            "reason": f"{label}_mast_blocked:{mast_result.get('reason')}",
            "send_result": mast_result,
            "before": before,
            "reading": before,
            "duration_ms": int(mast_duration_ms),
        }
    if mast_duration_ms > 0:
        time.sleep(float(mast_duration_ms) / 1000.0)
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        after_mast = _read_brick_measurement(vision)
    else:
        after_mast = before
    if bool(step2.get("blind_mast_only", False)):
        return {
            "success": True,
            "target_met": True,
            "reason": f"{label}_blind_mast_only",
            "send_result": {"skipped": True, "reason": f"{label}_blind_mast_only_no_drive"},
            "mast_result": mast_result,
            "before": before,
            "after_mast": after_mast,
            "reading": after_mast,
            "duration_ms": int(mast_duration_ms),
            "mast_duration_ms": int(mast_duration_ms),
            "drive_duration_ms": 0,
            "creep_attempts": 0,
            "visibility_recovery_creeps": 0,
            "precision_counts": {},
            "closeness": _step2_target_closeness_from_reading(after_mast, step2) if bool(after_mast.get("confident")) else None,
        }
    if _pickup_suspected_reading(after_mast):
        _stop_robot(robot)
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_pickup_suspected_far_low",
            "send_result": {"skipped": True, "reason": "pickup_suspected_far_low"},
            "mast_result": mast_result,
            "before": before,
            "after_mast": after_mast,
            "reading": after_mast,
            "duration_ms": int(mast_duration_ms),
            "mast_duration_ms": int(mast_duration_ms),
            "drive_duration_ms": 0,
            "creep_attempts": 0,
            "visibility_recovery_creeps": 0,
            "precision_counts": {"pickup_suspected_stop": 1},
            "closeness": _step2_target_closeness_from_reading(after_mast, step2),
        }
    if _deadline_expired(deadline):
        _stop_robot(robot)
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_step_timeout",
            "mast_result": mast_result,
            "before": before,
            "after_mast": after_mast,
            "reading": after_mast,
            "duration_ms": int(mast_duration_ms),
            "mast_duration_ms": int(mast_duration_ms),
            "drive_duration_ms": 0,
            "step_timeout_s": float(step_timeout_s),
        }
    if bool(probe_before_forward):
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_probe_before_forward",
            "mast_result": mast_result,
            "before": before,
            "after_mast": after_mast,
            "reading": after_mast,
            "duration_ms": int(mast_duration_ms),
            "mast_duration_ms": int(mast_duration_ms),
            "drive_duration_ms": 0,
            "creep_attempts": 0,
            "precision_counts": {},
            "closeness": _step2_target_closeness_from_reading(after_mast, step2) if bool(after_mast.get("confident")) else None,
            "probe": True,
        }
    if not bool(after_mast.get("confident")):
        if initial_xz_lock_reading is not None:
            print(f"[{label.upper()}] x/z locked before visibility loss; waiting without wheel motion.", flush=True)
            frozen_after_mast = _step2_freeze_xz_reading(after_mast, initial_xz_lock_reading)
            after = _wait_for_visibility_recovery(
                vision,
                robot,
                frozen_after_mast,
                timeout_s=step2.get("visibility_recovery_wait_s"),
                sample_s=step2.get("visibility_recovery_wait_poll_s"),
                context=f"{label}_xz_frozen_visibility",
            )
            after = _step2_freeze_xz_reading(after, initial_xz_lock_reading)
            visibility_recovery_creeps = 0
        else:
            after, visibility_recovery_creeps = _step2_recover_visibility_with_forward_creeps(
                vision,
                robot,
                after_mast,
                step2,
            )
        precision_counts = {"fwd": 0, "bck": 0, "mast_u": 0, "mast_d": 0, "blocked": 0}
        creep_attempts = 0
        if bool(after.get("confident")):
            if bool(step2.get("precision_settle_enabled", True)):
                after, precision_counts = _step2_precision_settle_to_targets(
                    vision,
                    robot,
                    after,
                    step2,
                    deadline=deadline,
                )
                creep_attempts = int((precision_counts or {}).get("fwd", 0))
            if bool((precision_counts or {}).get("pickup_suspected_stop")) or _pickup_suspected_reading(after):
                _stop_robot(robot)
                return {
                    "success": True,
                    "target_met": False,
                    "reason": f"{label}_pickup_suspected_far_low",
                    "send_result": {"skipped": True, "reason": f"no_blind_{label}_drive"},
                    "mast_result": mast_result,
                    "before": before,
                    "after_mast": after_mast,
                    "reading": after,
                    "duration_ms": int(mast_duration_ms),
                    "mast_duration_ms": int(mast_duration_ms),
                    "drive_duration_ms": 0,
                    "creep_attempts": int(creep_attempts),
                    "visibility_recovery_creeps": int(visibility_recovery_creeps),
                    "precision_counts": dict(precision_counts or {}),
                    "closeness": _step2_target_closeness_from_reading(after, step2),
                }
            target_met, target_reason, closeness = _step2_targets_ready(after, step2)
            if bool((precision_counts or {}).get("target_hit_confirm_failed")):
                target_reason = str((precision_counts or {}).get("target_hit_reason") or target_reason)
            elif bool((precision_counts or {}).get("step_timeout")) and not bool(target_met):
                target_reason = f"{label}_step_timeout"
            elif not bool(target_met):
                target_reason = f"{label}_visibility_recovered_targets_scored"
            return {
                "success": True,
                "target_met": bool(target_met),
                "reason": target_reason,
                "send_result": {"skipped": True, "reason": f"no_blind_{label}_drive"},
                "mast_result": mast_result,
                "before": before,
                "after_mast": after_mast,
                "reading": after,
                "duration_ms": int(mast_duration_ms),
                "mast_duration_ms": int(mast_duration_ms),
                "drive_duration_ms": 0,
                "creep_attempts": int(creep_attempts),
                "visibility_recovery_creeps": int(visibility_recovery_creeps),
                "precision_counts": dict(precision_counts or {}),
                "closeness": closeness,
            }
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_unconfirmed_no_final_visibility",
            "send_result": {"skipped": True, "reason": "visibility_recovery_failed"},
            "mast_result": mast_result,
            "before": before,
            "after_mast": after_mast,
            "reading": after,
            "duration_ms": int(mast_duration_ms),
            "mast_duration_ms": int(mast_duration_ms),
            "drive_duration_ms": 0,
            "creep_attempts": 0,
            "visibility_recovery_creeps": int(visibility_recovery_creeps),
            "precision_counts": dict(precision_counts or {}),
            "closeness": None,
        }
    if bool(step2.get("precision_settle_enabled", True)):
        after, precision_counts = _step2_precision_settle_to_targets(
            vision,
            robot,
            after_mast,
            step2,
            deadline=deadline,
        )
        creep_attempts = int((precision_counts or {}).get("fwd", 0))
    else:
        after, creep_attempts = _step2_creep_forward_if_short(vision, robot, after_mast, step2)
        precision_counts = {"fwd": int(creep_attempts), "bck": 0, "mast_u": 0, "mast_d": 0, "blocked": 0}
    if bool((precision_counts or {}).get("pickup_suspected_stop")) or _pickup_suspected_reading(after):
        _stop_robot(robot)
        return {
            "success": True,
            "target_met": False,
            "reason": f"{label}_pickup_suspected_far_low",
            "send_result": {"skipped": True, "reason": f"no_blind_{label}_drive"},
            "mast_result": mast_result,
            "before": before,
            "after_mast": after_mast,
            "reading": after,
            "duration_ms": int(mast_duration_ms),
            "mast_duration_ms": int(mast_duration_ms),
            "drive_duration_ms": 0,
            "creep_attempts": int(creep_attempts),
            "visibility_recovery_creeps": 0,
            "precision_counts": dict(precision_counts or {}),
            "closeness": _step2_target_closeness_from_reading(after, step2),
        }
    target_met = False
    target_reason = f"{label}_targets_scored"
    closeness = None
    visibility_recovery_creeps = 0
    recovery_cycles = 0
    max_recovery_cycles = _coerce_int(
        step2.get("post_precision_recovery_cycles"),
        DEFAULT_STEP2_CONFIG["post_precision_recovery_cycles"],
        minimum=0,
        maximum=3,
    )
    target_hit_confirm_failed = bool((precision_counts or {}).get("target_hit_confirm_failed"))
    if bool(target_hit_confirm_failed):
        target_reason = str((precision_counts or {}).get("target_hit_reason") or f"{label}_target_drifted_after_stop")
        closeness = _step2_target_closeness_from_reading(after, step2) if bool(after.get("confident")) else None
    while True:
        if bool(target_hit_confirm_failed):
            break
        if bool(after.get("confident")):
            if _pickup_suspected_reading(after):
                _stop_robot(robot)
                target_met = False
                target_reason = f"{label}_pickup_suspected_far_low"
                break
            target_met, target_reason, closeness = _step2_targets_ready(after, step2)
            if bool(target_met):
                precision_already_latched = bool(
                    (precision_counts or {}).get("target_hit_confirmed")
                    or (precision_counts or {}).get("target_hit_drifted_after_stop")
                )
                if not bool(precision_already_latched):
                    after, confirmed, confirm_reason, confirmed_closeness = _step2_stop_and_confirm_target_hit(
                        vision,
                        robot,
                        after,
                        step2,
                        label=label,
                    )
                    target_met = bool(confirmed)
                    target_reason = target_reason if bool(confirmed) else confirm_reason
                    closeness = confirmed_closeness
                    precision_counts["target_hit_stop"] = int(precision_counts.get("target_hit_stop", 0)) + 1
                    if bool(confirmed):
                        precision_counts["target_hit_confirmed"] = int(
                            precision_counts.get("target_hit_confirmed", 0)
                        ) + 1
                    else:
                        precision_counts["target_hit_drifted_after_stop"] = int(
                            precision_counts.get("target_hit_drifted_after_stop", 0)
                        ) + 1
                        precision_counts["target_hit_confirm_failed"] = int(
                            precision_counts.get("target_hit_confirm_failed", 0)
                        ) + 1
                        precision_counts["target_hit_reason"] = str(confirm_reason)
                break
        if _deadline_expired(deadline):
            target_reason = f"{label}_step_timeout"
            break
        if recovery_cycles >= int(max_recovery_cycles):
            if not bool(after.get("confident")):
                target_met, target_reason, closeness = False, f"{label}_unconfirmed_no_final_visibility", None
            elif not bool(target_met):
                target_reason = f"{label}_visibility_recovered_targets_scored"
            break
        after, recovered_creeps = _step2_recover_visibility_with_forward_creeps(
            vision,
            robot,
            after,
            step2,
        )
        visibility_recovery_creeps += int(recovered_creeps)
        if not bool(after.get("confident")):
            recovery_cycles += 1
            continue
        if bool(step2.get("precision_settle_enabled", True)):
            after, recovery_precision_counts = _step2_precision_settle_to_targets(
                vision,
                robot,
                after,
                step2,
                deadline=deadline,
            )
            _merge_precision_counts(precision_counts, recovery_precision_counts)
            creep_attempts = int((precision_counts or {}).get("fwd", 0))
        recovery_cycles += 1
    return {
        "success": True,
        "target_met": bool(target_met),
        "reason": target_reason,
        "send_result": {"skipped": True, "reason": f"no_blind_{label}_drive"},
        "mast_result": mast_result,
        "before": before,
        "after_mast": after_mast,
        "reading": after,
        "duration_ms": int(mast_duration_ms),
        "mast_duration_ms": int(mast_duration_ms),
        "drive_duration_ms": 0,
        "creep_attempts": int(creep_attempts),
        "visibility_recovery_creeps": int(visibility_recovery_creeps),
        "precision_counts": dict(precision_counts or {}),
        "closeness": closeness,
    }


def _run_step3_seat_sequence(vision: BrickDetector, robot: Robot) -> dict:
    return _run_step2_seat_sequence(
        vision,
        robot,
        step_cfg=_follow_step3_config(),
        step_key="step3",
    )


def _step3_retreat_target_dist_mm(step3_cfg: dict | None = None) -> float:
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step3_config()
    explicit = cfg.get("target_dist_mm") if isinstance(cfg, dict) else None
    if explicit is not None:
        return _coerce_float(explicit, TARGET_DIST_MM, minimum=0.0)
    profile = str(
        (cfg or {}).get("target_dist_profile")
        or DEFAULT_STEP3_RETREAT_CONFIG["target_dist_profile"]
    ).strip().lower()
    return float(_profile_step1_dist_target_mm(profile))


def _step3_retreat_target_ready(reading: dict | None, step3_cfg: dict | None = None) -> bool:
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return False
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step3_config()
    try:
        dist_mm = float(reading.get("dist_mm"))
    except (TypeError, ValueError):
        return False
    target = float(_step3_retreat_target_dist_mm(cfg))
    tol = _coerce_float(
        (cfg or {}).get("target_dist_tol_mm"),
        DEFAULT_STEP3_RETREAT_CONFIG["target_dist_tol_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    if bool((cfg or {}).get("stop_when_dist_at_or_beyond", True)):
        return dist_mm >= (target - tol)
    return abs(dist_mm - target) <= tol


def _run_step3_retreat_sequence(vision: BrickDetector, robot: Robot) -> dict:
    step3 = _follow_step3_config()
    target_dist = float(_step3_retreat_target_dist_mm(step3))
    max_duration_ms = _coerce_int(
        step3.get("max_duration_ms"),
        DEFAULT_STEP3_RETREAT_CONFIG["max_duration_ms"],
        minimum=1,
        maximum=30000,
    )
    chunk_ms = _coerce_int(
        step3.get("chunk_ms"),
        DEFAULT_STEP3_RETREAT_CONFIG["chunk_ms"],
        minimum=1,
        maximum=1000,
    )
    settle_s = _coerce_float(
        step3.get("settle_s"),
        DEFAULT_STEP3_RETREAT_CONFIG["settle_s"],
        minimum=0.0,
        maximum=2.0,
    )
    cmd = str(step3.get("drive_cmd") or DEFAULT_STEP3_RETREAT_CONFIG["drive_cmd"]).strip().lower()
    if cmd not in {"f", "b"}:
        cmd = DEFAULT_STEP3_RETREAT_CONFIG["drive_cmd"]
    pwm = _clamp_to_approved_straight_drive_pwm(
        cmd,
        step3.get("drive_pwm", DEFAULT_STEP3_RETREAT_CONFIG["drive_pwm"]),
    )
    min_duration_ms = int(_min_motion_duration_ms(cmd))
    chunk_ms = max(int(chunk_ms), int(min_duration_ms))

    before = _read_brick_measurement(vision)
    if not bool(before.get("confident")):
        before = _wait_for_visibility_recovery(
            vision,
            robot,
            before,
            context="step3_retreat_start",
        )
    if not bool(before.get("confident")):
        _stop_robot(robot)
        return {
            "success": False,
            "target_met": False,
            "reason": "brick_not_confident_before_step3_retreat",
            "before": before,
            "reading": before,
            "duration_ms": 0,
            "drive_duration_ms": 0,
            "attempts": 0,
        }

    print(
        f"[STEP3] Retreat: straight {cmd.upper()} until dist>={target_dist:.1f}mm "
        f"or {int(max_duration_ms)}ms max.",
        flush=True,
    )

    current = before
    elapsed_ms = 0
    attempts = 0
    send_result = None
    while elapsed_ms < int(max_duration_ms):
        if _step3_retreat_target_ready(current, step3):
            break
        remaining_ms = int(max_duration_ms) - int(elapsed_ms)
        if remaining_ms < min_duration_ms:
            break
        duration_ms = min(int(chunk_ms), int(remaining_ms))
        send_result = guarded_send_command_pwm(
            robot,
            cmd,
            pwm,
            duration_ms=duration_ms,
            reading=current,
            context="step3_retreat",
        )
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            _stop_robot(robot)
            return {
                "success": False,
                "target_met": False,
                "reason": f"step3_retreat_blocked:{send_result.get('reason')}",
                "before": before,
                "reading": current,
                "send_result": send_result,
                "duration_ms": int(elapsed_ms),
                "drive_duration_ms": int(elapsed_ms),
                "attempts": int(attempts),
            }
        attempts += 1
        elapsed_ms += int(duration_ms)
        time.sleep(max(float(LOOP_S), (float(duration_ms) / 1000.0) + float(settle_s)))
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        current = _read_brick_measurement(vision)
        if not bool(current.get("confident")):
            current = _wait_for_visibility_recovery(
                vision,
                robot,
                current,
                context="step3_retreat_after_drive",
            )
        if not bool(current.get("confident")):
            break

    target_met = _step3_retreat_target_ready(current, step3)
    reason = "step3_retreat_target_dist_seen" if bool(target_met) else "step3_retreat_time_budget_elapsed"
    if not bool(current.get("confident")):
        reason = "step3_retreat_lost_confident_brick"
    return {
        "success": bool(current.get("confident")),
        "target_met": bool(target_met),
        "reason": reason,
        "before": before,
        "reading": current,
        "send_result": send_result,
        "duration_ms": int(elapsed_ms),
        "drive_duration_ms": int(elapsed_ms),
        "attempts": int(attempts),
        "target_dist_mm": float(target_dist),
    }


def _run_step2_settle_sequence(vision: BrickDetector, robot: Robot) -> dict:
    step2 = _follow_step2_config()
    before = _read_brick_measurement(vision)
    if not bool(before.get("confident")):
        before, visibility_recovery_creeps = _step2_recover_visibility_with_forward_creeps(
            vision,
            robot,
            before,
            step2,
        )
    else:
        visibility_recovery_creeps = 0
    if not bool(before.get("confident")):
        return {
            "success": False,
            "target_met": False,
            "reason": "brick_not_confident_before_step2_settle",
            "before": before,
            "reading": before,
            "visibility_recovery_creeps": int(visibility_recovery_creeps),
            "precision_counts": {},
        }
    after, precision_counts = _step2_precision_settle_to_targets(vision, robot, before, step2)
    if bool(after.get("confident")):
        target_met, target_reason, closeness = _step2_targets_ready(after, step2)
    else:
        after, recovered_creeps = _step2_recover_visibility_with_forward_creeps(
            vision,
            robot,
            after,
            step2,
        )
        visibility_recovery_creeps += int(recovered_creeps)
        if bool(after.get("confident")):
            if bool(step2.get("precision_settle_enabled", True)):
                after, recovery_precision_counts = _step2_precision_settle_to_targets(vision, robot, after, step2)
                _merge_precision_counts(precision_counts, recovery_precision_counts)
            target_met, target_reason, closeness = _step2_targets_ready(after, step2)
            if not bool(target_met):
                target_reason = "step2_visibility_recovered_targets_scored"
        else:
            target_met, target_reason, closeness = False, "step2_unconfirmed_no_final_visibility", None
    return {
        "success": True,
        "target_met": bool(target_met),
        "reason": target_reason,
        "before": before,
        "reading": after,
        "duration_ms": 0,
        "mast_duration_ms": 0,
        "drive_duration_ms": 0,
        "creep_attempts": int((precision_counts or {}).get("fwd", 0)),
        "visibility_recovery_creeps": int(visibility_recovery_creeps),
        "precision_counts": dict(precision_counts or {}),
        "closeness": closeness,
    }


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
    duration_ms = _cap_mast_duration_ms(cmd, mast_duration_ms, y_cfg.get("mast_pulse_ms", PULSE_MS), minimum=1)
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
    duration_ms = _cap_mast_duration_ms("d", y_cfg.get("lock_on_pulse_ms"), PULSE_MS, minimum=1)
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


def _reset_straight_back_first_config(reset_cfg: dict | None = None) -> dict:
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    raw = cfg.get("straight_back_first") if isinstance(cfg.get("straight_back_first"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["enabled"])),
        "duration_ms": _coerce_int(
            raw.get("duration_ms"),
            DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["duration_ms"],
            minimum=1,
            maximum=5000,
        ),
        "duration_min_ms": _coerce_int(
            raw.get("duration_min_ms"),
            0,
            minimum=0,
            maximum=5000,
        ),
        "duration_max_ms": _coerce_int(
            raw.get("duration_max_ms"),
            0,
            minimum=0,
            maximum=5000,
        ),
        "pwm": _coerce_int(
            raw.get("pwm"),
            DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["pwm"],
            minimum=1,
            maximum=255,
        ),
        "mast_up_delay_fraction": _coerce_float(
            raw.get("mast_up_delay_fraction"),
            DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["mast_up_delay_fraction"],
            minimum=0.0,
            maximum=0.95,
        ),
    }


def _reset_x_goal_curve_config(reset_cfg: dict | None = None) -> dict:
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    raw = cfg.get("x_goal_curve") if isinstance(cfg.get("x_goal_curve"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["enabled"])),
        "target_fraction": _coerce_float(
            raw.get("target_fraction"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["target_fraction"],
            minimum=0.0,
            maximum=1.0,
        ),
        "max_duration_ms": _coerce_int(
            raw.get("max_duration_ms"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["max_duration_ms"],
            minimum=1,
            maximum=3000,
        ),
        "min_duration_ms": _coerce_int(
            raw.get("min_duration_ms"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["min_duration_ms"],
            minimum=0,
            maximum=3000,
        ),
        "chunk_ms": _coerce_int(
            raw.get("chunk_ms"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["chunk_ms"],
            minimum=50,
            maximum=1000,
        ),
        "settle_s": _coerce_float(
            raw.get("settle_s"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["settle_s"],
            minimum=0.0,
            maximum=1.0,
        ),
        "inner_pwm": _coerce_int(
            raw.get("inner_pwm"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["inner_pwm"],
            minimum=0,
            maximum=255,
        ),
        "outer_pwm": _coerce_int(
            raw.get("outer_pwm"),
            DEFAULT_RESET_X_GOAL_CURVE_CONFIG["outer_pwm"],
            minimum=1,
            maximum=255,
        ),
    }


def _reset_reverse_turn(
    robot: Robot,
    direction: str,
    reading: dict,
    *,
    rng=None,
) -> dict | None:
    """Send the first reset act without changing mast height."""
    turn_cmd = str(direction or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        return None
    cfg = _reset_motion_config().get("reverse_turn")
    reset_cfg = cfg if isinstance(cfg, dict) else {}
    straight_cfg = _reset_straight_back_first_config(reset_cfg)
    if not bool(straight_cfg.get("enabled", True)):
        return {
            "wheel_ms": 0,
            "straight_back_ms": 0,
            "gentle_ms": 0,
            "sharp_finish_ms": 0,
            "mast_up_ms": 0,
            "mast_settle_s": 0.0,
            "duration_ms": 0,
            "actions": [],
            "curve": {
                "drive_mode": "none",
                "strength": "straight_back_first_disabled",
                "inner_pwm": 0,
                "outer_pwm": 0,
                "turn_cmd": str(turn_cmd),
            },
        }
    min_duration_ms = _coerce_int(straight_cfg.get("duration_min_ms"), 0, minimum=0, maximum=5000)
    max_duration_ms = _coerce_int(straight_cfg.get("duration_max_ms"), 0, minimum=0, maximum=5000)
    if min_duration_ms > 0 and max_duration_ms >= min_duration_ms:
        random_source = rng if rng is not None else random
        duration_ms = int(round(float(random_source.uniform(float(min_duration_ms), float(max_duration_ms)))))
    else:
        duration_ms = _coerce_int(straight_cfg.get("duration_ms"), DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["duration_ms"], minimum=1)
    drive_pwm = _clamp_to_approved_straight_drive_pwm("b", straight_cfg.get("pwm"))
    actions = _straight_drive_actions("b", drive_pwm)
    for action in actions:
        action["duration_ms"] = int(duration_ms)
    scaled_actions = _scaled_actions(actions)
    mast_action, mast_up_ms, mast_settle_s = None, 0, 0.0
    delay_fraction = _coerce_float(
        straight_cfg.get("mast_up_delay_fraction"),
        DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["mast_up_delay_fraction"],
        minimum=0.0,
        maximum=0.95,
    )
    delayed_actions = []
    second_duration_ms = int(duration_ms)
    total_duration_ms = int(duration_ms)
    if mast_action is not None and delay_fraction > 0.0:
        delay_ms = max(1, min(int(duration_ms) - 1, int(round(float(duration_ms) * float(delay_fraction)))))
        remaining_ms = max(1, int(duration_ms) - int(delay_ms))
        second_duration_ms = int(max(int(remaining_ms), int(mast_up_ms)))
        total_duration_ms = int(delay_ms) + int(second_duration_ms)
        first_actions = _straight_drive_actions("b", drive_pwm)
        for action in first_actions:
            action["duration_ms"] = int(delay_ms)
        first_scaled = _scaled_actions(first_actions)
        first_result = _send_reset_custom_actions_pwm(
            robot,
            "b",
            first_scaled,
            duration_ms=delay_ms,
            reading=reading,
            context="reset_straight_back_first_delay",
        )
        if isinstance(first_result, dict) and bool(first_result.get("blocked")):
            return None
        actions = _straight_drive_actions("b", drive_pwm)
        for action in actions:
            action["duration_ms"] = int(remaining_ms)
        scaled_actions = _scaled_actions(actions)
        delayed_actions = [dict(action) for action in first_scaled]
    if mast_action is not None:
        scaled_actions.append(mast_action)
    send_result = _send_reset_custom_actions_pwm(
        robot,
        "b",
        scaled_actions,
        duration_ms=max(int(second_duration_ms), int(mast_up_ms)),
        reading=reading,
        context="reset_straight_back_first",
    )
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        return None
    return {
        "wheel_ms": int(duration_ms),
        "straight_back_ms": int(duration_ms),
        "gentle_ms": int(duration_ms),
        "sharp_finish_ms": 0,
        "mast_up_ms": int(mast_up_ms),
        "mast_settle_s": float(mast_settle_s),
        "duration_ms": int(max(int(total_duration_ms), int(mast_up_ms))),
        "actions": [dict(action) for action in delayed_actions + scaled_actions],
        "curve": {
            "drive_mode": "backward",
            "strength": "straight_back_first",
            "inner_pwm": int(drive_pwm),
            "outer_pwm": int(drive_pwm),
            "turn_cmd": str(turn_cmd),
        },
    }


def _reset_mast_up_action_spec(*, rng=None, reading: dict | None = None) -> tuple[dict | None, int, float]:
    """Build the reset mast-up custom action for the combined reset packet."""
    reset_cfg = _reset_motion_config()
    cfg = reset_cfg.get("mast_up") if isinstance(reset_cfg.get("mast_up"), dict) else {}
    settle_s = _coerce_float(cfg.get("settle_s"), RESET_MAST_UP_SETTLE_S, minimum=0.0)
    if not bool(cfg.get("enabled", True)):
        return None, 0, float(settle_s)
    min_ms = _coerce_int(
        cfg.get("min_duration_ms"),
        RESET_MAST_UP_MIN_MS,
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    max_ms = _coerce_int(
        cfg.get("max_duration_ms"),
        RESET_MAST_UP_MAX_MS,
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    if min_ms > max_ms:
        min_ms, max_ms = max_ms, min_ms
    random_source = rng if rng is not None else random
    try:
        duration_ms = int(round(float(random_source.uniform(float(min_ms), float(max_ms)))))
    except AttributeError:
        duration_ms = int(round((float(min_ms) + float(max_ms)) / 2.0))
    pwm = _coerce_int(cfg.get("pwm"), RESET_MAST_UP_PWM, minimum=1, maximum=255)
    cmd = str(cfg.get("cmd", RESET_MAST_UP_CMD) or "").strip().lower()
    if cmd not in {"u", "d"}:
        cmd = RESET_MAST_UP_CMD
    duration_ms = _cap_mast_duration_ms(cmd, duration_ms, RESET_MAST_UP_MIN_MS, minimum=1)
    if cmd == "u":
        target = float((reset_cfg.get("reverse_turn") or {}).get("y_target_mm", RESET_Y_TARGET_MM))
        blocked, cap_ms, y_mm, ceiling = _mast_up_ceiling_status(reading, target_mm=target)
        if blocked:
            try:
                y_text = f"{float(y_mm):+.1f}mm"
            except (TypeError, ValueError):
                y_text = "N/A"
            print(
                f"[RESET] Mast U skipped by reset y ceiling; current y={y_text}, ceiling={ceiling:+.1f}mm.",
                flush=True,
            )
            return None, 0, float(settle_s)
        if cap_ms is not None:
            duration_ms = min(int(duration_ms), int(cap_ms))
        if int(duration_ms) <= 0:
            return None, 0, float(settle_s)
    wire_action = cmd
    return {
        "target": "m",
        "action": wire_action,
        "pwm": _scaled_pwm_for_cmd(cmd, pwm),
        "duration_ms": int(duration_ms),
    }, int(duration_ms), float(settle_s)


def _reset_mast_up_enabled() -> bool:
    reset_cfg = _reset_motion_config()
    cfg = reset_cfg.get("mast_up") if isinstance(reset_cfg.get("mast_up"), dict) else {}
    return bool(cfg.get("enabled", True))


def _reset_low_x_extra_sharp_config(reset_cfg: dict | None = None) -> dict:
    cfg = reset_cfg if isinstance(reset_cfg, dict) else _reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    raw = cfg.get("low_x_extra_sharp_turn") if isinstance(cfg.get("low_x_extra_sharp_turn"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "threshold_abs_x_mm": _coerce_float(
            raw.get("threshold_abs_x_mm"),
            RESET_LOW_X_EXTRA_TURN_THRESHOLD_MM,
            minimum=0.0,
        ),
        "duration_ms": _coerce_int(
            raw.get("duration_ms"),
            RESET_LOW_X_EXTRA_TURN_DURATION_MS,
            minimum=0,
            maximum=1000,
        ),
        "slower_pwm": _coerce_int(
            raw.get("slower_pwm"),
            RESET_LOW_X_EXTRA_TURN_SLOWER_PWM,
            minimum=1,
            maximum=255,
        ),
        "faster_pwm": _coerce_int(
            raw.get("faster_pwm"),
            RESET_LOW_X_EXTRA_TURN_FASTER_PWM,
            minimum=1,
            maximum=255,
        ),
    }


def _reset_extra_sharp_turn_if_low_x(
    vision: BrickDetector,
    robot: Robot,
    *,
    direction: str,
    reading: dict,
    reset_cfg: dict,
) -> tuple[dict, bool]:
    """Add one ultra-sharp reset correction when the reset x offset is too small."""
    turn_cmd = str(direction or "").strip().lower()
    if turn_cmd not in {"l", "r"} or not isinstance(reading, dict):
        return reading, False
    extra_cfg = _reset_low_x_extra_sharp_config(reset_cfg)
    if not bool(extra_cfg.get("enabled")):
        return reading, False
    duration_ms = int(extra_cfg.get("duration_ms", 0) or 0)
    threshold = float(extra_cfg.get("threshold_abs_x_mm", RESET_LOW_X_EXTRA_TURN_THRESHOLD_MM))
    if duration_ms <= 0 or threshold <= 0.0:
        return reading, False
    try:
        abs_x = abs(float(reading.get("x_mm")))
    except (TypeError, ValueError):
        return reading, False
    if abs_x >= threshold:
        return reading, False

    slower_pwm = int(extra_cfg.get("slower_pwm", RESET_LOW_X_EXTRA_TURN_SLOWER_PWM))
    faster_pwm = int(extra_cfg.get("faster_pwm", RESET_LOW_X_EXTRA_TURN_FASTER_PWM))
    curve = {
        "inner_pwm": int(slower_pwm),
        "outer_pwm": int(faster_pwm),
        "slower_pwm": int(slower_pwm),
        "faster_pwm": int(faster_pwm),
        "x_gap_mm": max(0.0, threshold - abs_x),
    }
    actions = _scaled_actions(_turn_curve_actions(drive_mode="backward", cmd=turn_cmd, curve=curve))
    if not actions:
        return reading, False
    send_result = _send_reset_custom_actions_pwm(
        robot,
        turn_cmd,
        actions,
        duration_ms=duration_ms,
        reading=reading,
        context=f"reset_low_x_extra_sharp_{turn_cmd}_{abs_x:.1f}mm",
    )
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        return reading, False
    print(
        f"[RESET] Extra sharp BACK_TURN_{turn_cmd.upper()}: "
        f"|x|={abs_x:.1f}mm < {threshold:.0f}mm, "
        f"duration={duration_ms}ms faster_pwm={faster_pwm} slower_pwm={slower_pwm}",
        flush=True,
    )
    settle_s = _coerce_float(reset_cfg.get("settle_s"), RESET_REVERSE_TURN_SETTLE_S, minimum=0.0)
    time.sleep(max(LOOP_S, (float(duration_ms) / 1000.0) + float(settle_s)))
    _stop_robot(robot)
    _reset_follow_reading_history(vision)
    after = _read_brick_measurement(vision)
    if not bool(after.get("confident")):
        after = _wait_for_visibility_recovery(
            vision,
            robot,
            after,
            context="reset_after_extra_sharp_turn",
        )
    return after if isinstance(after, dict) else reading, True


def _reset_curve_until_x_goal_fraction(
    vision: BrickDetector,
    robot: Robot,
    *,
    direction: str,
    reading: dict,
    reset_cfg: dict,
    target_abs_x_mm: float,
) -> tuple[dict, dict]:
    """Open reset x with strong bounded curve pulses and read after each pulse."""
    curve_cfg = _reset_x_goal_curve_config(reset_cfg)
    current = reading if isinstance(reading, dict) else {}
    target_fraction = float(curve_cfg.get("target_fraction", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["target_fraction"]))
    x_goal = max(0.0, float(target_abs_x_mm) * max(0.0, min(1.0, target_fraction)))
    max_ms = int(curve_cfg.get("max_duration_ms", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["max_duration_ms"]))
    min_ms = min(
        int(curve_cfg.get("min_duration_ms", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["min_duration_ms"])),
        int(max_ms),
    )
    chunk_ms = min(int(curve_cfg.get("chunk_ms", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["chunk_ms"])), max_ms)
    settle_s = float(curve_cfg.get("settle_s", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["settle_s"]))
    turn_cmd = str(direction or "").strip().lower()
    if turn_cmd not in {"l", "r"}:
        return current, {"enabled": False, "reason": "invalid_turn_direction"}
    if not bool(curve_cfg.get("enabled", True)) or max_ms <= 0 or chunk_ms <= 0 or x_goal <= 0.0:
        return current, {"enabled": False, "reason": "disabled_or_no_goal", "x_goal_mm": float(x_goal)}

    try:
        x_min = _coerce_float(reset_cfg.get("x_offset_min_mm"), RESET_X_OFFSET_MIN_MM, minimum=0.0)
        x_max = _coerce_float(reset_cfg.get("x_offset_max_mm"), RESET_X_OFFSET_MAX_MM, minimum=0.0)
        current_abs_x = abs(float(current.get("x_mm")))
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        ready_floor = min(float(x_max), float(x_min) + 2.0)
        if ready_floor <= current_abs_x <= x_max:
            return current, {
                "enabled": True,
                "reason": "x_target_already_ready",
                "attempts": 0,
                "elapsed_ms": 0,
                "x_goal_mm": float(x_goal),
                "abs_x_mm": float(current_abs_x),
            }
    except (TypeError, ValueError):
        pass

    elapsed_ms = 0
    attempts = 0
    last_abs_x = None
    while elapsed_ms < max_ms and isinstance(current, dict) and bool(current.get("confident")):
        try:
            last_abs_x = abs(float(current.get("x_mm")))
        except (TypeError, ValueError):
            break
        if last_abs_x >= x_goal and elapsed_ms >= min_ms:
            return current, {
                "enabled": True,
                "reason": "x_goal_fraction_hit",
                "attempts": int(attempts),
                "elapsed_ms": int(elapsed_ms),
                "x_goal_mm": float(x_goal),
                "abs_x_mm": float(last_abs_x),
            }
        this_ms = min(int(chunk_ms), int(max_ms) - int(elapsed_ms))
        if elapsed_ms < min_ms:
            this_ms = min(max(int(this_ms), int(min_ms) - int(elapsed_ms)), int(max_ms) - int(elapsed_ms))
        curve = {
            "inner_pwm": int(curve_cfg.get("inner_pwm", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["inner_pwm"])),
            "outer_pwm": int(curve_cfg.get("outer_pwm", DEFAULT_RESET_X_GOAL_CURVE_CONFIG["outer_pwm"])),
            "drive_mode": "backward",
            "strength": "x_goal_strong",
        }
        actions = _scaled_actions(_turn_curve_actions(drive_mode="backward", cmd=turn_cmd, curve=curve))
        if not actions:
            return current, {
                "enabled": True,
                "reason": "no_curve_actions",
                "attempts": int(attempts),
                "elapsed_ms": int(elapsed_ms),
                "x_goal_mm": float(x_goal),
                "abs_x_mm": None if last_abs_x is None else float(last_abs_x),
            }
        send_result = _send_reset_custom_actions_pwm(
            robot,
            turn_cmd,
            actions,
            duration_ms=int(this_ms),
            reading=current,
            context=f"reset_x_goal_curve_{turn_cmd}_{x_goal:.1f}mm",
        )
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            return current, {
                "enabled": True,
                "reason": f"curve_blocked:{send_result.get('reason')}",
                "attempts": int(attempts),
                "elapsed_ms": int(elapsed_ms),
                "x_goal_mm": float(x_goal),
                "abs_x_mm": float(last_abs_x),
            }
        attempts += 1
        elapsed_ms += int(this_ms)
        time.sleep(max(float(LOOP_S), (float(this_ms) / 1000.0) + float(settle_s)))
        _stop_robot(robot)
        _reset_follow_reading_history(vision)
        current = _read_brick_measurement(vision)
        if not bool(current.get("confident")):
            current = _wait_for_visibility_recovery(
                vision,
                robot,
                current,
                context="reset_x_goal_curve",
            )

    try:
        final_abs_x = abs(float((current or {}).get("x_mm")))
    except (TypeError, ValueError):
        final_abs_x = None
    reason = "x_goal_curve_max_duration" if final_abs_x is not None and final_abs_x < x_goal else "x_goal_curve_complete"
    return current, {
        "enabled": True,
        "reason": reason,
        "attempts": int(attempts),
        "elapsed_ms": int(elapsed_ms),
        "x_goal_mm": float(x_goal),
        "abs_x_mm": None if final_abs_x is None else float(final_abs_x),
    }


def _reset_xy_target_ready(reading: dict, reset_cfg: dict) -> bool:
    """Reset readiness for the operator target: dist and abs x only."""
    if not isinstance(reading, dict):
        return False
    try:
        dist_mm = float(reading.get("dist_mm"))
        x_mm = float(reading.get("x_mm"))
        if dist_mm < _reset_min_attempt_dist_mm(reset_cfg):
            return False
        return bool(_reset_x_offset_ready(x_mm, dist_mm, reset_cfg, y_mm=None))
    except (TypeError, ValueError):
        return False


def _reset_adjustment_config(reset_cfg: dict) -> dict:
    raw = reset_cfg.get("adjustment") if isinstance(reset_cfg.get("adjustment"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_attempts": _coerce_int(raw.get("max_attempts"), 6, minimum=0, maximum=20),
        "pulse_min_ms": _coerce_int(raw.get("pulse_min_ms"), 60, minimum=1, maximum=400),
        "pulse_max_ms": _coerce_int(raw.get("pulse_max_ms"), 260, minimum=1, maximum=500),
        "settle_s": _coerce_float(raw.get("settle_s"), 0.12, minimum=0.0, maximum=2.0),
    }


def _reset_adjustment_pulse_ms(gap_mm: float, cfg: dict) -> int:
    min_ms = int(cfg.get("pulse_min_ms", 60))
    max_ms = int(cfg.get("pulse_max_ms", 260))
    if min_ms > max_ms:
        min_ms, max_ms = max_ms, min_ms
    scaled = int(round(float(min_ms) + min(abs(float(gap_mm)), 40.0) / 40.0 * (float(max_ms) - float(min_ms))))
    return max(min_ms, min(max_ms, scaled))


def _reset_small_turn_adjust(
    robot: Robot,
    *,
    turn_cmd: str,
    reading: dict,
    duration_ms: int,
    reset_cfg: dict,
    reason: str,
):
    extra_cfg = _reset_low_x_extra_sharp_config(reset_cfg)
    slower_pwm = int(extra_cfg.get("slower_pwm", RESET_LOW_X_EXTRA_TURN_SLOWER_PWM))
    faster_pwm = int(extra_cfg.get("faster_pwm", RESET_LOW_X_EXTRA_TURN_FASTER_PWM))
    curve = {
        "inner_pwm": int(slower_pwm),
        "outer_pwm": int(faster_pwm),
        "slower_pwm": int(slower_pwm),
        "faster_pwm": int(faster_pwm),
        "x_gap_mm": 0.0,
    }
    actions = _scaled_actions(_turn_curve_actions(drive_mode="backward", cmd=turn_cmd, curve=curve))
    if not actions:
        return {"blocked": True, "reason": "reset_adjust_no_actions", "cmd": turn_cmd}
    return _send_reset_custom_actions_pwm(
        robot,
        turn_cmd,
        actions,
        duration_ms=int(duration_ms),
        reading=reading,
        context=f"reset_adjust_{reason}_{turn_cmd}",
    )


def _recover_reset_visibility_with_backoff(
    vision: BrickDetector,
    robot: Robot,
    reading: dict | None,
    *,
    context: str,
    attempts: int = 5,
    pulse_ms: int = 250,
) -> dict:
    """Recover reset visibility after a forward polish likely moved too close."""
    current = reading if isinstance(reading, dict) else {}
    pwm = _approved_straight_drive_pwm("b")
    for idx in range(max(0, int(attempts))):
        if isinstance(current, dict) and bool(current.get("confident")):
            return current
        print(
            f"[RESET] Visibility recovery {idx + 1}/{int(attempts)} after {context}: "
            f"backing away {int(pulse_ms)}ms to reacquire stack.",
            flush=True,
        )
        try:
            robot.send_command_pwm("b", pwm, duration_ms=int(pulse_ms))
            time.sleep(max(float(LOOP_S), float(pulse_ms) / 1000.0 + 0.15))
        finally:
            _stop_robot(robot)
        _reset_follow_reading_history(vision, allow_large_dist_jump=True)
        current = _read_brick_measurement(vision)
        if not bool(current.get("confident")):
            current = _wait_for_visibility_recovery(
                vision,
                robot,
                current,
                timeout_s=1.0,
                sample_s=0.1,
                context=f"{context}_backoff",
            )
    return current if isinstance(current, dict) else {}


def _reset_read_after_adjustment(
    vision: BrickDetector,
    robot: Robot,
    *,
    duration_ms: int,
    settle_s: float,
    context: str,
    fallback: dict,
) -> dict:
    time.sleep(max(LOOP_S, (float(duration_ms) / 1000.0) + float(settle_s)))
    _stop_robot(robot)
    _reset_follow_reading_history(vision)
    after = _read_brick_measurement(vision)
    if not bool(after.get("confident")):
        after = _wait_for_visibility_recovery(vision, robot, after, context=context)
    if not bool(after.get("confident")) and "dist_forward" in str(context):
        after = _recover_reset_visibility_with_backoff(
            vision,
            robot,
            after,
            context=context,
        )
    return after if isinstance(after, dict) else fallback


def _adjust_reset_until_xy_target(
    vision: BrickDetector,
    robot: Robot,
    *,
    reading: dict,
    reset_cfg: dict,
    initial_turn_cmd: str,
) -> tuple[dict, bool, int]:
    """Bounded reset polishing loop for dist + abs-x target."""
    cfg = _reset_adjustment_config(reset_cfg)
    current = reading
    if not bool(cfg.get("enabled")) or _reset_xy_target_ready(current, reset_cfg):
        return current, _reset_xy_target_ready(current, reset_cfg), 0

    x_min = float(reset_cfg.get("x_offset_min_mm", RESET_X_OFFSET_MIN_MM))
    x_max = float(reset_cfg.get("x_offset_max_mm", RESET_X_OFFSET_MAX_MM))
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    dist_target = float(reset_cfg.get("dist_target_mm", RESET_DIST_TARGET_MM))
    dist_tol = float(reset_cfg.get("dist_tol_mm", RESET_DIST_TOL_MM))
    dist_low = max(float(dist_target) - float(dist_tol), _reset_min_attempt_dist_mm(reset_cfg))
    settle_s = float(cfg.get("settle_s", 0.12))
    attempts = 0
    turn_cmd = str(initial_turn_cmd or "r").strip().lower()
    turn_cmd = turn_cmd if turn_cmd in {"l", "r"} else "r"

    for attempt_idx in range(int(cfg.get("max_attempts", 0))):
        if not isinstance(current, dict) or not bool(current.get("confident")):
            break
        try:
            dist_mm = float(current.get("dist_mm"))
            x_mm = float(current.get("x_mm"))
        except (TypeError, ValueError):
            break
        abs_x = abs(float(x_mm))
        if _reset_xy_target_ready(current, reset_cfg):
            return current, True, attempts

        if dist_mm < dist_low:
            cmd = "b"
            gap = dist_low - dist_mm
            duration_ms = _reset_adjustment_pulse_ms(gap, cfg)
            send_result = guarded_send_command_pwm(
                robot,
                cmd,
                _approved_straight_drive_pwm(cmd),
                duration_ms=duration_ms,
                reading=current,
                context="reset_adjust_dist_back",
            )
            reason = "dist_back"
        elif dist_mm > dist_target + dist_tol:
            cmd = "f"
            gap = dist_mm - (dist_target + dist_tol)
            duration_ms = _reset_adjustment_pulse_ms(gap, cfg)
            send_result = guarded_send_command_pwm(
                robot,
                cmd,
                _approved_straight_drive_pwm(cmd),
                duration_ms=duration_ms,
                reading=current,
                context="reset_adjust_dist_forward",
            )
            reason = "dist_forward"
        elif abs_x < x_min:
            gap = x_min - abs_x
            duration_ms = _reset_adjustment_pulse_ms(gap, cfg)
            open_turn = _turn_cmd_to_open_x_gap(x_mm, turn_cmd)
            send_result = _reset_small_turn_adjust(
                robot,
                turn_cmd=open_turn,
                reading=current,
                duration_ms=duration_ms,
                reset_cfg=reset_cfg,
                reason="increase_x",
            )
            reason = "increase_x"
        else:
            gap = abs_x - x_max
            duration_ms = _reset_adjustment_pulse_ms(gap, cfg)
            correction_turn = _turn_cmd_to_close_x_gap(x_mm) or ("l" if turn_cmd == "r" else "r")
            send_result = _reset_small_turn_adjust(
                robot,
                turn_cmd=correction_turn,
                reading=current,
                duration_ms=duration_ms,
                reset_cfg=reset_cfg,
                reason="reduce_x",
            )
            reason = "reduce_x"

        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            print(f"[RESET] Adjustment blocked: {reason} ({send_result.get('reason')})", flush=True)
            break
        attempts += 1
        print(f"[RESET] Adjustment {attempt_idx + 1}: {reason} pulse={duration_ms}ms", flush=True)
        current = _reset_read_after_adjustment(
            vision,
            robot,
            duration_ms=duration_ms,
            settle_s=settle_s,
            context=f"reset_adjust_{reason}",
            fallback=current,
        )
        if _reset_xy_target_ready(current, reset_cfg):
            return current, True, attempts

    return current, _reset_xy_target_ready(current, reset_cfg), attempts


def _reset_final_x_offset_polish(
    vision: BrickDetector,
    robot: Robot,
    *,
    reading: dict,
    reset_cfg: dict,
    initial_turn_cmd: str,
) -> tuple[dict, bool, str]:
    """End reset with a visible x-offset polish after distance-only adjustments."""
    current = reading if isinstance(reading, dict) else {}
    if not bool(current.get("confident")):
        return current, False, "not_confident"
    try:
        dist_mm = float(current.get("dist_mm"))
        x_mm = float(current.get("x_mm"))
    except (TypeError, ValueError):
        return current, False, "invalid_reading"

    x_min = float(reset_cfg.get("x_offset_min_mm", RESET_X_OFFSET_MIN_MM))
    x_max = float(reset_cfg.get("x_offset_max_mm", RESET_X_OFFSET_MAX_MM))
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    target_abs_x = _coerce_float(
        reset_cfg.get("target_abs_x_mm"),
        (float(x_min) + float(x_max)) / 2.0,
        minimum=0.0,
    )
    if target_abs_x < x_min or target_abs_x > x_max:
        target_abs_x = (float(x_min) + float(x_max)) / 2.0

    dist_target = float(reset_cfg.get("dist_target_mm", RESET_DIST_TARGET_MM))
    dist_tol = float(reset_cfg.get("dist_tol_mm", RESET_DIST_TOL_MM))
    dist_low = max(float(dist_target) - float(dist_tol), _reset_min_attempt_dist_mm(reset_cfg))
    if dist_mm < dist_low or dist_mm > (float(dist_target) + float(dist_tol)):
        print(
            f"[RESET] Final x-offset polish skipped: dist={dist_mm:.1f}mm outside reset-ready band "
            f"{dist_low:.1f}-{float(dist_target) + float(dist_tol):.1f}mm.",
            flush=True,
        )
        return current, False, "dist_not_ready"

    abs_x = abs(float(x_mm))
    if abs_x >= float(target_abs_x) - float(RESET_FINAL_X_POLISH_MARGIN_MM):
        print(
            f"[RESET] Final x-offset polish skipped: |x|={abs_x:.1f}mm already near "
            f"{target_abs_x:.1f}mm target.",
            flush=True,
        )
        return current, False, "already_near_target"
    if abs_x >= float(x_max) - float(RESET_FINAL_X_POLISH_MARGIN_MM):
        print(
            f"[RESET] Final x-offset polish skipped: |x|={abs_x:.1f}mm already near max "
            f"{x_max:.1f}mm.",
            flush=True,
        )
        return current, False, "already_near_max"

    cfg = _reset_adjustment_config(reset_cfg)
    gap = max(0.0, float(target_abs_x) - float(abs_x))
    duration_ms = _reset_adjustment_pulse_ms(gap, cfg)
    duration_ms = max(
        int(RESET_FINAL_X_POLISH_MIN_MS),
        min(int(RESET_FINAL_X_POLISH_MAX_MS), int(duration_ms)),
    )
    turn_cmd = str(initial_turn_cmd or "r").strip().lower()
    turn_cmd = turn_cmd if turn_cmd in {"l", "r"} else "r"
    open_turn = _turn_cmd_to_open_x_gap(x_mm, turn_cmd)
    send_result = _reset_small_turn_adjust(
        robot,
        turn_cmd=open_turn,
        reading=current,
        duration_ms=duration_ms,
        reset_cfg=reset_cfg,
        reason="final_x_offset",
    )
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        print(
            f"[RESET] Final x-offset polish blocked: {send_result.get('reason')}",
            flush=True,
        )
        return current, False, "blocked"

    print(
        f"[RESET] Final x-offset polish: BACK_CURVE_{open_turn.upper()} "
        f"pulse={duration_ms}ms |x|={abs_x:.1f}->{target_abs_x:.1f}mm",
        flush=True,
    )
    after = _reset_read_after_adjustment(
        vision,
        robot,
        duration_ms=duration_ms,
        settle_s=float(cfg.get("settle_s", 0.12)),
        context="reset_final_x_offset_polish",
        fallback=current,
    )
    return after if isinstance(after, dict) else current, True, "sent"


def _warmup(vision: BrickDetector) -> None:
    log.info("Warming up camera pipeline (%d reads)...", WARMUP_READS)
    for _ in range(WARMUP_READS):
        try:
            vision.read()
        except Exception:
            pass
        time.sleep(0.06)
    log.info("Camera ready.")


def _wait_for_visibility_recovery(
    vision: BrickDetector,
    robot: Robot | None = None,
    reading: dict | None = None,
    *,
    timeout_s: float | None = None,
    sample_s: float | None = None,
    context: str = "visibility_recovery",
    jump_guard: bool = False,
) -> dict:
    current = reading if isinstance(reading, dict) else brick_motion_measurement_from_result(None)
    if bool(current.get("confident")):
        return current
    cfg = _visibility_recovery_config()
    wait_s = cfg["wait_s"] if timeout_s is None else _coerce_float(timeout_s, cfg["wait_s"], minimum=0.0, maximum=10.0)
    poll_s = cfg["poll_s"] if sample_s is None else _coerce_float(sample_s, cfg["poll_s"], minimum=0.01, maximum=2.0)
    if wait_s <= 0.0:
        return current
    reset_context = str(context or "").strip().lower().startswith("reset")
    if robot is not None and not reset_context:
        _stop_robot(robot)
        down_ms = _cap_mast_duration_ms(
            "d",
            cfg.get("mast_down_duration_ms"),
            DEFAULT_VISIBILITY_RECOVERY_CONFIG["mast_down_duration_ms"],
            minimum=0,
        )
        if down_ms > 0:
            down_pwm = _coerce_int(
                cfg.get("mast_down_pwm"),
                DEFAULT_VISIBILITY_RECOVERY_CONFIG["mast_down_pwm"],
                minimum=1,
                maximum=255,
            )
            print(
                f"[FOLLOW] Visibility not confident after {context}; "
                f"recovering with mast D for {int(down_ms)}ms, no wheel motion.",
                flush=True,
            )
            robot.send_command_pwm("d", down_pwm, duration_ms=down_ms)
            time.sleep(float(down_ms) / 1000.0)
            _stop_robot(robot)
            _reset_follow_reading_history(vision, allow_large_dist_jump=True)
            current = _read_brick_measurement(vision, jump_guard=jump_guard)
            if bool(current.get("confident")):
                try:
                    print(
                        f"[FOLLOW] Visibility recovered after mast D: "
                        f"dist={float(current.get('dist_mm')):.1f}mm "
                        f"x={float(current.get('x_mm')):+.1f}mm "
                        f"y={float(current.get('y_mm')):+.1f}mm",
                        flush=True,
                    )
                except (TypeError, ValueError):
                    print("[FOLLOW] Visibility recovered after mast D.", flush=True)
                return current
    deadline = time.monotonic() + float(wait_s)
    while time.monotonic() < deadline and not bool(current.get("confident")):
        time.sleep(min(float(poll_s), max(0.0, deadline - time.monotonic())))
        current = _read_brick_measurement(vision, jump_guard=jump_guard)
    if bool(current.get("confident")):
        try:
            print(
                f"[FOLLOW] Visibility recovered after {context}: "
                f"dist={float(current.get('dist_mm')):.1f}mm "
                f"x={float(current.get('x_mm')):+.1f}mm "
                f"y={float(current.get('y_mm')):+.1f}mm",
                flush=True,
            )
        except (TypeError, ValueError):
            print(f"[FOLLOW] Visibility recovered after {context}.", flush=True)
    return current


def _observe_after_hard_ghost_jump(
    vision: BrickDetector,
    robot: Robot,
    stats: dict,
    reading: dict,
    *,
    context: str,
) -> tuple[dict, bool]:
    cfg = _vision_jump_guard_config()
    observe_s = _coerce_float(
        cfg.get("hard_reject_observe_s"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["hard_reject_observe_s"],
        minimum=0.0,
        maximum=10.0,
    )
    poll_s = _coerce_float(
        cfg.get("hard_reject_observe_poll_s"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["hard_reject_observe_poll_s"],
        minimum=0.01,
        maximum=2.0,
    )
    delta = reading.get("ghost_jump_delta") if isinstance(reading.get("ghost_jump_delta"), dict) else {}
    _stop_robot(robot)
    _bump_stat_count(stats, "miss_reasons", "ghost_jump_hard_rejected")
    recovery_count = int(stats.get("hard_ghost_jump_recovery_count", 0) or 0)
    max_recoveries = _coerce_int(
        cfg.get("hard_reject_max_recoveries_per_step"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["hard_reject_max_recoveries_per_step"],
        minimum=0,
        maximum=10,
    )
    print(
        "[FOLLOW] HARD_GHOST_JUMP "
        f"dist={float(delta.get('dist', 0.0)):.1f}mm "
        f"x={float(delta.get('x', 0.0)):.1f}mm "
        f"y={float(delta.get('y', 0.0) or 0.0):.1f}mm; "
        f"holding still and observing for {float(observe_s):.1f}s.",
        flush=True,
    )
    recovered = _wait_for_visibility_recovery(
        vision,
        robot,
        reading,
        timeout_s=float(observe_s),
        sample_s=float(poll_s),
        context=context,
        jump_guard=True,
    )
    if bool(recovered.get("confident")) and recovery_count < max_recoveries:
        stats["hard_ghost_jump_recovery_count"] = recovery_count + 1
        print("[FOLLOW] Hard ghost jump cleared during still observation; resuming from stable brick.", flush=True)
        return recovered, False
    stats["hard_ghost_jump_stop"] = True
    stats["debug_stop_reading"] = dict(recovered) if isinstance(recovered, dict) else recovered
    stats["last_action"] = "HARD_GHOST_JUMP_STOP"
    print(
        "[FOLLOW] HARD STOP: brick candidate jumped >= "
        f"{float(cfg.get('hard_reject_dist_jump_mm', 0.0) or 0.0):.0f}mm while Leia was not doing a distance move. "
        "Treating it as a ghost stack and refusing to close gaps from it.",
        flush=True,
    )
    return recovered, True


def _wait_for_confident_brick(
    vision: BrickDetector,
    *,
    timeout_s: float | None = None,
    sample_s: float | None = None,
) -> dict:
    cfg = _cautious_visibility_config()
    timeout_val = cfg["pregame_timeout_s"] if timeout_s is None else float(timeout_s)
    sample_val = cfg["pregame_sample_s"] if sample_s is None else float(sample_s)
    sample_frames = int(cfg["pregame_sample_frames"])
    required_frames = int(cfg["pregame_required_frames"])
    deadline = time.monotonic() + max(0.0, float(timeout_val))
    last_reading = brick_motion_measurement_from_result(None)
    passing_readings: list[dict] = []
    best_reading: dict | None = None
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    sample_count = 0
    first_sample = True
    while first_sample or time.monotonic() <= deadline:
        first_sample = False
        last_reading = _read_brick_measurement(vision)
        sample_count += 1
        status_counts[str(last_reading.get("vision_status") or "unknown")] += 1
        reason_counts[str(last_reading.get("reason") or "unknown")] += 1
        geometry_counts[str(last_reading.get("vision_geometry_source") or "unknown")] += 1
        try:
            best_conf = float(best_reading.get("conf")) if isinstance(best_reading, dict) and best_reading.get("conf") is not None else -1.0
            conf = float(last_reading.get("conf")) if last_reading.get("conf") is not None else -1.0
        except (TypeError, ValueError):
            best_conf, conf = -1.0, -1.0
        if best_reading is None or conf > best_conf:
            best_reading = dict(last_reading)
        if bool(last_reading.get("confident")):
            passing_readings.append(dict(last_reading))
            passing_readings = passing_readings[-sample_frames:]
        if len(passing_readings) >= required_frames:
            chosen = passing_readings[-1]
            try:
                dist_mm = float(chosen.get("dist_mm"))
                x_mm = float(chosen.get("x_mm"))
                y_mm = float(chosen.get("y_mm"))
                conf = float(chosen.get("conf"))
                print(
                    f"[FOLLOW] Pregame visibility ok: dist={dist_mm:.1f}mm "
                    f"x={x_mm:+.1f}mm y={y_mm:+.1f}mm conf={conf:.0f}% "
                    f"frames={len(passing_readings)}/{required_frames}",
                    flush=True,
                )
            except (TypeError, ValueError):
                print("[FOLLOW] Pregame visibility ok.", flush=True)
            return chosen
        time.sleep(max(0.01, float(sample_val)))
    diagnostic = best_reading if isinstance(best_reading, dict) else last_reading
    try:
        conf_text = f"{float(diagnostic.get('conf')):.0f}%"
    except (TypeError, ValueError):
        conf_text = "N/A"
    try:
        dist_text = f"{float(diagnostic.get('dist_mm')):.1f}mm"
        x_text = f"{float(diagnostic.get('x_mm')):+.1f}mm"
        y_text = f"{float(diagnostic.get('y_mm')):+.1f}mm"
    except (TypeError, ValueError):
        dist_text = x_text = y_text = "N/A"
    status_text = ", ".join(f"{key}={count}" for key, count in status_counts.most_common(4)) or "none"
    reason_text = ", ".join(f"{key}={count}" for key, count in reason_counts.most_common(4)) or "none"
    geometry_text = ", ".join(f"{key}={count}" for key, count in geometry_counts.most_common(4)) or "none"
    print(
        "[FOLLOW] Pregame visibility blocked: brick was not confident; "
        "no robot motion was started. "
        f"best reason={diagnostic.get('reason')} conf={conf_text} "
        f"dist={dist_text} x={x_text} y={y_text} "
        f"passing_frames={len(passing_readings)}/{required_frames} "
        f"samples={int(sample_count)} timeout={float(timeout_val):.1f}s "
        f"detector_statuses=[{status_text}] reasons=[{reason_text}] geometry=[{geometry_text}]",
        flush=True,
    )
    return last_reading


def _mask_held_brick_for_target_frame(frame, holding_result: dict | None):
    """Return a frame copy that hides held-brick pixels from target detection."""
    return mask_held_brick_for_target_frame(frame, holding_result)


def _is_placeholder_reading(reading: dict | None) -> bool:
    if not isinstance(reading, dict) or not bool(reading.get("visible")):
        return False
    try:
        dist = float(reading.get("dist_mm", reading.get("dist")))
        x = abs(float(reading.get("x_mm", reading.get("x"))))
        y = abs(float(reading.get("y_mm", reading.get("y"))))
        confidence = float(reading.get("conf", reading.get("confidence")))
    except (TypeError, ValueError):
        return False
    return dist >= 490.0 and x <= 0.1 and y <= 0.1 and confidence <= 75.0


def _reading_median(rows: list[dict]) -> dict:
    out = dict(rows[-1])
    for key in ("dist_mm", "x_mm", "y_mm", "conf"):
        values = []
        for row in rows:
            try:
                values.append(float(row.get(key)))
            except (TypeError, ValueError):
                pass
        if values:
            values.sort()
            out[key] = float(values[len(values) // 2])
    out["temporal_filtered"] = True
    out["temporal_filter_samples"] = len(rows)
    return out


def _reading_jump_values(reading: dict | None) -> tuple[float, float, float | None] | None:
    if not isinstance(reading, dict):
        return None
    try:
        dist_mm = float(reading.get("dist_mm"))
        x_mm = float(reading.get("x_mm"))
    except (TypeError, ValueError):
        return None
    y_mm = None
    try:
        raw_y = reading.get("y_mm")
        if raw_y is not None:
            y_mm = float(raw_y)
    except (TypeError, ValueError):
        y_mm = None
    return float(dist_mm), float(x_mm), y_mm


def _reading_jump_delta(previous: dict | None, current: dict | None) -> dict | None:
    prev_values = _reading_jump_values(previous)
    cur_values = _reading_jump_values(current)
    if prev_values is None or cur_values is None:
        return None
    prev_dist, prev_x, prev_y = prev_values
    cur_dist, cur_x, cur_y = cur_values
    delta = {
        "dist": abs(float(cur_dist) - float(prev_dist)),
        "x": abs(float(cur_x) - float(prev_x)),
    }
    vector_terms = [float(delta["dist"]) ** 2, float(delta["x"]) ** 2]
    if prev_y is not None and cur_y is not None:
        delta["y"] = abs(float(cur_y) - float(prev_y))
        vector_terms.append(float(delta["y"]) ** 2)
    else:
        delta["y"] = None
    delta["vector"] = float(sum(vector_terms) ** 0.5)
    return delta


def _reading_jump_suspicious(previous: dict | None, current: dict | None, cfg: dict) -> tuple[bool, dict | None]:
    delta = _reading_jump_delta(previous, current)
    if delta is None:
        return False, None
    checks = (
        ("dist", "max_dist_jump_mm"),
        ("x", "max_x_jump_mm"),
        ("y", "max_y_jump_mm"),
        ("vector", "max_vector_jump_mm"),
    )
    for delta_key, cfg_key in checks:
        value = delta.get(delta_key)
        if value is None:
            continue
        limit = float(cfg.get(cfg_key, 0.0) or 0.0)
        if limit > 0.0 and float(value) > limit:
            return True, delta
    return False, delta


def _reading_hard_ghost_jump(previous: dict | None, current: dict | None, cfg: dict) -> dict | None:
    delta = _reading_jump_delta(previous, current)
    if delta is None:
        return None
    limit = float(cfg.get("hard_reject_dist_jump_mm", 0.0) or 0.0)
    if limit > 0.0 and float(delta.get("dist", 0.0)) >= limit:
        return delta
    return None


def _ghost_jump_rejected_reading(reading: dict, delta: dict | None, reason: str) -> dict:
    rejected = dict(reading)
    rejected["confident"] = False
    rejected["reason"] = str(reason)
    rejected[str(reason)] = True
    if delta is not None:
        rejected["ghost_jump_delta"] = dict(delta)
    return rejected


def _readings_close_for_jump_confirmation(first: dict | None, second: dict | None, *, window_mm: float) -> bool:
    delta = _reading_jump_delta(first, second)
    if delta is None:
        return False
    limit = max(0.0, float(window_mm))
    return bool(
        float(delta.get("dist", 0.0)) <= limit
        and float(delta.get("x", 0.0)) <= limit
        and (
            delta.get("y") is None
            or float(delta.get("y", 0.0)) <= limit
        )
    )


def _set_follow_last_stable_reading(vision: BrickDetector, reading: dict) -> None:
    try:
        setattr(vision, "_follow_last_stable_reading", dict(reading))
    except Exception:
        pass


def _temporal_filter_brick_reading(vision: BrickDetector, reading: dict, jump_guard: bool = False) -> dict:
    """Small 3-frame median plus jump-confirm guard around the detector."""
    if not isinstance(reading, dict):
        return reading
    jump_guard = bool(jump_guard or getattr(vision, "_follow_jump_guard_requested", False))
    history = getattr(vision, "_follow_reading_history", None)
    if history is None:
        history = deque(maxlen=3)
        try:
            setattr(vision, "_follow_reading_history", history)
        except Exception:
            pass
    if _is_placeholder_reading(reading):
        reading = dict(reading)
        reading["confident"] = False
        reading["reason"] = "placeholder_500_0_0_rejected"
        reading["placeholder_rejected"] = True
        try:
            history.clear()
        except Exception:
            pass
        try:
            setattr(vision, "_follow_jump_candidate", None)
        except Exception:
            pass
        return reading
    if bool(reading.get("confident")):
        jump_cfg = _vision_jump_guard_config()
        if bool(jump_guard) and bool(getattr(vision, "_follow_reacquire_after_motion", False)):
            reacquire_frames = _coerce_int(
                jump_cfg.get("reacquire_frames_after_motion"),
                DEFAULT_VISION_JUMP_GUARD_CONFIG["reacquire_frames_after_motion"],
                minimum=0,
                maximum=10,
            )
            if reacquire_frames <= 1 and bool(getattr(vision, "_follow_allow_large_dist_reacquire", True)):
                try:
                    setattr(vision, "_follow_reacquire_after_motion", False)
                    setattr(vision, "_follow_jump_candidate", None)
                except Exception:
                    pass
                _set_follow_last_stable_reading(vision, reading)
                return reading
            if reacquire_frames > 1:
                window_mm = _coerce_float(
                    jump_cfg.get("reacquire_window_mm"),
                    DEFAULT_VISION_JUMP_GUARD_CONFIG["reacquire_window_mm"],
                    minimum=0.0,
                    maximum=500.0,
                )
                if not bool(getattr(vision, "_follow_allow_large_dist_reacquire", True)):
                    stable_before_motion = getattr(vision, "_follow_last_stable_reading", None)
                    hard_delta = _reading_hard_ghost_jump(stable_before_motion, reading, jump_cfg)
                    if hard_delta is not None:
                        rejected = _ghost_jump_rejected_reading(
                            reading,
                            hard_delta,
                            "ghost_jump_hard_rejected",
                        )
                        try:
                            setattr(
                                vision,
                                "_follow_jump_candidate",
                                {"reading": dict(reading), "count": 1},
                            )
                        except Exception:
                            pass
                        return rejected
                last_reacquire = history[-1] if len(history) else None
                if last_reacquire is not None and not _readings_close_for_jump_confirmation(
                    last_reacquire,
                    reading,
                    window_mm=float(window_mm),
                ):
                    history.clear()
                history.append(dict(reading))
                if len(history) < int(reacquire_frames):
                    rejected = dict(reading)
                    rejected["confident"] = False
                    rejected["reason"] = "reacquiring_stable_brick_lock"
                    rejected["reacquiring_stable_brick_lock"] = True
                    return rejected
                filtered = _reading_median([dict(row) for row in list(history)])
                filtered["reason"] = str(reading.get("reason") or "confident_visible")
                try:
                    setattr(vision, "_follow_reacquire_after_motion", False)
                    setattr(vision, "_follow_jump_candidate", None)
                except Exception:
                    pass
                _set_follow_last_stable_reading(vision, filtered)
                return filtered
        max_abs_dist = float(jump_cfg.get("max_abs_dist_mm", 0.0) or 0.0)
        if max_abs_dist > 0.0:
            try:
                dist_val = float(reading.get("dist_mm"))
                x_val = abs(float(reading.get("x_mm")))
                conf_val = float(reading.get("conf") or 0.0)
                edge_recovery = (
                    x_val >= float(jump_cfg.get("edge_recovery_min_abs_x_mm", 200.0) or 0.0)
                    and conf_val >= float(jump_cfg.get("edge_recovery_min_confidence_pct", 90.0) or 0.0)
                    and dist_val <= float(jump_cfg.get("edge_recovery_max_abs_dist_mm", 500.0) or 0.0)
                )
                if dist_val > max_abs_dist and not bool(edge_recovery):
                    rejected = dict(reading)
                    rejected["confident"] = False
                    rejected["reason"] = "ghost_dist_implausible"
                    rejected["ghost_dist_implausible"] = True
                    return rejected
            except (TypeError, ValueError):
                pass
        stable = getattr(vision, "_follow_last_stable_reading", None)
        suspicious, delta = (
            _reading_jump_suspicious(stable, reading, jump_cfg)
            if bool(jump_guard) and bool(jump_cfg.get("enabled"))
            else (False, None)
        )
        if suspicious:
            hard_delta = _reading_hard_ghost_jump(stable, reading, jump_cfg)
            if hard_delta is not None:
                rejected = _ghost_jump_rejected_reading(
                    reading,
                    hard_delta,
                    "ghost_jump_hard_rejected",
                )
                try:
                    setattr(vision, "_follow_jump_candidate", {"reading": dict(reading), "count": 1})
                except Exception:
                    pass
                return rejected
            candidate = getattr(vision, "_follow_jump_candidate", None)
            candidate_reading = candidate.get("reading") if isinstance(candidate, dict) else None
            if _readings_close_for_jump_confirmation(
                candidate_reading,
                reading,
                window_mm=float(jump_cfg.get("confirm_window_mm", 10.0)),
            ):
                count = int(candidate.get("count", 1) if isinstance(candidate, dict) else 1) + 1
            else:
                count = 1
            confirm_frames = int(jump_cfg.get("confirm_frames", 2) or 2)
            if count < confirm_frames:
                rejected = dict(reading)
                rejected["confident"] = False
                rejected["reason"] = "ghost_jump_unconfirmed"
                rejected["ghost_jump_unconfirmed"] = True
                if delta is not None:
                    rejected["ghost_jump_delta"] = dict(delta)
                try:
                    setattr(vision, "_follow_jump_candidate", {"reading": dict(reading), "count": int(count)})
                except Exception:
                    pass
                return rejected
            if not bool(jump_cfg.get("accept_confirmed_jumps", True)):
                rejected = dict(reading)
                rejected["confident"] = False
                rejected["reason"] = "ghost_jump_confirmed_rejected"
                rejected["ghost_jump_confirmed_rejected"] = True
                if delta is not None:
                    rejected["ghost_jump_delta"] = dict(delta)
                try:
                    setattr(vision, "_follow_jump_candidate", {"reading": dict(reading), "count": int(count)})
                except Exception:
                    pass
                return rejected
            try:
                setattr(vision, "_follow_jump_candidate", None)
                history.clear()
            except Exception:
                pass
            reading = dict(reading)
            reading["jump_confirmed"] = True
            if delta is not None:
                reading["ghost_jump_delta"] = dict(delta)
        else:
            try:
                setattr(vision, "_follow_jump_candidate", None)
            except Exception:
                pass
        history.append(dict(reading))
        if len(history) >= 3:
            filtered = _reading_median([dict(row) for row in list(history)])
            filtered["reason"] = str(reading.get("reason") or "confident_visible")
            _set_follow_last_stable_reading(vision, filtered)
            return filtered
        _set_follow_last_stable_reading(vision, reading)
    return reading


def _apply_temporal_filter_brick_reading(
    vision: BrickDetector,
    reading: dict,
    *,
    jump_guard: bool = False,
) -> dict:
    if not bool(jump_guard):
        return _temporal_filter_brick_reading(vision, reading)
    sentinel = object()
    previous = getattr(vision, "_follow_jump_guard_requested", sentinel)
    try:
        setattr(vision, "_follow_jump_guard_requested", True)
    except Exception:
        pass
    try:
        return _temporal_filter_brick_reading(vision, reading)
    finally:
        try:
            if previous is sentinel:
                delattr(vision, "_follow_jump_guard_requested")
            else:
                setattr(vision, "_follow_jump_guard_requested", previous)
        except Exception:
            pass


def _reset_follow_reading_history(
    vision: BrickDetector,
    *,
    allow_large_dist_jump: bool = True,
) -> None:
    try:
        setattr(vision, "_follow_jump_candidate", None)
    except Exception:
        pass
    try:
        setattr(vision, "_follow_allow_large_dist_reacquire", bool(allow_large_dist_jump))
    except Exception:
        pass
    if bool(allow_large_dist_jump):
        try:
            setattr(vision, "_follow_last_stable_reading", None)
        except Exception:
            pass
    try:
        jump_cfg = _vision_jump_guard_config()
        setattr(
            vision,
            "_follow_reacquire_after_motion",
            _coerce_int(
                jump_cfg.get("reacquire_frames_after_motion"),
                DEFAULT_VISION_JUMP_GUARD_CONFIG["reacquire_frames_after_motion"],
                minimum=0,
                maximum=10,
            )
            > 1,
        )
    except Exception:
        pass
    history = getattr(vision, "_follow_reading_history", None)
    clear = getattr(history, "clear", None)
    if callable(clear):
        try:
            clear()
            return
        except Exception:
            pass
    try:
        setattr(vision, "_follow_reading_history", deque(maxlen=3))
    except Exception:
        pass


def _send_result_blocked(send_result) -> bool:
    return bool(isinstance(send_result, dict) and send_result.get("blocked"))


def _annotate_reading_with_vision_debug(vision: BrickDetector, reading: dict) -> dict:
    if not isinstance(reading, dict):
        return reading
    debug_fields = {
        "vision_status": "last_status",
        "vision_geometry_source": "last_geometry_source",
        "vision_candidate_count": "last_candidate_count",
        "vision_raw_prediction_count": "last_raw_prediction_count",
        "vision_stable_dist_samples": "last_stable_dist_sample_count",
        "vision_stable_dist_inliers": "last_stable_dist_inlier_count",
        "vision_stable_dist_spread_mm": "last_stable_dist_spread_mm",
    }
    for out_key, attr in debug_fields.items():
        try:
            value = getattr(vision, attr)
        except Exception:
            continue
        reading[out_key] = value
    return reading


def _read_brick_measurement(vision: BrickDetector, *, jump_guard: bool = False) -> dict:
    """Return a fresh brick reading, masking held bricks out of target vision."""
    try:
        result = vision.read()
    except Exception as exc:
        log.warning("Vision read error: %s", exc)
        return brick_motion_measurement_from_result(None)
    reading = brick_motion_measurement_from_result(
        result,
        min_confidence_pct=float(_cautious_visibility_config()["motion_min_confidence_pct"]),
    )
    _annotate_reading_with_vision_debug(vision, reading)
    frame = getattr(vision, "raw_frame", None)
    if _active_game_profile() == "empty":
        reading["holding"] = False
        reading["holding_reason"] = "empty_profile_skips_holding_mask"
        return _apply_temporal_filter_brick_reading(vision, reading, jump_guard=jump_guard)
    holding_result = detect_holding_brick(frame)
    reading["holding"] = bool(holding_result.get("holding"))
    reading["holding_reason"] = holding_result.get("reason")
    if not bool(holding_result.get("holding")):
        return _apply_temporal_filter_brick_reading(vision, reading, jump_guard=jump_guard)
    masked = _mask_held_brick_for_target_frame(frame, holding_result)
    if masked is None:
        reading["target_masked_for_holding"] = False
        return reading
    holding_target_cfg = _holding_target_vision_config()
    contour_result = detect_masked_target_brick_contour(masked, detector=vision)
    if bool(contour_result.get("found")):
        contour_reading = brick_motion_measurement_from_result(
            contour_target_result_tuple(contour_result),
            min_confidence_pct=float(holding_target_cfg["min_confidence_pct"]),
        )
        contour_reading["holding"] = True
        contour_reading["holding_reason"] = holding_result.get("reason")
        contour_reading["target_masked_for_holding"] = True
        contour_reading["holding_target_relaxed_confidence"] = True
        contour_reading["holding_target_contour"] = contour_result
        contour_reading["unmasked_target_reading"] = reading
        contour_reading = _apply_unmasked_stack_xz_if_configured(contour_reading, reading)
        contour_reading = _apply_holding_target_distance_calibration(contour_reading)
        try:
            vision.raw_frame = frame.copy()
        except Exception:
            pass
        return _apply_temporal_filter_brick_reading(vision, contour_reading, jump_guard=jump_guard)

    try:
        set_tuning = getattr(vision, "set_runtime_tuning", None)
        if callable(set_tuning):
            set_tuning(**_holding_target_runtime_tuning())
        masked_result = vision.read_frame(masked)
    except Exception as exc:
        log.warning("Holding-masked target read error: %s", exc)
        reading["target_masked_for_holding"] = False
        return reading
    finally:
        try:
            set_tuning = getattr(vision, "set_runtime_tuning", None)
            if callable(set_tuning):
                set_tuning(**dict(CROWN_PROFILE_TUNING))
        except Exception:
            pass
    masked_reading = brick_motion_measurement_from_result(
        masked_result,
        min_confidence_pct=float(holding_target_cfg["min_confidence_pct"]),
    )
    masked_reading["holding"] = True
    masked_reading["holding_reason"] = holding_result.get("reason")
    masked_reading["target_masked_for_holding"] = True
    masked_reading["holding_target_relaxed_confidence"] = True
    masked_reading["unmasked_target_reading"] = reading
    masked_reading = _apply_unmasked_stack_xz_if_configured(masked_reading, reading)
    masked_reading = _apply_holding_target_distance_calibration(masked_reading)
    if _is_placeholder_reading(masked_reading):
        masked_reading["confident"] = False
        masked_reading["reason"] = "holding_target_placeholder_rejected"
    try:
        vision.raw_frame = frame.copy()
    except Exception:
        pass
    return _apply_temporal_filter_brick_reading(vision, masked_reading, jump_guard=jump_guard)


def _confirm_pickup_suspected_reading(
    vision: BrickDetector,
    reading: dict,
) -> tuple[bool, dict, list[dict]]:
    cfg = _pickup_suspect_config()
    frames = _coerce_int(
        cfg.get("confirm_frames"),
        DEFAULT_PICKUP_SUSPECT_CONFIG["confirm_frames"],
        minimum=1,
        maximum=5,
    )
    poll_s = _coerce_float(
        cfg.get("confirm_poll_s"),
        DEFAULT_PICKUP_SUSPECT_CONFIG["confirm_poll_s"],
        minimum=0.0,
        maximum=1.0,
    )
    samples = [dict(reading)] if isinstance(reading, dict) else []
    current = reading if isinstance(reading, dict) else {}
    while len(samples) < int(frames):
        if poll_s > 0.0:
            time.sleep(float(poll_s))
        current = _read_brick_measurement(vision, jump_guard=True)
        samples.append(dict(current) if isinstance(current, dict) else {})
    suspect_count = sum(1 for sample in samples if _pickup_suspected_reading(sample))
    return bool(suspect_count >= int(frames)), current, samples


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
        "near_target_creep_band_mm": _coerce_float(
            raw.get("near_target_creep_band_mm"),
            DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"],
            minimum=0.0,
        ),
        "near_target_min_pulse_ms": _coerce_int(
            raw.get("near_target_min_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["near_target_min_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "near_target_max_pulse_ms": _coerce_int(
            raw.get("near_target_max_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["near_target_max_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "close_target_creep_band_mm": _coerce_float(
            raw.get("close_target_creep_band_mm"),
            DEFAULT_DIST_APPROACH_POLICY["close_target_creep_band_mm"],
            minimum=0.0,
        ),
        "close_target_max_pulse_ms": _coerce_int(
            raw.get("close_target_max_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["close_target_max_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "very_close_target_creep_band_mm": _coerce_float(
            raw.get("very_close_target_creep_band_mm"),
            DEFAULT_DIST_APPROACH_POLICY["very_close_target_creep_band_mm"],
            minimum=0.0,
        ),
        "very_close_target_max_pulse_ms": _coerce_int(
            raw.get("very_close_target_max_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["very_close_target_max_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "pingpong_confirm_abs_dist_err_mm": _coerce_float(
            raw.get("pingpong_confirm_abs_dist_err_mm"),
            DEFAULT_DIST_APPROACH_POLICY["pingpong_confirm_abs_dist_err_mm"],
            minimum=0.0,
        ),
        "micro_nudge_abs_dist_err_mm": _coerce_float(
            raw.get("micro_nudge_abs_dist_err_mm"),
            DEFAULT_DIST_APPROACH_POLICY["micro_nudge_abs_dist_err_mm"],
            minimum=0.0,
        ),
        "micro_nudge_pulse_ms": _coerce_int(
            raw.get("micro_nudge_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "micro_nudge_min_effective_pulse_ms": _coerce_int(
            raw.get("micro_nudge_min_effective_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["micro_nudge_min_effective_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "min_effective_drive_pulse_ms": _coerce_int(
            raw.get("min_effective_drive_pulse_ms"),
            DEFAULT_DIST_APPROACH_POLICY["min_effective_drive_pulse_ms"],
            minimum=1,
            maximum=_max_act_ms(),
        ),
        "micro_nudge_pwm": _coerce_int(
            raw.get("micro_nudge_pwm"),
            DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pwm"],
            minimum=1,
            maximum=255,
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
    if int(out["near_target_min_pulse_ms"]) > int(out["near_target_max_pulse_ms"]):
        out["near_target_min_pulse_ms"], out["near_target_max_pulse_ms"] = (
            out["near_target_max_pulse_ms"],
            out["near_target_min_pulse_ms"],
        )
    for max_key in ("close_target_max_pulse_ms", "very_close_target_max_pulse_ms"):
        out[max_key] = max(int(out["near_target_min_pulse_ms"]), int(out[max_key]))
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
        "simultaneous_dist_outside_min_mm": _coerce_float(
            raw.get("simultaneous_dist_outside_min_mm"),
            DEFAULT_X_PRIORITY_POLICY["simultaneous_dist_outside_min_mm"],
            minimum=0.0,
        ),
        "simultaneous_x_abs_max_mm": _coerce_float(
            raw.get("simultaneous_x_abs_max_mm"),
            DEFAULT_X_PRIORITY_POLICY["simultaneous_x_abs_max_mm"],
            minimum=0.0,
        ),
        "turn_settle_s": _coerce_float(
            raw.get("turn_settle_s"),
            DEFAULT_X_PRIORITY_POLICY["turn_settle_s"],
            minimum=0.0,
            maximum=2.0,
        ),
        "attach_y_to_turns": bool(
            raw.get(
                "attach_y_to_turns",
                DEFAULT_X_PRIORITY_POLICY["attach_y_to_turns"],
            )
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
        "avoid_forward_left_bias": bool(
            raw.get(
                "avoid_forward_left_bias",
                DEFAULT_X_PRIORITY_POLICY["avoid_forward_left_bias"],
            )
        ),
        "avoid_forward_left_bias_min_x_outside_mm": _coerce_float(
            raw.get("avoid_forward_left_bias_min_x_outside_mm"),
            DEFAULT_X_PRIORITY_POLICY["avoid_forward_left_bias_min_x_outside_mm"],
            minimum=0.0,
        ),
        "avoid_forward_left_bias_min_dist_outside_mm": _coerce_float(
            raw.get("avoid_forward_left_bias_min_dist_outside_mm"),
            DEFAULT_X_PRIORITY_POLICY["avoid_forward_left_bias_min_dist_outside_mm"],
            minimum=0.0,
        ),
        "tiny_x_outside_no_turn_mm": _coerce_float(
            raw.get("tiny_x_outside_no_turn_mm"),
            DEFAULT_X_PRIORITY_POLICY["tiny_x_outside_no_turn_mm"],
            minimum=0.0,
        ),
        "tiny_x_only_backoff_enabled": bool(
            raw.get(
                "tiny_x_only_backoff_enabled",
                DEFAULT_X_PRIORITY_POLICY["tiny_x_only_backoff_enabled"],
            )
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
    out["combined_bias_max_pulse_ms"] = _coerce_int(
        raw.get("combined_bias_max_pulse_ms"),
        DEFAULT_X_DIST_CURVE_POLICY["combined_bias_max_pulse_ms"],
        minimum=0,
        maximum=_max_act_ms(),
    )
    out["large_dist_small_x_strength"] = _coerce_curve_strength(
        raw.get("large_dist_small_x_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["large_dist_small_x_strength"],
    )
    out["combined_x_dist_strength"] = _coerce_curve_strength(
        raw.get("combined_x_dist_strength"),
        DEFAULT_X_DIST_CURVE_POLICY["combined_x_dist_strength"],
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
        "single_win_allowance_s": _coerce_float(
            raw.get("single_win_allowance_s"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["single_win_allowance_s"],
            minimum=0.0,
            maximum=30.0,
        ),
        "active_attempt_budget_s": (
            _coerce_float(
                raw.get("active_attempt_budget_s"),
                0.0,
                minimum=0.0,
                maximum=120.0,
            )
            if "active_attempt_budget_s" in raw
            else None
        ),
        "timeout_y_correction_grace_acts": _coerce_int(
            raw.get("timeout_y_correction_grace_acts"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["timeout_y_correction_grace_acts"],
            minimum=0,
            maximum=10,
        ),
        "min_axis_closeness_pct": _coerce_float(
            raw.get("min_axis_closeness_pct"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["min_axis_closeness_pct"],
            minimum=0.0,
            maximum=100.0,
        ),
        "min_confidence_pct": _coerce_float(
            raw.get("min_confidence_pct"),
            DEFAULT_WIN_CONFIRMATION_CONFIG["min_confidence_pct"],
            minimum=0.0,
            maximum=100.0,
        ),
        "accept_live_happy_after_stop": bool(
            raw.get(
                "accept_live_happy_after_stop",
                DEFAULT_WIN_CONFIRMATION_CONFIG["accept_live_happy_after_stop"],
            )
        ),
    }


def _step_attempt_limit_s() -> float:
    cfg = _win_confirmation_config()
    if cfg.get("active_attempt_budget_s") is not None:
        return _coerce_float(
            cfg.get("active_attempt_budget_s"),
            0.0,
            minimum=0.0,
            maximum=120.0,
        )
    return _coerce_float(
        cfg.get("single_win_allowance_s"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["single_win_allowance_s"],
        minimum=0.0,
        maximum=60.0,
    )


def _cautious_visibility_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("cautious_visibility") if isinstance(cfg.get("cautious_visibility"), dict) else {}
    requested_motion_min = _coerce_float(
        raw.get("motion_min_confidence_pct"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["motion_min_confidence_pct"],
        minimum=0.0,
        maximum=100.0,
    )
    sample_frames = _coerce_int(
        raw.get("pregame_sample_frames"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_sample_frames"],
        minimum=1,
        maximum=60,
    )
    required_frames = _coerce_int(
        raw.get("pregame_required_frames"),
        DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_required_frames"],
        minimum=1,
        maximum=sample_frames,
    )
    return {
        "motion_min_confidence_pct": max(float(requested_motion_min), float(_hard_motion_min_confidence_pct())),
        "pregame_timeout_s": _coerce_float(
            raw.get("pregame_timeout_s"),
            DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_timeout_s"],
            minimum=0.0,
            maximum=30.0,
        ),
        "pregame_sample_s": _coerce_float(
            raw.get("pregame_sample_s"),
            DEFAULT_CAUTIOUS_VISIBILITY_CONFIG["pregame_sample_s"],
            minimum=0.01,
            maximum=2.0,
        ),
        "pregame_sample_frames": sample_frames,
        "pregame_required_frames": required_frames,
    }


def _vision_jump_guard_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("vision_jump_guard") if isinstance(cfg.get("vision_jump_guard"), dict) else {}
    out = dict(DEFAULT_VISION_JUMP_GUARD_CONFIG)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    out["accept_confirmed_jumps"] = bool(raw.get("accept_confirmed_jumps", out["accept_confirmed_jumps"]))
    out["confirm_frames"] = _coerce_int(
        raw.get("confirm_frames"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["confirm_frames"],
        minimum=1,
        maximum=10,
    )
    out["reacquire_frames_after_motion"] = _coerce_int(
        raw.get("reacquire_frames_after_motion"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["reacquire_frames_after_motion"],
        minimum=0,
        maximum=10,
    )
    for key in ("hard_reject_dist_jump_mm", "hard_reject_observe_s", "hard_reject_observe_poll_s", "post_act_stabilize_s"):
        out[key] = _coerce_float(
            raw.get(key),
            DEFAULT_VISION_JUMP_GUARD_CONFIG[key],
            minimum=0.0,
            maximum=500.0 if key.endswith("_mm") else 10.0,
        )
    out["hard_reject_max_recoveries_per_step"] = _coerce_int(
        raw.get("hard_reject_max_recoveries_per_step"),
        DEFAULT_VISION_JUMP_GUARD_CONFIG["hard_reject_max_recoveries_per_step"],
        minimum=0,
        maximum=10,
    )
    for key in (
        "max_dist_jump_mm",
        "max_x_jump_mm",
        "max_y_jump_mm",
        "max_vector_jump_mm",
        "max_abs_dist_mm",
        "edge_recovery_min_abs_x_mm",
        "edge_recovery_max_abs_dist_mm",
        "edge_recovery_min_confidence_pct",
        "confirm_window_mm",
        "reacquire_window_mm",
    ):
        out[key] = _coerce_float(
            raw.get(key),
            DEFAULT_VISION_JUMP_GUARD_CONFIG[key],
            minimum=0.0,
            maximum=500.0,
        )
    return out


def _pickup_suspect_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("pickup_suspect") if isinstance(cfg.get("pickup_suspect"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", DEFAULT_PICKUP_SUSPECT_CONFIG["enabled"])),
        "min_dist_mm": _coerce_float(
            raw.get("min_dist_mm"),
            DEFAULT_PICKUP_SUSPECT_CONFIG["min_dist_mm"],
            minimum=0.0,
            maximum=1000.0,
        ),
        "max_y_mm": _coerce_float(
            raw.get("max_y_mm"),
            DEFAULT_PICKUP_SUSPECT_CONFIG["max_y_mm"],
            minimum=-1000.0,
            maximum=1000.0,
        ),
        "confirm_frames": _coerce_int(
            raw.get("confirm_frames"),
            DEFAULT_PICKUP_SUSPECT_CONFIG["confirm_frames"],
            minimum=1,
            maximum=5,
        ),
        "confirm_poll_s": _coerce_float(
            raw.get("confirm_poll_s"),
            DEFAULT_PICKUP_SUSPECT_CONFIG["confirm_poll_s"],
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _pickup_suspected_reading(reading: dict | None) -> bool:
    cfg = _pickup_suspect_config()
    if not bool(cfg.get("enabled", True)):
        return False
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return False
    try:
        dist_mm = float(reading.get("raw_dist_mm") if reading.get("raw_dist_mm") is not None else reading.get("dist_mm"))
        y_mm = float(reading.get("y_mm"))
    except (TypeError, ValueError):
        return False
    return bool(
        float(dist_mm) >= float(cfg.get("min_dist_mm", DEFAULT_PICKUP_SUSPECT_CONFIG["min_dist_mm"]))
        and float(y_mm) <= float(cfg.get("max_y_mm", DEFAULT_PICKUP_SUSPECT_CONFIG["max_y_mm"]))
    )


def _follow_y_axis_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    out = dict(DEFAULT_FOLLOW_Y_AXIS_CONFIG)
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    for key, fallback in DEFAULT_FOLLOW_Y_AXIS_CONFIG.items():
        if key == "enabled":
            continue
        if key in {"protect_requires_brick_below", "lock_on_enabled", "mast_duration_uses_tolerance_gap"}:
            out[key] = bool(raw.get(key, fallback))
            continue
        if key in {"mast_up_duration_curve", "mast_down_duration_curve"}:
            out[key] = _sanitize_y_duration_curve(raw.get(key, fallback))
            continue
        if key == "hard_floor_y_mm":
            out[key] = _coerce_optional_float(raw.get(key), fallback)
            continue
        minimum = 0.0 if key not in {"win_target_mm", "reset_target_mm"} else None
        out[key] = _coerce_float(raw.get(key), fallback, minimum=minimum)
    return out


def _follow_x_axis_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("x_axis") if isinstance(cfg.get("x_axis"), dict) else {}
    out = dict(DEFAULT_FOLLOW_X_AXIS_CONFIG)
    out["win_target_mm"] = _coerce_float(raw.get("win_target_mm"), out["win_target_mm"])
    out["win_tol_mm"] = _coerce_float(raw.get("win_tol_mm"), out["win_tol_mm"], minimum=0.0)
    positive_turn_cmd = str(raw.get("positive_error_turn_cmd", out["positive_error_turn_cmd"])).strip().lower()
    out["positive_error_turn_cmd"] = positive_turn_cmd if positive_turn_cmd in {"l", "r"} else "r"
    return out


def _follow_dist_axis_config() -> dict:
    cfg = _follow_motion_config()
    raw = cfg.get("dist_axis") if isinstance(cfg.get("dist_axis"), dict) else {}
    out = dict(DEFAULT_FOLLOW_DIST_AXIS_CONFIG)
    out["win_target_mm"] = _coerce_float(raw.get("win_target_mm"), out["win_target_mm"], minimum=0.0)
    out["win_tol_mm"] = _coerce_float(raw.get("win_tol_mm"), out["win_tol_mm"], minimum=0.0)
    positive_cmd = str(raw.get("positive_error_cmd", out.get("positive_error_cmd", "f"))).strip().lower()
    out["positive_error_cmd"] = positive_cmd if positive_cmd in {"f", "b"} else "f"
    return out


def _dist_target_mm() -> float:
    return float(_follow_dist_axis_config().get("win_target_mm", TARGET_DIST_MM))


def _dist_tol_mm() -> float:
    return float(_follow_dist_axis_config().get("win_tol_mm", DIST_TOL_MM))


def _dist_positive_error_cmd() -> str:
    cmd = str(_follow_dist_axis_config().get("positive_error_cmd", "f")).strip().lower()
    return cmd if cmd in {"f", "b"} else "f"


def _dist_cmd_for_error(dist_err: float) -> str:
    positive_cmd = _dist_positive_error_cmd()
    negative_cmd = "b" if positive_cmd == "f" else "f"
    return positive_cmd if float(dist_err) >= 0.0 else negative_cmd


def _drive_mode_for_dist_error(dist_err: float) -> str:
    return "forward" if _dist_cmd_for_error(dist_err) == "f" else "backward"


def _x_target_mm() -> float:
    return float(_follow_x_axis_config().get("win_target_mm", X_TARGET_MM))


def _x_tol_mm() -> float:
    return float(_follow_x_axis_config().get("win_tol_mm", X_TOL_MM))


def _y_win_target_mm() -> float:
    return float(_follow_y_axis_config().get("win_target_mm", Y_TARGET_MM))


def _y_win_tol_mm() -> float:
    return float(_follow_y_axis_config().get("win_tol_mm", Y_TOL_MM))


def _x_err_for_reading(reading: dict | None, *, target: float | None = None) -> float | None:
    try:
        x_mm = float((reading or {}).get("x_mm"))
    except (TypeError, ValueError):
        return None
    target_val = float(_x_target_mm() if target is None else target)
    return float(x_mm - target_val)


def _x_outside_gate_mm(x_err: float) -> float:
    return max(0.0, abs(float(x_err)) - float(_x_tol_mm()))


def _dist_outside_gate_mm(dist_err: float) -> float:
    return max(0.0, abs(float(dist_err)) - float(_dist_tol_mm()))


def _y_err_for_reading(reading: dict, *, target: float | None = None) -> float | None:
    try:
        y_mm = float((reading or {}).get("y_mm"))
        dist_mm = float((reading or {}).get("dist_mm"))
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


def _fmt_signed_mm(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):+.1f}mm"


def _fmt_gate_target(target: float | None, tol: float | None, *, signed: bool = False) -> str:
    target_text = _fmt_signed_mm(target) if signed else _fmt_mm(target)
    tol_text = _fmt_mm(tol)
    return f"{target_text} +/- {tol_text}"


def _step2_gate_text(step2_cfg: dict | None = None) -> str:
    cfg = step2_cfg if isinstance(step2_cfg, dict) else _follow_step2_config()
    targets = cfg.get("targets") if isinstance(cfg.get("targets"), dict) else {}
    missing = _step2_missing_target_keys(cfg)
    if missing:
        return "pending targets: " + ", ".join(missing)
    parts = []
    for label, value_key, tol_key in _configured_step2_target_axes(cfg):
        parts.append(
            f"{label}={_fmt_gate_target(targets.get(value_key), targets.get(tol_key), signed=(label != 'dist'))}"
        )
    return ", ".join(parts) if parts else "pending targets: targets"


def _step2_semi_happy_text(step2_cfg: dict | None = None) -> str:
    cfg = step2_cfg if isinstance(step2_cfg, dict) else _follow_step2_config()
    targets = cfg.get("semi_happy_targets") if isinstance(cfg.get("semi_happy_targets"), dict) else {}
    if not targets:
        return "none"
    return (
        f"dist={_fmt_gate_target(targets.get('dist_mm'), targets.get('dist_tol_mm'))}, "
        f"x={_fmt_gate_target(targets.get('x_mm'), targets.get('x_tol_mm'), signed=True)}, "
        f"y={_fmt_gate_target(targets.get('y_mm'), targets.get('y_tol_mm'), signed=True)}"
    )


def _step3_gate_text(step3_cfg: dict | None = None) -> str:
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step4_config()
    targets = cfg.get("targets") if isinstance(cfg.get("targets"), dict) else {}
    missing = _step3_missing_target_keys(cfg)
    if missing:
        return "pending targets: " + ", ".join(missing)
    return f"y={_fmt_gate_target(targets.get('y_mm'), targets.get('y_tol_mm'), signed=True)}"


def _step3_retreat_gate_text(step3_cfg: dict | None = None) -> str:
    cfg = step3_cfg if isinstance(step3_cfg, dict) else _follow_step3_config()
    target = _step3_retreat_target_dist_mm(cfg)
    max_ms = _coerce_int(
        cfg.get("max_duration_ms"),
        DEFAULT_STEP3_RETREAT_CONFIG["max_duration_ms"],
        minimum=1,
        maximum=5000,
    )
    cmd = str(cfg.get("drive_cmd") or DEFAULT_STEP3_RETREAT_CONFIG["drive_cmd"]).strip().upper()
    if cmd not in {"F", "B"}:
        cmd = DEFAULT_STEP3_RETREAT_CONFIG["drive_cmd"].upper()
    comparator = ">=" if bool(cfg.get("stop_when_dist_at_or_beyond", True)) else "="
    return f"straight {cmd} until dist{comparator}{target:.1f}mm or {max_ms}ms"


def _success_gate_summary_lines() -> list[str]:
    profile = _active_game_profile()
    win_cfg = _win_confirmation_config()
    y_cfg = _follow_y_axis_config()
    step2 = _follow_step2_config()
    step3 = _follow_step3_config()
    step4 = _follow_step4_config()
    reset_raw = _reset_motion_config().get("reverse_turn")
    reset_cfg = reset_raw if isinstance(reset_raw, dict) else {}
    reset_x_min = _coerce_float(reset_cfg.get("x_offset_min_mm"), RESET_X_OFFSET_MIN_MM, minimum=0.0)
    reset_x_max = _coerce_float(reset_cfg.get("x_offset_max_mm"), RESET_X_OFFSET_MAX_MM, minimum=0.0)
    if reset_x_min > reset_x_max:
        reset_x_min, reset_x_max = reset_x_max, reset_x_min
    reset_target_abs_x = _coerce_float(
        reset_cfg.get("target_abs_x_mm"),
        (float(reset_x_min) + float(reset_x_max)) / 2.0,
        minimum=0.0,
    )
    if reset_target_abs_x < reset_x_min or reset_target_abs_x > reset_x_max:
        reset_target_abs_x = (float(reset_x_min) + float(reset_x_max)) / 2.0
    y_gate = (
        f"y={_fmt_gate_target(y_cfg.get('win_target_mm'), _win_effective_tolerance(y_cfg.get('win_tol_mm')), signed=True)}"
        if bool(y_cfg.get("enabled"))
        else "y=disabled"
    )
    lock_on_ms = int(
        _coerce_int(
            y_cfg.get("lock_on_pulse_ms"),
            DEFAULT_FOLLOW_Y_AXIS_CONFIG["lock_on_pulse_ms"],
            minimum=1,
            maximum=_max_mast_act_ms(),
        )
    )
    finish_mast_ms = int(
        _coerce_int(
            y_cfg.get("finish_mast_pulse_ms"),
            DEFAULT_FOLLOW_Y_AXIS_CONFIG["finish_mast_pulse_ms"],
            minimum=1,
            maximum=_max_mast_act_ms(),
        )
    )
    y_commit_gate = (
        (
            "[GATES] Step 1 / Y-COMMIT: "
            f"after lock-on lower D for at least {lock_on_ms}ms, freeze x/z and resolve "
            f"{y_gate} with finish mast pulses of {finish_mast_ms}ms"
        )
        if bool(y_cfg.get("enabled")) and bool(y_cfg.get("lock_on_enabled", True))
        else "[GATES] Step 1 / Y-COMMIT: disabled"
    )
    step1_attempt_limit_s = float(_step_attempt_limit_s())
    active_budget_text = (
        "game timer only"
        if step1_attempt_limit_s <= 0.0
        else f"{step1_attempt_limit_s:.1f}s"
    )
    fallback_text = ""
    if bool(step4.get("no_visibility_fallback_enabled")):
        fallback_text = (
            f"; no-visibility fallback can unlock reset with "
            f"{str(step4.get('no_visibility_fallback_mast_cmd', 'u')).upper()} "
            f"{int(step4.get('no_visibility_fallback_duration_ms', 0) or 0)}ms"
        )
    step2_label = str(step2.get("nickname") or "align y").strip().upper()
    lines = [
        f"[GATES] Active a_follow_the_brick.py success gates for {profile} game:",
        (
            "[GATES] Step 1 / HAPPY: visible+confident, "
            f"conf>={float(win_cfg.get('min_confidence_pct', 0.0)):.0f}%, "
            f"dist={_fmt_gate_target(_dist_target_mm(), _win_effective_tolerance(_dist_tol_mm()))}, "
            f"x={_fmt_gate_target(_x_target_mm(), _win_effective_tolerance(_x_tol_mm()), signed=True)}, "
            f"{y_gate}; stopped confirmation="
            f"{int(win_cfg.get('confirm_frames', 1))} stopped happy hit(s) within "
            f"{float(win_cfg.get('single_win_allowance_s', 0.0)):.1f}s after "
            f"{float(win_cfg.get('settle_s', 0.0)):.2f}s; "
            f"active attempt budget={active_budget_text}"
        ),
        y_commit_gate,
        (
            "[GATES] Step 1 Reset / BACK_TURN_LR: "
            f"dist={_fmt_gate_target(reset_cfg.get('dist_target_mm'), reset_cfg.get('dist_tol_mm'))}, "
            f"min_attempt_dist={_reset_min_attempt_dist_mm(reset_cfg):.1f}mm, "
            f"|x|={reset_x_min:.1f}-{reset_x_max:.1f}mm "
            f"(target {reset_target_abs_x:.1f}mm); "
            f"y={_fmt_gate_target(reset_cfg.get('y_target_mm'), reset_cfg.get('y_tol_mm'), signed=True)} "
            "is logged only, not a reset hit gate"
        ),
        (
            f"[GATES] Step 2 / {step2_label}: "
            f"{_step2_gate_text(step2)}; semi-happy={_step2_semi_happy_text(step2)}; "
            f"freeze_xz_after_xz_target={bool(step2.get('freeze_xz_after_xz_target'))}"
        ),
    ]
    if profile == "holding":
        lines.insert(
            1,
            "[GATES] Holding reminder: reset into holding and holding Step 1 leave mast/y alone; Step 1 judges dist+x only.",
        )
    if _game_complete_after_step2():
        if _step3_kind(step3) == "retreat":
            step3_label = str(step3.get("nickname") or "retreat").strip().upper()
            lines.append(
                f"[GATES] Step 3 / {step3_label}: "
                f"{_step3_retreat_gate_text(step3)}; then reset, then next game profile is empty."
            )
            return lines
        lines.append("[GATES] After Step 2 / PLACE: reset, then next game profile is empty.")
        return lines
    lines.extend([
        (
            "[GATES] Step 3 / SEAT: "
            f"{_step2_gate_text(step3)}; semi-happy={_step2_semi_happy_text(step3)}; "
            f"freeze_xz_after_xz_target={bool(step3.get('freeze_xz_after_xz_target'))}"
        ),
        (
            "[GATES] Step 4 / LIFT: "
            f"{_step3_gate_text(step4)}; mast_cmd="
            f"{str(step4.get('lift_mast_cmd', 'u')).upper()}{fallback_text}"
        ),
    ])
    return lines


def _print_active_success_gates() -> None:
    for line in _success_gate_summary_lines():
        print(line, flush=True)


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
    combined_strength = str(policy.get("combined_x_dist_strength") or "").strip().lower()
    if combined_strength in {"adaptive", "gentle", "medium", "strong"}:
        return combined_strength
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


def _cap_near_target_distance_duration_ms(duration_ms: int, abs_dist_err_mm: float, policy: dict) -> int:
    duration = _bounded_act_duration_ms(duration_ms)
    for band_key, max_key in (
        ("close_target_creep_band_mm", "close_target_max_pulse_ms"),
        ("very_close_target_creep_band_mm", "very_close_target_max_pulse_ms"),
    ):
        band_mm = _coerce_float(
            policy.get(band_key),
            DEFAULT_DIST_APPROACH_POLICY[band_key],
            minimum=0.0,
        )
        if band_mm <= 0.0 or float(abs_dist_err_mm) > float(band_mm):
            continue
        cap_ms = _coerce_int(
            policy.get(max_key),
            DEFAULT_DIST_APPROACH_POLICY[max_key],
            minimum=1,
            maximum=_max_act_ms(),
        )
        duration = min(int(duration), max(int(policy.get("near_target_min_pulse_ms", PULSE_MS)), int(cap_ms)))
    return int(duration)


def _near_target_distance_creep_duration_ms(dist_err: float, policy: dict | None = None) -> int | None:
    cfg = policy if isinstance(policy, dict) else _follow_dist_approach_policy()
    band_mm = _coerce_float(
        cfg.get("near_target_creep_band_mm"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"],
        minimum=0.0,
    )
    if band_mm <= 0.0:
        return None
    abs_err = abs(float(dist_err))
    if abs_err > float(band_mm):
        return None
    duration = _proportional_duration_ms(
        gap_mm=abs_err,
        min_ms=int(cfg.get("near_target_min_pulse_ms", DEFAULT_DIST_APPROACH_POLICY["near_target_min_pulse_ms"])),
        max_ms=int(cfg.get("near_target_max_pulse_ms", DEFAULT_DIST_APPROACH_POLICY["near_target_max_pulse_ms"])),
        full_gap_mm=float(band_mm),
    )
    return _cap_near_target_distance_duration_ms(duration, abs_err, cfg)


def _distance_creep_duration_ms(dist_err: float) -> int:
    policy = _follow_dist_approach_policy()
    cmd = _dist_cmd_for_error(dist_err)
    floor_ms = int(_min_effective_drive_duration_ms(cmd))
    near_duration = _near_target_distance_creep_duration_ms(dist_err, policy)
    if near_duration is not None:
        return int(max(int(near_duration), int(floor_ms)))
    near_band = float(policy.get("near_target_creep_band_mm", DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"]))
    near_max_ms = int(policy.get("near_target_max_pulse_ms", DEFAULT_DIST_APPROACH_POLICY["near_target_max_pulse_ms"]))
    full_gap = max(
        0.1,
        float(policy.get("full_forward_gap_mm", DEFAULT_DIST_APPROACH_POLICY["full_forward_gap_mm"])) - float(near_band),
    )
    gap = max(0.0, abs(float(dist_err)) - float(near_band))
    duration = _proportional_duration_ms(
        gap_mm=gap,
        min_ms=near_max_ms,
        max_ms=int(policy.get("max_forward_pulse_ms", PULSE_MS)),
        full_gap_mm=full_gap,
    )
    return int(max(int(duration), int(floor_ms)))


def _min_motion_duration_ms(cmd: str | None) -> int:
    cmd_key = str(cmd or "").strip().lower()
    try:
        _, duration_ms = _speed_pwm_duration(cmd_key, _telemetry_robot.SPEED_SCORE_MIN)
    except Exception:
        return int(PULSE_MS)
    return int(max(1, int(duration_ms)))


def _min_effective_drive_duration_ms(cmd: str | None) -> int:
    policy = _follow_dist_approach_policy()
    configured_floor = _coerce_int(
        policy.get("min_effective_drive_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["min_effective_drive_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    return int(max(int(configured_floor), int(_min_motion_duration_ms(cmd))))


def _combined_drive_bias_duration_ms(dist_err: float, *, drive_mode: str = "forward") -> int:
    duration = int(_distance_creep_duration_ms(dist_err))
    logical_cmd = "b" if str(drive_mode or "").strip().lower() == "backward" else "f"
    floor_ms = int(_min_effective_drive_duration_ms(logical_cmd))
    cap_ms = _coerce_int(
        _follow_x_dist_curve_policy().get("combined_bias_max_pulse_ms"),
        DEFAULT_X_DIST_CURVE_POLICY["combined_bias_max_pulse_ms"],
        minimum=0,
        maximum=_max_act_ms(),
    )
    if cap_ms <= 0:
        return int(max(int(duration), int(floor_ms)))
    return int(max(int(floor_ms), min(int(duration), int(cap_ms))))


def _distance_correction_duration_ms(dist_err: float) -> int:
    policy = _follow_dist_approach_policy()
    cmd = _dist_cmd_for_error(dist_err)
    floor_ms = int(_min_effective_drive_duration_ms(cmd))
    near_duration = _near_target_distance_creep_duration_ms(dist_err, policy)
    if near_duration is not None:
        return int(max(int(near_duration), int(floor_ms)))
    near_band = float(policy.get("near_target_creep_band_mm", DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"]))
    near_max_ms = int(policy.get("near_target_max_pulse_ms", DEFAULT_DIST_APPROACH_POLICY["near_target_max_pulse_ms"]))
    full_gap = max(
        0.1,
        float(policy.get("full_forward_gap_mm", DEFAULT_DIST_APPROACH_POLICY["full_forward_gap_mm"])) - float(near_band),
    )
    gap = max(0.0, abs(float(dist_err)) - float(near_band))
    duration = _proportional_duration_ms(
        gap_mm=gap,
        min_ms=near_max_ms,
        max_ms=int(policy.get("max_forward_pulse_ms", DEFAULT_DIST_APPROACH_POLICY["max_forward_pulse_ms"])),
        full_gap_mm=full_gap,
    )
    return int(max(int(duration), int(floor_ms)))


def _too_close_escape_duration_ms(dist_err: float, escape_policy: dict | None = None) -> int:
    policy = escape_policy if isinstance(escape_policy, dict) else _too_close_escape_policy()
    gap = max(0.0, abs(float(dist_err)) - _win_effective_tolerance(_dist_tol_mm()))
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
    if float(dist_err) > _win_effective_tolerance(_dist_tol_mm()):
        dist_outside = _dist_outside_gate_mm(float(dist_err))
        simultaneous_dist_min = float(
            policy.get(
                "simultaneous_dist_outside_min_mm",
                DEFAULT_X_PRIORITY_POLICY["simultaneous_dist_outside_min_mm"],
            )
        )
        simultaneous_x_max = float(
            policy.get(
                "simultaneous_x_abs_max_mm",
                DEFAULT_X_PRIORITY_POLICY["simultaneous_x_abs_max_mm"],
            )
        )
        if dist_outside >= simultaneous_dist_min and (
            simultaneous_x_max <= 0.0 or abs_x <= simultaneous_x_max
        ):
            return False
    if (
        float(dist_err) >= float(policy.get("huge_dist_gap_mm", DEFAULT_X_PRIORITY_POLICY["huge_dist_gap_mm"]))
        and abs_x <= float(policy.get("huge_dist_tiny_abs_x_mm", DEFAULT_X_PRIORITY_POLICY["huge_dist_tiny_abs_x_mm"]))
    ):
        return False
    return True


def _tiny_x_outside_no_turn(x_outside_mm: float) -> bool:
    policy = _follow_x_priority_policy()
    threshold = _coerce_float(
        policy.get("tiny_x_outside_no_turn_mm"),
        DEFAULT_X_PRIORITY_POLICY["tiny_x_outside_no_turn_mm"],
        minimum=0.0,
    )
    return bool(threshold > 0.0 and 0.0 < float(x_outside_mm) <= float(threshold))


def _near_target_forward_veto_active(*, dist_err: float, x_ok: bool, y_ok: bool) -> bool:
    policy = _follow_dist_approach_policy()
    veto_mm = float(policy.get("near_target_forward_veto_mm", 0.0))
    if veto_mm <= 0.0:
        return False
    if bool(x_ok) and bool(y_ok):
        return False
    return _win_effective_tolerance(_dist_tol_mm()) < float(dist_err) <= float(veto_mm)


def _distance_micro_nudge_plan(reading: dict, *, dist_err: float, x_err: float, y_plan: dict | None) -> dict | None:
    policy = _follow_dist_approach_policy()
    band_mm = _coerce_float(
        policy.get("micro_nudge_abs_dist_err_mm"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_abs_dist_err_mm"],
        minimum=0.0,
    )
    if band_mm <= 0.0:
        return None
    if abs(float(dist_err)) <= _win_effective_tolerance(_dist_tol_mm()):
        return None
    if abs(float(dist_err)) > float(band_mm):
        return None
    if not _win_axis_ok(float(x_err), _x_tol_mm()):
        return None
    cmd = "b" if float(dist_err) < 0.0 else "f"
    turn_cmd = _turn_cmd_to_close_x_gap(float(x_err))
    if turn_cmd not in {"l", "r"}:
        turn_cmd = "l" if float(x_err) <= 0.0 else "r"
    duration_ms = _coerce_int(
        policy.get("micro_nudge_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    pwm = _coerce_int(
        policy.get("micro_nudge_pwm"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pwm"],
        minimum=1,
        maximum=255,
    )
    micro_floor_ms = _coerce_int(
        policy.get("micro_nudge_min_effective_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_min_effective_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    duration_ms = max(int(duration_ms), int(micro_floor_ms), int(_min_motion_duration_ms(cmd)))
    pwm = max(int(pwm), int(_pwm_floor_for_cmd(cmd)))
    action_prefix = "BCK" if cmd == "b" else "FWD"
    if _win_axis_ok(float(x_err), _x_tol_mm()):
        return _attach_mast_to_plan({
            "kind": "drive",
            "cmd": cmd,
            "action": f"{action_prefix}_MICRO",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "duration_ms": int(duration_ms),
            "pwm": int(pwm),
            "distance_creep": True,
            "reason": "dist_micro_straight",
        }, y_plan)
    return _attach_mast_to_plan({
        "kind": "drive_nudge",
        "cmd": cmd,
        "turn_cmd": turn_cmd,
        "drive_mode": "backward" if cmd == "b" else "forward",
        "action": f"{action_prefix}_NUDGE_{turn_cmd.upper()}",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "duration_ms": int(duration_ms),
        "pwm": int(pwm),
        "distance_creep": True,
        "reason": "dist_micro_nudge",
    }, y_plan)


def _distance_micro_straight_plan(reading: dict, *, dist_err: float, x_err: float, y_plan: dict | None) -> dict | None:
    policy = _follow_dist_approach_policy()
    band_mm = _coerce_float(
        policy.get("micro_nudge_abs_dist_err_mm"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_abs_dist_err_mm"],
        minimum=0.0,
    )
    if band_mm <= 0.0 or abs(float(dist_err)) > float(band_mm):
        return None
    cmd = "b" if float(dist_err) < 0.0 else "f"
    duration_ms = _coerce_int(
        policy.get("micro_nudge_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    micro_floor_ms = _coerce_int(
        policy.get("micro_nudge_min_effective_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["micro_nudge_min_effective_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    duration_ms = max(int(duration_ms), int(micro_floor_ms), int(_min_motion_duration_ms(cmd)))
    action_prefix = "BCK" if cmd == "b" else "FWD"
    return _attach_mast_to_plan({
        "kind": "drive",
        "cmd": cmd,
        "action": f"{action_prefix}_MICRO",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "duration_ms": int(duration_ms),
        "pwm": max(
            int(policy.get("micro_nudge_pwm", DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pwm"])),
            int(_pwm_floor_for_cmd(cmd)),
        ),
        "distance_creep": True,
        "reason": "tiny_x_dist_micro_straight",
    }, y_plan)


def _finish_y_dist_ok(dist_err: float, y_cfg: dict | None = None) -> bool:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    dist_fallback = _coerce_float(
        cfg.get("finish_y_only_dist_deadband_mm"),
        _dist_tol_mm(),
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
    if str(y_plan.get("reason") or "") == "y_min_act_would_overshoot":
        return False
    y_cfg = _follow_y_axis_config()
    x_deadband = _coerce_float(
        y_cfg.get("finish_y_only_x_deadband_mm"),
        _x_tol_mm(),
        minimum=0.0,
    )
    return _finish_y_dist_ok(dist_err, y_cfg) and abs(float(x_err)) <= float(x_deadband)


def _y_win_gate_state(reading: dict) -> tuple[bool, float | None]:
    y_cfg = _follow_y_axis_config()
    if not bool(y_cfg.get("enabled")):
        return True, None
    try:
        target = float(y_cfg.get("win_target_mm", Y_TARGET_MM))
        tol = float(y_cfg.get("win_tol_mm", Y_TOL_MM))
    except (TypeError, ValueError):
        return False, None
    y_err = _y_err_for_reading(reading, target=target)
    if y_err is None:
        return False, None
    return bool(_win_axis_ok(float(y_err), tol)), float(y_err)


def _y_commit_action_plan(reading: dict, *, dist_err: float, x_err: float) -> dict:
    y_cfg = _follow_y_axis_config()
    y_ok, y_err = _y_win_gate_state(reading)
    target = float(y_cfg.get("win_target_mm", Y_TARGET_MM))
    if y_ok:
        return {
            "kind": "hold",
            "action": "Y_COMMIT_TARGET_HIT",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": y_err,
            "y_target_mm": float(target),
            "reason": "y_commit_target_hit",
        }
    if y_err is None:
        return {
            "kind": "wait",
            "action": "Y_COMMIT_WAIT_Y",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": None,
            "y_target_mm": float(target),
            "reason": "missing_y_after_commit",
    }
    cmd = "u" if float(y_err) > 0.0 else "d"
    duration_ms = _adaptive_y_mast_duration_ms(float(y_err), y_cfg, near_end=True, cmd=cmd)
    return {
        "kind": "mast",
        "cmd": cmd,
        "action": f"Y_COMMIT_MAST_{cmd.upper()}",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "y_err": float(y_err),
        "y_mm": (reading or {}).get("y_mm"),
        "y_target_mm": float(target),
        "pwm": _y_mast_pwm(y_cfg, cmd=cmd, near_end=True),
        "duration_ms": int(duration_ms),
        "reason": "y_commit_final_y",
    }


def _y_mast_rate_mm_per_100ms(y_cfg: dict | None, *, cmd: str | None) -> float:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    cmd_key = str(cmd or "").strip().lower()
    direction_key = "mast_down_mm_per_100ms" if cmd_key == "d" else "mast_up_mm_per_100ms"
    fallback = cfg.get("mast_mm_per_100ms", DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_mm_per_100ms"])
    return _coerce_float(
        cfg.get(direction_key),
        fallback,
        minimum=0.1,
        maximum=100.0,
    )


def _y_mast_pwm(y_cfg: dict | None, *, cmd: str | None, near_end: bool) -> int:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    cmd_key = str(cmd or "").strip().lower()
    prefix = "finish_mast" if bool(near_end) else "mast"
    direction_key = f"{prefix}_{'down' if cmd_key == 'd' else 'up'}_pwm"
    fallback_key = f"{prefix}_pwm"
    fallback = cfg.get(fallback_key, cfg.get("mast_pwm", 40))
    return int(_coerce_int(cfg.get(direction_key), fallback, minimum=1, maximum=255))


def _duration_from_y_curve(y_cfg: dict, *, cmd: str | None, closes_mm: float, min_ms: int, max_ms: int) -> int | None:
    cmd_key = str(cmd or "").strip().lower()
    curve_key = "mast_up_duration_curve" if cmd_key == "u" else "mast_down_duration_curve"
    curve = _sanitize_y_duration_curve(y_cfg.get(curve_key))
    if not curve:
        return None
    target_mm = max(0.0, float(closes_mm))
    if target_mm <= 0.0:
        return None
    if target_mm <= float(curve[0]["closes_mm"]):
        return max(int(min_ms), min(int(max_ms), int(curve[0]["duration_ms"])))
    for left, right in zip(curve, curve[1:]):
        left_mm = float(left["closes_mm"])
        right_mm = float(right["closes_mm"])
        if target_mm > right_mm:
            continue
        span_mm = max(0.001, right_mm - left_mm)
        ratio = (target_mm - left_mm) / span_mm
        duration = int(round(float(left["duration_ms"]) + ratio * (float(right["duration_ms"]) - float(left["duration_ms"]))))
        return max(int(min_ms), min(int(max_ms), int(duration)))
    return max(int(min_ms), min(int(max_ms), int(curve[-1]["duration_ms"])))


def _min_y_curve_closes_mm(y_cfg: dict | None, *, cmd: str | None) -> float | None:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    cmd_key = str(cmd or "").strip().lower()
    curve_key = "mast_up_duration_curve" if cmd_key == "u" else "mast_down_duration_curve"
    curve = _sanitize_y_duration_curve(cfg.get(curve_key))
    if not curve:
        return None
    try:
        return float(curve[0]["closes_mm"])
    except (TypeError, ValueError, KeyError):
        return None


def _adaptive_y_mast_duration_ms(
    y_err: float,
    y_cfg: dict | None = None,
    *,
    near_end: bool,
    cmd: str | None = None,
    duration_err: float | None = None,
) -> int:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    prefix = "finish_mast" if bool(near_end) else "mast"
    fallback_ms = cfg.get(f"{prefix}_pulse_ms", cfg.get("mast_pulse_ms", 220))
    min_ms = _coerce_int(
        cfg.get(f"{prefix}_min_pulse_ms"),
        min(int(fallback_ms), 80 if near_end else 100),
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    max_ms = _coerce_int(
        cfg.get(f"{prefix}_max_pulse_ms"),
        fallback_ms,
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    if min_ms > max_ms:
        min_ms, max_ms = max_ms, min_ms
    mm_per_100ms = _y_mast_rate_mm_per_100ms(cfg, cmd=cmd)
    correction_fraction = _coerce_float(
        cfg.get("mast_correction_fraction"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_correction_fraction"],
        minimum=0.05,
        maximum=1.0,
    )
    error_for_duration = abs(float(y_err))
    if duration_err is not None:
        try:
            error_for_duration = max(0.0, abs(float(duration_err)))
        except (TypeError, ValueError):
            error_for_duration = abs(float(y_err))
    curve_ms = _duration_from_y_curve(
        cfg,
        cmd=cmd,
        closes_mm=float(error_for_duration),
        min_ms=int(min_ms),
        max_ms=int(max_ms),
    )
    if curve_ms is not None:
        return max(1, min(_max_mast_act_ms(), int(max_ms), int(curve_ms)))
    desired_ms = (float(error_for_duration) * float(correction_fraction) * 100.0) / float(mm_per_100ms)
    clamped = max(int(min_ms), min(int(max_ms), int(round(desired_ms))))
    return max(1, min(_max_mast_act_ms(), int(max_ms), int(clamped)))


def _y_motion_coast_settle_s(y_cfg: dict | None = None) -> float:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    return _coerce_float(
        cfg.get("mast_coast_settle_s"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_coast_settle_s"],
        minimum=0.0,
        maximum=1.0,
    )


def _tiny_y_no_observed_wait_s(y_cfg: dict | None = None) -> float:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    return _coerce_float(
        cfg.get("tiny_y_no_observed_wait_s"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["tiny_y_no_observed_wait_s"],
        minimum=0.0,
        maximum=1.0,
    )


def _should_wait_after_tiny_y_no_observed(detail: dict | None, y_cfg: dict | None = None) -> bool:
    if not isinstance(detail, dict) or bool(detail.get("observed")):
        return False
    if not _pending_action_is_y_action(str(detail.get("action") or "")):
        return False
    try:
        y_err = abs(float(detail.get("y_err")))
    except (TypeError, ValueError):
        return False
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    threshold = _coerce_float(
        cfg.get("tiny_y_no_observed_abs_err_mm"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["tiny_y_no_observed_abs_err_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    return threshold > 0.0 and y_err <= threshold


def _plan_commits_y_lowering(plan: dict | None) -> bool:
    if not isinstance(plan, dict):
        return False
    if str(plan.get("kind") or "").strip().lower() != "mast":
        return False
    if str(plan.get("cmd") or "").strip().lower() != "d":
        return False
    return str(plan.get("reason") or "").strip() == "final_y"


def _y_axis_action_plan(reading: dict, *, dist_err: float, x_err: float) -> dict | None:
    y_cfg = _follow_y_axis_config()
    if not bool(y_cfg.get("enabled")):
        return None
    try:
        y_mm = float((reading or {}).get("y_mm"))
        dist_mm = float((reading or {}).get("dist_mm"))
    except (TypeError, ValueError):
        return None
    target = float(y_cfg.get("win_target_mm", Y_TARGET_MM))
    tol = float(y_cfg.get("win_tol_mm", Y_TOL_MM))
    high_target = target * float(y_cfg.get("approach_high_factor", 1.3))
    protect_below = float(y_cfg.get("protect_below_y_mm", max(high_target + tol, target + (3.0 * tol))))
    inside_step1_x_dist_gate = _win_axis_ok(float(dist_err), _dist_tol_mm()) and _win_axis_ok(
        float(x_err),
        _x_tol_mm(),
    )
    near_end = bool(inside_step1_x_dist_gate) or (
        abs(float(dist_err)) <= float(y_cfg.get("endgame_dist_tol_mm", _dist_tol_mm()))
        and abs(float(x_err)) <= float(y_cfg.get("endgame_x_tol_mm", _x_tol_mm()))
    ) or (
        _finish_y_dist_ok(dist_err, y_cfg)
        and abs(float(x_err)) <= float(y_cfg.get("finish_y_only_x_deadband_mm", _x_tol_mm()))
    )
    brick_below = bool((reading or {}).get("brick_below"))
    protect_requires_below = bool(y_cfg.get("protect_requires_brick_below", True))
    protect_dist_limit = _coerce_float(
        y_cfg.get("protect_disabled_within_dist_mm"),
        177.0,
        minimum=0.0,
    )
    protect_allowed_by_dist = protect_dist_limit <= 0.0 or float(dist_mm) > float(protect_dist_limit)
    protect_triggered = (
        (brick_below or ((not protect_requires_below) and y_mm > protect_below))
        and bool(protect_allowed_by_dist)
    )
    if protect_triggered:
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
    min_effect_mm = _coerce_float(
        y_cfg.get("mast_min_effect_mm"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_min_effect_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    overshoot_margin_mm = _coerce_float(
        y_cfg.get("mast_overshoot_guard_margin_mm"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_overshoot_guard_margin_mm"],
        minimum=0.0,
        maximum=100.0,
    )
    gap_to_happy_edge = max(0.0, abs(float(y_err)) - float(active_tol))
    if min_effect_mm > 0.0 and gap_to_happy_edge <= (float(min_effect_mm) + float(overshoot_margin_mm)):
        return {
            "kind": "wait",
            "action": "Y_OVERSHOOT_GUARD_WAIT",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": float(y_err),
            "y_mm": float(y_mm),
            "y_target_mm": float(active_target),
            "duration_ms": 0,
            "reason": "y_min_act_would_overshoot",
        }
    cmd = "d" if y_err > 0.0 else "u"
    if cmd == "u" and _step1_mast_up_budget_ms(y_cfg) <= 0:
        return None
    # Hard ceiling: never raise y above the win zone (win target + win tol). There is
    # no reason to take y higher than the step-1 win, and over-raising wrecks the pose.
    win_ceiling_y = float(target) + float(tol)
    if cmd == "u" and float(y_mm) >= win_ceiling_y:
        return {
            "kind": "wait",
            "action": "Y_HARD_CEILING_WAIT",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": float(y_err),
            "y_mm": float(y_mm),
            "y_target_mm": float(active_target),
            "duration_ms": 0,
            "reason": "y_hard_ceiling",
        }
    hard_floor_raw = y_cfg.get("hard_floor_y_mm")
    hard_floor_y = None
    try:
        if hard_floor_raw is not None:
            hard_floor_y = float(hard_floor_raw)
    except (TypeError, ValueError):
        hard_floor_y = None
    if hard_floor_y is not None and cmd == "d" and float(y_mm) <= hard_floor_y:
        return {
            "kind": "wait",
            "action": "Y_HARD_FLOOR_WAIT",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": float(y_err),
            "y_mm": float(y_mm),
            "y_target_mm": float(active_target),
            "duration_ms": 0,
            "reason": "y_hard_floor",
        }
    duration_err = gap_to_happy_edge if bool(y_cfg.get("mast_duration_uses_tolerance_gap")) else abs(float(y_err))
    duration_ms = _adaptive_y_mast_duration_ms(
        float(y_err),
        y_cfg,
        near_end=near_end,
        cmd=cmd,
        duration_err=duration_err,
    )
    if hard_floor_y is not None and cmd == "d":
        min_down_mm = _min_y_curve_closes_mm(y_cfg, cmd="d")
        if min_down_mm is None:
            min_down_mm = min_effect_mm
        if float(y_mm) - float(min_down_mm) <= hard_floor_y:
            return {
                "kind": "wait",
                "action": "Y_HARD_FLOOR_WAIT",
                "dist_err": float(dist_err),
                "x_err": float(x_err),
                "y_err": float(y_err),
                "y_mm": float(y_mm),
                "y_target_mm": float(active_target),
                "duration_ms": 0,
                "reason": "y_hard_floor_projected",
            }
    return {
        "kind": "mast",
        "cmd": cmd,
        "action": f"MAST_{cmd.upper()}",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "y_err": float(y_err),
        "y_mm": float(y_mm),
        "y_target_mm": float(active_target),
        "pwm": _y_mast_pwm(y_cfg, cmd=cmd, near_end=near_end),
        "duration_ms": int(duration_ms),
        "reason": "final_y" if near_end else "approach_high_y",
    }


def _y_plan_closes_win_y(y_plan: dict | None) -> bool:
    if not isinstance(y_plan, dict):
        return False
    if str(y_plan.get("kind") or "").strip().lower() != "mast":
        return False
    cmd = str(y_plan.get("cmd") or "").strip().lower()
    if cmd not in {"u", "d"}:
        return False
    y_cfg = _follow_y_axis_config()
    if not bool(y_cfg.get("enabled")):
        return False
    try:
        y_mm = float(y_plan.get("y_mm"))
        target = float(y_cfg.get("win_target_mm", Y_TARGET_MM))
        tol = float(y_cfg.get("win_tol_mm", Y_TOL_MM))
    except (TypeError, ValueError):
        return False
    final_y_err = float(y_mm) - float(target)
    if _win_axis_ok(final_y_err, tol):
        return False
    return (final_y_err > 0.0 and cmd == "d") or (final_y_err < 0.0 and cmd == "u")


def _plan_mast_cmd(plan: dict | None) -> str | None:
    if not isinstance(plan, dict):
        return None
    kind = str(plan.get("kind") or "").strip().lower()
    cmd = str(plan.get("cmd") or "").strip().lower() if kind == "mast" else ""
    if not cmd:
        cmd = str(plan.get("mast_cmd") or "").strip().lower()
    return cmd if cmd in {"u", "d"} else None


def _plan_mast_duration_ms(plan: dict | None) -> int:
    if not isinstance(plan, dict):
        return 0
    kind = str(plan.get("kind") or "").strip().lower()
    raw = plan.get("duration_ms") if kind == "mast" else plan.get("mast_duration_ms")
    return _coerce_int(raw, 0, minimum=0, maximum=_max_mast_act_ms())


def _step1_mast_up_budget_ms(y_cfg: dict | None = None) -> int:
    cfg = y_cfg if isinstance(y_cfg, dict) else _follow_y_axis_config()
    return _coerce_int(
        cfg.get("max_step1_mast_up_ms"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["max_step1_mast_up_ms"],
        minimum=0,
        maximum=_max_mast_act_ms(),
    )


def _strip_attached_mast_up_for_budget(
    plan: dict | None,
    *,
    used_ms: int,
    planned_ms: int,
    max_up_ms: int,
    remaining_ms: int,
) -> dict | None:
    if not isinstance(plan, dict):
        return plan
    if str(plan.get("kind") or "").strip().lower() == "mast":
        return plan
    out = dict(plan)
    for key in ("mast_cmd", "mast_pwm", "mast_duration_ms", "mast_reason", "mast_y_err", "mast_y_target_mm"):
        out.pop(key, None)
    out["mast_up_budget_blocked"] = True
    out["mast_up_budget_used_ms"] = int(used_ms)
    out["mast_up_budget_planned_ms"] = int(planned_ms)
    out["mast_up_budget_max_ms"] = int(max_up_ms)
    out["mast_up_budget_remaining_ms"] = int(remaining_ms)
    action = str(out.get("action") or "").strip()
    out["action"] = f"{action}_NO_MAST_U_BUDGET" if action else "NO_MAST_U_BUDGET"
    return out


def _cap_mast_plan_to_max_duration(plan: dict | None) -> dict | None:
    if not isinstance(plan, dict):
        return plan
    if _plan_mast_cmd(plan) not in {"u", "d"}:
        return plan
    planned_ms = _plan_mast_duration_ms(plan)
    key = "duration_ms" if str(plan.get("kind") or "").strip().lower() == "mast" else "mast_duration_ms"
    try:
        raw_ms = int(round(float(plan.get(key))))
    except (TypeError, ValueError):
        raw_ms = planned_ms
    if raw_ms <= _max_mast_act_ms():
        return plan
    out = dict(plan)
    out[key] = int(planned_ms)
    out["mast_duration_capped"] = True
    out["mast_duration_original_ms"] = int(raw_ms)
    return out


def _cap_near_target_wheel_plan_to_crawl(plan: dict | None) -> dict | None:
    if not isinstance(plan, dict):
        return plan
    kind = str(plan.get("kind") or "").strip().lower()
    if kind not in {"drive", "drive_bias", "drive_nudge", "turn"}:
        return plan
    if bool(plan.get("skip_near_target_crawl_cap")):
        return plan
    try:
        dist_err = float(plan.get("dist_err"))
        current_ms = int(round(float(plan.get("duration_ms"))))
    except (TypeError, ValueError):
        return plan
    policy = _follow_dist_approach_policy()
    band_mm = _coerce_float(
        policy.get("near_target_creep_band_mm"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"],
        minimum=0.0,
    )
    if band_mm <= 0.0 or abs(float(dist_err)) > float(band_mm):
        return plan
    cap_ms = _coerce_int(
        policy.get("near_target_max_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_max_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    cmd = str(plan.get("cmd") or "").strip().lower()
    if kind == "turn":
        motion_cmd = cmd if cmd in {"l", "r"} else str(plan.get("turn_cmd") or "r").strip().lower()
    elif kind == "drive_bias":
        motion_cmd = str(plan.get("turn_cmd") or cmd or "r").strip().lower()
    else:
        motion_cmd = cmd
    floor_ms = int(_min_motion_duration_ms(motion_cmd))
    capped_ms = int(max(int(floor_ms), min(int(current_ms), int(cap_ms))))
    if capped_ms >= int(current_ms):
        return plan
    out = dict(plan)
    out["duration_ms"] = int(capped_ms)
    out["near_target_crawl_cap"] = True
    out["near_target_crawl_band_mm"] = float(band_mm)
    out["near_target_crawl_original_ms"] = int(current_ms)
    action = str(out.get("action") or "").strip()
    if action and not action.endswith("_CRAWL"):
        out["action"] = f"{action}_CRAWL"
    return out


def _mast_raise_ceiling_target_from_plan(plan: dict | None) -> float:
    if isinstance(plan, dict):
        for key in ("y_target_mm", "mast_y_target_mm"):
            try:
                return float(plan.get(key))
            except (TypeError, ValueError):
                pass
    return _y_win_target_mm()


def _mast_raise_ceiling_margin_mm() -> float:
    return float(MAST_RAISE_CEILING_ABOVE_TARGET_MM)


def _mast_up_ceiling_status(
    reading: dict | None,
    *,
    target_mm: float,
) -> tuple[bool, int | None, float | None, float]:
    try:
        y_mm = float((reading or {}).get("y_mm"))
        target = float(target_mm)
    except (TypeError, ValueError):
        fallback_target = _y_win_target_mm()
        return False, None, None, float(fallback_target) + _mast_raise_ceiling_margin_mm()
    ceiling = float(target) + _mast_raise_ceiling_margin_mm()
    if y_mm >= ceiling:
        return True, 0, y_mm, ceiling
    remaining_mm = max(0.0, float(ceiling) - float(y_mm))
    y_cfg = _follow_y_axis_config()
    rate = _coerce_float(
        y_cfg.get("mast_up_mm_per_100ms"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_up_mm_per_100ms"],
        minimum=0.01,
        maximum=100.0,
    )
    cap_ms = int(max(0, round((remaining_mm / float(rate)) * 100.0)))
    return False, cap_ms, y_mm, ceiling


def _mast_up_ceiling_wait_plan(plan: dict, *, y_mm: float | None, target_mm: float, ceiling_mm: float) -> dict:
    out = {
        "kind": "wait",
        "action": "MAST_UP_CEILING_WAIT",
        "duration_ms": 0,
        "reason": "mast_up_ceiling",
        "y_target_mm": float(target_mm),
        "mast_up_ceiling_y_mm": y_mm,
        "mast_up_ceiling_mm": float(ceiling_mm),
    }
    for key in ("dist_err", "x_err", "y_err"):
        if key in plan:
            out[key] = plan.get(key)
    return out


def _cap_mast_up_plan_to_y_ceiling(plan: dict | None, reading: dict | None) -> dict | None:
    if not isinstance(plan, dict) or _plan_mast_cmd(plan) != "u":
        return plan
    target = _mast_raise_ceiling_target_from_plan(plan)
    blocked, cap_ms, y_mm, ceiling = _mast_up_ceiling_status(reading, target_mm=target)
    kind = str(plan.get("kind") or "").strip().lower()
    if blocked:
        if kind == "mast":
            return _mast_up_ceiling_wait_plan(plan, y_mm=y_mm, target_mm=target, ceiling_mm=ceiling)
        out = dict(plan)
        for key in ("mast_cmd", "mast_pwm", "mast_duration_ms", "mast_reason", "mast_y_err", "mast_y_target_mm"):
            out.pop(key, None)
        out["mast_up_ceiling_blocked"] = True
        out["mast_up_ceiling_y_mm"] = y_mm
        out["mast_up_ceiling_mm"] = float(ceiling)
        action = str(out.get("action") or "").strip()
        out["action"] = f"{action}_NO_MAST_U_CEILING" if action else "NO_MAST_U_CEILING"
        return out
    planned_ms = _plan_mast_duration_ms(plan)
    if cap_ms is None or planned_ms <= int(cap_ms):
        return plan
    y_cfg = _follow_y_axis_config()
    min_up_ms = _coerce_int(
        y_cfg.get("mast_min_pulse_ms"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_min_pulse_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    if int(cap_ms) < int(min_up_ms):
        if kind == "mast":
            return _mast_up_ceiling_wait_plan(plan, y_mm=y_mm, target_mm=target, ceiling_mm=ceiling)
        out = dict(plan)
        for key in ("mast_cmd", "mast_pwm", "mast_duration_ms", "mast_reason", "mast_y_err", "mast_y_target_mm"):
            out.pop(key, None)
        out["mast_up_ceiling_blocked"] = True
        out["mast_up_ceiling_cap_ms"] = int(cap_ms)
        action = str(out.get("action") or "").strip()
        out["action"] = f"{action}_NO_MAST_U_CEILING" if action else "NO_MAST_U_CEILING"
        return out
    capped = dict(plan)
    key = "duration_ms" if kind == "mast" else "mast_duration_ms"
    capped[key] = int(cap_ms)
    capped["mast_up_ceiling_capped"] = True
    capped["mast_up_ceiling_original_ms"] = int(planned_ms)
    capped["mast_up_ceiling_cap_ms"] = int(cap_ms)
    capped["mast_up_ceiling_y_mm"] = y_mm
    capped["mast_up_ceiling_mm"] = float(ceiling)
    return capped


def _mast_up_budget_exceeded(stats: dict, plan: dict | None) -> tuple[bool, int, int, int]:
    if _plan_mast_cmd(plan) != "u":
        return False, 0, 0, 0
    max_up_ms = _step1_mast_up_budget_ms()
    used_ms = _coerce_int((stats or {}).get("step1_mast_up_ms"), 0, minimum=0, maximum=100000)
    planned_ms = _plan_mast_duration_ms(plan)
    if max_up_ms <= 0:
        return True, used_ms, planned_ms, max_up_ms
    return (used_ms + planned_ms) > max_up_ms, used_ms, planned_ms, max_up_ms


def _cap_mast_up_plan_to_budget(stats: dict, plan: dict | None) -> dict | None:
    if not isinstance(plan, dict) or _plan_mast_cmd(plan) != "u":
        return plan
    y_cfg = _follow_y_axis_config()
    max_up_ms = _step1_mast_up_budget_ms(y_cfg)
    used_ms = _coerce_int((stats or {}).get("step1_mast_up_ms"), 0, minimum=0, maximum=100000)
    planned_ms = _plan_mast_duration_ms(plan)
    if max_up_ms <= 0:
        return _strip_attached_mast_up_for_budget(
            plan,
            used_ms=used_ms,
            planned_ms=planned_ms,
            max_up_ms=max_up_ms,
            remaining_ms=0,
        )
    remaining_ms = max(0, int(max_up_ms) - int(used_ms))
    if planned_ms <= remaining_ms:
        return plan
    if remaining_ms <= 0:
        return _strip_attached_mast_up_for_budget(
            plan,
            used_ms=used_ms,
            planned_ms=planned_ms,
            max_up_ms=max_up_ms,
            remaining_ms=remaining_ms,
        )
    min_up_ms = _coerce_int(
        y_cfg.get("mast_min_pulse_ms"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["mast_min_pulse_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    if remaining_ms < min_up_ms:
        return _strip_attached_mast_up_for_budget(
            plan,
            used_ms=used_ms,
            planned_ms=planned_ms,
            max_up_ms=max_up_ms,
            remaining_ms=remaining_ms,
        )
    capped = dict(plan)
    key = "duration_ms" if str(capped.get("kind") or "").strip().lower() == "mast" else "mast_duration_ms"
    capped[key] = int(remaining_ms)
    capped["mast_up_budget_capped"] = True
    capped["mast_up_budget_original_ms"] = int(planned_ms)
    capped["mast_up_budget_remaining_ms"] = int(remaining_ms)
    return capped


def _record_mast_up_budget(stats: dict, plan: dict | None, send_result: dict | None) -> None:
    if not isinstance(stats, dict) or _plan_mast_cmd(plan) != "u":
        return
    duration = 0
    if isinstance(send_result, dict):
        duration = _coerce_int(send_result.get("duration_ms"), 0, minimum=0, maximum=100000)
    if duration <= 0:
        duration = _plan_mast_duration_ms(plan)
    stats["step1_mast_up_ms"] = int(stats.get("step1_mast_up_ms", 0) or 0) + int(duration)


def _plan_closes_win_y_now(plan: dict | None, reading: dict | None) -> bool:
    cmd = _plan_mast_cmd(plan)
    if cmd not in {"u", "d"}:
        return False
    y_cfg = _follow_y_axis_config()
    if not bool(y_cfg.get("enabled")):
        return False
    try:
        target = float(y_cfg.get("win_target_mm", Y_TARGET_MM))
        tol = float(y_cfg.get("win_tol_mm", Y_TOL_MM))
        y_err = float(_y_err_for_reading(reading or {}, target=target))
    except (TypeError, ValueError):
        return False
    if _win_axis_ok(y_err, tol):
        return False
    return (y_err > 0.0 and cmd == "u") or (y_err < 0.0 and cmd == "d")


def _should_grace_step1_timeout_for_y(stats: dict, plan: dict, reading: dict) -> bool:
    if not _plan_closes_win_y_now(plan, reading):
        return False
    cfg = _win_confirmation_config()
    max_grace = _coerce_int(
        cfg.get("timeout_y_correction_grace_acts"),
        DEFAULT_WIN_CONFIRMATION_CONFIG["timeout_y_correction_grace_acts"],
        minimum=0,
        maximum=10,
    )
    if max_grace <= 0:
        return False
    used = int((stats or {}).get("step1_timeout_y_correction_grace_count", 0) or 0)
    return used < max_grace


def _prepare_next_step1_attempt(stats: dict) -> None:
    if not isinstance(stats, dict):
        return
    stats["y_lock_on_armed"] = True
    stats["step1_timeout_y_correction_grace_count"] = 0
    stats["y_commit_active"] = False
    stats["pending_observation"] = None


def _attach_mast_to_plan(plan: dict, y_plan: dict | None) -> dict:
    if not isinstance(plan, dict) or not isinstance(y_plan, dict):
        return plan
    y_reason = str(y_plan.get("reason") or "")
    plan_kind = str(plan.get("kind") or "").strip().lower()
    allowed_kinds = {"drive", "drive_bias", "drive_nudge"}
    if bool(_follow_x_priority_policy().get("attach_y_to_turns", False)):
        allowed_kinds.add("turn")
    if y_reason != "protect_lower_edge" and plan_kind not in allowed_kinds:
        return plan
    if y_reason != "protect_lower_edge" and not _y_plan_closes_win_y(y_plan):
        return plan
    if y_reason != "protect_lower_edge":
        drive_cmd = str(plan.get("cmd") or "").strip().lower()
        drive_mode = str(plan.get("drive_mode") or "").strip().lower()
        try:
            dist_err = float(plan.get("dist_err"))
        except (TypeError, ValueError):
            dist_err = 0.0
        if (
            (drive_cmd == "b" or drive_mode == "backward")
            and dist_err < 0.0
        ):
            return plan
    if y_reason != "protect_lower_edge":
        y_cfg = _follow_y_axis_config()
        attach_min_abs_err = _coerce_float(
            y_cfg.get("attach_y_min_abs_err_mm"),
            DEFAULT_FOLLOW_Y_AXIS_CONFIG["attach_y_min_abs_err_mm"],
            minimum=0.0,
            maximum=100.0,
        )
        try:
            if abs(float(y_plan.get("y_err"))) < float(attach_min_abs_err):
                return plan
        except (TypeError, ValueError):
            return plan
        attach_max_abs_dist = _coerce_float(
            y_cfg.get("attach_y_max_abs_dist_err_mm"),
            DEFAULT_FOLLOW_Y_AXIS_CONFIG["attach_y_max_abs_dist_err_mm"],
            minimum=0.0,
            maximum=500.0,
        )
        if attach_max_abs_dist > 0.0:
            try:
                if abs(float(plan.get("dist_err"))) > float(attach_max_abs_dist):
                    return plan
            except (TypeError, ValueError):
                return plan
    cmd = str(y_plan.get("cmd") or "").strip().lower()
    if cmd not in {"u", "d"}:
        return plan
    out = dict(plan)
    out["mast_cmd"] = cmd
    out["mast_reason"] = y_plan.get("reason")
    out["mast_y_err"] = y_plan.get("y_err")
    out["mast_y_target_mm"] = y_plan.get("y_target_mm")
    out["mast_pwm"] = y_plan.get("pwm")
    out["mast_duration_ms"] = y_plan.get("duration_ms")
    action = str(out.get("action") or "").strip()
    if action:
        out["action"] = f"{action}_MAST_{cmd.upper()}"
    return out


def _x_dist_drive_bias_plan(
    *,
    turn_cmd: str,
    drive_mode: str,
    dist_err: float,
    x_err: float,
    x_outside: float,
    dist_outside: float,
    y_plan: dict | None,
    reason: str,
) -> dict:
    mode = str(drive_mode or "forward").strip().lower()
    if mode not in {"forward", "backward"}:
        mode = "forward"
    strength = _bias_strength_for_dist_x(dist_err=dist_err, x_err=x_err, x_outside_mm=x_outside)
    return _attach_mast_to_plan({
        "kind": "drive_bias",
        "cmd": "b" if mode == "backward" else "f",
        "turn_cmd": str(turn_cmd or "r").strip().lower(),
        "drive_mode": mode,
        "strength": strength,
        "action": f"BIAS_{str(turn_cmd or 'r').upper()}_{strength.upper()}",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "x_outside_mm": float(x_outside),
        "dist_outside_mm": float(dist_outside),
        "duration_ms": _combined_drive_bias_duration_ms(dist_err, drive_mode=mode),
        "distance_creep": True,
        "reason": str(reason),
    }, y_plan)


def _should_avoid_forward_left_bias(
    *,
    turn_cmd: str,
    dist_err: float,
    x_err: float,
    x_outside: float,
    dist_outside: float,
) -> bool:
    policy = _follow_x_priority_policy()
    if not bool(policy.get("avoid_forward_left_bias", False)):
        return False
    if str(turn_cmd or "").strip().lower() != "l":
        return False
    if float(dist_err) <= _win_effective_tolerance(_dist_tol_mm()):
        return False
    if float(x_err) >= 0.0:
        return False
    return (
        float(x_outside) >= float(policy.get("avoid_forward_left_bias_min_x_outside_mm", 0.0))
        and float(dist_outside) >= float(policy.get("avoid_forward_left_bias_min_dist_outside_mm", 0.0))
    )


def _x_turn_duration_ms(*, dist_err: float, turn_cmd: str) -> int:
    """Return turn duration, capped shorter when dist is already near the target gate."""
    policy = _follow_dist_approach_policy()
    near_band = float(policy.get("near_target_creep_band_mm", DEFAULT_DIST_APPROACH_POLICY["near_target_creep_band_mm"]))
    near_max_ms = _coerce_int(
        policy.get("near_target_max_pulse_ms"),
        DEFAULT_DIST_APPROACH_POLICY["near_target_max_pulse_ms"],
        minimum=1,
        maximum=_max_act_ms(),
    )
    min_ms = int(_min_motion_duration_ms(turn_cmd))
    if abs(dist_err) <= near_band:
        return int(max(int(min_ms), int(near_max_ms)))
    return int(max(300, int(PULSE_MS), min_ms))


def _x_only_turn_plan(
    *,
    reading: dict,
    turn_cmd: str,
    drive_mode: str,
    strength: str,
    dist_err: float,
    x_err: float,
    x_outside: float,
    dist_outside: float,
    y_plan: dict | None,
    reason: str,
    use_production_curve: bool = False,
) -> dict:
    plan = {
        "kind": "turn",
        "cmd": str(turn_cmd or "r").strip().lower(),
        "drive_mode": str(drive_mode or _x_only_turn_drive_mode_for_dist(dist_err)).strip().lower(),
        "strength": str(strength or _follow_x_priority_policy().get("x_first_turn_strength", "strong")),
        "action": f"TURN_{str(turn_cmd or 'r').upper()}",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "x_outside_mm": float(x_outside),
        "dist_outside_mm": float(dist_outside),
        "duration_ms": _x_turn_duration_ms(dist_err=dist_err, turn_cmd=turn_cmd),
        "reason": str(reason),
    }
    if bool(use_production_curve):
        plan["use_production_turn_curve"] = True
        production_curve = _production_turn_curve_for_reading(
            cmd=str(turn_cmd or "r").strip().lower(),
            drive_mode=plan["drive_mode"],
            reading=reading,
            x_err_mm=float(x_err),
        )
        if isinstance(production_curve, tuple):
            curve, production_duration_ms = production_curve
            if production_duration_ms is not None:
                _near_cap_ms = _x_turn_duration_ms(dist_err=dist_err, turn_cmd=turn_cmd)
                _prod_ms = int(max(300, _bounded_act_duration_ms(production_duration_ms)))
                if _near_cap_ms < 300:
                    plan["duration_ms"] = int(min(_prod_ms, _near_cap_ms))
                    plan["use_production_turn_curve"] = False
                    plan["strength"] = "gentle"
                    plan["near_target_production_curve_disabled"] = True
                else:
                    plan["duration_ms"] = _prod_ms
            plan["production_curve_name"] = curve.get("curve_name")
            plan["production_curve_value_mm"] = curve.get("curve_value_mm")
    return _attach_mast_to_plan(plan, y_plan)


def _follow_action_plan(reading: dict) -> dict:
    dist_mm = float(reading["dist_mm"])
    x_mm = float(reading["x_mm"])
    dist_err = dist_mm - _dist_target_mm()
    x_err = float(x_mm - _x_target_mm())
    x_ok = _win_axis_ok(x_err, _x_tol_mm())
    dist_ok = _win_axis_ok(dist_err, _dist_tol_mm())
    dist_happy_tol = _win_effective_tolerance(_dist_tol_mm())
    y_cfg = _follow_y_axis_config()
    y_err = _y_err_for_reading(reading, target=float(y_cfg.get("win_target_mm", Y_TARGET_MM)))
    if _pickup_suspected_reading(reading):
        return {
            "kind": "wait",
            "action": "PICKUP_SUSPECT_STOP",
            "dist_err": float(dist_err),
            "x_err": float(x_err),
            "y_err": y_err,
            "duration_ms": 0,
            "reason": "pickup_suspected_far_low",
        }
    y_plan = _y_axis_action_plan(reading, dist_err=dist_err, x_err=x_err)
    y_ok = (
        True
        if y_err is None or not bool(y_cfg.get("enabled"))
        or (float(y_err) < 0.0 and _step1_mast_up_budget_ms(y_cfg) <= 0)
        else _win_axis_ok(float(y_err), float(y_cfg.get("win_tol_mm", Y_TOL_MM)))
    )

    if x_ok and dist_ok and y_ok:
        return {"kind": "hold", "action": "HAPPY", "dist_err": dist_err, "x_err": x_err, "y_err": y_err}
    if isinstance(y_plan, dict) and str(y_plan.get("reason") or "") == "protect_lower_edge":
        return y_plan
    if _active_game_profile() == "holding":
        holding_x_outside = _x_outside_gate_mm(x_err)
        if dist_ok and not x_ok:
            turn_cmd = _turn_cmd_to_close_x_gap(x_err) or "r"
            return _attach_mast_to_plan({
                "kind": "drive_bias",
                "cmd": "b",
                "turn_cmd": turn_cmd,
                "drive_mode": "backward",
                "strength": "gentle",
                "action": f"BIAS_{turn_cmd.upper()}_GENTLE_BACKOFF",
                "dist_err": float(dist_err),
                "x_err": float(x_err),
                "x_outside_mm": float(holding_x_outside),
                "dist_outside_mm": float(_dist_outside_gate_mm(dist_err)),
                "duration_ms": int(HOLDING_S1_RETRY_BACKOFF_MS),
                "distance_creep": True,
                "allow_long_duration": True,
                "skip_near_target_crawl_cap": True,
                "reason": "holding_s1_dist_ok_x_retry_long_gentle_backoff",
            }, y_plan)
        holding_x_polish_dist_band_mm = min(10.0, max(3.0, float(_dist_tol_mm()) * 0.35))
        must_center_dist_before_x = (not bool(dist_ok)) or (
            not bool(x_ok) and abs(float(dist_err)) > float(holding_x_polish_dist_band_mm)
        )
    else:
        must_center_dist_before_x = False
        holding_x_polish_dist_band_mm = 0.0
    if bool(must_center_dist_before_x):
        dist_cmd = _dist_cmd_for_error(dist_err)
        dist_outside = _dist_outside_gate_mm(dist_err)
        duration_ms = _distance_creep_duration_ms(dist_err)
        if bool(dist_ok):
            duration_ms = min(int(duration_ms), 80)
        return {
            "kind": "drive",
            "cmd": dist_cmd,
            "action": "BCK" if dist_cmd == "b" else "FWD",
            "dist_err": dist_err,
            "x_err": x_err,
            "x_outside_mm": float(_x_outside_gate_mm(x_err)),
            "dist_outside_mm": float(dist_outside),
            "duration_ms": int(duration_ms),
            "distance_creep": True,
            "reason": (
                "holding_s1_dist_first_before_x"
                if not bool(dist_ok)
                else f"holding_s1_center_dist_before_x_{holding_x_polish_dist_band_mm:.1f}mm"
            ),
        }
    if dist_err < -dist_happy_tol:
        if not x_ok:
            turn_cmd = _turn_cmd_to_close_x_gap(x_err) or "r"
            x_outside = _x_outside_gate_mm(x_err)
            dist_outside = _dist_outside_gate_mm(dist_err)
            if _tiny_x_outside_no_turn(x_outside):
                micro_plan = _distance_micro_straight_plan(
                    reading,
                    dist_err=dist_err,
                    x_err=x_err,
                    y_plan=y_plan,
                )
                if micro_plan is not None:
                    return micro_plan
                return _attach_mast_to_plan({
                    "kind": "drive",
                    "cmd": _dist_cmd_for_error(dist_err),
                    "action": "BCK" if _dist_cmd_for_error(dist_err) == "b" else "FWD",
                    "dist_err": dist_err,
                    "x_err": x_err,
                    "x_outside_mm": float(x_outside),
                    "dist_outside_mm": float(dist_outside),
                    "duration_ms": _distance_correction_duration_ms(dist_err),
                    "distance_creep": True,
                    "reason": "tiny_x_gap_backoff_instead_of_turn",
                }, y_plan)
            if dist_outside >= float(_follow_combined_gap_policy().get("straight_dist_outside_min_mm", 0.0)):
                dist_cmd = _dist_cmd_for_error(dist_err)
                return _attach_mast_to_plan({
                    "kind": "drive",
                    "cmd": dist_cmd,
                    "action": "BCK" if dist_cmd == "b" else "FWD",
                    "dist_err": dist_err,
                    "x_err": x_err,
                    "x_outside_mm": float(x_outside),
                    "dist_outside_mm": float(dist_outside),
                    "duration_ms": _distance_correction_duration_ms(dist_err),
                    "distance_creep": True,
                    "reason": "too_close_dist_first_before_x",
                }, y_plan)
            if not _sharp_x_only_turn_allowed(dist_err):
                return _x_dist_drive_bias_plan(
                    turn_cmd=turn_cmd,
                    drive_mode=_drive_mode_for_dist_error(dist_err),
                    dist_err=dist_err,
                    x_err=x_err,
                    x_outside=x_outside,
                    dist_outside=dist_outside,
                    y_plan=y_plan,
                    reason="x_polish_while_backing_dist",
                )
            return _x_only_turn_plan(
                reading=reading,
                turn_cmd=turn_cmd,
                drive_mode="backward",
                strength=str(_follow_x_priority_policy().get("x_first_turn_strength", "strong")),
                dist_err=dist_err,
                x_err=x_err,
                x_outside=x_outside,
                dist_outside=dist_outside,
                y_plan=y_plan,
                reason="sharp_x_only_tiny_dist",
                use_production_curve=True,
            )
        nudge_plan = _distance_micro_nudge_plan(reading, dist_err=dist_err, x_err=x_err, y_plan=y_plan)
        if nudge_plan is not None:
            return nudge_plan
        return {
            "kind": "drive",
            "cmd": _dist_cmd_for_error(dist_err),
            "action": "BCK" if _dist_cmd_for_error(dist_err) == "b" else "FWD",
            "dist_err": dist_err,
            "x_err": x_err,
            "duration_ms": _distance_correction_duration_ms(dist_err),
            "distance_creep": True,
            "reason": "dist_only_creep",
        }
    if _should_finish_y_before_wheels(y_plan, dist_err=dist_err, x_err=x_err):
        return y_plan
    dist_approach = _follow_dist_approach_policy()
    if not x_ok:
        turn_cmd = _turn_cmd_to_close_x_gap(x_err) or "r"
        x_outside = _x_outside_gate_mm(x_err)
        dist_outside = _dist_outside_gate_mm(dist_err)
        if (
            dist_err < -0.75 * float(dist_happy_tol)
            and x_outside <= float(_follow_combined_gap_policy().get("straight_x_outside_max_mm", 0.0))
        ):
            return _attach_mast_to_plan({
                "kind": "drive",
                "cmd": "b",
                "action": "BCK",
                "dist_err": dist_err,
                "x_err": x_err,
                "x_outside_mm": float(x_outside),
                "dist_outside_mm": float(dist_outside),
                "duration_ms": _distance_creep_duration_ms(dist_err),
                "distance_creep": True,
                "reason": "near_close_edge_backoff_before_x",
            }, y_plan)
        if _tiny_x_outside_no_turn(x_outside):
            if dist_err > dist_happy_tol:
                micro_plan = _distance_micro_straight_plan(
                    reading,
                    dist_err=dist_err,
                    x_err=x_err,
                    y_plan=y_plan,
                )
                if micro_plan is not None:
                    return micro_plan
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
                    "reason": "tiny_x_gap_forward_instead_of_turn",
                }, y_plan)
            if y_plan is not None:
                return y_plan
            if bool(_follow_x_priority_policy().get("tiny_x_only_backoff_enabled", True)):
                if dist_ok:
                    return _x_only_turn_plan(
                        reading=reading,
                        turn_cmd=turn_cmd,
                        drive_mode=_x_only_turn_drive_mode_for_dist(dist_err),
                        strength=str(_follow_x_priority_policy().get("x_first_turn_strength", "adaptive")),
                        dist_err=dist_err,
                        x_err=x_err,
                        x_outside=x_outside,
                        dist_outside=dist_outside,
                        y_plan=y_plan,
                        reason="tiny_x_only_turn_at_happy_dist",
                        use_production_curve=False,
                    )
                return {
                    "kind": "drive",
                    "cmd": "b",
                    "action": "BCK",
                    "dist_err": dist_err,
                    "x_err": x_err,
                    "x_outside_mm": float(x_outside),
                    "dist_outside_mm": float(dist_outside),
                    "duration_ms": _min_effective_drive_duration_ms("b"),
                    "distance_creep": True,
                    "reason": "tiny_x_only_backoff_instead_of_turn",
                }
            return {
                "kind": "wait",
                "action": "TINY_X_OBSERVE",
                "dist_err": dist_err,
                "x_err": x_err,
                "x_outside_mm": float(x_outside),
                "dist_outside_mm": float(dist_outside),
                "duration_ms": 0,
                "reason": "tiny_x_gap_no_subfloor_turn",
            }
        sharp_x_only_allowed = _sharp_x_only_turn_allowed(dist_err)
        if _near_target_forward_veto_active(dist_err=dist_err, x_ok=x_ok, y_ok=y_ok) and sharp_x_only_allowed:
            return _x_only_turn_plan(
                reading=reading,
                turn_cmd=turn_cmd,
                drive_mode=_x_only_turn_drive_mode_for_dist(dist_err),
                strength=_near_wide_x_turn_strength(abs(x_err)),
                dist_err=dist_err,
                x_err=x_err,
                x_outside=x_outside,
                dist_outside=dist_outside,
                y_plan=y_plan,
                reason="near_target_x_before_forward",
                use_production_curve=True,
            )
        x_first_before_dist = _should_close_x_before_distance(abs_x_err=abs(x_err), dist_err=dist_err)
        if x_first_before_dist and (sharp_x_only_allowed or dist_err > dist_happy_tol):
            edge_turn_cmd = turn_cmd
            reason = "sharp_x_only_tiny_dist" if sharp_x_only_allowed else "wide_x_before_dist"
            edge_threshold = float(
                _vision_jump_guard_config().get("edge_recovery_min_abs_x_mm", 200.0) or 200.0
            )
            if not sharp_x_only_allowed and dist_err > dist_happy_tol and abs(float(x_err)) >= edge_threshold:
                edge_turn_cmd = _opposite_turn_cmd(turn_cmd) or turn_cmd
                reason = "wide_edge_x_recovery_opposite_turn"
            return _x_only_turn_plan(
                reading=reading,
                turn_cmd=edge_turn_cmd,
                drive_mode=_x_only_turn_drive_mode_for_dist(dist_err),
                strength=str(_follow_x_priority_policy().get("x_first_turn_strength", "strong")),
                dist_err=dist_err,
                x_err=x_err,
                x_outside=x_outside,
                dist_outside=dist_outside,
                y_plan=y_plan,
                reason=reason,
                use_production_curve=bool(sharp_x_only_allowed),
            )
        if dist_err > dist_happy_tol:
            policy = _follow_combined_gap_policy()
            if (
                x_outside <= float(policy.get("straight_x_outside_max_mm", 0.0))
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
            if _should_avoid_forward_left_bias(
                turn_cmd=turn_cmd,
                dist_err=dist_err,
                x_err=x_err,
                x_outside=x_outside,
                dist_outside=dist_outside,
            ):
                return _x_only_turn_plan(
                    reading=reading,
                    turn_cmd=turn_cmd,
                    drive_mode=_x_only_turn_drive_mode_for_dist(dist_err),
                    strength=str(_follow_x_priority_policy().get("x_first_turn_strength", "strong")),
                    dist_err=dist_err,
                    x_err=x_err,
                    x_outside=x_outside,
                    dist_outside=dist_outside,
                    y_plan=y_plan,
                    reason="forward_left_bias_untrusted_x_first",
                    use_production_curve=bool(sharp_x_only_allowed),
                )
            return _x_dist_drive_bias_plan(
                turn_cmd=turn_cmd,
                drive_mode=_drive_mode_for_dist_error(dist_err),
                dist_err=dist_err,
                x_err=x_err,
                x_outside=x_outside,
                dist_outside=dist_outside,
                y_plan=y_plan,
                reason="x_polish_while_creeping_dist",
            )
        return _x_only_turn_plan(
            reading=reading,
            turn_cmd=turn_cmd,
            drive_mode=_x_only_turn_drive_mode_for_dist(dist_err),
            strength=str(_follow_x_priority_policy().get("x_first_turn_strength", "strong")),
            dist_err=dist_err,
            x_err=x_err,
            x_outside=x_outside,
            dist_outside=dist_outside,
            y_plan=y_plan,
            reason="sharp_x_only_tiny_dist" if sharp_x_only_allowed else "x_first_before_dist",
            use_production_curve=bool(sharp_x_only_allowed),
        )

    dist_cmd = _dist_cmd_for_error(dist_err)
    nudge_plan = _distance_micro_nudge_plan(reading, dist_err=dist_err, x_err=x_err, y_plan=y_plan)
    if nudge_plan is not None:
        return nudge_plan
    return _attach_mast_to_plan({
        "kind": "drive",
        "cmd": dist_cmd,
        "action": "BCK" if dist_cmd == "b" else "FWD",
        "dist_err": dist_err,
        "x_err": x_err,
        "duration_ms": _distance_correction_duration_ms(dist_err),
        "distance_creep": True,
        "reason": "dist_only_creep",
    }, y_plan)


def _holding_s1_transition_commit_plan(stats: dict, reading: dict, base_plan: dict | None = None) -> dict | None:
    if _active_game_profile() != "holding":
        return None
    if not isinstance(stats, dict) or bool(stats.get("holding_s1_transition_commit_sent")):
        return None
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return None
    base_kind = str((base_plan or {}).get("kind") or "").strip().lower()
    if base_kind in {"wait", "mast"}:
        return None
    try:
        dist_err = float(reading["dist_mm"]) - float(_dist_target_mm())
        x_err = float(reading["x_mm"]) - float(_x_target_mm())
    except (KeyError, TypeError, ValueError):
        return None
    if dist_err <= 0.0:
        return None
    max_dist_err = max(float(_dist_tol_mm()), float(HOLDING_S1_TRANSITION_MAX_DIST_ERR_MM))
    if dist_err > float(max_dist_err):
        return None
    x_outside = float(_x_outside_gate_mm(x_err))
    if x_outside > float(HOLDING_S1_TRANSITION_MAX_X_OUTSIDE_MM):
        return None
    return {
        "kind": "drive",
        "cmd": "f",
        "action": "HOLDING_S1_COMMIT_FWD",
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "x_outside_mm": float(x_outside),
        "dist_outside_mm": float(_dist_outside_gate_mm(dist_err)),
        "duration_ms": int(HOLDING_S1_TRANSITION_COMMIT_MS),
        "distance_creep": True,
        "allow_long_duration": True,
        "skip_near_target_crawl_cap": True,
        "reason": "holding_s1_near_far_transition_commit_800ms",
    }


def _execute_follow_action(robot: Robot, plan: dict, reading: dict) -> None:
    kind = str((plan or {}).get("kind") or "").strip().lower()
    if kind == "wait":
        _stop_robot(robot)
        return {"cmd_sent": "s", "pwm": 0, "power": 0.0, "duration_ms": 0, "skipped": True}
    if kind == "hold":
        _stop_robot(robot)
        return {"cmd_sent": "s", "pwm": 0, "power": 0.0, "duration_ms": 0, "skipped": True}
    if kind == "drive":
        return _drive(
            robot,
            str(plan.get("cmd") or "f"),
            reading,
            mast_cmd=plan.get("mast_cmd"),
            mast_pwm=plan.get("mast_pwm"),
            mast_duration_ms=plan.get("mast_duration_ms"),
            pwm=plan.get("pwm"),
            duration_ms=plan.get("duration_ms"),
            recovery_boost_scale=plan.get("recovery_boost_scale"),
            allow_long_duration=bool(plan.get("allow_long_duration")),
        )
    elif kind == "drive_bias":
        return _send_drive_bias(
            robot,
            turn_cmd=str(plan.get("turn_cmd") or "l"),
            drive_mode=str(plan.get("drive_mode") or "forward"),
            strength=str(plan.get("strength") or "micro"),
            duration_ms=_wheel_act_duration_ms(
                plan.get("duration_ms", PULSE_MS),
                allow_long_duration=bool(plan.get("allow_long_duration")),
            ),
            reading=reading,
            context="follow_drive_bias",
            mast_cmd=plan.get("mast_cmd"),
            mast_pwm=plan.get("mast_pwm"),
            mast_duration_ms=plan.get("mast_duration_ms"),
            recovery_boost_scale=plan.get("recovery_boost_scale"),
            allow_long_duration=bool(plan.get("allow_long_duration")),
        )
    elif kind == "drive_nudge":
        return _send_drive_nudge(
            robot,
            cmd=str(plan.get("cmd") or "f"),
            turn_cmd=str(plan.get("turn_cmd") or "l"),
            pwm=_coerce_int(plan.get("pwm"), DEFAULT_DIST_APPROACH_POLICY["micro_nudge_pwm"], minimum=1, maximum=255),
            duration_ms=_bounded_act_duration_ms(plan.get("duration_ms", PULSE_MS)),
            reading=reading,
            mast_cmd=plan.get("mast_cmd"),
            mast_pwm=plan.get("mast_pwm"),
            mast_duration_ms=plan.get("mast_duration_ms"),
            recovery_boost_scale=plan.get("recovery_boost_scale"),
        )
    elif kind == "turn":
        return _send_turn_curve(
            robot,
            cmd=str(plan.get("cmd") or "l"),
            drive_mode=str(plan.get("drive_mode") or _x_only_turn_drive_mode()),
            strength=str(plan.get("strength") or _curve_strength_for_reading(reading)),
            duration_ms=_bounded_act_duration_ms(plan.get("duration_ms", PULSE_MS)),
            reading=reading,
            context="follow_x_only_turn_curve",
            mast_cmd=plan.get("mast_cmd"),
            mast_pwm=plan.get("mast_pwm"),
            mast_duration_ms=plan.get("mast_duration_ms"),
            use_production_curve=bool(plan.get("use_production_turn_curve")),
            recovery_boost_scale=plan.get("recovery_boost_scale"),
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
    if bool(plan.get("use_production_turn_curve")):
        turn_cmd = str(plan.get("cmd") or "l").strip().lower()
        try:
            prod = _production_turn_curve_for_reading(
                cmd=turn_cmd,
                drive_mode=drive_mode,
                reading=reading,
                x_err_mm=float(plan.get("x_err", x_abs)),
            )
        except (TypeError, ValueError):
            prod = None
        if isinstance(prod, tuple):
            curve, _duration = prod
            return dict(curve)
    return (
        _adaptive_turn_curve_for_drive_mode(drive_mode, x_abs)
        if strength == "adaptive"
        else _turn_curve_for_drive_mode(drive_mode, strength)
    )


def _sent_motion_duration_ms(plan: dict, send_result) -> float:
    try:
        return max(1.0, float((send_result or {}).get("duration_ms")))
    except (AttributeError, TypeError, ValueError):
        try:
            return max(1.0, float(plan.get("duration_ms")))
        except (TypeError, ValueError):
            return max(1.0, float(PULSE_MS))


def _sent_motion_duration_text(plan: dict, send_result) -> str:
    if _send_result_blocked(send_result):
        return ""
    try:
        packet_ms = int(round(_sent_motion_duration_ms(plan, send_result)))
    except (TypeError, ValueError):
        return ""
    wheel_ms = None
    if isinstance(send_result, dict):
        wheel_durations = []
        for action in send_result.get("actions") or []:
            if not isinstance(action, dict):
                continue
            if str(action.get("target") or "").strip().lower() not in {"l", "r"}:
                continue
            if str(action.get("action") or "").strip().lower() in {"", "s", "stop"}:
                continue
            try:
                wheel_durations.append(int(round(float(action.get("duration_ms")))))
            except (TypeError, ValueError):
                pass
        if wheel_durations:
            wheel_ms = max(wheel_durations)
    if wheel_ms is not None and int(wheel_ms) != int(packet_ms):
        text = f" wheel={int(wheel_ms)}ms packet={int(packet_ms)}ms"
    else:
        text = f" t={packet_ms}ms"
    try:
        boost_scale = float((plan or {}).get("recovery_boost_scale", 1.0) or 1.0)
    except (AttributeError, TypeError, ValueError):
        boost_scale = 1.0
    if boost_scale > 1.0:
        text = f"{text} boost={boost_scale:.2f}x"
    return text


def _distance_bearing_plan(plan: dict | None) -> bool:
    if not isinstance(plan, dict):
        return False
    if not bool(plan.get("distance_creep")):
        return False
    kind = str(plan.get("kind") or "").strip().lower()
    if kind not in {"drive", "drive_bias", "drive_nudge"}:
        return False
    return str(plan.get("cmd") or "").strip().lower() in {"f", "b"}


def _plan_allows_large_dist_reacquire(plan: dict | None) -> bool:
    if not isinstance(plan, dict):
        return False
    kind = str(plan.get("kind") or "").strip().lower()
    return kind in {"drive", "drive_bias", "drive_nudge", "mast"}


def _record_distance_act_state(stats: dict, action: str, plan: dict) -> None:
    if not isinstance(stats, dict) or not _distance_bearing_plan(plan):
        return
    try:
        dist_err = float(plan.get("dist_err"))
    except (TypeError, ValueError):
        return
    stats["last_distance_act_dist_err"] = float(dist_err)
    stats["last_distance_act_action"] = str(action or plan.get("action") or "")
    stats["last_distance_act_cmd"] = str(plan.get("cmd") or "").strip().lower()


def _should_stop_confirm_for_dist_pingpong(stats: dict, plan: dict) -> bool:
    if not isinstance(stats, dict) or not _distance_bearing_plan(plan):
        return False
    if bool(stats.get("last_act_was_mast")):
        return False
    try:
        previous_dist_err = float(stats.get("last_distance_act_dist_err"))
        current_dist_err = float(plan.get("dist_err"))
    except (TypeError, ValueError):
        return False
    if previous_dist_err == 0.0 or current_dist_err == 0.0:
        return False
    if previous_dist_err * current_dist_err >= 0.0:
        return False
    policy = _follow_dist_approach_policy()
    threshold = float(
        policy.get(
            "pingpong_confirm_abs_dist_err_mm",
            DEFAULT_DIST_APPROACH_POLICY["pingpong_confirm_abs_dist_err_mm"],
        )
    )
    return threshold > 0.0 and abs(current_dist_err) <= threshold


def _post_action_wait_s(plan: dict, send_result) -> float:
    if not isinstance(plan, dict):
        return float(LOOP_S)
    duration_s = _sent_motion_duration_ms(plan, send_result) / 1000.0
    has_mast = str(plan.get("kind") or "").strip().lower() == "mast" or bool(plan.get("mast_cmd"))
    coast_s = _y_motion_coast_settle_s() if has_mast else 0.0
    jump_stabilize_s = float(
        _vision_jump_guard_config().get(
            "post_act_stabilize_s",
            DEFAULT_VISION_JUMP_GUARD_CONFIG["post_act_stabilize_s"],
        )
        or 0.0
    )
    if not bool(plan.get("distance_creep")):
        if str(plan.get("kind") or "").strip().lower() == "turn":
            turn_settle_s = float(_follow_x_priority_policy().get("turn_settle_s", 0.0))
            return max(
                float(LOOP_S),
                float(duration_s) + max(0.0, turn_settle_s),
                float(duration_s) + float(coast_s),
                float(duration_s) + max(0.0, jump_stabilize_s),
            )
        return max(float(LOOP_S), float(duration_s) + float(coast_s), float(duration_s) + max(0.0, jump_stabilize_s))
    policy = _follow_dist_approach_policy()
    shots = float(policy.get("closure_shots", DEFAULT_DIST_APPROACH_POLICY["closure_shots"]))
    settle_s = float(policy.get("settle_after_act_s", DEFAULT_DIST_APPROACH_POLICY["settle_after_act_s"]))
    readback_spacing_s = max(0.0, float(shots) - 1.0) * float(LOOP_S)
    return max(
        float(LOOP_S),
        float(duration_s) + max(0.0, settle_s) + readback_spacing_s + float(coast_s),
        float(duration_s) + max(0.0, jump_stabilize_s),
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
    """Reset sequence: straight back, then bounded strong curve to open x."""
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
        f"[RESET] Two-phase reset: STRAIGHT_BACK then BACK_CURVE_{turn_cmd.upper()} "
        f"target dist={dist_target:.0f}±{dist_tol:.0f}mm, "
        f"|x|~{target_abs_x:.0f}mm ({x_min:.0f}-{x_max:.0f}mm), "
        f"logged y={y_target:+.0f}±{y_tol:.0f}mm (not a hit gate)",
        flush=True,
    )

    before_reading = _read_brick_measurement(vision)
    if not bool(before_reading.get("confident")):
        before_reading = _wait_for_visibility_recovery(
            vision,
            robot,
            before_reading,
            context="reset_start",
        )
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

    if _reset_xy_target_ready(before_reading, reset_cfg):
        print(
            f"[RESET] Already inside reset gate: dist={before_dist:.1f}mm x={before_x:+.1f}mm y={before_y_text}",
            flush=True,
        )
        return True, "target_already_ready", before_reading

    before_abs_x = abs(float(before_x))
    if before_abs_x < float(x_min):
        turn_cmd = _turn_cmd_to_open_x_gap(before_x, turn_cmd)
    elif before_abs_x > float(x_max):
        turn_cmd = _turn_cmd_to_close_x_gap(before_x) or turn_cmd

    if before_dist > (float(dist_target) + float(dist_tol)) or before_abs_x > float(x_max):
        print(
            "[RESET] Skipping opening straight-back/curve because reset is already beyond the gate: "
            f"dist={before_dist:.1f}mm x={before_x:+.1f}mm. Polishing directly toward the reset target.",
            flush=True,
        )
        after_reading, target_met_xy, adjustment_attempts = _adjust_reset_until_xy_target(
            vision,
            robot,
            reading=before_reading,
            reset_cfg=reset_cfg,
            initial_turn_cmd=turn_cmd,
        )
        if not bool(after_reading.get("confident")):
            return False, "lost_confident_brick_after_reset_adjustment", after_reading
        after_reading, final_x_polished, final_x_reason = _reset_final_x_offset_polish(
            vision,
            robot,
            reading=after_reading,
            reset_cfg=reset_cfg,
            initial_turn_cmd=turn_cmd,
        )
        if not bool(after_reading.get("confident")):
            return False, "lost_confident_brick_after_final_x_polish", after_reading
        target_met_xy = _reset_xy_target_ready(after_reading, reset_cfg)
        try:
            after_dist = float(after_reading["dist_mm"])
            after_x = float(after_reading["x_mm"])
        except (TypeError, ValueError):
            return False, "invalid_reset_after_reading", after_reading
        try:
            after_y_text = f"{float(after_reading.get('y_mm')):+.1f}mm"
        except (TypeError, ValueError):
            after_y_text = "N/A"
        closeness = _reset_closeness_from_reading(after_reading, reset_cfg)
        close_text = ""
        if closeness is not None:
            dist_close, x_close, y_close, combined_close = closeness
            y_text = "" if y_close is None else f" y_log={y_close:.0f}%"
            close_text = (
                f" close={combined_close:.0f}% "
                f"(dist={dist_close:.0f}% x={x_close:.0f}%{y_text})"
            )
        print(
            f"[RESET] Reset complete: dist={after_dist:.1f}mm x={after_x:+.1f}mm y={after_y_text} "
            f"target_dist_abs_x={'hit' if bool(target_met_xy) else 'miss'} adjustments={int(adjustment_attempts)} "
            f"final_x={'sent' if bool(final_x_polished) else 'skip:' + str(final_x_reason)}{close_text}",
            flush=True,
        )
        if bool(target_met_xy):
            reason = "adjusted_target_hit" if int(adjustment_attempts) > 0 else "target_hit"
        elif int(adjustment_attempts) > 0:
            reason = "adjustment_limit_complete"
        else:
            reason = "outside_gate_no_adjustment"
        return True, reason, after_reading

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

    print(
        f"[RESET] STRAIGHT_BACK sent before BACK_CURVE_{turn_cmd.upper()}: "
        f"before dist={before_dist:.1f}mm x={before_x:+.1f}mm y={before_y_text} "
        f"wheel_pulse={int(pulse_ms)}ms{mast_text}",
        flush=True,
    )
    wheel_wait_s = (float(pulse_ms) / 1000.0) + float(settle_s)
    mast_wait_s = ((float(mast_up_ms) / 1000.0) + float(mast_settle_s)) if mast_up_ms > 0 else 0.0
    time.sleep(max(LOOP_S, float(wheel_wait_s), float(mast_wait_s)))
    _stop_robot(robot)

    _reset_follow_reading_history(vision)
    after_reading = _read_brick_measurement(vision)
    if not bool(after_reading.get("confident")):
        after_reading = _wait_for_visibility_recovery(
            vision,
            robot,
            after_reading,
            context="reset_after_straight_back",
        )
    if not bool(after_reading.get("confident")):
        return False, "lost_confident_brick_after_reset_straight_back", after_reading
    try:
        straight_dist = float(after_reading.get("dist_mm"))
        straight_x = float(after_reading.get("x_mm"))
        straight_y = float(after_reading.get("y_mm"))
        print(
            f"[RESET] After STRAIGHT_BACK read: "
            f"dist={straight_dist:.1f}mm x={straight_x:+.1f}mm y={straight_y:+.1f}mm",
            flush=True,
        )
    except (TypeError, ValueError):
        print("[RESET] After STRAIGHT_BACK read: visible, numeric read unavailable.", flush=True)

    after_reading, curve_meta = _reset_curve_until_x_goal_fraction(
        vision,
        robot,
        direction=turn_cmd,
        reading=after_reading,
        reset_cfg=reset_cfg,
        target_abs_x_mm=target_abs_x,
    )
    curve_abs = curve_meta.get("abs_x_mm") if isinstance(curve_meta, dict) else None
    curve_goal = curve_meta.get("x_goal_mm") if isinstance(curve_meta, dict) else None
    curve_abs_text = "N/A" if curve_abs is None else f"{float(curve_abs):.1f}mm"
    curve_goal_text = "N/A" if curve_goal is None else f"{float(curve_goal):.1f}mm"
    print(
        f"[RESET] BACK_CURVE_{turn_cmd.upper()} complete: "
        f"reason={curve_meta.get('reason') if isinstance(curve_meta, dict) else 'unknown'} "
        f"elapsed={int(curve_meta.get('elapsed_ms', 0) if isinstance(curve_meta, dict) else 0)}ms "
        f"|x|={curve_abs_text} goal80={curve_goal_text}",
        flush=True,
    )
    if not bool(after_reading.get("confident")):
        return False, "lost_confident_brick_after_reset_x_curve", after_reading

    pause_s = _reset_post_pause_s()
    if pause_s > 0.0:
        print(f"[RESET] Pause {pause_s:.1f}s before measuring reset result.", flush=True)
        time.sleep(pause_s)

    _reset_follow_reading_history(vision)
    after_reading = _read_brick_measurement(vision)
    if not bool(after_reading.get("confident")):
        after_reading = _wait_for_visibility_recovery(
            vision,
            robot,
            after_reading,
            context="reset_after_motion",
        )
    if not bool(after_reading.get("confident")):
        return False, "lost_confident_brick_after_reset", after_reading

    after_reading, target_met_xy, adjustment_attempts = _adjust_reset_until_xy_target(
        vision,
        robot,
        reading=after_reading,
        reset_cfg=reset_cfg,
        initial_turn_cmd=turn_cmd,
    )
    if not bool(after_reading.get("confident")):
        return False, "lost_confident_brick_after_reset_adjustment", after_reading
    after_reading, final_x_polished, final_x_reason = _reset_final_x_offset_polish(
        vision,
        robot,
        reading=after_reading,
        reset_cfg=reset_cfg,
        initial_turn_cmd=turn_cmd,
    )
    if not bool(after_reading.get("confident")):
        return False, "lost_confident_brick_after_final_x_polish", after_reading
    target_met_xy = _reset_xy_target_ready(after_reading, reset_cfg)

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
    target_met = bool(target_met_xy)
    closeness = _reset_closeness_from_reading(after_reading, reset_cfg)
    close_text = ""
    if closeness is not None:
        dist_close, x_close, y_close, combined_close = closeness
        y_text = "" if y_close is None else f" y_log={y_close:.0f}%"
        close_text = (
            f" close={combined_close:.0f}% "
            f"(dist={dist_close:.0f}% x={x_close:.0f}%{y_text})"
        )
    print(
        f"[RESET] Reset complete: dist={after_dist:.1f}mm x={after_x:+.1f}mm y={after_y_text} "
        f"target_dist_abs_x={'hit' if target_met else 'miss'} adjustments={int(adjustment_attempts)} "
        f"final_x={'sent' if bool(final_x_polished) else 'skip:' + str(final_x_reason)}{close_text}",
        flush=True,
    )
    if target_met:
        reason = "target_hit" if int(adjustment_attempts) <= 0 else "adjusted_target_hit"
    elif int(adjustment_attempts) > 0:
        reason = "adjustment_limit_complete"
    else:
        reason = "one_act_complete"
    return True, reason, after_reading


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
        cfg = _reset_motion_config().get("reverse_turn")
        target_met = _reset_xy_target_ready(offset_reading, cfg if isinstance(cfg, dict) else {})
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
        "no_observed_change_streak_action": None,
        "no_observed_change_streak_count": 0,
        "no_observed_change_streak_duration_ms": 0,
        "stall_recovery_boost": None,
        "stall_recovery_boost_count": 0,
        "stall_guard_triggered": False,
        "stall_guard_reason": "",
        "stall_guard_detail": {},
        "x_curve_samples": [],
        "gap_closure_samples": [],
        "miss_reasons": {},
        "non_win_dist_target_closeness_pct": [],
        "non_win_x_target_closeness_pct": [],
        "non_win_y_target_closeness_pct": [],
        "non_win_target_closeness_pct": [],
        "latest_step1_gap": None,
        "last_step1_win": None,
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
        "step2_attempt_count": 0,
        "step2_count": 0,
        "step2_target_met_count": 0,
        "step2_confirmed_win_count": 0,
        "step2_unconfirmed_win_count": 0,
        "step2_creep_attempt_count": 0,
        "step2_dist_after_mm": [],
        "step2_x_after_mm": [],
        "step2_y_after_mm": [],
        "step2_dist_target_closeness_pct": [],
        "step2_x_target_closeness_pct": [],
        "step2_y_target_closeness_pct": [],
        "step2_target_closeness_pct": [],
        "last_step2_reason": None,
        "y_lock_on_armed": True,
        "y_commit_active": False,
        "y_commit_target_hit_count": 0,
        "holding_s1_transition_commit_sent": False,
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


def _step1_gap_snapshot(
    reading: dict,
    plan: dict,
    *,
    action: str | None = None,
    reason: str | None = None,
) -> dict | None:
    try:
        dist_err = float((plan or {}).get("dist_err"))
        x_err = float((plan or {}).get("x_err"))
        dist_mm = float((reading or {}).get("dist_mm"))
        x_mm = float((reading or {}).get("x_mm"))
    except (TypeError, ValueError):
        return None
    y_cfg = _follow_y_axis_config()
    y_enabled = bool(y_cfg.get("enabled"))
    y_mm = None
    y_err = None
    y_closeness = None
    if y_enabled:
        try:
            y_mm = float((reading or {}).get("y_mm"))
            y_err = float(y_mm - _y_win_target_mm())
            y_closeness = _target_closeness_pct(y_err, _y_win_tol_mm())
        except (TypeError, ValueError):
            y_mm = None
            y_err = None
            y_closeness = None
    dist_closeness = _target_closeness_pct(dist_err, _dist_tol_mm())
    x_closeness = _target_closeness_pct(x_err, _x_tol_mm())
    closeness_values = [float(dist_closeness), float(x_closeness)]
    if y_closeness is not None:
        closeness_values.append(float(y_closeness))
    combined_closeness = sum(closeness_values) / max(1, len(closeness_values))
    snapshot = {
        "action": str(action or (plan or {}).get("action") or "UNKNOWN"),
        "reason": str(reason or (plan or {}).get("reason") or "unknown"),
        "dist_mm": float(dist_mm),
        "x_mm": float(x_mm),
        "dist_err": float(dist_err),
        "x_err": float(x_err),
        "dist_closeness_pct": float(dist_closeness),
        "x_closeness_pct": float(x_closeness),
        "closeness_pct": float(combined_closeness),
    }
    if y_closeness is not None and y_mm is not None and y_err is not None:
        snapshot["y_mm"] = float(y_mm)
        snapshot["y_err"] = float(y_err)
        snapshot["y_closeness_pct"] = float(y_closeness)
    return snapshot


def _update_latest_step1_gap(
    stats: dict,
    reading: dict,
    plan: dict,
    *,
    action: str | None = None,
    reason: str | None = None,
) -> dict | None:
    if not isinstance(stats, dict):
        return None
    snapshot = _step1_gap_snapshot(reading, plan, action=action, reason=reason)
    if not isinstance(snapshot, dict):
        return None
    stats["latest_step1_gap"] = dict(snapshot)
    return snapshot


def _record_win_stats(stats: dict, reading: dict, plan: dict) -> None:
    if not isinstance(stats, dict):
        return
    try:
        dist_err = float((plan or {}).get("dist_err"))
        x_err = float((plan or {}).get("x_err"))
    except (TypeError, ValueError):
        try:
            dist_err = float((reading or {}).get("dist_mm")) - float(_dist_target_mm())
            x_err = float((reading or {}).get("x_mm")) - float(_x_target_mm())
        except (TypeError, ValueError):
            return
    dist_closeness = _target_closeness_pct(dist_err, _dist_tol_mm())
    x_closeness = _target_closeness_pct(x_err, _x_tol_mm())
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
    snapshot = _step1_gap_snapshot(reading, plan, action="HAPPY", reason="win")
    if isinstance(snapshot, dict):
        stats["last_step1_win"] = dict(snapshot)
        stats["latest_step1_gap"] = dict(snapshot)


def _miss_reason_for_plan(plan: dict, reading: dict | None = None) -> str:
    try:
        dist_err = float((plan or {}).get("dist_err"))
        x_err = float((plan or {}).get("x_err"))
    except (TypeError, ValueError):
        return "invalid_reading"
    dist_outside = abs(dist_err) > float(_dist_tol_mm())
    x_outside = abs(x_err) > float(_x_tol_mm())
    y_outside = False
    if bool(_follow_y_axis_config().get("enabled")):
        y_err = None
        if isinstance(reading, dict):
            y_err = _y_err_for_reading(reading, target=_y_win_target_mm())
        if y_err is None:
            try:
                y_err = float((plan or {}).get("y_err"))
            except (TypeError, ValueError):
                y_err = None
        y_outside = y_err is None or not _win_axis_ok(float(y_err), _y_win_tol_mm())
    outside_axes = []
    if dist_outside:
        outside_axes.append("dist")
    if x_outside:
        outside_axes.append("x")
    if y_outside:
        outside_axes.append("y")
    if len(outside_axes) > 1:
        return "_and_".join(outside_axes) + "_outside"
    if x_outside:
        return "x_outside"
    if y_outside:
        return "y_outside"
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
    reason_text = str(reason or _miss_reason_for_plan(plan, reading))
    snapshot = _step1_gap_snapshot(reading, plan, action=action_text, reason=reason_text)
    if not isinstance(snapshot, dict):
        return
    stats.setdefault("non_win_dist_target_closeness_pct", []).append(float(snapshot["dist_closeness_pct"]))
    stats.setdefault("non_win_x_target_closeness_pct", []).append(float(snapshot["x_closeness_pct"]))
    if snapshot.get("y_closeness_pct") is not None:
        stats.setdefault("non_win_y_target_closeness_pct", []).append(float(snapshot["y_closeness_pct"]))
    stats.setdefault("non_win_target_closeness_pct", []).append(float(snapshot["closeness_pct"]))
    stats["latest_step1_gap"] = dict(snapshot)
    stats["last_non_win"] = dict(snapshot)
    closest = stats.get("closest_non_win")
    if not isinstance(closest, dict) or float(snapshot["closeness_pct"]) > float(closest.get("closeness_pct", -1.0)):
        stats["closest_non_win"] = dict(snapshot)


def _record_follow_attempt_stats(stats: dict, reading: dict, plan: dict) -> None:
    if not isinstance(stats, dict):
        return
    stats["follow_attempt_count"] = int(stats.get("follow_attempt_count", 0)) + 1
    action = str((plan or {}).get("action") or "UNKNOWN")
    _bump_stat_count(stats, "act_counts", action)
    reason = _miss_reason_for_plan(plan, reading)
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


def _pending_action_is_y_action(action: str) -> bool:
    text = str(action or "").strip().upper()
    return text.startswith("MAST_") or "_MAST_" in text or text.startswith("Y_LOCK_")


def _y_action_observed_progress(
    pending: dict,
    reading: dict,
    *,
    min_delta_mm: float,
) -> tuple[bool, dict]:
    """For mast moves, progress means y moved toward its y target, not x/dist jitter."""
    detail = classify_mast_y_effect(
        before_y_mm=pending.get("y_mm"),
        after_y_mm=(reading or {}).get("y_mm"),
        cmd=pending.get("cmd") or pending.get("mast_cmd"),
        target_y_mm=pending.get("y_target_mm"),
        min_delta_mm=float(min_delta_mm),
    )
    return bool(detail.get("observed")), detail


def _mark_mast_spool_unreliable(stats: dict, detail: dict) -> None:
    if not isinstance(stats, dict):
        return
    y_cfg = _follow_y_axis_config()
    count = _coerce_int(
        y_cfg.get("spool_reversal_short_act_count"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["spool_reversal_short_act_count"],
        minimum=0,
        maximum=20,
    )
    stats["mast_spool_unreliable_countdown"] = int(count)
    stats["mast_spool_last_unreliable_detail"] = dict(detail or {})
    _bump_stat_count(stats, "miss_reasons", "mast_spool_direction_unreliable")


def _cap_mast_plan_for_unreliable_spool(stats: dict, plan: dict) -> dict:
    if not isinstance(stats, dict) or not isinstance(plan, dict):
        return plan
    if int(stats.get("mast_spool_unreliable_countdown", 0) or 0) <= 0:
        return plan
    if str(plan.get("kind") or "").strip().lower() != "mast":
        return plan
    cmd = str(plan.get("cmd") or plan.get("mast_cmd") or "").strip().lower()
    if cmd not in {"u", "d"}:
        return plan
    y_cfg = _follow_y_axis_config()
    max_ms = _coerce_int(
        y_cfg.get("spool_reversal_mast_max_ms"),
        DEFAULT_FOLLOW_Y_AXIS_CONFIG["spool_reversal_mast_max_ms"],
        minimum=1,
        maximum=_max_mast_act_ms(),
    )
    out = dict(plan)
    try:
        duration_ms = int(round(float(out.get("duration_ms", max_ms))))
    except (TypeError, ValueError):
        duration_ms = int(max_ms)
    if duration_ms > int(max_ms):
        out["duration_ms"] = int(max_ms)
        out["spool_unreliable_capped"] = True
        out["reason"] = f"{out.get('reason', 'mast')}_spool_unreliable_probe"
    return out


def _consume_unreliable_spool_probe(stats: dict, plan: dict) -> None:
    if not isinstance(stats, dict) or not isinstance(plan, dict):
        return
    if not bool(plan.get("spool_unreliable_capped")):
        return
    remaining = int(stats.get("mast_spool_unreliable_countdown", 0) or 0)
    stats["mast_spool_unreliable_countdown"] = max(0, remaining - 1)


def _reset_motion_was_sent(reset_result: dict) -> bool:
    reason = str((reset_result or {}).get("reason") or "").strip()
    return reason not in {
        "invalid_turn_direction",
        "brick_not_confident_before_reset_motion",
        "invalid_reset_start_reading",
        "reverse_turn_unavailable",
    }


def _stall_recovery_next_boost_scale(pending: dict, guard_cfg: dict) -> float:
    try:
        current_scale = float((pending or {}).get("recovery_boost_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        current_scale = 1.0
    step_pct = _coerce_float(
        guard_cfg.get("recovery_boost_step_pct"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["recovery_boost_step_pct"],
        minimum=0.0,
        maximum=100.0,
    )
    max_scale = _coerce_float(
        guard_cfg.get("recovery_boost_max_scale"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["recovery_boost_max_scale"],
        minimum=1.0,
        maximum=3.0,
    )
    if step_pct <= 0.0:
        return float(current_scale)
    return float(min(float(max_scale), max(1.0, float(current_scale)) * (1.0 + (float(step_pct) / 100.0))))


def _stall_recovery_boost_available(pending: dict, guard_cfg: dict, *, is_y_action: bool) -> bool:
    if not bool(guard_cfg.get("recovery_boost_enabled", True)):
        return False
    if bool(is_y_action):
        return False
    try:
        current_scale = float((pending or {}).get("recovery_boost_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        current_scale = 1.0
    max_scale = _coerce_float(
        guard_cfg.get("recovery_boost_max_scale"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["recovery_boost_max_scale"],
        minimum=1.0,
        maximum=3.0,
    )
    return float(current_scale) < float(max_scale) - 1e-6


def _close_trap_stall_limit(pending: dict, guard_cfg: dict, *, is_y_action: bool) -> int | None:
    if bool(is_y_action) or not bool(guard_cfg.get("close_trap_enabled", True)):
        return None
    action = str((pending or {}).get("action") or "").strip().upper()
    cmd = str((pending or {}).get("cmd") or "").strip().lower()
    if cmd != "b" and not action.startswith("BCK"):
        return None
    try:
        dist_err = float((pending or {}).get("dist_err"))
    except (TypeError, ValueError):
        return None
    min_abs = _coerce_float(
        guard_cfg.get("close_trap_min_abs_dist_err_mm"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_min_abs_dist_err_mm"],
        minimum=0.0,
        maximum=300.0,
    )
    if float(dist_err) > -float(min_abs):
        return None
    return _coerce_int(
        guard_cfg.get("close_trap_max_no_change_tries"),
        DEFAULT_ACT_STALL_GUARD_CONFIG["close_trap_max_no_change_tries"],
        minimum=1,
        maximum=10,
    )


def _arm_stall_recovery_boost(stats: dict, pending: dict, guard_cfg: dict) -> None:
    if not isinstance(stats, dict) or not isinstance(pending, dict):
        return
    action = str(pending.get("action") or "UNKNOWN")
    next_scale = _stall_recovery_next_boost_scale(pending, guard_cfg)
    stats["stall_recovery_boost"] = {
        "action": action,
        "scale": float(next_scale),
    }
    stats["stall_recovery_boost_count"] = int(stats.get("stall_recovery_boost_count", 0) or 0) + 1


def _apply_stall_recovery_boost(stats: dict, plan: dict) -> dict:
    if not isinstance(stats, dict) or not isinstance(plan, dict):
        return plan
    boost = stats.get("stall_recovery_boost")
    if not isinstance(boost, dict):
        return plan
    action = str(plan.get("action") or "").strip()
    if action != str(boost.get("action") or "").strip():
        stats["stall_recovery_boost"] = None
        return plan
    kind = str(plan.get("kind") or "").strip().lower()
    if kind not in {"drive", "drive_bias", "drive_nudge", "turn"}:
        stats["stall_recovery_boost"] = None
        return plan
    try:
        scale = float(boost.get("scale"))
    except (TypeError, ValueError):
        stats["stall_recovery_boost"] = None
        return plan
    if scale <= 1.0:
        stats["stall_recovery_boost"] = None
        return plan
    out = dict(plan)
    out["recovery_boost_scale"] = float(scale)
    return out


def _record_observed_after_pending_act(stats: dict, reading: dict) -> dict | None:
    if not isinstance(stats, dict):
        return None
    pending = stats.get("pending_observation")
    if not isinstance(pending, dict):
        return None
    stats["pending_observation"] = None
    try:
        prev_dist = float(pending.get("dist_mm"))
        prev_x = float(pending.get("x_mm"))
        dist_mm = float((reading or {}).get("dist_mm"))
        x_mm = float((reading or {}).get("x_mm"))
    except (TypeError, ValueError):
        return None
    delta_y = 0.0
    try:
        delta_y = abs(float((reading or {}).get("y_mm")) - float(pending.get("y_mm")))
    except (TypeError, ValueError):
        delta_y = 0.0
    delta_dist = abs(float(dist_mm) - float(prev_dist))
    delta_x = abs(float(x_mm) - float(prev_x))
    action = str(pending.get("action") or "UNKNOWN")
    try:
        pending_duration_ms = int(pending.get("duration_ms", 0) or 0)
    except (TypeError, ValueError):
        pending_duration_ms = 0
    guard_cfg = _act_stall_guard_config()
    min_delta = float(guard_cfg.get("min_axis_delta_mm", DEFAULT_ACT_STALL_GUARD_CONFIG["min_axis_delta_mm"]))
    is_y_action = _pending_action_is_y_action(action)
    y_progress_detail = {}
    if is_y_action:
        observed, y_progress_detail = _y_action_observed_progress(
            pending,
            reading,
            min_delta_mm=min_delta,
        )
    else:
        observed = bool(delta_dist >= min_delta or delta_x >= min_delta or delta_y >= min_delta)
    if observed:
        _bump_stat_count(stats, "observed_after_act_counts", action)
        stats["no_observed_change_streak_action"] = None
        stats["no_observed_change_streak_count"] = 0
        stats["no_observed_change_streak_duration_ms"] = 0
        stats["stall_recovery_boost"] = None
    else:
        _bump_stat_count(stats, "no_observed_after_act_counts", action)
        _bump_stat_count(stats, "miss_reasons", "no_observed_change_after_act")
        if is_y_action and mast_effect_is_reversal(y_progress_detail):
            _mark_mast_spool_unreliable(stats, y_progress_detail)
            _bump_stat_count(stats, "miss_reasons", str(y_progress_detail.get("status") or "mast_spool_reversal"))
        if stats.get("no_observed_change_streak_action") == action:
            streak = int(stats.get("no_observed_change_streak_count", 0)) + 1
            streak_duration_ms = int(stats.get("no_observed_change_streak_duration_ms", 0)) + max(0, pending_duration_ms)
        else:
            streak = 1
            streak_duration_ms = max(0, pending_duration_ms)
        stats["no_observed_change_streak_action"] = action
        stats["no_observed_change_streak_count"] = int(streak)
        stats["no_observed_change_streak_duration_ms"] = int(streak_duration_ms)
        if is_y_action:
            max_tries = int(
                guard_cfg.get(
                    "y_max_no_change_tries",
                    DEFAULT_ACT_STALL_GUARD_CONFIG["y_max_no_change_tries"],
                )
            )
            max_duration_ms = int(
                guard_cfg.get(
                    "y_max_no_change_duration_ms",
                    DEFAULT_ACT_STALL_GUARD_CONFIG["y_max_no_change_duration_ms"],
                )
            )
            stalled = int(streak) >= int(max_tries) or int(streak_duration_ms) >= int(max_duration_ms)
        else:
            max_tries = int(guard_cfg.get("max_no_change_tries", DEFAULT_ACT_STALL_GUARD_CONFIG["max_no_change_tries"]))
            max_duration_ms = None
            close_trap_limit = _close_trap_stall_limit(pending, guard_cfg, is_y_action=bool(is_y_action))
            close_trap_limited = close_trap_limit is not None
            if close_trap_limited:
                max_tries = min(int(max_tries), int(close_trap_limit))
            stalled = int(streak) >= int(max_tries)
        if bool(is_y_action):
            close_trap_limited = False
        recovery_boost_available = False
        if not bool(close_trap_limited):
            recovery_boost_available = _stall_recovery_boost_available(
                pending,
                guard_cfg,
                is_y_action=bool(is_y_action),
            )
        if bool(recovery_boost_available):
            _arm_stall_recovery_boost(stats, pending, guard_cfg)
            stalled = False
        if bool(guard_cfg.get("enabled", True)) and bool(stalled):
            stats["stall_guard_triggered"] = True
            stats["stall_guard_reason"] = f"act_stall_no_observed_change:{action}"
            stats["stall_guard_detail"] = {
                "action": action,
                "streak": int(streak),
                "max_no_change_tries": int(max_tries),
                "streak_duration_ms": int(streak_duration_ms),
                "max_no_change_duration_ms": max_duration_ms,
                "min_axis_delta_mm": float(min_delta),
                "delta_dist_mm": float(delta_dist),
                "delta_x_mm": float(delta_x),
                "delta_y_mm": float(delta_y),
                "observed_mode": y_progress_detail.get("mode", "any_axis_delta" if not is_y_action else "y_progress"),
                "close_trap_limited": bool(close_trap_limited),
                "y_progress_detail": dict(y_progress_detail),
                "before": {
                    "dist_mm": float(prev_dist),
                    "x_mm": float(prev_x),
                    "y_mm": pending.get("y_mm"),
                },
                "after": {
                    "dist_mm": float(dist_mm),
                    "x_mm": float(x_mm),
                    "y_mm": (reading or {}).get("y_mm"),
                },
            }
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
    _record_gap_closure_sample(stats, pending, reading, action=action)
    return {
        "action": action,
        "observed": bool(observed),
        "delta_dist_mm": float(delta_dist),
        "delta_x_mm": float(delta_x),
        "delta_y_mm": float(delta_y),
        "y_err": pending.get("y_err"),
        "streak": int(stats.get("no_observed_change_streak_count", 0) or 0),
    }


def _axis_error_sample(pending: dict, reading: dict, *, axis: str) -> dict | None:
    value_key = f"{axis}_mm"
    target_key = f"{axis}_target_mm"
    before_key = f"{axis}_err"
    target = pending.get(target_key)
    if target is None:
        if axis == "dist":
            target = _dist_target_mm()
        elif axis == "x":
            target = _x_target_mm()
        elif axis == "y":
            target = pending.get("mast_y_target_mm", pending.get("y_target_mm", _y_win_target_mm()))
    try:
        before_value = float(pending.get(value_key))
        after_value = float((reading or {}).get(value_key))
        target_value = float(target)
    except (TypeError, ValueError):
        return None
    try:
        before_err = float(pending.get(before_key))
    except (TypeError, ValueError):
        before_err = float(before_value - target_value)
    after_err = float(after_value - target_value)
    before_abs = abs(before_err)
    after_abs = abs(after_err)
    tol = _dist_tol_mm() if axis == "dist" else (_x_tol_mm() if axis == "x" else _y_win_tol_mm())
    overshot = bool(before_err and after_err and (before_err > 0.0) != (after_err > 0.0))
    regression = float(after_abs - before_abs)
    return {
        "axis": axis,
        "before_err": float(before_err),
        "after_err": float(after_err),
        "before_abs": float(before_abs),
        "after_abs": float(after_abs),
        "reduction": float(before_abs - after_abs),
        "regressed": bool(regression > GAP_REGRESSION_EPSILON_MM),
        "regression_mm": float(max(0.0, regression)),
        "overshot": bool(overshot),
        "overshot_within_tolerance": bool(overshot and after_abs <= float(tol)),
        "tol": float(tol),
    }


def _record_gap_closure_sample(stats: dict, pending: dict, reading: dict, *, action: str) -> None:
    if not isinstance(stats, dict) or not isinstance(pending, dict) or not isinstance(reading, dict):
        return
    axes = []
    text = str(action or "").upper()
    cmd = str(pending.get("cmd") or "").strip().lower()
    mast_cmd = str(pending.get("mast_cmd") or "").strip().lower()
    if cmd in {"f", "b"} or text.startswith(("FWD", "BCK")) or pending.get("dist_err") is not None:
        axes.append("dist")
    if cmd in {"l", "r"} or "TURN" in text or "BIAS" in text or "NUDGE" in text or pending.get("x_err") is not None:
        axes.append("x")
    if (
        mast_cmd in {"u", "d"}
        or _pending_action_is_y_action(text)
        or pending.get("y_err") is not None
        or pending.get("mast_y_err") is not None
    ):
        axes.append("y")
    seen = set()
    for axis in axes:
        if axis in seen:
            continue
        seen.add(axis)
        sample = _axis_error_sample(pending, reading, axis=axis)
        if sample is None:
            continue
        sample.update({"action": str(action or "UNKNOWN"), "duration_ms": int(pending.get("duration_ms", 0) or 0)})
        stats.setdefault("gap_closure_samples", []).append(sample)


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


def _record_step2_stats(stats: dict, step2_result: dict) -> None:
    if not isinstance(stats, dict) or not isinstance(step2_result, dict):
        return
    stats["step2_attempt_count"] = int(stats.get("step2_attempt_count", 0)) + 1
    stats["last_step2_reason"] = step2_result.get("reason")
    _bump_stat_count(stats, "act_counts", "STEP2_SEAT")
    if bool(step2_result.get("success")):
        _bump_stat_count(stats, "sent_act_counts", "STEP2_SEAT")
    try:
        creep_attempts = int(step2_result.get("creep_attempts", 0) or 0)
    except (TypeError, ValueError):
        creep_attempts = 0
    if creep_attempts > 0:
        stats["step2_creep_attempt_count"] = int(stats.get("step2_creep_attempt_count", 0)) + int(creep_attempts)
        _bump_stat_count(stats, "act_counts", "STEP2_CREEP_FWD", creep_attempts)
        _bump_stat_count(stats, "sent_act_counts", "STEP2_CREEP_FWD", creep_attempts)
    try:
        visibility_creeps = int(step2_result.get("visibility_recovery_creeps", 0) or 0)
    except (TypeError, ValueError):
        visibility_creeps = 0
    if visibility_creeps > 0:
        stats["step2_visibility_recovery_creep_count"] = int(
            stats.get("step2_visibility_recovery_creep_count", 0)
        ) + int(visibility_creeps)
        _bump_stat_count(stats, "act_counts", "STEP2_VISIBILITY_CREEP_FWD", visibility_creeps)
        _bump_stat_count(stats, "sent_act_counts", "STEP2_VISIBILITY_CREEP_FWD", visibility_creeps)
    precision_counts = step2_result.get("precision_counts")
    if isinstance(precision_counts, dict):
        precision_samples = precision_counts.get("gap_closure_samples")
        if isinstance(precision_samples, list):
            stats.setdefault("gap_closure_samples", []).extend(
                dict(sample) for sample in precision_samples if isinstance(sample, dict)
            )
        for key, action in (
            ("bck", "STEP2_PRECISION_BCK"),
            ("mast_u", "STEP2_PRECISION_MAST_U"),
            ("mast_d", "STEP2_PRECISION_MAST_D"),
        ):
            try:
                count = int(precision_counts.get(key, 0) or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                _bump_stat_count(stats, "act_counts", action, count)
                _bump_stat_count(stats, "sent_act_counts", action, count)
    reading = step2_result.get("reading")
    if not isinstance(reading, dict):
        return
    stats["step2_count"] = int(stats.get("step2_count", 0)) + 1
    if bool(step2_result.get("target_met")):
        stats["step2_target_met_count"] = int(stats.get("step2_target_met_count", 0)) + 1
        stats["step2_confirmed_win_count"] = int(stats.get("step2_confirmed_win_count", 0)) + 1
    elif bool(step2_result.get("success")):
        stats["step2_unconfirmed_win_count"] = int(stats.get("step2_unconfirmed_win_count", 0)) + 1
    for axis in ("dist", "x", "y"):
        try:
            stats.setdefault(f"step2_{axis}_after_mm", []).append(float(reading.get(f"{axis}_mm")))
        except (TypeError, ValueError):
            pass
    closeness = step2_result.get("closeness")
    if not isinstance(closeness, dict):
        closeness = _step2_target_closeness_from_reading(reading)
    if isinstance(closeness, dict):
        for axis in ("dist", "x", "y"):
            value = closeness.get(f"{axis}_target_closeness_pct")
            if value is not None:
                stats.setdefault(f"step2_{axis}_target_closeness_pct", []).append(float(value))
        combined = closeness.get("target_closeness_pct")
        if combined is not None:
            stats.setdefault("step2_target_closeness_pct", []).append(float(combined))


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
        f"avg={reset_close}{_bar_pct(reset_close_val)} dist/x/y_log={reset_dist_close}/{reset_x_close}/{reset_y_close} "
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
        dist_mm = float(snapshot.get("dist_mm"))
        x_mm = float(snapshot.get("x_mm"))
    except (TypeError, ValueError):
        return "N/A"
    y_text = ""
    y_close = snapshot.get("y_closeness_pct")
    if y_close is not None:
        try:
            y_text = (
                f" y={float(y_close):.0f}%, "
                f"y={float(snapshot.get('y_mm')):+.1f}mm y_err={float(snapshot.get('y_err')):+.1f}mm"
            )
        except (TypeError, ValueError):
            y_text = ""
    return (
        f"{close:.0f}% (dist={dist_close:.0f}% x={x_close:.0f}%{y_text}, "
        f"dist={dist_mm:.1f}mm x={x_mm:+.1f}mm, "
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


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(val) < 10.0 and abs(val - round(val)) >= 0.05:
        return f"{val:.1f}ms"
    return f"{val:.0f}ms"


def _gap_duration_wish(sample: dict | None) -> dict | None:
    if not isinstance(sample, dict):
        return None
    try:
        duration_ms = float(sample.get("duration_ms"))
        before_err = float(sample.get("before_err"))
        after_err = float(sample.get("after_err"))
        tol = float(sample.get("tol", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if duration_ms <= 0.0:
        return None
    before_abs = abs(float(before_err))
    after_abs = abs(float(after_err))
    if before_abs <= 0.0:
        return None
    if bool(sample.get("overshot")):
        travel = before_abs + after_abs
        if travel <= 0.0:
            return None
        ideal_ms = duration_ms * before_abs / travel
        delta_ms = max(0.1, duration_ms - ideal_ms)
        return {
            "kind": "overshot",
            "direction": "lower",
            "delta_ms": float(delta_ms),
            "ideal_ms": float(ideal_ms),
            "duration_ms": float(duration_ms),
        }
    if bool(sample.get("regressed")):
        ideal_ms = duration_ms * (before_abs / after_abs) if after_abs > 0.0 else 0.0
        ideal_ms = max(1.0, min(float(duration_ms), float(ideal_ms)))
        delta_ms = max(0.1, duration_ms - ideal_ms)
        return {
            "kind": "regressed",
            "direction": "lower",
            "delta_ms": float(delta_ms),
            "ideal_ms": float(ideal_ms),
            "duration_ms": float(duration_ms),
        }
    reduction = before_abs - after_abs
    if reduction > 0.0 and after_abs > max(0.0, tol):
        ideal_ms = duration_ms * before_abs / reduction
        delta_ms = max(0.1, ideal_ms - duration_ms)
        return {
            "kind": "undershot",
            "direction": "higher",
            "delta_ms": float(delta_ms),
            "ideal_ms": float(ideal_ms),
            "duration_ms": float(duration_ms),
        }
    return None


def _format_gap_duration_wish(sample: dict | None) -> str:
    wish = _gap_duration_wish(sample)
    if not isinstance(wish, dict):
        return "no actionable duration wish yet"
    try:
        before_err = float((sample or {}).get("before_err"))
        after_err = float((sample or {}).get("after_err"))
    except (TypeError, ValueError):
        before_err = 0.0
        after_err = 0.0
    return (
        f"{wish['kind']}: I wish our duration was {_fmt_ms(wish['delta_ms'])} "
        f"{wish['direction']} ({_fmt_ms(wish['duration_ms'])}->{_fmt_ms(wish['ideal_ms'])}, "
        f"{(sample or {}).get('action', 'UNKNOWN')} {before_err:+.1f}->{after_err:+.1f}mm)"
    )


def _problem_axis_name(axis: str | None) -> str:
    axis_key = str(axis or "").strip().lower()
    if axis_key == "dist":
        return "dist"
    if axis_key == "x":
        return "x"
    if axis_key == "y":
        return "y"
    return "motion"


def _gap_problem_candidate(sample: dict | None) -> dict | None:
    if not isinstance(sample, dict):
        return None
    try:
        after_abs = abs(float(sample.get("after_err")))
        before_abs = abs(float(sample.get("before_err")))
        tol = max(0.0, float(sample.get("tol", 0.0) or 0.0))
        reduction = float(sample.get("reduction", before_abs - after_abs))
    except (TypeError, ValueError):
        return None
    axis = _problem_axis_name(sample.get("axis"))
    action = str(sample.get("action") or "UNKNOWN")
    excess = max(0.0, after_abs - tol)
    wish = _format_gap_duration_wish(sample)
    if bool(sample.get("overshot")) and not bool(sample.get("overshot_within_tolerance")):
        return {
            "severity": float(after_abs + excess),
            "problem": f"{axis} overshot after {action}: ended {_fmt_mm(excess)} outside tolerance",
            "fix": f"Shorten that curve/pulse next time; {wish}.",
        }
    if bool(sample.get("overshot_within_tolerance")):
        return None
    if bool(sample.get("regressed")):
        regression = max(0.0, after_abs - before_abs)
        return {
            "severity": float(after_abs + excess + regression),
            "problem": (
                f"{axis} regressed after {action}: gap worsened by {_fmt_mm(regression)} "
                f"to {_fmt_mm(after_abs)}"
            ),
            "fix": f"Shorten/split that curve or avoid that coupling near target; {wish}.",
        }
    if reduction <= 0.0:
        return {
            "severity": float(after_abs + abs(reduction)),
            "problem": f"{axis} did not close after {action}: gap stayed at {_fmt_mm(after_abs)}",
            "fix": f"Increase effect only if this repeats; {wish}.",
        }
    if excess > 0.0:
        return {
            "severity": float(excess),
            "problem": f"{axis} under-closed after {action}: still {_fmt_mm(excess)} outside tolerance",
            "fix": f"Raise that curve/pulse enough to finish the remaining gap; {wish}.",
        }
    return None


def _top_miss_reason_problem(stats: dict) -> dict | None:
    miss_reasons = stats.get("miss_reasons") if isinstance(stats, dict) else None
    if not isinstance(miss_reasons, dict) or not miss_reasons:
        return None
    reason, count = sorted(miss_reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[0]
    reason_text = str(reason)
    fixes = {
        "too_far": "increase forward-distance effect or use a longer forward pulse for this band",
        "too_close": "increase backward-distance effect or lower forward overshoot into this band",
        "x_outside": "increase x turn effect for the current x-gap band",
        "dist_and_x_outside": "prefer simultaneous dist+x correction with a stronger x component",
        "happy_not_stopped": "tighten stopped confirmation or add a shorter settle before accepting live happy",
        "no_observed_change_after_act": "raise the minimum effective pulse for acts that produce no measured change",
        "brick_not_confident": "improve candidate lock or reject ghost frames before motion",
        "y_outside": "keep correcting the mast until the y gate is inside tolerance",
        "dist_and_y_outside": "close dist and y together, then confirm from rest",
        "x_and_y_outside": "close x and y together with smaller coupled acts near target",
        "dist_and_x_and_y_outside": "keep simultaneous gap closure active; all three gates are outside",
    }
    fix = fixes.get(reason_text, "inspect the top miss reason and tune the matching curve/policy")
    return {
        "severity": float(count),
        "problem": f"{reason_text} was the top miss reason ({int(count)}x)",
        "fix": f"{fix}.",
    }


def _weakest_win_axis_problem(stats: dict) -> dict | None:
    values = {
        "dist": _avg(stats.get("win_dist_target_closeness_pct") if isinstance(stats, dict) else None),
        "x": _avg(stats.get("win_x_target_closeness_pct") if isinstance(stats, dict) else None),
        "y": _avg(stats.get("win_y_target_closeness_pct") if isinstance(stats, dict) else None),
    }
    values = {axis: value for axis, value in values.items() if value is not None}
    if not values:
        return None
    axis, value = min(values.items(), key=lambda item: float(item[1]))
    return {
        "severity": float(100.0 - float(value)),
        "problem": f"Step won; weakest axis was {axis} at {float(value):.0f}% closeness",
        "fix": "No emergency change; if this repeats, tune that axis curve toward the center of tolerance.",
    }


def _most_egregious_problem(stats: dict) -> dict:
    candidates = []
    samples = stats.get("gap_closure_samples") if isinstance(stats, dict) else None
    if isinstance(samples, list):
        for sample in samples:
            candidate = _gap_problem_candidate(sample)
            if candidate is not None:
                candidates.append(candidate)
    miss_candidate = _top_miss_reason_problem(stats)
    if miss_candidate is not None:
        candidates.append(miss_candidate)
    win_candidate = _weakest_win_axis_problem(stats)
    if win_candidate is not None:
        candidates.append(win_candidate)
    if not candidates:
        return {
            "severity": 0.0,
            "problem": "No single egregious problem captured",
            "fix": "Keep collecting per-act gap samples; add instrumentation to any path that still reports zero samples.",
        }
    return max(candidates, key=lambda item: float(item.get("severity", 0.0)))


def _format_most_egregious_problem(stats: dict) -> list[str]:
    problem = _most_egregious_problem(stats)
    return [
        "",
        "| Single Biggest Problem | What must change |",
        "|---|---|",
        f"| {problem.get('problem')} | {problem.get('fix')} |",
    ]


def _format_gap_closure_learning(samples: list | None) -> list[str]:
    rows = [
        "",
        "| Gap Closure Learning | Value |",
        "|---|---:|",
    ]
    if not isinstance(samples, list) or not samples:
        rows.append("| Samples | 0 |")
        return rows
    valid = [s for s in samples if isinstance(s, dict)]
    rows.append(f"| Samples | {len(valid)} |")
    for axis in ("dist", "x", "y"):
        axis_samples = [s for s in valid if str(s.get("axis")) == axis]
        if not axis_samples:
            rows.append(f"| {axis} samples | 0 |")
            continue
        reductions = [float(s.get("reduction", 0.0)) for s in axis_samples]
        improved = sum(1 for value in reductions if value > GAP_REGRESSION_EPSILON_MM)
        overshoots = sum(1 for s in axis_samples if bool(s.get("overshot")))
        meaningful_overshoots = sum(
            1
            for s in axis_samples
            if bool(s.get("overshot")) and not bool(s.get("overshot_within_tolerance"))
        )
        regressions = sum(1 for s in axis_samples if bool(s.get("regressed")))
        no_close = sum(1 for value in reductions if abs(float(value)) <= GAP_REGRESSION_EPSILON_MM)
        underclosed = sum(
            1
            for s in axis_samples
            if (
                float(s.get("reduction", 0.0)) > GAP_REGRESSION_EPSILON_MM
                and float(s.get("after_abs", 0.0)) > float(s.get("tol", 0.0))
                and not bool(s.get("overshot"))
            )
        )
        worst_regression = max(
            (s for s in axis_samples if bool(s.get("regressed"))),
            key=lambda row: float(row.get("regression_mm", 0.0)),
            default=None,
        )
        latest = axis_samples[-1]
        latest_wish = next(
            (
                s
                for s in reversed(axis_samples)
                if _gap_duration_wish(s) is not None
            ),
            None,
        )
        rows.extend(
            [
                f"| {axis} improved | {improved}/{len(axis_samples)} |",
                f"| {axis} under-closed | {underclosed}/{len(axis_samples)} |",
                f"| {axis} no-close | {no_close}/{len(axis_samples)} |",
                f"| {axis} regressed | {regressions}/{len(axis_samples)} |",
                f"| {axis} overshot | {overshoots}/{len(axis_samples)} |",
                f"| {axis} meaningful overshot | {meaningful_overshoots}/{len(axis_samples)} |",
                f"| {axis} avg reduction | {_fmt_mm(_avg(reductions))} |",
                (
                    f"| {axis} last | {latest.get('action')} "
                    f"{float(latest.get('before_err', 0.0)):+.1f}->{float(latest.get('after_err', 0.0)):+.1f}mm |"
                ),
                f"| {axis} duration wish | {_format_gap_duration_wish(latest_wish)} |",
            ]
        )
        if isinstance(worst_regression, dict):
            rows.append(
                f"| {axis} worst regression | {worst_regression.get('action')} "
                f"{float(worst_regression.get('before_err', 0.0)):+.1f}->{float(worst_regression.get('after_err', 0.0)):+.1f}mm "
                f"({_fmt_mm(float(worst_regression.get('regression_mm', 0.0)))} worse) |"
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
    avg_step2_close_val = _avg_stat(stats, "step2_target_closeness_pct")
    win_close_values = stats.get("win_target_closeness_pct")
    win_dist_values = stats.get("win_dist_target_closeness_pct")
    win_x_values = stats.get("win_x_target_closeness_pct")
    win_y_values = stats.get("win_y_target_closeness_pct")
    reset_close_values = stats.get("reset_target_closeness_pct")
    reset_dist_values = stats.get("reset_dist_target_closeness_pct")
    reset_x_values = stats.get("reset_x_target_closeness_pct")
    reset_y_values = stats.get("reset_y_target_closeness_pct")
    step2_close_values = stats.get("step2_target_closeness_pct")
    step2_dist_values = stats.get("step2_dist_target_closeness_pct")
    step2_x_values = stats.get("step2_x_target_closeness_pct")
    step2_y_values = stats.get("step2_y_target_closeness_pct")
    step2_dist_after = _avg(stats.get("step2_dist_after_mm") if isinstance(stats, dict) else None)
    step2_x_after = _avg(stats.get("step2_x_after_mm") if isinstance(stats, dict) else None)
    step2_y_after = _avg(stats.get("step2_y_after_mm") if isinstance(stats, dict) else None)
    step2_avg_after_text = "N/A"
    if step2_dist_after is not None or step2_x_after is not None or step2_y_after is not None:
        step2_avg_after_text = (
            f"dist {_fmt_mm(step2_dist_after)}, x {_fmt_mm(step2_x_after)}, y {_fmt_mm(step2_y_after)}"
        )
    step2_missing = _step2_missing_target_keys()
    if step2_missing:
        step2_avg_after_text = "targets pending: " + ", ".join(step2_missing)
    attempts = int(stats.get("follow_attempt_count", 0))
    confident = int(stats.get("confident_sample_count", 0))
    sample_count = int(stats.get("sample_count", 0))
    not_confident = int(stats.get("not_confident_count", 0))
    wins = int(stats.get("win_count", 0))
    reset_attempts = int(stats.get("reset_attempt_count", 0))
    reset_hits = int(stats.get("reset_target_met_count", 0))
    step2_attempts = int(stats.get("step2_attempt_count", 0))
    step2_hits = int(stats.get("step2_target_met_count", 0))
    step2_confirmed = int(stats.get("step2_confirmed_win_count", 0))
    step2_unconfirmed = int(stats.get("step2_unconfirmed_win_count", 0))
    step2_creeps = int(stats.get("step2_creep_attempt_count", 0))
    act_counts = _format_count_items(stats.get("act_counts"))
    sent_act_counts = _format_count_items(stats.get("sent_act_counts"))
    blocked_act_counts = _format_count_items(stats.get("blocked_act_counts"))
    observed_counts = _format_count_items(stats.get("observed_after_act_counts"))
    no_observed_counts = _format_count_items(stats.get("no_observed_after_act_counts"))
    miss_reasons = _format_count_items(stats.get("miss_reasons"))
    latest_step1_gap = _format_snapshot(stats.get("latest_step1_gap"))
    last_step1_win = _format_snapshot(stats.get("last_step1_win"))
    closest_non_win = _format_snapshot(stats.get("closest_non_win"))
    last_non_win = _format_snapshot(stats.get("last_non_win"))
    rows = [
            "| Target | Samples | Hits | Close avg±sd | Dist avg±sd | X avg±sd | Y avg±sd | Avg after |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            f"| Step 1 Win | {int(stats.get('win_count', 0))} | {int(stats.get('win_count', 0))} | "
            f"{_pct_avg_std_text(win_close_values)} {_bar_pct(avg_win_close_val)} | "
            f"{_pct_avg_std_text(win_dist_values)} | {_pct_avg_std_text(win_x_values)} | {_pct_avg_std_text(win_y_values)} | "
            f"target dist {_dist_target_mm():.1f}mm, x {_x_target_mm():.1f}mm, y {_y_win_target_mm():.1f}mm |",
            f"| Step 1 Reset | {int(stats.get('reset_count', 0))}/{int(stats.get('reset_attempt_count', 0))} | "
            f"{int(stats.get('reset_target_met_count', 0))}/{int(stats.get('reset_count', 0))} | "
            f"{_pct_avg_std_text(reset_close_values)} {_bar_pct(avg_reset_close_val)} | "
            f"{_pct_avg_std_text(reset_dist_values)} | {_pct_avg_std_text(reset_x_values)} | {_pct_avg_std_text(reset_y_values)} | {avg_after_text} |",
            f"| Step 2 Win | {int(stats.get('step2_count', 0))}/{int(stats.get('step2_attempt_count', 0))} | "
            f"{int(stats.get('step2_target_met_count', 0))}/{int(stats.get('step2_count', 0))} | "
            f"{_pct_avg_std_text(step2_close_values)} {_bar_pct(avg_step2_close_val)} | "
            f"{_pct_avg_std_text(step2_dist_values)} | {_pct_avg_std_text(step2_x_values)} | {_pct_avg_std_text(step2_y_values)} | {step2_avg_after_text} |",
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
            f"| Step 2 attempts | {step2_attempts} |",
            f"| Step 2 target hits | {step2_hits} |",
            f"| Step 2 confirmed wins | {step2_confirmed} |",
            f"| Step 2 unconfirmed wins | {step2_unconfirmed} |",
            f"| Step 2 recovery creeps | {step2_creeps} |",
            "",
            "| Movement acts | Count |",
            "|---|---:|",
            f"| Planned: {act_counts} | {sum(int(v) for v in (stats.get('act_counts') or {}).values())} |",
            f"| Sent: {sent_act_counts} | {sum(int(v) for v in (stats.get('sent_act_counts') or {}).values())} |",
            f"| Blocked: {blocked_act_counts} | {sum(int(v) for v in (stats.get('blocked_act_counts') or {}).values())} |",
            f"| Observed change after act: {observed_counts} | {sum(int(v) for v in (stats.get('observed_after_act_counts') or {}).values())} |",
            f"| No observed change after act: {no_observed_counts} | {sum(int(v) for v in (stats.get('no_observed_after_act_counts') or {}).values())} |",
            f"| Stall guard | {'TRIGGERED: ' + str(stats.get('stall_guard_reason')) if bool(stats.get('stall_guard_triggered')) else 'ok'} |",
            "",
            "| Why not more wins? | Evidence |",
            "|---|---|",
            f"| Miss reasons | {miss_reasons} |",
            f"| Final Step 1 read | {latest_step1_gap} |",
            f"| Last Step 1 win | {last_step1_win} |",
            f"| Closest non-win | {closest_non_win} |",
            f"| Last non-win | {last_non_win} |",
        ]
    rows.extend(_format_most_egregious_problem(stats))
    rows.extend(_format_x_curve_learning(stats.get("x_curve_samples")))
    rows.extend(_format_gap_closure_learning(stats.get("gap_closure_samples")))
    return "\n".join(rows)


INVISIBLE_STOP_FRAMES = 1   # stop motors on the first not-visible frame


def _confirm_stopped_happy(
    vision: BrickDetector,
    robot: Robot,
    stats: dict,
) -> tuple[bool, str, dict | None, dict | None]:
    """Stop, let motion settle, then wait briefly for happy readings from rest."""
    cfg = _win_confirmation_config()
    _stop_robot(robot)
    settle_s = float(cfg.get("settle_s", DEFAULT_WIN_CONFIRMATION_CONFIG["settle_s"]))
    if settle_s > 0.0:
        time.sleep(settle_s)

    confirm_frames = int(cfg.get("confirm_frames", DEFAULT_WIN_CONFIRMATION_CONFIG["confirm_frames"]))
    allowance_s = float(
        cfg.get(
            "single_win_allowance_s",
            DEFAULT_WIN_CONFIRMATION_CONFIG["single_win_allowance_s"],
        )
    )
    deadline = time.monotonic() + max(0.0, float(allowance_s))
    max_confirmation_samples = max(
        max(1, confirm_frames),
        int((max(0.0, float(allowance_s)) / max(0.001, float(LOOP_S))) + 2),
    )
    passing_frames = 0
    passing_samples = 0
    last_reading = None
    last_plan = None
    last_reason = "happy_not_stopped"
    first_sample = True
    sample_count = 0
    while sample_count < max_confirmation_samples and (first_sample or time.monotonic() <= deadline):
        first_sample = False
        sample_count += 1
        reading = _read_brick_measurement(vision, jump_guard=True)
        last_reading = reading
        stats["sample_count"] = int(stats.get("sample_count", 0)) + 1
        if not bool(reading.get("confident")):
            reading = _wait_for_visibility_recovery(
                vision,
                robot,
                reading,
                context="happy_confirmation",
            )
            last_reading = reading
        if not bool(reading.get("confident")):
            passing_frames = 0
            last_plan = None
            last_reason = "brick_not_confident_after_stop"
            if time.monotonic() > deadline:
                break
            time.sleep(min(LOOP_S, max(0.0, deadline - time.monotonic())))
            continue
        try:
            conf_pct = float(reading.get("conf"))
        except (TypeError, ValueError):
            conf_pct = -1.0
        min_win_conf = float(cfg.get("min_confidence_pct", DEFAULT_WIN_CONFIRMATION_CONFIG["min_confidence_pct"]))
        if conf_pct < min_win_conf:
            passing_frames = 0
            last_plan = None
            last_reason = "win_low_confidence_after_stop"
            if time.monotonic() > deadline:
                break
            time.sleep(min(LOOP_S, max(0.0, deadline - time.monotonic())))
            continue
        stats["confident_sample_count"] = int(stats.get("confident_sample_count", 0)) + 1
        plan = _follow_action_plan(reading)
        last_plan = plan
        if _holding_s1_transition_commit_plan(stats, reading, plan) is not None:
            passing_frames = 0
            last_reason = "holding_s1_transition_commit_pending"
            if time.monotonic() > deadline:
                break
            time.sleep(min(LOOP_S, max(0.0, deadline - time.monotonic())))
            continue
        if str(plan.get("kind")) != "hold":
            passing_frames = 0
            last_reason = "happy_not_stopped"
            if time.monotonic() > deadline:
                break
            time.sleep(min(LOOP_S, max(0.0, deadline - time.monotonic())))
            continue
        passing_frames += 1
        passing_samples += 1
        if passing_frames >= max(1, confirm_frames):
            return True, "stopped_happy_confirmed", last_reading, last_plan
        if passing_samples >= max(1, confirm_frames):
            return True, "stopped_happy_confirmed_within_allowance", last_reading, last_plan
        if time.monotonic() > deadline:
            break
        time.sleep(min(LOOP_S, max(0.0, deadline - time.monotonic())))
    if last_reason in {"brick_not_confident_after_stop", "win_low_confidence_after_stop"}:
        stats["not_confident_count"] = int(stats.get("not_confident_count", 0)) + 1
        _bump_stat_count(stats, "miss_reasons", last_reason)
    else:
        _bump_stat_count(stats, "miss_reasons", "happy_not_stopped")
        if isinstance(last_plan, dict) and isinstance(last_reading, dict):
            _record_non_win_stats(
                stats,
                last_reading,
                last_plan,
                action="HAPPY_REJECT",
                reason="happy_not_stopped",
            )
    return False, last_reason, last_reading, last_plan


def _record_and_print_step1_win(stats: dict, reading: dict, plan: dict) -> tuple[float, float, str, float]:
    dist_err = float(plan["dist_err"])
    x_err = float(plan["x_err"])
    y_mm = reading.get("y_mm")
    try:
        y_text = f" y_err={float(y_mm) - _y_win_target_mm():+.1f}mm"
    except (TypeError, ValueError):
        y_text = ""
    try:
        conf = float(reading["conf"])
    except (TypeError, ValueError):
        conf = 0.0
    stats["win_count"] = int(stats.get("win_count", 0)) + 1
    _record_win_stats(stats, reading, plan)
    win_dist_close = _target_closeness_pct(dist_err, _dist_tol_mm())
    win_x_close = _target_closeness_pct(x_err, _x_tol_mm())
    win_y_close = None
    win_y_err = _y_err_for_reading(reading, target=_y_win_target_mm())
    if win_y_err is not None and bool(_follow_y_axis_config().get("enabled")):
        win_y_close = _target_closeness_pct(win_y_err, _y_win_tol_mm())
    close_text = f"dist={win_dist_close:.0f}% x={win_x_close:.0f}%"
    if win_y_close is not None:
        close_text = f"{close_text} y={win_y_close:.0f}%"
    print(
        f"[FOLLOW] WIN #{int(stats['win_count'])}: "
        f"dist_err={dist_err:+.1f}mm x_err={x_err:+.1f}mm "
        f"{y_text} close=({close_text}) conf={conf:.0f}%",
        flush=True,
    )
    return dist_err, x_err, y_text, conf


def _follow_loop(
    vision: BrickDetector,
    robot: Robot,
    duration_s: float = 40.0,
    *,
    reset_after_win: bool = True,
    stop_after_win: bool = False,
    stop_after_step2: bool = False,
    complete_after_step3: bool = False,
    max_cycles: int | None = None,
    step2_probe_before_forward: bool = False,
    debug_mode: bool = False,
) -> dict:
    last_action = ""
    print_ticker = 0
    miss_count = 0
    completed_cycles = 0
    deadline = time.monotonic() + duration_s
    step1_started_at = time.monotonic()
    step1_attempt_limit_s = _step_attempt_limit_s()
    stats = _new_game_stats()

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        loop_wait_s = float(LOOP_S)

        reading = _read_brick_measurement(vision, jump_guard=True)
        stats["sample_count"] = int(stats.get("sample_count", 0)) + 1
        found = bool(reading.get("confident"))

        if not found:
            reading_reason = str(reading.get("reason") or "")
            if reading_reason == "ghost_jump_hard_rejected":
                reading, hard_stop = _observe_after_hard_ghost_jump(
                    vision,
                    robot,
                    stats,
                    reading,
                    context="hard_ghost_jump",
                )
                if bool(hard_stop):
                    return stats
                found = bool(reading.get("confident"))
            elif reading_reason in {"ghost_jump_unconfirmed", "ghost_jump_confirmed_rejected"}:
                _bump_stat_count(stats, "miss_reasons", reading_reason)
                delta = reading.get("ghost_jump_delta") if isinstance(reading.get("ghost_jump_delta"), dict) else {}
                print(
                    "[FOLLOW] GHOST_JUMP? "
                    f"dist={float(delta.get('dist', 0.0)):.1f}mm "
                    f"x={float(delta.get('x', 0.0)):.1f}mm "
                    f"y={float(delta.get('y', 0.0) or 0.0):.1f}mm; "
                    "holding still and not closing gaps on this candidate.",
                    flush=True,
                )
            elif reading_reason == "reacquiring_stable_brick_lock":
                _bump_stat_count(stats, "miss_reasons", "reacquiring_stable_brick_lock")
                if isinstance(stats.get("pending_observation"), dict):
                    stats["pending_observation"] = None
                    _bump_stat_count(stats, "miss_reasons", "post_act_observation_cleared_for_reacquire")
            if not found:
                reading = _wait_for_visibility_recovery(
                    vision,
                    robot,
                    reading,
                    context="follow_loop",
                    jump_guard=True,
                )
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
        observed_detail = _record_observed_after_pending_act(stats, reading)
        if bool(stats.get("stall_guard_triggered")):
            detail = stats.get("stall_guard_detail") if isinstance(stats.get("stall_guard_detail"), dict) else {}
            _stop_robot(robot)
            duration_text = ""
            if detail.get("max_no_change_duration_ms") is not None:
                duration_text = (
                    f" or {int(detail.get('streak_duration_ms', 0) or 0)}/"
                    f"{int(detail.get('max_no_change_duration_ms', 0) or 0)}ms aggregate movement"
                )
            print(
                "[FOLLOW] HARD STOP: act stall guard triggered. "
                f"{detail.get('action', 'UNKNOWN')} produced no >= "
                f"{float(detail.get('min_axis_delta_mm', 1.0)):.1f}mm "
                f"{'y-target progress' if _pending_action_is_y_action(str(detail.get('action', ''))) else 'change in dist/x/y'} "
                f"for {int(detail.get('streak', 0) or 0)}/"
                f"{int(detail.get('max_no_change_tries', 0) or 0)} tries"
                f"{duration_text}. "
                f"last_delta=(dist {float(detail.get('delta_dist_mm', 0.0)):.1f}, "
                f"x {float(detail.get('delta_x_mm', 0.0)):.1f}, "
                f"y {float(detail.get('delta_y_mm', 0.0)):.1f})mm",
                flush=True,
            )
            break
        miss_count = 0
        dist_mm = float(reading["dist_mm"])
        x_mm = float(reading["x_mm"])
        y_mm = reading.get("y_mm")
        y_text = ""
        try:
            y_text = f" y_err={float(y_mm) - _y_win_target_mm():+.1f}mm"
        except (TypeError, ValueError):
            y_text = ""
        conf = float(reading["conf"])
        dist_err_now = float(dist_mm - _dist_target_mm())
        x_err_now = float(x_mm - _x_target_mm())

        if bool(stats.get("y_commit_active")) and not (
            _win_axis_ok(dist_err_now, _dist_tol_mm()) and _win_axis_ok(x_err_now, _x_tol_mm())
        ):
            _stop_robot(robot)
            stats["y_commit_active"] = False
            _bump_stat_count(stats, "miss_reasons", "y_commit_released_xz_outside")
            action = "Y_COMMIT_RELEASE"
            print(
                f"[FOLLOW] {action:<10} dist_err={dist_err_now:+.1f}mm  "
                f"x_err={x_err_now:+.1f}mm {y_text} conf={conf:.0f}% "
                "(x/z left gate; resuming normal follow)",
                flush=True,
            )
            last_action = action

        if bool(stats.get("y_commit_active")):
            plan = _y_commit_action_plan(reading, dist_err=dist_err_now, x_err=x_err_now)
            action = str(plan.get("action") or "Y_COMMIT")
            kind = str(plan.get("kind") or "").strip().lower()
            if kind == "hold":
                _stop_robot(robot)
                stats["y_commit_active"] = False
                stats["y_commit_target_hit_count"] = int(stats.get("y_commit_target_hit_count", 0)) + 1
                print(
                    f"[FOLLOW] {action:<10} dist_err={dist_err_now:+.1f}mm  "
                    f"x_err={x_err_now:+.1f}mm {y_text}  conf={conf:.0f}% "
                    "(x/z held during y commit)",
                    flush=True,
                )
                last_action = action
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue
            if kind == "wait":
                _stop_robot(robot)
                _bump_stat_count(stats, "miss_reasons", "missing_y_after_commit")
                print(
                    f"[FOLLOW] {action:<10} dist_err={dist_err_now:+.1f}mm  "
                    f"x_err={x_err_now:+.1f}mm y_err=N/A  conf={conf:.0f}% "
                    "(x/z held during y commit)",
                    flush=True,
                )
                last_action = action
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue

            plan = _cap_mast_plan_for_unreliable_spool(stats, plan)
            plan = _cap_mast_up_plan_to_y_ceiling(plan, reading)
            plan = _cap_mast_up_plan_to_budget(stats, plan)
            plan = _cap_mast_plan_to_max_duration(plan)
            plan = _cap_near_target_wheel_plan_to_crawl(plan)
            action = str(plan.get("action") or action)
            exceeded, used_up_ms, planned_up_ms, max_up_ms = _mast_up_budget_exceeded(stats, plan)
            if exceeded:
                _stop_robot(robot)
                _bump_stat_count(stats, "miss_reasons", "mast_up_budget_guard")
                stats["debug_stop_reading"] = dict(reading) if isinstance(reading, dict) else reading
                stats["last_action"] = "MAST_UP_BUDGET_GUARD"
                print(
                    f"[FOLLOW] HARD STOP: mast-up budget guard blocked {action}. "
                    f"Already used {used_up_ms}ms; planned {planned_up_ms}ms would exceed "
                    f"{max_up_ms}ms. Holding still instead of risking a high-mast overshoot.",
                    flush=True,
                )
                return stats

            _record_follow_attempt_stats(stats, reading, plan)
            send_result = _execute_follow_action(robot, plan, reading)
            _record_send_result(stats, action, send_result)
            if not _send_result_blocked(send_result):
                _record_mast_up_budget(stats, plan, send_result)
            duration_text = _sent_motion_duration_text(plan, send_result)
            if not _send_result_blocked(send_result):
                if _plan_commits_y_lowering(plan):
                    stats["y_commit_active"] = True
                _consume_unreliable_spool_probe(stats, plan)
                _reset_follow_reading_history(
                    vision,
                    allow_large_dist_jump=_plan_allows_large_dist_reacquire(plan),
                )
                loop_wait_s = _post_action_wait_s(plan, send_result)
                pending = {
                    "action": action,
                    "dist_mm": dist_mm,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "cmd": plan.get("cmd"),
                    "dist_err": dist_err_now,
                    "x_err": x_err_now,
                    "y_target_mm": plan.get("y_target_mm"),
                    "y_err": plan.get("y_err"),
                }
                try:
                    duration_val = (send_result or {}).get("duration_ms") if isinstance(send_result, dict) else None
                    pending["duration_ms"] = int(duration_val if duration_val is not None else plan.get("duration_ms", PULSE_MS))
                except (TypeError, ValueError):
                    pending["duration_ms"] = int(PULSE_MS)
                stats["pending_observation"] = pending
                _record_distance_act_state(stats, action, plan)
            print(
                f"[FOLLOW] {action:<10} dist_err={dist_err_now:+.1f}mm  "
                f"x_err={x_err_now:+.1f}mm {y_text} {duration_text} conf={conf:.0f}% "
                "(x/z held during y commit)",
                flush=True,
            )
            last_action = action
            elapsed = time.monotonic() - loop_start
            remaining = float(loop_wait_s) - elapsed
            if remaining > 0:
                time.sleep(remaining)
            if not _send_result_blocked(send_result):
                _stop_robot(robot)
            continue

        if _should_run_y_lock_on(stats, reading):
            action = "Y_LOCK_MAST_D"
            send_result = _lock_on_mast_down(robot, reading)
            lock_plan = {
                "kind": "mast",
                "action": action,
                "duration_ms": int((send_result or {}).get("duration_ms", 400) if isinstance(send_result, dict) else 400),
            }
            duration_text = _sent_motion_duration_text(lock_plan, send_result)
            stats["follow_attempt_count"] = int(stats.get("follow_attempt_count", 0)) + 1
            _bump_stat_count(stats, "act_counts", action)
            _bump_stat_count(stats, "miss_reasons", "y_lock_on")
            _record_send_result(stats, action, send_result)
            stats["y_lock_on_armed"] = False
            if not (isinstance(send_result, dict) and bool(send_result.get("blocked"))):
                stats["y_commit_active"] = True
                stats["pending_observation"] = {
                    "action": action,
                    "dist_mm": dist_mm,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "cmd": "d",
                    "duration_ms": int((send_result or {}).get("duration_ms", 400) if isinstance(send_result, dict) else 400),
                    "y_target_mm": _y_win_target_mm(),
                }
            print(
                f"[FOLLOW] {action:<10} dist={dist_mm:.1f}mm x={x_mm:+.1f}mm {y_text} {duration_text} conf={conf:.0f}%",
                flush=True,
            )
            lock_wait_s = float(LOOP_S)
            if not _send_result_blocked(send_result):
                _reset_follow_reading_history(vision, allow_large_dist_jump=False)
                lock_wait_s = _post_action_wait_s(lock_plan, send_result)
            elapsed = time.monotonic() - loop_start
            if (remaining := float(lock_wait_s) - elapsed) > 0:
                time.sleep(remaining)
            if not _send_result_blocked(send_result):
                _stop_robot(robot)
            continue

        plan = _follow_action_plan(reading)
        transition_plan = _holding_s1_transition_commit_plan(stats, reading, plan)
        if transition_plan is not None:
            plan = transition_plan
            stats["holding_s1_transition_commit_sent"] = True
            _bump_stat_count(stats, "miss_reasons", "holding_s1_transition_commit")
        dist_err = float(plan["dist_err"])
        x_err = float(plan["x_err"])
        action = str(plan.get("action") or "HOLD")
        _update_latest_step1_gap(stats, reading, plan, action=action, reason=plan.get("reason"))
        if str(plan.get("kind") or "").strip().lower() == "wait":
            _stop_robot(robot)
            wait_reason = str(plan.get("reason") or "wait")
            _bump_stat_count(stats, "miss_reasons", wait_reason)
            if wait_reason == "pickup_suspected_far_low":
                confirmed_pickup, confirmed_reading, pickup_samples = _confirm_pickup_suspected_reading(
                    vision,
                    reading,
                )
                if not bool(confirmed_pickup):
                    stats["pickup_suspected_unconfirmed_count"] = int(
                        stats.get("pickup_suspected_unconfirmed_count", 0) or 0
                    ) + 1
                    _bump_stat_count(stats, "miss_reasons", "pickup_suspect_unconfirmed")
                    print(
                        "[FOLLOW] Pickup-suspect read was not confirmed across "
                        f"{len(pickup_samples)} frames; continuing from stopped observation.",
                        flush=True,
                    )
                    continue
                stats["pickup_suspected_stop"] = True
                stats["pickup_suspected_confirm_frames"] = len(pickup_samples)
                stats["pickup_suspected_confirm_samples"] = pickup_samples
                stats["debug_stop_reading"] = dict(confirmed_reading) if isinstance(confirmed_reading, dict) else confirmed_reading
                stats["last_action"] = action
                _record_non_win_stats(stats, confirmed_reading, plan, action=action, reason=wait_reason)
                print(
                    f"[FOLLOW] HARD STOP: pickup suspected "
                    f"(confirmed over {len(pickup_samples)} frames; "
                    f"dist={dist_mm:.1f}mm x={x_mm:+.1f}mm {y_text} conf={conf:.0f}%). "
                    "Holding still until the brick is cleared.",
                    flush=True,
                )
                return stats
            print_ticker += 1
            if action != last_action or print_ticker >= 20:
                print(
                    f"[FOLLOW] {action:<10} dist_err={dist_err:+.1f}mm  "
                    f"x_err={x_err:+.1f}mm {y_text} conf={conf:.0f}%",
                    flush=True,
                )
                print_ticker = 0
            last_action = action
            elapsed = time.monotonic() - loop_start
            wait_s = max(float(LOOP_S), _y_motion_coast_settle_s())
            if (remaining := wait_s - elapsed) > 0:
                time.sleep(remaining)
            continue
        if (
            _should_wait_after_tiny_y_no_observed(observed_detail)
            and str(plan.get("kind") or "").strip().lower() != "hold"
        ):
            _stop_robot(robot)
            wait_s = _tiny_y_no_observed_wait_s()
            _bump_stat_count(stats, "miss_reasons", "tiny_y_coast_observe")
            if wait_s > 0.0:
                time.sleep(wait_s)
            last_action = "Y_COAST_OBSERVE"
            continue
        if (
            float(step1_attempt_limit_s) > 0.0
            and str(plan.get("kind") or "").strip().lower() != "hold"
            and (time.monotonic() - float(step1_started_at)) >= float(step1_attempt_limit_s)
        ):
            if _should_grace_step1_timeout_for_y(stats, plan, reading):
                stats["step1_timeout_y_correction_grace_count"] = (
                    int(stats.get("step1_timeout_y_correction_grace_count", 0) or 0) + 1
                )
                _bump_stat_count(stats, "miss_reasons", "step1_timeout_y_correction_grace")
                print(
                    f"[FOLLOW] STEP1_TIMEOUT_Y_GRACE "
                    f"{int(stats['step1_timeout_y_correction_grace_count'])}/"
                    f"{int(_win_confirmation_config().get('timeout_y_correction_grace_acts', 0) or 0)}: "
                    f"dist_err={dist_err:+.1f}mm x_err={x_err:+.1f}mm {y_text} "
                    f"conf={conf:.0f}%; allowing corrective {action} act.",
                    flush=True,
                )
            else:
                _stop_robot(robot)
                stats["step1_attempt_timeout"] = True
                stats["debug_stop_reading"] = dict(reading) if isinstance(reading, dict) else reading
                stats["last_action"] = "STEP1_ATTEMPT_TIMEOUT"
                _bump_stat_count(stats, "miss_reasons", "step1_attempt_timeout")
                _record_non_win_stats(
                    stats,
                    reading,
                    plan,
                    action="STEP1_ATTEMPT_TIMEOUT",
                    reason="step1_attempt_timeout",
                )
                print(
                    f"[FOLLOW] STEP1_TIMEOUT after {float(step1_attempt_limit_s):.1f}s: "
                    f"dist_err={dist_err:+.1f}mm x_err={x_err:+.1f}mm {y_text} "
                    f"conf={conf:.0f}%; stopped before another {action} act.",
                    flush=True,
                )
                if bool(debug_mode):
                    stats["debug_mode_terminated"] = True
                return stats

        if (bool(debug_mode) or bool(stop_after_win)) and _should_stop_confirm_for_dist_pingpong(stats, plan):
            previous_dist_err = float(stats.get("last_distance_act_dist_err"))
            previous_action = str(stats.get("last_distance_act_action") or "DIST")
            print(
                f"[FOLLOW] DIST_PINGPONG_CHECK previous={previous_action} "
                f"prev_dist_err={previous_dist_err:+.1f}mm now_dist_err={dist_err:+.1f}mm; "
                "stopping for confirmation instead of reversing immediately.",
                flush=True,
            )
            confirmed, confirm_reason, confirmed_reading, confirmed_plan = _confirm_stopped_happy(
                vision,
                robot,
                stats,
            )
            if confirmed:
                reading = confirmed_reading if isinstance(confirmed_reading, dict) else reading
                plan = confirmed_plan if isinstance(confirmed_plan, dict) else plan
                dist_err, x_err, y_text, conf = _record_and_print_step1_win(stats, reading, plan)
                if bool(debug_mode):
                    _stop_robot(robot)
                    stats["debug_mode_terminated"] = True
                    stats["last_action"] = "DEBUG_STEP1_TERMINATE"
                    print("[DEBUG] Step 1 win confirmed; stopping for operator confirmation.", flush=True)
                    return stats
                last_action = "HAPPY"
                stats["stop_after_win_triggered"] = True
                break
            _bump_stat_count(stats, "miss_reasons", "dist_pingpong_confirm_failed")
            print(
                f"[FOLLOW] DIST_PINGPONG_REJECT stopped_check={confirm_reason}; "
                + (
                    "will re-read from rest before the next correction."
                    if bool(stop_after_win) and not bool(debug_mode)
                    else "stopped before another opposite wheel correction."
                ),
                flush=True,
            )
            _stop_robot(robot)
            if bool(stop_after_win) and not bool(debug_mode):
                last_action = "DIST_PINGPONG_REJECT"
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue
            stats["dist_pingpong_guard_terminated"] = True
            stats["happy_reject_terminated"] = True
            if bool(debug_mode):
                stats["debug_mode_terminated"] = True
            stats["debug_stop_reading"] = (
                dict(confirmed_reading)
                if isinstance(confirmed_reading, dict)
                else confirmed_reading
            )
            stats["last_action"] = (
                "DEBUG_DIST_PINGPONG_GUARD_TERMINATE"
                if bool(debug_mode)
                else "PARK_DIST_PINGPONG_GUARD_TERMINATE"
            )
            return stats

        if str(plan.get("kind")) == "hold":
            action = "HAPPY"
            if bool(_win_confirmation_config().get("accept_live_happy_after_stop", False)):
                _stop_robot(robot)
                dist_err, x_err, y_text, conf = _record_and_print_step1_win(stats, reading, plan)
                if bool(debug_mode):
                    stats["debug_mode_terminated"] = True
                    stats["last_action"] = "DEBUG_STEP1_TERMINATE"
                    print("[DEBUG] Step 1 win accepted from live happy; stopping for operator confirmation.", flush=True)
                    return stats
                if bool(stop_after_win):
                    last_action = "HAPPY"
                    stats["stop_after_win_triggered"] = True
                    break
                continue
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
                        rejected_y_text = f" y_err={float(confirmed_reading.get('y_mm')) - _y_win_target_mm():+.1f}mm"
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
            dist_err, x_err, y_text, conf = _record_and_print_step1_win(stats, reading, plan)
            if bool(debug_mode):
                _stop_robot(robot)
                stats["debug_mode_terminated"] = True
                stats["last_action"] = "DEBUG_STEP1_TERMINATE"
                print("[DEBUG] Step 1 win confirmed; stopping for operator confirmation.", flush=True)
                return stats
            if bool(stop_after_win):
                last_action = "HAPPY"
                stats["stop_after_win_triggered"] = True
                break
            if bool(_follow_motion_config().get("step2_suspended", False)):
                if bool(reset_after_win):
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
                    _set_game_profile("empty")
                    print("[STEP2] Suspended for empty profile; reset done; next game profile is empty.", flush=True)
                    step1_started_at = time.monotonic()
                    last_action = "RESET"
                    stats["y_lock_on_armed"] = True
                    print_ticker = 0
                    elapsed = time.monotonic() - loop_start
                    if (remaining := LOOP_S - elapsed) > 0:
                        time.sleep(remaining)
                    continue
                last_action = "STEP2_SUSPENDED"
                break
            step2_result = _run_step2_seat_sequence(
                vision,
                robot,
                probe_before_forward=bool(step2_probe_before_forward),
            )
            _record_step2_stats(stats, step2_result)
            step2_reading = step2_result.get("reading") if isinstance(step2_result, dict) else None
            step2_dist = step2_x = step2_y = "N/A"
            if isinstance(step2_reading, dict):
                try:
                    step2_dist = f"{float(step2_reading.get('dist_mm')):.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    step2_x = f"{float(step2_reading.get('x_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    step2_y = f"{float(step2_reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            step2_xz_frozen = bool(isinstance(step2_reading, dict) and step2_reading.get("xz_frozen"))
            print(
                f"[STEP2] Attempt after step 1 win: reason={step2_result.get('reason')} "
                f"target_met={bool(step2_result.get('target_met'))} "
                f"xz_frozen={step2_xz_frozen} "
                f"after=dist {step2_dist}, x {step2_x}, y {step2_y}",
                flush=True,
            )
            if bool(step2_result.get("probe")):
                print("[STEP2] Probe complete before forward drive; stopped without reset.", flush=True)
                last_action = "STEP2_PROBE"
                break
            if not bool(step2_result.get("success")):
                print(f"[STEP2] Failed gracefully: {step2_result.get('reason')}", flush=True)
                break
            if not bool(step2_result.get("target_met")):
                print("[STEP2] Not starting step 3 until step 2 is an honest target hit.", flush=True)
                last_action = "STEP2_NEEDS_WORK"
                break
            if bool(debug_mode):
                _stop_robot(robot)
                stats["debug_mode_terminated"] = True
                stats["last_action"] = "DEBUG_STEP2_TERMINATE"
                print("[DEBUG] Robot stopped after Step 2 win; waiting for operator confirmation.", flush=True)
                return stats
            if _game_complete_after_step2():
                ran_step3_retreat = False
                if _step3_kind() == "retreat":
                    ran_step3_retreat = True
                    step3_result = _run_step3_retreat_sequence(vision, robot)
                    step3_reading = step3_result.get("reading") if isinstance(step3_result, dict) else None
                    step3_dist = step3_x = step3_y = "N/A"
                    if isinstance(step3_reading, dict):
                        try:
                            step3_dist = f"{float(step3_reading.get('dist_mm')):.1f}mm"
                        except (TypeError, ValueError):
                            pass
                        try:
                            step3_x = f"{float(step3_reading.get('x_mm')):+.1f}mm"
                        except (TypeError, ValueError):
                            pass
                        try:
                            step3_y = f"{float(step3_reading.get('y_mm')):+.1f}mm"
                        except (TypeError, ValueError):
                            pass
                    print(
                        f"[STEP3] Retreat after step 2: reason={step3_result.get('reason')} "
                        f"target_met={bool(step3_result.get('target_met'))} "
                        f"duration={int(step3_result.get('drive_duration_ms', 0) or 0)}ms "
                        f"after=dist {step3_dist}, x {step3_x}, y {step3_y}",
                        flush=True,
                    )
                    if not bool(step3_result.get("success")):
                        print(f"[STEP3] Failed gracefully: {step3_result.get('reason')}", flush=True)
                        break
                    if not bool(step3_result.get("target_met")):
                        print("[STEP3] Not resetting until retreat target is honestly reached; parked.", flush=True)
                        last_action = "STEP3_RETREAT_NEEDS_WORK"
                        break
                if bool(reset_after_win):
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
                    _set_game_profile("empty")
                    if ran_step3_retreat:
                        print("[PROFILE] Holding place/retreat complete; reset done; next game profile is empty.", flush=True)
                    else:
                        print("[PROFILE] Holding place complete; reset done; next game profile is empty.", flush=True)
                    completed_cycles += 1
                    if max_cycles is not None and completed_cycles >= int(max_cycles):
                        last_action = "E2E_CYCLES_DONE"
                        break
                    step1_started_at = time.monotonic()
                    last_action = "RESET"
                    stats["y_lock_on_armed"] = True
                    print_ticker = 0
                    elapsed = time.monotonic() - loop_start
                    if (remaining := LOOP_S - elapsed) > 0:
                        time.sleep(remaining)
                    continue
                last_action = "STEP2_GAME_COMPLETE"
                break
            if bool(stop_after_step2):
                print("[STEP2] Parked at step 2; stopped without reset.", flush=True)
                last_action = "STEP2_PARK"
                stats["stop_after_step2_triggered"] = True
                break
            step3_result = _run_step3_seat_sequence(vision, robot)
            step3_reading = step3_result.get("reading") if isinstance(step3_result, dict) else None
            step3_dist = step3_x = step3_y = "N/A"
            if isinstance(step3_reading, dict):
                try:
                    step3_dist = f"{float(step3_reading.get('dist_mm')):.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    step3_x = f"{float(step3_reading.get('x_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    step3_y = f"{float(step3_reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            print(
                f"[STEP3] Seat after step 2: reason={step3_result.get('reason')} "
                f"target_met={bool(step3_result.get('target_met'))} "
                f"after=dist {step3_dist}, x {step3_x}, y {step3_y}",
                flush=True,
            )
            if not bool(step3_result.get("success")):
                print(f"[STEP3] Failed gracefully: {step3_result.get('reason')}", flush=True)
                break
            if not bool(step3_result.get("target_met")):
                print("[STEP3] Not starting step 4 until step 3 is an honest target hit.", flush=True)
                last_action = "STEP3_NEEDS_WORK"
                break
            if bool(complete_after_step3):
                stats["step3_win_count"] = int(stats.get("step3_win_count", 0)) + 1
                print("[STEP3] Step 3 complete (steps 1-3 drill); skipping step 4 lift.", flush=True)
                if not bool(reset_after_win):
                    last_action = "STEP3_COMPLETE"
                    break
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
                _set_game_profile("empty")
                completed_cycles += 1
                if max_cycles is not None and completed_cycles >= int(max_cycles):
                    last_action = "STEPS123_CYCLES_DONE"
                    break
                step1_started_at = time.monotonic()
                last_action = "RESET"
                stats["y_lock_on_armed"] = True
                print_ticker = 0
                elapsed = time.monotonic() - loop_start
                if (remaining := LOOP_S - elapsed) > 0:
                    time.sleep(remaining)
                continue
            if bool(debug_mode):
                _stop_robot(robot)
                stats["debug_mode_terminated"] = True
                stats["last_action"] = "DEBUG_STEP3_TERMINATE"
                print("[DEBUG] Robot stopped after Step 3 win; waiting for operator confirmation.", flush=True)
                return stats
            step4_result = _run_step3_lift_sequence(vision, robot)
            step4_reading = step4_result.get("reading") if isinstance(step4_result, dict) else None
            step4_y = "N/A"
            if isinstance(step4_reading, dict):
                try:
                    step4_y = f"{float(step4_reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            print(
                f"[STEP4] Lift after step 3: reason={step4_result.get('reason')} "
                f"target_met={bool(step4_result.get('target_met'))} holding={bool(step4_result.get('holding'))} "
                f"after_y={step4_y}",
                flush=True,
            )
            if bool(step4_result.get("fallback_reset_ok")):
                print(
                    f"[STEP4] No visibility fallback completed: "
                    f"{step4_result.get('fallback_cmd')} for {step4_result.get('fallback_duration_ms')}ms; "
                    "continuing to reset.",
                    flush=True,
                )
            if bool(step4_result.get("holding")):
                _set_game_profile("holding")
            if not bool(step4_result.get("success")):
                print(f"[STEP4] Failed gracefully: {step4_result.get('reason')}", flush=True)
                break
            if not (bool(step4_result.get("target_met")) or bool(step4_result.get("fallback_reset_ok"))):
                print("[STEP4] Not resetting until step 4 is an honest target hit.", flush=True)
                last_action = "STEP4_NEEDS_WORK"
                break
            if bool(debug_mode) and bool(step4_result.get("target_met")):
                _stop_robot(robot)
                stats["debug_mode_terminated"] = True
                stats["last_action"] = "DEBUG_STEP4_TERMINATE"
                print("[DEBUG] Robot stopped after Step 4 win; waiting for operator confirmation.", flush=True)
                return stats
            if _step2_result_freezes_xz(step2_result):
                _stop_robot(robot)
                print(
                    "[FOLLOW] Placement complete with x/z frozen; parked without reset.",
                    flush=True,
                )
                last_action = "XZ_FROZEN_PARK"
                break
            if not bool(reset_after_win):
                last_action = "HAPPY"
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
            step1_started_at = time.monotonic()
            last_action = "RESET"
            stats["y_lock_on_armed"] = True
            print_ticker = 0
            elapsed = time.monotonic() - loop_start
            if (remaining := LOOP_S - elapsed) > 0:
                time.sleep(remaining)
            continue
        else:
            plan = _apply_stall_recovery_boost(stats, plan)
            plan = _cap_mast_plan_for_unreliable_spool(stats, plan)
            plan = _cap_mast_up_plan_to_y_ceiling(plan, reading)
            plan = _cap_mast_up_plan_to_budget(stats, plan)
            plan = _cap_mast_plan_to_max_duration(plan)
            plan = _cap_near_target_wheel_plan_to_crawl(plan)
            action = str(plan.get("action") or action)
            exceeded, used_up_ms, planned_up_ms, max_up_ms = _mast_up_budget_exceeded(stats, plan)
            if exceeded:
                _stop_robot(robot)
                _bump_stat_count(stats, "miss_reasons", "mast_up_budget_guard")
                stats["debug_stop_reading"] = dict(reading) if isinstance(reading, dict) else reading
                stats["last_action"] = "MAST_UP_BUDGET_GUARD"
                print(
                    f"[FOLLOW] HARD STOP: mast-up budget guard blocked {action}. "
                    f"Already used {used_up_ms}ms; planned {planned_up_ms}ms would exceed "
                    f"{max_up_ms}ms. Holding still instead of risking a high-mast overshoot.",
                    flush=True,
                )
                return stats
            _record_follow_attempt_stats(stats, reading, plan)
            send_result = _execute_follow_action(robot, plan, reading)
            _record_send_result(stats, action, send_result)
            if not _send_result_blocked(send_result):
                _record_mast_up_budget(stats, plan, send_result)
                _consume_unreliable_spool_probe(stats, plan)
            duration_text = _sent_motion_duration_text(plan, send_result)
            if not _send_result_blocked(send_result):
                _reset_follow_reading_history(
                    vision,
                    allow_large_dist_jump=_plan_allows_large_dist_reacquire(plan),
                )
                loop_wait_s = _post_action_wait_s(plan, send_result)
                pending = {
                    "action": action,
                    "dist_mm": dist_mm,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "cmd": plan.get("cmd"),
                    "mast_cmd": plan.get("mast_cmd"),
                    "dist_err": plan.get("dist_err", dist_err),
                    "x_err": plan.get("x_err", x_err),
                    "y_target_mm": plan.get("y_target_mm", plan.get("mast_y_target_mm")),
                    "y_err": plan.get("y_err", plan.get("mast_y_err")),
                }
                if plan.get("recovery_boost_scale") is not None:
                    pending["recovery_boost_scale"] = plan.get("recovery_boost_scale")
                try:
                    duration_val = (send_result or {}).get("duration_ms") if isinstance(send_result, dict) else None
                    pending["duration_ms"] = int(duration_val if duration_val is not None else plan.get("duration_ms", PULSE_MS))
                except (TypeError, ValueError):
                    pending["duration_ms"] = int(PULSE_MS)
                if isinstance(send_result, dict) and isinstance(send_result.get("x_curve"), dict):
                    pending["x_curve"] = dict(send_result["x_curve"])
                else:
                    plan_curve = _x_curve_for_plan(plan, reading)
                    if isinstance(plan_curve, dict):
                        pending["x_curve"] = dict(plan_curve)
                stats["pending_observation"] = pending
                _record_distance_act_state(stats, action, plan)

        # Print on state change or every 20 ticks (~1 s) to avoid flooding
        print_ticker += 1
        if action != last_action or print_ticker >= 20:
            print(
                f"[FOLLOW] {action:<10} dist_err={dist_err:+.1f}mm  "
                f"x_err={x_err:+.1f}mm {y_text} {duration_text} conf={conf:.0f}%",
                flush=True,
            )
            print_ticker = 0
        last_action = action
        stats["last_act_was_mast"] = str(plan.get("kind") or "").strip().lower() in {"mast", "y"}
        elapsed = time.monotonic() - loop_start
        remaining = float(loop_wait_s) - elapsed
        if remaining > 0:
            time.sleep(remaining)
        if not _send_result_blocked(send_result):
            _stop_robot(robot)
    if last_action:
        stats["last_action"] = last_action
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
        "--park-step2",
        action="store_true",
        help="Move to step 1 happy, run the step 2 seat act, then stop without reset.",
    )
    parser.add_argument(
        "--step2-seat-once",
        action="store_true",
        help="Run one step 2 seat act: mast down plus slow forward, then measure.",
    )
    parser.add_argument(
        "--step2-lock-once",
        action="store_true",
        help="Run only step 2a lock: lower the mast, measure, and stop.",
    )
    parser.add_argument(
        "--step2-settle-only",
        action="store_true",
        help="Run only step 2 precision settling from the current pose, with no blind seat/reset.",
    )
    parser.add_argument(
        "--step2-probe-before-forward",
        action="store_true",
        help="After a step 1 win, lower for step 2, report the pre-forward reading, then stop.",
    )
    parser.add_argument(
        "--step3-lift-once",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--step3-seat-once",
        action="store_true",
        help="Run one step 3 seat settle to the active profile's dist/x target, then measure.",
    )
    parser.add_argument(
        "--step3-retreat-once",
        action="store_true",
        help="Run one step 3 retreat using the active profile's retreat config, then measure.",
    )
    parser.add_argument(
        "--step4-lift-once",
        action="store_true",
        help="Run one step 4 lift to the active profile's y target, then measure.",
    )
    parser.add_argument(
        "--e2e-trial",
        action="store_true",
        help="Run one full empty+holding end-to-end trial: reset, play, reset, then stop.",
    )
    parser.add_argument(
        "--e2e-max-attempts",
        type=int,
        default=5,
        help="Maximum reset-and-retry attempts for an e2e trial before parking as a last resort.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Run with verbose/debug gates and stop after each confirmed step win for operator confirmation.",
    )
    parser.add_argument(
        "--game-profile",
        choices=("auto", "empty", "holding"),
        default="auto",
        help="Use auto detection, or force the empty/holding target profile.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=25.0,
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


def _worker_argv(args: argparse.Namespace, *, skip_vision_preflight: bool | None = None) -> list[str]:
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
    if bool(args.park_step2):
        argv.append("--park-step2")
    if bool(args.step2_seat_once):
        argv.append("--step2-seat-once")
    if bool(args.step2_lock_once):
        argv.append("--step2-lock-once")
    if bool(args.step2_settle_only):
        argv.append("--step2-settle-only")
    if bool(args.step2_probe_before_forward):
        argv.append("--step2-probe-before-forward")
    if bool(args.step3_lift_once):
        argv.append("--step3-lift-once")
    if bool(args.step3_seat_once):
        argv.append("--step3-seat-once")
    if bool(args.step3_retreat_once):
        argv.append("--step3-retreat-once")
    if bool(args.step4_lift_once):
        argv.append("--step4-lift-once")
    if bool(args.e2e_trial):
        argv.append("--e2e-trial")
        argv.extend(["--e2e-max-attempts", str(int(max(1, args.e2e_max_attempts)))])
    if bool(args.debug_mode):
        argv.append("--debug-mode")
    argv.extend(["--game-profile", str(args.game_profile)])
    skip_preflight = bool(args.skip_vision_preflight) if skip_vision_preflight is None else bool(skip_vision_preflight)
    if bool(skip_preflight):
        argv.append("--skip-vision-preflight")
    return argv


def _stop_stale_vision_processes() -> int:
    """Stop local camera owners that commonly leave OAK/TensorRT resources stale."""
    patterns = (
        "livestream_crown_vision.py",
    )
    stopped = 0
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return 0
    own_pid = int(os.getpid())
    for line in (result.stdout or "").splitlines():
        text = str(line or "").strip()
        if not text:
            continue
        try:
            pid_text, cmdline = text.split(None, 1)
            pid = int(pid_text)
        except (ValueError, TypeError):
            continue
        if int(pid) == own_pid:
            continue
        if not any(pattern in cmdline for pattern in patterns):
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
            stopped += 1
        except ProcessLookupError:
            pass
        except Exception:
            continue
    if stopped:
        time.sleep(1.0)
    return int(stopped)


def _recover_vision_startup(reason: str, *, attempt: int) -> tuple[bool, str]:
    """Best-effort cleanup before retrying camera/TensorRT startup."""
    stopped = _stop_stale_vision_processes()
    wait_s = min(4.0, 1.0 + float(max(0, attempt)))
    if stopped:
        detail = f"stopped {stopped} stale livestream process(es)"
    else:
        detail = "no stale livestream process found"
    print(
        f"[FOLLOW] Vision startup recovery {attempt}: {reason}; {detail}; "
        f"waiting {wait_s:.1f}s before retry.",
        flush=True,
    )
    time.sleep(wait_s)
    ok, check_reason = _vision_memory_preflight()
    return bool(ok), str(check_reason)


def _recover_pregame_visibility(reason: str, *, attempt: int) -> None:
    """Best-effort camera teardown pause before retrying pregame brick lock."""
    stopped = _stop_stale_vision_processes()
    wait_s = min(4.0, 1.0 + float(max(0, attempt)))
    if stopped:
        detail = f"stopped {stopped} stale livestream process(es)"
    else:
        detail = "camera worker exited; no stale livestream process found"
    print(
        f"[FOLLOW] Pregame visibility recovery {attempt}: {reason}; {detail}; "
        f"waiting {wait_s:.1f}s before restarting vision.",
        flush=True,
    )
    time.sleep(wait_s)


def _supervise_run(args: argparse.Namespace) -> int:
    worker = None
    bypass_preflight = bool(args.skip_vision_preflight)
    pregame_recovery_attempts = 0
    max_attempts = (
        1
        + int(VISION_RECOVERY_RETRIES)
        + 1
        + int(PREGAME_VISIBILITY_RECOVERY_RETRIES)
    )
    for attempt in range(1, max_attempts + 1):
        print(
            "[FOLLOW] Starting supervised follow worker"
            + (" with vision preflight bypass." if bypass_preflight else "."),
            flush=True,
        )
        try:
            worker = subprocess.Popen(
                _worker_argv(args, skip_vision_preflight=bypass_preflight),
                cwd=str(Path(__file__).resolve().parent),
            )
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
        if returncode == VISION_PREFLIGHT_BLOCK_EXIT and not bool(args.skip_vision_preflight):
            if attempt <= int(VISION_RECOVERY_RETRIES):
                ok, check_reason = _recover_vision_startup(
                    "preflight reported fragmented memory",
                    attempt=attempt,
                )
                if bool(ok):
                    print(f"[FOLLOW] Vision recovery ok: {check_reason}; retrying game.", flush=True)
                else:
                    print(f"[FOLLOW] Vision still fragmented: {check_reason}; retrying cleanup path.", flush=True)
                continue
            if not bypass_preflight:
                bypass_preflight = True
                print(
                    "[FOLLOW] Vision preflight still pessimistic after cleanup; "
                    "trying one direct vision startup. No robot motion starts unless vision opens.",
                    flush=True,
                )
                continue

        if returncode == PREGAME_VISIBILITY_BLOCK_EXIT:
            pregame_recovery_attempts += 1
            if pregame_recovery_attempts <= int(PREGAME_VISIBILITY_RECOVERY_RETRIES):
                _recover_pregame_visibility(
                    "brick lock timed out before motion",
                    attempt=pregame_recovery_attempts,
                )
                continue

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
    _emergency_stop_robot()
    print("[FOLLOW] Worker did not recover after vision startup retries; recovered with stop.", flush=True)
    return 1


def _run_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    requested_profile = "empty" if bool(args.e2e_trial) else str(args.game_profile)
    _set_game_profile("empty" if requested_profile == "auto" else requested_profile)

    if not bool(args.skip_vision_preflight):
        ok, reason = _vision_memory_preflight(min_lfb_mb=float(args.min_lfb_mb))
        if not bool(ok):
            print(
                f"[FOLLOW] Vision startup blocked: {reason}. "
                "Recovery: no robot motion was started.",
                flush=True,
            )
            return VISION_PREFLIGHT_BLOCK_EXIT
        log.info("Vision preflight ok: %s", reason)

    vision = None
    robot = None
    print(
        f"[FOLLOW] Target: dist={_dist_target_mm():.0f}mm ±{_dist_tol_mm():.0f}mm  "
        f"x={_x_target_mm():+.1f}mm ±{_x_tol_mm():.0f}mm  |  pulse: {PULSE_MS}ms  |  loop: {int(1/LOOP_S)}Hz  "
        f"|  max act: {_max_act_ms()}ms  |  normal score: {_normal_speed_score()}  "
        f"|  power scale: {_motion_power_scale():.2f}x",
        flush=True,
    )
    print("[FOLLOW] Press Ctrl-C to stop.", flush=True)

    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(CROWN_PROFILE_TUNING))
        _warmup(vision)
        if requested_profile == "auto":
            selected_profile, profile_vote = _auto_select_game_profile(vision)
            _set_game_profile(selected_profile)
            vote_details = []
            for idx, vote in enumerate(profile_vote.get("votes", []), start=1):
                vote_details.append(
                    f"{idx}:{'holding' if bool(vote.get('holding')) else 'empty'}"
                    f"/{vote.get('reason', 'unknown')}"
                )
            print(
                f"[PROFILE] holding_detector: {profile_vote['holding_count']}/{profile_vote['samples']} "
                f"holding=True -> using {selected_profile} game ({', '.join(vote_details)})",
                flush=True,
            )
        else:
            print(f"[PROFILE] forced -> using {requested_profile} game", flush=True)
        _print_active_success_gates()
        robot = Robot()
        pregame_reading = _wait_for_confident_brick(vision)
        if not bool(pregame_reading.get("confident")):
            if bool(args.e2e_trial):
                pregame_reading = _recover_reset_visibility_with_backoff(
                    vision,
                    robot,
                    pregame_reading,
                    context="e2e_pregame_visibility",
                )
        if not bool(pregame_reading.get("confident")):
            print(
                "[FOLLOW] Pregame visibility failed; staying still. "
                "Reset is only automatic after an empty/holding game win, "
                "or when --reset-only is explicitly requested.",
                flush=True,
            )
            stats = _new_game_stats()
            stats["sample_count"] = 1
            stats["not_confident_count"] = 1
            _bump_stat_count(stats, "miss_reasons", "brick_not_confident")
            print("[RESULTS]", flush=True)
            print(_format_game_results_table(stats), flush=True)
            return PREGAME_VISIBILITY_BLOCK_EXIT
        if bool(args.e2e_trial):
            _set_game_profile("empty")
            print("[E2E] Starting end-to-end trial: initial reset, empty game, holding game, final reset.", flush=True)
            print(
                "[E2E] No global trial timer; each step gets its configured budget, "
                f"and misses reset/retry up to {int(max(1, args.e2e_max_attempts))} attempt(s) before parking.",
                flush=True,
            )
            stats = None
            max_attempts = int(max(1, args.e2e_max_attempts))
            for e2e_attempt in range(1, max_attempts + 1):
                _set_game_profile("empty")
                if e2e_attempt > 1:
                    print(f"[E2E] Retry {e2e_attempt}/{max_attempts}: reset and run the script again.", flush=True)
                reset_result = _run_reset_sequence(vision, robot)
                if bool(reset_result.get("success")):
                    print(
                        f"[E2E] Reset before attempt {e2e_attempt} done: "
                        f"turn={str(reset_result.get('turn_cmd')).upper()} reason={reset_result.get('reason')}",
                        flush=True,
                    )
                else:
                    print(
                        f"[E2E] Reset before attempt {e2e_attempt} failed during {reset_result.get('phase')}: "
                        f"{reset_result.get('reason')}; trying next reset if budget remains.",
                        flush=True,
                    )
                    continue
                stats = _follow_loop(
                    vision,
                    robot,
                    duration_s=365.0 * 24.0 * 60.0 * 60.0,
                    reset_after_win=True,
                    stop_after_win=False,
                    stop_after_step2=False,
                    step2_probe_before_forward=False,
                    debug_mode=False,
                    max_cycles=1,
                )
                print(f"[RESULTS attempt {e2e_attempt}/{max_attempts}]", flush=True)
                print(_format_game_results_table(stats), flush=True)
                if str(stats.get("last_action") or "") == "E2E_CYCLES_DONE":
                    print("[E2E] Trial complete: empty+holding cycle won and final reset completed.", flush=True)
                    return 0
                last_action = str(stats.get("last_action") or "unknown")
                print(f"[E2E] Attempt {e2e_attempt}/{max_attempts} missed: {last_action}", flush=True)
                if bool(stats.get("pickup_suspected_stop")):
                    print("[E2E] Pickup suspected; parking as the last-resort safety stop.", flush=True)
                    return 1
            last = "unknown" if not isinstance(stats, dict) else str(stats.get("last_action") or "unknown")
            print(f"[E2E] Trial parked after {max_attempts} attempt(s): {last}", flush=True)
            return 1
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
        elif bool(args.step2_seat_once) or bool(args.step2_settle_only) or bool(args.step2_lock_once):
            if bool(args.step2_settle_only):
                result = _run_step2_settle_sequence(vision, robot)
                step2_label = "Settle"
            elif bool(args.step2_lock_once):
                result = _run_step2_seat_sequence(vision, robot, probe_before_forward=True)
                step2_label = "Lock"
            else:
                result = _run_step2_seat_sequence(vision, robot)
                step2_label = "Seat"
            stats = _new_game_stats()
            _record_step2_stats(stats, result)
            reading = result.get("reading") if isinstance(result, dict) else None
            dist_text = x_text = y_text = "N/A"
            if isinstance(reading, dict):
                try:
                    dist_text = f"{float(reading.get('dist_mm')):.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    x_text = f"{float(reading.get('x_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    y_text = f"{float(reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            xz_frozen = bool(isinstance(reading, dict) and reading.get("xz_frozen"))
            print(
                f"[STEP2] {step2_label} done: reason={result.get('reason')} "
                f"target_met={bool(result.get('target_met'))} "
                f"xz_frozen={xz_frozen} "
                f"after=dist {dist_text}, x {x_text}, y {y_text}",
                flush=True,
            )
            print("[RESULTS]", flush=True)
            print(_format_game_results_table(stats), flush=True)
            if not bool(result.get("success")):
                return 1
            if bool(args.debug_mode) and bool(result.get("target_met")):
                _stop_robot(robot)
                print("[DEBUG] Robot stopped after Step 2 win; process exiting cleanly.", flush=True)
                return 0
        elif bool(args.step3_retreat_once):
            result = _run_step3_retreat_sequence(vision, robot)
            reading = result.get("reading") if isinstance(result, dict) else None
            dist_text = x_text = y_text = "N/A"
            if isinstance(reading, dict):
                try:
                    dist_text = f"{float(reading.get('dist_mm')):.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    x_text = f"{float(reading.get('x_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    y_text = f"{float(reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            print(
                f"[STEP3] Retreat done: reason={result.get('reason')} "
                f"target_met={bool(result.get('target_met'))} "
                f"duration={int(result.get('drive_duration_ms', 0) or 0)}ms "
                f"after=dist {dist_text}, x {x_text}, y {y_text}",
                flush=True,
            )
            if not bool(result.get("success")):
                return 1
            if bool(args.debug_mode) and bool(result.get("target_met")):
                _stop_robot(robot)
                print("[DEBUG] Robot stopped after Step 3 retreat; process exiting cleanly.", flush=True)
                return 0
        elif bool(args.step3_seat_once):
            result = _run_step3_seat_sequence(vision, robot)
            reading = result.get("reading") if isinstance(result, dict) else None
            dist_text = x_text = y_text = "N/A"
            if isinstance(reading, dict):
                try:
                    dist_text = f"{float(reading.get('dist_mm')):.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    x_text = f"{float(reading.get('x_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
                try:
                    y_text = f"{float(reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            print(
                f"[STEP3] Seat done: reason={result.get('reason')} "
                f"target_met={bool(result.get('target_met'))} "
                f"after=dist {dist_text}, x {x_text}, y {y_text}",
                flush=True,
            )
            if not bool(result.get("success")):
                return 1
            if bool(args.debug_mode) and bool(result.get("target_met")):
                _stop_robot(robot)
                print("[DEBUG] Robot stopped after Step 3 win; process exiting cleanly.", flush=True)
                return 0
        elif bool(args.step4_lift_once) or bool(args.step3_lift_once):
            result = _run_step3_lift_sequence(vision, robot)
            reading = result.get("reading") if isinstance(result, dict) else None
            y_text = "N/A"
            if isinstance(reading, dict):
                try:
                    y_text = f"{float(reading.get('y_mm')):+.1f}mm"
                except (TypeError, ValueError):
                    pass
            print(
                f"[STEP4] Lift done: reason={result.get('reason')} "
                f"target_met={bool(result.get('target_met'))} holding={bool(result.get('holding'))} y={y_text}",
                flush=True,
            )
            if bool(result.get("holding")):
                _set_game_profile("holding")
            if not bool(result.get("success")):
                return 1
            if bool(args.debug_mode) and bool(result.get("target_met")):
                _stop_robot(robot)
                print("[DEBUG] Robot stopped after Step 3 win; process exiting cleanly.", flush=True)
                return 0
        else:
            stats = _follow_loop(
                vision,
                robot,
                duration_s=float(args.duration_s),
                reset_after_win=not (bool(args.park_happy) or bool(args.park_step2)),
                stop_after_win=bool(args.park_happy),
                stop_after_step2=bool(args.park_step2),
                step2_probe_before_forward=bool(args.step2_probe_before_forward),
                debug_mode=bool(args.debug_mode),
            )
            if bool(stats.get("debug_mode_terminated")):
                print("[DEBUG] Debug mode complete; process exiting cleanly.", flush=True)
                print("[RESULTS]", flush=True)
                print(_format_game_results_table(stats), flush=True)
                return 0
            if bool(stats.get("happy_reject_terminated")):
                print(
                    "[FOLLOW] Step 1 live happy failed stopped confirmation; "
                    "stopped by --park-happy before another correction.",
                    flush=True,
                )
            elif bool(stats.get("stop_after_win_triggered")):
                print("[FOLLOW] Step 1 win reached; stopped by --park-happy.", flush=True)
            elif bool(stats.get("stop_after_step2_triggered")):
                print("[FOLLOW] Step 2 win reached; stopped by --park-step2.", flush=True)
            else:
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
