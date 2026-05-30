#!/usr/bin/env python3
"""Practice empty Step 1 repeatedly with a half-distance reset."""

from __future__ import annotations

import copy
import argparse
import json
import random
import time
from pathlib import Path

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot


TRIALS = 70
FOLLOW_DURATION_S = 15.0
OUT_PATH = Path("runs/practice_step1_empty_70.jsonl")
RESET_RECOVERY_WAIT_S = 4.0
RESET_RECOVERY_POLL_S = 0.15
START_RECOVERY_WAIT_S = 6.0
START_RECOVERY_POLL_S = 0.15
MAX_RECOVERY_SKIPS = 40
LOST_BRICK_SHARP_RIGHT_RECOVERY_ENABLED = True
LOST_BRICK_SHARP_RIGHT_MAX_MS = 3000
LOST_BRICK_SHARP_RIGHT_CHUNK_MS = 1000
LOST_BRICK_SHARP_RIGHT_PWM = 103
LOST_BRICK_SHARP_RIGHT_RECHECK_S = 1.0
LOST_BRICK_BACKUP_RECOVERY_ENABLED = True
LOST_BRICK_BACKUP_RECOVERY_MS = 1000
LOST_BRICK_BACKUP_RECOVERY_PWM = 103
LOST_BRICK_BACKUP_RECHECK_S = 1.0
START_Y_RECOVERY_ABS_ERR_MM = 10.0
# Mast-down is ~0.2mm/100ms (much slower than up), so when y starts too high it
# needs many small down acts to reach the start band. Allow enough acts to grind
# it down; the "worsening" guard still bails out if the mast isn't helping.
START_Y_RECOVERY_MAX_ACTS = 20
START_Y_RECOVERY_SAMPLE_S = 0.15
PRACTICE_RESET_STRAIGHT_SCALE = 1.0
PRACTICE_RESET_STRAIGHT_MIN_MS = 800
PRACTICE_RESET_STRAIGHT_MAX_MS = 1500
PRACTICE_RESET_DIST_OFFSET_MM = 75.0
PRACTICE_RESET_DIST_TOL_MM = 85.0
PRACTICE_RESET_X_CURVE_MAX_MS = 240
PRACTICE_RESET_X_CURVE_MIN_MS = 160
PRACTICE_RESET_X_OFFSET_MIN_MM = 5.0
PRACTICE_RESET_X_OFFSET_MAX_MM = 35.0
PRACTICE_RESET_TARGET_ABS_X_MM = 12.0
PRACTICE_RESET_POST_PAUSE_S = 0.2
PRACTICE_RESET_SETTLE_S = 0.0
PRACTICE_STEP1_POST_ACT_STABILIZE_S = 0.0
PRACTICE_STEP1_MAST_COAST_SETTLE_S = 0.35
PRACTICE_STEP1_WIN_SETTLE_S = 0.0
PRACTICE_STEP1_WIN_CONFIRM_FRAMES = 1
PRACTICE_FAR_DIST_FULL_GAP_MM = 120.0
PRACTICE_FAR_DIST_MAX_PULSE_MS = 285
PRACTICE_NEAR_DIST_MAX_PULSE_MS = 240
PRACTICE_DRIVE_MIN_EFFECTIVE_MS = 240
PRACTICE_BIAS_MAX_PULSE_MS = 240
PRACTICE_BACKWARD_MAX_PULSE_MS = 260
PRACTICE_DIST_PINGPONG_CONFIRM_MM = 35.0
PRACTICE_Y_DIST_WINDOW_MM = 75.0
PRACTICE_Y_X_WINDOW_MM = 18.0
PRACTICE_Y_SMALL_GAP_PULSE_MS = 240
PRACTICE_Y_MEDIUM_GAP_PULSE_MS = 360
PRACTICE_Y_LARGE_GAP_PULSE_MS = 500
PRACTICE_CLEAN_RESET_LOW_X_ENABLED = True
PRACTICE_CLEAN_RESET_LOW_X_MIN_MM = 5.0
PRACTICE_CLEAN_RESET_LOW_X_TARGET_MM = 12.0
PRACTICE_CLEAN_RESET_LOW_X_MAX_ATTEMPTS = 4
PRACTICE_CLEAN_RESET_LOW_X_PULSE_MS = 250
PRACTICE_CLEAN_RESET_HIGH_X_MAX_ATTEMPTS = 5
PRACTICE_CLEAN_RESET_HIGH_X_PULSE_MS = 300
PRACTICE_CLEAN_RESET_DIST_ENABLED = False
PRACTICE_CLEAN_RESET_DIST_MAX_ATTEMPTS = 4
PRACTICE_CLEAN_RESET_DIST_PULSE_MS = 240
PRACTICE_RESET_MAX_TARGET_RETRIES = 3
PRACTICE_RESET_FINAL_ONE_WHEEL_FINISH_ENABLED = True
PRACTICE_RESET_FINAL_ONE_WHEEL_FINISH_MS = 250


def _practice_low_x_one_wheel_turn(
    robot: Robot,
    *,
    turn_cmd: str,
    reading: dict,
    reset_cfg: dict,
    duration_ms: int,
) -> dict | None:
    x_goal_curve = reset_cfg.get("x_goal_curve") if isinstance(reset_cfg.get("x_goal_curve"), dict) else {}
    extra_turn = reset_cfg.get("low_x_extra_sharp_turn") if isinstance(reset_cfg.get("low_x_extra_sharp_turn"), dict) else {}
    try:
        outer_pwm = int(round(float(x_goal_curve.get("outer_pwm", extra_turn.get("faster_pwm", 162)))))
    except (TypeError, ValueError):
        outer_pwm = 162
    curve = {
        "inner_pwm": 0,
        "outer_pwm": int(outer_pwm),
        "drive_mode": "backward",
        "strength": "practice_low_x_one_wheel_finish",
    }
    actions = follow._scaled_actions(
        follow._turn_curve_actions(drive_mode="backward", cmd=turn_cmd, curve=curve)
    )
    if not actions:
        return {"blocked": True, "reason": "practice_low_x_one_wheel_no_actions", "cmd": turn_cmd}
    return follow.guarded_send_custom_actions_pwm(
        robot,
        turn_cmd,
        actions,
        duration_ms=int(duration_ms),
        reading=reading,
        context=f"practice_clean_reset_low_x_one_wheel_{turn_cmd}",
    )


def _practice_final_one_wheel_finish(
    reset_result: dict | None,
    vision: BrickDetector,
    robot: Robot,
    reset_cfg: dict,
) -> dict | None:
    if not PRACTICE_RESET_FINAL_ONE_WHEEL_FINISH_ENABLED or not isinstance(reset_result, dict):
        return reset_result
    reading = reset_result.get("reading") if isinstance(reset_result.get("reading"), dict) else None
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return reset_result
    turn_cmd = random.choice(("l", "r"))
    duration_ms = int(PRACTICE_RESET_FINAL_ONE_WHEEL_FINISH_MS)
    send_result = _practice_low_x_one_wheel_turn(
        robot,
        turn_cmd=turn_cmd,
        reading=reading,
        reset_cfg=reset_cfg,
        duration_ms=duration_ms,
    )
    print(
        f"[PRACTICE] final reset one-wheel finish: "
        f"{turn_cmd.upper()} {duration_ms}ms after reset",
        flush=True,
    )
    result = dict(reset_result)
    result["practice_final_one_wheel_finish"] = {
        "turn_cmd": turn_cmd,
        "duration_ms": duration_ms,
        "send_result": send_result,
    }
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        return result
    current = follow._reset_read_after_adjustment(
        vision,
        robot,
        duration_ms=duration_ms,
        settle_s=0.05,
        context="practice_final_one_wheel_finish",
        fallback=reading,
    )
    result["reading"] = current
    result["target_met"] = bool(follow._reset_xy_target_ready(current, reset_cfg))
    if result["target_met"]:
        result["reason"] = "target_hit_after_final_one_wheel_finish"
    return result


def _half_dist_reset_config(base_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    reverse = cfg.get("reverse_turn") if isinstance(cfg.get("reverse_turn"), dict) else {}
    reverse["post_pause_s"] = PRACTICE_RESET_POST_PAUSE_S
    reverse["settle_s"] = PRACTICE_RESET_SETTLE_S
    reverse["dist_target_mm"] = float(follow._dist_target_mm()) + float(PRACTICE_RESET_DIST_OFFSET_MM)
    reverse["dist_tol_mm"] = float(PRACTICE_RESET_DIST_TOL_MM)
    reverse["x_offset_min_mm"] = PRACTICE_RESET_X_OFFSET_MIN_MM
    reverse["x_offset_max_mm"] = PRACTICE_RESET_X_OFFSET_MAX_MM
    reverse["target_abs_x_mm"] = PRACTICE_RESET_TARGET_ABS_X_MM
    straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
    try:
        original_ms = int(round(float(straight.get("duration_ms"))))
    except (TypeError, ValueError):
        original_ms = int(follow.DEFAULT_RESET_STRAIGHT_BACK_FIRST_CONFIG["duration_ms"])
    straight["enabled"] = True
    straight["duration_ms"] = max(1, int(round(float(original_ms) * PRACTICE_RESET_STRAIGHT_SCALE)))
    straight["duration_min_ms"] = int(PRACTICE_RESET_STRAIGHT_MIN_MS)
    straight["duration_max_ms"] = int(PRACTICE_RESET_STRAIGHT_MAX_MS)
    straight["practice_original_duration_ms"] = int(original_ms)
    reverse["straight_back_first"] = straight
    x_goal_curve = reverse.get("x_goal_curve") if isinstance(reverse.get("x_goal_curve"), dict) else {}
    if x_goal_curve:
        try:
            original_curve_max_ms = int(round(float(x_goal_curve.get("max_duration_ms"))))
        except (TypeError, ValueError):
            original_curve_max_ms = 375
        x_goal_curve["practice_original_max_duration_ms"] = int(original_curve_max_ms)
        x_goal_curve["max_duration_ms"] = min(int(original_curve_max_ms), PRACTICE_RESET_X_CURVE_MAX_MS)
        x_goal_curve["min_duration_ms"] = min(PRACTICE_RESET_X_CURVE_MIN_MS, int(x_goal_curve["max_duration_ms"]))
        x_goal_curve["settle_s"] = PRACTICE_RESET_SETTLE_S
        x_goal_curve["practice_cap_reason"] = "avoid_losing_brick_after_step1_practice_reset_turn"
        reverse["x_goal_curve"] = x_goal_curve
    adjustment = reverse.get("adjustment") if isinstance(reverse.get("adjustment"), dict) else {}
    adjustment["enabled"] = False
    adjustment["max_attempts"] = 0
    adjustment["pulse_min_ms"] = 80
    adjustment["pulse_max_ms"] = 160
    adjustment["settle_s"] = 0.05
    adjustment["practice_reason"] = "step1_empty_practice_disables_reset_polish_after_observed_x_regressions"
    reverse["adjustment"] = adjustment
    cfg["reverse_turn"] = reverse
    mast_up = cfg.get("mast_up") if isinstance(cfg.get("mast_up"), dict) else {}
    mast_up["enabled"] = False
    mast_up["practice_disabled_reason"] = "step1_empty_practice_keeps_mast_in_vision_zone"
    cfg["mast_up"] = mast_up
    return cfg


def _smooth_step1_practice_config(base_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["max_act_ms"] = max(
        int(cfg.get("max_act_ms", 0) or 0),
        PRACTICE_FAR_DIST_MAX_PULSE_MS,
    )
    dist_policy = cfg.get("dist_approach_policy") if isinstance(cfg.get("dist_approach_policy"), dict) else {}
    dist_policy["settle_after_act_s"] = 0.0
    dist_policy["max_forward_pulse_ms"] = PRACTICE_FAR_DIST_MAX_PULSE_MS
    dist_policy["full_forward_gap_mm"] = PRACTICE_FAR_DIST_FULL_GAP_MM
    dist_policy["near_target_min_pulse_ms"] = PRACTICE_DRIVE_MIN_EFFECTIVE_MS
    dist_policy["near_target_max_pulse_ms"] = PRACTICE_NEAR_DIST_MAX_PULSE_MS
    dist_policy["close_target_max_pulse_ms"] = PRACTICE_DRIVE_MIN_EFFECTIVE_MS
    dist_policy["very_close_target_max_pulse_ms"] = PRACTICE_DRIVE_MIN_EFFECTIVE_MS
    dist_policy["micro_nudge_pulse_ms"] = PRACTICE_DRIVE_MIN_EFFECTIVE_MS
    dist_policy["micro_nudge_min_effective_pulse_ms"] = PRACTICE_DRIVE_MIN_EFFECTIVE_MS
    dist_policy["min_effective_drive_pulse_ms"] = PRACTICE_DRIVE_MIN_EFFECTIVE_MS
    dist_policy["pingpong_confirm_abs_dist_err_mm"] = PRACTICE_DIST_PINGPONG_CONFIRM_MM
    cfg["dist_approach_policy"] = dist_policy

    x_dist_policy = cfg.get("x_dist_curve_policy") if isinstance(cfg.get("x_dist_curve_policy"), dict) else {}
    x_dist_policy["combined_bias_max_pulse_ms"] = PRACTICE_BIAS_MAX_PULSE_MS
    cfg["x_dist_curve_policy"] = x_dist_policy

    x_policy = cfg.get("x_priority_policy") if isinstance(cfg.get("x_priority_policy"), dict) else {}
    x_policy["turn_settle_s"] = 0.0
    x_policy["simultaneous_x_abs_max_mm"] = 0.1
    x_policy["tiny_x_outside_no_turn_mm"] = 0.0
    x_policy["tiny_x_only_backoff_enabled"] = False
    cfg["x_priority_policy"] = x_policy

    jump_guard = cfg.get("vision_jump_guard") if isinstance(cfg.get("vision_jump_guard"), dict) else {}
    jump_guard["accept_confirmed_jumps"] = True
    jump_guard["confirm_frames"] = 1
    jump_guard["confirm_window_mm"] = 12.0
    jump_guard["max_x_jump_mm"] = 45.0
    jump_guard["max_y_jump_mm"] = 18.0
    jump_guard["max_vector_jump_mm"] = 60.0
    jump_guard["reacquire_frames_after_motion"] = 0
    jump_guard["post_act_stabilize_s"] = PRACTICE_STEP1_POST_ACT_STABILIZE_S
    cfg["vision_jump_guard"] = jump_guard

    win_cfg = cfg.get("win_confirmation") if isinstance(cfg.get("win_confirmation"), dict) else {}
    win_cfg["settle_s"] = PRACTICE_STEP1_WIN_SETTLE_S
    win_cfg["confirm_frames"] = PRACTICE_STEP1_WIN_CONFIRM_FRAMES
    win_cfg["accept_live_happy_after_stop"] = True
    cfg["win_confirmation"] = win_cfg

    cautious = cfg.get("cautious_visibility") if isinstance(cfg.get("cautious_visibility"), dict) else {}
    cautious["pregame_required_frames"] = 2
    cfg["cautious_visibility"] = cautious

    # _set_game_profile("empty") materializes the active profile into the
    # top-level y_axis before this practice override is installed.
    y_axis = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    y_axis["mast_coast_settle_s"] = max(
        PRACTICE_STEP1_MAST_COAST_SETTLE_S,
        float(y_axis.get("mast_coast_settle_s", PRACTICE_STEP1_MAST_COAST_SETTLE_S) or 0.0),
    )
    y_axis["priority_abs_err_mm"] = 999.0
    y_axis["attach_y_min_abs_err_mm"] = 999.0
    y_axis["practice_reason"] = (
        "smooth_step1_practice_keeps_mast_corrections_separate_from_wheel_acts"
    )
    cfg["y_axis"] = y_axis

    profiles = cfg.get("game_profiles") if isinstance(cfg.get("game_profiles"), dict) else {}
    empty = profiles.get("empty") if isinstance(profiles.get("empty"), dict) else {}
    profile_x_policy = empty.get("x_priority_policy") if isinstance(empty.get("x_priority_policy"), dict) else {}
    profile_x_policy["turn_settle_s"] = 0.0
    profile_x_policy["simultaneous_x_abs_max_mm"] = 0.1
    profile_x_policy["tiny_x_outside_no_turn_mm"] = 0.0
    profile_x_policy["tiny_x_only_backoff_enabled"] = False
    empty["x_priority_policy"] = profile_x_policy
    profile_x_dist_policy = empty.get("x_dist_curve_policy") if isinstance(empty.get("x_dist_curve_policy"), dict) else {}
    profile_x_dist_policy["combined_bias_max_pulse_ms"] = PRACTICE_BIAS_MAX_PULSE_MS
    empty["x_dist_curve_policy"] = profile_x_dist_policy
    profile_y_axis = empty.get("y_axis") if isinstance(empty.get("y_axis"), dict) else {}
    profile_y_axis.update({
        "mast_coast_settle_s": y_axis["mast_coast_settle_s"],
        "priority_abs_err_mm": 999.0,
        "attach_y_min_abs_err_mm": 999.0,
        "practice_reason": y_axis["practice_reason"],
    })
    empty["y_axis"] = profile_y_axis
    profiles["empty"] = empty
    cfg["game_profiles"] = profiles
    return cfg


def _install_practice_duration_overrides():
    """Practice-only asymmetric dist curves: backward correction is shorter."""
    original_creep = follow._distance_creep_duration_ms
    original_correction = follow._distance_correction_duration_ms
    original_y_plan = follow._y_axis_action_plan

    def _cap_backward(dist_err: float, duration_ms: int) -> int:
        if float(dist_err) >= 0.0:
            return int(duration_ms)
        return min(int(duration_ms), int(PRACTICE_BACKWARD_MAX_PULSE_MS))

    def _practice_distance_creep_duration_ms(dist_err: float) -> int:
        return _cap_backward(dist_err, original_creep(dist_err))

    def _practice_distance_correction_duration_ms(dist_err: float) -> int:
        return _cap_backward(dist_err, original_correction(dist_err))

    def _practice_y_axis_action_plan(reading: dict, *, dist_err: float, x_err: float) -> dict | None:
        if abs(float(dist_err)) > PRACTICE_Y_DIST_WINDOW_MM or abs(float(x_err)) > PRACTICE_Y_X_WINDOW_MM:
            return None
        plan = original_y_plan(reading, dist_err=dist_err, x_err=x_err)
        if isinstance(plan, dict) and str(plan.get("kind")) == "mast":
            try:
                abs_y_err = abs(float(plan.get("y_err")))
            except (TypeError, ValueError):
                abs_y_err = PRACTICE_Y_LARGE_GAP_PULSE_MS
            if abs_y_err <= 8.0:
                cap_ms = PRACTICE_Y_SMALL_GAP_PULSE_MS
            elif abs_y_err <= 16.0:
                cap_ms = PRACTICE_Y_MEDIUM_GAP_PULSE_MS
            else:
                cap_ms = PRACTICE_Y_LARGE_GAP_PULSE_MS
            plan = dict(plan)
            plan["duration_ms"] = min(
                int(plan.get("duration_ms", cap_ms) or cap_ms),
                cap_ms,
            )
            plan["practice_y_cap_ms"] = cap_ms
        return plan

    follow._distance_creep_duration_ms = _practice_distance_creep_duration_ms
    follow._distance_correction_duration_ms = _practice_distance_correction_duration_ms
    follow._y_axis_action_plan = _practice_y_axis_action_plan

    def restore() -> None:
        follow._distance_creep_duration_ms = original_creep
        follow._distance_correction_duration_ms = original_correction
        follow._y_axis_action_plan = original_y_plan

    return restore


def _step1_record(trial: int, stats: dict, reset_result: dict | None) -> dict:
    win = stats.get("last_step1_win") if isinstance(stats.get("last_step1_win"), dict) else None
    latest = stats.get("latest_step1_gap") if isinstance(stats.get("latest_step1_gap"), dict) else None
    diagnosis = _diagnose_step1_attempt(stats, reset_result)
    return {
        "trial": int(trial),
        "won": bool(int(stats.get("win_count", 0) or 0) > 0),
        "win": win,
        "latest": latest,
        "movement_attempts": int(stats.get("follow_attempt_count", 0) or 0),
        "stall_guard_triggered": bool(stats.get("stall_guard_triggered")),
        "stall_recovery_boost_count": int(stats.get("stall_recovery_boost_count", 0) or 0),
        "miss_reasons": dict(stats.get("miss_reasons") or {}),
        "diagnosis": diagnosis,
        "reset": reset_result,
    }


def _diagnose_step1_attempt(stats: dict, reset_result: dict | None) -> dict:
    won = bool(int(stats.get("win_count", 0) or 0) > 0)
    latest = stats.get("latest_step1_gap") if isinstance(stats.get("latest_step1_gap"), dict) else {}
    miss_reasons = stats.get("miss_reasons") if isinstance(stats.get("miss_reasons"), dict) else {}
    reset = reset_result if isinstance(reset_result, dict) else {}
    reset_reading = reset.get("reading") if isinstance(reset.get("reading"), dict) else {}

    axis_errors: dict[str, float] = {}
    for axis in ("dist", "x", "y"):
        try:
            axis_errors[axis] = abs(float(latest.get(f"{axis}_err")))
        except (TypeError, ValueError):
            pass
    worst_axis = max(axis_errors, key=axis_errors.get) if axis_errors else None
    worst_axis_error = axis_errors.get(worst_axis) if worst_axis is not None else None

    reset_issue = None
    if reset and not bool(reset.get("target_met")):
        reset_issue = str(reset.get("reason") or "reset_target_missed")
    try:
        reset_x = float(reset_reading.get("x_mm"))
        if abs(reset_x) > PRACTICE_RESET_X_OFFSET_MAX_MM:
            reset_issue = "reset_x_outside_honest_band"
    except (TypeError, ValueError):
        pass
    try:
        reset_dist = float(reset_reading.get("dist_mm"))
        reset_cfg = reset.get("config") if isinstance(reset.get("config"), dict) else {}
        dist_target = float(reset_cfg.get("dist_target_mm", follow._dist_target_mm() + PRACTICE_RESET_DIST_OFFSET_MM))
        dist_tol = float(reset_cfg.get("dist_tol_mm", PRACTICE_RESET_DIST_TOL_MM))
        if abs(reset_dist - dist_target) > dist_tol:
            reset_issue = "reset_dist_outside_honest_band"
    except (TypeError, ValueError):
        pass

    primary = "won"
    fix_hint = "keep current curve settings"
    must_investigate = False
    if not won:
        must_investigate = True
        if bool(stats.get("stall_guard_triggered")):
            primary = "stall_or_below_floor_motion"
            fix_hint = "raise or reroute the selected act; do not repeat a command that produces no observed movement"
        elif reset_issue is not None:
            primary = reset_issue
            fix_hint = "fix reset before counting this as an honest trial"
        elif "dist_pingpong_confirm_failed" in miss_reasons:
            primary = "dist_pingpong_or_momentum_overshoot"
            fix_hint = "shorten the previous dist act or add stronger stopped-read confirmation before reversing"
        elif worst_axis is not None:
            primary = f"{worst_axis}_gap_not_closed"
            fix_hint = f"adjust the {worst_axis} curve or command direction before more bulk trials"
        elif int(stats.get("follow_attempt_count", 0) or 0) <= 0:
            primary = "visibility_or_pregame_failed"
            fix_hint = "recover brick visibility before counting another attempt"
        else:
            primary = "unknown_miss"
            fix_hint = "inspect the raw stats before continuing"

    return {
        "primary": primary,
        "fix_hint": fix_hint,
        "must_investigate": bool(must_investigate),
        "worst_axis": worst_axis,
        "worst_axis_abs_err_mm": worst_axis_error,
        "reset_issue": reset_issue,
    }


def _print_step1_diagnosis(record: dict) -> None:
    diagnosis = record.get("diagnosis") if isinstance(record.get("diagnosis"), dict) else {}
    if bool(record.get("won")):
        print(
            f"[PRACTICE] trial {int(record.get('trial', 0) or 0)} diagnosis: win; "
            f"{diagnosis.get('fix_hint', 'keep current curve settings')}",
            flush=True,
        )
        return
    print(
        f"[PRACTICE] trial {int(record.get('trial', 0) or 0)} diagnosis: "
        f"{diagnosis.get('primary', 'unknown_miss')} — {diagnosis.get('fix_hint', 'inspect before continuing')}",
        flush=True,
    )


def _reset_target_met(reset_result: dict | None) -> bool:
    return bool(isinstance(reset_result, dict) and reset_result.get("target_met"))


def _format_win(win: dict | None) -> str:
    if not isinstance(win, dict):
        return "no Step 1 win"
    try:
        return (
            f"dist_err={float(win.get('dist_err_mm')):+.1f}mm "
            f"x_err={float(win.get('x_err_mm')):+.1f}mm "
            f"y_err={float(win.get('y_err_mm')):+.1f}mm"
        )
    except (TypeError, ValueError):
        return "Step 1 win recorded"


def _reset_result_recovered(reset_result: dict, vision: BrickDetector, robot: Robot) -> dict:
    result = dict(reset_result or {})
    if bool(result.get("success")):
        if result.get("target_met") is False:
            result["practice_target_miss_accepted"] = True
            result["reason"] = f"{result.get('reason') or 'reset'}_practice_partial_target_missed"
        return result
    reason = str(result.get("reason") or "")
    if reason not in {
        "lost_confident_brick_after_reset_x_curve",
        "lost_confident_brick_after_reset",
        "lost_confident_brick_after_reset_straight_back",
    }:
        return result
    print(
        f"[PRACTICE] reset visibility recovery: {reason}; pausing and reacquiring before giving up.",
        flush=True,
    )
    time.sleep(max(0.0, float(follow._reset_post_pause_s())))
    reading = result.get("reading") if isinstance(result.get("reading"), dict) else {}
    recovered = follow._wait_for_visibility_recovery(
        vision,
        robot,
        reading,
        timeout_s=RESET_RECOVERY_WAIT_S,
        sample_s=RESET_RECOVERY_POLL_S,
        context="practice_reset_recovery",
    )
    result["reading"] = recovered
    result["recovery_attempted"] = True
    if bool(recovered.get("confident")):
        reset_cfg = follow._reset_motion_config().get("reverse_turn")
        target_met = follow._reset_xy_target_ready(recovered, reset_cfg if isinstance(reset_cfg, dict) else {})
        result["success"] = True
        result["target_met"] = bool(target_met)
        result["reason"] = f"{reason}_visibility_recovered"
        return result
    result["recovery_failed"] = True
    return result


def _clean_reset_low_x_pose(reset_result: dict | None, vision: BrickDetector, robot: Robot, reset_cfg: dict) -> dict | None:
    if not PRACTICE_CLEAN_RESET_LOW_X_ENABLED or not isinstance(reset_result, dict):
        return reset_result
    reading = reset_result.get("reading") if isinstance(reset_result.get("reading"), dict) else None
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return reset_result
    try:
        x_mm = float(reading.get("x_mm"))
    except (TypeError, ValueError):
        return reset_result
    min_abs_x = float(PRACTICE_CLEAN_RESET_LOW_X_MIN_MM)
    max_abs_x = float(PRACTICE_RESET_X_OFFSET_MAX_MM)
    if min_abs_x <= abs(x_mm) <= max_abs_x:
        return reset_result
    result = dict(reset_result)
    attempts = []
    current = dict(reading)
    if abs(x_mm) > max_abs_x:
        last_abs_x = abs(x_mm)
        turn_cmd = None
        for attempt_idx in range(PRACTICE_CLEAN_RESET_HIGH_X_MAX_ATTEMPTS):
            try:
                current_x = float(current.get("x_mm"))
            except (TypeError, ValueError):
                break
            current_abs_x = abs(current_x)
            if min_abs_x <= current_abs_x <= max_abs_x:
                break
            if turn_cmd is None:
                turn_cmd = "r" if current_x > 0.0 else "l"
            elif current_abs_x > last_abs_x + 1.0:
                turn_cmd = "l" if turn_cmd == "r" else "r"
            last_abs_x = current_abs_x
            duration_ms = int(PRACTICE_CLEAN_RESET_HIGH_X_PULSE_MS)
            print(
                f"[PRACTICE] clean reset high-x turn {attempt_idx + 1}: "
                f"{turn_cmd.upper()} {duration_ms}ms from x={current_x:+.1f}mm",
                flush=True,
            )
            send_result = follow._send_turn_curve(
                robot,
                cmd=turn_cmd,
                drive_mode="forward",
                strength="strong",
                reading=current,
                context="practice_clean_reset_high_x",
                duration_ms=duration_ms,
            )
            if send_result is None:
                send_result = robot.send_command_pwm(
                    turn_cmd,
                    103,
                    duration_ms=duration_ms,
                )
            follow._stop_robot(robot)
            attempts.append({
                "turn_cmd": turn_cmd,
                "duration_ms": duration_ms,
                "before_x_mm": current_x,
                "before_abs_x_mm": current_abs_x,
                "send_result": send_result,
                "reason": "high_x_curve_turn",
            })
            current = follow._reset_read_after_adjustment(
                vision,
                robot,
                duration_ms=duration_ms,
                settle_s=0.05,
                context="practice_clean_reset_high_x",
                fallback=current,
            )
            if not bool(current.get("confident")):
                break
        result["reading"] = current
        result["practice_clean_low_x_attempts"] = attempts
        try:
            cleaned_x = float(current.get("x_mm"))
            result["practice_clean_low_x_success"] = bool(min_abs_x <= abs(cleaned_x) <= max_abs_x)
            result["practice_clean_low_x_target_mm"] = float(PRACTICE_CLEAN_RESET_LOW_X_TARGET_MM)
            result["target_met"] = bool(follow._reset_xy_target_ready(current, reset_cfg))
            if result["target_met"]:
                result["reason"] = "target_hit_after_practice_high_x_cleanup"
        except (TypeError, ValueError):
            result["practice_clean_low_x_success"] = False
        return result
    last_abs_x = abs(x_mm)
    turn_cmd = random.choice(("l", "r"))
    for attempt_idx in range(PRACTICE_CLEAN_RESET_LOW_X_MAX_ATTEMPTS):
        try:
            current_x = float(current.get("x_mm"))
        except (TypeError, ValueError):
            break
        current_abs_x = abs(current_x)
        if min_abs_x <= current_abs_x <= max_abs_x:
            break
        if attempts and current_abs_x > last_abs_x + 2.0:
            turn_cmd = "r" if turn_cmd == "l" else "l"
        last_abs_x = current_abs_x
        send_result = _practice_low_x_one_wheel_turn(
            robot,
            turn_cmd=turn_cmd,
            reading=current,
            duration_ms=PRACTICE_CLEAN_RESET_LOW_X_PULSE_MS,
            reset_cfg=reset_cfg,
        )
        print(
            f"[PRACTICE] clean reset low-x one-wheel finish {attempt_idx + 1}: "
            f"{turn_cmd.upper()} {PRACTICE_CLEAN_RESET_LOW_X_PULSE_MS}ms from x={current_x:+.1f}mm",
            flush=True,
        )
        attempts.append({
            "turn_cmd": turn_cmd,
            "duration_ms": PRACTICE_CLEAN_RESET_LOW_X_PULSE_MS,
            "before_x_mm": current_x,
            "before_abs_x_mm": current_abs_x,
            "send_result": send_result,
            "reason": "low_x_random_one_wheel_finish",
        })
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            break
        current = follow._reset_read_after_adjustment(
            vision,
            robot,
            duration_ms=PRACTICE_CLEAN_RESET_LOW_X_PULSE_MS,
            settle_s=0.05,
            context="practice_clean_reset_low_x",
            fallback=current,
        )
        if not bool(current.get("confident")):
            break
    result["reading"] = current
    result["practice_clean_low_x_attempts"] = attempts
    try:
        cleaned_x = float(current.get("x_mm"))
        result["practice_clean_low_x_success"] = bool(min_abs_x <= abs(cleaned_x) <= max_abs_x)
        result["practice_clean_low_x_target_mm"] = float(PRACTICE_CLEAN_RESET_LOW_X_TARGET_MM)
        result["target_met"] = bool(follow._reset_xy_target_ready(current, reset_cfg))
        if result["target_met"]:
            result["reason"] = "target_hit_after_practice_x_cleanup"
    except (TypeError, ValueError):
        result["practice_clean_low_x_success"] = False
    return result


def _clean_reset_dist_pose(reset_result: dict | None, vision: BrickDetector, robot: Robot, reset_cfg: dict) -> dict | None:
    if not PRACTICE_CLEAN_RESET_DIST_ENABLED or not isinstance(reset_result, dict):
        return reset_result
    reading = reset_result.get("reading") if isinstance(reset_result.get("reading"), dict) else None
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return reset_result
    try:
        target = float(reset_cfg.get("dist_target_mm"))
        tol = float(reset_cfg.get("dist_tol_mm"))
    except (TypeError, ValueError):
        return reset_result
    low = target - tol
    high = target + tol
    result = dict(reset_result)
    current = dict(reading)
    attempts = []
    for attempt_idx in range(PRACTICE_CLEAN_RESET_DIST_MAX_ATTEMPTS):
        try:
            dist_mm = float(current.get("dist_mm"))
        except (TypeError, ValueError):
            break
        if low <= dist_mm <= high:
            break
        cmd = "f" if dist_mm > high else "b"
        print(
            f"[PRACTICE] clean reset dist {attempt_idx + 1}: "
            f"{cmd.upper()} {PRACTICE_CLEAN_RESET_DIST_PULSE_MS}ms from dist={dist_mm:.1f}mm "
            f"target={target:.1f}+/-{tol:.1f}mm",
            flush=True,
        )
        send_result = follow._drive(
            robot,
            cmd,
            current,
            duration_ms=PRACTICE_CLEAN_RESET_DIST_PULSE_MS,
        )
        attempts.append({
            "cmd": cmd,
            "duration_ms": PRACTICE_CLEAN_RESET_DIST_PULSE_MS,
            "before_dist_mm": dist_mm,
            "send_result": send_result,
        })
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            break
        current = follow._reset_read_after_adjustment(
            vision,
            robot,
            duration_ms=PRACTICE_CLEAN_RESET_DIST_PULSE_MS,
            settle_s=0.05,
            context="practice_clean_reset_dist",
            fallback=current,
        )
        if not bool(current.get("confident")):
            break
    result["reading"] = current
    result["practice_clean_dist_attempts"] = attempts
    try:
        cleaned_dist = float(current.get("dist_mm"))
        result["practice_clean_dist_success"] = bool(low <= cleaned_dist <= high)
        result["target_met"] = bool(follow._reset_xy_target_ready(current, reset_cfg))
        if result["target_met"]:
            result["reason"] = "target_hit_after_practice_dist_cleanup"
    except (TypeError, ValueError):
        result["practice_clean_dist_success"] = False
    return result


def _run_practice_reset_until_target(
    vision: BrickDetector,
    robot: Robot,
    reset_cfg: dict,
    original_reset_config_fn,
    *,
    label: str,
) -> dict | None:
    last_result = None
    for reset_attempt in range(1, PRACTICE_RESET_MAX_TARGET_RETRIES + 1):
        follow._reset_motion_config = lambda: copy.deepcopy(reset_cfg)
        try:
            reset_result = follow._run_reset_sequence(vision, robot, rng=random)
            reset_result = _reset_result_recovered(reset_result, vision, robot)
            reset_result = _clean_reset_dist_pose(reset_result, vision, robot, reset_cfg.get("reverse_turn", {}))
            reset_result = _clean_reset_low_x_pose(reset_result, vision, robot, reset_cfg.get("reverse_turn", {}))
            reset_result = _practice_final_one_wheel_finish(reset_result, vision, robot, reset_cfg.get("reverse_turn", {}))
        finally:
            follow._reset_motion_config = original_reset_config_fn
        if isinstance(reset_result, dict):
            reset_result["practice_reset_attempt"] = int(reset_attempt)
            reset_result["practice_reset_target_retry_max"] = int(PRACTICE_RESET_MAX_TARGET_RETRIES)
        last_result = reset_result
        target_met = bool((reset_result or {}).get("target_met"))
        print(
            f"[PRACTICE] {label} reset attempt {reset_attempt}/{PRACTICE_RESET_MAX_TARGET_RETRIES}: "
            f"success={bool((reset_result or {}).get('success'))} "
            f"target_met={target_met} reason={(reset_result or {}).get('reason')} "
            f"turn={str((reset_result or {}).get('turn_cmd') or '').upper()}",
            flush=True,
        )
        if target_met:
            return reset_result
        if reset_attempt < PRACTICE_RESET_MAX_TARGET_RETRIES:
            _recover_start_visibility(
                vision,
                robot,
                reason=f"{label} reset target miss retry",
            )
    return last_result


def _recover_start_visibility(vision: BrickDetector, robot: Robot, *, reason: str) -> bool:
    print(
        f"[PRACTICE] visibility recovery before counting next trial: {reason}; waiting for brick lock.",
        flush=True,
    )
    follow._stop_robot(robot)
    recovered = follow._wait_for_confident_brick(
        vision,
        timeout_s=START_RECOVERY_WAIT_S,
        sample_s=START_RECOVERY_POLL_S,
    )
    if bool(recovered.get("confident")):
        return True
    if not bool(LOST_BRICK_SHARP_RIGHT_RECOVERY_ENABLED):
        return False
    elapsed_ms = 0
    max_ms = int(LOST_BRICK_SHARP_RIGHT_MAX_MS)
    chunk_ms = max(1, int(LOST_BRICK_SHARP_RIGHT_CHUNK_MS))
    print(
        f"[PRACTICE] brick still not visible; sharp-right recovery up to {max_ms}ms.",
        flush=True,
    )
    while elapsed_ms < max_ms:
        this_ms = min(chunk_ms, max_ms - elapsed_ms)
        robot.send_command_pwm("r", int(LOST_BRICK_SHARP_RIGHT_PWM), duration_ms=int(this_ms))
        follow._stop_robot(robot)
        elapsed_ms += int(this_ms)
        recovered = follow._wait_for_confident_brick(
            vision,
            timeout_s=float(LOST_BRICK_SHARP_RIGHT_RECHECK_S),
            sample_s=START_RECOVERY_POLL_S,
        )
        if bool(recovered.get("confident")):
            print(
                f"[PRACTICE] sharp-right recovery restored brick visibility after {elapsed_ms}ms.",
                flush=True,
            )
            return True
    print(
        f"[PRACTICE] sharp-right recovery did not restore brick visibility after {elapsed_ms}ms.",
        flush=True,
    )
    if bool(LOST_BRICK_BACKUP_RECOVERY_ENABLED):
        backup_ms = max(1, int(LOST_BRICK_BACKUP_RECOVERY_MS))
        print(
            f"[PRACTICE] trying bounded straight-back visibility escape: B {backup_ms}ms.",
            flush=True,
        )
        robot.send_command_pwm("b", int(LOST_BRICK_BACKUP_RECOVERY_PWM), duration_ms=int(backup_ms))
        follow._stop_robot(robot)
        recovered = follow._wait_for_confident_brick(
            vision,
            timeout_s=float(LOST_BRICK_BACKUP_RECHECK_S),
            sample_s=START_RECOVERY_POLL_S,
        )
        if bool(recovered.get("confident")):
            print("[PRACTICE] straight-back recovery restored brick visibility.", flush=True)
            return True
    return False


def _recover_start_y_zone(vision: BrickDetector, robot: Robot) -> bool:
    """Best-effort y pre-alignment.

    Practice runs should persist into the real attempt even if this helper
    cannot make y tidy. The follow loop is the truth source for winning.
    """
    y_cfg = follow._follow_y_axis_config()
    if not bool(y_cfg.get("enabled")):
        return True
    try:
        target = float(y_cfg.get("win_target_mm", follow.Y_TARGET_MM))
        tol = float(y_cfg.get("win_tol_mm", follow.Y_TOL_MM))
    except (TypeError, ValueError):
        return True
    ready_abs_err = max(START_Y_RECOVERY_ABS_ERR_MM, float(tol) * 2.0)
    last_err = None
    previous_abs_err = None
    worsening_count = 0
    for act_index in range(START_Y_RECOVERY_MAX_ACTS + 1):
        reading = follow._read_brick_measurement(vision, jump_guard=True)
        if not bool((reading or {}).get("confident")):
            print("[PRACTICE] y pre-recovery paused: brick confidence lost; proceeding to attempt.", flush=True)
            return True
        try:
            y_mm = float(reading.get("y_mm"))
        except (TypeError, ValueError):
            return False
        y_err = float(y_mm) - float(target)
        last_err = y_err
        abs_err = abs(y_err)
        if previous_abs_err is not None:
            if abs_err > previous_abs_err + 1.0:
                worsening_count += 1
            elif abs_err < previous_abs_err - 0.5:
                worsening_count = 0
            if worsening_count >= 2:
                print(
                    f"[PRACTICE] y pre-recovery paused: mast acts are worsening y "
                    f"({previous_abs_err:.1f}mm -> {abs_err:.1f}mm); proceeding to attempt.",
                    flush=True,
                )
                return True
        previous_abs_err = abs_err
        if abs(y_err) <= ready_abs_err:
            if act_index > 0:
                print(
                    f"[PRACTICE] y pre-recovery ready: y_err={y_err:+.1f}mm "
                    f"after {act_index} mast act(s).",
                    flush=True,
                )
            return True
        if act_index >= START_Y_RECOVERY_MAX_ACTS:
            break
        plan = follow._y_axis_action_plan(reading, dist_err=0.0, x_err=0.0)
        if not isinstance(plan, dict) or str(plan.get("kind") or "") != "mast":
            print(
                f"[PRACTICE] y pre-recovery blocked: y_err={y_err:+.1f}mm "
                f"plan={str((plan or {}).get('action') if isinstance(plan, dict) else plan)}; "
                "proceeding to attempt.",
                flush=True,
            )
            return True
        cmd = str(plan.get("cmd") or "").strip().lower()
        duration_ms = int(plan.get("duration_ms") or 0)
        pwm = int(plan.get("pwm") or 255)
        print(
            f"[PRACTICE] y pre-recovery {act_index + 1}/{START_Y_RECOVERY_MAX_ACTS}: "
            f"{cmd.upper()} {duration_ms}ms y_err={y_err:+.1f}mm.",
            flush=True,
        )
        result = follow.guarded_send_command_pwm(
            robot,
            cmd,
            pwm,
            duration_ms=duration_ms,
            reading=reading,
            context="practice_step1_start_y_recovery",
        )
        if isinstance(result, dict) and bool(result.get("blocked")):
            print(
                f"[PRACTICE] y pre-recovery command blocked: {result.get('reason')}; "
                "proceeding to attempt.",
                flush=True,
            )
            return True
        time.sleep(max(START_Y_RECOVERY_SAMPLE_S, float(follow._y_motion_coast_settle_s(y_cfg))))
    print(
        f"[PRACTICE] y pre-recovery did not reach start band: "
        f"last_y_err={float(last_err):+.1f}mm target_band=+/-{ready_abs_err:.1f}mm; "
        "proceeding to attempt.",
        flush=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--target-wins", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--continue-on-miss", action="store_true")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    trials = max(1, int(args.trials))
    target_wins = max(0, int(args.target_wins))
    max_attempts = max(0, int(args.max_attempts))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    follow._set_game_profile("empty")
    base_reset_cfg = follow._reset_motion_config()
    half_reset_cfg = _half_dist_reset_config(base_reset_cfg)
    base_straight = base_reset_cfg.get("reverse_turn", {}).get("straight_back_first", {})
    half_straight = half_reset_cfg.get("reverse_turn", {}).get("straight_back_first", {})
    print(
        f"[PRACTICE] Empty Step 1 x{trials}. Reset after each win uses half straight-back "
        f"distance duration: {int(base_straight.get('duration_ms', 0) or 0)}ms -> "
        f"{int(half_straight.get('duration_ms', 0) or 0)}ms; "
        f"practice reset x-curve capped at {PRACTICE_RESET_X_CURVE_MAX_MS}ms; "
        "reset mast lift disabled for Step 1 practice; "
        f"smooth mode post-act settle={PRACTICE_STEP1_POST_ACT_STABILIZE_S:.2f}s; "
        f"far dist crawl max={PRACTICE_FAR_DIST_MAX_PULSE_MS}ms at "
        f"{PRACTICE_FAR_DIST_FULL_GAP_MM:.0f}mm+ gaps.",
        flush=True,
    )
    vision = None
    robot = None
    completed = 0
    wins = 0
    original_reset_config_fn = follow._reset_motion_config
    original_follow_config = copy.deepcopy(follow._follow_motion_config())
    restore_duration_overrides = _install_practice_duration_overrides()
    try:
        follow._follow_motion_config._cache = _smooth_step1_practice_config(original_follow_config)
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        print("[PRACTICE] Camera ready.", flush=True)
        follow._print_active_success_gates()
        robot = Robot()
        with out_path.open("w", encoding="utf-8") as out:
            trial = 1
            recovery_skips = 0
            while True:
                if target_wins > 0:
                    if wins >= target_wins:
                        break
                    if max_attempts > 0 and completed >= max_attempts:
                        break
                elif trial > trials:
                    break
                label_total = f"target_wins={target_wins}" if target_wins > 0 else str(trials)
                print(f"[PRACTICE] trial {trial}/{label_total}: empty Step 1 (wins={wins})", flush=True)
                if not _recover_start_visibility(
                    vision,
                    robot,
                    reason="pre-trial visibility gate",
                ):
                    print(
                        "[PRACTICE] unable to recover brick vision after one full pre-trial recovery sequence; stopping.",
                        flush=True,
                    )
                    return 6
                if not _recover_start_y_zone(vision, robot):
                    recovery_skips += 1
                    if recovery_skips <= MAX_RECOVERY_SKIPS:
                        continue
                    print(
                        f"[PRACTICE] unable to recover Step 1 y zone after {recovery_skips} attempts; stopping.",
                        flush=True,
                    )
                    return 7
                stats = follow._follow_loop(
                    vision,
                    robot,
                    duration_s=FOLLOW_DURATION_S,
                    reset_after_win=False,
                    stop_after_win=True,
                    debug_mode=False,
                )
                completed += 1
                won = bool(int(stats.get("win_count", 0) or 0) > 0)
                wins += 1 if won else 0
                reset_result = None
                if not won:
                    if int(stats.get("follow_attempt_count", 0) or 0) <= 0 and _recover_start_visibility(
                        vision,
                        robot,
                        reason="no confident brick before any movement",
                    ):
                        recovery_skips += 1
                        if recovery_skips <= MAX_RECOVERY_SKIPS:
                            completed -= 1
                            continue
                    if int(stats.get("follow_attempt_count", 0) or 0) <= 0 and recovery_skips <= MAX_RECOVERY_SKIPS:
                        recovery_skips += 1
                        completed -= 1
                        continue
                    recovered_before_reset = _recover_start_visibility(
                        vision,
                        robot,
                        reason="trial missed Step 1 before practice reset",
                    )
                    if recovered_before_reset:
                        reset_result = _run_practice_reset_until_target(
                            vision,
                            robot,
                            half_reset_cfg,
                            original_reset_config_fn,
                            label=f"miss recovery after trial {trial}",
                        )
                        recovery_skips += 1
                        if recovery_skips <= MAX_RECOVERY_SKIPS:
                            record = _step1_record(trial, stats, reset_result)
                            record["recovered_with_practice_reset"] = True
                            _print_step1_diagnosis(record)
                            out.write(json.dumps(record, sort_keys=True) + "\n")
                            out.flush()
                            if not _reset_target_met(reset_result) and not bool(args.continue_on_miss):
                                print(
                                    "[PRACTICE] reset after miss was not honest; stopping before the next counted attempt.",
                                    flush=True,
                                )
                                return 8
                            trial += 1
                            continue
                    print(
                        "[PRACTICE] cannot see brick after miss recovery; hard-stopping block before more motion.",
                        flush=True,
                    )
                    record = _step1_record(trial, stats, reset_result)
                    _print_step1_diagnosis(record)
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    return 6
                    record = _step1_record(trial, stats, reset_result)
                    _print_step1_diagnosis(record)
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    if not bool(args.continue_on_miss):
                        print(
                            f"[PRACTICE] trial {trial} did not win; stopping before reset. "
                            f"Data: {out_path}",
                            flush=True,
                        )
                        return 4
                    print(
                        f"[PRACTICE] trial {trial} did not win; continuing without reset. "
                        f"Data: {out_path}",
                        flush=True,
                    )
                    trial += 1
                    continue
                more_to_run = (
                    (target_wins > 0 and wins < target_wins)
                    or (target_wins <= 0 and trial < trials)
                )
                if more_to_run:
                    print(f"[PRACTICE] trial {trial} win: {_format_win(stats.get('last_step1_win'))}", flush=True)
                    reset_result = _run_practice_reset_until_target(
                        vision,
                        robot,
                        half_reset_cfg,
                        original_reset_config_fn,
                        label=f"after win trial {trial}",
                    )
                    if not bool(reset_result.get("success")):
                        if _recover_start_visibility(
                            vision,
                            robot,
                            reason=f"reset failed: {reset_result.get('reason')}",
                        ):
                            recovery_skips += 1
                            if recovery_skips <= MAX_RECOVERY_SKIPS:
                                record = _step1_record(trial, stats, reset_result)
                                record["recovered_after_failed_reset"] = True
                                _print_step1_diagnosis(record)
                                out.write(json.dumps(record, sort_keys=True) + "\n")
                                out.flush()
                                if not _reset_target_met(reset_result) and not bool(args.continue_on_miss):
                                    print(
                                        "[PRACTICE] reset recovery failed to reach honest reset band; stopping before next counted attempt.",
                                        flush=True,
                                    )
                                    return 8
                                trial += 1
                                continue
                        record = _step1_record(trial, stats, reset_result)
                        _print_step1_diagnosis(record)
                        out.write(json.dumps(record, sort_keys=True) + "\n")
                        out.flush()
                        if not bool(args.continue_on_miss):
                            return 5
                        print(
                            f"[PRACTICE] reset failed after trial {trial}; continuing after recovery path. "
                            f"Data: {out_path}",
                            flush=True,
                        )
                    if reset_result is not None and not _reset_target_met(reset_result):
                        if not bool(args.continue_on_miss):
                            record = _step1_record(trial, stats, reset_result)
                            record["reset_not_honest_stop"] = True
                            _print_step1_diagnosis(record)
                            out.write(json.dumps(record, sort_keys=True) + "\n")
                            out.flush()
                            print(
                                "[PRACTICE] reset after win was not honest; stopping before the next counted attempt.",
                                flush=True,
                            )
                            return 8
                        print(
                            "[PRACTICE] reset after win was not honest; continue-on-miss: proceeding to next trial.",
                            flush=True,
                        )
                    time.sleep(0.0)
                record = _step1_record(trial, stats, reset_result)
                _print_step1_diagnosis(record)
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
                trial += 1
        print(f"[PRACTICE] completed {completed} trials; wins={wins}/{completed}; data={out_path}", flush=True)
        return 0
    finally:
        follow._reset_motion_config = original_reset_config_fn
        restore_duration_overrides()
        follow._follow_motion_config._cache = original_follow_config
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
