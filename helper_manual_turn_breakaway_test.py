#!/usr/bin/env python3
"""Manual raw-turn PWM floor search for Q/E hotkeys."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import deque
from pathlib import Path

import telemetry_robot as telemetry_robot_module
from helper_manual_drive_breakaway_test import (
    DEFAULT_PAUSE_BETWEEN_PULSES_MS,
    DEFAULT_VISION_MODE,
    POST_ACT_SETTLE_S,
    RECENTER_OBSERVE_SAMPLES,
    RECENTER_OBSERVE_TIMEOUT_S,
    RECENTER_TIMEOUT_S,
    RECENTER_TOLERANCE_DIST_MM,
    RECENTER_TOLERANCE_X_AXIS_MM,
    RECOVERY_MAX_INVERSE_ACTS,
    RECOVERY_OBSERVE_TIMEOUT_S,
    _build_vision,
    _clamp_positive_int,
    _collect_pose_samples,
    _format_pose_text,
    _hotkey_spec,
    _median_pose_or_none,
    _prompt_with_default,
    _recenter_status,
)
from helper_robot_control import Robot
from telemetry_process import reset_repeat_act_guard
from telemetry_robot import StepState, WorldModel


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_turn_breakaway_test.json"
TURN_HOTKEYS_DEFAULT = ("q", "e")
TURN_BREAKAWAY_ACT_DURATION_MS = 130
DEFAULT_DURATION_MS = TURN_BREAKAWAY_ACT_DURATION_MS
TURN_COARSE_SEARCH_TRIALS = 1
TURN_REFINE_SEARCH_TRIALS = 2
TURN_TRIAL_BUDGET_PER_HOTKEY = 20
TURN_PWM_BATCH_STEP = 5
TURN_COARSE_PWM_STEP_INITIAL = 12
TURN_FINAL_BRACKET_TOLERANCE_PWM = 2
DEFAULT_MOVEMENT_THRESHOLD_MM = 3.0
TURN_ABORT_CONSECUTIVE_DEAD_CANDIDATES = 2
TURN_ABORT_X_DRIFT_MM = 12.0
TURN_ABORT_DIST_DRIFT_MM = 10.0
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def _clamp_turn_breakaway_duration_ms(value, default: int = DEFAULT_DURATION_MS) -> int:
    duration_ms = _clamp_positive_int(value, default, minimum=1)
    return int(max(1, int(duration_ms)))


def _turn_curve_candidate_for_hotkey(hotkey: str, *, duration_ms: int) -> dict:
    spec = _hotkey_spec(hotkey)
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported hotkey: {hotkey}")
    cmd = str(spec.get("cmd") or "")
    score_map = telemetry_robot_module.score_power_pwm_for_cmd(cmd)
    row = score_map.get(1) if isinstance(score_map, dict) else None
    try:
        pwm = int(round(float((row or {}).get("pwm"))))
    except (TypeError, ValueError):
        pwm = 0
    if pwm <= 0:
        raise ValueError(f"Missing raw 1% PWM curve row for {cmd}")
    try:
        power = float((row or {}).get("power"))
    except (TypeError, ValueError):
        power = float(telemetry_robot_module.pwm_to_power(pwm) or 0.0)
    return {
        "hotkey": str(hotkey),
        "cmd": str(cmd),
        "display_label": str(spec.get("display_label") or str(hotkey).upper()),
        "metric_key": str(spec.get("metric_key") or "x_axis_mm"),
        "metric_label": str(spec.get("metric_label") or "x_axis"),
        "curve_score": 1,
        "curve_pwm": int(pwm),
        "curve_power": float(power),
        "curve_duration_ms": int(duration_ms),
        "pwm": int(pwm),
        "power": float(power),
        "duration_ms": int(duration_ms),
    }


def _turn_candidate_from_pwm(base_candidate: dict, *, pwm: int, phase: str, probe_index: int) -> dict:
    candidate = dict(base_candidate or {})
    pwm_value = max(1, int(round(float(pwm))))
    candidate["phase"] = str(phase)
    candidate["probe_index"] = int(probe_index)
    candidate["pwm"] = int(pwm_value)
    candidate["power"] = float(telemetry_robot_module.pwm_to_power(pwm_value) or 0.0)
    candidate["duration_ms"] = int(base_candidate.get("duration_ms") or DEFAULT_DURATION_MS)
    return candidate


def _next_coarse_pwm_step(previous_step: int, remaining_gap: int) -> int:
    gap = max(0, int(remaining_gap))
    if gap <= 0:
        return 0
    base_step = max(int(TURN_PWM_BATCH_STEP), int(TURN_COARSE_PWM_STEP_INITIAL))
    if int(previous_step) <= 0:
        next_step = int(base_step)
    else:
        next_step = max(int(base_step), int(previous_step) * 2)
    return max(1, min(int(gap), int(next_step)))


def _log_turn_trial(logger, row: dict) -> None:
    trial_number = int(row.get("trial_number") or 0)
    pwm_text = "unknown"
    power_text = "unknown"
    duration_text = "unknown"
    raw_delta_text = "unknown"
    try:
        pwm_text = str(int(round(float(row.get("pwm")))))
    except (TypeError, ValueError):
        pass
    try:
        power_text = f"{float(row.get('power')):.3f}"
    except (TypeError, ValueError):
        pass
    try:
        duration_text = f"{int(round(float(row.get('duration_ms'))))}ms"
    except (TypeError, ValueError):
        pass
    try:
        raw_delta_text = f"{float(row.get('raw_delta_mm')):+.3f}mm"
    except (TypeError, ValueError):
        pass
    result_label = str(row.get("result_label") or row.get("error") or "unknown")
    moved = bool(row.get("moved"))
    if moved and raw_delta_text != "unknown":
        raw_delta_text = f"{ANSI_GREEN}{raw_delta_text}{ANSI_RESET}"
    elif str(result_label).strip().lower() == "no movement":
        result_label = f"{ANSI_RED}{result_label}{ANSI_RESET}"
    logger(
        f"Trial {int(trial_number)}: {str(row.get('display_label') or '').strip()} "
        f"pwm={pwm_text}, pwr={power_text}, t={duration_text}; "
        f"x_delta={raw_delta_text}; result: {result_label}"
    )


def _record_recent_turn_pwm_act(recent_turn_acts, *, cmd: str, pwm: int, duration_ms: int) -> None:
    if recent_turn_acts is None:
        return
    cmd_norm = str(cmd or "").strip().lower()
    if cmd_norm not in {"l", "r"}:
        return
    act = {
        "cmd": str(cmd_norm),
        "pwm": max(1, int(round(float(pwm)))),
        "duration_ms": max(1, int(round(float(duration_ms)))),
        "timestamp": float(time.time()),
    }
    try:
        recent_turn_acts.append(act)
    except AttributeError:
        recent_turn_acts[:] = list(recent_turn_acts[-(int(RECOVERY_MAX_INVERSE_ACTS) - 1) :]) + [act]


def _inverse_turn_cmd(cmd: str | None) -> str | None:
    token = str(cmd or "").strip().lower()
    if token == "l":
        return "r"
    if token == "r":
        return "l"
    return None


def _recover_pose_from_recent_turn_pwm_acts(
    *,
    robot,
    vision,
    world,
    recent_turn_acts,
    source_label: str,
    target_pose: dict | None = None,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    history = [
        dict(item)
        for item in list(recent_turn_acts or [])
        if isinstance(item, dict) and _inverse_turn_cmd(item.get("cmd")) in {"l", "r"}
    ]
    if not history:
        return {
            "ok": False,
            "reason": "no_recent_turn_acts",
        }
    plan = []
    for act in reversed(history[-int(RECOVERY_MAX_INVERSE_ACTS) :]):
        inverse_cmd = _inverse_turn_cmd(act.get("cmd"))
        if inverse_cmd not in {"l", "r"}:
            continue
        plan.append(
            {
                "cmd": str(inverse_cmd),
                "pwm": max(1, int(round(float(act.get("pwm") or 0)))),
                "duration_ms": max(1, int(round(float(act.get("duration_ms") or DEFAULT_DURATION_MS)))),
                "origin_cmd": str(act.get("cmd") or ""),
            }
        )
    if not plan:
        return {
            "ok": False,
            "reason": "no_recovery_plan",
        }

    logger(
        f"[TURN BREAKAWAY TEST] Recovery: replay inverse of the most recent {int(len(plan))} turn act(s) after {str(source_label)}."
    )
    steps = []
    last_pose = None
    for index, step in enumerate(plan, start=1):
        reset_repeat_act_guard(world, robot)
        send_result = robot.send_command_pwm(
            str(step["cmd"]),
            int(step["pwm"]),
            duration_ms=int(step["duration_ms"]),
        )
        try:
            send_duration_ms = int(round(float((send_result or {}).get("duration_ms") or step.get("duration_ms") or 0)))
        except (TypeError, ValueError):
            send_duration_ms = int(step.get("duration_ms") or 1)
        time.sleep(max(float(POST_ACT_SETTLE_S), (float(send_duration_ms) / 1000.0) + 0.02))
        pose = _median_pose_or_none(
            _collect_pose_samples(
                vision,
                world,
                samples=1,
                timeout_s=float(RECOVERY_OBSERVE_TIMEOUT_S),
            )
        )
        step_result = {
            "index": int(index),
            "cmd": str(step["cmd"]),
            "pwm": int(step["pwm"]),
            "duration_ms": int(send_duration_ms),
            "origin_cmd": str(step.get("origin_cmd") or ""),
            "pose_found": bool(isinstance(pose, dict)),
        }
        if isinstance(pose, dict):
            last_pose = dict(pose)
            step_result["pose"] = dict(pose)
            if isinstance(target_pose, dict):
                step_result["target_status"] = _recenter_status(
                    pose,
                    target_pose,
                    tolerance_dist_mm=float(RECENTER_TOLERANCE_DIST_MM),
                    tolerance_x_axis_mm=float(RECENTER_TOLERANCE_X_AXIS_MM),
                )
        steps.append(step_result)
        if isinstance(pose, dict):
            if not isinstance(target_pose, dict) or bool((step_result.get("target_status") or {}).get("ok")):
                logger(
                    f"[TURN BREAKAWAY TEST] Recovery step {int(index)}/{int(len(plan))} regained vision with "
                    f"{str(step['cmd']).upper()} pwm={int(step['pwm'])} for {int(send_duration_ms)}ms."
                )
                return {
                    "ok": True,
                    "pose": dict(pose),
                    "steps": steps,
                    "source_label": str(source_label),
                }
            logger(
                f"[TURN BREAKAWAY TEST] Recovery step {int(index)}/{int(len(plan))} regained vision but is still off-baseline "
                f"({str(step['cmd']).upper()} pwm={int(step['pwm'])} for {int(send_duration_ms)}ms)."
            )
        logger(
            f"[TURN BREAKAWAY TEST] Recovery step {int(index)}/{int(len(plan))} did not regain vision "
            f"({str(step['cmd']).upper()} pwm={int(step['pwm'])} for {int(send_duration_ms)}ms)."
        )
    return {
        "ok": False,
        "reason": "recovery_pose_unavailable" if last_pose is None else "recovery_not_recentered",
        "pose": last_pose,
        "steps": steps,
        "source_label": str(source_label),
    }


def _observe_pose_with_optional_recovery(
    *,
    robot,
    vision,
    world,
    recent_turn_acts,
    source_label: str,
    log_fn=None,
) -> tuple[dict | None, dict | None]:
    pose = _median_pose_or_none(_collect_pose_samples(vision, world))
    if isinstance(pose, dict):
        return pose, None
    recovery_result = _recover_pose_from_recent_turn_pwm_acts(
        robot=robot,
        vision=vision,
        world=world,
        recent_turn_acts=recent_turn_acts,
        source_label=str(source_label),
        log_fn=log_fn,
    )
    recovered_pose = (recovery_result or {}).get("pose")
    if isinstance(recovered_pose, dict):
        return dict(recovered_pose), dict(recovery_result)
    return None, dict(recovery_result or {})


def _run_turn_pwm_recenter_checkpoint(
    *,
    robot,
    vision,
    world,
    baseline_pose: dict | None,
    hotkey: str,
    recent_turn_acts,
    checkpoint_label: str,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    if not isinstance(baseline_pose, dict):
        logger(f"[TURN BREAKAWAY TEST] Recenter checkpoint before {str(checkpoint_label)}: skipped (baseline unavailable).")
        return {
            "ok": False,
            "skipped": True,
            "reason": "baseline_unavailable",
        }
    logger(
        "[TURN BREAKAWAY TEST] Recenter checkpoint: "
        f"return near {_format_pose_text(baseline_pose)} before {str(checkpoint_label)}. "
        f"Window: +/-{float(RECENTER_TOLERANCE_DIST_MM):.1f}mm dist, "
        f"+/-{float(RECENTER_TOLERANCE_X_AXIS_MM):.1f}mm x_axis."
    )
    started = time.time()
    last_pose = None
    last_status = None
    recovery_result = None
    recovery_attempted = False
    while (time.time() - started) < float(RECENTER_TIMEOUT_S):
        pose = _median_pose_or_none(
            _collect_pose_samples(
                vision,
                world,
                samples=int(RECENTER_OBSERVE_SAMPLES),
                timeout_s=float(RECENTER_OBSERVE_TIMEOUT_S),
            )
        )
        status = _recenter_status(
            pose,
            baseline_pose,
            tolerance_dist_mm=float(RECENTER_TOLERANCE_DIST_MM),
            tolerance_x_axis_mm=float(RECENTER_TOLERANCE_X_AXIS_MM),
        )
        last_pose = pose
        last_status = status
        if not bool(status.get("ok")) and not recovery_attempted:
            recovery_attempted = True
            recovery_result = _recover_pose_from_recent_turn_pwm_acts(
                robot=robot,
                vision=vision,
                world=world,
                recent_turn_acts=recent_turn_acts,
                source_label=f"recenter before {str(checkpoint_label)}",
                target_pose=baseline_pose,
                log_fn=logger,
            )
            recovered_pose = (recovery_result or {}).get("pose")
            if isinstance(recovered_pose, dict):
                pose = dict(recovered_pose)
                status = _recenter_status(
                    pose,
                    baseline_pose,
                    tolerance_dist_mm=float(RECENTER_TOLERANCE_DIST_MM),
                    tolerance_x_axis_mm=float(RECENTER_TOLERANCE_X_AXIS_MM),
                )
                last_pose = pose
                last_status = status
        if bool(status.get("ok")):
            result = {
                "ok": True,
                "hotkey": str(hotkey),
                "pose": dict(pose or {}),
                "dist_error_mm": float(status.get("dist_error_mm") or 0.0),
                "x_axis_error_mm": float(status.get("x_axis_error_mm") or 0.0),
                "seconds": float(max(0.0, time.time() - started)),
            }
            if isinstance(recovery_result, dict):
                result["recovery"] = dict(recovery_result)
            logger(
                "[TURN BREAKAWAY TEST] Recenter checkpoint passed: "
                f"{_format_pose_text(pose)} "
                f"(dist_err={float(status.get('dist_error_mm') or 0.0):+.2f}mm, "
                f"x_err={float(status.get('x_axis_error_mm') or 0.0):+.2f}mm)."
            )
            return result
    result = {
        "ok": False,
        "timed_out": True,
        "hotkey": str(hotkey),
        "pose": dict(last_pose or {}),
        "dist_error_mm": None if not isinstance(last_status, dict) else last_status.get("dist_error_mm"),
        "x_axis_error_mm": None if not isinstance(last_status, dict) else last_status.get("x_axis_error_mm"),
        "seconds": float(max(0.0, time.time() - started)),
    }
    if isinstance(recovery_result, dict):
        result["recovery"] = dict(recovery_result)
    logger(
        "[TURN BREAKAWAY TEST] Recenter checkpoint timed out: "
        f"current {_format_pose_text(last_pose)} "
        f"(dist_err={float((last_status or {}).get('dist_error_mm') or 0.0):+.2f}mm, "
        f"x_err={float((last_status or {}).get('x_axis_error_mm') or 0.0):+.2f}mm). "
        "Continuing anyway."
    )
    return result


def _run_turn_pwm_trial(
    *,
    robot,
    vision,
    world,
    hotkey: str,
    pwm: int,
    duration_ms: int,
    movement_threshold_mm: float,
    recent_turn_acts,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    spec = _hotkey_spec(hotkey)
    if not isinstance(spec, dict):
        return {
            "ok": False,
            "hotkey": str(hotkey or ""),
            "error": "unsupported_hotkey",
        }

    cmd = str(spec.get("cmd") or "")
    metric_key = str(spec.get("metric_key") or "x_axis_mm")
    display_label = str(spec.get("display_label") or str(hotkey).upper())
    pre_pose, pre_recovery = _observe_pose_with_optional_recovery(
        robot=robot,
        vision=vision,
        world=world,
        recent_turn_acts=recent_turn_acts,
        source_label=f"{display_label} pwm={int(pwm)} pre_pose",
        log_fn=logger,
    )
    if pre_pose is None:
        row = {
            "ok": False,
            "hotkey": str(hotkey),
            "cmd": str(cmd),
            "pwm": int(pwm),
            "power": float(telemetry_robot_module.pwm_to_power(int(pwm)) or 0.0),
            "duration_ms": int(duration_ms),
            "display_label": display_label,
            "metric_key": metric_key,
            "metric_label": str(spec.get("metric_label") or "x_axis"),
            "error": "pre_pose_unavailable",
        }
        if isinstance(pre_recovery, dict):
            row["recovery"] = dict(pre_recovery)
        return row

    pre_metric_mm = pre_pose.get(metric_key)
    if pre_metric_mm is None:
        return {
            "ok": False,
            "hotkey": str(hotkey),
            "cmd": str(cmd),
            "pwm": int(pwm),
            "power": float(telemetry_robot_module.pwm_to_power(int(pwm)) or 0.0),
            "duration_ms": int(duration_ms),
            "display_label": display_label,
            "metric_key": metric_key,
            "metric_label": str(spec.get("metric_label") or "x_axis"),
            "error": "pre_metric_unavailable",
        }

    reset_repeat_act_guard(world, robot)
    send_result = robot.send_command_pwm(
        cmd,
        int(pwm),
        duration_ms=int(duration_ms),
    )
    try:
        send_duration_ms = int(round(float((send_result or {}).get("duration_ms") or duration_ms)))
    except (TypeError, ValueError):
        send_duration_ms = int(duration_ms)
    _record_recent_turn_pwm_act(
        recent_turn_acts,
        cmd=str(cmd),
        pwm=int(pwm),
        duration_ms=int(send_duration_ms),
    )
    time.sleep(max(float(POST_ACT_SETTLE_S), (float(send_duration_ms) / 1000.0) + 0.02))

    post_pose, post_recovery = _observe_pose_with_optional_recovery(
        robot=robot,
        vision=vision,
        world=world,
        recent_turn_acts=recent_turn_acts,
        source_label=f"{display_label} pwm={int(pwm)} post_pose",
        log_fn=logger,
    )
    if post_pose is None:
        row = {
            "ok": False,
            "hotkey": str(hotkey),
            "cmd": str(cmd),
            "pwm": int(pwm),
            "power": float(telemetry_robot_module.pwm_to_power(int(pwm)) or 0.0),
            "duration_ms": int(send_duration_ms),
            "display_label": display_label,
            "metric_key": metric_key,
            "metric_label": str(spec.get("metric_label") or "x_axis"),
            "pre_metric_mm": float(pre_metric_mm),
            "error": "post_pose_unavailable",
        }
        if isinstance(post_recovery, dict):
            row["recovery"] = dict(post_recovery)
        return row
    post_metric_mm = post_pose.get(metric_key)
    if post_metric_mm is None:
        return {
            "ok": False,
            "hotkey": str(hotkey),
            "cmd": str(cmd),
            "pwm": int(pwm),
            "power": float(telemetry_robot_module.pwm_to_power(int(pwm)) or 0.0),
            "duration_ms": int(send_duration_ms),
            "display_label": display_label,
            "metric_key": metric_key,
            "metric_label": str(spec.get("metric_label") or "x_axis"),
            "pre_metric_mm": float(pre_metric_mm),
            "error": "post_metric_unavailable",
        }

    raw_delta_mm = float(post_metric_mm) - float(pre_metric_mm)
    abs_delta_mm = abs(float(raw_delta_mm))
    moved = bool(abs_delta_mm >= float(movement_threshold_mm))
    return {
        "ok": True,
        "hotkey": str(hotkey),
        "cmd": str(cmd),
        "pwm": int(pwm),
        "power": float(telemetry_robot_module.pwm_to_power(int(pwm)) or 0.0),
        "duration_ms": int(send_duration_ms),
        "cmd_sent": (send_result or {}).get("cmd_sent"),
        "display_label": display_label,
        "metric_key": metric_key,
        "metric_label": str(spec.get("metric_label") or "x_axis"),
        "pre_metric_mm": float(pre_metric_mm),
        "post_metric_mm": float(post_metric_mm),
        "pre_dist_mm": float(pre_pose.get("dist_mm")),
        "post_dist_mm": float(post_pose.get("dist_mm")),
        "pre_x_axis_mm": float(pre_pose.get("x_axis_mm")),
        "post_x_axis_mm": float(post_pose.get("x_axis_mm")),
        "raw_delta_mm": float(raw_delta_mm),
        "abs_delta_mm": float(abs_delta_mm),
        "moved": bool(moved),
        "result_label": "movement" if bool(moved) else "no movement",
    }


def _summarize_candidate_rows(rows: list[dict], *, movement_threshold_mm: float) -> dict:
    valid_rows = [dict(row) for row in list(rows or []) if isinstance(row, dict) and bool(row.get("ok"))]
    if not valid_rows:
        return {
            "valid_trial_count": 0,
            "movement_count": 0,
            "median_abs_delta_mm": None,
            "mean_abs_delta_mm": None,
            "median_raw_delta_mm": None,
            "moved": False,
            "result_label": "unusable",
        }
    abs_deltas = [float(row.get("abs_delta_mm") or 0.0) for row in valid_rows]
    raw_deltas = [float(row.get("raw_delta_mm") or 0.0) for row in valid_rows]
    median_abs_delta_mm = float(statistics.median(abs_deltas)) if abs_deltas else None
    mean_abs_delta_mm = float(statistics.mean(abs_deltas)) if abs_deltas else None
    moved = bool(
        median_abs_delta_mm is not None and float(median_abs_delta_mm) >= float(movement_threshold_mm)
    )
    movement_count = sum(1 for row in valid_rows if bool(row.get("moved")))
    return {
        "valid_trial_count": int(len(valid_rows)),
        "movement_count": int(movement_count),
        "median_abs_delta_mm": median_abs_delta_mm,
        "mean_abs_delta_mm": mean_abs_delta_mm,
        "median_raw_delta_mm": float(statistics.median(raw_deltas)) if raw_deltas else None,
        "moved": bool(moved),
        "result_label": "movement" if bool(moved) else "no movement",
    }


def _summarize_candidate_checkpoints(checkpoints: list[dict]) -> dict:
    checkpoint_rows = [dict(item) for item in list(checkpoints or []) if isinstance(item, dict)]
    abs_dist_errors = []
    abs_x_errors = []
    timed_out_count = 0
    for checkpoint in checkpoint_rows:
        try:
            abs_dist_errors.append(abs(float(checkpoint.get("dist_error_mm") or 0.0)))
        except (TypeError, ValueError):
            pass
        try:
            abs_x_errors.append(abs(float(checkpoint.get("x_axis_error_mm") or 0.0)))
        except (TypeError, ValueError):
            pass
        if bool(checkpoint.get("timed_out")):
            timed_out_count += 1
    return {
        "checkpoint_count": int(len(checkpoint_rows)),
        "timed_out_count": int(timed_out_count),
        "max_abs_dist_error_mm": float(max(abs_dist_errors)) if abs_dist_errors else 0.0,
        "max_abs_x_axis_error_mm": float(max(abs_x_errors)) if abs_x_errors else 0.0,
    }


def _dead_direction_abort_info(candidate_result: dict | None, consecutive_dead_candidates: int) -> dict | None:
    if not isinstance(candidate_result, dict):
        return None
    if not bool(candidate_result.get("ok")) or bool(candidate_result.get("moved")):
        return None
    if int(consecutive_dead_candidates) < int(TURN_ABORT_CONSECUTIVE_DEAD_CANDIDATES):
        return None
    drift_summary = dict(candidate_result.get("drift_summary") or {})
    timed_out_count = int(drift_summary.get("timed_out_count") or 0)
    max_abs_x_axis_error_mm = float(drift_summary.get("max_abs_x_axis_error_mm") or 0.0)
    max_abs_dist_error_mm = float(drift_summary.get("max_abs_dist_error_mm") or 0.0)
    if int(timed_out_count) < 1:
        return None
    if (
        float(max_abs_x_axis_error_mm) < float(TURN_ABORT_X_DRIFT_MM)
        and float(max_abs_dist_error_mm) < float(TURN_ABORT_DIST_DRIFT_MM)
    ):
        return None
    return {
        "reason": "dead_direction_abort",
        "consecutive_dead_candidates": int(consecutive_dead_candidates),
        "timed_out_count": int(timed_out_count),
        "max_abs_x_axis_error_mm": float(max_abs_x_axis_error_mm),
        "max_abs_dist_error_mm": float(max_abs_dist_error_mm),
    }


def _build_hotkey_summary(
    *,
    seed_candidate: dict,
    duration_ms: int,
    movement_threshold_mm: float,
    floor_result: dict | None,
    ceiling_result: dict | None,
    error: str | None = None,
    abort_info: dict | None = None,
) -> dict:
    summary = {
        "recommended_score": None,
        "recommended_pwm": None,
        "recommended_power": None,
        "recommended_duration_ms": int(duration_ms),
        "seed_curve_score": 1,
        "seed_curve_pwm": int(seed_candidate.get("curve_pwm") or 0),
        "seed_curve_power": float(seed_candidate.get("curve_power") or 0.0),
        "seed_curve_duration_ms": int(seed_candidate.get("curve_duration_ms") or duration_ms),
        "movement_threshold_mm": float(movement_threshold_mm),
        "scores": [],
        "floor_candidate": _candidate_snapshot(floor_result),
        "ceiling_candidate": _candidate_snapshot(ceiling_result),
    }
    recommended_candidate = (ceiling_result or {}).get("candidate") if isinstance(ceiling_result, dict) else None
    if isinstance(recommended_candidate, dict) and bool((ceiling_result or {}).get("moved")):
        recommended_pwm = int(recommended_candidate.get("pwm") or 0)
        summary["recommended_pwm"] = int(recommended_pwm)
        summary["recommended_power"] = float(recommended_candidate.get("power") or 0.0)
        summary["recommended_duration_ms"] = int(recommended_candidate.get("duration_ms") or duration_ms)
        if int(recommended_pwm) == int(seed_candidate.get("curve_pwm") or 0):
            summary["recommended_score"] = 1
    if error:
        summary["error"] = str(error)
    if isinstance(abort_info, dict):
        summary["abort_info"] = dict(abort_info)
    return summary


def _evaluate_turn_candidate(
    *,
    robot,
    vision,
    world,
    candidate: dict,
    baseline_pose: dict | None,
    recent_turn_acts,
    rows: list[dict],
    recenter_checkpoints: list[dict],
    trial_journal: list[dict],
    consistency_trials: int,
    movement_threshold_mm: float,
    pause_between_pulses_ms: int,
    log_fn=None,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    candidate_rows = []
    candidate_checkpoints = []
    for repeat_idx in range(1, int(consistency_trials) + 1):
        checkpoint = _run_turn_pwm_recenter_checkpoint(
            robot=robot,
            vision=vision,
            world=world,
            baseline_pose=baseline_pose,
            hotkey=str(candidate.get("hotkey") or ""),
            recent_turn_acts=recent_turn_acts,
            checkpoint_label=(
                f"{str(candidate.get('display_label') or '').strip()} "
                f"{str(candidate.get('phase') or 'probe')} "
                f"pwm={int(candidate.get('pwm') or 0)} "
                f"trial {int(repeat_idx)}/{int(consistency_trials)}"
            ),
            log_fn=logger,
        )
        checkpoint["hotkey"] = str(candidate.get("hotkey") or "")
        checkpoint["phase"] = str(candidate.get("phase") or "")
        checkpoint["pwm"] = int(candidate.get("pwm") or 0)
        checkpoint["trial_index"] = int(repeat_idx)
        recenter_checkpoints.append(checkpoint)
        candidate_checkpoints.append(dict(checkpoint))

        row = _run_turn_pwm_trial(
            robot=robot,
            vision=vision,
            world=world,
            hotkey=str(candidate.get("hotkey") or ""),
            pwm=int(candidate.get("pwm") or 0),
            duration_ms=int(candidate.get("duration_ms") or DEFAULT_DURATION_MS),
            movement_threshold_mm=float(movement_threshold_mm),
            recent_turn_acts=recent_turn_acts,
            log_fn=logger,
        )
        row["phase"] = str(candidate.get("phase") or "")
        row["probe_index"] = int(candidate.get("probe_index") or 0)
        row["trial_index"] = int(repeat_idx)
        row["trial_number"] = int(len(trial_journal) + 1)
        rows.append(row)
        trial_journal.append(dict(row))
        candidate_rows.append(dict(row))
        _log_turn_trial(logger, row)
        if int(repeat_idx) < int(consistency_trials) and int(pause_between_pulses_ms) > 0:
            time.sleep(float(pause_between_pulses_ms) / 1000.0)

    summary = _summarize_candidate_rows(
        candidate_rows,
        movement_threshold_mm=float(movement_threshold_mm),
    )
    result = {
        "candidate": dict(candidate),
        "rows": candidate_rows,
        "checkpoints": candidate_checkpoints,
        "ok": bool(summary.get("valid_trial_count")),
        "valid_trial_count": int(summary.get("valid_trial_count") or 0),
        "movement_count": int(summary.get("movement_count") or 0),
        "median_abs_delta_mm": summary.get("median_abs_delta_mm"),
        "mean_abs_delta_mm": summary.get("mean_abs_delta_mm"),
        "median_raw_delta_mm": summary.get("median_raw_delta_mm"),
        "moved": bool(summary.get("moved")),
        "result_label": str(summary.get("result_label") or "no movement"),
    }
    result["drift_summary"] = _summarize_candidate_checkpoints(candidate_checkpoints)
    logger(
        f"[TURN BREAKAWAY TEST] Candidate {str(candidate.get('display_label') or '').strip()} "
        f"{str(candidate.get('phase') or 'probe')} pwm={int(candidate.get('pwm') or 0)} "
        f"-> {int(result['movement_count'])}/{int(consistency_trials)} movement "
        f"(median_abs={float(result['median_abs_delta_mm'] or 0.0):.3f}mm); "
        f"classification: {str(result['result_label'])}."
    )
    return result


def _candidate_snapshot(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return None
    candidate = dict(result.get("candidate") or {})
    return {
        "hotkey": str(candidate.get("hotkey") or ""),
        "display_label": str(candidate.get("display_label") or ""),
        "phase": str(candidate.get("phase") or ""),
        "probe_index": int(candidate.get("probe_index") or 0),
        "pwm": int(candidate.get("pwm") or 0),
        "power": float(candidate.get("power") or 0.0),
        "duration_ms": int(candidate.get("duration_ms") or DEFAULT_DURATION_MS),
        "valid_trial_count": int(result.get("valid_trial_count") or 0),
        "movement_count": int(result.get("movement_count") or 0),
        "median_abs_delta_mm": result.get("median_abs_delta_mm"),
        "median_raw_delta_mm": result.get("median_raw_delta_mm"),
        "result_label": str(result.get("result_label") or "unknown"),
    }


def run_turn_breakaway_test(
    *,
    robot,
    vision,
    world,
    hotkeys: tuple[str, ...] = TURN_HOTKEYS_DEFAULT,
    trial_budget: int | None = None,
    ping_pong_max_trials: int | None = None,
    duration_ms: int = DEFAULT_DURATION_MS,
    movement_threshold_mm: float = DEFAULT_MOVEMENT_THRESHOLD_MM,
    pause_between_pulses_ms: int = DEFAULT_PAUSE_BETWEEN_PULSES_MS,
    log_path: Path = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
    **_ignored,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    started = time.time()
    rows = []
    recenter_checkpoints = []
    trial_journal = []
    phase_baselines = {}
    hotkeys_norm = tuple(str(item).strip().lower() for item in hotkeys if str(item).strip())
    if trial_budget is None:
        trial_budget = ping_pong_max_trials
    trial_budget = _clamp_positive_int(
        trial_budget,
        TURN_TRIAL_BUDGET_PER_HOTKEY,
        minimum=1,
    )
    duration_ms = _clamp_turn_breakaway_duration_ms(duration_ms, DEFAULT_DURATION_MS)
    movement_threshold_mm = max(0.01, float(movement_threshold_mm))
    pause_between_pulses_ms = _clamp_positive_int(
        pause_between_pulses_ms,
        DEFAULT_PAUSE_BETWEEN_PULSES_MS,
        minimum=0,
    )

    logger("[TURN BREAKAWAY TEST] Goal: find the slowest raw turn PWM that still moves the brick on x_axis.")
    logger(
        f"[TURN BREAKAWAY TEST] Seed from the raw turn 1% curve row, push upward quickly to find a movement ceiling, "
        f"then narrow downward toward the floor. Fixed {int(duration_ms)}ms acts, threshold={float(movement_threshold_mm):.2f}mm, "
        f"coarse_step={int(TURN_COARSE_PWM_STEP_INITIAL)} PWM, budget={int(trial_budget)} trial(s) per hotkey."
    )

    baseline_pose = _median_pose_or_none(
        _collect_pose_samples(
            vision,
            world,
            samples=int(RECENTER_OBSERVE_SAMPLES),
            timeout_s=float(RECENTER_OBSERVE_TIMEOUT_S),
        )
    )
    if isinstance(baseline_pose, dict):
        logger(f"[TURN BREAKAWAY TEST] Baseline pose: {_format_pose_text(baseline_pose)}.")
    else:
        logger("[TURN BREAKAWAY TEST] Baseline pose unavailable; each hotkey will attempt its own local baseline.")

    summary_by_hotkey = {}
    candidate_search_paths = {}
    for hotkey in hotkeys_norm:
        spec = _hotkey_spec(hotkey)
        if not isinstance(spec, dict):
            continue
        phase_recent_turn_acts = deque(maxlen=int(RECOVERY_MAX_INVERSE_ACTS))
        phase_baseline = _median_pose_or_none(
            _collect_pose_samples(
                vision,
                world,
                samples=int(RECENTER_OBSERVE_SAMPLES),
                timeout_s=float(RECENTER_OBSERVE_TIMEOUT_S),
            )
        )
        phase_baselines[str(hotkey)] = phase_baseline
        candidate_search_paths[str(hotkey)] = []
        display_label = str(spec.get("display_label") or hotkey.upper())
        if isinstance(phase_baseline, dict):
            logger(f"[TURN BREAKAWAY TEST] Phase baseline for {display_label}: {_format_pose_text(phase_baseline)}.")
        else:
            logger(f"[TURN BREAKAWAY TEST] Phase baseline for {display_label}: unavailable.")

        try:
            seed_candidate = _turn_curve_candidate_for_hotkey(
                str(hotkey),
                duration_ms=int(duration_ms),
            )
        except ValueError as exc:
            logger(f"[TURN BREAKAWAY TEST] {display_label}: skipped ({exc}).")
            summary_by_hotkey[str(hotkey)] = {
                "recommended_score": None,
                "recommended_pwm": None,
                "recommended_power": None,
                "recommended_duration_ms": int(duration_ms),
                "scores": [],
                "error": str(exc),
            }
            continue

        logger(
            f"[TURN BREAKAWAY TEST] Seed for {display_label}: raw curve 1% -> "
            f"pwm={int(seed_candidate['curve_pwm'])}, pwr={float(seed_candidate['curve_power']):.3f}, "
            f"t={int(seed_candidate['duration_ms'])}ms."
        )

        probe_index = 1
        hotkey_trial_count = 0
        floor_result = None
        ceiling_result = None
        dead_candidate_streak = 0
        abort_info = None

        def _remaining_trials() -> int:
            return max(0, int(trial_budget) - int(hotkey_trial_count))

        def _evaluate_budgeted(candidate: dict, *, requested_trials: int) -> dict | None:
            nonlocal hotkey_trial_count
            remaining_trials = _remaining_trials()
            if remaining_trials <= 0:
                logger(
                    f"[TURN BREAKAWAY TEST] {display_label}: trial budget exhausted after "
                    f"{int(hotkey_trial_count)}/{int(trial_budget)} trial(s)."
                )
                return None
            actual_trials = max(1, min(int(requested_trials), int(remaining_trials)))
            result = _evaluate_turn_candidate(
                robot=robot,
                vision=vision,
                world=world,
                candidate=dict(candidate),
                baseline_pose=phase_baseline,
                recent_turn_acts=phase_recent_turn_acts,
                rows=rows,
                recenter_checkpoints=recenter_checkpoints,
                trial_journal=trial_journal,
                consistency_trials=int(actual_trials),
                movement_threshold_mm=float(movement_threshold_mm),
                pause_between_pulses_ms=int(pause_between_pulses_ms),
                log_fn=logger,
            )
            hotkey_trial_count += int(actual_trials)
            candidate_search_paths[str(hotkey)].append(_candidate_snapshot(result))
            return result

        seed_result = _evaluate_budgeted(
            _turn_candidate_from_pwm(
                seed_candidate,
                pwm=int(seed_candidate["curve_pwm"]),
                phase="seed",
                probe_index=int(probe_index),
            ),
            requested_trials=int(TURN_COARSE_SEARCH_TRIALS),
        )
        if seed_result is None:
            summary_by_hotkey[str(hotkey)] = _build_hotkey_summary(
                seed_candidate=seed_candidate,
                duration_ms=int(duration_ms),
                movement_threshold_mm=float(movement_threshold_mm),
                floor_result=None,
                ceiling_result=None,
                error="trial_budget_exhausted",
            )
            continue
        if not bool(seed_result.get("ok")):
            logger(f"[TURN BREAKAWAY TEST] Seed for {display_label}: unusable; stopping this hotkey.")
            summary_by_hotkey[str(hotkey)] = _build_hotkey_summary(
                seed_candidate=seed_candidate,
                duration_ms=int(duration_ms),
                movement_threshold_mm=float(movement_threshold_mm),
                floor_result=seed_result,
                ceiling_result=None,
                error="seed_unusable",
            )
            continue
        dead_candidate_streak = 0 if bool(seed_result.get("moved")) else 1
        abort_info = _dead_direction_abort_info(seed_result, dead_candidate_streak)
        if isinstance(abort_info, dict):
            logger(
                f"[TURN BREAKAWAY TEST] {display_label}: stopping early after {int(abort_info['consecutive_dead_candidates'])} "
                "dead candidate(s) with recenter drift."
            )
            summary_by_hotkey[str(hotkey)] = _build_hotkey_summary(
                seed_candidate=seed_candidate,
                duration_ms=int(duration_ms),
                movement_threshold_mm=float(movement_threshold_mm),
                floor_result=seed_result,
                ceiling_result=None,
                error="dead_direction_abort",
                abort_info=abort_info,
            )
            continue
        max_pwm = int(getattr(robot, "MAX_PWM", 255) or 255)
        if bool(seed_result.get("moved")):
            ceiling_result = seed_result
        else:
            floor_result = seed_result

        coarse_step = int(TURN_COARSE_PWM_STEP_INITIAL)
        if not isinstance(abort_info, dict):
            if ceiling_result is None:
                current_pwm = int((floor_result.get("candidate") or {}).get("pwm") or 0)
                while ceiling_result is None and int(current_pwm) < int(max_pwm):
                    remaining_gap = int(max_pwm) - int(current_pwm)
                    step_size = _next_coarse_pwm_step(int(coarse_step), int(remaining_gap))
                    if step_size <= 0:
                        break
                    next_pwm = min(int(max_pwm), int(current_pwm) + int(step_size))
                    if int(next_pwm) <= int(current_pwm):
                        break
                    probe_index += 1
                    probe_result = _evaluate_budgeted(
                        _turn_candidate_from_pwm(
                            seed_candidate,
                            pwm=int(next_pwm),
                            phase="coarse_faster",
                            probe_index=int(probe_index),
                        ),
                        requested_trials=int(TURN_COARSE_SEARCH_TRIALS),
                    )
                    if probe_result is None:
                        break
                    if not bool(probe_result.get("ok")):
                        logger(
                            f"[TURN BREAKAWAY TEST] {display_label} faster probe pwm={int(next_pwm)} was unusable; stopping coarse search."
                        )
                        break
                    dead_candidate_streak = 0 if bool(probe_result.get("moved")) else int(dead_candidate_streak) + 1
                    abort_info = _dead_direction_abort_info(probe_result, dead_candidate_streak)
                    if isinstance(abort_info, dict):
                        logger(
                            f"[TURN BREAKAWAY TEST] {display_label}: stopping early after {int(abort_info['consecutive_dead_candidates'])} "
                            "dead candidate(s) with recenter drift."
                        )
                        floor_result = probe_result
                        break
                    if bool(probe_result.get("moved")):
                        ceiling_result = probe_result
                        break
                    floor_result = probe_result
                    current_pwm = int(next_pwm)
                    coarse_step = int(step_size)

            if ceiling_result is not None and floor_result is None:
                current_pwm = int((ceiling_result.get("candidate") or {}).get("pwm") or 0)
                while int(current_pwm) > 1:
                    remaining_gap = int(current_pwm) - 1
                    step_size = _next_coarse_pwm_step(int(coarse_step), int(remaining_gap))
                    if step_size <= 0:
                        break
                    next_pwm = max(1, int(current_pwm) - int(step_size))
                    if int(next_pwm) >= int(current_pwm):
                        break
                    probe_index += 1
                    probe_result = _evaluate_budgeted(
                        _turn_candidate_from_pwm(
                            seed_candidate,
                            pwm=int(next_pwm),
                            phase="coarse_slower",
                            probe_index=int(probe_index),
                        ),
                        requested_trials=int(TURN_COARSE_SEARCH_TRIALS),
                    )
                    if probe_result is None:
                        break
                    if not bool(probe_result.get("ok")):
                        logger(
                            f"[TURN BREAKAWAY TEST] {display_label} slower probe pwm={int(next_pwm)} was unusable; stopping coarse search."
                        )
                        break
                    dead_candidate_streak = 0 if bool(probe_result.get("moved")) else int(dead_candidate_streak) + 1
                    abort_info = _dead_direction_abort_info(probe_result, dead_candidate_streak)
                    if isinstance(abort_info, dict):
                        logger(
                            f"[TURN BREAKAWAY TEST] {display_label}: stopping early after {int(abort_info['consecutive_dead_candidates'])} "
                            "dead candidate(s) with recenter drift."
                        )
                        floor_result = probe_result
                        break
                    if bool(probe_result.get("moved")):
                        ceiling_result = probe_result
                        current_pwm = int(next_pwm)
                        coarse_step = int(step_size)
                        if int(next_pwm) <= 1:
                            floor_result = {
                                "candidate": _turn_candidate_from_pwm(
                                    seed_candidate,
                                    pwm=1,
                                    phase="synthetic_floor",
                                    probe_index=int(probe_index + 1),
                                ),
                                "valid_trial_count": 0,
                                "movement_count": 0,
                                "median_abs_delta_mm": None,
                                "median_raw_delta_mm": None,
                                "moved": False,
                                "result_label": "assumed below floor",
                            }
                            break
                        continue
                    floor_result = probe_result
                    break

        if not isinstance(abort_info, dict) and isinstance(floor_result, dict) and isinstance(ceiling_result, dict):
            floor_pwm = int((floor_result.get("candidate") or {}).get("pwm") or 0)
            ceiling_pwm = int((ceiling_result.get("candidate") or {}).get("pwm") or 0)
            logger(
                f"[TURN BREAKAWAY TEST] Initial bracket for {display_label}: "
                f"floor pwm={int(floor_pwm)}, ceiling pwm={int(ceiling_pwm)}."
            )
            while _remaining_trials() > 0:
                floor_pwm = int((floor_result.get("candidate") or {}).get("pwm") or 0)
                ceiling_pwm = int((ceiling_result.get("candidate") or {}).get("pwm") or 0)
                if int(ceiling_pwm) - int(floor_pwm) <= int(TURN_FINAL_BRACKET_TOLERANCE_PWM):
                    break
                midpoint_pwm = int(floor_pwm + ((int(ceiling_pwm) - int(floor_pwm)) // 2))
                if int(midpoint_pwm) in {int(floor_pwm), int(ceiling_pwm)}:
                    break
                probe_index += 1
                probe_result = _evaluate_budgeted(
                    _turn_candidate_from_pwm(
                        seed_candidate,
                        pwm=int(midpoint_pwm),
                        phase="refine_downward",
                        probe_index=int(probe_index),
                    ),
                    requested_trials=int(TURN_REFINE_SEARCH_TRIALS),
                )
                if probe_result is None:
                    break
                if not bool(probe_result.get("ok")):
                    logger(
                        f"[TURN BREAKAWAY TEST] {display_label} downward refinement probe pwm={int(midpoint_pwm)} was unusable; stopping refinement."
                    )
                    break
                dead_candidate_streak = 0 if bool(probe_result.get("moved")) else int(dead_candidate_streak) + 1
                abort_info = _dead_direction_abort_info(probe_result, dead_candidate_streak)
                if isinstance(abort_info, dict):
                    logger(
                        f"[TURN BREAKAWAY TEST] {display_label}: stopping early after {int(abort_info['consecutive_dead_candidates'])} "
                        "dead candidate(s) with recenter drift."
                    )
                    floor_result = probe_result
                    break
                if bool(probe_result.get("moved")):
                    ceiling_result = probe_result
                else:
                    floor_result = probe_result

        recommended_candidate = (ceiling_result or {}).get("candidate") if isinstance(ceiling_result, dict) else None
        recommended_pwm = None
        recommended_power = None
        recommended_duration_ms = int(duration_ms)
        recommended_score = None
        if isinstance(recommended_candidate, dict) and bool((ceiling_result or {}).get("moved")):
            recommended_pwm = int(recommended_candidate.get("pwm") or 0)
            recommended_power = float(recommended_candidate.get("power") or 0.0)
            recommended_duration_ms = int(recommended_candidate.get("duration_ms") or duration_ms)
            if int(recommended_pwm) == int(seed_candidate.get("curve_pwm") or 0):
                recommended_score = 1
            logger(
                f"[TURN BREAKAWAY TEST] Recommendation for {display_label}: "
                f"pwm={int(recommended_pwm)}, pwr={float(recommended_power):.3f}, t={int(recommended_duration_ms)}ms."
            )
        else:
            logger(f"[TURN BREAKAWAY TEST] Recommendation for {display_label}: no movement ceiling found.")

        summary_by_hotkey[str(hotkey)] = _build_hotkey_summary(
            seed_candidate=seed_candidate,
            duration_ms=int(duration_ms),
            movement_threshold_mm=float(movement_threshold_mm),
            floor_result=floor_result,
            ceiling_result=ceiling_result,
            error="dead_direction_abort" if isinstance(abort_info, dict) else None,
            abort_info=abort_info,
        )

    result = {
        "ok": True,
        "seconds": float(max(0.0, time.time() - started)),
        "settings": {
            "hotkeys": list(hotkeys_norm),
            "duration_ms": int(duration_ms),
            "movement_threshold_mm": float(movement_threshold_mm),
            "pause_between_pulses_ms": int(pause_between_pulses_ms),
            "trial_budget_per_hotkey": int(trial_budget),
            "coarse_search_trials": int(TURN_COARSE_SEARCH_TRIALS),
            "refine_search_trials": int(TURN_REFINE_SEARCH_TRIALS),
            "batch_pwm_step": int(TURN_PWM_BATCH_STEP),
            "coarse_pwm_step_initial": int(TURN_COARSE_PWM_STEP_INITIAL),
            "final_bracket_tolerance_pwm": int(TURN_FINAL_BRACKET_TOLERANCE_PWM),
            "seed_mode": "raw_curve_1pct_pwm",
        },
        "baseline_pose": baseline_pose,
        "phase_baselines": phase_baselines,
        "candidate_search_paths": candidate_search_paths,
        "trial_journal": trial_journal,
        "recenter_checkpoints": recenter_checkpoints,
        "rows": rows,
        "summary": {
            "required_success_ratio": None,
            "required_success_count": None,
            "by_hotkey": summary_by_hotkey,
        },
    }
    try:
        Path(log_path).write_text(json.dumps(result, indent=2) + "\n")
    except OSError as exc:
        result["write_error"] = str(exc)
    return result


def run_interactive_turn_breakaway_test(
    *,
    robot,
    vision,
    world,
    prompt_fn=input,
    log_fn=None,
    log_path: Path = RUN_LOG_FILE_DEFAULT,
) -> dict:
    logger = log_fn if callable(log_fn) else print
    logger("[TURN BREAKAWAY TEST] Raw turn PWM floor search for Q/E.")
    logger("[TURN BREAKAWAY TEST] Fixed 130ms acts. Seed from the raw turn 1% curve row, climb quickly to a movement ceiling, then narrow downward to the floor.")
    logger(
        f"[TURN BREAKAWAY TEST] Coarse ceiling probes use {int(TURN_COARSE_SEARCH_TRIALS)} trial each, downward refinement uses {int(TURN_REFINE_SEARCH_TRIALS)} trial(s), and each hotkey is capped at {int(TURN_TRIAL_BUDGET_PER_HOTKEY)} total trials."
    )

    trial_budget = _clamp_positive_int(
        _prompt_with_default(prompt_fn, "  Trial budget per hotkey", str(TURN_TRIAL_BUDGET_PER_HOTKEY)),
        TURN_TRIAL_BUDGET_PER_HOTKEY,
        minimum=1,
    )
    movement_threshold_mm = max(
        0.01,
        float(
            _prompt_with_default(
                prompt_fn,
                "  Movement threshold mm",
                f"{DEFAULT_MOVEMENT_THRESHOLD_MM:.2f}",
            )
        ),
    )
    return run_turn_breakaway_test(
        robot=robot,
        vision=vision,
        world=world,
        hotkeys=TURN_HOTKEYS_DEFAULT,
        trial_budget=int(trial_budget),
        duration_ms=int(DEFAULT_DURATION_MS),
        movement_threshold_mm=float(movement_threshold_mm),
        pause_between_pulses_ms=int(DEFAULT_PAUSE_BETWEEN_PULSES_MS),
        log_path=Path(log_path),
        log_fn=logger,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a raw-turn PWM floor search for Q/E hotkeys.")
    parser.add_argument("--trial-budget", type=int, default=TURN_TRIAL_BUDGET_PER_HOTKEY)
    parser.add_argument("--ping-pong-trials", dest="trial_budget", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--movement-threshold-mm", type=float, default=DEFAULT_MOVEMENT_THRESHOLD_MM)
    parser.add_argument("--log", type=str, default=str(RUN_LOG_FILE_DEFAULT))
    parser.add_argument(
        "--vision",
        type=str,
        choices=("cyan", "yolo", "leia", "aruco"),
        default=str(DEFAULT_VISION_MODE),
        help="Vision mode for the standalone turn breakaway test.",
    )
    args = parser.parse_args()

    robot = Robot()
    vision = _build_vision(str(args.vision))
    world = WorldModel()
    world.step_state = StepState.ALIGN_BRICK
    try:
        result = run_turn_breakaway_test(
            robot=robot,
            vision=vision,
            world=world,
            hotkeys=TURN_HOTKEYS_DEFAULT,
            trial_budget=args.trial_budget,
            duration_ms=int(DEFAULT_DURATION_MS),
            movement_threshold_mm=args.movement_threshold_mm,
            pause_between_pulses_ms=DEFAULT_PAUSE_BETWEEN_PULSES_MS,
            log_path=Path(args.log),
        )
        print(json.dumps(result, indent=2))
        return 0 if bool(result.get("ok")) else 1
    except KeyboardInterrupt:
        print("\n[TURN BREAKAWAY TEST] Interrupted.")
        return 130
    finally:
        try:
            vision.close()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
