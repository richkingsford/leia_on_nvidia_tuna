#!/usr/bin/env python3
"""
Repeatable y-axis stroke probe.

Observe the current y offset, send one fixed mast command, observe the new y
offset, and record how far the stroke moved in command direction. Optionally
send a fixed reposition command between trials so the start state walks through
the vertical travel range without requiring operator input between samples.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
from telemetry_process import (
    _average_smoothed_frames as telemetry_average_smoothed_frames,
    _latest_unique_smoothed_frames as telemetry_latest_unique_smoothed_frames,
    lite_gate_unique_frames,
    send_robot_command,
    update_world_from_vision,
)
from telemetry_robot import StepState, WorldModel, normalize_speed_score, speed_power_pwm_for_cmd


OBSERVE_SLEEP_S = 0.02
OBSERVE_TIMEOUT_S = 1.8
OBSERVE_SAMPLES = 5
POST_ACT_SETTLE_S = 0.10
MIN_LITE_UNIQUE_FRAMES = 3
RESULTS_LOG_DEFAULT = "zdebug_yaxis_consistency_results.json"


@dataclass
class TrialResult:
    trial: int
    duration_group_ms: int | None
    repeat_in_group: int | None
    cmd: str
    score_requested: int
    duration_override_ms: int | None
    cmd_sent: str | None
    pwm: int | None
    power: float | None
    duration_ms: int | None
    pre_y_mm: float
    post_y_mm: float
    pre_dist_mm: float
    post_dist_mm: float
    raw_delta_mm: float
    cmd_delta_mm: float
    dist_delta_mm: float
    pre_pose_source: str | None
    post_pose_source: str | None
    pre_lite_required_frames: int | None
    post_lite_required_frames: int | None
    target_gap_mm: float | None = None
    off_center_mm: float | None = None
    success_margin_mm: float | None = None
    hit_target: bool | None = None
    story_line: str | None = None
    between_cmd_sent: str | None = None
    between_pwm: int | None = None
    between_power: float | None = None
    between_duration_ms: int | None = None
    between_pre_y_mm: float | None = None
    between_post_y_mm: float | None = None
    between_pre_dist_mm: float | None = None
    between_post_dist_mm: float | None = None
    between_raw_delta_mm: float | None = None
    between_cmd_delta_mm: float | None = None
    between_dist_delta_mm: float | None = None
    between_pre_pose_source: str | None = None
    between_post_pose_source: str | None = None
    between_pre_lite_required_frames: int | None = None
    between_post_lite_required_frames: int | None = None
    reset_acts: int | None = None
    reset_cmds: list[str] | None = None
    reset_modes: list[str] | None = None
    reset_initial_y_mm: float | None = None
    reset_final_y_error_mm: float | None = None
    reset_target_y_mm: float | None = None
    setup_acts: int | None = None
    setup_cmds: list[str] | None = None
    setup_modes: list[str] | None = None
    setup_target_y_mm: float | None = None
    setup_target_y_tol_mm: float | None = None
    setup_initial_y_mm: float | None = None
    setup_initial_dist_mm: float | None = None
    setup_final_y_error_mm: float | None = None
    setup_final_dist_error_mm: float | None = None
    setup_approach_cmd: str | None = None


def log_line(message: str) -> None:
    print(str(message), flush=True)


def _coerce_float(value, fallback=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value, fallback=None):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _parse_int_list(raw: str | None) -> list[int]:
    values: list[int] = []
    for token in str(raw or "").split(","):
        text = token.strip()
        if not text:
            continue
        value = _coerce_int(text)
        if value is None or int(value) <= 0:
            raise ValueError(f"Invalid positive integer list value: {text!r}")
        values.append(int(value))
    return values


def _normalize_cmd(value: str, *, allow_none: bool = False, allow_auto: bool = False) -> str | None:
    text = str(value or "").strip().lower()
    if allow_none and text in ("", "none", "skip", "off"):
        return None
    if allow_auto and text in ("auto", "alternate", "center"):
        return "auto"
    if text not in ("u", "d"):
        raise ValueError("Allowed y-axis commands are only 'u' or 'd'.")
    return text


def _inverse_cmd(cmd: str | None) -> str | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "u":
        return "d"
    if cmd_key == "d":
        return "u"
    return None


def _command_delta_mm(cmd: str, pre_y_mm: float, post_y_mm: float) -> float:
    if cmd == "u":
        return float(pre_y_mm) - float(post_y_mm)
    return float(post_y_mm) - float(pre_y_mm)


def _auto_first_cmd_for_y(
    curr_y_mm: float,
    target_y_mm: float = 0.0,
    tol_mm: float = 0.0,
    *,
    center_y_mm: float = 0.0,
) -> str:
    next_cmd = _setup_cmd_for_target_y(float(curr_y_mm), float(target_y_mm), float(tol_mm))
    if next_cmd is not None:
        return str(next_cmd)
    center_cmd = _setup_cmd_for_target_y(float(curr_y_mm), float(center_y_mm), 0.0)
    if center_cmd is not None:
        return str(center_cmd)
    return "d"


def _raw_delta_direction_text(raw_delta_mm: float | None) -> tuple[str, float]:
    value = abs(float(_coerce_float(raw_delta_mm, 0.0) or 0.0))
    raw_val = float(_coerce_float(raw_delta_mm, 0.0) or 0.0)
    if raw_val > 0.0:
        return "lowering", float(value)
    if raw_val < 0.0:
        return "raising", float(value)
    return "holding", 0.0


def _trial_story_line(
    *,
    cmd: str,
    score: int,
    duration_ms: int | None,
    pre_y_mm: float,
    post_y_mm: float,
    raw_delta_mm: float,
    center_y_mm: float = 0.0,
) -> str:
    desired_text = "up" if str(cmd).lower() == "u" else "down"
    target_mm = abs(float(pre_y_mm) - float(center_y_mm))
    off_mm = abs(float(post_y_mm) - float(center_y_mm))
    motion_word, actual_mm = _raw_delta_direction_text(raw_delta_mm)
    duration_text = "model" if duration_ms is None else f"{int(duration_ms)}ms"
    act_text = f"{str(cmd).upper()} {int(score)}% {duration_text}"
    return (
        f'I wanted to go {desired_text} {target_mm:.2f}mm, so I did the "{act_text}" act, '
        f"resulting in the brick {motion_word} {actual_mm:.2f}mm ({off_mm:.2f}mm off)."
    )


def _trial_target_y_for_cmd(cmd: str, *, center_y_mm: float, band_mm: float) -> float:
    if str(cmd).strip().lower() == "u":
        return float(center_y_mm) + abs(float(band_mm))
    return float(center_y_mm) - abs(float(band_mm))


def _setup_cmd_for_target_y(curr_y_mm: float, target_y_mm: float, tol_mm: float) -> str | None:
    if float(curr_y_mm) < (float(target_y_mm) - float(tol_mm)):
        return "d"
    if float(curr_y_mm) > (float(target_y_mm) + float(tol_mm)):
        return "u"
    return None


def _setup_next_cmd_for_target_y(
    curr_y_mm: float,
    target_y_mm: float,
    tol_mm: float,
    *,
    approach_cmd: str | None = None,
    approach_margin_mm: float = 0.0,
    approach_ready: bool = False,
) -> tuple[bool, bool, str | None]:
    lower = float(target_y_mm) - float(tol_mm)
    upper = float(target_y_mm) + float(tol_mm)
    curr = float(curr_y_mm)
    approach = _normalize_cmd(approach_cmd, allow_none=True)
    if approach is None:
        next_cmd = _setup_cmd_for_target_y(curr, float(target_y_mm), float(tol_mm))
        return next_cmd is None, bool(approach_ready), next_cmd

    margin = max(0.0, float(approach_margin_mm))
    if approach == "d":
        preload_cmd = "u"
        preload_target = lower - margin
        if not bool(approach_ready):
            if curr <= preload_target:
                approach_ready = True
            else:
                return False, False, preload_cmd
        if lower <= curr <= upper:
            return True, True, None
        if curr < lower:
            return False, True, "d"
        return False, False, preload_cmd

    preload_cmd = "d"
    preload_target = upper + margin
    if not bool(approach_ready):
        if curr >= preload_target:
            approach_ready = True
        else:
            return False, False, preload_cmd
    if lower <= curr <= upper:
        return True, True, None
    if curr > upper:
        return False, True, "u"
    return False, False, preload_cmd


def _default_setup_fine_duration_ms(
    setup_duration_ms: int | None,
    setup_fine_duration_ms: int | None,
) -> int:
    explicit = _coerce_int(setup_fine_duration_ms)
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    coarse = _coerce_int(setup_duration_ms)
    if coarse is not None and int(coarse) > 0:
        return max(60, int(round(int(coarse) * 0.5)))
    return 150


def _setup_remaining_mm_to_next_boundary(
    curr_y_mm: float,
    target_y_mm: float,
    tol_mm: float,
    *,
    approach_cmd: str | None = None,
    approach_margin_mm: float = 0.0,
    approach_ready: bool = False,
) -> float:
    lower = float(target_y_mm) - float(tol_mm)
    upper = float(target_y_mm) + float(tol_mm)
    curr = float(curr_y_mm)
    approach = _normalize_cmd(approach_cmd, allow_none=True)
    if approach is None:
        if curr < lower:
            return float(lower - curr)
        if curr > upper:
            return float(curr - upper)
        return 0.0

    margin = max(0.0, float(approach_margin_mm))
    ready = bool(approach_ready)
    if approach == "d":
        preload_target = upper + margin
        if not ready and curr >= preload_target:
            ready = True
        if not ready:
            return float(max(0.0, preload_target - curr))
        if curr > upper:
            return float(curr - upper)
        if curr < lower:
            return float((upper + margin) - curr)
        return 0.0

    preload_target = lower - margin
    if not ready and curr <= preload_target:
        ready = True
    if not ready:
        return float(max(0.0, curr - preload_target))
    if curr < lower:
        return float(lower - curr)
    if curr > upper:
        return float(curr - (lower - margin))
    return 0.0


def _select_setup_motion(
    curr_y_mm: float,
    target_y_mm: float,
    tol_mm: float,
    *,
    setup_score: int,
    setup_duration_ms: int | None,
    setup_fine_score: int,
    setup_fine_duration_ms: int | None,
    setup_fine_window_mm: float,
    approach_cmd: str | None = None,
    approach_margin_mm: float = 0.0,
    approach_ready: bool = False,
) -> dict:
    remaining_mm = _setup_remaining_mm_to_next_boundary(
        curr_y_mm,
        target_y_mm,
        tol_mm,
        approach_cmd=approach_cmd,
        approach_margin_mm=approach_margin_mm,
        approach_ready=approach_ready,
    )
    use_fine = float(setup_fine_window_mm) > 0.0 and float(remaining_mm) <= float(setup_fine_window_mm)
    return {
        "mode": "fine" if use_fine else "coarse",
        "score": int(setup_fine_score if use_fine else setup_score),
        "duration_ms": int(setup_fine_duration_ms) if use_fine and setup_fine_duration_ms is not None else setup_duration_ms,
        "remaining_mm": float(remaining_mm),
    }


def _approach_shelf_target_y(target_y_mm: float, tol_mm: float, *, approach_cmd: str | None, approach_margin_mm: float) -> float | None:
    approach = _normalize_cmd(approach_cmd, allow_none=True)
    if approach is None:
        return None
    upper = float(target_y_mm) + float(tol_mm)
    lower = float(target_y_mm) - float(tol_mm)
    margin = max(0.0, float(approach_margin_mm))
    if approach == "d":
        return float(lower - margin)
    return float(upper + margin)


def _pose_effect(cmd: str, pre_pose: dict, post_pose: dict) -> dict:
    pre_y_mm = float(pre_pose["offset_y"])
    post_y_mm = float(post_pose["offset_y"])
    pre_dist_mm = float(pre_pose["dist"])
    post_dist_mm = float(post_pose["dist"])
    return {
        "pre_y_mm": pre_y_mm,
        "post_y_mm": post_y_mm,
        "pre_dist_mm": pre_dist_mm,
        "post_dist_mm": post_dist_mm,
        "raw_delta_mm": float(post_y_mm - pre_y_mm),
        "cmd_delta_mm": _command_delta_mm(cmd, pre_y_mm, post_y_mm),
        "dist_delta_mm": float(post_dist_mm - pre_dist_mm),
    }


def _stats(values: list[float]) -> dict:
    clean = [float(value) for value in values]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "max_abs_deviation_from_median": None,
        }
    median_val = float(statistics.median(clean))
    stdev_val = float(statistics.stdev(clean)) if len(clean) > 1 else 0.0
    max_abs_dev = max(abs(float(value) - median_val) for value in clean)
    return {
        "count": len(clean),
        "mean": float(statistics.mean(clean)),
        "median": median_val,
        "stdev": stdev_val,
        "min": float(min(clean)),
        "max": float(max(clean)),
        "max_abs_deviation_from_median": float(max_abs_dev),
    }


def _rate_display(numerator: int | None, denominator: int | None) -> str | None:
    try:
        num = int(numerator)
        den = int(denominator)
    except (TypeError, ValueError):
        return None
    if den <= 0:
        return None
    pct = (float(num) / float(den)) * 100.0
    return f"{pct:.1f}% ({num}/{den})"


def _consistency_label(median_mm: float | None, stdev_mm: float | None) -> str:
    try:
        median_val = abs(float(median_mm))
        stdev_val = abs(float(stdev_mm))
    except (TypeError, ValueError):
        return "unknown"
    if stdev_val <= max(0.35, median_val * 0.15):
        return "tight"
    if stdev_val <= max(1.0, median_val * 0.35):
        return "semi-tight"
    return "all over the place"


def _consistency_summary_line(
    *,
    trial_count: int,
    cmd: str,
    score: int,
    duration_ms: int | None,
    summary: dict,
) -> str:
    cmd_delta = summary.get("cmd_delta_mm") if isinstance(summary, dict) else {}
    median_delta = _coerce_float(cmd_delta.get("median"))
    stdev_delta = _coerce_float(cmd_delta.get("stdev"))
    label = _consistency_label(median_delta, stdev_delta)
    duration_text = "model" if duration_ms is None else f"{int(duration_ms)}ms"
    median_text = "n/a" if median_delta is None else f"{float(median_delta):.2f}mm"
    stdev_text = "n/a" if stdev_delta is None else f"{float(stdev_delta):.2f}mm"
    return (
        f"{int(trial_count)} trials; speed sent: {str(cmd).upper()} {int(score)}% {duration_text}; "
        f"median distance covered: {median_text}; standard deviation: {stdev_text} ({label})"
    )


def _world_step_label(world) -> str:
    step_state = getattr(world, "step_state", None)
    step_value = getattr(step_state, "value", step_state)
    step_text = str(step_value or "ALIGN_BRICK").strip()
    return step_text or "ALIGN_BRICK"


def _pose_from_measurement(
    measurement: dict,
    *,
    obs_ts: float,
    pose_source: str,
    lite_required_frames: int | None = None,
) -> dict | None:
    if not isinstance(measurement, dict) or not bool(measurement.get("visible")):
        return None
    try:
        return {
            "offset_y": float(measurement.get("offset_y", measurement.get("y_axis", 0.0)) or 0.0),
            "offset_x": float(measurement.get("offset_x", measurement.get("x_axis", 0.0)) or 0.0),
            "dist": float(measurement.get("dist", 0.0) or 0.0),
            "angle": float(measurement.get("angle", 0.0) or 0.0),
            "confidence": float(measurement.get("confidence", 0.0) or 0.0),
            "obs_ts": float(obs_ts),
            "pose_source": str(pose_source),
            "lite_required_frames": int(lite_required_frames) if lite_required_frames is not None else None,
        }
    except (TypeError, ValueError):
        return None


def _lite_pose_from_world(world, *, step: str, samples: int, obs_ts: float) -> dict | None:
    required_frames = max(1, int(lite_gate_unique_frames(step) or 1))
    required_frames = max(required_frames, min(max(1, int(samples)), int(MIN_LITE_UNIQUE_FRAMES)))
    frames = telemetry_latest_unique_smoothed_frames(world, required_frames)
    if len(frames) < int(required_frames):
        return None
    measurement = telemetry_average_smoothed_frames(
        frames,
        step=step,
        process_rules=getattr(world, "process_rules", None),
    )
    return _pose_from_measurement(
        measurement,
        obs_ts=obs_ts,
        pose_source="lite_smoothed",
        lite_required_frames=required_frames,
    )


def _brick_pose_from_world(world, *, obs_ts: float) -> dict | None:
    brick = getattr(world, "brick", None)
    if not isinstance(brick, dict):
        return None
    return _pose_from_measurement(brick, obs_ts=obs_ts, pose_source="brick_state")


def read_pose(
    vision,
    world,
    *,
    samples: int = OBSERVE_SAMPLES,
    timeout_s: float = OBSERVE_TIMEOUT_S,
    min_sample_time: float | None = None,
) -> dict | None:
    poses = []
    start_t = time.time()
    step_label = _world_step_label(world)
    while len(poses) < int(samples) and (time.time() - start_t) < float(timeout_s):
        now = time.time()
        if min_sample_time is not None and now < float(min_sample_time):
            time.sleep(OBSERVE_SLEEP_S)
            continue
        pose = None
        try:
            update_world_from_vision(world, vision, log=False)
            now = time.time()
            if min_sample_time is not None and now < float(min_sample_time):
                time.sleep(OBSERVE_SLEEP_S)
                continue
            pose = _lite_pose_from_world(world, step=step_label, samples=int(samples), obs_ts=now)
            if pose is None:
                pose = _brick_pose_from_world(world, obs_ts=now)
        except Exception:
            pose = None
        if pose is None:
            found, angle, dist, offset_x, conf, cam_h, _above, _below = vision.read()
            world.update_vision(found, dist, angle, conf, offset_x, cam_h)
            if not found:
                time.sleep(OBSERVE_SLEEP_S)
                continue
            pose = {
                "offset_y": float(cam_h),
                "offset_x": float(offset_x),
                "dist": float(dist),
                "angle": float(angle),
                "confidence": float(conf),
                "obs_ts": float(now),
                "pose_source": "raw_visible",
                "lite_required_frames": None,
            }
        poses.append(pose)
        if len(poses) < int(samples):
            time.sleep(OBSERVE_SLEEP_S)
    if not poses:
        return None
    return {
        "offset_y": float(statistics.median([float(item["offset_y"]) for item in poses])),
        "offset_x": float(statistics.median([float(item["offset_x"]) for item in poses])),
        "dist": float(statistics.median([float(item["dist"]) for item in poses])),
        "angle": float(statistics.median([float(item["angle"]) for item in poses])),
        "confidence": float(statistics.median([float(item["confidence"]) for item in poses])),
        "obs_ts": float(max(float(item["obs_ts"]) for item in poses)),
        "pose_source": str(poses[-1].get("pose_source") or "unknown"),
        "lite_required_frames": _coerce_int(poses[-1].get("lite_required_frames")),
        "samples_used": len(poses),
    }


def _action_description(cmd: str, score: int, meta: dict | None) -> str:
    score_used = _coerce_int(meta.get("score_model") if isinstance(meta, dict) else None, fallback=score)
    pwm = _coerce_int(meta.get("pwm") if isinstance(meta, dict) else None)
    power = _coerce_float(meta.get("power") if isinstance(meta, dict) else None)
    duration_ms = _coerce_int(meta.get("duration_ms") if isinstance(meta, dict) else None)
    power_text = "n/a" if power is None else f"{power:.3f}"
    pwm_text = "n/a" if pwm is None else str(pwm)
    duration_text = "n/a" if duration_ms is None else f"{duration_ms}ms"
    return f"{cmd.upper()} {int(score_used)}% (pwm={pwm_text}, pwr={power_text}, t={duration_text})"


def _planned_action_meta(cmd: str, score: int, duration_override_ms: int | None) -> dict:
    power, pwm, score_used, duration_ms = speed_power_pwm_for_cmd(cmd, score)
    if duration_override_ms is not None and int(duration_override_ms) > 0:
        duration_ms = int(duration_override_ms)
    return {
        "power": float(power),
        "pwm": int(pwm),
        "score_model": int(score_used),
        "duration_ms": int(duration_ms),
    }


def _send_fixed_score_command(
    *,
    robot,
    world,
    step: str,
    cmd: str,
    score: int,
    duration_override_ms: int | None,
) -> dict | None:
    return send_robot_command(
        robot,
        world,
        step,
        cmd,
        speed=0.0,
        speed_score=int(score),
        duration_override_ms=duration_override_ms,
    )


def _write_results(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))
    log_line(f"[Y-CONSISTENCY] Wrote log to {path}")


def _within_dist_guard(pose: dict | None, target_dist_mm: float | None, dist_tol_mm: float | None) -> bool:
    if pose is None or target_dist_mm is None or dist_tol_mm is None:
        return True
    try:
        return abs(float(pose["dist"]) - float(target_dist_mm)) <= float(dist_tol_mm)
    except (KeyError, TypeError, ValueError):
        return False


def _build_trial_summary(trials: list[TrialResult]) -> dict:
    cmd_deltas = [item.cmd_delta_mm for item in trials]
    raw_deltas = [item.raw_delta_mm for item in trials]
    pre_vals = [item.pre_y_mm for item in trials]
    post_vals = [item.post_y_mm for item in trials]
    pre_dist_vals = [item.pre_dist_mm for item in trials]
    post_dist_vals = [item.post_dist_mm for item in trials]
    dist_deltas = [item.dist_delta_mm for item in trials]
    between_cmd_deltas = [item.between_cmd_delta_mm for item in trials if item.between_cmd_delta_mm is not None]
    between_pre_vals = [item.between_pre_y_mm for item in trials if item.between_pre_y_mm is not None]
    between_post_vals = [item.between_post_y_mm for item in trials if item.between_post_y_mm is not None]
    between_pre_dist_vals = [item.between_pre_dist_mm for item in trials if item.between_pre_dist_mm is not None]
    between_post_dist_vals = [item.between_post_dist_mm for item in trials if item.between_post_dist_mm is not None]
    between_dist_deltas = [item.between_dist_delta_mm for item in trials if item.between_dist_delta_mm is not None]
    setup_acts = [item.setup_acts for item in trials if item.setup_acts is not None]
    target_gaps = [item.target_gap_mm for item in trials if item.target_gap_mm is not None]
    off_center_vals = [item.off_center_mm for item in trials if item.off_center_mm is not None]
    reset_acts = [item.reset_acts for item in trials if item.reset_acts is not None]
    reset_final_y_errors = [item.reset_final_y_error_mm for item in trials if item.reset_final_y_error_mm is not None]
    hit_targets = [bool(item.hit_target) for item in trials if item.hit_target is not None]
    hit_count = sum(1 for value in hit_targets if bool(value))
    hit_total = len(hit_targets)
    cmd_direction_success_count = sum(1 for value in cmd_deltas if float(value) > 0.0)
    cmd_direction_success_total = len(cmd_deltas)
    return {
        "trial_count": len(trials),
        "cmd_delta_mm": _stats(cmd_deltas),
        "cmd_direction_success_count": int(cmd_direction_success_count),
        "cmd_direction_success_total": int(cmd_direction_success_total),
        "cmd_direction_success_rate": (
            None
            if int(cmd_direction_success_total) <= 0
            else float(cmd_direction_success_count) / float(cmd_direction_success_total)
        ),
        "cmd_direction_success_rate_display": _rate_display(
            int(cmd_direction_success_count),
            int(cmd_direction_success_total),
        ),
        "raw_delta_mm": _stats(raw_deltas),
        "pre_y_mm": _stats(pre_vals),
        "post_y_mm": _stats(post_vals),
        "pre_dist_mm": _stats(pre_dist_vals),
        "post_dist_mm": _stats(post_dist_vals),
        "dist_delta_mm": _stats(dist_deltas),
        "between_cmd_delta_mm": _stats(between_cmd_deltas),
        "between_pre_y_mm": _stats(between_pre_vals),
        "between_post_y_mm": _stats(between_post_vals),
        "between_pre_dist_mm": _stats(between_pre_dist_vals),
        "between_post_dist_mm": _stats(between_post_dist_vals),
        "between_dist_delta_mm": _stats(between_dist_deltas),
        "setup_acts": _stats(setup_acts),
        "target_gap_mm": _stats(target_gaps),
        "off_center_mm": _stats(off_center_vals),
        "hit_count": int(hit_count),
        "hit_total": int(hit_total),
        "hit_rate": None if int(hit_total) <= 0 else float(hit_count) / float(hit_total),
        "hit_rate_display": _rate_display(int(hit_count), int(hit_total)),
        "reset_acts": _stats(reset_acts),
        "reset_final_y_error_mm": _stats(reset_final_y_errors),
    }


def _acquire_target_y_band(
    *,
    vision,
    world,
    robot,
    target_y_mm: float,
    target_y_tol_mm: float,
    setup_score: int,
    setup_duration_ms: int | None,
    setup_fine_score: int,
    setup_fine_duration_ms: int | None,
    setup_fine_window_mm: float,
    setup_max_acts: int,
    setup_approach_cmd: str | None,
    setup_approach_margin_mm: float,
    observe_samples: int,
    observe_timeout_s: float,
    settle_s: float,
    target_dist_mm: float | None,
    dist_tol_mm: float | None,
) -> tuple[dict | None, dict]:
    pose = read_pose(
        vision,
        world,
        samples=observe_samples,
        timeout_s=observe_timeout_s,
    )
    if pose is None:
        return None, {"reason": "setup_pre_pose_unavailable"}
    if not _within_dist_guard(pose, target_dist_mm, dist_tol_mm):
        return None, {
            "reason": "setup_dist_out_of_band_before",
            "initial_pose": pose,
        }
    initial_pose = pose
    cmds: list[str] = []
    modes: list[str] = []
    acts = 0
    approach_ready = False
    while acts <= int(setup_max_acts):
        done, approach_ready, next_cmd = _setup_next_cmd_for_target_y(
            float(pose["offset_y"]),
            float(target_y_mm),
            float(target_y_tol_mm),
            approach_cmd=setup_approach_cmd,
            approach_margin_mm=float(setup_approach_margin_mm),
            approach_ready=bool(approach_ready),
        )
        if bool(done):
            return pose, {
                "reason": "ok",
                "setup_acts": acts,
                "setup_cmds": cmds,
                "setup_modes": modes,
                "initial_pose": initial_pose,
                "final_pose": pose,
                "setup_approach_cmd": setup_approach_cmd,
            }
        if acts >= int(setup_max_acts):
            break
        setup_motion = _select_setup_motion(
            float(pose["offset_y"]),
            float(target_y_mm),
            float(target_y_tol_mm),
            setup_score=int(setup_score),
            setup_duration_ms=setup_duration_ms,
            setup_fine_score=int(setup_fine_score),
            setup_fine_duration_ms=setup_fine_duration_ms,
            setup_fine_window_mm=float(setup_fine_window_mm),
            approach_cmd=setup_approach_cmd,
            approach_margin_mm=float(setup_approach_margin_mm),
            approach_ready=bool(approach_ready),
        )
        act_start_ts = time.time()
        action_meta = _send_fixed_score_command(
            robot=robot,
            world=world,
            step="DEBUG_YAXIS_CONSISTENCY_SETUP",
            cmd=next_cmd,
            score=int(setup_motion["score"]),
            duration_override_ms=_coerce_int(setup_motion.get("duration_ms")),
        )
        if not isinstance(action_meta, dict):
            return None, {
                "reason": "setup_send_failed",
                "setup_acts": acts,
                "setup_cmds": cmds,
                "setup_modes": modes,
                "initial_pose": initial_pose,
                "setup_approach_cmd": setup_approach_cmd,
            }
        duration_ms_used = _coerce_int(action_meta.get("duration_ms"))
        cmds.append(next_cmd)
        modes.append(str(setup_motion.get("mode") or "coarse"))
        acts += 1
        pose = read_pose(
            vision,
            world,
            samples=observe_samples,
            timeout_s=observe_timeout_s,
            min_sample_time=act_start_ts + (float(duration_ms_used or 0) / 1000.0) + float(settle_s),
        )
        if pose is None:
            return None, {
                "reason": "setup_post_pose_unavailable",
                "setup_acts": acts,
                "setup_cmds": cmds,
                "setup_modes": modes,
                "initial_pose": initial_pose,
                "setup_approach_cmd": setup_approach_cmd,
            }
        if not _within_dist_guard(pose, target_dist_mm, dist_tol_mm):
            return None, {
                "reason": "setup_dist_out_of_band_after",
                "setup_acts": acts,
                "setup_cmds": cmds,
                "setup_modes": modes,
                "initial_pose": initial_pose,
                "final_pose": pose,
                "setup_approach_cmd": setup_approach_cmd,
            }
    return None, {
        "reason": "setup_target_unreached",
        "setup_acts": acts,
        "setup_cmds": cmds,
        "setup_modes": modes,
        "initial_pose": initial_pose,
        "final_pose": pose,
        "setup_approach_cmd": setup_approach_cmd,
    }


def _acquire_target_y_plain(
    *,
    vision,
    world,
    robot,
    target_y_mm: float,
    target_y_tol_mm: float,
    setup_score: int,
    setup_duration_ms: int | None,
    setup_fine_score: int,
    setup_fine_duration_ms: int | None,
    setup_fine_window_mm: float,
    setup_max_acts: int,
    observe_samples: int,
    observe_timeout_s: float,
    settle_s: float,
    target_dist_mm: float | None,
    dist_tol_mm: float | None,
) -> tuple[dict | None, dict]:
    pose = read_pose(
        vision,
        world,
        samples=observe_samples,
        timeout_s=observe_timeout_s,
    )
    if pose is None:
        return None, {"reason": "reset_pre_pose_unavailable"}
    if not _within_dist_guard(pose, target_dist_mm, dist_tol_mm):
        return None, {
            "reason": "reset_dist_out_of_band_before",
            "initial_pose": pose,
        }
    initial_pose = pose
    cmds: list[str] = []
    modes: list[str] = []
    acts = 0
    while acts <= int(setup_max_acts):
        next_cmd = _setup_cmd_for_target_y(
            float(pose["offset_y"]),
            float(target_y_mm),
            float(target_y_tol_mm),
        )
        if next_cmd is None:
            return pose, {
                "reason": "ok",
                "reset_acts": acts,
                "reset_cmds": cmds,
                "reset_modes": modes,
                "initial_pose": initial_pose,
                "final_pose": pose,
            }
        if acts >= int(setup_max_acts):
            break
        setup_motion = _select_setup_motion(
            float(pose["offset_y"]),
            float(target_y_mm),
            float(target_y_tol_mm),
            setup_score=int(setup_score),
            setup_duration_ms=setup_duration_ms,
            setup_fine_score=int(setup_fine_score),
            setup_fine_duration_ms=setup_fine_duration_ms,
            setup_fine_window_mm=float(setup_fine_window_mm),
            approach_cmd=None,
            approach_margin_mm=0.0,
            approach_ready=False,
        )
        act_start_ts = time.time()
        action_meta = _send_fixed_score_command(
            robot=robot,
            world=world,
            step="DEBUG_YAXIS_CONSISTENCY_RESET",
            cmd=next_cmd,
            score=int(setup_motion["score"]),
            duration_override_ms=_coerce_int(setup_motion.get("duration_ms")),
        )
        if not isinstance(action_meta, dict):
            return None, {
                "reason": "reset_send_failed",
                "reset_acts": acts,
                "reset_cmds": cmds,
                "reset_modes": modes,
                "initial_pose": initial_pose,
            }
        duration_ms_used = _coerce_int(action_meta.get("duration_ms"))
        cmds.append(next_cmd)
        modes.append(str(setup_motion.get("mode") or "coarse"))
        acts += 1
        pose = read_pose(
            vision,
            world,
            samples=observe_samples,
            timeout_s=observe_timeout_s,
            min_sample_time=act_start_ts + (float(duration_ms_used or 0) / 1000.0) + float(settle_s),
        )
        if pose is None:
            return None, {
                "reason": "reset_post_pose_unavailable",
                "reset_acts": acts,
                "reset_cmds": cmds,
                "reset_modes": modes,
                "initial_pose": initial_pose,
            }
        if not _within_dist_guard(pose, target_dist_mm, dist_tol_mm):
            return None, {
                "reason": "reset_dist_out_of_band_after",
                "reset_acts": acts,
                "reset_cmds": cmds,
                "reset_modes": modes,
                "initial_pose": initial_pose,
                "final_pose": pose,
            }
    return None, {
        "reason": "reset_target_unreached",
        "reset_acts": acts,
        "reset_cmds": cmds,
        "reset_modes": modes,
        "initial_pose": initial_pose,
        "final_pose": pose,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe y-axis repeatability for a fixed mast command.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of observe-act-observe trials (default: 5).")
    parser.add_argument("--cmd", type=str, default="auto", help="Test mast command: u, d, or auto (default: auto).")
    parser.add_argument("--alternate-cmds", type=int, default=1, help="Alternate u/d after each trial (default: 1).")
    parser.add_argument("--center-y-mm", type=float, default=0.0, help="Centerline used for auto first-command selection (default: 0.0).")
    parser.add_argument("--speed-score", type=int, default=1, help="Fixed mast speed score for the test stroke (default: 1).")
    parser.add_argument("--duration-ms", type=int, default=None, help="Override duration in ms for the test stroke.")
    parser.add_argument("--duration-ms-list", type=str, default=None, help="Comma-separated duration sweep in ms. If provided, each duration gets its own repeat block.")
    parser.add_argument("--between-cmd", type=str, default="none", help="Reposition command between trials: u, d, or none (default: none).")
    parser.add_argument("--between-score", type=int, default=1, help="Speed score for between-trial repositioning (default: 1).")
    parser.add_argument("--between-duration-ms", type=int, default=None, help="Override duration in ms for the between-trial reposition act.")
    parser.add_argument("--between-settle-s", type=float, default=0.12, help="Extra wait after between-trial repositioning (default: 0.12).")
    parser.add_argument("--target-y-mm", type=float, default=None, help="Pre-act y band center. If set, the script nudges with small mast acts until this band is reached.")
    parser.add_argument("--target-y-tol-mm", type=float, default=1.0, help="Tolerance for --target-y-mm (default: 1.0).")
    parser.add_argument("--alternate-target-band-mm", type=float, default=None, help="If set, each measured trial starts from a cmd-specific band around center: +band for U, -band for D.")
    parser.add_argument("--alternate-target-band-tol-mm", type=float, default=0.75, help="Tolerance for --alternate-target-band-mm (default: 0.75).")
    parser.add_argument("--success-margin-mm", type=float, default=None, help="Count a one-shot hit when the post-act y is within this absolute distance from center.")
    parser.add_argument("--setup-score", type=int, default=1, help="Speed score for setup nudges into the target y band (default: 1).")
    parser.add_argument("--setup-duration-ms", type=int, default=None, help="Override duration in ms for setup nudges.")
    parser.add_argument("--setup-fine-score", type=int, default=None, help="Optional finer speed score for near-band setup nudges. Defaults to --setup-score.")
    parser.add_argument("--setup-fine-duration-ms", type=int, default=None, help="Optional finer duration override for near-band setup nudges. Defaults to half of --setup-duration-ms, or 150 ms.")
    parser.add_argument("--setup-fine-window-mm", type=float, default=1.5, help="Switch setup to fine nudges when the next setup boundary is this close in mm (default: 1.5).")
    parser.add_argument("--setup-max-acts", type=int, default=12, help="Maximum setup nudges allowed before each test stroke (default: 12).")
    parser.add_argument("--setup-approach-cmd", type=str, default=None, help="Force the final setup approach direction into the target band: u, d, or none.")
    parser.add_argument("--setup-approach-margin-mm", type=float, default=0.5, help="Extra margin beyond the band before final approach begins (default: 0.5).")
    parser.add_argument("--between-reset-to-approach-shelf", action="store_true", help="After each measured trial, reset to the forced-approach preload shelf instead of using a generic between command.")
    parser.add_argument("--between-reset-tol-mm", type=float, default=0.5, help="Tolerance around the approach shelf target when --between-reset-to-approach-shelf is enabled (default: 0.5).")
    parser.add_argument("--between-reset-max-acts", type=int, default=None, help="Maximum acts for the approach-shelf reset. Defaults to --setup-max-acts.")
    parser.add_argument("--target-dist-mm", type=float, default=None, help="Distance band center in mm. If omitted, use --lock-initial-dist to keep the first observed distance.")
    parser.add_argument("--dist-tol-mm", type=float, default=None, help="Allowed absolute distance drift from target distance in mm.")
    parser.add_argument("--lock-initial-dist", action="store_true", help="Lock the first observed distance as the reference distance band.")
    parser.add_argument("--observe-samples", type=int, default=OBSERVE_SAMPLES, help=f"Observation samples per pose (default: {OBSERVE_SAMPLES}).")
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S, help=f"Observation timeout in seconds (default: {OBSERVE_TIMEOUT_S}).")
    parser.add_argument("--post-act-settle-s", type=float, default=POST_ACT_SETTLE_S, help=f"Extra wait after the test stroke before observation (default: {POST_ACT_SETTLE_S}).")
    parser.add_argument("--results-file", type=str, default=RESULTS_LOG_DEFAULT, help=f"JSON output path (default: {RESULTS_LOG_DEFAULT}).")
    args = parser.parse_args()

    try:
        test_cmd = _normalize_cmd(args.cmd, allow_auto=True)
        between_cmd = _normalize_cmd(args.between_cmd, allow_none=True)
    except ValueError as exc:
        log_line(f"[Y-CONSISTENCY] {exc}")
        return 2

    repeats = max(1, int(args.repeats))
    alternate_cmds = bool(int(args.alternate_cmds))
    score = normalize_speed_score(args.speed_score)
    between_score = normalize_speed_score(args.between_score)
    try:
        duration_sweep_ms = _parse_int_list(args.duration_ms_list)
    except ValueError as exc:
        log_line(f"[Y-CONSISTENCY] {exc}")
        return 2
    duration_override_ms = _coerce_int(args.duration_ms)
    if duration_sweep_ms:
        duration_schedule_ms = duration_sweep_ms
    else:
        duration_schedule_ms = [duration_override_ms]
    between_duration_ms = _coerce_int(args.between_duration_ms)
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    post_act_settle_s = max(0.0, float(args.post_act_settle_s))
    between_settle_s = max(0.0, float(args.between_settle_s))
    target_y_mm = _coerce_float(args.target_y_mm)
    target_y_tol_mm = max(0.1, float(args.target_y_tol_mm))
    alternate_target_band_mm = _coerce_float(args.alternate_target_band_mm)
    if alternate_target_band_mm is not None and float(alternate_target_band_mm) <= 0.0:
        log_line("[Y-CONSISTENCY] --alternate-target-band-mm must be > 0.")
        return 2
    alternate_target_band_tol_mm = max(0.1, float(args.alternate_target_band_tol_mm))
    success_margin_mm = _coerce_float(args.success_margin_mm)
    if success_margin_mm is not None and float(success_margin_mm) <= 0.0:
        log_line("[Y-CONSISTENCY] --success-margin-mm must be > 0.")
        return 2
    center_y_mm = float(args.center_y_mm)
    setup_score = normalize_speed_score(args.setup_score)
    setup_duration_ms = _coerce_int(args.setup_duration_ms)
    setup_fine_score = normalize_speed_score(args.setup_fine_score) if args.setup_fine_score is not None else int(setup_score)
    setup_fine_duration_ms = _default_setup_fine_duration_ms(setup_duration_ms, args.setup_fine_duration_ms)
    setup_fine_window_mm = max(0.0, float(args.setup_fine_window_mm))
    setup_max_acts = max(0, int(args.setup_max_acts))
    try:
        setup_approach_cmd = _normalize_cmd(args.setup_approach_cmd, allow_none=True)
    except ValueError as exc:
        log_line(f"[Y-CONSISTENCY] {exc}")
        return 2
    setup_approach_margin_mm = max(0.0, float(args.setup_approach_margin_mm))
    between_reset_to_approach_shelf = bool(args.between_reset_to_approach_shelf)
    between_reset_tol_mm = max(0.1, float(args.between_reset_tol_mm))
    between_reset_max_acts = max(0, int(args.between_reset_max_acts if args.between_reset_max_acts is not None else setup_max_acts))
    target_dist_mm = _coerce_float(args.target_dist_mm)
    dist_tol_mm = _coerce_float(args.dist_tol_mm)
    results_path = Path(args.results_file)
    approach_shelf_target_y_mm = (
        _approach_shelf_target_y(
            float(target_y_mm),
            float(target_y_tol_mm),
            approach_cmd=setup_approach_cmd,
            approach_margin_mm=float(setup_approach_margin_mm),
        )
        if target_y_mm is not None
        else None
    )

    planned_cmd = "u" if test_cmd == "auto" else str(test_cmd)
    planned_test_actions = [
        _planned_action_meta(planned_cmd, score, item_duration_ms)
        for item_duration_ms in duration_schedule_ms
    ]
    planned_test = planned_test_actions[0] if planned_test_actions else None
    planned_between = None
    if between_cmd is not None:
        planned_between = _planned_action_meta(between_cmd, between_score, between_duration_ms)

    log_line("[Y-CONSISTENCY] Starting fixed-command y-axis repeatability probe.")
    log_line(
        f"[Y-CONSISTENCY] repeats_per_duration={repeats} duration_groups={len(duration_schedule_ms)} "
        f"observe_samples={observe_samples} timeout={observe_timeout_s:.2f}s"
    )
    log_line(
        "[Y-CONSISTENCY] test="
        f"{('AUTO alternating' if bool(alternate_cmds) else 'AUTO') if test_cmd == 'auto' else _action_description(test_cmd, score, planned_test)}"
    )
    if bool(alternate_cmds):
        log_line(f"[Y-CONSISTENCY] command_schedule=alternate first_cmd_from_y(center={center_y_mm:+.2f}mm)")
    if len(duration_schedule_ms) > 1:
        log_line(f"[Y-CONSISTENCY] duration_sweep_ms={duration_schedule_ms}")
    if planned_between is not None:
        log_line(
            "[Y-CONSISTENCY] between="
            f"{_action_description(between_cmd, between_score, planned_between)}"
        )
    else:
        log_line("[Y-CONSISTENCY] between=disabled")
    if between_reset_to_approach_shelf:
        if target_y_mm is None or setup_approach_cmd is None or approach_shelf_target_y_mm is None:
            log_line("[Y-CONSISTENCY] between-reset-to-approach-shelf requires --target-y-mm and --setup-approach-cmd.")
            return 2
        log_line(
            f"[Y-CONSISTENCY] between_reset=approach shelf at {approach_shelf_target_y_mm:.2f} +/- {between_reset_tol_mm:.2f} mm "
            f"(max {between_reset_max_acts} acts)"
        )
    if target_y_mm is not None:
        log_line(
            f"[Y-CONSISTENCY] target_y={target_y_mm:.2f} +/- {target_y_tol_mm:.2f} mm "
            f"(setup coarse {setup_score}%/{setup_duration_ms or 'model'}ms, "
            f"fine {setup_fine_score}%/{setup_fine_duration_ms}ms inside {setup_fine_window_mm:.2f} mm, "
            f"max {setup_max_acts} acts)"
        )
        if setup_approach_cmd is not None:
            log_line(
                f"[Y-CONSISTENCY] setup_final_approach={setup_approach_cmd.upper()} "
                f"(preload margin {setup_approach_margin_mm:.2f} mm)"
            )
    if alternate_target_band_mm is not None:
        log_line(
            f"[Y-CONSISTENCY] alternating_target_band=center +/- {alternate_target_band_mm:.2f} mm "
            f"(per-side tol +/- {alternate_target_band_tol_mm:.2f} mm)"
        )
    if success_margin_mm is not None:
        log_line(f"[Y-CONSISTENCY] success_margin=+/- {success_margin_mm:.2f} mm from center")
    if target_dist_mm is not None:
        log_line(f"[Y-CONSISTENCY] target_dist={target_dist_mm:.2f} +/- {float(dist_tol_mm or 0.0):.2f} mm")
    elif bool(args.lock_initial_dist):
        log_line(
            f"[Y-CONSISTENCY] target_dist=lock first observed distance "
            f"(tol +/- {float(dist_tol_mm or 0.0):.2f} mm)"
        )

    robot = None
    vision = None
    world = None
    started_at = time.time()
    trials: list[TrialResult] = []
    status = "completed"
    abort_reason = None
    locked_initial_dist_mm = None
    last_test_cmd: str | None = None
    try:
        world = WorldModel()
        world.step_state = StepState.ALIGN_BRICK
        world._post_action_observe_delay_s = 0.0
        vision = ArucoBrickVision(debug=False)
        robot = Robot()

        trial_idx = 0
        total_trials = int(repeats) * len(duration_schedule_ms)
        for duration_group_idx, group_duration_ms in enumerate(duration_schedule_ms, start=1):
            log_line(
                f"[Y-CONSISTENCY] Duration group {duration_group_idx}/{len(duration_schedule_ms)}: "
                f"{int(group_duration_ms) if group_duration_ms is not None else 'model'}ms"
            )
            for repeat_idx in range(1, repeats + 1):
                trial_idx += 1
                preview_pose = None
                pre_pose = None
                setup_meta = {
                    "setup_acts": 0,
                    "setup_cmds": [],
                    "initial_pose": None,
                    "final_pose": None,
                }
                effective_target_dist_mm = target_dist_mm if target_dist_mm is not None else locked_initial_dist_mm
                preview_needed = (
                    (bool(alternate_cmds) and last_test_cmd is None)
                    or (not bool(alternate_cmds) and test_cmd == "auto")
                )
                if preview_needed:
                    preview_pose = read_pose(
                        vision,
                        world,
                        samples=observe_samples,
                        timeout_s=observe_timeout_s,
                    )
                    if preview_pose is None:
                        status = "aborted"
                        abort_reason = f"preview_pose_unavailable_trial_{trial_idx}"
                        log_line(f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: no visible pose before planning. Aborting.")
                        break

                if bool(alternate_cmds):
                    if last_test_cmd is None:
                        cmd_for_trial = _auto_first_cmd_for_y(
                            float((preview_pose or {}).get("offset_y")),
                            float(target_y_mm if target_y_mm is not None else center_y_mm),
                            float(target_y_tol_mm if target_y_mm is not None else 0.0),
                            center_y_mm=float(center_y_mm),
                        )
                    else:
                        cmd_for_trial = str(_inverse_cmd(last_test_cmd) or last_test_cmd)
                elif test_cmd == "auto":
                    cmd_for_trial = _auto_first_cmd_for_y(
                        float((preview_pose or {}).get("offset_y")),
                        float(target_y_mm if target_y_mm is not None else center_y_mm),
                        float(target_y_tol_mm if target_y_mm is not None else 0.0),
                        center_y_mm=float(center_y_mm),
                    )
                else:
                    cmd_for_trial = str(test_cmd)

                effective_setup_target_y_mm = target_y_mm
                effective_setup_target_y_tol_mm = target_y_tol_mm if target_y_mm is not None else None
                if alternate_target_band_mm is not None:
                    effective_setup_target_y_mm = _trial_target_y_for_cmd(
                        cmd_for_trial,
                        center_y_mm=float(center_y_mm),
                        band_mm=float(alternate_target_band_mm),
                    )
                    effective_setup_target_y_tol_mm = float(alternate_target_band_tol_mm)

                if effective_setup_target_y_mm is not None and effective_setup_target_y_tol_mm is not None:
                    pre_pose, setup_meta = _acquire_target_y_band(
                        vision=vision,
                        world=world,
                        robot=robot,
                        target_y_mm=float(effective_setup_target_y_mm),
                        target_y_tol_mm=float(effective_setup_target_y_tol_mm),
                        setup_score=int(setup_score),
                        setup_duration_ms=setup_duration_ms,
                        setup_fine_score=int(setup_fine_score),
                        setup_fine_duration_ms=setup_fine_duration_ms,
                        setup_fine_window_mm=float(setup_fine_window_mm),
                        setup_max_acts=int(setup_max_acts),
                        setup_approach_cmd=setup_approach_cmd,
                        setup_approach_margin_mm=float(setup_approach_margin_mm),
                        observe_samples=observe_samples,
                        observe_timeout_s=observe_timeout_s,
                        settle_s=post_act_settle_s,
                        target_dist_mm=effective_target_dist_mm,
                        dist_tol_mm=dist_tol_mm,
                    )
                    if pre_pose is None:
                        status = "aborted"
                        abort_reason = f"{setup_meta.get('reason')}_trial_{trial_idx}"
                        log_line(
                            f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: setup failed "
                            f"({setup_meta.get('reason')}). Aborting."
                        )
                        break
                else:
                    pre_pose = preview_pose
                    if pre_pose is None:
                        pre_pose = read_pose(
                            vision,
                            world,
                            samples=observe_samples,
                            timeout_s=observe_timeout_s,
                        )
                    if pre_pose is None:
                        status = "aborted"
                        abort_reason = f"pre_pose_unavailable_trial_{trial_idx}"
                        log_line(f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: no visible pose before act. Aborting.")
                        break
                if bool(args.lock_initial_dist) and locked_initial_dist_mm is None:
                    locked_initial_dist_mm = float(pre_pose["dist"])
                    effective_target_dist_mm = float(locked_initial_dist_mm)
                    log_line(f"[Y-CONSISTENCY] Locked initial distance at {locked_initial_dist_mm:.2f} mm.")
                if not _within_dist_guard(pre_pose, effective_target_dist_mm, dist_tol_mm):
                    status = "aborted"
                    abort_reason = f"pre_pose_dist_out_of_band_trial_{trial_idx}"
                    log_line(
                        f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: pre-act distance "
                        f"{float(pre_pose['dist']):.2f} mm outside band. Aborting."
                    )
                    break

                planned_group_action = _planned_action_meta(cmd_for_trial, score, group_duration_ms)

                act_start_ts = time.time()
                action_meta = _send_fixed_score_command(
                    robot=robot,
                    world=world,
                    step="DEBUG_YAXIS_CONSISTENCY",
                    cmd=cmd_for_trial,
                    score=score,
                    duration_override_ms=group_duration_ms,
                )
                if not isinstance(action_meta, dict):
                    status = "aborted"
                    abort_reason = f"send_failed_trial_{trial_idx}"
                    log_line(f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: send failed. Aborting.")
                    break
                duration_ms_used = _coerce_int(action_meta.get("duration_ms"), planned_group_action["duration_ms"])
                post_pose = read_pose(
                    vision,
                    world,
                    samples=observe_samples,
                    timeout_s=observe_timeout_s,
                    min_sample_time=act_start_ts + (float(duration_ms_used) / 1000.0) + post_act_settle_s,
                )
                if post_pose is None:
                    status = "aborted"
                    abort_reason = f"post_pose_unavailable_trial_{trial_idx}"
                    log_line(f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: no visible pose after act. Aborting.")
                    break
                if not _within_dist_guard(post_pose, effective_target_dist_mm, dist_tol_mm):
                    status = "aborted"
                    abort_reason = f"post_pose_dist_out_of_band_trial_{trial_idx}"
                    log_line(
                        f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: post-act distance "
                        f"{float(post_pose['dist']):.2f} mm outside band. Aborting."
                    )
                    break

                effect = _pose_effect(cmd_for_trial, pre_pose, post_pose)
                pre_y_mm = float(effect["pre_y_mm"])
                post_y_mm = float(effect["post_y_mm"])
                pre_dist_mm = float(effect["pre_dist_mm"])
                post_dist_mm = float(effect["post_dist_mm"])
                raw_delta_mm = float(effect["raw_delta_mm"])
                cmd_delta_mm = float(effect["cmd_delta_mm"])
                dist_delta_mm = float(effect["dist_delta_mm"])
                target_gap_mm = abs(float(pre_y_mm) - float(center_y_mm))
                off_center_mm = abs(float(post_y_mm) - float(center_y_mm))
                hit_target = None if success_margin_mm is None else bool(off_center_mm <= float(success_margin_mm))
                story_line = _trial_story_line(
                    cmd=cmd_for_trial,
                    score=int(score),
                    duration_ms=duration_ms_used,
                    pre_y_mm=pre_y_mm,
                    post_y_mm=post_y_mm,
                    raw_delta_mm=raw_delta_mm,
                    center_y_mm=float(center_y_mm),
                )

                initial_setup_pose = setup_meta.get("initial_pose") if isinstance(setup_meta, dict) else None
                final_setup_pose = setup_meta.get("final_pose") if isinstance(setup_meta, dict) else None
                trial = TrialResult(
                    trial=trial_idx,
                    duration_group_ms=group_duration_ms,
                    repeat_in_group=repeat_idx,
                    cmd=cmd_for_trial,
                    score_requested=int(score),
                    duration_override_ms=group_duration_ms,
                    cmd_sent=str(action_meta.get("cmd_sent") or cmd_for_trial),
                    pwm=_coerce_int(action_meta.get("pwm")),
                    power=_coerce_float(action_meta.get("power")),
                    duration_ms=duration_ms_used,
                    pre_y_mm=pre_y_mm,
                    post_y_mm=post_y_mm,
                    pre_dist_mm=pre_dist_mm,
                    post_dist_mm=post_dist_mm,
                    raw_delta_mm=raw_delta_mm,
                    cmd_delta_mm=cmd_delta_mm,
                    dist_delta_mm=dist_delta_mm,
                    pre_pose_source=str(pre_pose.get("pose_source") or "unknown"),
                    post_pose_source=str(post_pose.get("pose_source") or "unknown"),
                    pre_lite_required_frames=_coerce_int(pre_pose.get("lite_required_frames")),
                    post_lite_required_frames=_coerce_int(post_pose.get("lite_required_frames")),
                    target_gap_mm=target_gap_mm,
                    off_center_mm=off_center_mm,
                    success_margin_mm=success_margin_mm,
                    hit_target=hit_target,
                    story_line=story_line,
                    setup_acts=_coerce_int(setup_meta.get("setup_acts"), 0),
                    setup_cmds=list(setup_meta.get("setup_cmds") or []),
                    setup_modes=list(setup_meta.get("setup_modes") or []),
                    setup_target_y_mm=effective_setup_target_y_mm,
                    setup_target_y_tol_mm=effective_setup_target_y_tol_mm,
                    setup_initial_y_mm=_coerce_float(initial_setup_pose.get("offset_y")) if isinstance(initial_setup_pose, dict) else None,
                    setup_initial_dist_mm=_coerce_float(initial_setup_pose.get("dist")) if isinstance(initial_setup_pose, dict) else None,
                    setup_approach_cmd=str(setup_meta.get("setup_approach_cmd") or setup_approach_cmd or "") or None,
                    setup_final_y_error_mm=(
                        _coerce_float(final_setup_pose.get("offset_y")) - float(effective_setup_target_y_mm)
                        if isinstance(final_setup_pose, dict) and effective_setup_target_y_mm is not None
                        else None
                    ),
                    setup_final_dist_error_mm=(
                        _coerce_float(final_setup_pose.get("dist")) - float(effective_target_dist_mm)
                        if isinstance(final_setup_pose, dict) and effective_target_dist_mm is not None
                        else None
                    ),
                )

                if effective_setup_target_y_mm is not None:
                    log_line(
                        "[Y-CONSISTENCY] "
                        f"Trial {trial_idx}/{total_trials} setup: "
                        f"start_y={float(trial.setup_initial_y_mm or pre_y_mm):+.2f}mm "
                        f"-> pre_act_y={pre_y_mm:+.2f}mm using {int(trial.setup_acts or 0)} act(s) "
                        f"{trial.setup_cmds or []} modes={trial.setup_modes or []}"
                    )

                log_line(
                    "[Y-CONSISTENCY] "
                    f"Trial {trial_idx}/{total_trials}: start_y={pre_y_mm:+.2f}mm @ {pre_dist_mm:.2f}mm "
                    f"-> end_y={post_y_mm:+.2f}mm @ {post_dist_mm:.2f}mm "
                    f"(raw_delta={raw_delta_mm:+.2f}mm, cmd_delta={cmd_delta_mm:+.2f}mm, dist_delta={dist_delta_mm:+.2f}mm) "
                    f"via {_action_description(cmd_for_trial, score, action_meta)} "
                    f"src={trial.pre_pose_source}->{trial.post_pose_source}"
                )
                log_line(f"[Y-CONSISTENCY] {story_line}")
                last_test_cmd = str(cmd_for_trial)

                if between_cmd is not None and repeat_idx < repeats:
                    between_start_ts = time.time()
                    between_meta = _send_fixed_score_command(
                        robot=robot,
                        world=world,
                        step="DEBUG_YAXIS_CONSISTENCY",
                        cmd=between_cmd,
                        score=between_score,
                        duration_override_ms=between_duration_ms,
                    )
                    if not isinstance(between_meta, dict):
                        status = "aborted"
                        abort_reason = f"between_send_failed_trial_{trial_idx}"
                        log_line(f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: between-step send failed. Aborting.")
                        break
                    between_duration_used = _coerce_int(
                        between_meta.get("duration_ms"),
                        planned_between["duration_ms"] if planned_between is not None else None,
                    )
                    trial.between_cmd_sent = str(between_meta.get("cmd_sent") or between_cmd)
                    trial.between_pwm = _coerce_int(between_meta.get("pwm"))
                    trial.between_power = _coerce_float(between_meta.get("power"))
                    trial.between_duration_ms = between_duration_used
                    between_post_pose = read_pose(
                        vision,
                        world,
                        samples=observe_samples,
                        timeout_s=observe_timeout_s,
                        min_sample_time=between_start_ts + (float(between_duration_used or 0) / 1000.0) + between_settle_s,
                    )
                    if between_post_pose is None:
                        log_line(
                            f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: warning, no observed pose after between-step "
                            f"{_action_description(between_cmd, between_score, between_meta)}."
                        )
                    else:
                        between_effect = _pose_effect(between_cmd, post_pose, between_post_pose)
                        trial.between_pre_y_mm = float(between_effect["pre_y_mm"])
                        trial.between_post_y_mm = float(between_effect["post_y_mm"])
                        trial.between_pre_dist_mm = float(between_effect["pre_dist_mm"])
                        trial.between_post_dist_mm = float(between_effect["post_dist_mm"])
                        trial.between_raw_delta_mm = float(between_effect["raw_delta_mm"])
                        trial.between_cmd_delta_mm = float(between_effect["cmd_delta_mm"])
                        trial.between_dist_delta_mm = float(between_effect["dist_delta_mm"])
                        trial.between_pre_pose_source = str(post_pose.get("pose_source") or "unknown")
                        trial.between_post_pose_source = str(between_post_pose.get("pose_source") or "unknown")
                        trial.between_pre_lite_required_frames = _coerce_int(post_pose.get("lite_required_frames"))
                        trial.between_post_lite_required_frames = _coerce_int(between_post_pose.get("lite_required_frames"))
                        log_line(
                            "[Y-CONSISTENCY] "
                            f"Trial {trial_idx}/{total_trials} between-step: "
                            f"start_y={trial.between_pre_y_mm:+.2f}mm @ {trial.between_pre_dist_mm:.2f}mm "
                            f"-> end_y={trial.between_post_y_mm:+.2f}mm @ {trial.between_post_dist_mm:.2f}mm "
                            f"(cmd_delta={trial.between_cmd_delta_mm:+.2f}mm, dist_delta={trial.between_dist_delta_mm:+.2f}mm) "
                            f"via {_action_description(between_cmd, between_score, between_meta)} "
                            f"src={trial.between_pre_pose_source}->{trial.between_post_pose_source}"
                        )

                remaining_trials_after_this = total_trials - trial_idx
                if between_reset_to_approach_shelf and remaining_trials_after_this > 0:
                    reset_pose, reset_meta = _acquire_target_y_plain(
                        vision=vision,
                        world=world,
                        robot=robot,
                        target_y_mm=float(approach_shelf_target_y_mm),
                        target_y_tol_mm=float(between_reset_tol_mm),
                        setup_score=int(setup_score),
                        setup_duration_ms=setup_duration_ms,
                        setup_fine_score=int(setup_fine_score),
                        setup_fine_duration_ms=setup_fine_duration_ms,
                        setup_fine_window_mm=float(setup_fine_window_mm),
                        setup_max_acts=int(between_reset_max_acts),
                        observe_samples=observe_samples,
                        observe_timeout_s=observe_timeout_s,
                        settle_s=post_act_settle_s,
                        target_dist_mm=effective_target_dist_mm,
                        dist_tol_mm=dist_tol_mm,
                    )
                    trial.reset_acts = _coerce_int(reset_meta.get("reset_acts"), 0)
                    trial.reset_cmds = list(reset_meta.get("reset_cmds") or [])
                    trial.reset_modes = list(reset_meta.get("reset_modes") or [])
                    trial.reset_target_y_mm = float(approach_shelf_target_y_mm)
                    reset_initial_pose = reset_meta.get("initial_pose") if isinstance(reset_meta, dict) else None
                    reset_final_pose = reset_meta.get("final_pose") if isinstance(reset_meta, dict) else None
                    trial.reset_initial_y_mm = _coerce_float(reset_initial_pose.get("offset_y")) if isinstance(reset_initial_pose, dict) else None
                    trial.reset_final_y_error_mm = (
                        _coerce_float(reset_final_pose.get("offset_y")) - float(approach_shelf_target_y_mm)
                        if isinstance(reset_final_pose, dict)
                        else None
                    )
                    if reset_pose is None:
                        status = "aborted"
                        abort_reason = f"{reset_meta.get('reason')}_trial_{trial_idx}"
                        log_line(
                            f"[Y-CONSISTENCY] Trial {trial_idx}/{total_trials}: between-reset failed "
                            f"({reset_meta.get('reason')}). Aborting."
                        )
                        trials.append(trial)
                        break
                    log_line(
                        "[Y-CONSISTENCY] "
                        f"Trial {trial_idx}/{total_trials} between-reset: "
                        f"start_y={float(trial.reset_initial_y_mm or post_y_mm):+.2f}mm "
                        f"-> shelf_y={float(reset_pose['offset_y']):+.2f}mm using {int(trial.reset_acts or 0)} act(s) "
                        f"{trial.reset_cmds or []} modes={trial.reset_modes or []}"
                    )

                trials.append(trial)
            if status != "completed":
                break
    finally:
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass

    summary = _build_trial_summary(trials)
    summary["status"] = status
    summary["abort_reason"] = abort_reason
    summary_by_duration_ms = {}
    summary_by_cmd = {}
    summary_by_cmd_duration = {}
    grouped_trials: dict[int | None, list[TrialResult]] = defaultdict(list)
    grouped_trials_by_cmd: dict[str, list[TrialResult]] = defaultdict(list)
    grouped_trials_by_cmd_duration: dict[str, list[TrialResult]] = defaultdict(list)
    for item in trials:
        grouped_trials[item.duration_group_ms].append(item)
        grouped_trials_by_cmd[str(item.cmd)].append(item)
        duration_label = "default" if item.duration_group_ms is None else str(int(item.duration_group_ms))
        grouped_trials_by_cmd_duration[f"{str(item.cmd)}:{duration_label}"].append(item)
    for duration_key, subset in grouped_trials.items():
        label = "default" if duration_key is None else str(int(duration_key))
        summary_by_duration_ms[label] = _build_trial_summary(subset)
    for cmd_key, subset in grouped_trials_by_cmd.items():
        summary_by_cmd[str(cmd_key)] = _build_trial_summary(subset)
    for group_key, subset in grouped_trials_by_cmd_duration.items():
        summary_by_cmd_duration[str(group_key)] = _build_trial_summary(subset)

    summary_lines = []
    for group_key in sorted(summary_by_cmd_duration.keys()):
        cmd_key, _, duration_key = str(group_key).partition(":")
        duration_val = None if duration_key == "default" else _coerce_int(duration_key)
        group_summary = summary_by_cmd_duration[group_key]
        summary_lines.append(
            _consistency_summary_line(
                trial_count=int(group_summary.get("trial_count") or 0),
                cmd=str(cmd_key),
                score=int(score),
                duration_ms=duration_val,
                summary=group_summary,
            )
        )

    payload = {
        "schema_version": 1,
        "source": "zdebug_Yaxis_consistency",
        "generated_at": time.time(),
        "started_at": started_at,
        "finished_at": time.time(),
        "step": "ALIGN_BRICK",
        "config": {
            "repeats": repeats,
            "cmd": test_cmd,
            "alternate_cmds": bool(alternate_cmds),
            "center_y_mm": float(center_y_mm),
            "speed_score": int(score),
            "duration_override_ms": duration_override_ms,
            "duration_ms_list": duration_schedule_ms,
            "between_cmd": between_cmd,
            "between_score": int(between_score),
            "between_duration_ms": between_duration_ms,
            "target_y_mm": target_y_mm,
            "target_y_tol_mm": target_y_tol_mm if target_y_mm is not None else None,
            "alternate_target_band_mm": alternate_target_band_mm,
            "alternate_target_band_tol_mm": alternate_target_band_tol_mm if alternate_target_band_mm is not None else None,
            "success_margin_mm": success_margin_mm,
            "setup_score": int(setup_score),
            "setup_duration_ms": setup_duration_ms,
            "setup_fine_score": int(setup_fine_score),
            "setup_fine_duration_ms": setup_fine_duration_ms,
            "setup_fine_window_mm": setup_fine_window_mm,
            "setup_max_acts": setup_max_acts,
            "setup_approach_cmd": setup_approach_cmd,
            "setup_approach_margin_mm": setup_approach_margin_mm,
            "between_reset_to_approach_shelf": between_reset_to_approach_shelf,
            "between_reset_tol_mm": between_reset_tol_mm if between_reset_to_approach_shelf else None,
            "between_reset_max_acts": between_reset_max_acts if between_reset_to_approach_shelf else None,
            "approach_shelf_target_y_mm": approach_shelf_target_y_mm if between_reset_to_approach_shelf else None,
            "target_dist_mm": target_dist_mm,
            "dist_tol_mm": dist_tol_mm,
            "lock_initial_dist": bool(args.lock_initial_dist),
            "locked_initial_dist_mm": locked_initial_dist_mm,
            "observe_samples": observe_samples,
            "observe_timeout_s": observe_timeout_s,
            "post_act_settle_s": post_act_settle_s,
            "between_settle_s": between_settle_s,
        },
        "planned_test_action": planned_test,
        "planned_test_actions": planned_test_actions,
        "planned_between_action": planned_between,
        "summary": summary,
        "summary_by_duration_ms": summary_by_duration_ms,
        "summary_by_cmd": summary_by_cmd,
        "summary_by_cmd_duration": summary_by_cmd_duration,
        "summary_lines": summary_lines,
        "story_lines": [str(item.story_line) for item in trials if item.story_line],
        "trials": [asdict(item) for item in trials],
    }
    _write_results(results_path, payload)

    if trials:
        summary_meta = summary["cmd_delta_mm"]
        median_delta = _coerce_float(summary_meta.get("median"), 0.0)
        max_abs_dev = _coerce_float(summary_meta.get("max_abs_deviation_from_median"), 0.0)
        min_delta = _coerce_float(summary_meta.get("min"), 0.0)
        max_delta = _coerce_float(summary_meta.get("max"), 0.0)
        stdev_delta = _coerce_float(summary_meta.get("stdev"), 0.0)
        direction_success_text = str(summary.get("cmd_direction_success_rate_display") or "n/a")
        observed_meta = {
            "score_model": int(score),
            "pwm": trials[0].pwm,
            "power": trials[0].power,
            "duration_ms": trials[0].duration_ms,
        }
        action_desc = (
            f"AUTO alternating schedule @ {int(score)}%"
            if bool(alternate_cmds)
            else (
                f"AUTO center-aware @ {int(score)}%"
                if test_cmd == "auto"
                else _action_description(test_cmd, score, observed_meta)
            )
        )
        log_line(
            "[Y-CONSISTENCY] Summary: "
            f'when we send "{action_desc}", '
            f"the command-direction stroke is median {median_delta:.2f} mm "
            f"(min {min_delta:.2f}, max {max_delta:.2f}, stdev {stdev_delta:.2f}, "
            f"inconsistencyRealityNumber +/-{max_abs_dev:.2f} mm)."
        )
        log_line(f"[Y-CONSISTENCY] Directionally correct acts: {direction_success_text}")
        for line in summary_lines:
            log_line(f"[Y-CONSISTENCY] {line}")
        dist_summary_meta = summary["dist_delta_mm"]
        log_line(
            "[Y-CONSISTENCY] Distance summary: "
            f"median dist_delta {(_coerce_float(dist_summary_meta.get('median'), 0.0)):+.2f} mm "
            f"(stdev {(_coerce_float(dist_summary_meta.get('stdev'), 0.0)):.2f} mm)."
        )
        if int(summary.get("hit_total") or 0) > 0:
            log_line(
                "[Y-CONSISTENCY] One-shot hit rate: "
                f"{str(summary.get('hit_rate_display') or 'n/a')} within +/- {float(success_margin_mm or 0.0):.2f} mm of center."
            )
        between_summary_meta = summary["between_cmd_delta_mm"]
        if int(between_summary_meta.get("count") or 0) > 0:
            log_line(
                "[Y-CONSISTENCY] Between-step summary: "
                f"median command-direction stroke {(_coerce_float(between_summary_meta.get('median'), 0.0)):.2f} mm "
                f"(stdev {(_coerce_float(between_summary_meta.get('stdev'), 0.0)):.2f} mm)."
            )
        reset_summary_meta = summary["reset_acts"]
        if int(reset_summary_meta.get("count") or 0) > 0:
            log_line(
                "[Y-CONSISTENCY] Between-reset summary: "
                f"median reset acts {(_coerce_float(reset_summary_meta.get('median'), 0.0)):.1f} "
                f"(stdev {(_coerce_float(reset_summary_meta.get('stdev'), 0.0)):.2f}), "
                f"median shelf error {(_coerce_float(summary['reset_final_y_error_mm'].get('median'), 0.0)):+.2f} mm."
            )
        if len(summary_by_duration_ms) > 1:
            for duration_key, item in summary_by_duration_ms.items():
                item_meta = item["cmd_delta_mm"]
                log_line(
                    "[Y-CONSISTENCY] Duration group summary: "
                    f"{duration_key}ms -> median {(_coerce_float(item_meta.get('median'), 0.0)):.2f} mm "
                    f"(stdev {(_coerce_float(item_meta.get('stdev'), 0.0)):.2f} mm, "
                    f"setup acts median {(_coerce_float(item['setup_acts'].get('median'), 0.0)):.1f})"
                )
    else:
        log_line("[Y-CONSISTENCY] No completed trials were recorded.")

    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
