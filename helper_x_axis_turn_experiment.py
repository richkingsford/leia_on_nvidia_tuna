#!/usr/bin/env python3
"""Semi-manual x-axis observe-while-moving trial helper."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import telemetry_robot as telemetry_robot_module
from telemetry_process import send_robot_command
from telemetry_robot import StepState


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_x_axis_turn_experiment.json"
DEFAULT_SCORE = 1
DEFAULT_SAMPLE_HZ = 10.0
MAX_DURATION_MS = 5000
EXPERIMENT_STEP = StepState.ALIGN_BRICK


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_triplet(value):
    numeric = _coerce_float(value, None)
    if numeric is None:
        return None
    return round(float(numeric), 3)


def parse_observe_while_moving_trial_input(raw, *, max_duration_ms: int = MAX_DURATION_MS):
    text = str(raw or "").strip()
    if not text:
        return None, "[OBSERVE] Expected: <direction[L/R]> <duration_ms> <target_x_axis_mm>."
    parts = text.split()
    if len(parts) != 3:
        return None, "[OBSERVE] Expected exactly 3 params: <direction[L/R]> <duration_ms> <target_x_axis_mm>."

    direction_token = str(parts[0]).strip().lower()
    if direction_token not in {"l", "r"}:
        return None, "[OBSERVE] Direction must be L or R."

    try:
        duration_ms = int(round(float(parts[1])))
    except (TypeError, ValueError):
        return None, "[OBSERVE] Duration must be a number of milliseconds."
    if duration_ms <= 0:
        return None, "[OBSERVE] Duration must be greater than 0 ms."
    if duration_ms > int(max_duration_ms):
        return None, (
            f"[OBSERVE] Duration {int(duration_ms)} ms exceeds the hard max of "
            f"{int(max_duration_ms)} ms. Keep the trial at or below {int(max_duration_ms)} ms "
            "so Leia can keep tracking while moving."
        )

    try:
        target_x_axis_mm = float(parts[2])
    except (TypeError, ValueError):
        return None, "[OBSERVE] Target x_axis must be numeric."

    return {
        "direction": str(direction_token),
        "duration_ms": int(duration_ms),
        "target_x_axis_mm": float(target_x_axis_mm),
    }, None


def _read_world_pose(world, vision, *, vision_io_lock=None):
    if world is None or vision is None:
        return {
            "visible": False,
            "offset_x": None,
            "offset_y": None,
            "dist": None,
            "angle": None,
            "confidence": None,
            "obs_ts": _round_triplet(time.time()),
        }

    if vision_io_lock is None:
        found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
    else:
        with vision_io_lock:
            found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()

    world.update_vision(
        bool(found),
        float(dist),
        float(angle),
        float(conf),
        float(offset_x),
        float(cam_h),
        bool(brick_above),
        bool(brick_below),
    )
    brick = getattr(world, "brick", {}) or {}
    pose_source = str(brick.get("pose_source") or "")
    return {
        "visible": bool(brick.get("visible")),
        "offset_x": _round_triplet(brick.get("offset_x")),
        "offset_y": _round_triplet(brick.get("offset_y")),
        "dist": _round_triplet(brick.get("dist")),
        "angle": _round_triplet(brick.get("angle")),
        "confidence": _round_triplet(brick.get("confidence")),
        "pose_source": pose_source,
        "obs_ts": _round_triplet(time.time()),
    }


def _sample_row(*, pose, sample_index: int, stage_elapsed_s: float, target_x_axis_mm: float):
    x_axis_mm = _coerce_float((pose or {}).get("offset_x"), None)
    target_error_mm = None
    if x_axis_mm is not None:
        target_error_mm = float(x_axis_mm) - float(target_x_axis_mm)
    return {
        "sample_index": int(sample_index),
        "stage_elapsed_s": _round_triplet(stage_elapsed_s),
        "visible": bool((pose or {}).get("visible")),
        "offset_x": _round_triplet(x_axis_mm),
        "offset_y": _round_triplet((pose or {}).get("offset_y")),
        "dist": _round_triplet((pose or {}).get("dist")),
        "angle": _round_triplet((pose or {}).get("angle")),
        "confidence": _round_triplet((pose or {}).get("confidence")),
        "target_x_axis_mm": _round_triplet(target_x_axis_mm),
        "target_error_mm": _round_triplet(target_error_mm),
        "pose_source": str((pose or {}).get("pose_source") or ""),
        "obs_ts": _round_triplet((pose or {}).get("obs_ts")),
    }


def _timeline_line(sample):
    elapsed_ms = int(round(float(_coerce_float((sample or {}).get("stage_elapsed_s"), 0.0) or 0.0) * 1000.0))
    x_axis_mm = _coerce_float((sample or {}).get("offset_x"), None)
    if not bool((sample or {}).get("visible")) or x_axis_mm is None:
        return f"[OBSERVE] {int(elapsed_ms)}ms: Not visible"
    return f"[OBSERVE] {int(elapsed_ms)}ms: Visible; xaxis: {_round_triplet(x_axis_mm)}"


def _percent_off_target(*, result_x_axis_mm, target_x_axis_mm, start_x_axis_mm):
    result_x = _coerce_float(result_x_axis_mm, None)
    target_x = _coerce_float(target_x_axis_mm, None)
    if result_x is None or target_x is None:
        return None, None
    error_mm = abs(float(result_x) - float(target_x))
    if abs(float(target_x)) > 1e-6:
        return _round_triplet((error_mm / abs(float(target_x))) * 100.0), "target_abs"
    start_x = _coerce_float(start_x_axis_mm, None)
    if start_x is not None and abs(float(start_x) - float(target_x)) > 1e-6:
        return _round_triplet((error_mm / abs(float(start_x) - float(target_x))) * 100.0), "start_gap"
    return None, None


def _summarize_samples(samples):
    if not isinstance(samples, list):
        samples = []
    visible_samples = [row for row in samples if bool((row or {}).get("visible"))]
    x_values = [
        float(row["offset_x"])
        for row in visible_samples
        if _coerce_float((row or {}).get("offset_x"), None) is not None
    ]
    target_errors = [
        abs(float(row["target_error_mm"]))
        for row in visible_samples
        if _coerce_float((row or {}).get("target_error_mm"), None) is not None
    ]
    visible_rate = None
    if samples:
        visible_rate = float(len(visible_samples)) / float(len(samples))
    return {
        "sample_count": int(len(samples)),
        "visible_sample_count": int(len(visible_samples)),
        "visible_rate": _round_triplet(visible_rate),
        "x_axis_mm": {
            "start": _round_triplet(x_values[0]) if x_values else None,
            "end": _round_triplet(x_values[-1]) if x_values else None,
            "median": _round_triplet(statistics.median(x_values)) if x_values else None,
        },
        "best_target_error_mm": _round_triplet(min(target_errors)) if target_errors else None,
        "last_visible_x_axis_mm": _round_triplet(x_values[-1]) if x_values else None,
    }


def run_observe_while_moving_trial(
    *,
    robot,
    world,
    vision,
    direction: str,
    duration_ms: int,
    target_x_axis_mm: float,
    vision_io_lock=None,
    sample_hz: float = DEFAULT_SAMPLE_HZ,
    speed_score: int = DEFAULT_SCORE,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
):
    logger = log_fn if callable(log_fn) else (lambda *_args, **_kwargs: None)
    cmd = str(direction or "").strip().lower()
    if cmd not in {"l", "r"}:
        raise ValueError("direction must be 'l' or 'r'")
    trial_duration_ms = int(duration_ms)
    if trial_duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")
    if trial_duration_ms > int(MAX_DURATION_MS):
        raise ValueError(f"duration_ms must be <= {int(MAX_DURATION_MS)}")

    start_pose = _read_world_pose(world, vision, vision_io_lock=vision_io_lock)
    logger(
        "[OBSERVE] Objective: detect whether Leia can track the brick and hit the target x_axis value "
        "without completely stopping."
    )
    logger(
        f"[OBSERVE] Running {str(cmd).upper()} for {int(trial_duration_ms)} ms toward target "
        f"x_axis={float(target_x_axis_mm):.3f} mm."
    )

    stage_started = time.monotonic()
    stage_deadline = float(stage_started) + (float(trial_duration_ms) / 1000.0)
    sample_period_s = 1.0 / max(1.0, float(sample_hz))
    next_sample_time = float(stage_started)
    send_results = []
    first_send_result = None
    samples = []
    state = {
        "last_visible_pose": None,
        "first_visible_pose": None,
        "first_visible_elapsed_ms": None,
        "first_visible_x_axis_mm": None,
        "best_target_pose": None,
        "best_target_error_mm": None,
    }

    try:
        send_result = send_robot_command(
            robot,
            world,
            EXPERIMENT_STEP,
            str(cmd),
            0.0,
            speed_score=int(speed_score),
            duration_override_ms=int(trial_duration_ms),
            auto_mode=False,
            ease_in_out_enabled=False,
            half_first_turn_pulse=False,
        )
        if isinstance(send_result, dict):
            row = dict(send_result or {})
            row["duration_requested_ms"] = int(trial_duration_ms)
            send_results.append(row)
            first_send_result = dict(row)

        def _capture_sample(stage_elapsed_s: float):
            pose = _read_world_pose(world, vision, vision_io_lock=vision_io_lock)
            sample = _sample_row(
                pose=pose,
                sample_index=int(len(samples) + 1),
                stage_elapsed_s=max(0.0, float(stage_elapsed_s)),
                target_x_axis_mm=float(target_x_axis_mm),
            )
            samples.append(sample)
            logger(_timeline_line(sample))
            if bool(sample.get("visible")) and _coerce_float(sample.get("offset_x"), None) is not None:
                state["last_visible_pose"] = dict(pose)
                if state.get("first_visible_pose") is None:
                    state["first_visible_pose"] = dict(pose)
                    state["first_visible_elapsed_ms"] = int(
                        round(float(_coerce_float(sample.get("stage_elapsed_s"), 0.0) or 0.0) * 1000.0)
                    )
                    state["first_visible_x_axis_mm"] = _round_triplet(sample.get("offset_x"))
                target_error_abs = abs(float(sample["target_error_mm"]))
                best_target_error = _coerce_float(state.get("best_target_error_mm"), None)
                if best_target_error is None or target_error_abs < float(best_target_error):
                    state["best_target_error_mm"] = float(target_error_abs)
                    state["best_target_pose"] = dict(pose)

        _capture_sample(0.0)
        next_sample_time = float(stage_started) + float(sample_period_s)

        while True:
            now = time.monotonic()
            if now >= float(next_sample_time) or not samples:
                _capture_sample(max(0.0, float(now) - float(stage_started)))
                next_sample_time += float(sample_period_s)

            if now >= float(stage_deadline):
                break
            sleep_until = min(float(next_sample_time), float(stage_deadline))
            delay_s = max(0.0, float(sleep_until) - float(time.monotonic()))
            if delay_s > 0.0:
                time.sleep(delay_s)
    finally:
        stop_robot = getattr(robot, "stop", None)
        if callable(stop_robot):
            stop_robot()

    end_pose = _read_world_pose(world, vision, vision_io_lock=vision_io_lock)
    if bool((end_pose or {}).get("visible")) and state.get("first_visible_pose") is None:
        state["first_visible_pose"] = dict(end_pose)
        state["first_visible_elapsed_ms"] = int(trial_duration_ms)
        state["first_visible_x_axis_mm"] = _round_triplet((end_pose or {}).get("offset_x"))
    result_pose = dict(end_pose) if bool((end_pose or {}).get("visible")) else dict(state.get("last_visible_pose") or {})
    result_x_axis_mm = _coerce_float((result_pose or {}).get("offset_x"), None)
    start_x_axis_mm = _coerce_float((start_pose or {}).get("offset_x"), None)
    result_target_error_mm = None
    if result_x_axis_mm is not None:
        result_target_error_mm = abs(float(result_x_axis_mm) - float(target_x_axis_mm))
    percent_off_target, percent_basis = _percent_off_target(
        result_x_axis_mm=result_x_axis_mm,
        target_x_axis_mm=float(target_x_axis_mm),
        start_x_axis_mm=start_x_axis_mm,
    )
    if result_x_axis_mm is None:
        status = "never_visible"
    else:
        status = "completed"

    result = {
        "ok": True,
        "experiment_type": "semi_manual_observe_while_moving_x_axis",
        "objective": "detect whether Leia can track the brick and hit the target x_axis value without completely stopping",
        "direction": str(cmd),
        "duration_ms": int(trial_duration_ms),
        "target_x_axis_mm": _round_triplet(target_x_axis_mm),
        "speed_score": int(speed_score),
        "continuous_duration_ms": int(trial_duration_ms),
        "start_pose": start_pose,
        "end_pose": end_pose,
        "result_pose": result_pose or None,
        "result_x_axis_mm": _round_triplet(result_x_axis_mm),
        "result_target_error_mm": _round_triplet(result_target_error_mm),
        "percent_off_target": _round_triplet(percent_off_target),
        "percent_off_target_basis": percent_basis,
        "status": str(status),
        "send_result": dict(first_send_result or {}),
        "send_results": list(send_results),
        "samples": samples,
        "sample_summary": _summarize_samples(samples),
        "first_visible_pose": state.get("first_visible_pose"),
        "first_visible_elapsed_ms": state.get("first_visible_elapsed_ms"),
        "first_visible_x_axis_mm": state.get("first_visible_x_axis_mm"),
        "best_target_pose": state.get("best_target_pose"),
        "best_target_error_mm": _round_triplet(state.get("best_target_error_mm")),
        "seconds": _round_triplet(max(0.0, time.monotonic() - float(stage_started))),
    }
    if log_path is not None:
        log_path = Path(log_path)
        result["log_path"] = str(log_path)
        log_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
