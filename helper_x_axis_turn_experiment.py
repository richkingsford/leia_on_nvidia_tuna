#!/usr/bin/env python3
"""Semi-manual x-axis observe-while-moving trial helper."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import telemetry_robot as telemetry_robot_module
from telemetry_process import send_robot_command_pwm
from telemetry_robot import StepState


RUN_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "world_model_x_axis_turn_experiment.json"
DEFAULT_SCORE = 1
DEFAULT_SAMPLE_HZ = 10.0
MAX_DURATION_MS = 8000
DEFAULT_TARGET_SUCCESS_BAND_MM = 0.4
DEFAULT_TARGET_X_AXIS_MM = 0.0
DEFAULT_MAX_ACTS = 15
PWM_STEP_UP = 2
PWM_STEP_DOWN = 2
STALL_PROGRESS_MM = 0.75
VISIBILITY_LOSS_PAUSE_FRAMES = 2
CAUTIOUS_OBSERVE_ACT_WITHIN_MM = 40.0
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
        return None, "[OBSERVE] Expected: <direction[L/R]>."
    parts = text.split()
    if len(parts) != 1:
        return None, "[OBSERVE] Expected exactly 1 param: <direction[L/R]>."

    direction_token = str(parts[0]).strip().lower()
    if direction_token not in {"l", "r"}:
        return None, "[OBSERVE] Direction must be L or R."

    return {
        "direction": str(direction_token),
        "max_act_duration_ms": int(max_duration_ms),
        "target_x_axis_mm": float(DEFAULT_TARGET_X_AXIS_MM),
    }, None


def _target_error_mm(x_axis_mm, target_x_axis_mm):
    x_val = _coerce_float(x_axis_mm, None)
    target_val = _coerce_float(target_x_axis_mm, None)
    if x_val is None or target_val is None:
        return None
    return float(x_val) - float(target_val)


def _sign(value):
    numeric = _coerce_float(value, None)
    if numeric is None:
        return 0
    if float(numeric) > 0.0:
        return 1
    if float(numeric) < 0.0:
        return -1
    return 0


def _inverse_turn_cmd(cmd):
    cmd_key = str(cmd or "").strip().lower()
    return "l" if cmd_key == "r" else "r"


def _turn_profile_for_cmd(cmd):
    cmd_key = str(cmd or "").strip().lower()
    profile = telemetry_robot_module.breakaway_recommended_turn_profile_for_cmd(cmd_key)
    if not isinstance(profile, dict):
        power, pwm, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd_key, DEFAULT_SCORE)
        return {
            "cmd": cmd_key,
            "pwm": int(max(0, int(round(float(pwm or 0))))),
            "power": float(power or 0.0),
            "duration_ms": int(max(1, int(round(float(duration_ms or 1))))),
        }
    try:
        pwm = int(round(float(profile.get("pwm") or 0)))
    except (TypeError, ValueError):
        pwm = 0
    try:
        power = float(profile.get("power"))
    except (TypeError, ValueError):
        power = float(telemetry_robot_module.pwm_to_power(pwm) or 0.0)
    try:
        duration_ms = int(round(float(profile.get("duration_ms") or 0)))
    except (TypeError, ValueError):
        duration_ms = 0
    if duration_ms <= 0:
        _, _, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd_key, DEFAULT_SCORE)
    return {
        "cmd": cmd_key,
        "pwm": int(max(0, int(pwm))),
        "power": float(max(0.0, min(1.0, float(power or 0.0)))),
        "duration_ms": int(max(1, int(duration_ms))),
    }


def _desired_pwm_for_abs_x(abs_x_mm, floor_pwm):
    abs_x = _coerce_float(abs_x_mm, None)
    base_pwm = max(0, int(round(float(floor_pwm or 0))))
    if abs_x is None:
        return int(base_pwm)
    if float(abs_x) >= 60.0:
        boost = 12
    elif float(abs_x) >= 40.0:
        boost = 10
    elif float(abs_x) >= 25.0:
        boost = 8
    elif float(abs_x) >= 15.0:
        boost = 6
    elif float(abs_x) >= 8.0:
        boost = 4
    elif float(abs_x) >= 3.0:
        boost = 2
    else:
        boost = 0
    return int(base_pwm + boost)


def _desired_act_duration_ms(abs_x_mm, *, max_act_duration_ms: int = MAX_DURATION_MS):
    abs_x = _coerce_float(abs_x_mm, None)
    max_ms = max(1, int(max_act_duration_ms))
    if abs_x is None:
        target_ms = max_ms
    elif float(abs_x) >= 60.0:
        target_ms = 2600
    elif float(abs_x) >= 35.0:
        target_ms = 1800
    elif float(abs_x) >= 20.0:
        target_ms = 1200
    elif float(abs_x) >= 10.0:
        target_ms = 800
    elif float(abs_x) >= 4.0:
        target_ms = 450
    elif float(abs_x) >= 1.5:
        target_ms = 260
    else:
        target_ms = 180
    return int(max(1, min(int(target_ms), int(max_ms))))


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
    visible_now = bool(brick.get("visible"))
    return {
        "visible": visible_now,
        "offset_x": _round_triplet(brick.get("offset_x")) if visible_now else None,
        "offset_y": _round_triplet(brick.get("offset_y")) if visible_now else None,
        "dist": _round_triplet(brick.get("dist")) if visible_now else None,
        "angle": _round_triplet(brick.get("angle")) if visible_now else None,
        "confidence": _round_triplet(brick.get("confidence")) if visible_now else None,
        "pose_source": pose_source,
        "obs_ts": _round_triplet(time.time()),
    }


def _sample_row(
    *,
    pose,
    sample_index: int,
    stage_elapsed_s: float,
    target_x_axis_mm: float,
    act_index: int | None = None,
    act_cmd: str | None = None,
    act_elapsed_s: float | None = None,
):
    x_axis_mm = _coerce_float((pose or {}).get("offset_x"), None)
    target_error_mm = _target_error_mm(x_axis_mm, target_x_axis_mm)
    row = {
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
    if act_index is not None:
        row["act_index"] = int(act_index)
    if act_cmd is not None:
        row["act_cmd"] = str(act_cmd)
    if act_elapsed_s is not None:
        row["act_elapsed_s"] = _round_triplet(act_elapsed_s)
    return row


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
    vision_io_lock=None,
    sample_hz: float = DEFAULT_SAMPLE_HZ,
    speed_score: int = DEFAULT_SCORE,
    target_x_axis_mm: float = DEFAULT_TARGET_X_AXIS_MM,
    target_success_band_mm: float = DEFAULT_TARGET_SUCCESS_BAND_MM,
    max_act_duration_ms: int = MAX_DURATION_MS,
    max_acts: int = DEFAULT_MAX_ACTS,
    cancel_event=None,
    cancel_check=None,
    log_path: Path | None = RUN_LOG_FILE_DEFAULT,
    log_fn=None,
):
    logger = log_fn if callable(log_fn) else (lambda *_args, **_kwargs: None)
    cmd = str(direction or "").strip().lower()
    if cmd not in {"l", "r"}:
        raise ValueError("direction must be 'l' or 'r'")
    act_duration_cap_ms = int(max_act_duration_ms)
    if act_duration_cap_ms <= 0:
        raise ValueError("max_act_duration_ms must be > 0")
    if act_duration_cap_ms > int(MAX_DURATION_MS):
        raise ValueError(f"max_act_duration_ms must be <= {int(MAX_DURATION_MS)}")
    max_act_count = max(1, int(max_acts))
    success_band_mm = max(
        0.0,
        float(_coerce_float(target_success_band_mm, DEFAULT_TARGET_SUCCESS_BAND_MM) or DEFAULT_TARGET_SUCCESS_BAND_MM),
    )

    start_pose = _read_world_pose(world, vision, vision_io_lock=vision_io_lock)
    floor_profiles = {key: _turn_profile_for_cmd(key) for key in ("l", "r")}
    current_pwm = {
        key: int(max(0, int((floor_profiles.get(key) or {}).get("pwm") or 0)))
        for key in ("l", "r")
    }
    logger(
        "[OBSERVE] Objective: detect whether Leia can track the brick and hit the target x_axis value "
        f"within +/-{float(success_band_mm):.3f} mm without completely stopping."
    )
    logger(
        f"[OBSERVE] Running adaptive {str(cmd).upper()} toward target "
        f"x_axis={float(target_x_axis_mm):.3f} mm with a hard max of {int(act_duration_cap_ms)} ms per act."
    )

    def _cancel_requested():
        try:
            if cancel_event is not None and bool(cancel_event.is_set()):
                return True
        except Exception:
            pass
        try:
            if callable(cancel_check) and bool(cancel_check()):
                return True
        except Exception:
            pass
        return False

    stage_started = time.monotonic()
    sample_period_s = 1.0 / max(1.0, float(sample_hz))
    send_results = []
    first_send_result = None
    acts = []
    samples = []
    state = {
        "last_visible_pose": dict(start_pose) if bool((start_pose or {}).get("visible")) else None,
        "last_visible_x_mm": _coerce_float((start_pose or {}).get("offset_x"), None),
        "last_observed_visible": bool((start_pose or {}).get("visible")),
        "first_visible_pose": None,
        "first_visible_elapsed_ms": None,
        "first_visible_x_axis_mm": None,
        "best_target_pose": None,
        "best_target_error_mm": None,
        "band_hit_pose": None,
        "band_hit_elapsed_ms": None,
        "band_hit_x_axis_mm": None,
        "band_hit_target_error_mm": None,
        "direction_reversals": 0,
        "stall_recoveries": 0,
        "visibility_recoveries": 0,
        "cautious_mode_acts": 0,
    }

    def _capture_sample(*, act_index: int, act_cmd: str, act_started_s: float):
        trial_elapsed_s = max(0.0, float(time.monotonic()) - float(stage_started))
        act_elapsed_s = max(0.0, float(time.monotonic()) - float(act_started_s))
        pose = _read_world_pose(world, vision, vision_io_lock=vision_io_lock)
        sample = _sample_row(
            pose=pose,
            sample_index=int(len(samples) + 1),
            stage_elapsed_s=trial_elapsed_s,
            target_x_axis_mm=float(target_x_axis_mm),
            act_index=int(act_index),
            act_cmd=str(act_cmd),
            act_elapsed_s=act_elapsed_s,
        )
        samples.append(sample)
        logger(_timeline_line(sample))

        visible = bool(sample.get("visible"))
        state["last_observed_visible"] = bool(visible)
        x_axis_mm = _coerce_float(sample.get("offset_x"), None)
        if not visible or x_axis_mm is None:
            return {"sample": sample}

        prev_visible_x = _coerce_float(state.get("last_visible_x_mm"), None)
        state["last_visible_pose"] = dict(pose)
        state["last_visible_x_mm"] = float(x_axis_mm)
        if state.get("first_visible_pose") is None:
            state["first_visible_pose"] = dict(pose)
            state["first_visible_elapsed_ms"] = int(round(float(trial_elapsed_s) * 1000.0))
            state["first_visible_x_axis_mm"] = _round_triplet(x_axis_mm)

        target_error_abs = abs(float(sample["target_error_mm"]))
        best_target_error = _coerce_float(state.get("best_target_error_mm"), None)
        if best_target_error is None or target_error_abs < float(best_target_error):
            state["best_target_error_mm"] = float(target_error_abs)
            state["best_target_pose"] = dict(pose)

        if float(target_error_abs) <= float(success_band_mm):
            state["band_hit_pose"] = dict(pose)
            state["band_hit_elapsed_ms"] = int(round(float(trial_elapsed_s) * 1000.0))
            state["band_hit_x_axis_mm"] = _round_triplet(x_axis_mm)
            state["band_hit_target_error_mm"] = _round_triplet(target_error_abs)
            logger(
                "[OBSERVE] Target band reached at "
                f"{int(state['band_hit_elapsed_ms'])}ms: xaxis={_round_triplet(x_axis_mm)}; "
                f"error={_round_triplet(target_error_abs)} mm within +/-{float(success_band_mm):.3f} mm. "
                "Stopping trial."
            )
            return {"sample": sample, "band_hit": True}

        if prev_visible_x is not None and _sign(prev_visible_x) != 0 and _sign(x_axis_mm) != 0 and _sign(prev_visible_x) != _sign(x_axis_mm):
            reverse_cmd = _inverse_turn_cmd(act_cmd)
            state["direction_reversals"] = int(state.get("direction_reversals") or 0) + 1
            logger(
                "[OBSERVE] Overshoot detected at "
                f"{int(round(float(trial_elapsed_s) * 1000.0))}ms: xaxis crossed zero "
                f"({ _round_triplet(prev_visible_x) } -> { _round_triplet(x_axis_mm) }). "
                f"Reversing to {str(reverse_cmd).upper()}."
            )
            return {"sample": sample, "reverse_cmd": str(reverse_cmd)}

        return {"sample": sample}

    def _capture_act_sample(*, act_state, act_index: int, act_cmd: str, act_started_s: float):
        capture = _capture_sample(
            act_index=int(act_index),
            act_cmd=str(act_cmd),
            act_started_s=float(act_started_s),
        )
        sample = capture.get("sample") or {}
        act_state["samples"] = int(act_state.get("samples") or 0) + 1
        if bool(sample.get("visible")) and _coerce_float(sample.get("offset_x"), None) is not None:
            act_state["visible_samples"] = int(act_state.get("visible_samples") or 0) + 1
        return capture, sample

    active_cmd = str(cmd)
    stop_robot = getattr(robot, "stop", None)
    status = "completed"
    try:
        for act_index in range(1, int(max_act_count) + 1):
            if _cancel_requested():
                status = "cancelled"
                break
            if state.get("band_hit_pose") is not None:
                break

            reference_pose = state.get("last_visible_pose") if bool(state.get("last_observed_visible")) else None
            reference_x = _coerce_float((reference_pose or {}).get("offset_x"), None)
            if reference_x is None and bool(state.get("last_observed_visible")):
                reference_x = _coerce_float((start_pose or {}).get("offset_x"), None)
            reference_abs_x = abs(float(reference_x)) if reference_x is not None else None
            reference_target_error_abs = (
                abs(float(reference_x) - float(target_x_axis_mm))
                if reference_x is not None
                else None
            )
            cautious_cycle = bool(
                reference_target_error_abs is not None
                and float(reference_target_error_abs) <= float(CAUTIOUS_OBSERVE_ACT_WITHIN_MM)
            )

            floor_pwm = int((floor_profiles.get(active_cmd) or {}).get("pwm") or 0)
            pwm_used = int(max(floor_pwm, int(current_pwm.get(active_cmd) or floor_pwm)))
            if reference_abs_x is not None:
                heuristic_pwm = _desired_pwm_for_abs_x(reference_abs_x, floor_pwm)
                if pwm_used > heuristic_pwm:
                    pwm_used = max(int(heuristic_pwm), int(pwm_used - int(PWM_STEP_DOWN)))
                elif pwm_used < heuristic_pwm:
                    pwm_used = min(int(heuristic_pwm), int(pwm_used + int(PWM_STEP_UP)))
            current_pwm[active_cmd] = int(max(floor_pwm, pwm_used))
            act_duration_ms = _desired_act_duration_ms(reference_abs_x, max_act_duration_ms=int(act_duration_cap_ms))
            power_used = float(telemetry_robot_module.pwm_to_power(current_pwm[active_cmd]) or (floor_profiles.get(active_cmd) or {}).get("power") or 0.0)

            logger(
                f"[OBSERVE] Act {int(act_index)}: {str(active_cmd).upper()} pwm={int(current_pwm[active_cmd])}, "
                f"pwr={float(power_used):.3f}, max={int(act_duration_ms)}ms."
            )
            if cautious_cycle:
                state["cautious_mode_acts"] = int(state.get("cautious_mode_acts") or 0) + 1
                logger(
                    "[OBSERVE] Within 40mm of target: switching to cautious observe->act cycle "
                    "(no in-motion observe samples for this act)."
                )
            send_result = send_robot_command_pwm(
                robot,
                world,
                EXPERIMENT_STEP,
                str(active_cmd),
                float(power_used),
                int(current_pwm[active_cmd]),
                int(act_duration_ms),
                speed_score=int(speed_score),
                auto_mode=False,
                half_first_turn_pulse=False,
                ease_in_out_enabled=False,
                respect_requested_duration=True,
            )
            send_row = dict(send_result or {}) if isinstance(send_result, dict) else {}
            send_row["duration_requested_ms"] = int(act_duration_ms)
            send_results.append(send_row)
            if first_send_result is None:
                first_send_result = dict(send_row)

            act_started_s = time.monotonic()
            act_deadline_s = float(act_started_s) + (float(act_duration_ms) / 1000.0)
            next_sample_time = float(act_started_s)
            act_state = {
                "index": int(act_index),
                "cmd": str(active_cmd),
                "pwm": int(current_pwm[active_cmd]),
                "power": _round_triplet(power_used),
                "duration_requested_ms": int(act_duration_ms),
                "start_x_axis_mm": _round_triplet(reference_x),
                "result": "completed",
                "samples": 0,
                "visible_samples": 0,
                "max_invisible_streak": 0,
            }
            reverse_cmd = None
            invisible_streak = 0

            if cautious_cycle:
                while True:
                    if _cancel_requested():
                        act_state["result"] = "cancelled"
                        status = "cancelled"
                        if callable(stop_robot):
                            stop_robot()
                        break
                    now = time.monotonic()
                    if now >= float(act_deadline_s):
                        break
                    time.sleep(min(0.01, max(0.0, float(act_deadline_s) - float(now))))

                if act_state["result"] != "cancelled":
                    if callable(stop_robot):
                        stop_robot()
                    capture, sample = _capture_act_sample(
                        act_state=act_state,
                        act_index=int(act_index),
                        act_cmd=str(active_cmd),
                        act_started_s=float(act_started_s),
                    )
                    sample_has_pose = bool(sample.get("visible")) and _coerce_float(sample.get("offset_x"), None) is not None
                    if sample_has_pose:
                        invisible_streak = 0
                    else:
                        invisible_streak = int(invisible_streak) + 1
                        act_state["max_invisible_streak"] = max(
                            int(act_state.get("max_invisible_streak") or 0),
                            int(invisible_streak),
                        )

                    if capture.get("band_hit"):
                        act_state["result"] = "target_band_reached"
                        if callable(stop_robot):
                            stop_robot()
                    else:
                        reverse_cmd = capture.get("reverse_cmd")
                        if reverse_cmd:
                            act_state["result"] = "overshoot_reversed"
                            if callable(stop_robot):
                                stop_robot()
            else:
                while True:
                    if _cancel_requested():
                        act_state["result"] = "cancelled"
                        status = "cancelled"
                        if callable(stop_robot):
                            stop_robot()
                        break
                    now = time.monotonic()
                    if now >= float(next_sample_time):
                        capture, sample = _capture_act_sample(
                            act_state=act_state,
                            act_index=int(act_index),
                            act_cmd=str(active_cmd),
                            act_started_s=float(act_started_s),
                        )
                        sample_has_pose = bool(sample.get("visible")) and _coerce_float(sample.get("offset_x"), None) is not None
                        if sample_has_pose:
                            invisible_streak = 0
                        else:
                            invisible_streak = int(invisible_streak) + 1
                            act_state["max_invisible_streak"] = max(
                                int(act_state.get("max_invisible_streak") or 0),
                                int(invisible_streak),
                            )

                        if capture.get("band_hit"):
                            act_state["result"] = "target_band_reached"
                            if callable(stop_robot):
                                stop_robot()
                            break
                        reverse_cmd = capture.get("reverse_cmd")
                        if reverse_cmd:
                            act_state["result"] = "overshoot_reversed"
                            if callable(stop_robot):
                                stop_robot()
                            break
                        next_sample_time += float(sample_period_s)

                    if now >= float(act_deadline_s):
                        break
                    sleep_until = min(float(next_sample_time), float(act_deadline_s))
                    delay_s = max(0.0, float(sleep_until) - float(time.monotonic()))
                    if delay_s > 0.0:
                        time.sleep(delay_s)

            act_state["end_x_axis_mm"] = _round_triplet(
                _coerce_float(((state.get("last_visible_pose") or {}) if isinstance(state.get("last_visible_pose"), dict) else {}).get("offset_x"), None)
            )
            end_x_axis_mm = _coerce_float(act_state.get("end_x_axis_mm"), None)
            end_abs_x = abs(float(end_x_axis_mm)) if end_x_axis_mm is not None else None
            if act_state["result"] == "completed":
                if not bool(state.get("last_observed_visible")) and int(act_state.get("max_invisible_streak") or 0) >= int(VISIBILITY_LOSS_PAUSE_FRAMES):
                    previous_pwm = int(current_pwm.get(active_cmd) or floor_pwm)
                    current_pwm[active_cmd] = int(previous_pwm) + int(PWM_STEP_UP)
                    state["visibility_recoveries"] = int(state.get("visibility_recoveries") or 0) + 1
                    act_state["result"] = "continue_blind_crawl"
                    logger(
                        f"[OBSERVE] Brick visibility lost for {int(act_state.get('max_invisible_streak') or 0)} consecutive frames during "
                        f"act {int(act_index)}. Keeping movement active and increasing {str(active_cmd).upper()} PWM "
                        f"from {int(previous_pwm)} to {int(current_pwm[active_cmd])} for the next act."
                    )
                elif (
                    act_state.get("visible_samples")
                    and reference_abs_x is not None
                    and end_abs_x is not None
                    and float(end_abs_x) > float(success_band_mm)
                ):
                    progress_mm = max(0.0, float(reference_abs_x) - float(end_abs_x))
                    act_state["progress_mm"] = _round_triplet(progress_mm)
                    if float(progress_mm) < float(STALL_PROGRESS_MM):
                        previous_pwm = int(current_pwm.get(active_cmd) or floor_pwm)
                        current_pwm[active_cmd] = int(previous_pwm) + int(PWM_STEP_UP)
                        state["stall_recoveries"] = int(state.get("stall_recoveries") or 0) + 1
                        act_state["result"] = "stalled_below_breakaway"
                        logger(
                            f"[OBSERVE] Progress stalled during act {int(act_index)}. "
                            f"Increasing {str(active_cmd).upper()} PWM from {int(previous_pwm)} "
                            f"to {int(current_pwm[active_cmd])} for the next act."
                        )
            act_state["seconds"] = _round_triplet(max(0.0, float(time.monotonic()) - float(act_started_s)))
            acts.append(act_state)

            if state.get("band_hit_pose") is not None:
                status = "target_band_reached"
                break
            if status == "cancelled":
                break
            if reverse_cmd:
                active_cmd = str(reverse_cmd)
                other_floor = int((floor_profiles.get(active_cmd) or {}).get("pwm") or 0)
                current_pwm[active_cmd] = max(int(current_pwm.get(active_cmd) or other_floor), int(other_floor))
                continue
        else:
            status = "max_acts_reached"
    finally:
        if callable(stop_robot):
            stop_robot()

    end_pose = _read_world_pose(world, vision, vision_io_lock=vision_io_lock)
    if bool((end_pose or {}).get("visible")) and state.get("first_visible_pose") is None:
        state["first_visible_pose"] = dict(end_pose)
        state["first_visible_elapsed_ms"] = int(round(max(0.0, time.monotonic() - float(stage_started)) * 1000.0))
        state["first_visible_x_axis_mm"] = _round_triplet((end_pose or {}).get("offset_x"))
    result_pose = dict(state.get("band_hit_pose") or {})
    if not result_pose:
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
    band_hit_error_mm = _coerce_float(state.get("band_hit_target_error_mm"), None)
    if band_hit_error_mm is not None:
        status = "target_band_reached"
    elif status == "cancelled":
        status = "cancelled"
    elif result_x_axis_mm is None:
        status = "never_visible"
    elif status not in {"completed", "max_acts_reached", "cancelled"}:
        status = "completed"

    result = {
        "ok": True,
        "experiment_type": "semi_manual_observe_while_moving_x_axis",
        "objective": "detect whether Leia can track the brick and hit the target x_axis value without completely stopping",
        "direction": str(cmd),
        "duration_ms": None,
        "max_act_duration_ms": int(act_duration_cap_ms),
        "max_acts": int(max_act_count),
        "target_x_axis_mm": _round_triplet(target_x_axis_mm),
        "target_success_band_mm": _round_triplet(success_band_mm),
        "speed_score": int(speed_score),
        "continuous_duration_ms": None,
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
        "acts": acts,
        "samples": samples,
        "sample_summary": _summarize_samples(samples),
        "first_visible_pose": state.get("first_visible_pose"),
        "first_visible_elapsed_ms": state.get("first_visible_elapsed_ms"),
        "first_visible_x_axis_mm": state.get("first_visible_x_axis_mm"),
        "band_hit_pose": state.get("band_hit_pose"),
        "band_hit_elapsed_ms": state.get("band_hit_elapsed_ms"),
        "band_hit_x_axis_mm": state.get("band_hit_x_axis_mm"),
        "band_hit_target_error_mm": _round_triplet(state.get("band_hit_target_error_mm")),
        "best_target_pose": state.get("best_target_pose"),
        "best_target_error_mm": _round_triplet(state.get("best_target_error_mm")),
        "direction_reversals": int(state.get("direction_reversals") or 0),
        "stall_recoveries": int(state.get("stall_recoveries") or 0),
        "visibility_recoveries": int(state.get("visibility_recoveries") or 0),
        "cautious_mode_acts": int(state.get("cautious_mode_acts") or 0),
        "seconds": _round_triplet(max(0.0, time.monotonic() - float(stage_started))),
    }
    if log_path is not None:
        log_path = Path(log_path)
        result["log_path"] = str(log_path)
        log_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
