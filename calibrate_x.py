#!/usr/bin/env python3
"""
Minimal x-axis duration probe.

Observe the current x offset with a 3-frame confidence read, send one turn act
at a fixed 1% speed score and a deterministic duration, observe again, and plot
total x distance traveled in mm against duration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    _MATPLOTLIB_AVAILABLE = False

from helper_calibrate import (
    CalibrationLivePlot,
    build_linear_duration_schedule,
    build_payload as build_shared_payload,
    build_repeated_trial_plan,
    cleanup_old_run_files,
    coerce_finite_float as shared_coerce_finite_float,
    coerce_float as shared_coerce_float,
    coerce_int as shared_coerce_int,
    ensure_run_dir,
    observed_brick_distances_mm as shared_observed_brick_distances_mm,
    planned_durations_ms as shared_planned_durations_ms,
    plot_offsets as shared_plot_offsets,
    plot_series_phase as shared_plot_series_phase,
    trial_label_text as shared_trial_label_text,
    write_results as shared_write_results,
)
from helper_robot_control import Robot
from helper_vision_leia import LeiaVision

# The Yolo brick detector is used by manual training; it's optional here and
# only imported if the module is available.  Using Yolo often gives much more
# robust cyan‑brick tracking than the simple LeiaVision edge detector.
try:
    from brick_detector_yolo import BrickDetector as YoloBrickDetector
except ImportError:
    YoloBrickDetector = None
from telemetry_process import (
    _average_smoothed_frames as telemetry_average_smoothed_frames,
    _latest_unique_smoothed_frames as telemetry_latest_unique_smoothed_frames,
    lite_gate_unique_frames,
    send_robot_command,
    send_robot_command_pwm,
    update_world_from_vision,
)
from telemetry_robot import (
    HOTKEY_SPEED_SCORES,
    StepState,
    WorldModel,
    clamp_pwm,
    normalize_speed_score,
    power_to_pwm,
    pwm_to_power,
    speed_power_pwm_for_cmd,
)

OBSERVE_SLEEP_S = 0.02
OBSERVE_TIMEOUT_S = 1.8
POST_ACT_SETTLE_S = 0.10
OBSERVE_SAMPLES_DEFAULT = 3
X_AXIS_TARGET_MM_DEFAULT = 0.0
SPEED_SCORE_DEFAULT = 20
DURATION_CEILING_MS = 900
DURATION_STEP_MS_DEFAULT = 10
X_AXIS_POSITIVE_CMD_DEFAULT = "r"
X_AXIS_INVERT = False  # Set to True to invert l/r command mapping
PRIMARY_TRIAL_CMD_SEQUENCE_DEFAULT = ("l", "r", "l", "r")

# reference distance associated with the current regression equation.  The
# calibration data assume the brick was this far from the camera (mm).  This is
# written to the world model so downstream code knows its validity range.
REFERENCE_BRICK_DISTANCE_MM: float | None = None
PLOT_TITLE_BRICK_DISTANCE_MM_DEFAULT = 166.0
PLOT_COLOR_BY_CMD = {
    "l": "#1f77b4",
    "r": "#ff7f0e",
}
PLOT_REPEAT_COLOR_BY_CMD = {
    "l": "#00dd77",
    "r": "#dd00dd",
}
PLOT_REPEAT_FAIL_COLOR_BY_CMD = {
    "l": "#ff1493",
    "r": "#ff69b4",
}
REPEAT_RESULT_ERROR_MARGIN_MM = 1.5
MIN_LITE_UNIQUE_FRAMES = 3
REOBSERVE_HOLD_S = 0.12
REOBSERVE_ROUNDS = 2
RELAXED_OBSERVE_TIMEOUT_S = 2.8
RECOVERY_MAX_INVERSE_ACTS = 5
RECOVERY_OBSERVE_TIMEOUT_S = 1.0
# Leave defaults None so live JSON is explicit and plot PNG output stays opt-in.
RESULTS_FILE_DEFAULT: str | None = None
PLOT_FILE_DEFAULT: str | None = None

# Folder where calibration runs deposit live files.
RUN_DIR = Path("Runs - aruco")


def _cleanup_old_run_files():
    cleanup_old_run_files(
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_y_live.json",
            "calibrate_dist_live.json",
        },
    )


def _ensure_run_dir():
    ensure_run_dir(
        run_dir=RUN_DIR,
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_y_live.json",
            "calibrate_dist_live.json",
        },
    )

BRICK_DISTANCE_SOURCE = "vision.dist"
BRICK_DISTANCE_DEFINITION = "Camera-to-brick distance reported by vision at observation time (mm)."
PLOT_TITLE_FONT_SIZE = 10
PLOT_LABEL_FONT_SIZE = 9
PLOT_TICK_FONT_SIZE = 8
PLOT_LEGEND_FONT_SIZE = 8
PLOT_ANNOTATION_FONT_SIZE = 7
ANSI_CYAN_BRIGHT = "\033[96m"
ANSI_BLUE_BRIGHT = "\033[94m"
ANSI_MAGENTA_BRIGHT = "\033[95m"
ANSI_GREEN_BRIGHT = "\033[92m"
ANSI_RED_BRIGHT = "\033[91m"
ANSI_YELLOW_BRIGHT = "\033[93m"


@dataclass
class TrialResult:
    trial: int
    duration_ms: int
    cmd: str
    score_requested: int
    cmd_sent: str | None
    pwm: int | None
    power: float | None
    pre_x_mm: float
    post_x_mm: float
    raw_delta_mm: float
    signed_cmd_delta_mm: float
    cmd_delta_mm: float
    wrong_way: bool
    pre_dist_mm: float
    post_dist_mm: float
    pre_brick_dist_mm: float
    post_brick_dist_mm: float
    pre_confidence: float
    post_confidence: float
    pre_samples_used: int | None
    post_samples_used: int | None
    pre_pose_source: str | None
    post_pose_source: str | None
    pre_observation_mode: str | None
    post_observation_mode: str | None
    post_reobserved: bool
    lost_visibility: bool = False
    recovered_visibility: bool = False
    recovery_mode: str | None = None
    recovery_inverse_acts: int | None = None
    pre_lite_required_frames: int | None = None
    post_lite_required_frames: int | None = None
    phase: str = "primary"
    source_trial: int | None = None


@dataclass
class ResetEffort:
    trial: int
    reset_act: int
    cmd: str
    score_requested: int
    cmd_sent: str | None
    pwm: int | None
    power: float | None
    duration_ms: int
    pre_x_mm: float
    post_x_mm: float
    raw_delta_mm: float
    signed_cmd_delta_mm: float
    cmd_delta_mm: float
    wrong_way: bool
    pre_brick_dist_mm: float
    post_brick_dist_mm: float
    pre_confidence: float
    post_confidence: float
    pre_pose_source: str | None
    post_pose_source: str | None
    post_observation_mode: str | None
    phase: str = "primary"
    source_trial: int | None = None


def log_line(message: str) -> None:
    print(str(message), flush=True)


def _supports_ansi_color() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _colorize(text: str, color_code: str) -> str:
    if not _supports_ansi_color():
        return str(text)
    return f"{str(color_code)}{str(text)}\033[0m"


def _highlight_number_text(text: str) -> str:
    return _colorize(str(text), ANSI_CYAN_BRIGHT)


def _highlight_mm(value: float, *, signed: bool = False) -> str:
    number = float(value)
    fmt = f"{number:+.2f}mm" if signed else f"{abs(number):.2f}mm"
    return _highlight_number_text(fmt)


def _highlight_duration_ms(duration_ms: int) -> str:
    return _highlight_number_text(f"{int(duration_ms)}ms")


def _highlight_turn_letter(cmd: str) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == "l":
        return _colorize("L", ANSI_BLUE_BRIGHT)
    return _colorize("R", ANSI_MAGENTA_BRIGHT)


def _highlight_side_text(side: str) -> str:
    side_key = str(side or "").strip().lower()
    if side_key == "left":
        return _colorize("left", ANSI_BLUE_BRIGHT)
    if side_key == "right":
        return _colorize("right", ANSI_MAGENTA_BRIGHT)
    if side_key == "center":
        return _colorize("centered", ANSI_YELLOW_BRIGHT)
    return str(side)


def _highlight_progress_text(status: str) -> str:
    status_key = str(status or "").strip().lower()
    if status_key == "closer":
        return _colorize("closer to x=0", ANSI_GREEN_BRIGHT)
    if status_key == "further":
        return _colorize("further from x=0", ANSI_RED_BRIGHT)
    return _colorize("still the same distance from x=0", ANSI_YELLOW_BRIGHT)


def _coerce_float(value, fallback=None):
    return shared_coerce_float(value, fallback)


def _coerce_int(value, fallback=None):
    return shared_coerce_int(value, fallback)


def _normalize_cmd(value: str, *, allow_auto: bool = False) -> str:
    text = str(value or "").strip().lower()
    if allow_auto and text in ("auto", "center"):
        return "auto"
    if text not in ("l", "r"):
        raise ValueError("Allowed x-axis commands are only 'l', 'r', 'auto', or 'center'.")
    return text


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


def _aggregate_pose_samples(poses: list[dict]) -> dict | None:
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


def read_pose(
    vision,
    world,
    *,
    samples: int = OBSERVE_SAMPLES_DEFAULT,
    timeout_s: float = OBSERVE_TIMEOUT_S,
    min_sample_time: float | None = None,
    min_samples_required: int | None = None,
) -> dict | None:
    poses = []
    start_t = time.time()
    step_label = _world_step_label(world)
    target_samples = max(1, int(samples))
    required_samples = target_samples if min_samples_required is None else max(1, int(min_samples_required))
    deadline = float(start_t) + float(timeout_s)
    if min_sample_time is not None:
        try:
            deadline = max(float(deadline), float(min_sample_time) + float(timeout_s))
        except (TypeError, ValueError):
            pass
    while len(poses) < int(target_samples) and time.time() < float(deadline):
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
        if len(poses) < int(target_samples):
            time.sleep(OBSERVE_SLEEP_S)
    if len(poses) < int(required_samples):
        return None
    return _aggregate_pose_samples(poses)


def _observe_pose_with_reobserve(
    *,
    vision,
    world,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None = None,
    hold_s: float = REOBSERVE_HOLD_S,
    reobserve_rounds: int = REOBSERVE_ROUNDS,
    relaxed_timeout_s: float = RELAXED_OBSERVE_TIMEOUT_S,
) -> tuple[dict | None, dict]:
    target_samples = max(1, int(samples))
    strict_pose = read_pose(
        vision,
        world,
        samples=target_samples,
        timeout_s=float(timeout_s),
        min_sample_time=min_sample_time,
        min_samples_required=target_samples,
    )
    if strict_pose is not None:
        return strict_pose, {
            "mode": "primary_full",
            "reobserved": False,
        }

    rounds = max(1, int(reobserve_rounds))
    for round_idx in range(1, rounds + 1):
        if float(hold_s) > 0.0:
            time.sleep(float(hold_s))
        relaxed_pose = read_pose(
            vision,
            world,
            samples=target_samples,
            timeout_s=max(float(timeout_s), float(relaxed_timeout_s)),
            min_sample_time=None,
            min_samples_required=1,
        )
        if relaxed_pose is None:
            if round_idx < rounds:
                log_line(
                    f"[CALIBRATE_X] Observation hold/reobserve {round_idx}/{rounds}: still no usable pose."
                )
            continue

        mode = (
            "hold_reobserve_full"
            if int(relaxed_pose.get("samples_used") or 0) >= int(target_samples)
            else "hold_reobserve_partial"
        )
        if int(relaxed_pose.get("samples_used") or 0) < int(target_samples):
            confirm_pose = read_pose(
                vision,
                world,
                samples=target_samples,
                timeout_s=max(1.0, float(timeout_s)),
                min_sample_time=None,
                min_samples_required=1,
            )
            if confirm_pose is not None and int(confirm_pose.get("samples_used") or 0) >= int(relaxed_pose.get("samples_used") or 0):
                relaxed_pose = confirm_pose
                mode = (
                    "hold_reobserve_confirmed_full"
                    if int(relaxed_pose.get("samples_used") or 0) >= int(target_samples)
                    else "hold_reobserve_confirmed_partial"
                )
        log_line(
            f"[CALIBRATE_X] Observation rescue: accepted {int(relaxed_pose.get('samples_used') or 0)}/{int(target_samples)} samples "
            f"via {mode}."
        )
        return relaxed_pose, {
            "mode": mode,
            "reobserved": True,
        }

    return None, {
        "mode": "unavailable",
        "reobserved": True,
    }


def _command_delta_mm(cmd: str, pre_x_mm: float, post_x_mm: float) -> float:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == _x_cmd_for_positive_motion():
        return float(post_x_mm) - float(pre_x_mm)
    return float(pre_x_mm) - float(post_x_mm)


def _travel_distance_mm(pre_x_mm: float, post_x_mm: float) -> float:
    return abs(float(post_x_mm) - float(pre_x_mm))


def _movement_metrics(cmd: str, pre_x_mm: float, post_x_mm: float) -> dict:
    raw_delta_mm = float(post_x_mm) - float(pre_x_mm)
    signed_cmd_delta_mm = _command_delta_mm(cmd, pre_x_mm, post_x_mm)
    travel_distance_mm = _travel_distance_mm(pre_x_mm, post_x_mm)
    return {
        "raw_delta_mm": float(raw_delta_mm),
        "signed_cmd_delta_mm": float(signed_cmd_delta_mm),
        "cmd_delta_mm": float(travel_distance_mm),
        "wrong_way": bool(float(signed_cmd_delta_mm) < 0.0),
    }


def _wrong_way_reason_text(
    *,
    pre_x_mm: float,
    post_x_mm: float,
    target_x_mm: float,
) -> str:
    travel_distance_mm = _travel_distance_mm(pre_x_mm, post_x_mm)
    near_center = (
        min(
            abs(float(pre_x_mm) - float(target_x_mm)),
            abs(float(post_x_mm) - float(target_x_mm)),
        )
        <= max(1.0, float(travel_distance_mm))
    )
    tiny_motion = float(travel_distance_mm) <= 1.0
    if near_center and tiny_motion:
        return "Likely because x_axis was already near 0 and the measured move was tiny, so vision jitter or settle noise can flip the direction label."
    if near_center:
        return "Likely because x_axis was already near 0, so a small overshoot or settle jitter can flip the direction label."
    if tiny_motion:
        return "Likely because the measured move was tiny, so vision jitter or settle noise can flip the direction label."
    return "Likely because of vision jitter, settle timing, or a small overshoot."


def _inverse_cmd(cmd: str | None) -> str | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "l":
        return "r"
    if cmd_key == "r":
        return "l"
    return None


def _x_cmd_for_positive_motion() -> str:
    cmd = _normalize_cmd(X_AXIS_POSITIVE_CMD_DEFAULT, allow_auto=False)
    if X_AXIS_INVERT:
        cmd = _inverse_cmd(cmd) or cmd
    return cmd


def _x_cmd_for_negative_motion() -> str:
    cmd = str(_inverse_cmd(_x_cmd_for_positive_motion()) or "r")
    return cmd


def _auto_cmd_for_x(
    curr_x_mm: float,
    *,
    center_x_mm: float = 0.0,
) -> str:
    if float(curr_x_mm) > float(center_x_mm):
        return _x_cmd_for_negative_motion()
    return _x_cmd_for_positive_motion()


def _turn_label_for_cmd(cmd: str) -> str:
    return "left_turn" if _normalize_cmd(cmd, allow_auto=False) == "l" else "right_turn"


def _turn_hotkey_for_cmd(cmd: str) -> str:
    return "q" if _normalize_cmd(cmd, allow_auto=False) == "l" else "e"


def _turn_hotkey_profile(cmd: str, score: int) -> dict | None:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key not in ("l", "r"):
        return None
    hotkey = _turn_hotkey_for_cmd(cmd_key)
    rows = HOTKEY_SPEED_SCORES if isinstance(HOTKEY_SPEED_SCORES, dict) else {}
    row = rows.get(hotkey)
    if not isinstance(row, dict):
        return None
    row_cmd = str(row.get("cmd") or "").strip().lower()
    if row_cmd != cmd_key:
        return None
    try:
        row_score = normalize_speed_score(row.get("score"))
    except Exception:
        return None
    if int(row_score) != int(normalize_speed_score(score)):
        return None

    pwm = None
    power = None
    try:
        pwm = clamp_pwm(int(round(float(row.get("pwm")))))
    except (TypeError, ValueError):
        pwm = None
    if pwm is None or int(pwm) <= 0:
        try:
            power_raw = float(row.get("power"))
        except (TypeError, ValueError):
            power_raw = None
        if power_raw is not None and float(power_raw) > 0.0:
            pwm_from_power = power_to_pwm(float(power_raw))
            if pwm_from_power is not None:
                pwm = clamp_pwm(int(pwm_from_power))
    if pwm is not None and int(pwm) > 0:
        power = pwm_to_power(int(pwm))
        if power is None:
            power = 0.0

    return {
        "hotkey": str(hotkey),
        "cmd": str(cmd_key),
        "score": int(row_score),
        "pwm": None if pwm is None or int(pwm) <= 0 else int(pwm),
        "power": None if power is None else float(power),
    }


def _plot_series_phase(kind: str | None = None) -> str:
    return shared_plot_series_phase(kind)


def _plot_color_for_cmd(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if repeat_status == "fail":
        return str(PLOT_REPEAT_FAIL_COLOR_BY_CMD.get(cmd_key) or "#ff1493")
    phase = _plot_series_phase(kind)
    if phase == "repeat":
        return str(PLOT_REPEAT_COLOR_BY_CMD.get(cmd_key) or "#5e81ac")
    return str(PLOT_COLOR_BY_CMD.get(cmd_key) or "#4c566a")


def _plot_series_key(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    phase = _plot_series_phase(kind)
    if phase == "primary":
        return str(cmd_key)
    if repeat_status == "fail":
        return f"{cmd_key}:repeat_fail"
    return f"{cmd_key}:repeat"


def _plot_series_label(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    phase = _plot_series_phase(kind)
    if repeat_status == "fail":
        return "Repeat fail"
    if phase == "repeat":
        return "Repeat left" if cmd_key == "l" else "Repeat right"
    return _turn_label_for_cmd(cmd_key)


def _plot_offsets(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    return shared_plot_offsets(xs, ys)


def _coerce_finite_float(value) -> float | None:
    return shared_coerce_finite_float(value)


def _plot_title_text(brick_distances_mm: list[float]) -> str:
    observed = [float(value) for value in brick_distances_mm if _coerce_finite_float(value) is not None]
    title_distance_mm = _coerce_finite_float(REFERENCE_BRICK_DISTANCE_MM)
    if title_distance_mm is None and observed:
        title_distance_mm = float(statistics.median(observed))
    if title_distance_mm is None:
        title_distance_mm = float(PLOT_TITLE_BRICK_DISTANCE_MM_DEFAULT)
    return f"X Calibration at {int(round(float(title_distance_mm)))}mm"


def _trial_label_text(
    trial_idx: int,
    trials_planned: int,
    *,
    phase: str = "primary",
    source_trial: int | None = None,
) -> str:
    return shared_trial_label_text(
        trial_idx,
        trials_planned,
        phase=phase,
        source_trial=source_trial,
    )


def _center_target_status_line(x_mm: float, *, target_x_mm: float) -> str:
    error_mm = float(x_mm) - float(target_x_mm)
    if abs(float(error_mm)) <= 0.5:
        status_text = _colorize("Near target", "\033[92m")
    else:
        status_text = _colorize("Off target", "\033[93m")
    return (
        f"x_axis: {float(x_mm):+.2f}. {status_text} center target "
        f"({float(target_x_mm):+.2f}mm; error {float(error_mm):+.2f}mm)."
    )


def _relative_side_of_brick(pre_x_mm: float, *, target_x_mm: float) -> str:
    error_mm = float(pre_x_mm) - float(target_x_mm)
    if error_mm > 0.0:
        return "left"
    if error_mm < 0.0:
        return "right"
    return "center"


def _distance_progress_status(pre_x_mm: float, post_x_mm: float, *, target_x_mm: float) -> tuple[float, str]:
    pre_err_mm = abs(float(pre_x_mm) - float(target_x_mm))
    post_err_mm = abs(float(post_x_mm) - float(target_x_mm))
    delta_mm = abs(float(post_err_mm) - float(pre_err_mm))
    if post_err_mm + 1e-9 < pre_err_mm:
        return float(delta_mm), "closer"
    if post_err_mm > pre_err_mm + 1e-9:
        return float(delta_mm), "further"
    return float(delta_mm), "unchanged"


def _build_duration_schedule(
    *,
    trials: int | None,
    min_duration_ms: int,
    max_duration_ms: int,
    duration_step_ms: int = DURATION_STEP_MS_DEFAULT,
) -> list[int]:
    return build_linear_duration_schedule(
        trials=trials,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        duration_step_ms=duration_step_ms,
    )


def _primary_trial_cmd_sequence() -> tuple[str, ...]:
    return tuple(_normalize_cmd(cmd, allow_auto=False) for cmd in PRIMARY_TRIAL_CMD_SEQUENCE_DEFAULT)


def _build_trial_plan(
    *,
    durations_ms: list[int],
    trials: int | None = None,
) -> list[dict]:
    return build_repeated_trial_plan(
        durations_ms=durations_ms,
        cmd_sequence=_primary_trial_cmd_sequence(),
        normalize_cmd=lambda value: _normalize_cmd(value, allow_auto=False),
        trials=trials,
    )


def _planned_durations_ms(trial_plan: list[dict]) -> list[int]:
    return shared_planned_durations_ms(trial_plan)


def _planned_action_meta(cmd: str, score: int, duration_override_ms: int) -> dict:
    power, pwm, score_used, duration_ms = speed_power_pwm_for_cmd(cmd, score)
    hotkey_profile = _turn_hotkey_profile(cmd, score)
    if isinstance(hotkey_profile, dict):
        if hotkey_profile.get("power") is not None:
            power = float(hotkey_profile["power"])
        if hotkey_profile.get("pwm") is not None:
            pwm = int(hotkey_profile["pwm"])
    if duration_override_ms is not None and int(duration_override_ms) > 0:
        duration_ms = int(duration_override_ms)
    duration_ms = min(int(duration_ms), int(DURATION_CEILING_MS))
    return {
        "power": float(power),
        "pwm": int(pwm),
        "score_model": int(score_used),
        "duration_ms": int(duration_ms),
        "hotkey": str(hotkey_profile.get("hotkey")).upper() if isinstance(hotkey_profile, dict) else None,
    }


def _send_fixed_score_command(
    *,
    robot,
    world,
    step: str,
    cmd: str,
    score: int,
    duration_override_ms: int,
) -> dict | None:
    duration_override_ms = min(max(1, int(duration_override_ms)), int(DURATION_CEILING_MS))
    hotkey_profile = _turn_hotkey_profile(cmd, score)
    if isinstance(hotkey_profile, dict) and hotkey_profile.get("pwm") is not None and hotkey_profile.get("power") is not None:
        return send_robot_command_pwm(
            robot,
            world,
            step,
            cmd,
            float(hotkey_profile["power"]),
            int(hotkey_profile["pwm"]),
            int(duration_override_ms),
            speed_score=int(score),
            auto_mode=False,
            half_first_turn_pulse=False,
        )
    return send_robot_command(
        robot,
        world,
        step,
        cmd,
        speed=0.0,
        speed_score=int(score),
        duration_override_ms=int(duration_override_ms),
        half_first_turn_pulse=False,
    )


def _recovery_plan_step_line(step: dict, *, idx: int, total: int) -> str:
    cmd = _normalize_cmd(step.get("cmd"), allow_auto=False)
    undo_cmd = _normalize_cmd(step.get("undo_cmd"), allow_auto=False)
    score = max(1, int(_coerce_int(step.get("score"), 1) or 1))
    duration_ms = max(1, int(_coerce_int(step.get("duration_ms"), 1) or 1))
    return (
        f"[RECOVERY]   Act {int(idx)}/{int(total)}: {_turn_label_for_cmd(cmd)} "
        f"score={int(score)}% duration={int(duration_ms)}ms "
        f"(undo {_turn_label_for_cmd(undo_cmd)})"
    )


def _recover_visibility(
    *,
    vision,
    world,
    robot,
    recent_acts,
    max_acts: int = RECOVERY_MAX_INVERSE_ACTS,
) -> tuple[dict | None, dict]:
    history = list(recent_acts)[-int(max_acts):]
    if not history:
        return None, {"mode": "inverse_unavailable", "inverse_acts": 0}

    plan = []
    for row in reversed(history):
        inv = _inverse_cmd(row.get("cmd"))
        if inv is None:
            continue
        plan.append(
            {
                "cmd": str(inv),
                "undo_cmd": str(row.get("cmd") or ""),
                "duration_ms": _coerce_int(row.get("duration_ms"), 0),
                "score": _coerce_int(row.get("score_requested"), 1),
            }
        )
    if not plan:
        return None, {"mode": "inverse_unavailable", "inverse_acts": 0}

    log_line(f"[RECOVERY] Vision lost. Reversing last {len(plan)} act(s).")
    for idx, step in enumerate(plan, start=1):
        log_line(_recovery_plan_step_line(step, idx=idx, total=len(plan)))
    for idx, step in enumerate(plan, start=1):
        act_start_ts = time.time()
        action_meta = _send_fixed_score_command(
            robot=robot,
            world=world,
            step="CALIBRATE_X_RECOVER",
            cmd=str(step["cmd"]),
            score=int(step["score"] or 1),
            duration_override_ms=max(1, int(step["duration_ms"] or 1)),
        )
        if not isinstance(action_meta, dict):
            continue
        duration_ms_used = _coerce_int(action_meta.get("duration_ms"), step["duration_ms"])
        pose, observe_meta = _observe_pose_with_reobserve(
            vision=vision,
            world=world,
            samples=1,
            timeout_s=float(RECOVERY_OBSERVE_TIMEOUT_S),
            min_sample_time=act_start_ts + (float(duration_ms_used or 0) / 1000.0) + float(POST_ACT_SETTLE_S),
        )
        if pose is not None:
            log_line(f"[RECOVERY] Reacquired vision after {idx} inverse act(s).")
            return pose, {
                "mode": f"inverse_{str(observe_meta.get('mode') or 'unknown')}",
                "inverse_acts": int(idx),
            }
    log_line("[RECOVERY] Inverse recovery failed.")
    return None, {"mode": "inverse_failed", "inverse_acts": len(plan)}


def _attempt_recovery(
    *,
    vision,
    world,
    robot,
    recent_acts,
) -> tuple[dict | None, dict]:
    rounds = max(1, int(REOBSERVE_ROUNDS))
    for idx in range(1, rounds + 1):
        pose = read_pose(
            vision,
            world,
            samples=1,
            timeout_s=float(RECOVERY_OBSERVE_TIMEOUT_S),
            min_samples_required=1,
        )
        if pose is not None:
            if idx > 1:
                log_line(f"[RECOVERY] Reacquired vision by holding still (round {idx}/{rounds}).")
            return pose, {
                "mode": "hold_reobserve",
                "inverse_acts": 0,
            }
        if idx < rounds:
            log_line(f"[RECOVERY] Still not visible after hold/reobserve {idx}/{rounds}.")
    return _recover_visibility(
        vision=vision,
        world=world,
        robot=robot,
        recent_acts=recent_acts,
    )


def _recover_pose_for_trial(
    *,
    vision,
    world,
    robot,
    recent_acts,
    trial_idx: int,
    trials_requested: int,
    stage_label: str,
    trial_label: str | None = None,
) -> tuple[dict | None, dict]:
    display_label = str(trial_label or _trial_label_text(int(trial_idx), int(trials_requested)))
    log_line(
        f"[CALIBRATE_X] {display_label}: no visible brick {stage_label}. Attempting recovery."
    )
    pose, recovery_meta = _attempt_recovery(
        vision=vision,
        world=world,
        robot=robot,
        recent_acts=recent_acts,
    )
    if pose is None:
        return None, {
            "mode": "unavailable",
            "reobserved": True,
            "inverse_acts": 0,
        }
    recovery_mode = str((recovery_meta or {}).get("mode") or "unknown")
    log_line(
        f"[CALIBRATE_X] {display_label}: recovered visibility {stage_label} via {recovery_mode}."
    )
    return pose, {
        "mode": recovery_mode,
        "reobserved": True,
        "inverse_acts": _coerce_int((recovery_meta or {}).get("inverse_acts"), 0),
    }


def _diagnose_vision_loss_event(
    trial_label: str,
    cmd: str,
    pre_x_mm: float,
    center_target_x_mm: float,
    setup_score: int,
    duration_used_ms: int,
) -> None:
    """Log comprehensive diagnostics for a vision loss event after movement."""
    target_err_mm = float(pre_x_mm) - float(center_target_x_mm)
    
    log_line(
        f"[CALIBRATE_X_VISION_LOSS] {trial_label}: "
        f"We started from x_axis={pre_x_mm:+.2f}mm "
        f"(target {float(center_target_x_mm):+.2f}mm; error {float(target_err_mm):+.2f}mm) "
        f"so we did cmd={cmd.upper()} score={setup_score}% duration={duration_used_ms}ms "
        f"and then lost vision."
    )


def _run_trial_action(
    *,
    trial_idx: int,
    trials_planned: int,
    trial_label: str,
    cmd: str,
    duration_ms: int,
    phase: str,
    source_trial: int | None,
    action_step: str,
    plot_kind: str,
    vision,
    world,
    robot,
    recent_acts,
    setup_score: int,
    center_target_x_mm: float,
    observe_samples: int,
    observe_timeout_s: float,
    post_act_settle_s: float,
    plotter=None,
    initial_pre_pose: dict | None = None,
    initial_pre_obs_meta: dict | None = None,
    compare_to_distance: float | None = None,
) -> tuple[TrialResult | None, str | None]:
    phase_key = str(phase or "primary")
    abort_prefix = "repeat_" if phase_key == "repeat" else ""

    pre_pose = dict(initial_pre_pose) if isinstance(initial_pre_pose, dict) else None
    pre_obs_meta = dict(initial_pre_obs_meta) if isinstance(initial_pre_obs_meta, dict) else None
    if pre_pose is None:
        pre_pose, pre_obs_meta = _observe_pose_with_reobserve(
            vision=vision,
            world=world,
            samples=observe_samples,
            timeout_s=observe_timeout_s,
        )
    if pre_pose is None:
        pre_pose, pre_obs_meta = _recover_pose_for_trial(
            vision=vision,
            world=world,
            robot=robot,
            recent_acts=recent_acts,
            trial_idx=trial_idx,
            trials_requested=trials_planned,
            stage_label="before act",
            trial_label=trial_label,
        )
        if pre_pose is None:
            log_line(f"[CALIBRATE_X] {trial_label}: recovery failed before act. Aborting.")
            return None, f"{abort_prefix}pre_pose_unavailable_trial_{trial_idx}"
    if not isinstance(pre_obs_meta, dict):
        pre_obs_meta = {"mode": "unknown", "reobserved": False}

    act_plan = _planned_action_meta(cmd, setup_score, duration_ms)
    pre_x_mm = float(pre_pose.get("offset_x") or 0.0)
    pre_error_mm = float(pre_x_mm) - float(center_target_x_mm)
    pre_offset_mm = abs(float(pre_error_mm))
    side = _relative_side_of_brick(pre_x_mm, target_x_mm=float(center_target_x_mm))
    if side == "center":
        where_text = f"{_highlight_mm(pre_offset_mm)} {_highlight_side_text('center')} on the brick"
    else:
        where_text = f"{_highlight_mm(pre_offset_mm)} to the {_highlight_side_text(side)} of the brick"
    trial_plan_text = "this repeat trial will turn" if phase_key == "repeat" else "this planned trial will turn"
    log_line(
        f"[CALIBRATE_X] {trial_label}: I see that I'm {where_text}, "
        f"and {trial_plan_text} {_highlight_turn_letter(cmd)} {_highlight_number_text(f'{int(setup_score)}%')} "
        f"for {_highlight_duration_ms(int(act_plan['duration_ms']))}."
    )

    act_start_ts = time.time()
    action_meta = _send_fixed_score_command(
        robot=robot,
        world=world,
        step=str(action_step),
        cmd=cmd,
        score=int(setup_score),
        duration_override_ms=int(duration_ms),
    )
    if not isinstance(action_meta, dict):
        log_line(f"[CALIBRATE_X] {trial_label}: send failed. Aborting.")
        return None, f"{abort_prefix}send_failed_trial_{trial_idx}"

    duration_used_ms = _coerce_int(action_meta.get("duration_ms"), act_plan["duration_ms"])
    recent_acts.append(
        {
            "cmd": str(cmd),
            "duration_ms": int(duration_used_ms or 0),
            "score_requested": int(setup_score),
            "timestamp": time.time(),
        }
    )
    post_pose, post_obs_meta = _observe_pose_with_reobserve(
        vision=vision,
        world=world,
        samples=observe_samples,
        timeout_s=observe_timeout_s,
        min_sample_time=act_start_ts + (float(duration_used_ms or 0) / 1000.0) + float(post_act_settle_s),
    )
    lost_visibility = False
    recovered_visibility = False
    recovery_mode = None
    recovery_inverse_acts = 0
    if post_pose is None:
        lost_visibility = True
        # Log detailed diagnostics of what we were doing when vision was lost
        _diagnose_vision_loss_event(
            trial_label=trial_label,
            cmd=cmd,
            pre_x_mm=float(pre_pose["offset_x"]),
            center_target_x_mm=float(center_target_x_mm),
            setup_score=int(setup_score),
            duration_used_ms=int(duration_used_ms or 0),
        )
        post_pose, post_obs_meta = _recover_pose_for_trial(
            vision=vision,
            world=world,
            robot=robot,
            recent_acts=recent_acts,
            trial_idx=trial_idx,
            trials_requested=trials_planned,
            stage_label="after act",
            trial_label=trial_label,
        )
        if post_pose is None:
            log_line(f"[CALIBRATE_X] {trial_label}: recovery failed after act. Aborting.")
            return None, f"{abort_prefix}post_pose_unavailable_trial_{trial_idx}"
        recovered_visibility = True
        recovery_mode = str(post_obs_meta.get("mode") or "unknown")
        recovery_inverse_acts = _coerce_int(post_obs_meta.get("inverse_acts"), 0)

    post_x_mm = float(post_pose["offset_x"])
    movement = _movement_metrics(cmd, pre_x_mm, post_x_mm)
    raw_delta_mm = float(movement["raw_delta_mm"])
    signed_cmd_delta_mm = float(movement["signed_cmd_delta_mm"])
    cmd_delta_mm = float(movement["cmd_delta_mm"])
    wrong_way = bool(movement["wrong_way"])
    cmd_sent_effective = str(action_meta.get("cmd_sent") or cmd)
    source_trial_value = _coerce_int(source_trial, trial_idx)

    row = TrialResult(
        trial=int(trial_idx),
        duration_ms=int(duration_used_ms or 0),
        cmd=str(cmd),
        score_requested=int(setup_score),
        cmd_sent=str(cmd_sent_effective),
        pwm=_coerce_int(action_meta.get("pwm")),
        power=_coerce_float(action_meta.get("power")),
        pre_x_mm=pre_x_mm,
        post_x_mm=post_x_mm,
        raw_delta_mm=raw_delta_mm,
        signed_cmd_delta_mm=signed_cmd_delta_mm,
        cmd_delta_mm=cmd_delta_mm,
        wrong_way=bool(wrong_way),
        pre_dist_mm=float(pre_pose["dist"]),
        post_dist_mm=float(post_pose["dist"]),
        pre_brick_dist_mm=float(pre_pose["dist"]),
        post_brick_dist_mm=float(post_pose["dist"]),
        pre_confidence=float(pre_pose["confidence"]),
        post_confidence=float(post_pose["confidence"]),
        pre_samples_used=_coerce_int(pre_pose.get("samples_used")),
        post_samples_used=_coerce_int(post_pose.get("samples_used")),
        pre_pose_source=str(pre_pose.get("pose_source") or "unknown"),
        post_pose_source=str(post_pose.get("pose_source") or "unknown"),
        pre_observation_mode=str(pre_obs_meta.get("mode") or "unknown"),
        post_observation_mode=str(post_obs_meta.get("mode") or "unknown"),
        post_reobserved=bool(post_obs_meta.get("reobserved")),
        lost_visibility=bool(lost_visibility),
        recovered_visibility=bool(recovered_visibility),
        recovery_mode=recovery_mode,
        recovery_inverse_acts=_coerce_int(recovery_inverse_acts),
        pre_lite_required_frames=_coerce_int(pre_pose.get("lite_required_frames")),
        post_lite_required_frames=_coerce_int(post_pose.get("lite_required_frames")),
        phase=str(phase_key),
        source_trial=_coerce_int(source_trial_value),
    )

    goal_delta_mm, goal_status = _distance_progress_status(
        pre_x_mm,
        post_x_mm,
        target_x_mm=float(center_target_x_mm),
    )
    log_line(
        f"[CALIBRATE_X] {trial_label}: That act resulted in {_highlight_mm(goal_delta_mm)} difference "
        f"and I'm {_highlight_progress_text(goal_status)} "
        f"({_highlight_number_text(f'x_axis={post_x_mm:+.2f}mm')})."
    )

    repeat_status = None
    if phase_key == "repeat" and compare_to_distance is not None:
        delta_from_source = abs(float(cmd_delta_mm) - float(compare_to_distance))
        if delta_from_source > REPEAT_RESULT_ERROR_MARGIN_MM:
            repeat_status = "fail"
            log_line(
                f"[CALIBRATE_X] {trial_label}: repeat result differs by {delta_from_source:.2f}mm "
                f"(original: {compare_to_distance:.2f}mm, repeat: {cmd_delta_mm:.2f}mm, margin: ±{REPEAT_RESULT_ERROR_MARGIN_MM}mm)"
            )
        else:
            repeat_status = "success"

    if plotter is not None:
        plotter.add_point(
            duration_ms=int(duration_used_ms or 0),
            distance_mm=float(cmd_delta_mm),
            trial=int(trial_idx),
            cmd=str(cmd),
            kind=str(plot_kind),
            pre_brick_distance_mm=_coerce_finite_float(pre_pose.get("dist")),
            post_brick_distance_mm=_coerce_finite_float(post_pose.get("dist")),
            annotation_label=None,
            repeat_status=repeat_status,
        )
    return row, None


class LivePlot:
    def __init__(self, *, show_plot: bool, plot_path: Path | None):
        self._plot = CalibrationLivePlot(
            show_plot=show_plot,
            plot_path=plot_path,
            cmds=("l", "r"),
            normalize_cmd=lambda value: _normalize_cmd(value, allow_auto=False),
            plot_series_key=lambda cmd, kind=None, repeat_status=None: _plot_series_key(
                cmd,
                kind,
                repeat_status=repeat_status,
            ),
            plot_color=lambda cmd, kind=None, repeat_status=None: _plot_color_for_cmd(
                cmd,
                kind,
                repeat_status=repeat_status,
            ),
            plot_series_label=lambda cmd, kind=None, repeat_status=None: _plot_series_label(
                cmd,
                kind,
                repeat_status=repeat_status,
            ),
            plot_title=_plot_title_text,
            x_label="Distance Covered (mm)",
            y_label="Duration (ms)",
            title_font_size=int(PLOT_TITLE_FONT_SIZE),
            label_font_size=int(PLOT_LABEL_FONT_SIZE),
            tick_font_size=int(PLOT_TICK_FONT_SIZE),
            legend_font_size=int(PLOT_LEGEND_FONT_SIZE),
        )

    def add_point(
        self,
        *,
        duration_ms: int,
        distance_mm: float,
        trial: int,
        cmd: str,
        kind: str = "trial",
        pre_brick_distance_mm: float | None = None,
        post_brick_distance_mm: float | None = None,
        annotation_label: str | None = None,
        repeat_status: str | None = None,
    ) -> None:
        self._plot.add_point(
            duration_ms=duration_ms,
            distance_mm=distance_mm,
            trial=trial,
            cmd=cmd,
            kind=kind,
            pre_brick_distance_mm=pre_brick_distance_mm,
            post_brick_distance_mm=post_brick_distance_mm,
            annotation_label=annotation_label,
            repeat_status=repeat_status,
        )

    def finish(self) -> None:
        self._plot.finish()


def _write_results(path: Path, payload: dict) -> None:
    shared_write_results(path, payload)


def _observed_brick_distances_mm(
    *,
    trials: list[TrialResult],
    reset_efforts: list[ResetEffort],
) -> list[float]:
    return shared_observed_brick_distances_mm(trials=trials, reset_efforts=reset_efforts)


def _build_payload(
    *,
    config: dict,
    durations_ms: list[int],
    trials: list[TrialResult],
    reset_efforts: list[ResetEffort],
    status: str,
    abort_reason: str | None,
) -> dict:
    return build_shared_payload(
        source="calibrate_x",
        config=config,
        durations_ms=durations_ms,
        trials=trials,
        reset_efforts=reset_efforts,
        status=status,
        abort_reason=abort_reason,
    )


def _exit_as_script(exit_code: int) -> None:
    if sys.gettrace() is not None:
        return
    raise SystemExit(int(exit_code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal x-axis duration probe with live scatter updates.")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Optional primary-trial cap; default runs the full deterministic duration schedule.",
    )
    parser.add_argument(
        "--center-x-mm",
        type=float,
        default=X_AXIS_TARGET_MM_DEFAULT,
        help=f"X-axis center target used for status logging (default: {X_AXIS_TARGET_MM_DEFAULT}).",
    )
    parser.add_argument(
        "--vision",
        choices=["leia", "yolo", "aruco"],
        default="aruco",
        help="Which vision backend to use: aruco markers (default), leia edges, or yolo cyan bricks.",
    )

    parser.add_argument("--min-duration-ms", type=int, default=250, help="Minimum deterministic duration in ms (default: 250).")
    parser.add_argument("--max-duration-ms", type=int, default=500, help="Maximum deterministic duration in ms (default: 500).")
    parser.add_argument("--observe-samples", type=int, default=OBSERVE_SAMPLES_DEFAULT, help="Observation samples per pose; use 3 for 3-frame confidence (default: 3).")
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S, help=f"Observation timeout in seconds (default: {OBSERVE_TIMEOUT_S}).")
    parser.add_argument("--post-act-settle-s", type=float, default=POST_ACT_SETTLE_S, help=f"Extra wait after the act before re-observing (default: {POST_ACT_SETTLE_S}).")
    parser.add_argument("--show-plot", action="store_true", help="Open an interactive Matplotlib window and update it after each trial.")
    parser.add_argument("--plot-path", type=str, default=PLOT_FILE_DEFAULT, help="Optional PNG file to rewrite after each trial.")
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_DEFAULT, help="JSON output path (default: run-specific file in ./runs).")
    parser.add_argument("--reference-distance-mm", type=float, default=None, help="Assumed brick distance (mm) for this calibration set")
    parser.add_argument("--invert-x-axis", action="store_true", help="Invert l/r command mapping.")
    args = parser.parse_args()
    _ensure_run_dir()
    # Use a stable live JSON file in the Runs - aruco folder.
    if args.results_file is None:
        args.results_file = str(RUN_DIR / "calibrate_x_live.json")
    trials_requested = None if args.trials is None else max(1, int(args.trials))
    speed_score = normalize_speed_score(SPEED_SCORE_DEFAULT)
    center_x_mm = float(args.center_x_mm)
    min_duration_ms = max(1, int(args.min_duration_ms))
    max_duration_ms = max(min_duration_ms, int(args.max_duration_ms))
    duration_ceiling_ms = max(1, int(DURATION_CEILING_MS))
    if max_duration_ms > duration_ceiling_ms:
        log_line(
            f"[CALIBRATE_X] Clamping requested max_duration_ms={int(max_duration_ms)}ms "
            f"to ceiling {int(duration_ceiling_ms)}ms."
        )
        max_duration_ms = int(duration_ceiling_ms)
    if min_duration_ms > duration_ceiling_ms:
        log_line(
            f"[CALIBRATE_X] Clamping requested min_duration_ms={int(min_duration_ms)}ms "
            f"to ceiling {int(duration_ceiling_ms)}ms."
        )
        min_duration_ms = int(duration_ceiling_ms)
    max_duration_ms = max(int(min_duration_ms), int(max_duration_ms))
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    post_act_settle_s = max(0.0, float(args.post_act_settle_s))
    global X_AXIS_INVERT
    if args.invert_x_axis:
        X_AXIS_INVERT = True
    if args.reference_distance_mm is not None:
        global REFERENCE_BRICK_DISTANCE_MM
        REFERENCE_BRICK_DISTANCE_MM = float(args.reference_distance_mm)
    full_durations_ms = _build_duration_schedule(
        trials=None,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
    )
    full_trial_plan = _build_trial_plan(
        durations_ms=full_durations_ms,
        trials=None,
    )
    trial_plan = _build_trial_plan(
        durations_ms=full_durations_ms,
        trials=trials_requested,
    )
    primary_cmd_sequence = _primary_trial_cmd_sequence()
    durations_ms = _planned_durations_ms(trial_plan)
    if trials_requested is not None and int(trials_requested) > len(full_trial_plan):
        log_line(
            f"[CALIBRATE_X] Requested {int(trials_requested)} trial(s), but the deterministic "
            f"{int(min_duration_ms)}..{int(max_duration_ms)}ms sweep with per-duration "
            f"{'/'.join(str(cmd).upper() for cmd in primary_cmd_sequence)} only provides "
            f"{len(full_trial_plan)} planned trial(s)."
        )
    trials_planned = len(trial_plan)
    results_path = Path(args.results_file)
    plot_path = Path(args.plot_path) if args.plot_path else None

    config = {
        "trials": int(trials_planned),
        "requested_trials": None if trials_requested is None else int(trials_requested),
        "repeat_pass_enabled": True,
        "duration_ceiling_ms": int(duration_ceiling_ms),
        "speed_score": int(speed_score),
        "center_x_mm": float(center_x_mm),
        "min_duration_ms": int(min_duration_ms),
        "max_duration_ms": int(max_duration_ms),
        "duration_step_ms": int(DURATION_STEP_MS_DEFAULT),
        "x_axis_positive_cmd": str(_x_cmd_for_positive_motion()),
        "x_axis_negative_cmd": str(_x_cmd_for_negative_motion()),
        "observe_samples": int(observe_samples),
        "observe_timeout_s": float(observe_timeout_s),
        "post_act_settle_s": float(post_act_settle_s),
        "half_first_turn_pulse": False,
        "primary_trial_cmd_schedule": "per_duration_sequence",
        "primary_trial_cmd_sequence": [str(cmd).upper() for cmd in primary_cmd_sequence],
        "primary_trial_repetitions_per_direction": 2,
        "x_axis_center_target_mm": float(center_x_mm),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "brick_distance_source": str(BRICK_DISTANCE_SOURCE),
        "brick_distance_definition": str(BRICK_DISTANCE_DEFINITION),
    }
    if REFERENCE_BRICK_DISTANCE_MM is not None:
        config["reference_brick_distance_mm"] = float(REFERENCE_BRICK_DISTANCE_MM)

    log_line("[CALIBRATE_X] Starting x-axis duration probe.")
    log_line(
        f"[CALIBRATE_X] trials={trials_planned} score={int(speed_score)}% durations_ms={durations_ms} "
        f"observe_samples={observe_samples}"
    )
    log_line(
        f"[CALIBRATE_X] deterministic duration climb: start={int(min_duration_ms)}ms "
        f"stop={int(max_duration_ms)}ms step={int(DURATION_STEP_MS_DEFAULT)}ms."
    )
    log_line(
        f"[CALIBRATE_X] x-axis motion sign: {_turn_label_for_cmd(_x_cmd_for_positive_motion())} increases x_axis, "
        f"{_turn_label_for_cmd(_x_cmd_for_negative_motion())} decreases x_axis."
    )
    log_line(
        f"[CALIBRATE_X] repeat_pass=enabled; repeats are deferred until all {int(trials_planned)} "
        f"primary trial(s) finish."
    )
    log_line("[CALIBRATE_X] duration fidelity: exact requested turn duration is used; first-turn halving is disabled.")
    log_line(f"[CALIBRATE_X] center target: x_axis={float(center_x_mm):+.2f}mm.")
    log_line("[CALIBRATE_X] turn source: left turns follow hotkey Q; right turns follow hotkey E.")
    log_line(
        f"[CALIBRATE_X] primary trial command schedule: per duration run "
        f"{_turn_label_for_cmd(primary_cmd_sequence[0])}, "
        f"{_turn_label_for_cmd(primary_cmd_sequence[1])}, "
        f"{_turn_label_for_cmd(primary_cmd_sequence[2])}, "
        f"{_turn_label_for_cmd(primary_cmd_sequence[3])}."
    )
    if bool(args.show_plot):
        if _MATPLOTLIB_AVAILABLE:
            log_line("[CALIBRATE_X] Live plot enabled.")
        else:
            log_line("[CALIBRATE_X] Matplotlib unavailable; continuing without live plot.")
    if plot_path is not None:
        log_line(f"[CALIBRATE_X] Plot PNG will update at {plot_path}")

    plotter = LivePlot(show_plot=bool(args.show_plot), plot_path=plot_path)
    robot = None
    vision = None
    world = None
    recent_acts = deque(maxlen=32)
    trial_rows: list[TrialResult] = []
    reset_rows: list[ResetEffort] = []
    status = "completed"
    abort_reason = None

    try:
        world = WorldModel()
        world.step_state = StepState.ALIGN_BRICK
        world._post_action_observe_delay_s = 0.0
        robot = Robot()
        # instantiate vision according to the requested backend
        if args.vision == "yolo":
            if YoloBrickDetector is None:
                raise RuntimeError("YOLO detector module not installed; cannot use --vision yolo")
            vision = YoloBrickDetector(debug=False)
        elif args.vision == "aruco":
            # keep in case someone still wants marker calibration
            from helper_vision_aruco import ArucoBrickVision
            vision = ArucoBrickVision(debug=False)
        else:  # leia
            vision = LeiaVision(debug=False)

        for trial_idx, plan_step in enumerate(trial_plan, start=1):
            trial_label = _trial_label_text(trial_idx, trials_planned)
            cmd = str(plan_step.get("cmd") or "")
            duration_ms = max(1, int(_coerce_int(plan_step.get("duration_ms"), 1) or 1))
            log_line(
                f"[CALIBRATE_X] {trial_label}: scheduled cmd={str(cmd).upper()} "
                f"({_turn_label_for_cmd(cmd)}) for {_highlight_duration_ms(int(duration_ms))}."
            )

            row, trial_abort_reason = _run_trial_action(
                trial_idx=trial_idx,
                trials_planned=trials_planned,
                trial_label=trial_label,
                cmd=str(cmd),
                duration_ms=int(duration_ms),
                phase="primary",
                source_trial=trial_idx,
                action_step="CALIBRATE_X",
                plot_kind="trial",
                vision=vision,
                world=world,
                robot=robot,
                recent_acts=recent_acts,
                setup_score=int(speed_score),
                center_target_x_mm=float(center_x_mm),
                observe_samples=observe_samples,
                observe_timeout_s=observe_timeout_s,
                post_act_settle_s=post_act_settle_s,
                plotter=plotter,
            )
            if row is None:
                status = "aborted"
                abort_reason = str(trial_abort_reason or f"trial_failed_{trial_idx}")
                break
            if bool(row.wrong_way):
                wrong_way_reason = _wrong_way_reason_text(
                    pre_x_mm=float(row.pre_x_mm),
                    post_x_mm=float(row.post_x_mm),
                    target_x_mm=float(center_x_mm),
                )
                log_line(
                    f"[CALIBRATE_X] ⚠️  Trial {trial_idx}: wrong_way detected. "
                    f"Plotting it anyway. {wrong_way_reason}"
                )
            trial_rows.append(row)
            _write_results(
                results_path,
                _build_payload(
                    config=config,
                    durations_ms=durations_ms,
                    trials=trial_rows,
                    reset_efforts=reset_rows,
                    status=status,
                    abort_reason=abort_reason,
                ),
            )

        if status == "completed":
            repeat_plan = [row for row in trial_rows if str(getattr(row, "phase", "primary")) != "repeat"]
            log_line(
                f"[CALIBRATE_X] Primary pass complete. Starting repeat pass over "
                f"{len(repeat_plan)} recorded trial(s)."
            )
            for repeat_idx, source_row in enumerate(repeat_plan, start=1):
                repeat_label = _trial_label_text(
                    repeat_idx,
                    len(repeat_plan),
                    phase="repeat",
                    source_trial=_coerce_int(source_row.source_trial, source_row.trial),
                )
                repeat_row, repeat_abort_reason = _run_trial_action(
                    trial_idx=repeat_idx,
                    trials_planned=len(repeat_plan),
                    trial_label=repeat_label,
                    cmd=str(source_row.cmd),
                    duration_ms=int(source_row.duration_ms),
                    phase="repeat",
                    source_trial=_coerce_int(source_row.source_trial, source_row.trial),
                    action_step="CALIBRATE_X_REPEAT",
                    plot_kind="repeat",
                    vision=vision,
                    world=world,
                    robot=robot,
                    recent_acts=recent_acts,
                    setup_score=int(speed_score),
                    center_target_x_mm=float(center_x_mm),
                    observe_samples=observe_samples,
                    observe_timeout_s=observe_timeout_s,
                    post_act_settle_s=post_act_settle_s,
                    plotter=plotter,
                    compare_to_distance=float(source_row.cmd_delta_mm),
                )
                if repeat_row is None:
                    status = "aborted"
                    abort_reason = str(repeat_abort_reason or f"repeat_trial_failed_{repeat_idx}")
                    break
                if bool(repeat_row.wrong_way):
                    wrong_way_reason = _wrong_way_reason_text(
                        pre_x_mm=float(repeat_row.pre_x_mm),
                        post_x_mm=float(repeat_row.post_x_mm),
                        target_x_mm=float(center_x_mm),
                    )
                    log_line(
                        f"[CALIBRATE_X] ⚠️  Repeat {repeat_idx}: wrong_way detected. "
                        f"Plotting it anyway. {wrong_way_reason}"
                    )
                trial_rows.append(repeat_row)
                _write_results(
                    results_path,
                    _build_payload(
                        config=config,
                        durations_ms=durations_ms,
                        trials=trial_rows,
                        reset_efforts=reset_rows,
                        status=status,
                        abort_reason=abort_reason,
                    ),
                )
    except KeyboardInterrupt:
        status = "interrupted"
        abort_reason = "keyboard_interrupt"
        log_line("[CALIBRATE_X] Interrupted by user.")
    finally:
        _write_results(
            results_path,
            _build_payload(
                config=config,
                durations_ms=durations_ms,
                trials=trial_rows,
                reset_efforts=reset_rows,
                status=status,
                abort_reason=abort_reason,
            ),
        )
        plotter.finish()
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

    log_line(f"[CALIBRATE_X] Wrote results to {results_path}")
    if plot_path is not None:
        log_line(f"[CALIBRATE_X] Updated plot at {plot_path}")
    if status != "completed":
        detail = f" reason={abort_reason}" if abort_reason else ""
        log_line(f"[CALIBRATE_X] Finished with status={status}{detail}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    _exit_as_script(main())
