#!/usr/bin/env python3
"""
Minimal distance-gap duration probe.

Observe the current brick distance, send one forward/backward act at a fixed
speed score and deterministic duration, observe again, and plot total distance
traveled in mm against duration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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
    observe_pose_with_reobserve as shared_observe_pose_with_reobserve,
    planned_durations_ms as shared_planned_durations_ms,
    read_pose as shared_read_pose,
    trial_label_text as shared_trial_label_text,
    write_results as shared_write_results,
)
from helper_robot_control import Robot
from helper_vision_leia import LeiaVision

try:
    from brick_detector_yolo import BrickDetector as YoloBrickDetector
except ImportError:
    YoloBrickDetector = None

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
POST_ACT_SETTLE_S = 0.10
OBSERVE_SAMPLES_DEFAULT = 3
DIST_TARGET_MM_DEFAULT: float | None = None
SPEED_SCORE_DEFAULT = 5
DURATION_CEILING_MS = 900
DURATION_STEP_MS_DEFAULT = 20
DIST_POSITIVE_CMD_DEFAULT = "b"
DIST_INVERT = False
PRIMARY_TRIAL_CMD_SEQUENCE_DEFAULT = ("f", "b", "f", "b")
REFERENCE_BRICK_DISTANCE_MM: float | None = None
PLOT_TITLE_DISTANCE_MM_DEFAULT = 166.0
PLOT_COLOR_BY_CMD = {
    "f": "#1f77b4",
    "b": "#ff7f0e",
}
PLOT_REPEAT_COLOR_BY_CMD = {
    "f": "#00dd77",
    "b": "#dd00dd",
}
PLOT_REPEAT_FAIL_COLOR_BY_CMD = {
    "f": "#ff1493",
    "b": "#ff69b4",
}
REPEAT_RESULT_ERROR_MARGIN_MM = 2.0
MIN_LITE_UNIQUE_FRAMES = 3
REOBSERVE_HOLD_S = 0.12
REOBSERVE_ROUNDS = 2
RELAXED_OBSERVE_TIMEOUT_S = 2.8
RECOVERY_MAX_INVERSE_ACTS = 5
RECOVERY_OBSERVE_TIMEOUT_S = 1.0
RESULTS_FILE_DEFAULT: str | None = None
PLOT_FILE_DEFAULT: str | None = None
RUN_DIR = Path("Runs - aruco")
BRICK_DISTANCE_SOURCE = "vision.dist"
BRICK_DISTANCE_DEFINITION = "Camera-to-brick distance reported by vision at observation time (mm)."
PLOT_TITLE_FONT_SIZE = 10
PLOT_LABEL_FONT_SIZE = 9
PLOT_TICK_FONT_SIZE = 8
PLOT_LEGEND_FONT_SIZE = 8
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
    pre_dist_mm: float
    post_dist_mm: float
    raw_delta_mm: float
    signed_cmd_delta_mm: float
    cmd_delta_mm: float
    wrong_way: bool
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


def _cleanup_old_run_files():
    cleanup_old_run_files(
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_x_live.png",
            "calibrate_y_live.json",
            "calibrate_y_live.png",
            "calibrate_dist_live.json",
            "calibrate_dist_live.png",
        }
    )


def _ensure_run_dir():
    ensure_run_dir(
        run_dir=RUN_DIR,
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_x_live.png",
            "calibrate_y_live.json",
            "calibrate_y_live.png",
            "calibrate_dist_live.json",
            "calibrate_dist_live.png",
        },
    )

def _coerce_float(value, fallback=None):
    return shared_coerce_float(value, fallback)


def _coerce_int(value, fallback=None):
    return shared_coerce_int(value, fallback)


def _coerce_finite_float(value) -> float | None:
    return shared_coerce_finite_float(value)


def _normalize_cmd(value: str, *, allow_auto: bool = False) -> str:
    text = str(value or "").strip().lower()
    if allow_auto and text in ("auto", "target"):
        return "auto"
    if text not in ("f", "b"):
        raise ValueError("Allowed dist commands are only 'f', 'b', 'auto', or 'target'.")
    return text


def read_pose(
    vision,
    world,
    *,
    samples: int = OBSERVE_SAMPLES_DEFAULT,
    timeout_s: float = OBSERVE_TIMEOUT_S,
    min_sample_time: float | None = None,
    min_samples_required: int | None = None,
) -> dict | None:
    return shared_read_pose(
        vision,
        world,
        samples=samples,
        timeout_s=timeout_s,
        min_sample_time=min_sample_time,
        min_samples_required=min_samples_required,
        observe_sleep_s=OBSERVE_SLEEP_S,
        fallback_step_label="ALIGN_BRICK",
        update_world_from_vision=update_world_from_vision,
        latest_unique_smoothed_frames=telemetry_latest_unique_smoothed_frames,
        average_smoothed_frames=telemetry_average_smoothed_frames,
        lite_gate_unique_frames=lite_gate_unique_frames,
        min_lite_unique_frames=MIN_LITE_UNIQUE_FRAMES,
    )


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
    return shared_observe_pose_with_reobserve(
        read_pose_fn=read_pose,
        log_fn=log_line,
        log_prefix="[CALIBRATE_DIST]",
        vision=vision,
        world=world,
        samples=samples,
        timeout_s=timeout_s,
        min_sample_time=min_sample_time,
        hold_s=hold_s,
        reobserve_rounds=reobserve_rounds,
        relaxed_timeout_s=relaxed_timeout_s,
    )


def _inverse_cmd(cmd: str | None) -> str | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "f":
        return "b"
    if cmd_key == "b":
        return "f"
    return None


def _dist_cmd_for_positive_motion() -> str:
    cmd = _normalize_cmd(DIST_POSITIVE_CMD_DEFAULT, allow_auto=False)
    if DIST_INVERT:
        cmd = _inverse_cmd(cmd) or cmd
    return cmd


def _dist_cmd_for_negative_motion() -> str:
    return str(_inverse_cmd(_dist_cmd_for_positive_motion()) or "f")


def _command_delta_mm(cmd: str, pre_dist_mm: float, post_dist_mm: float) -> float:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == _dist_cmd_for_positive_motion():
        return float(post_dist_mm) - float(pre_dist_mm)
    return float(pre_dist_mm) - float(post_dist_mm)


def _movement_metrics(cmd: str, pre_dist_mm: float, post_dist_mm: float) -> dict:
    raw_delta_mm = float(post_dist_mm) - float(pre_dist_mm)
    signed_cmd_delta_mm = _command_delta_mm(cmd, pre_dist_mm, post_dist_mm)
    travel_distance_mm = abs(float(post_dist_mm) - float(pre_dist_mm))
    return {
        "raw_delta_mm": float(raw_delta_mm),
        "signed_cmd_delta_mm": float(signed_cmd_delta_mm),
        "cmd_delta_mm": float(travel_distance_mm),
        "wrong_way": bool(float(signed_cmd_delta_mm) < 0.0),
    }


def _drive_label_for_cmd(cmd: str) -> str:
    return "forward" if _normalize_cmd(cmd, allow_auto=False) == "f" else "backward"


def _highlight_drive_letter(cmd: str) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == "f":
        return _colorize("F", ANSI_BLUE_BRIGHT)
    return _colorize("B", ANSI_MAGENTA_BRIGHT)


def _target_distance_status(pre_dist_mm: float, *, target_dist_mm: float) -> tuple[float, str]:
    error_mm = float(pre_dist_mm) - float(target_dist_mm)
    if error_mm > 0.0:
        return abs(float(error_mm)), "too far from the brick"
    if error_mm < 0.0:
        return abs(float(error_mm)), "too close to the brick"
    return 0.0, "at the target distance"


def _highlight_progress_text(status: str) -> str:
    status_key = str(status or "").strip().lower()
    if status_key == "closer":
        return _colorize("closer to the target distance", ANSI_GREEN_BRIGHT)
    if status_key == "further":
        return _colorize("further from the target distance", ANSI_RED_BRIGHT)
    return _colorize("still the same distance from the target", ANSI_YELLOW_BRIGHT)


def _distance_progress_status(pre_dist_mm: float, post_dist_mm: float, *, target_dist_mm: float) -> tuple[float, str]:
    pre_err_mm = abs(float(pre_dist_mm) - float(target_dist_mm))
    post_err_mm = abs(float(post_dist_mm) - float(target_dist_mm))
    delta_mm = abs(float(post_err_mm) - float(pre_err_mm))
    if post_err_mm + 1e-9 < pre_err_mm:
        return float(delta_mm), "closer"
    if post_err_mm > pre_err_mm + 1e-9:
        return float(delta_mm), "further"
    return float(delta_mm), "unchanged"


def _plot_color_for_cmd(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    phase = "repeat" if str(kind or "").strip().lower() == "repeat" else "primary"
    if repeat_status == "fail":
        return str(PLOT_REPEAT_FAIL_COLOR_BY_CMD.get(cmd_key) or "#ff1493")
    if phase == "repeat":
        return str(PLOT_REPEAT_COLOR_BY_CMD.get(cmd_key) or "#5e81ac")
    return str(PLOT_COLOR_BY_CMD.get(cmd_key) or "#4c566a")


def _plot_series_key(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    phase = "repeat" if str(kind or "").strip().lower() == "repeat" else "primary"
    if phase == "primary":
        return str(cmd_key)
    if repeat_status == "fail":
        return f"{cmd_key}:repeat_fail"
    return f"{cmd_key}:repeat"


def _plot_series_label(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    phase = "repeat" if str(kind or "").strip().lower() == "repeat" else "primary"
    if repeat_status == "fail":
        return "Repeat fail"
    if phase == "repeat":
        return "Repeat forward" if cmd_key == "f" else "Repeat backward"
    return _drive_label_for_cmd(cmd_key)


def _plot_title_text(brick_distances_mm: list[float]) -> str:
    observed = [float(value) for value in brick_distances_mm if _coerce_finite_float(value) is not None]
    title_distance_mm = _coerce_finite_float(REFERENCE_BRICK_DISTANCE_MM)
    if title_distance_mm is None and observed:
        title_distance_mm = float(statistics.median(observed))
    if title_distance_mm is None:
        title_distance_mm = float(PLOT_TITLE_DISTANCE_MM_DEFAULT)
    return f"Distance Calibration at {int(round(float(title_distance_mm)))}mm"


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
    if duration_override_ms is not None and int(duration_override_ms) > 0:
        duration_ms = int(duration_override_ms)
    duration_ms = min(int(duration_ms), int(DURATION_CEILING_MS))
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
    duration_override_ms: int,
) -> dict | None:
    duration_override_ms = min(max(1, int(duration_override_ms)), int(DURATION_CEILING_MS))
    return send_robot_command(
        robot,
        world,
        step,
        cmd,
        speed=0.0,
        speed_score=int(score),
        duration_override_ms=int(duration_override_ms),
    )


def _recovery_plan_step_line(step: dict, *, idx: int, total: int) -> str:
    cmd = _normalize_cmd(step.get("cmd"), allow_auto=False)
    undo_cmd = _normalize_cmd(step.get("undo_cmd"), allow_auto=False)
    score = max(1, int(_coerce_int(step.get("score"), 1) or 1))
    duration_ms = max(1, int(_coerce_int(step.get("duration_ms"), 1) or 1))
    return (
        f"[RECOVERY]   Act {int(idx)}/{int(total)}: {_drive_label_for_cmd(cmd)} "
        f"score={int(score)}% duration={int(duration_ms)}ms "
        f"(undo {_drive_label_for_cmd(undo_cmd)})"
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
            step="CALIBRATE_DIST_RECOVER",
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
    log_line(f"[CALIBRATE_DIST] {display_label}: no visible brick {stage_label}. Attempting recovery.")
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
        f"[CALIBRATE_DIST] {display_label}: recovered visibility {stage_label} via {recovery_mode}."
    )
    return pose, {
        "mode": recovery_mode,
        "reobserved": True,
        "inverse_acts": _coerce_int((recovery_meta or {}).get("inverse_acts"), 0),
    }


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
    target_dist_mm: float,
    observe_samples: int,
    observe_timeout_s: float,
    post_act_settle_s: float,
    plotter=None,
    compare_to_distance: float | None = None,
) -> tuple[TrialResult | None, str | None]:
    phase_key = str(phase or "primary")
    abort_prefix = "repeat_" if phase_key == "repeat" else ""

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
            log_line(f"[CALIBRATE_DIST] {trial_label}: recovery failed before act. Aborting.")
            return None, f"{abort_prefix}pre_pose_unavailable_trial_{trial_idx}"
    if not isinstance(pre_obs_meta, dict):
        pre_obs_meta = {"mode": "unknown", "reobserved": False}

    act_plan = _planned_action_meta(cmd, setup_score, duration_ms)
    pre_dist_mm = float(pre_pose.get("dist") or 0.0)
    gap_mm, where_text = _target_distance_status(pre_dist_mm, target_dist_mm=float(target_dist_mm))
    trial_plan_text = "this repeat trial will move" if phase_key == "repeat" else "this planned trial will move"
    log_line(
        f"[CALIBRATE_DIST] {trial_label}: I see that I'm {_highlight_mm(gap_mm)} {where_text}, "
        f"and {trial_plan_text} {_highlight_drive_letter(cmd)} {_highlight_number_text(f'{int(setup_score)}%')} "
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
        log_line(f"[CALIBRATE_DIST] {trial_label}: send failed. Aborting.")
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
            log_line(f"[CALIBRATE_DIST] {trial_label}: recovery failed after act. Aborting.")
            return None, f"{abort_prefix}post_pose_unavailable_trial_{trial_idx}"
        recovered_visibility = True
        recovery_mode = str(post_obs_meta.get("mode") or "unknown")
        recovery_inverse_acts = _coerce_int(post_obs_meta.get("inverse_acts"), 0)

    post_dist_mm = float(post_pose.get("dist") or 0.0)
    movement = _movement_metrics(cmd, pre_dist_mm, post_dist_mm)
    cmd_delta_mm = float(movement["cmd_delta_mm"])
    source_trial_value = _coerce_int(source_trial, trial_idx)
    row = TrialResult(
        trial=int(trial_idx),
        duration_ms=int(duration_used_ms or 0),
        cmd=str(cmd),
        score_requested=int(setup_score),
        cmd_sent=str(action_meta.get("cmd_sent") or cmd),
        pwm=_coerce_int(action_meta.get("pwm")),
        power=_coerce_float(action_meta.get("power")),
        pre_dist_mm=pre_dist_mm,
        post_dist_mm=post_dist_mm,
        raw_delta_mm=float(movement["raw_delta_mm"]),
        signed_cmd_delta_mm=float(movement["signed_cmd_delta_mm"]),
        cmd_delta_mm=cmd_delta_mm,
        wrong_way=bool(movement["wrong_way"]),
        pre_brick_dist_mm=pre_dist_mm,
        post_brick_dist_mm=post_dist_mm,
        pre_confidence=float(pre_pose.get("confidence") or 0.0),
        post_confidence=float(post_pose.get("confidence") or 0.0),
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
        pre_dist_mm,
        post_dist_mm,
        target_dist_mm=float(target_dist_mm),
    )
    log_line(
        f"[CALIBRATE_DIST] {trial_label}: That act resulted in {_highlight_mm(goal_delta_mm)} difference "
        f"and I'm {_highlight_progress_text(goal_status)} "
        f"({_highlight_number_text(f'dist={post_dist_mm:.2f}mm')})."
    )

    repeat_status = None
    if phase_key == "repeat" and compare_to_distance is not None:
        delta_from_source = abs(float(cmd_delta_mm) - float(compare_to_distance))
        if delta_from_source > REPEAT_RESULT_ERROR_MARGIN_MM:
            repeat_status = "fail"
            log_line(
                f"[CALIBRATE_DIST] {trial_label}: repeat result differs by {delta_from_source:.2f}mm "
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
            cmds=("f", "b"),
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

    def add_point(self, **kwargs) -> None:
        self._plot.add_point(**kwargs)

    def finish(self) -> None:
        self._plot.finish()


def _write_results(path: Path, payload: dict) -> None:
    shared_write_results(path, payload)


def _build_payload(
    *,
    config: dict,
    durations_ms: list[int],
    trials: list[TrialResult],
    status: str,
    abort_reason: str | None,
) -> dict:
    return build_shared_payload(
        source="calibrate_dist",
        config=config,
        durations_ms=durations_ms,
        trials=trials,
        reset_efforts=[],
        status=status,
        abort_reason=abort_reason,
    )


def _exit_as_script(exit_code: int) -> None:
    if sys.gettrace() is not None:
        return
    raise SystemExit(int(exit_code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal distance-gap duration probe with live scatter updates.")
    parser.add_argument("--trials", type=int, default=None, help="Optional primary-trial cap; default runs the full deterministic duration schedule.")
    parser.add_argument("--speed-score", type=int, default=SPEED_SCORE_DEFAULT, help=f"Fixed drive speed score (default: {SPEED_SCORE_DEFAULT}).")
    parser.add_argument(
        "--target-dist-mm",
        type=float,
        default=DIST_TARGET_MM_DEFAULT,
        help="Target brick distance in mm; if omitted, the first observed distance becomes the target.",
    )
    parser.add_argument("--vision", choices=["leia", "yolo", "aruco"], default="aruco")
    parser.add_argument("--min-duration-ms", type=int, default=200, help="Minimum deterministic duration in ms (default: 200).")
    parser.add_argument("--max-duration-ms", type=int, default=400, help="Maximum deterministic duration in ms (default: 400).")
    parser.add_argument("--observe-samples", type=int, default=OBSERVE_SAMPLES_DEFAULT)
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S)
    parser.add_argument("--post-act-settle-s", type=float, default=POST_ACT_SETTLE_S)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=PLOT_FILE_DEFAULT)
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_DEFAULT)
    parser.add_argument("--reference-distance-mm", type=float, default=None)
    args = parser.parse_args()
    _ensure_run_dir()

    if args.results_file is None:
        args.results_file = str(RUN_DIR / "calibrate_dist_live.json")
    if args.plot_path is None:
        args.plot_path = str(RUN_DIR / "calibrate_dist_live.png")

    results_path = Path(args.results_file)
    plot_path = Path(args.plot_path) if args.plot_path else None
    trials_requested = None if args.trials is None else max(1, int(args.trials))
    speed_score = int(normalize_speed_score(args.speed_score))
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    post_act_settle_s = max(0.0, float(args.post_act_settle_s))
    min_duration_ms = max(1, int(args.min_duration_ms))
    max_duration_ms = max(min_duration_ms, int(args.max_duration_ms))
    if args.reference_distance_mm is not None:
        global REFERENCE_BRICK_DISTANCE_MM
        REFERENCE_BRICK_DISTANCE_MM = float(args.reference_distance_mm)

    full_durations_ms = _build_duration_schedule(
        trials=None,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
    )
    full_trial_plan = _build_trial_plan(durations_ms=full_durations_ms, trials=None)
    trial_plan = _build_trial_plan(durations_ms=full_durations_ms, trials=trials_requested)
    durations_ms = _planned_durations_ms(trial_plan)
    trials_planned = len(trial_plan)

    config = {
        "trials": int(trials_planned),
        "requested_trials": None if trials_requested is None else int(trials_requested),
        "repeat_pass_enabled": True,
        "duration_ceiling_ms": int(DURATION_CEILING_MS),
        "speed_score": int(speed_score),
        "min_duration_ms": int(min_duration_ms),
        "max_duration_ms": int(max_duration_ms),
        "duration_step_ms": int(DURATION_STEP_MS_DEFAULT),
        "distance_positive_cmd": str(_dist_cmd_for_positive_motion()),
        "distance_negative_cmd": str(_dist_cmd_for_negative_motion()),
        "observe_samples": int(observe_samples),
        "observe_timeout_s": float(observe_timeout_s),
        "post_act_settle_s": float(post_act_settle_s),
        "primary_trial_cmd_schedule": "per_duration_sequence",
        "primary_trial_cmd_sequence": [str(cmd).upper() for cmd in _primary_trial_cmd_sequence()],
        "primary_trial_repetitions_per_direction": 2,
        "plot_path": str(plot_path) if plot_path is not None else None,
        "brick_distance_source": str(BRICK_DISTANCE_SOURCE),
        "brick_distance_definition": str(BRICK_DISTANCE_DEFINITION),
    }
    if REFERENCE_BRICK_DISTANCE_MM is not None:
        config["reference_brick_distance_mm"] = float(REFERENCE_BRICK_DISTANCE_MM)

    log_line("[CALIBRATE_DIST] Starting distance-gap duration probe.")
    log_line(
        f"[CALIBRATE_DIST] trials={trials_planned} score={int(speed_score)}% durations_ms={durations_ms} "
        f"observe_samples={observe_samples}"
    )
    log_line(
        f"[CALIBRATE_DIST] deterministic duration climb: start={int(min_duration_ms)}ms "
        f"stop={int(max_duration_ms)}ms step={int(DURATION_STEP_MS_DEFAULT)}ms."
    )
    log_line(
        f"[CALIBRATE_DIST] repeat_pass=enabled; repeats are deferred until all {int(trials_planned)} primary trial(s) finish."
    )

    plotter = LivePlot(show_plot=bool(args.show_plot), plot_path=plot_path)
    robot = None
    vision = None
    world = None
    recent_acts = deque(maxlen=32)
    trial_rows: list[TrialResult] = []
    status = "completed"
    abort_reason = None
    target_dist_mm = None if args.target_dist_mm is None else float(args.target_dist_mm)

    try:
        world = WorldModel()
        world.step_state = StepState.ALIGN_BRICK
        world._post_action_observe_delay_s = 0.0
        robot = Robot()
        if args.vision == "yolo":
            if YoloBrickDetector is None:
                raise RuntimeError("YOLO detector module not installed; cannot use --vision yolo")
            vision = YoloBrickDetector(debug=False)
        elif args.vision == "aruco":
            from helper_vision_aruco import ArucoBrickVision

            vision = ArucoBrickVision(debug=False)
        else:
            vision = LeiaVision(debug=False)

        if target_dist_mm is None:
            initial_pose, _initial_meta = _observe_pose_with_reobserve(
                vision=vision,
                world=world,
                samples=observe_samples,
                timeout_s=observe_timeout_s,
            )
            if initial_pose is None:
                status = "aborted"
                abort_reason = "initial_target_pose_unavailable"
                log_line("[CALIBRATE_DIST] Failed to observe initial target distance. Aborting.")
            else:
                target_dist_mm = float(initial_pose.get("dist") or 0.0)
                log_line(f"[CALIBRATE_DIST] Using initial observed distance as target: {target_dist_mm:.2f}mm.")
        if target_dist_mm is None:
            target_dist_mm = float(REFERENCE_BRICK_DISTANCE_MM or PLOT_TITLE_DISTANCE_MM_DEFAULT)
        config["target_dist_mm"] = float(target_dist_mm)
        log_line(f"[CALIBRATE_DIST] target distance: dist={float(target_dist_mm):.2f}mm.")

        if status == "completed":
            for trial_idx, plan_step in enumerate(trial_plan, start=1):
                trial_label = _trial_label_text(trial_idx, trials_planned)
                cmd = str(plan_step.get("cmd") or "")
                duration_ms = max(1, int(_coerce_int(plan_step.get("duration_ms"), 1) or 1))
                log_line(
                    f"[CALIBRATE_DIST] {trial_label}: scheduled cmd={str(cmd).upper()} "
                    f"({_drive_label_for_cmd(cmd)}) for {_highlight_duration_ms(int(duration_ms))}."
                )
                row, trial_abort_reason = _run_trial_action(
                    trial_idx=trial_idx,
                    trials_planned=trials_planned,
                    trial_label=trial_label,
                    cmd=str(cmd),
                    duration_ms=int(duration_ms),
                    phase="primary",
                    source_trial=trial_idx,
                    action_step="CALIBRATE_DIST",
                    plot_kind="trial",
                    vision=vision,
                    world=world,
                    robot=robot,
                    recent_acts=recent_acts,
                    setup_score=int(speed_score),
                    target_dist_mm=float(target_dist_mm),
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
                    log_line(f"[CALIBRATE_DIST] ⚠️  Trial {trial_idx}: wrong_way detected. Plotting it anyway.")
                trial_rows.append(row)
                _write_results(
                    results_path,
                    _build_payload(
                        config=config,
                        durations_ms=durations_ms,
                        trials=trial_rows,
                        status=status,
                        abort_reason=abort_reason,
                    ),
                )

        if status == "completed":
            repeat_plan = [row for row in trial_rows if str(getattr(row, "phase", "primary")) != "repeat"]
            log_line(
                f"[CALIBRATE_DIST] Primary pass complete. Starting repeat pass over {len(repeat_plan)} recorded trial(s)."
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
                    action_step="CALIBRATE_DIST_REPEAT",
                    plot_kind="repeat",
                    vision=vision,
                    world=world,
                    robot=robot,
                    recent_acts=recent_acts,
                    setup_score=int(speed_score),
                    target_dist_mm=float(target_dist_mm),
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
                    log_line(f"[CALIBRATE_DIST] ⚠️  Repeat {repeat_idx}: wrong_way detected. Plotting it anyway.")
                trial_rows.append(repeat_row)
                _write_results(
                    results_path,
                    _build_payload(
                        config=config,
                        durations_ms=durations_ms,
                        trials=trial_rows,
                        status=status,
                        abort_reason=abort_reason,
                    ),
                )
    except KeyboardInterrupt:
        status = "interrupted"
        abort_reason = "keyboard_interrupt"
        log_line("[CALIBRATE_DIST] Interrupted by user.")
    finally:
        _write_results(
            results_path,
            _build_payload(
                config=config,
                durations_ms=durations_ms,
                trials=trial_rows,
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

    log_line(f"[CALIBRATE_DIST] Wrote results to {results_path}")
    if plot_path is not None:
        log_line(f"[CALIBRATE_DIST] Updated plot at {plot_path}")
    if status != "completed":
        detail = f" reason={abort_reason}" if abort_reason else ""
        log_line(f"[CALIBRATE_DIST] Finished with status={status}{detail}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    _exit_as_script(main())
