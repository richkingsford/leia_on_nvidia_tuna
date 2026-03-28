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
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .helper_calibrate import (
    CALIBRATION_DURATION_LIMIT_MS,
    CalibrationLivePlot,
    build_linear_duration_schedule,
    build_payload as build_shared_payload,
    cleanup_old_run_files,
    coerce_finite_float as shared_coerce_finite_float,
    coerce_float as shared_coerce_float,
    coerce_int as shared_coerce_int,
    ensure_run_dir,
    get_shared_stream_runtime,
    load_calibration_trial_speed_profile as shared_load_calibration_trial_speed_profile,
    observe_pose_with_reobserve as shared_observe_pose_with_reobserve,
    planned_durations_ms as shared_planned_durations_ms,
    prediction_closeness_percentage as shared_prediction_closeness_percentage,
    prompt_calibration_run_settings as shared_prompt_calibration_run_settings,
    prepare_shared_stream_state,
    read_pose as shared_read_pose,
    resolve_calibration_trial_speed_score as shared_resolve_calibration_trial_speed_score,
    trial_label_text as shared_trial_label_text,
    write_results as shared_write_results,
)
from helper_close_gaps import load_axis_aruco_calibration
from helper_manual_config import load_manual_training_config
from helper_robot_control import Robot
from helper_stream_server import format_stream_url
from helper_streaming import start_stream_server
from helper_vision_leia import LeiaVision

try:
    from helper_brick_detector_yolo import BrickDetector as YoloBrickDetector
except ImportError:
    YoloBrickDetector = None

from telemetry_process import (
    _average_smoothed_frames as telemetry_average_smoothed_frames,
    _latest_unique_smoothed_frames as telemetry_latest_unique_smoothed_frames,
    lite_gate_unique_frames,
    send_robot_command,
    update_world_from_vision,
)
from telemetry_robot import (
    StepState,
    WorldModel,
    draw_telemetry_overlay,
    normalize_speed_score,
    one_percent_discovery_note as shared_one_percent_discovery_note,
    speed_power_pwm_for_cmd,
)
import helper_xyz_coords

OBSERVE_SLEEP_S = 0.02
OBSERVE_TIMEOUT_S = 1.8
POST_ACT_SETTLE_S = 0.10
OBSERVE_SAMPLES_DEFAULT = 3
DIST_TARGET_MM_DEFAULT: float | None = None
SPEED_SCORE_DEFAULT = 1
DURATION_CEILING_MS = CALIBRATION_DURATION_LIMIT_MS
MIN_DURATION_MS_DEFAULT = 200
MAX_DURATION_MS_DEFAULT = 400
DURATION_STEP_MS_DEFAULT = 20
DIST_POSITIVE_CMD_DEFAULT = "b"
DIST_INVERT = False
TRIAL_CMD_AUTO = "auto"
TRIAL_SPEED_MODE_DISTANCE_CURVE = "distance_curve"
TRIAL_SPEED_MODE_FIXED = "fixed"
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
RUN_DIR_ARUCO = Path("Runs - aruco")
RUN_DIR_CYAN = Path("Runs - cyan")
BRICK_DISTANCE_SOURCE = "vision.dist"
BRICK_DISTANCE_DEFINITION = "Camera-to-brick distance reported by vision at observation time (mm)."
DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM = 0.1
FAST_ALIGN_TIME_BUDGET_S = 1.5
PLOT_TITLE_FONT_SIZE = 10
PLOT_LABEL_FONT_SIZE = 9
PLOT_TICK_FONT_SIZE = 8
PLOT_LEGEND_FONT_SIZE = 8
ANSI_CYAN_BRIGHT = "\033[96m"
ANSI_WHITE_BRIGHT = "\033[97m"
ANSI_BLUE_BRIGHT = "\033[94m"
ANSI_MAGENTA_BRIGHT = "\033[95m"
ANSI_GREEN_BRIGHT = "\033[92m"
ANSI_RED_BRIGHT = "\033[91m"
ANSI_YELLOW_BRIGHT = "\033[93m"
ANSI_ORANGE_BRIGHT = "\033[38;5;208m"
ANSI_GRAY = "\033[90m"

_MANUAL_CONFIG = load_manual_training_config()
STREAM_HOST = str(_MANUAL_CONFIG.get("stream_host", "127.0.0.1"))
STREAM_PORT = int(_MANUAL_CONFIG.get("stream_port", 5000))
STREAM_FPS = int(_MANUAL_CONFIG.get("stream_fps", 10))
STREAM_JPEG_QUALITY = int(_MANUAL_CONFIG.get("stream_jpeg_quality", 85))
STREAM_IMG_WIDTH = int(_MANUAL_CONFIG.get("stream_img_width", 1600))


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
    no_meaningful_movement: bool = False
    lost_visibility: bool = False
    recovered_visibility: bool = False
    recovery_mode: str | None = None
    recovery_inverse_acts: int | None = None
    pre_lite_required_frames: int | None = None
    post_lite_required_frames: int | None = None
    phase: str = "primary"
    source_trial: int | None = None
    pre_observe_elapsed_s: float | None = None
    post_observe_elapsed_s: float | None = None
    total_trial_elapsed_s: float | None = None
    signed_effective_mm_per_s: float | None = None
    predicted_distance_mm: float | None = None
    curve_source: str | None = None
    absolute_difference_mm: float | None = None
    prediction_closeness_percentage: float | None = None


def log_line(message: str) -> None:
    print(str(message), flush=True)


def _current_vision_frame(vision):
    for attr in ("current_frame", "debug_frame", "raw_frame"):
        frame = getattr(vision, attr, None)
        if frame is not None:
            return frame
    return None


def _refresh_stream_state(
    *,
    stream_state: dict | None,
    vision,
    world,
    title_lines: list[str],
) -> None:
    if not isinstance(stream_state, dict):
        return
    frame = _current_vision_frame(vision)
    if frame is None:
        try:
            update_world_from_vision(world, vision, log=False)
            frame = _current_vision_frame(vision)
        except Exception:
            frame = _current_vision_frame(vision)
    text_lines = list(title_lines or [])
    try:
        helper_xyz_coords.sync_from_world(world, reason="vision", render=False)
    except Exception:
        pass
    xyz_workspace = getattr(world, "_xyz_workspace", None)
    if frame is not None:
        try:
            show_cl = bool(stream_state.get("show_center_line", True))
            frame = draw_telemetry_overlay(
                frame,
                world,
                show_prompt=False,
                draw_text=False,
                line_sink=text_lines,
                show_center_line=show_cl,
            )
        except Exception:
            pass
    lock = stream_state.get("lock")
    if lock is None:
        stream_state["frame"] = frame
        stream_state["text_lines"] = text_lines
        stream_state["xyz_workspace"] = xyz_workspace
        return
    with lock:
        stream_state["frame"] = frame
        stream_state["text_lines"] = text_lines
        stream_state["xyz_workspace"] = xyz_workspace


def _supports_ansi_color() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _colorize(text: str, color_code: str) -> str:
    if not _supports_ansi_color():
        return str(text)
    return f"{str(color_code)}{str(text)}\033[0m"


def _orange_text(text: str) -> str:
    return _colorize(str(text), ANSI_ORANGE_BRIGHT)


def _highlight_number_text(text: str) -> str:
    return _colorize(str(text), ANSI_CYAN_BRIGHT)


def _highlight_score_text(text: str) -> str:
    return _colorize(str(text), ANSI_WHITE_BRIGHT)


def _highlight_mm(value: float, *, signed: bool = False) -> str:
    number = float(value)
    fmt = f"{number:+.2f}mm" if signed else f"{abs(number):.2f}mm"
    return _highlight_number_text(fmt)


def _highlight_duration_ms(duration_ms: int) -> str:
    return _highlight_number_text(f"{int(duration_ms)}ms")


def _score_with_motion_details_text(
    cmd: str | None,
    score: int,
    *,
    pwm: int | None,
    power: float | None,
    duration_ms: int,
) -> str:
    pwm_text = "?" if pwm is None else str(int(pwm))
    power_text = "?" if power is None else f"{float(power):.3f}"
    detail_body = f"(pwm={pwm_text}, pwr={power_text}, t={int(duration_ms)}ms"
    discovery_note = shared_one_percent_discovery_note(cmd, score)
    if discovery_note:
        detail_body += f"; {str(discovery_note)}"
    detail_body += ")"
    detail_text = _colorize(detail_body, ANSI_GRAY)
    return f"{_highlight_score_text(f'{int(score)}%')} {detail_text}"


def _trial_speed_profile_for_mode(trial_speed_mode: str | None) -> dict | None:
    mode = str(trial_speed_mode or TRIAL_SPEED_MODE_DISTANCE_CURVE).strip().lower()
    if mode == TRIAL_SPEED_MODE_FIXED:
        return None
    return shared_load_calibration_trial_speed_profile("dist")


def _cleanup_old_run_files():
    cleanup_old_run_files(
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_y_live.json",
            "calibrate_dist_live.json",
        },
    )


def _run_dir_for_vision(vision_mode: str | None) -> Path:
    mode = str(vision_mode or "").strip().lower()
    if mode == "aruco":
        return Path(RUN_DIR_ARUCO)
    return Path(RUN_DIR_CYAN)


def _ensure_run_dir(*, run_dir: Path):
    ensure_run_dir(
        run_dir=Path(run_dir),
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_y_live.json",
            "calibrate_dist_live.json",
        },
    )

def _coerce_float(value, fallback=None):
    return shared_coerce_float(value, fallback)


def _coerce_int(value, fallback=None):
    return shared_coerce_int(value, fallback)


def _coerce_finite_float(value) -> float | None:
    return shared_coerce_finite_float(value)


def _prediction_metric_color_code(prediction_closeness_percentage: float | None) -> str | None:
    if prediction_closeness_percentage is None:
        return None
    return "\033[92m" if float(prediction_closeness_percentage) < 25.0 else "\033[91m"


def _format_prediction_comparison_fields(prediction_comparison: dict) -> str:
    closeness_value = prediction_comparison.get("prediction_closeness_percentage")
    absolute_difference_value = prediction_comparison.get("absolute_difference_mm")
    if closeness_value is None or absolute_difference_value is None:
        return ""
    color_code = _prediction_metric_color_code(closeness_value)
    absolute_difference_text = f"{float(absolute_difference_value):.2f}"
    prediction_closeness_text = f"{float(closeness_value):.1f}"
    if color_code is not None:
        absolute_difference_text = _colorize(absolute_difference_text, color_code)
        prediction_closeness_text = _colorize(prediction_closeness_text, color_code)
    return (
        f"absolute_difference={absolute_difference_text} "
        f"prediction_closeness={prediction_closeness_text}%"
    )


def _normalize_cmd(value: str, *, allow_auto: bool = False) -> str:
    text = str(value or "").strip().lower()
    if allow_auto and text in ("auto", "target"):
        return "auto"
    if text not in ("f", "b"):
        raise ValueError("Allowed dist commands are only 'f', 'b', 'auto', or 'target'.")
    return text


def _dist_curve_display_name(calibration: dict | None) -> str:
    if not isinstance(calibration, dict):
        return "no_curve"
    calib = dict(calibration)
    base_name = str(calib.get("source") or "aruco_marker_calibration").strip() or "aruco_marker_calibration"
    reference_distance_mm = _coerce_finite_float(calib.get("reference_distance_mm"))
    speed_score_pct = _coerce_finite_float(calib.get("speed_score_pct"))
    if reference_distance_mm is not None and speed_score_pct is not None:
        return (
            f"{base_name} at {float(reference_distance_mm):.0f}mm distance "
            f"at {int(round(float(speed_score_pct)))}% speed"
        )
    if reference_distance_mm is not None:
        return f"{base_name} at {float(reference_distance_mm):.0f}mm distance"
    if speed_score_pct is not None:
        return f"{base_name} at {int(round(float(speed_score_pct)))}% speed"
    return base_name


def _load_dist_duration_calibration() -> dict | None:
    calibration = load_axis_aruco_calibration("dist")
    if not isinstance(calibration, dict):
        return None
    by_cmd = calibration.get("by_cmd")
    if not isinstance(by_cmd, dict):
        return None
    if not any(isinstance(by_cmd.get(cmd), dict) for cmd in ("f", "b")):
        return None
    return dict(calibration)


def _predict_movement_from_curve(
    *,
    cmd: str,
    duration_ms: int,
    dist_calibration: dict | None,
) -> tuple[float | None, str]:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    calibration = dist_calibration if isinstance(dist_calibration, dict) else None
    if not isinstance(calibration, dict):
        return None, "no_curve"
    curve_name = _dist_curve_display_name(calibration)
    row = calibration.get("by_cmd", {}).get(cmd_key) if isinstance(calibration.get("by_cmd"), dict) else {}
    if not isinstance(row, dict):
        return None, curve_name
    try:
        slope = float(row.get("slope_mm_per_ms"))
        intercept = float(row.get("intercept_mm"))
    except Exception:
        return None, curve_name
    if slope <= 1e-9:
        return None, curve_name
    predicted_mm = float(slope) * float(duration_ms) + float(intercept)
    return max(0.0, predicted_mm), curve_name


def _calculate_prediction_comparison(
    *,
    actual_distance_mm: float,
    predicted_distance_mm: float | None,
    curve_source: str,
) -> dict:
    if predicted_distance_mm is None or predicted_distance_mm <= 0.0:
        return {
            "predicted_distance_mm": None,
            "curve_source": curve_source,
            "absolute_difference_mm": None,
            "prediction_closeness_percentage": None,
        }
    absolute_difference_mm = abs(float(actual_distance_mm) - float(predicted_distance_mm))
    prediction_closeness = shared_prediction_closeness_percentage(
        actual_distance_mm=actual_distance_mm,
        predicted_distance_mm=predicted_distance_mm,
    )
    return {
        "predicted_distance_mm": float(predicted_distance_mm),
        "curve_source": str(curve_source),
        "absolute_difference_mm": float(absolute_difference_mm),
        "prediction_closeness_percentage": (
            float(prediction_closeness) if prediction_closeness is not None else None
        ),
    }


def _trials_setup_log_line(
    *,
    observed_distance_mm: float | None,
    closest_curve_name: str,
) -> str:
    observed_distance = _coerce_finite_float(observed_distance_mm)
    observed_distance_text = (
        f"{float(observed_distance):.2f}mm" if observed_distance is not None else "unknown"
    )
    message = (
        "[TRIALS SETUP] "
        f"observed_distance={observed_distance_text} "
        f"closest_speed_curve={str(closest_curve_name or 'no_curve')}"
    )
    return _colorize(message, "\033[92m")


def read_pose(
    vision,
    world,
    *,
    samples: int = OBSERVE_SAMPLES_DEFAULT,
    timeout_s: float = OBSERVE_TIMEOUT_S,
    min_sample_time: float | None = None,
    min_samples_required: int | None = None,
    on_vision_update=None,
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
        on_vision_update=on_vision_update,
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
    on_vision_update=None,
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
        on_vision_update=on_vision_update,
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


def _auto_cmd_for_dist(
    curr_dist_mm: float,
    *,
    target_dist_mm: float,
) -> str:
    if float(curr_dist_mm) > float(target_dist_mm):
        return _dist_cmd_for_negative_motion()
    return _dist_cmd_for_positive_motion()


def _expected_distance_effect_for_cmd(cmd: str) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == "f":
        return "decrease"
    return "increase"


def _distance_effect_from_raw_delta(raw_delta_mm: float, *, threshold_mm: float) -> str | None:
    raw_delta = float(raw_delta_mm)
    if abs(raw_delta) < max(0.0, float(threshold_mm)):
        return None
    return "increase" if raw_delta > 0.0 else "decrease"


def _distance_effect_human(effect: str | None) -> str:
    effect_key = str(effect or "").strip().lower()
    if effect_key == "increase":
        return "increase"
    if effect_key == "decrease":
        return "decrease"
    return "inconclusive"


def _new_distance_direction_check_entry(cmd: str) -> dict:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    return {
        "label": _drive_label_for_cmd(cmd_key),
        "expected_effect": _expected_distance_effect_for_cmd(cmd_key),
        "status": "pending",
        "observations": 0,
        "evidence_count": 0,
        "match_count": 0,
        "mismatch_count": 0,
        "increase_count": 0,
        "decrease_count": 0,
        "no_movement_count": 0,
        "inconclusive_count": 0,
        "last_raw_delta_mm": None,
        "last_cmd_sent": None,
        "last_trial_label": None,
    }


def _new_distance_direction_check_state(*, threshold_mm: float = DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM) -> dict:
    return {
        "movement_threshold_mm": max(0.0, float(threshold_mm)),
        "by_cmd": {
            "f": _new_distance_direction_check_entry("f"),
            "b": _new_distance_direction_check_entry("b"),
        },
    }


def _distance_direction_check_entry(check_state: dict | None, cmd: str) -> dict | None:
    if not isinstance(check_state, dict):
        return None
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    by_cmd = check_state.setdefault("by_cmd", {})
    entry = by_cmd.get(cmd_key)
    if not isinstance(entry, dict):
        entry = _new_distance_direction_check_entry(cmd_key)
        by_cmd[cmd_key] = entry
    return entry


def _refresh_distance_direction_entry_status(entry: dict | None) -> str:
    if not isinstance(entry, dict):
        return "pending"
    matches = max(0, int(_coerce_int(entry.get("match_count"), 0) or 0))
    mismatches = max(0, int(_coerce_int(entry.get("mismatch_count"), 0) or 0))
    no_movement = max(
        0,
        int(
            _coerce_int(
                entry.get("no_movement_count"),
                _coerce_int(entry.get("inconclusive_count"), 0),
            )
            or 0
        ),
    )
    if matches > 0 and mismatches > 0:
        status = "mixed"
    elif mismatches > 0:
        status = "mismatch"
    elif matches > 0:
        status = "verified"
    elif no_movement > 0:
        status = "no_movement"
    else:
        status = "pending"
    entry["status"] = str(status)
    return str(status)


def _record_distance_direction_check(
    check_state: dict | None,
    *,
    trial_label: str,
    cmd: str,
    cmd_sent: str | None,
    raw_delta_mm: float,
) -> None:
    entry = _distance_direction_check_entry(check_state, cmd)
    if not isinstance(entry, dict):
        return

    threshold_mm = max(
        0.0,
        float(_coerce_float((check_state or {}).get("movement_threshold_mm"), DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM) or 0.0),
    )
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    expected_effect = str(entry.get("expected_effect") or _expected_distance_effect_for_cmd(cmd_key))
    actual_effect = _distance_effect_from_raw_delta(float(raw_delta_mm), threshold_mm=float(threshold_mm))
    prior_status = _refresh_distance_direction_entry_status(entry)

    entry["observations"] = max(0, int(_coerce_int(entry.get("observations"), 0) or 0)) + 1
    entry["last_raw_delta_mm"] = float(raw_delta_mm)
    entry["last_cmd_sent"] = str(cmd_sent).strip().lower() if cmd_sent is not None else None
    entry["last_trial_label"] = str(trial_label)

    if actual_effect is None:
        no_movement_count = max(0, int(_coerce_int(entry.get("no_movement_count"), 0) or 0)) + 1
        entry["no_movement_count"] = int(no_movement_count)
        # Keep the legacy field populated for backward-compatible payload readers.
        entry["inconclusive_count"] = int(no_movement_count)
        status = _refresh_distance_direction_entry_status(entry)
        log_line(
            f"[CALIBRATE_DIST_DIRECTION_CHECK] {trial_label}: {str(entry.get('label') or _drive_label_for_cmd(cmd_key))} "
            f"produced no meaningful distance change; recording a no-change sample "
            f"(raw_delta={float(raw_delta_mm):+.2f}mm, threshold={float(threshold_mm):.2f}mm, "
            f"wire={str(cmd_sent or cmd_key).strip().upper() or cmd_key.upper()})."
        )
        entry["status"] = str(status)
        return

    entry["evidence_count"] = max(0, int(_coerce_int(entry.get("evidence_count"), 0) or 0)) + 1
    effect_key = str(actual_effect)
    effect_count_key = f"{effect_key}_count"
    entry[effect_count_key] = max(0, int(_coerce_int(entry.get(effect_count_key), 0) or 0)) + 1
    if effect_key == expected_effect:
        entry["match_count"] = max(0, int(_coerce_int(entry.get("match_count"), 0) or 0)) + 1
    else:
        entry["mismatch_count"] = max(0, int(_coerce_int(entry.get("mismatch_count"), 0) or 0)) + 1
    status = _refresh_distance_direction_entry_status(entry)

    if effect_key == expected_effect:
        if prior_status != "verified":
            log_line(
                f"[CALIBRATE_DIST_DIRECTION_CHECK] {trial_label}: verified {str(entry.get('label') or _drive_label_for_cmd(cmd_key))} "
                f"-> distance {_distance_effect_human(effect_key).upper()} "
                f"(raw_delta={float(raw_delta_mm):+.2f}mm, wire={str(cmd_sent or cmd_key).strip().upper() or cmd_key.upper()})."
            )
        return

    if prior_status != status or int(entry.get("mismatch_count") or 0) == 1:
        log_line(
            f"[CALIBRATE_DIST_DIRECTION_CHECK] {trial_label}: WARNING {str(entry.get('label') or _drive_label_for_cmd(cmd_key))} "
            f"caused distance {_distance_effect_human(effect_key).upper()}; expected {_distance_effect_human(expected_effect).upper()} "
            f"(raw_delta={float(raw_delta_mm):+.2f}mm, wire={str(cmd_sent or cmd_key).strip().upper() or cmd_key.upper()})."
        )


def _observed_distance_effect_for_cmd(check_state: dict | None, cmd: str) -> str | None:
    entry = _distance_direction_check_entry(check_state, cmd)
    if not isinstance(entry, dict):
        return None
    increase_count = max(0, int(_coerce_int(entry.get("increase_count"), 0) or 0))
    decrease_count = max(0, int(_coerce_int(entry.get("decrease_count"), 0) or 0))
    if increase_count > decrease_count:
        return "increase"
    if decrease_count > increase_count:
        return "decrease"
    return None


def _inferred_distance_positive_cmd(check_state: dict | None) -> str | None:
    forward_effect = _observed_distance_effect_for_cmd(check_state, "f")
    backward_effect = _observed_distance_effect_for_cmd(check_state, "b")
    if forward_effect == "increase" and backward_effect == "decrease":
        return "f"
    if backward_effect == "increase" and forward_effect == "decrease":
        return "b"
    return None


def _distance_direction_check_summary_line(check_state: dict | None) -> str | None:
    if not isinstance(check_state, dict):
        return None
    parts = []
    for cmd_key in ("f", "b"):
        entry = _distance_direction_check_entry(check_state, cmd_key)
        if not isinstance(entry, dict):
            continue
        status = _refresh_distance_direction_entry_status(entry)
        label = str(entry.get("label") or _drive_label_for_cmd(cmd_key))
        expected_effect = str(entry.get("expected_effect") or _expected_distance_effect_for_cmd(cmd_key))
        observed_effect = _observed_distance_effect_for_cmd(check_state, cmd_key)
        no_movement_count = max(
            0,
            int(
                _coerce_int(
                    entry.get("no_movement_count"),
                    _coerce_int(entry.get("inconclusive_count"), 0),
                )
                or 0
            ),
        )
        observed_text = (
            observed_effect
            if observed_effect is not None
            else "no_change"
            if no_movement_count > 0
            else "unknown"
        )
        parts.append(
            f"{label}={status} expected={expected_effect} observed={observed_text} "
            f"match={int(_coerce_int(entry.get('match_count'), 0) or 0)} "
            f"mismatch={int(_coerce_int(entry.get('mismatch_count'), 0) or 0)} "
            f"no_change={int(no_movement_count)}"
        )
    inferred_positive_cmd = _inferred_distance_positive_cmd(check_state)
    if inferred_positive_cmd is not None and inferred_positive_cmd != _dist_cmd_for_positive_motion():
        parts.append(
            f"suspected_drive_inversion observed_distance_positive_cmd={str(inferred_positive_cmd).upper()} "
            f"configured={str(_dist_cmd_for_positive_motion()).upper()}"
        )
    if not parts:
        return None
    return "[CALIBRATE_DIST_DIRECTION_CHECK] " + "; ".join(parts)


def _should_abort_for_distance_inversion(check_state: dict | None) -> bool:
    if not isinstance(check_state, dict):
        return False
    forward_entry = _distance_direction_check_entry(check_state, "f")
    backward_entry = _distance_direction_check_entry(check_state, "b")
    if not isinstance(forward_entry, dict) or not isinstance(backward_entry, dict):
        return False
    if _refresh_distance_direction_entry_status(forward_entry) != "mismatch":
        return False
    if _refresh_distance_direction_entry_status(backward_entry) != "mismatch":
        return False
    return (
        int(_coerce_int(forward_entry.get("evidence_count"), 0) or 0) > 0
        and int(_coerce_int(backward_entry.get("evidence_count"), 0) or 0) > 0
    )


def _fast_alignment_analysis(trials: list[TrialResult], *, target_budget_s: float) -> dict:
    primary_trials = [row for row in trials if str(getattr(row, "phase", "primary")) != "repeat"]
    by_duration: dict[int, list[TrialResult]] = {}
    for row in primary_trials:
        duration_ms = max(1, int(_coerce_int(getattr(row, "duration_ms", None), 1) or 1))
        by_duration.setdefault(duration_ms, []).append(row)

    duration_rows = []
    recommended_row = None
    for duration_ms in sorted(by_duration):
        rows = by_duration[duration_ms]
        valid_rows = [
            row
            for row in rows
            if not bool(getattr(row, "wrong_way", False))
            and _coerce_finite_float(getattr(row, "total_trial_elapsed_s", None)) is not None
            and float(getattr(row, "total_trial_elapsed_s")) > 0.0
        ]
        analysis_row = {
            "duration_ms": int(duration_ms),
            "trial_count": len(rows),
            "valid_trial_count": len(valid_rows),
            "median_cmd_delta_mm": (
                float(statistics.median(float(getattr(row, "cmd_delta_mm", 0.0)) for row in rows))
                if rows
                else None
            ),
            "median_total_trial_elapsed_s": (
                float(statistics.median(float(getattr(row, "total_trial_elapsed_s")) for row in valid_rows))
                if valid_rows
                else None
            ),
            "median_signed_effective_mm_per_s": (
                float(statistics.median(float(getattr(row, "signed_effective_mm_per_s")) for row in valid_rows))
                if valid_rows
                else None
            ),
        }
        median_elapsed_s = _coerce_finite_float(analysis_row.get("median_total_trial_elapsed_s"))
        analysis_row["fits_time_budget"] = bool(
            median_elapsed_s is not None and float(median_elapsed_s) <= float(target_budget_s)
        )
        duration_rows.append(analysis_row)

        if not valid_rows:
            continue
        row_score = _coerce_finite_float(analysis_row.get("median_signed_effective_mm_per_s"))
        if row_score is None:
            continue
        if recommended_row is None:
            recommended_row = analysis_row
            continue
        current_fit = bool(recommended_row.get("fits_time_budget"))
        candidate_fit = bool(analysis_row.get("fits_time_budget"))
        current_score = _coerce_finite_float(recommended_row.get("median_signed_effective_mm_per_s"))
        if candidate_fit and not current_fit:
            recommended_row = analysis_row
            continue
        if candidate_fit == current_fit and float(row_score) > float(current_score or float("-inf")) + 1e-9:
            recommended_row = analysis_row

    valid_trial_count = sum(int(row.get("valid_trial_count") or 0) for row in duration_rows)
    analysis = {
        "target_budget_s": float(target_budget_s),
        "primary_trial_count": len(primary_trials),
        "directionally_valid_trial_count": int(valid_trial_count),
        "recommended_duration_ms": None if recommended_row is None else int(recommended_row["duration_ms"]),
        "recommended_reason": None,
        "by_duration_ms": duration_rows,
    }
    if recommended_row is None:
        analysis["recommended_reason"] = "no_directionally_valid_trials"
    elif bool(recommended_row.get("fits_time_budget")):
        analysis["recommended_reason"] = "best_valid_mm_per_s_within_budget"
    else:
        analysis["recommended_reason"] = "best_valid_mm_per_s_outside_budget"
    return analysis


def _log_fast_alignment_analysis(analysis: dict | None) -> None:
    if not isinstance(analysis, dict):
        return
    recommended_duration_ms = _coerce_int(analysis.get("recommended_duration_ms"))
    if recommended_duration_ms is None:
        log_line(
            f"[CALIBRATE_DIST] Fast-align analysis blocked: {str(analysis.get('recommended_reason') or 'unavailable')}."
        )
        return
    target_budget_s = float(_coerce_float(analysis.get("target_budget_s"), FAST_ALIGN_TIME_BUDGET_S) or FAST_ALIGN_TIME_BUDGET_S)
    selected = None
    for row in analysis.get("by_duration_ms") or []:
        if int(_coerce_int((row or {}).get("duration_ms"), -1) or -1) == int(recommended_duration_ms):
            selected = row
            break
    if not isinstance(selected, dict):
        return
    median_elapsed_s = _coerce_finite_float(selected.get("median_total_trial_elapsed_s"))
    median_rate = _coerce_finite_float(selected.get("median_signed_effective_mm_per_s"))
    fits_budget = bool(selected.get("fits_time_budget"))
    elapsed_text = "unknown" if median_elapsed_s is None else f"{float(median_elapsed_s):.2f}s"
    rate_text = "unknown" if median_rate is None else f"{float(median_rate):.2f}mm/s"
    budget_text = "within" if fits_budget else "outside"
    log_line(
        f"[CALIBRATE_DIST] Fast-align recommendation: duration={int(recommended_duration_ms)}ms "
        f"median_total_trial_time={elapsed_text} median_signed_rate={rate_text} ({budget_text} {float(target_budget_s):.2f}s budget)."
    )

def _command_delta_mm(cmd: str, pre_dist_mm: float, post_dist_mm: float) -> float:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == _dist_cmd_for_positive_motion():
        return float(post_dist_mm) - float(pre_dist_mm)
    return float(pre_dist_mm) - float(post_dist_mm)


def _movement_metrics(
    cmd: str,
    pre_dist_mm: float,
    post_dist_mm: float,
    *,
    threshold_mm: float = DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM,
) -> dict:
    raw_delta_mm = float(post_dist_mm) - float(pre_dist_mm)
    threshold_value = max(0.0, float(threshold_mm))
    if abs(float(raw_delta_mm)) < float(threshold_value):
        return {
            "raw_delta_mm": float(raw_delta_mm),
            "signed_cmd_delta_mm": 0.0,
            "cmd_delta_mm": 0.0,
            "wrong_way": False,
            "no_meaningful_movement": True,
        }
    signed_cmd_delta_mm = _command_delta_mm(cmd, pre_dist_mm, post_dist_mm)
    travel_distance_mm = abs(float(post_dist_mm) - float(pre_dist_mm))
    return {
        "raw_delta_mm": float(raw_delta_mm),
        "signed_cmd_delta_mm": float(signed_cmd_delta_mm),
        "cmd_delta_mm": float(travel_distance_mm),
        "wrong_way": bool(float(signed_cmd_delta_mm) < 0.0),
        "no_meaningful_movement": False,
    }


def _drive_label_for_cmd(cmd: str) -> str:
    return "forward" if _normalize_cmd(cmd, allow_auto=False) == "f" else "backward"


def _highlight_drive_letter(cmd: str) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == "f":
        return _colorize("F", ANSI_BLUE_BRIGHT)
    return _colorize("B", ANSI_MAGENTA_BRIGHT)


def _log_distance_command_inversion_detail(
    *,
    prefix: str,
    trial_label: str,
    logical_cmd: str,
    wire_cmd: str,
    raw_delta_mm: float | None = None,
    threshold_mm: float = DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM,
) -> None:
    logical_cmd_key = _normalize_cmd(logical_cmd, allow_auto=False)
    wire_cmd_key = _normalize_cmd(wire_cmd, allow_auto=False)
    expected_effect = _expected_distance_effect_for_cmd(logical_cmd_key)
    wire_effect = _expected_distance_effect_for_cmd(wire_cmd_key)
    if raw_delta_mm is None:
        log_line(
            f"{str(prefix)} {trial_label}: expected {_drive_label_for_cmd(logical_cmd_key)} to make distance "
            f"{_distance_effect_human(expected_effect)}, but wire {_drive_label_for_cmd(wire_cmd_key)} implies "
            f"{_distance_effect_human(wire_effect)}."
        )
        return

    actual_effect = _distance_effect_from_raw_delta(float(raw_delta_mm), threshold_mm=float(threshold_mm))
    if actual_effect is None:
        log_line(
            f"{str(prefix)} {trial_label}: expected {_drive_label_for_cmd(logical_cmd_key)} to make distance "
            f"{_distance_effect_human(expected_effect)}, but observed no meaningful distance change "
            f"(raw_delta={float(raw_delta_mm):+.2f}mm, below {float(threshold_mm):.2f}mm threshold). "
            f"Recording this as a no-change sample; wire {_drive_label_for_cmd(wire_cmd_key)} still implies "
            f"{_distance_effect_human(wire_effect)}."
        )
        return

    log_line(
        f"{str(prefix)} {trial_label}: expected {_drive_label_for_cmd(logical_cmd_key)} to make distance "
        f"{_distance_effect_human(expected_effect)}, but observed {_distance_effect_human(actual_effect)} "
        f"({abs(float(raw_delta_mm)):.2f}mm; raw_delta={float(raw_delta_mm):+.2f}mm). "
        f"Wire {_drive_label_for_cmd(wire_cmd_key)} also implies {_distance_effect_human(wire_effect)}."
    )


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


def _evenly_spaced_duration_indices(length: int, count: int) -> list[int]:
    total = max(0, int(length))
    needed = max(0, int(count))
    if total <= 0 or needed <= 0:
        return []
    if needed >= total:
        return list(range(total))
    if needed == 1:
        return [0]

    indices: list[int] = []
    last_idx = -1
    for idx in range(needed):
        raw_position = float(idx) * float(total - 1) / float(max(1, needed - 1))
        candidate = int(round(raw_position))
        min_allowed = int(last_idx + 1)
        max_allowed = int(total - (needed - idx))
        candidate = max(int(min_allowed), min(int(max_allowed), int(candidate)))
        indices.append(int(candidate))
        last_idx = int(candidate)
    return indices


def _spread_duration_schedule(durations_ms: list[int], count: int) -> list[int]:
    source = [int(value) for value in list(durations_ms or [])]
    remaining = max(0, int(count))
    if not source or remaining <= 0:
        return []

    schedule: list[int] = []
    while remaining > 0:
        round_count = min(len(source), int(remaining))
        for idx in _evenly_spaced_duration_indices(len(source), int(round_count)):
            schedule.append(int(source[idx]))
        remaining -= int(round_count)
    return schedule


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


def _build_trial_plan(
    *,
    durations_ms: list[int],
    trials: int | None = None,
) -> list[dict]:
    scheduled_durations = (
        _spread_duration_schedule(durations_ms, max(1, int(trials)))
        if trials is not None
        else [int(value) for value in list(durations_ms or [])]
    )
    return [
        {
            "duration_ms": int(duration_ms),
            "cmd": str(TRIAL_CMD_AUTO),
        }
        for duration_ms in scheduled_durations
    ]


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
    on_vision_update=None,
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
            on_vision_update=on_vision_update,
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
    on_vision_update=None,
) -> tuple[dict | None, dict]:
    rounds = max(1, int(REOBSERVE_ROUNDS))
    for idx in range(1, rounds + 1):
        pose = read_pose(
            vision,
            world,
            samples=1,
            timeout_s=float(RECOVERY_OBSERVE_TIMEOUT_S),
            min_samples_required=1,
            on_vision_update=on_vision_update,
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
        on_vision_update=on_vision_update,
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
    on_vision_update=None,
) -> tuple[dict | None, dict]:
    display_label = str(trial_label or _trial_label_text(int(trial_idx), int(trials_requested)))
    log_line(f"[CALIBRATE_DIST] {display_label}: no visible brick {stage_label}. Attempting recovery.")
    pose, recovery_meta = _attempt_recovery(
        vision=vision,
        world=world,
        robot=robot,
        recent_acts=recent_acts,
        on_vision_update=on_vision_update,
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
    distance_direction_check: dict | None = None,
    plotter=None,
    initial_pre_pose: dict | None = None,
    initial_pre_obs_meta: dict | None = None,
    dist_duration_cal: dict | None = None,
    trial_speed_profile: dict | None = None,
    compare_to_distance: float | None = None,
    stream_refresh_fn=None,
) -> tuple[TrialResult | None, str | None]:
    phase_key = str(phase or "primary")
    abort_prefix = "repeat_" if phase_key == "repeat" else ""
    trial_start_ts = time.time()

    def _on_vision_update():
        if callable(stream_refresh_fn):
            stream_refresh_fn()

    pre_pose = dict(initial_pre_pose) if isinstance(initial_pre_pose, dict) else None
    pre_obs_meta = dict(initial_pre_obs_meta) if isinstance(initial_pre_obs_meta, dict) else None
    pre_observe_start_ts = time.time()
    if pre_pose is None:
        pre_pose, pre_obs_meta = _observe_pose_with_reobserve(
            vision=vision,
            world=world,
            samples=observe_samples,
            timeout_s=observe_timeout_s,
            on_vision_update=_on_vision_update,
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
            on_vision_update=_on_vision_update,
        )
        if pre_pose is None:
            log_line(f"[CALIBRATE_DIST] {trial_label}: recovery failed before act. Aborting.")
            return None, f"{abort_prefix}pre_pose_unavailable_trial_{trial_idx}"
    pre_observe_elapsed_s = max(0.0, time.time() - float(pre_observe_start_ts))
    if not isinstance(pre_obs_meta, dict):
        pre_obs_meta = {"mode": "unknown", "reobserved": False}

    effective_score, effective_score_meta = shared_resolve_calibration_trial_speed_score(
        observed_distance_mm=_coerce_finite_float(pre_pose.get("dist")),
        requested_score=int(setup_score),
        speed_profile=trial_speed_profile,
    )
    if str((effective_score_meta or {}).get("source") or "") == "distance_curve":
        observed_dist_local = _coerce_finite_float(pre_pose.get("dist"))
        observed_dist_text = (
            f"{float(observed_dist_local):.2f}mm"
            if observed_dist_local is not None
            else "unknown"
        )
        log_line(
            f"[CALIBRATE_DIST] {trial_label}: trial speed curve "
            f"dist={observed_dist_text} -> score={int(effective_score)}% "
            f"(base {int(setup_score)}%)."
        )

    act_plan = _planned_action_meta(cmd, effective_score, duration_ms)
    pre_dist_mm = float(pre_pose.get("dist") or 0.0)
    gap_mm, where_text = _target_distance_status(pre_dist_mm, target_dist_mm=float(target_dist_mm))
    trial_plan_text = "this repeat trial will move" if phase_key == "repeat" else "this planned trial will move"
    log_line(
        f"[CALIBRATE_DIST] {trial_label}: I see that I'm {_highlight_mm(gap_mm)} {where_text}, "
        f"and {trial_plan_text} {_highlight_drive_letter(cmd)} "
        f"{_score_with_motion_details_text(cmd, int(effective_score), pwm=act_plan.get('pwm'), power=act_plan.get('power'), duration_ms=int(act_plan['duration_ms']))}."
    )

    act_start_ts = time.time()
    action_meta = _send_fixed_score_command(
        robot=robot,
        world=world,
        step=str(action_step),
        cmd=cmd,
        score=int(effective_score),
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
            "score_requested": int(effective_score),
            "timestamp": time.time(),
        }
    )
    settle_ms = int(round(float(post_act_settle_s) * 1000.0))
    duration_wait_ms = int(duration_used_ms or 0)
    total_pause_ms = duration_wait_ms + settle_ms
    log_line(
        f"\033[90m[CALIBRATE_DIST] {trial_label}: {int(total_pause_ms)}ms pause "
        f"(cmd={int(duration_wait_ms)}ms + settle={int(settle_ms)}ms)\033[0m"
    )
    post_observe_start_ts = time.time()
    post_pose, post_obs_meta = _observe_pose_with_reobserve(
        vision=vision,
        world=world,
        samples=observe_samples,
        timeout_s=observe_timeout_s,
        min_sample_time=act_start_ts + (float(duration_used_ms or 0) / 1000.0) + float(post_act_settle_s),
        on_vision_update=_on_vision_update,
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
            on_vision_update=_on_vision_update,
        )
        if post_pose is None:
            log_line(f"[CALIBRATE_DIST] {trial_label}: recovery failed after act. Aborting.")
            return None, f"{abort_prefix}post_pose_unavailable_trial_{trial_idx}"
        recovered_visibility = True
        recovery_mode = str(post_obs_meta.get("mode") or "unknown")
        recovery_inverse_acts = _coerce_int(post_obs_meta.get("inverse_acts"), 0)
    post_observe_elapsed_s = max(0.0, time.time() - float(post_observe_start_ts))

    post_dist_mm = float(post_pose.get("dist") or 0.0)
    movement_threshold_mm = float(
        _coerce_float(
            (distance_direction_check or {}).get("movement_threshold_mm"),
            DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM,
        )
        or DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM
    )
    movement = _movement_metrics(
        cmd,
        pre_dist_mm,
        post_dist_mm,
        threshold_mm=float(movement_threshold_mm),
    )
    raw_delta_mm = float(movement["raw_delta_mm"])
    signed_cmd_delta_mm = float(movement["signed_cmd_delta_mm"])
    cmd_delta_mm = float(movement["cmd_delta_mm"])
    cmd_sent_effective = str(action_meta.get("cmd_sent") or cmd)
    _record_distance_direction_check(
        distance_direction_check,
        trial_label=trial_label,
        cmd=str(cmd),
        cmd_sent=str(cmd_sent_effective),
        raw_delta_mm=float(raw_delta_mm),
    )
    source_trial_value = _coerce_int(source_trial, trial_idx)
    total_trial_elapsed_s = max(0.0, time.time() - float(trial_start_ts))
    signed_effective_mm_per_s = None
    if total_trial_elapsed_s > 0.0:
        signed_effective_mm_per_s = float(signed_cmd_delta_mm) / float(total_trial_elapsed_s)
    row = TrialResult(
        trial=int(trial_idx),
        duration_ms=int(duration_used_ms or 0),
        cmd=str(cmd),
        score_requested=int(effective_score),
        cmd_sent=str(cmd_sent_effective),
        pwm=_coerce_int(action_meta.get("pwm")),
        power=_coerce_float(action_meta.get("power")),
        pre_dist_mm=pre_dist_mm,
        post_dist_mm=post_dist_mm,
        raw_delta_mm=raw_delta_mm,
        signed_cmd_delta_mm=signed_cmd_delta_mm,
        cmd_delta_mm=cmd_delta_mm,
        wrong_way=bool(movement["wrong_way"]),
        no_meaningful_movement=bool(movement.get("no_meaningful_movement")),
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
        pre_observe_elapsed_s=float(pre_observe_elapsed_s),
        post_observe_elapsed_s=float(post_observe_elapsed_s),
        total_trial_elapsed_s=float(total_trial_elapsed_s),
        signed_effective_mm_per_s=None if signed_effective_mm_per_s is None else float(signed_effective_mm_per_s),
    )

    predicted_distance_mm, curve_source = _predict_movement_from_curve(
        cmd=cmd,
        duration_ms=int(duration_used_ms or 0),
        dist_calibration=dist_duration_cal,
    )
    prediction_comparison = _calculate_prediction_comparison(
        actual_distance_mm=cmd_delta_mm,
        predicted_distance_mm=predicted_distance_mm,
        curve_source=curve_source,
    )
    row.predicted_distance_mm = prediction_comparison["predicted_distance_mm"]
    row.curve_source = prediction_comparison["curve_source"]
    row.absolute_difference_mm = prediction_comparison["absolute_difference_mm"]
    row.prediction_closeness_percentage = prediction_comparison["prediction_closeness_percentage"]

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
    log_line(
        "[CALIBRATE_DIST] "
        f"{_trial_label_text(trial_idx, trials_planned, phase=phase_key, source_trial=source_trial_value)}: "
        f"cmd={cmd.upper()} score="
        f"{_score_with_motion_details_text(cmd, int(effective_score), pwm=row.pwm, power=row.power, duration_ms=int(duration_used_ms or 0))} "
        f"start_dist={pre_dist_mm:.2f}mm end_dist={post_dist_mm:.2f}mm "
        f"distance={cmd_delta_mm:.2f}mm signed={signed_cmd_delta_mm:+.2f}mm "
        f"wrong_way={bool(row.wrong_way)} no_change={bool(row.no_meaningful_movement)} raw_delta={raw_delta_mm:+.2f}mm "
        f"predicted={prediction_comparison['predicted_distance_mm']:.2f}mm "
        f"curve_source={prediction_comparison['curve_source']} "
        f"{_format_prediction_comparison_fields(prediction_comparison)}"
        if prediction_comparison["predicted_distance_mm"] is not None
        else f"[CALIBRATE_DIST] "
        f"{_trial_label_text(trial_idx, trials_planned, phase=phase_key, source_trial=source_trial_value)}: "
        f"cmd={cmd.upper()} score="
        f"{_score_with_motion_details_text(cmd, int(effective_score), pwm=row.pwm, power=row.power, duration_ms=int(duration_used_ms or 0))} "
        f"start_dist={pre_dist_mm:.2f}mm end_dist={post_dist_mm:.2f}mm "
        f"distance={cmd_delta_mm:.2f}mm signed={signed_cmd_delta_mm:+.2f}mm "
        f"wrong_way={bool(row.wrong_way)} no_change={bool(row.no_meaningful_movement)} raw_delta={raw_delta_mm:+.2f}mm "
        f"predicted=None curve_source={prediction_comparison['curve_source']}"
    )
    if str(cmd_sent_effective).strip().lower() != str(cmd).strip().lower():
        _log_distance_command_inversion_detail(
            prefix="[CALIBRATE_DIST] Note:"
            if bool(row.no_meaningful_movement)
            else "[CALIBRATE_DIST] ⚠️  Command inversion detail:",
            trial_label=trial_label,
            logical_cmd=str(cmd),
            wire_cmd=str(cmd_sent_effective),
            raw_delta_mm=float(raw_delta_mm),
            threshold_mm=float(movement_threshold_mm),
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
    if callable(stream_refresh_fn):
        stream_refresh_fn()
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
    distance_direction_check: dict | None = None,
    fast_alignment_analysis: dict | None = None,
) -> dict:
    return build_shared_payload(
        source="calibrate_dist",
        config=config,
        durations_ms=durations_ms,
        trials=trials,
        reset_efforts=[],
        status=status,
        abort_reason=abort_reason,
        extra_fields={
            "distance_direction_check": json.loads(json.dumps(distance_direction_check or {})),
            "fast_alignment_analysis": json.loads(json.dumps(fast_alignment_analysis or {})),
        },
    )


def _exit_as_script(exit_code: int) -> None:
    if sys.gettrace() is not None:
        return
    raise SystemExit(int(exit_code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal distance-gap duration probe with live scatter updates.")
    parser.add_argument("--trials", type=int, default=None, help="Optional primary-trial cap; default runs the full deterministic duration schedule.")
    parser.add_argument(
        "--repeat-trials",
        type=int,
        default=None,
        help="Optional repeat-trial count; repeats run after primaries using recorded cmd/duration.",
    )
    parser.add_argument("--speed-score", type=int, default=SPEED_SCORE_DEFAULT, help=f"Fixed drive speed score (default: {SPEED_SCORE_DEFAULT}).")
    parser.add_argument(
        "--trial-speed-mode",
        choices=[TRIAL_SPEED_MODE_DISTANCE_CURVE, TRIAL_SPEED_MODE_FIXED],
        default=TRIAL_SPEED_MODE_DISTANCE_CURVE,
        help="How to choose dist trial speed: distance_curve (default) or fixed (respect --speed-score).",
    )
    parser.add_argument(
        "--target-dist-mm",
        type=float,
        default=DIST_TARGET_MM_DEFAULT,
        help="Target brick distance in mm; if omitted, the first observed distance becomes the target.",
    )
    parser.add_argument(
        "--vision",
        choices=["leia", "yolo", "aruco"],
        default="yolo",
        help="Which vision backend to use: yolo cyan bricks (default), aruco markers, or leia edges.",
    )
    parser.add_argument("--min-duration-ms", type=int, default=MIN_DURATION_MS_DEFAULT, help=f"Minimum deterministic duration in ms (default: {MIN_DURATION_MS_DEFAULT}).")
    parser.add_argument("--max-duration-ms", type=int, default=MAX_DURATION_MS_DEFAULT, help=f"Maximum deterministic duration in ms (default: {MAX_DURATION_MS_DEFAULT}).")
    parser.add_argument("--observe-samples", type=int, default=OBSERVE_SAMPLES_DEFAULT)
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S)
    parser.add_argument("--post-act-settle-s", type=float, default=POST_ACT_SETTLE_S)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=PLOT_FILE_DEFAULT, help="Optional PNG file to rewrite after each trial.")
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_DEFAULT)
    parser.add_argument(
        "--preflight-check",
        action="store_true",
        help="Run a 1% movement preflight check before trials (disabled by default).",
    )
    parser.add_argument(
        "--livestream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable livestream with overlay for distance trial progress (default: enabled).",
    )
    parser.add_argument("--stream-host", type=str, default=STREAM_HOST)
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT)
    parser.add_argument("--stream-fps", type=int, default=STREAM_FPS)
    parser.add_argument("--stream-jpeg-quality", type=int, default=STREAM_JPEG_QUALITY)
    parser.add_argument("--stream-img-width", type=int, default=STREAM_IMG_WIDTH)
    parser.add_argument("--reference-distance-mm", type=float, default=None)
    parser.add_argument(
        "--fast-target-budget-s",
        type=float,
        default=FAST_ALIGN_TIME_BUDGET_S,
        help=f"Target total time budget for one distance-correction cycle analysis (default: {FAST_ALIGN_TIME_BUDGET_S}).",
    )
    parser.add_argument(
        "--allow-direction-mismatch",
        action="store_true",
        help="Continue the run even if forward/backward are both proven inverted against observed distance motion.",
    )
    args = parser.parse_args()
    run_dir = _run_dir_for_vision(args.vision)
    _ensure_run_dir(run_dir=run_dir)

    if args.results_file is None:
        args.results_file = str(Path(run_dir) / "calibrate_dist_live.json")
    results_path = Path(args.results_file)
    plot_path = Path(args.plot_path) if args.plot_path else None
    trials_requested = None if args.trials is None else max(1, int(args.trials))
    repeat_trials_requested = None if args.repeat_trials is None else max(0, int(args.repeat_trials))
    speed_score = int(normalize_speed_score(args.speed_score))
    requested_speed_score = int(speed_score)
    prompted_speed_score = False
    prompted_duration_bounds = False
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    post_act_settle_s = max(0.0, float(args.post_act_settle_s))
    min_duration_ms = max(1, int(args.min_duration_ms))
    max_duration_ms = max(min_duration_ms, int(args.max_duration_ms))
    fast_target_budget_s = max(0.1, float(args.fast_target_budget_s))
    trial_speed_mode = str(args.trial_speed_mode or TRIAL_SPEED_MODE_DISTANCE_CURVE).strip().lower()
    if args.reference_distance_mm is not None:
        global REFERENCE_BRICK_DISTANCE_MM
        REFERENCE_BRICK_DISTANCE_MM = float(args.reference_distance_mm)

    trial_speed_profile = _trial_speed_profile_for_mode(trial_speed_mode)
    dist_duration_cal = _load_dist_duration_calibration()
    full_durations_ms = _build_duration_schedule(
        trials=None,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
    )
    trial_plan = _build_trial_plan(durations_ms=full_durations_ms, trials=trials_requested)
    durations_ms = _planned_durations_ms(trial_plan)
    trials_planned = len(trial_plan)
    repeat_pass_enabled = bool(repeat_trials_requested is not None and int(repeat_trials_requested) > 0)

    config = {
        "trials": int(trials_planned),
        "requested_trials": None if trials_requested is None else int(trials_requested),
        "repeat_pass_enabled": bool(repeat_pass_enabled),
        "requested_repeat_trials": None if repeat_trials_requested is None else int(repeat_trials_requested),
        "duration_ceiling_ms": int(DURATION_CEILING_MS),
        "speed_score": int(speed_score),
        "requested_speed_score": int(requested_speed_score),
        "speed_score_source": "arg",
        "min_duration_ms": int(min_duration_ms),
        "max_duration_ms": int(max_duration_ms),
        "duration_step_ms": int(DURATION_STEP_MS_DEFAULT),
        "distance_positive_cmd": str(_dist_cmd_for_positive_motion()),
        "distance_negative_cmd": str(_dist_cmd_for_negative_motion()),
        "observe_samples": int(observe_samples),
        "observe_timeout_s": float(observe_timeout_s),
        "post_act_settle_s": float(post_act_settle_s),
        "trial_cmd_mode": str(TRIAL_CMD_AUTO),
        "primary_trial_cmd_schedule": "auto_target_by_dist_sign",
        "trial_cmd_rule": "if dist > target_dist_mm: forward else backward",
        "plot_path": str(plot_path) if plot_path is not None else None,
        "brick_distance_source": str(BRICK_DISTANCE_SOURCE),
        "brick_distance_definition": str(BRICK_DISTANCE_DEFINITION),
        "fast_target_budget_s": float(fast_target_budget_s),
        "direction_check_min_delta_mm": float(DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM),
        "abort_on_direction_mismatch": not bool(args.allow_direction_mismatch),
        "trial_speed_mode": str(trial_speed_mode),
        "trial_speed_score_source": "distance_curve" if isinstance(trial_speed_profile, dict) else "arg",
    }
    if REFERENCE_BRICK_DISTANCE_MM is not None:
        config["reference_brick_distance_mm"] = float(REFERENCE_BRICK_DISTANCE_MM)
    if isinstance(trial_speed_profile, dict):
        config["trial_speed_score_profile"] = json.loads(json.dumps(trial_speed_profile))

    plotter = LivePlot(show_plot=bool(args.show_plot), plot_path=plot_path)
    robot = None
    vision = None
    world = None
    recent_acts = deque(maxlen=32)
    trial_rows: list[TrialResult] = []
    distance_direction_check = _new_distance_direction_check_state()
    status = "completed"
    abort_reason = None
    target_dist_mm = None if args.target_dist_mm is None else float(args.target_dist_mm)
    stream_server = None
    stream_state = None
    stream_url = format_stream_url(str(args.stream_host), int(args.stream_port))
    stream_score_line = (
        "Score: distance curve"
        if isinstance(trial_speed_profile, dict)
        else f"Score: {int(speed_score)}%"
    )

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

        if bool(args.livestream):
            shared_stream_state, shared_stream_url = get_shared_stream_runtime()
            stream_state = prepare_shared_stream_state(
                shared_stream_state,
                vision_mode="aruco" if str(args.vision).strip().lower() == "aruco" else "cyan",
            )
            stream_extra_lines: list[str] = []
            if stream_state is None:
                stream_state = {
                    "frame": None,
                    "text_lines": [],
                    "lock": threading.Lock(),
                    "show_center_line": True,
                    "vision_mode": "aruco" if str(args.vision).strip().lower() == "aruco" else "cyan",
                }
            else:
                stream_url = str(shared_stream_url or stream_url)
            log_line(f"[CALIBRATE_DIST] Livestream URL: {_orange_text(stream_url)}")

            def _live_refresh(extra_lines=None):
                nonlocal stream_extra_lines
                if extra_lines is not None:
                    stream_extra_lines = [str(item) for item in list(extra_lines)]
                lines = [
                    f"Distance calibration",
                    f"Trials: {len(trial_rows)}/{int(trials_planned)}",
                    str(stream_score_line),
                ]
                lines.extend(list(stream_extra_lines))
                _refresh_stream_state(
                    stream_state=stream_state,
                    vision=vision,
                    world=world,
                    title_lines=lines,
                )

            _live_refresh([])
            if shared_stream_state is not None and stream_state is shared_stream_state:
                log_line("[CALIBRATE_DIST] Reusing existing manual-training livestream.")
            else:
                try:
                    stream_server, stream_url = start_stream_server(
                        stream_state,
                        title="Distance Speed Curve Calibration",
                        header="",
                        footer="<div class='footer-sections'><div class='footer-section'><div class='footer-title'>Distance Calibration</div><div>Live trial telemetry.</div></div></div>",
                        host=str(args.stream_host),
                        port=int(args.stream_port),
                        fps=max(1, int(args.stream_fps)),
                        jpeg_quality=max(1, min(100, int(args.stream_jpeg_quality))),
                        img_width=max(320, int(args.stream_img_width)),
                        vision_mode_options=[("aruco", "AruCo Markers"), ("cyan", "Cyan Bricks")],
                        xyz_workspace_getter=lambda: getattr(world, "_xyz_workspace", None),
                    )
                    log_line(f"[CALIBRATE_DIST] Livestream started: {_orange_text(stream_url)}")
                except Exception as exc:
                    log_line(f"[CALIBRATE_DIST] Livestream startup failed at {stream_url}: {exc}")
                    stream_server = None
        else:
            def _live_refresh(extra_lines=None):
                return

        # Optional preflight check: verify 1% speed produces detectable movement.
        if status == "completed" and bool(args.preflight_check):
            from .helper_calibrate import check_1pct_speed_movement
            cmd_to_test = "f"  # Use forward as the test command for distance
            log_line(
                f"[CALIBRATE_DIST] Running preflight check: probing {cmd_to_test.upper()} at fixed 250ms and escalating score until movement is detected..."
            )
            preflight_result = check_1pct_speed_movement(
                robot=robot,
                vision=vision,
                world=world,
                cmd=cmd_to_test,
                movement_threshold_mm=0.15,
                sample_frames=3,
                sample_timeout_s=1.5,
                observe_sleep_s=0.02,
                control_sleep_s=0.04,
                duration_override_ms=250,
                log=log_line,
            )
            if not preflight_result:
                log_line(
                    f"[CALIBRATE_DIST] ⚠️  PREFLIGHT FAILED: no tested score produced detectable {cmd_to_test.upper()} movement at 250ms!"
                )
                log_line("[CALIBRATE_DIST] Check: Is the robot powered on? Are drive motors stalled or distance readings unstable?")
                log_line("[CALIBRATE_DIST] Aborting calibration because --preflight-check is enabled.")
                status = "aborted"
                abort_reason = "preflight_no_detectable_movement_at_250ms"
            else:
                log_line(
                    f"[CALIBRATE_DIST] ✓ Preflight passed: first detectable movement was {cmd_to_test.upper()} at "
                    f"{int(preflight_result.get('score_used') or 0)}% for {int(preflight_result.get('duration_ms') or 0)}ms."
                )
        elif status == "completed":
            log_line("[CALIBRATE_DIST] Preflight check skipped (use --preflight-check to enable).")

        initial_trial_pre_pose = None
        initial_trial_pre_obs_meta = None
        if status == "completed" and trials_planned > 0:
            _live_refresh(
                [
                    "TRIALS SETUP",
                    "Observing current distance...",
                ]
            )
            initial_trial_pre_pose, initial_trial_pre_obs_meta = _observe_pose_with_reobserve(
                vision=vision,
                world=world,
                samples=observe_samples,
                timeout_s=observe_timeout_s,
                on_vision_update=_live_refresh,
            )
            if initial_trial_pre_pose is None:
                initial_trial_pre_pose, initial_trial_pre_obs_meta = _recover_pose_for_trial(
                    vision=vision,
                    world=world,
                    robot=robot,
                    recent_acts=recent_acts,
                    trial_idx=1,
                    trials_requested=trials_planned,
                    stage_label="before trials",
                    trial_label="TRIALS SETUP",
                    on_vision_update=_live_refresh,
                )
            if initial_trial_pre_pose is None:
                status = "aborted"
                abort_reason = "pre_pose_unavailable_before_trials"
                log_line("[CALIBRATE_DIST] TRIALS SETUP: unable to observe current distance. Aborting.")
            else:
                setup_distance_mm = _coerce_finite_float(initial_trial_pre_pose.get("dist"))
                if target_dist_mm is None:
                    target_dist_mm = float(
                        setup_distance_mm
                        if setup_distance_mm is not None
                        else (REFERENCE_BRICK_DISTANCE_MM or PLOT_TITLE_DISTANCE_MM_DEFAULT)
                    )
                    log_line(f"[CALIBRATE_DIST] Using initial observed distance as target: {float(target_dist_mm):.2f}mm.")
                setup_curve_name = _dist_curve_display_name(dist_duration_cal)
                log_line(
                    _trials_setup_log_line(
                        observed_distance_mm=setup_distance_mm,
                        closest_curve_name=setup_curve_name,
                    )
                )
                prompted_settings = shared_prompt_calibration_run_settings(
                    prefix="CALIBRATE_DIST",
                    observed_distance_mm=setup_distance_mm,
                    default_speed_score=int(requested_speed_score),
                    default_min_duration_ms=int(min_duration_ms),
                    default_max_duration_ms=int(max_duration_ms),
                    duration_ceiling_ms=int(DURATION_CEILING_MS),
                    log=log_line,
                )
                speed_score = int(normalize_speed_score(prompted_settings["speed_score"]))
                requested_speed_score = int(speed_score)
                min_duration_ms = max(1, int(prompted_settings["min_duration_ms"]))
                max_duration_ms = max(int(min_duration_ms), int(prompted_settings["max_duration_ms"]))
                prompted_speed_score = bool(prompted_settings.get("prompted_speed_score"))
                prompted_duration_bounds = bool(prompted_settings.get("prompted_duration_bounds"))
                full_durations_ms = _build_duration_schedule(
                    trials=None,
                    min_duration_ms=min_duration_ms,
                    max_duration_ms=max_duration_ms,
                )
                trial_plan = _build_trial_plan(durations_ms=full_durations_ms, trials=trials_requested)
                durations_ms = _planned_durations_ms(trial_plan)
                trials_planned = len(trial_plan)
                stream_score_line = (
                    f"Score: distance curve (base {int(speed_score)}%)"
                    if isinstance(trial_speed_profile, dict)
                    else f"Score: {int(speed_score)}%"
                )
                config = {
                    "trials": int(trials_planned),
                    "requested_trials": None if trials_requested is None else int(trials_requested),
                    "repeat_pass_enabled": bool(repeat_pass_enabled),
                    "requested_repeat_trials": None if repeat_trials_requested is None else int(repeat_trials_requested),
                    "duration_ceiling_ms": int(DURATION_CEILING_MS),
                    "speed_score": int(speed_score),
                    "requested_speed_score": int(requested_speed_score),
                    "speed_score_source": "prompt" if bool(prompted_speed_score) else "arg",
                    "prompted_speed_score": bool(prompted_speed_score),
                    "min_duration_ms": int(min_duration_ms),
                    "max_duration_ms": int(max_duration_ms),
                    "prompted_duration_bounds": bool(prompted_duration_bounds),
                    "duration_step_ms": int(DURATION_STEP_MS_DEFAULT),
                    "distance_positive_cmd": str(_dist_cmd_for_positive_motion()),
                    "distance_negative_cmd": str(_dist_cmd_for_negative_motion()),
                    "observe_samples": int(observe_samples),
                    "observe_timeout_s": float(observe_timeout_s),
                    "post_act_settle_s": float(post_act_settle_s),
                    "trial_cmd_mode": str(TRIAL_CMD_AUTO),
                    "primary_trial_cmd_schedule": "auto_target_by_dist_sign",
                    "trial_cmd_rule": "if dist > target_dist_mm: forward else backward",
                    "plot_path": str(plot_path) if plot_path is not None else None,
                    "brick_distance_source": str(BRICK_DISTANCE_SOURCE),
                    "brick_distance_definition": str(BRICK_DISTANCE_DEFINITION),
                    "fast_target_budget_s": float(fast_target_budget_s),
                    "direction_check_min_delta_mm": float(DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM),
                    "abort_on_direction_mismatch": not bool(args.allow_direction_mismatch),
                    "trial_speed_mode": str(trial_speed_mode),
                    "trial_speed_score_source": "distance_curve" if isinstance(trial_speed_profile, dict) else "arg",
                }
                if REFERENCE_BRICK_DISTANCE_MM is not None:
                    config["reference_brick_distance_mm"] = float(REFERENCE_BRICK_DISTANCE_MM)
                if isinstance(trial_speed_profile, dict):
                    config["trial_speed_score_profile"] = json.loads(json.dumps(trial_speed_profile))
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
                    f"[CALIBRATE_DIST] distance motion sign: {_drive_label_for_cmd(_dist_cmd_for_positive_motion())} increases distance, "
                    f"{_drive_label_for_cmd(_dist_cmd_for_negative_motion())} decreases distance."
                )
                if bool(repeat_pass_enabled):
                    log_line(
                        f"[CALIBRATE_DIST] repeat_pass=enabled; repeats are deferred until all {int(trials_planned)} primary trial(s) finish."
                    )
                else:
                    log_line("[CALIBRATE_DIST] repeat_pass=disabled by default; use --repeat-trials N to enable.")
                log_line(
                    "[CALIBRATE_DIST] primary trial command selection: auto by distance target; "
                    f"if dist > target use {_drive_label_for_cmd(_dist_cmd_for_negative_motion())}, "
                    f"else use {_drive_label_for_cmd(_dist_cmd_for_positive_motion())}."
                )
                log_line(
                    f"[CALIBRATE_DIST] direction check: forward should DECREASE distance and backward should INCREASE distance "
                    f"(threshold {float(DISTANCE_DIRECTION_VERIFY_MIN_DELTA_MM):.2f}mm)."
                )
                if isinstance(trial_speed_profile, dict):
                    curve_points = list(trial_speed_profile.get("curve_points") or [])
                    if len(curve_points) >= 2:
                        first_point = curve_points[0]
                        last_point = curve_points[-1]
                        log_line(
                            f"[CALIBRATE_DIST] trial speed curve: "
                            f"<= {float(first_point.get('distance_mm')):.0f}mm -> {int(first_point.get('speed_score'))}% ; "
                            f">= {float(last_point.get('distance_mm')):.0f}mm -> {int(last_point.get('speed_score'))}% ; "
                            "linear between points."
                        )
                else:
                    log_line(
                        f"[CALIBRATE_DIST] trial speed mode: fixed; using score={int(speed_score)}% for every trial."
                    )
                if not bool(args.allow_direction_mismatch):
                    log_line("[CALIBRATE_DIST] direction mismatch guard: aborting once both drive directions are proven inverted.")
                _live_refresh(
                    [
                        "TRIALS SETUP",
                        f"Observed dist: {float(setup_distance_mm):.2f}mm"
                        if setup_distance_mm is not None
                        else "Observed dist: unknown",
                        f"Target dist: {float(target_dist_mm):.2f}mm",
                        f"Closest curve: {setup_curve_name}",
                        str(stream_score_line),
                        f"Durations: {int(min_duration_ms)}..{int(max_duration_ms)}ms",
                    ]
                )
        if target_dist_mm is None:
            target_dist_mm = float(REFERENCE_BRICK_DISTANCE_MM or PLOT_TITLE_DISTANCE_MM_DEFAULT)
        config["target_dist_mm"] = float(target_dist_mm)
        log_line(f"[CALIBRATE_DIST] target distance: dist={float(target_dist_mm):.2f}mm.")

        if status == "completed":
            for trial_idx, plan_step in enumerate(trial_plan, start=1):
                trial_label = _trial_label_text(trial_idx, trials_planned)
                duration_ms = max(1, int(_coerce_int(plan_step.get("duration_ms"), 1) or 1))
                pre_pose = (
                    dict(initial_trial_pre_pose)
                    if trial_idx == 1 and isinstance(initial_trial_pre_pose, dict)
                    else None
                )
                pre_obs_meta = (
                    dict(initial_trial_pre_obs_meta)
                    if trial_idx == 1 and isinstance(initial_trial_pre_obs_meta, dict)
                    else None
                )
                _live_refresh(
                    [
                        f"Trial {int(trial_idx)}/{int(trials_planned)}",
                        "Observing...",
                    ]
                )
                if pre_pose is None:
                    pre_pose, pre_obs_meta = _observe_pose_with_reobserve(
                        vision=vision,
                        world=world,
                        samples=observe_samples,
                        timeout_s=observe_timeout_s,
                        on_vision_update=_live_refresh,
                    )
                if pre_pose is None:
                    pre_pose, pre_obs_meta = _recover_pose_for_trial(
                        vision=vision,
                        world=world,
                        robot=robot,
                        recent_acts=recent_acts,
                        trial_idx=trial_idx,
                        trials_requested=trials_planned,
                        stage_label="before command selection",
                        trial_label=trial_label,
                        on_vision_update=_live_refresh,
                    )
                if pre_pose is None:
                    status = "aborted"
                    abort_reason = f"pre_pose_unavailable_trial_{trial_idx}"
                    log_line(f"[CALIBRATE_DIST] {trial_label}: recovery failed before command selection. Aborting.")
                    break
                curr_dist = float(pre_pose.get("dist") or 0.0)
                cmd = _auto_cmd_for_dist(
                    curr_dist,
                    target_dist_mm=float(target_dist_mm),
                )
                log_line(
                    f"[CALIBRATE_DIST] {trial_label}: auto target selection "
                    f"current_dist={curr_dist:.2f}mm target_dist={float(target_dist_mm):.2f}mm "
                    f"-> cmd={str(cmd).upper()} ({_drive_label_for_cmd(cmd)})."
                )
                _live_refresh(
                    [
                        f"Trial {int(trial_idx)}/{int(trials_planned)}",
                        f"Current dist: {float(curr_dist):.2f}mm",
                        f"Planned: {str(cmd).upper()} {int(duration_ms)}ms",
                    ]
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
                    distance_direction_check=distance_direction_check,
                    plotter=plotter,
                    initial_pre_pose=pre_pose,
                    initial_pre_obs_meta=pre_obs_meta,
                    dist_duration_cal=dist_duration_cal,
                    trial_speed_profile=trial_speed_profile,
                    stream_refresh_fn=_live_refresh,
                )
                if row is None:
                    status = "aborted"
                    abort_reason = str(trial_abort_reason or f"trial_failed_{trial_idx}")
                    break
                if bool(row.wrong_way):
                    log_line(f"[CALIBRATE_DIST] ⚠️  Trial {trial_idx}: wrong_way detected. Plotting it anyway.")
                trial_rows.append(row)
                _live_refresh(
                    [
                        f"Trial {int(trial_idx)}/{int(trials_planned)}",
                        f"Cmd: {str(row.cmd).upper()} duration={int(row.duration_ms)}ms",
                        f"Distance: {float(row.cmd_delta_mm):.2f}mm",
                    ]
                )
                _write_results(
                    results_path,
                    _build_payload(
                        config=config,
                        durations_ms=durations_ms,
                        trials=trial_rows,
                        status=status,
                        abort_reason=abort_reason,
                        distance_direction_check=distance_direction_check,
                        fast_alignment_analysis=_fast_alignment_analysis(
                            trial_rows,
                            target_budget_s=float(fast_target_budget_s),
                        ),
                    ),
                )
                if not bool(args.allow_direction_mismatch) and _should_abort_for_distance_inversion(distance_direction_check):
                    status = "aborted"
                    abort_reason = "distance_command_inversion_suspected"
                    summary_line = _distance_direction_check_summary_line(distance_direction_check)
                    if summary_line:
                        log_line(summary_line)
                    inferred_positive_cmd = _inferred_distance_positive_cmd(distance_direction_check)
                    if inferred_positive_cmd is not None:
                        log_line(
                            f"[CALIBRATE_DIST] Direction guard: observed distance-positive cmd={str(inferred_positive_cmd).upper()} "
                            f"but configured={str(_dist_cmd_for_positive_motion()).upper()}. "
                            f"Inspect drive command remap before re-running."
                        )
                    break

        if status == "completed" and bool(repeat_pass_enabled):
            repeat_plan_source = [row for row in trial_rows if str(getattr(row, "phase", "primary")) != "repeat"]
            repeat_plan = []
            if repeat_plan_source and int(repeat_trials_requested or 0) > 0:
                for idx in range(int(repeat_trials_requested or 0)):
                    repeat_plan.append(repeat_plan_source[idx % len(repeat_plan_source)])
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
                _live_refresh(
                    [
                        f"Repeat {int(repeat_idx)}/{len(repeat_plan)}",
                        f"Planned: {str(source_row.cmd).upper()} {int(source_row.duration_ms)}ms",
                        "Observing...",
                    ]
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
                    distance_direction_check=distance_direction_check,
                    plotter=plotter,
                    dist_duration_cal=dist_duration_cal,
                    trial_speed_profile=trial_speed_profile,
                    compare_to_distance=float(source_row.cmd_delta_mm),
                    stream_refresh_fn=_live_refresh,
                )
                if repeat_row is None:
                    status = "aborted"
                    abort_reason = str(repeat_abort_reason or f"repeat_trial_failed_{repeat_idx}")
                    break
                if bool(repeat_row.wrong_way):
                    log_line(f"[CALIBRATE_DIST] ⚠️  Repeat {repeat_idx}: wrong_way detected. Plotting it anyway.")
                trial_rows.append(repeat_row)
                _live_refresh(
                    [
                        f"Repeat {int(repeat_idx)}/{len(repeat_plan)}",
                        f"Cmd: {str(repeat_row.cmd).upper()} duration={int(repeat_row.duration_ms)}ms",
                        f"Distance: {float(repeat_row.cmd_delta_mm):.2f}mm",
                    ]
                )
                _write_results(
                    results_path,
                    _build_payload(
                        config=config,
                        durations_ms=durations_ms,
                        trials=trial_rows,
                        status=status,
                        abort_reason=abort_reason,
                        distance_direction_check=distance_direction_check,
                        fast_alignment_analysis=_fast_alignment_analysis(
                            trial_rows,
                            target_budget_s=float(fast_target_budget_s),
                        ),
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
                distance_direction_check=distance_direction_check,
                fast_alignment_analysis=_fast_alignment_analysis(
                    trial_rows,
                    target_budget_s=float(fast_target_budget_s),
                ),
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
        if stream_server is not None:
            try:
                close_fn = getattr(stream_server, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass

    direction_summary = _distance_direction_check_summary_line(distance_direction_check)
    if direction_summary:
        log_line(direction_summary)
    _log_fast_alignment_analysis(
        _fast_alignment_analysis(
            trial_rows,
            target_budget_s=float(fast_target_budget_s),
        )
    )
    log_line(f"[CALIBRATE_DIST] Wrote results to {results_path}")
    if plot_path is not None:
        log_line(f"[CALIBRATE_DIST] Updated plot at {plot_path}")
    if status != "completed":
        detail = f" reason={abort_reason}" if abort_reason else ""
        log_line(f"[CALIBRATE_DIST] Finished with status={status}{detail}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    _exit_as_script(main())
