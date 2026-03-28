#!/usr/bin/env python3
"""
Minimal y-axis duration probe.

Observe the current y offset with a 3-frame confidence read, send one mast act
at a configured trial speed policy and random duration, observe again, and plot
command-direction distance traveled in mm against duration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
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

from .helper_calibrate import (
    CALIBRATION_DURATION_LIMIT_MS,
    CalibrationLivePlot,
    build_payload as build_shared_payload,
    cleanup_old_run_files,
    coerce_finite_float as shared_coerce_finite_float,
    coerce_float as shared_coerce_float,
    coerce_int as shared_coerce_int,
    ensure_run_dir,
    get_shared_stream_runtime,
    load_calibration_trial_speed_profile as shared_load_calibration_trial_speed_profile,
    observed_brick_distances_mm as shared_observed_brick_distances_mm,
    prediction_closeness_percentage as shared_prediction_closeness_percentage,
    prompt_calibration_run_settings as shared_prompt_calibration_run_settings,
    prepare_shared_stream_state,
    plot_offsets as shared_plot_offsets,
    resolve_calibration_trial_speed_score as shared_resolve_calibration_trial_speed_score,
    trial_label_text as shared_trial_label_text,
    write_results as shared_write_results,
)
from helper_robot_control import Robot
import helper_xyz_coords
from helper_manual_config import load_manual_training_config
from helper_stream_server import format_stream_url
from helper_streaming import start_stream_server
from helper_vision_leia import LeiaVision

# The Yolo brick detector is used by manual training; it's optional here and
# only imported if the module is available.  Using Yolo often gives much more
# robust cyan‑brick tracking than the simple LeiaVision edge detector.
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

OBSERVE_SLEEP_S = 0.02
OBSERVE_TIMEOUT_S = 1.8
POST_ACT_SETTLE_S = 0.10
OBSERVE_SAMPLES_DEFAULT = 3
Y_AXIS_SWEET_SPOT_MM_DEFAULT = 8.1
DURATION_CEILING_MS = CALIBRATION_DURATION_LIMIT_MS
DURATION_SECTION_STEP_MS_DEFAULT = 100
DURATION_SECTION_SPAN_MS_DEFAULT = 50
DURATION_SAMPLES_PER_SECTION_DEFAULT = 5
# This helper calibrates the camera-frame vertical offset used during mast
# setup/recovery. Its local convention is positive-downward, so mast_up is the
# positive-motion command for this script.
Y_AXIS_POSITIVE_CMD_DEFAULT = "u"

# reference distance associated with the current regression equation.  The
# calibration data assume the brick was this far from the camera (mm).  This is
# written to the world model so downstream code knows its validity range.
REFERENCE_BRICK_DISTANCE_MM: float | None = None
PLOT_COLOR_BY_CMD = {
    "u": "#1f77b4",
    "d": "#ff7f0e",
}
TRIAL_ALTERNATING_START_CMD = "u"
CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM = 0.03
UP_TRIAL_BAND_MIN_MM = 8.0
UP_TRIAL_BAND_MAX_MM = 16.0
DOWN_TRIAL_BAND_MIN_MM = 2.0
DOWN_TRIAL_BAND_MAX_MM = 8.0
RESET_DURATION_MIN_MS = 250
RESET_DURATION_MAX_MS = 500
RESET_DURATION_SECTION_STEP_MS = 100
RESET_DURATION_SECTION_SPAN_MS = 50
RESET_DURATION_SAMPLES_PER_SECTION = 5
MIN_LITE_UNIQUE_FRAMES = 3
REOBSERVE_HOLD_S = 0.12
REOBSERVE_ROUNDS = 2
RELAXED_OBSERVE_TIMEOUT_S = 2.8
RECOVERY_MAX_INVERSE_ACTS = 5
RECOVERY_OBSERVE_TIMEOUT_S = 1.0
# Leave defaults None so live JSON is explicit and plot PNG output stays opt-in.
RESULTS_FILE_DEFAULT: str | None = None
PLOT_FILE_DEFAULT: str | None = None

_MANUAL_CONFIG = load_manual_training_config()
STREAM_HOST = str(_MANUAL_CONFIG.get("stream_host", "127.0.0.1"))
STREAM_PORT = int(_MANUAL_CONFIG.get("stream_port", 5000))
STREAM_FPS = int(_MANUAL_CONFIG.get("stream_fps", 10))
STREAM_JPEG_QUALITY = int(_MANUAL_CONFIG.get("stream_jpeg_quality", 85))
STREAM_IMG_WIDTH = int(_MANUAL_CONFIG.get("stream_img_width", 1600))
ANSI_ORANGE_BRIGHT = "\033[38;5;208m"
ANSI_RESET = "\033[0m"

# Folder where calibration runs deposit live files.
RUN_DIR_ARUCO = Path("Runs - aruco")
RUN_DIR_CYAN = Path("Runs - cyan")


def _run_dir_for_vision(vision_mode: str | None) -> Path:
    mode = str(vision_mode or "").strip().lower()
    if mode == "aruco":
        return Path(RUN_DIR_ARUCO)
    return Path(RUN_DIR_CYAN)


def _cleanup_old_run_files():
    cleanup_old_run_files(
        preserve_live_files={
            "calibrate_x_live.json",
            "calibrate_y_live.json",
            "calibrate_dist_live.json",
        },
    )


def _ensure_run_dir(run_dir: Path):
    ensure_run_dir(
        run_dir=Path(run_dir),
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


@dataclass
class TrialResult:
    trial: int
    duration_ms: int
    cmd: str
    score_requested: int
    cmd_sent: str | None
    pwm: int | None
    power: float | None
    pre_y_mm: float
    post_y_mm: float
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
    phase: str = "trial"
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
    pre_y_mm: float
    post_y_mm: float
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
    phase: str = "trial"
    source_trial: int | None = None


def log_line(message: str) -> None:
    print(str(message), flush=True)


def _orange_text(text: str) -> str:
    return f"{ANSI_ORANGE_BRIGHT}{str(text)}{ANSI_RESET}"


def _parse_float_list(raw_text: str) -> list[float]:
    values: list[float] = []
    for token in str(raw_text or "").replace(";", ",").split(","):
        text = str(token or "").strip()
        if not text:
            continue
        values.append(float(text))
    return values


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


def _load_y_duration_calibration(path: Path | None = None) -> dict | None:
    curve_path = Path(path) if path is not None else (Path(__file__).resolve().parents[1] / "world_model_up_down_curve.json")
    if not curve_path.exists():
        return None
    try:
        payload = json.loads(curve_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    curves: list[dict] = []
    for curve_key, calib in payload.items():
        if not isinstance(calib, dict):
            continue
        by_cmd = calib.get("by_cmd")
        if not isinstance(by_cmd, dict):
            continue
        cmd_rows: dict[str, dict] = {}
        for cmd in ("u", "d"):
            row = by_cmd.get(cmd)
            if not isinstance(row, dict):
                continue
            try:
                cmd_rows[cmd] = {
                    "slope_mm_per_ms": float(row.get("slope_mm_per_ms")),
                    "intercept_mm": float(row.get("intercept_mm")),
                }
            except Exception:
                continue
        if not cmd_rows:
            continue
        curves.append(
            {
                "curve_key": str(curve_key),
                "curve_name": _y_curve_display_name(str(curve_key), calib),
                "reference_distance_mm": shared_coerce_finite_float(calib.get("reference_distance_mm")),
                "speed_score_pct": shared_coerce_finite_float(calib.get("speed_score_pct")),
                "source": str(calib.get("source") or curve_key),
                "by_cmd": cmd_rows,
            }
        )
    return {"curves": curves} if curves else None


def _y_curve_display_name(curve_key: str, calibration: dict | None) -> str:
    base_name = str(curve_key or "curve").strip() or "curve"
    calib = calibration if isinstance(calibration, dict) else {}
    reference_distance_mm = shared_coerce_finite_float(calib.get("reference_distance_mm"))
    speed_score_pct = shared_coerce_finite_float(calib.get("speed_score_pct"))
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


def _closest_y_duration_curve(
    *,
    y_calibration: dict | None,
    observed_distance_mm: float | None,
) -> dict | None:
    curves = y_calibration.get("curves") if isinstance(y_calibration, dict) else None
    if not isinstance(curves, list) or not curves:
        return None
    observed_distance = shared_coerce_finite_float(observed_distance_mm)
    if observed_distance is None:
        return dict(curves[0]) if isinstance(curves[0], dict) else None
    curves_with_reference = [
        dict(curve)
        for curve in curves
        if isinstance(curve, dict) and shared_coerce_finite_float(curve.get("reference_distance_mm")) is not None
    ]
    if not curves_with_reference:
        return dict(curves[0]) if isinstance(curves[0], dict) else None
    return min(
        curves_with_reference,
        key=lambda curve: (
            abs(float(curve.get("reference_distance_mm")) - float(observed_distance)),
            str(curve.get("curve_name") or curve.get("curve_key") or ""),
        ),
    )


def _closest_y_duration_curve_name(
    *,
    y_calibration: dict | None,
    observed_distance_mm: float | None,
) -> str:
    curve = _closest_y_duration_curve(
        y_calibration=y_calibration,
        observed_distance_mm=observed_distance_mm,
    )
    if not isinstance(curve, dict):
        return "no_curve"
    return str(curve.get("curve_name") or curve.get("curve_key") or "curve")


def _predict_movement_from_curve(
    *,
    cmd: str,
    duration_ms: int,
    y_calibration: dict | None,
    observed_distance_mm: float | None = None,
) -> tuple[float | None, str]:
    """Predict movement distance using existing calibration curve.
    
    Returns:
        (predicted_distance_mm, curve_source)
    """
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    selected_curve = _closest_y_duration_curve(
        y_calibration=y_calibration,
        observed_distance_mm=observed_distance_mm,
    )
    if not isinstance(selected_curve, dict):
        return None, "no_curve"
    curve_name = str(selected_curve.get("curve_name") or selected_curve.get("curve_key") or "curve")
    row = selected_curve.get("by_cmd", {}).get(cmd_key) if isinstance(selected_curve.get("by_cmd"), dict) else {}
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
    """Calculate prediction comparison metrics."""
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


def _predict_duration_for_target_delta_mm(
    *,
    cmd: str,
    abs_delta_mm: float,
    duration_min_ms: int,
    duration_max_ms: int,
    y_calibration: dict | None,
    observed_distance_mm: float | None = None,
) -> int:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    selected_curve = _closest_y_duration_curve(
        y_calibration=y_calibration,
        observed_distance_mm=observed_distance_mm,
    )
    if not isinstance(selected_curve, dict):
        return int(max(duration_min_ms, min(duration_max_ms, duration_min_ms)))
    row = selected_curve.get("by_cmd", {}).get(cmd_key) if isinstance(selected_curve.get("by_cmd"), dict) else {}
    if not isinstance(row, dict):
        return int(max(duration_min_ms, min(duration_max_ms, duration_min_ms)))
    try:
        slope = float(row.get("slope_mm_per_ms"))
        intercept = float(row.get("intercept_mm"))
    except Exception:
        return int(max(duration_min_ms, min(duration_max_ms, duration_min_ms)))
    if slope <= 1e-9:
        return int(max(duration_min_ms, min(duration_max_ms, duration_min_ms)))
    predicted = (max(0.0, float(abs_delta_mm)) - float(intercept)) / float(slope)
    predicted = max(float(duration_min_ms), min(float(duration_max_ms), float(predicted)))
    return int(round(predicted))


def _supports_ansi_color() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _colorize(text: str, color_code: str) -> str:
    if not _supports_ansi_color():
        return str(text)
    return f"{str(color_code)}{str(text)}\033[0m"


def _highlight_score_text(text: str) -> str:
    return _colorize(str(text), "\033[97m")


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
    detail_text = _colorize(detail_body, "\033[90m")
    return f"{_highlight_score_text(f'{int(score)}%')} {detail_text}"


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


def _trials_setup_log_line(
    *,
    observed_distance_mm: float | None,
    closest_curve_name: str,
) -> str:
    observed_distance = shared_coerce_finite_float(observed_distance_mm)
    observed_distance_text = (
        f"{float(observed_distance):.2f}mm" if observed_distance is not None else "unknown"
    )
    message = (
        "[TRIALS SETUP] "
        f"observed_distance={observed_distance_text} "
        f"closest_speed_curve={str(closest_curve_name or 'no_curve')}"
    )
    return _colorize(message, "\033[92m")


def _trial_result_status_text(*, useful: bool) -> str:
    return _colorize("USEFUL", "\033[92m") if bool(useful) else _colorize("FAIL", "\033[91m")


def _trial_result_label(
    *,
    trial_idx: int,
    trials_planned: int,
) -> str:
    return f"Trial {int(trial_idx)}/{int(trials_planned)}"


def _coerce_float(value, fallback=None):
    return shared_coerce_float(value, fallback)


def _coerce_int(value, fallback=None):
    return shared_coerce_int(value, fallback)


def _normalize_cmd(value: str, *, allow_auto: bool = False) -> str:
    text = str(value or "").strip().lower()
    if allow_auto and text in ("auto", "center"):
        return "auto"
    if text not in ("u", "d"):
        raise ValueError("Allowed y-axis commands are only 'u', 'd', 'auto', or 'center'.")
    return text


def _world_step_label(world) -> str:
    step_state = getattr(world, "step_state", None)
    step_value = getattr(step_state, "value", step_state)
    step_text = str(step_value or "ALIGN_BRICK").strip()
    return step_text or "ALIGN_BRICK"


def _measurement_offset_y_mm(measurement: dict, *, measurement_space: str) -> float:
    if not isinstance(measurement, dict):
        raise TypeError("measurement must be a dict")
    if str(measurement_space or "").strip().lower() == "world":
        # WorldModel y_axis uses the global normal-number-line convention
        # (positive up). This helper uses camera-frame vertical offset
        # semantics (positive down), so convert exactly once at intake.
        y_val = measurement.get("y_axis", measurement.get("offset_y", 0.0))
        return -float(y_val or 0.0)
    # Raw vision `cam_h` already uses positive-down camera semantics.
    return float(measurement.get("offset_y", measurement.get("cam_h", 0.0)) or 0.0)


def _pose_from_measurement(
    measurement: dict,
    *,
    obs_ts: float,
    pose_source: str,
    measurement_space: str = "world",
    lite_required_frames: int | None = None,
) -> dict | None:
    if not isinstance(measurement, dict) or not bool(measurement.get("visible")):
        return None
    try:
        return {
            "offset_y": float(_measurement_offset_y_mm(measurement, measurement_space=str(measurement_space))),
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
        # Smoothed frame averages come from the raw detector frame buffer, so
        # their vertical offset is already in camera space (positive down).
        measurement_space="camera",
        lite_required_frames=required_frames,
    )


def _brick_pose_from_world(world, *, obs_ts: float) -> dict | None:
    brick = getattr(world, "brick", None)
    if not isinstance(brick, dict):
        return None
    return _pose_from_measurement(
        brick,
        obs_ts=obs_ts,
        pose_source="brick_state",
        measurement_space="world",
    )


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


def _pose_meets_multiframe_requirement(
    pose: dict | None,
    *,
    required_samples: int,
    required_lite_frames: int = MIN_LITE_UNIQUE_FRAMES,
) -> bool:
    if not isinstance(pose, dict):
        return False
    if str(pose.get("pose_source") or "").strip().lower() != "lite_smoothed":
        return False
    samples_used = _coerce_int(pose.get("samples_used"), 0)
    lite_frames = _coerce_int(pose.get("lite_required_frames"), 0)
    return bool(
        int(samples_used) >= int(max(1, int(required_samples)))
        and int(lite_frames) >= int(max(1, int(required_lite_frames)))
    )


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
            if callable(on_vision_update):
                on_vision_update()
            now = time.time()
            if min_sample_time is not None and now < float(min_sample_time):
                time.sleep(OBSERVE_SLEEP_S)
                continue
            pose = _lite_pose_from_world(world, step=step_label, samples=int(samples), obs_ts=now)
        except Exception:
            pose = None
        if pose is None:
            time.sleep(OBSERVE_SLEEP_S)
            continue
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
    on_vision_update=None,
) -> tuple[dict | None, dict]:
    target_samples = max(1, int(samples))
    required_samples = max(target_samples, int(MIN_LITE_UNIQUE_FRAMES))
    strict_pose = read_pose(
        vision,
        world,
        samples=target_samples,
        timeout_s=float(timeout_s),
        min_sample_time=min_sample_time,
        min_samples_required=required_samples,
        on_vision_update=on_vision_update,
    )
    if _pose_meets_multiframe_requirement(
        strict_pose,
        required_samples=required_samples,
    ):
        return strict_pose, {
            "mode": "trial_full",
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
            min_samples_required=required_samples,
            on_vision_update=on_vision_update,
        )
        if not _pose_meets_multiframe_requirement(
            relaxed_pose,
            required_samples=required_samples,
        ):
            if round_idx < rounds:
                log_line(
                    f"[CALIBRATE_Y] Observation hold/reobserve {round_idx}/{rounds}: still no usable pose."
                )
            continue

        mode = "hold_reobserve_full"
        log_line(
            f"[CALIBRATE_Y] Observation rescue: accepted {int(relaxed_pose.get('samples_used') or 0)}/{int(target_samples)} samples "
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


def _command_delta_mm(cmd: str, pre_y_mm: float, post_y_mm: float) -> float:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == _y_cmd_for_positive_motion():
        return float(post_y_mm) - float(pre_y_mm)
    return float(pre_y_mm) - float(post_y_mm)


def _movement_metrics(cmd: str, pre_y_mm: float, post_y_mm: float) -> dict:
    raw_delta_mm = float(post_y_mm) - float(pre_y_mm)
    signed_cmd_delta_mm = _command_delta_mm(cmd, pre_y_mm, post_y_mm)
    return {
        "raw_delta_mm": float(raw_delta_mm),
        "signed_cmd_delta_mm": float(signed_cmd_delta_mm),
        "cmd_delta_mm": abs(float(signed_cmd_delta_mm)),
        "wrong_way": bool(float(signed_cmd_delta_mm) < 0.0),
    }


def _inverse_cmd(cmd: str | None) -> str | None:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "u":
        return "d"
    if cmd_key == "d":
        return "u"
    return None


def _y_cmd_for_positive_motion() -> str:
    return _normalize_cmd(Y_AXIS_POSITIVE_CMD_DEFAULT, allow_auto=False)


def _y_cmd_for_negative_motion() -> str:
    cmd = str(_inverse_cmd(_y_cmd_for_positive_motion()) or "d")
    return cmd


def _auto_cmd_for_y(
    curr_y_mm: float,
    *,
    center_y_mm: float = 0.0,
    deadband_mm: float = 0.5,
    fallback_cmd: str = "d",
) -> str:
    band_mm = abs(float(deadband_mm))
    if float(curr_y_mm) < (float(center_y_mm) - float(band_mm)):
        return _y_cmd_for_positive_motion()
    if float(curr_y_mm) > (float(center_y_mm) + float(band_mm)):
        return _y_cmd_for_negative_motion()
    return _normalize_cmd(fallback_cmd, allow_auto=False)


def _mast_label_for_cmd(cmd: str) -> str:
    return "mast_up" if _normalize_cmd(cmd, allow_auto=False) == "u" else "mast_down"


def _expected_camera_direction_for_cmd(cmd: str) -> str:
    return "down" if _normalize_cmd(cmd, allow_auto=False) == "u" else "up"


def _camera_direction_from_raw_delta(raw_delta_mm: float, *, threshold_mm: float) -> str | None:
    raw_delta = float(raw_delta_mm)
    if abs(float(raw_delta)) < max(0.0, float(threshold_mm)):
        return None
    # Within this helper, y deltas use camera-frame semantics: positive means
    # the brick moved downward on camera, negative means upward.
    return "down" if float(raw_delta) > 0.0 else "up"


def _camera_direction_human(direction: str | None, *, adverb: bool = False) -> str:
    direction_key = str(direction or "").strip().lower()
    if direction_key == "up":
        return "upwards" if adverb else "up"
    if direction_key == "down":
        return "downwards" if adverb else "down"
    return "inconclusive"


def _trial_prefixed_detail_prefix(prefix: str, trial_label: str) -> str:
    prefix_text = str(prefix or "").strip()
    label_text = str(trial_label or "").strip()
    if not prefix_text:
        return label_text
    if not label_text:
        return prefix_text
    if prefix_text.startswith("["):
        bracket_end = prefix_text.find("]")
        if bracket_end >= 0:
            channel = prefix_text[: bracket_end + 1]
            detail = prefix_text[bracket_end + 1 :].strip()
            if detail:
                return f"{channel} {label_text}: {detail}"
            return f"{channel} {label_text}:"
    return f"{prefix_text} {label_text}:"


def _log_command_inversion_detail(
    *,
    prefix: str,
    trial_label: str,
    logical_cmd: str,
    wire_cmd: str,
    raw_delta_mm: float | None = None,
    threshold_mm: float = CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM,
) -> None:
    logical_cmd_key = _normalize_cmd(logical_cmd, allow_auto=False)
    wire_cmd_key = _normalize_cmd(wire_cmd, allow_auto=False)
    expected_direction = _expected_camera_direction_for_cmd(logical_cmd_key)
    wire_direction = _expected_camera_direction_for_cmd(wire_cmd_key)
    expected_text = _camera_direction_human(expected_direction, adverb=True)
    wire_text = _camera_direction_human(wire_direction, adverb=True)
    detail_prefix = _trial_prefixed_detail_prefix(prefix, trial_label)

    if raw_delta_mm is None:
        log_line(
            f"{detail_prefix} logical {_mast_label_for_cmd(logical_cmd_key)} expects the brick to go "
            f"{expected_text} on camera, but wire {_mast_label_for_cmd(wire_cmd_key)} "
            f"would drive it {wire_text}."
        )
        return

    raw_delta = float(raw_delta_mm)
    observed_direction = _camera_direction_from_raw_delta(raw_delta, threshold_mm=float(threshold_mm))
    if observed_direction is None:
        log_line(
            f"{detail_prefix} expected the brick to go {expected_text} on camera, "
            f"but observed raw_delta={float(raw_delta):+.2f}mm which is below the "
            f"{float(threshold_mm):.2f}mm direction threshold."
        )
        return

    observed_text = _camera_direction_human(observed_direction, adverb=True)
    if observed_direction == expected_direction:
        observed_clause = f"and observed it move {observed_text}"
    else:
        observed_clause = f"but observed it move {observed_text}"
    log_line(
        f"{detail_prefix} logical {_mast_label_for_cmd(logical_cmd_key)} expects the brick to go {expected_text} on camera, "
        f"{observed_clause} ({abs(float(raw_delta)):.2f}mm; raw_delta={float(raw_delta):+.2f}mm). "
        f"Wire {_mast_label_for_cmd(wire_cmd_key)} implies {wire_text} camera motion."
    )


def _should_log_command_inversion_detail(
    *,
    logical_cmd: str,
    wire_cmd: str,
    raw_delta_mm: float | None,
    threshold_mm: float = CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM,
) -> bool:
    logical_cmd_key = _normalize_cmd(logical_cmd, allow_auto=False)
    wire_cmd_key = _normalize_cmd(wire_cmd, allow_auto=False)
    if logical_cmd_key == wire_cmd_key:
        return False
    if raw_delta_mm is None:
        return True
    observed_direction = _camera_direction_from_raw_delta(float(raw_delta_mm), threshold_mm=float(threshold_mm))
    if observed_direction is None:
        return True
    wire_direction = _expected_camera_direction_for_cmd(wire_cmd_key)
    return str(observed_direction) != str(wire_direction)


def _new_camera_direction_check_entry(cmd: str) -> dict:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    return {
        "label": str(_mast_label_for_cmd(cmd_key)),
        "expected_camera_direction": str(_expected_camera_direction_for_cmd(cmd_key)),
        "status": "pending",
        "observations": 0,
        "evidence_count": 0,
        "match_count": 0,
        "mismatch_count": 0,
        "inconclusive_count": 0,
        "last_raw_delta_mm": None,
        "last_cmd_sent": None,
        "last_trial_label": None,
    }


def _new_camera_direction_check_state() -> dict:
    return {
        "movement_threshold_mm": float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM),
        "by_cmd": {
            "u": _new_camera_direction_check_entry("u"),
            "d": _new_camera_direction_check_entry("d"),
        },
    }


def _camera_direction_check_entry(check_state: dict | None, cmd: str) -> dict | None:
    if not isinstance(check_state, dict):
        return None
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    by_cmd = check_state.setdefault("by_cmd", {})
    entry = by_cmd.get(cmd_key)
    if not isinstance(entry, dict):
        entry = _new_camera_direction_check_entry(cmd_key)
        by_cmd[cmd_key] = entry
    return entry


def _refresh_camera_direction_entry_status(entry: dict | None) -> str:
    if not isinstance(entry, dict):
        return "pending"
    matches = max(0, int(_coerce_int(entry.get("match_count"), 0) or 0))
    mismatches = max(0, int(_coerce_int(entry.get("mismatch_count"), 0) or 0))
    inconclusive = max(0, int(_coerce_int(entry.get("inconclusive_count"), 0) or 0))
    if matches > 0 and mismatches > 0:
        status = "mixed"
    elif mismatches > 0:
        status = "mismatch"
    elif matches > 0:
        status = "verified"
    elif inconclusive > 0:
        status = "inconclusive"
    else:
        status = "pending"
    entry["status"] = str(status)
    return str(status)


def _record_camera_direction_check(
    check_state: dict | None,
    *,
    trial_label: str,
    cmd: str,
    cmd_sent: str | None,
    raw_delta_mm: float,
) -> None:
    entry = _camera_direction_check_entry(check_state, cmd)
    if not isinstance(entry, dict):
        return

    threshold_mm = max(
        0.0,
        float(_coerce_float((check_state or {}).get("movement_threshold_mm"), CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM) or 0.0),
    )
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    expected_direction = str(entry.get("expected_camera_direction") or _expected_camera_direction_for_cmd(cmd_key))
    actual_direction = _camera_direction_from_raw_delta(float(raw_delta_mm), threshold_mm=threshold_mm)
    prior_status = _refresh_camera_direction_entry_status(entry)

    entry["observations"] = max(0, int(_coerce_int(entry.get("observations"), 0) or 0)) + 1
    entry["last_raw_delta_mm"] = float(raw_delta_mm)
    entry["last_cmd_sent"] = str(cmd_sent).strip().lower() if cmd_sent is not None else None
    entry["last_trial_label"] = str(trial_label)

    wire_cmd = str(cmd_sent or cmd_key).strip().upper() or str(cmd_key).upper()

    if actual_direction is None:
        entry["inconclusive_count"] = max(0, int(_coerce_int(entry.get("inconclusive_count"), 0) or 0)) + 1
        status = _refresh_camera_direction_entry_status(entry)
        if prior_status in {"pending", "inconclusive"} and int(entry.get("inconclusive_count") or 0) == 1:
            log_line(
                f"[CALIBRATE_Y_DIRECTION_CHECK] {trial_label}: pending {_mast_label_for_cmd(cmd_key)} verification; "
                f"raw_delta={float(raw_delta_mm):+.2f}mm is below {float(threshold_mm):.2f}mm."
            )
        entry["status"] = str(status)
        return

    entry["evidence_count"] = max(0, int(_coerce_int(entry.get("evidence_count"), 0) or 0)) + 1
    if actual_direction == expected_direction:
        entry["match_count"] = max(0, int(_coerce_int(entry.get("match_count"), 0) or 0)) + 1
    else:
        entry["mismatch_count"] = max(0, int(_coerce_int(entry.get("mismatch_count"), 0) or 0)) + 1
    status = _refresh_camera_direction_entry_status(entry)

    if actual_direction == expected_direction:
        if prior_status != "verified":
            log_line(
                f"[CALIBRATE_Y_DIRECTION_CHECK] {trial_label}: verified {_mast_label_for_cmd(cmd_key)} "
                f"-> brick moved {str(actual_direction).upper()} on camera "
                f"(raw_delta={float(raw_delta_mm):+.2f}mm, wire={wire_cmd})."
            )
        return

    if prior_status != status or int(entry.get("mismatch_count") or 0) == 1:
        log_line(
            f"[CALIBRATE_Y_DIRECTION_CHECK] {trial_label}: WARNING {_mast_label_for_cmd(cmd_key)} "
            f"moved brick {str(actual_direction).upper()} on camera; expected {str(expected_direction).upper()} "
            f"(raw_delta={float(raw_delta_mm):+.2f}mm, wire={wire_cmd})."
        )


def _camera_direction_check_hint(check_state: dict | None, cmd: str) -> str | None:
    entry = _camera_direction_check_entry(check_state, cmd)
    if not isinstance(entry, dict):
        return None
    status = _refresh_camera_direction_entry_status(entry)
    label = str(entry.get("label") or _mast_label_for_cmd(cmd))
    expected_direction = str(entry.get("expected_camera_direction") or _expected_camera_direction_for_cmd(cmd)).upper()
    if status == "verified":
        return f"{label} already verified this run (expected camera motion {expected_direction})."
    if status == "mismatch":
        return f"{label} previously mismatched camera motion this run (expected {expected_direction})."
    if status == "mixed":
        return f"{label} has mixed camera-direction evidence this run (expected {expected_direction})."
    if status == "inconclusive":
        return f"{label} not yet verified this run; only sub-threshold motion has been observed so far."
    return f"{label} has not yet been camera-verified this run (expected {expected_direction})."


def _camera_direction_check_summary_line(check_state: dict | None) -> str | None:
    if not isinstance(check_state, dict):
        return None
    parts = []
    for cmd_key in ("u", "d"):
        entry = _camera_direction_check_entry(check_state, cmd_key)
        if not isinstance(entry, dict):
            continue
        status = _refresh_camera_direction_entry_status(entry)
        label = str(entry.get("label") or _mast_label_for_cmd(cmd_key))
        expected_direction = str(entry.get("expected_camera_direction") or _expected_camera_direction_for_cmd(cmd_key))
        parts.append(
            f"{label}={status} expected={expected_direction} "
            f"match={int(_coerce_int(entry.get('match_count'), 0) or 0)} "
            f"mismatch={int(_coerce_int(entry.get('mismatch_count'), 0) or 0)}"
        )
    if not parts:
        return None
    threshold_mm = float(_coerce_float(check_state.get("movement_threshold_mm"), CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM) or 0.0)
    return (
        f"[CALIBRATE_Y_DIRECTION_CHECK] Summary: threshold={float(threshold_mm):.2f}mm; "
        + "; ".join(parts)
    )


def _plot_color_for_cmd(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    return str(PLOT_COLOR_BY_CMD.get(cmd_key) or "#4c566a")


def _plot_series_key(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    return str(cmd_key)


def _plot_series_label(cmd: str, kind: str | None = None, repeat_status: str | None = None) -> str:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    return _mast_label_for_cmd(cmd_key)


def _plot_offsets(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    return shared_plot_offsets(xs, ys)


def _coerce_finite_float(value) -> float | None:
    return shared_coerce_finite_float(value)


def _plot_title_text(brick_distances_mm: list[float]) -> str:
    detail = f"Brick distance = {BRICK_DISTANCE_SOURCE} ({BRICK_DISTANCE_DEFINITION})"
    observed = [float(value) for value in brick_distances_mm if _coerce_finite_float(value) is not None]
    if not observed:
        return f"Y Calibration\n{detail}"
    latest_mm = float(observed[-1])
    median_mm = float(statistics.median(observed))
    min_mm = float(min(observed))
    max_mm = float(max(observed))
    return (
        "Y Calibration\n"
        f"{detail}; latest={latest_mm:.1f}mm median={median_mm:.1f}mm "
        f"range={min_mm:.1f}..{max_mm:.1f}mm"
    )


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


def _scheduled_trial_cmd_for_trial(trial_idx: int) -> str:
    start_cmd = _normalize_cmd(TRIAL_ALTERNATING_START_CMD, allow_auto=False)
    if int(trial_idx) % 2 == 1:
        return str(start_cmd)
    return str(_inverse_cmd(start_cmd) or "u")


def _center_target_status_line(y_mm: float, *, target_y_mm: float) -> str:
    error_mm = float(y_mm) - float(target_y_mm)
    if abs(float(error_mm)) <= 0.5:
        status_text = _colorize("Near target", "\033[92m")
    else:
        status_text = _colorize("Off target", "\033[93m")
    return (
        f"y_axis: {float(y_mm):+.2f}. {status_text} center target "
        f"({float(target_y_mm):+.2f}mm; error {float(error_mm):+.2f}mm)."
    )


def _trial_band_for_cmd(cmd: str) -> dict:
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    if cmd_key == "u":
        return {
            "min_mm": float(UP_TRIAL_BAND_MIN_MM),
            "max_mm": float(UP_TRIAL_BAND_MAX_MM),
        }
    return {
        "min_mm": float(DOWN_TRIAL_BAND_MIN_MM),
        "max_mm": float(DOWN_TRIAL_BAND_MAX_MM),
    }


def _trial_band_correction_cmd(trial_cmd: str, current_y_mm: float) -> str:
    band = _trial_band_for_cmd(trial_cmd)
    y_val = float(current_y_mm)
    if y_val < float(band["min_mm"]):
        return "u"
    if y_val > float(band["max_mm"]):
        return "d"
    return _normalize_cmd(trial_cmd, allow_auto=False)


def _trial_band_status_line(cmd: str, current_y_mm: float) -> tuple[bool, str]:
    band = _trial_band_for_cmd(cmd)
    cmd_key = _normalize_cmd(cmd, allow_auto=False)
    within = float(band["min_mm"]) <= float(current_y_mm) <= float(band["max_mm"])
    status = "Within" if within else "Not within"
    line = (
        f"{status} trial band for {_mast_label_for_cmd(cmd_key)}: "
        f"y_axis={float(current_y_mm):+.2f}mm target_band="
        f"{float(band['min_mm']):+.1f}..{float(band['max_mm']):+.1f}mm."
    )
    return bool(within), line


def _build_duration_schedule(
    *,
    trials: int | None,
    min_duration_ms: int,
    max_duration_ms: int,
    rng: random.Random,
    section_step_ms: int = DURATION_SECTION_STEP_MS_DEFAULT,
    section_span_ms: int = DURATION_SECTION_SPAN_MS_DEFAULT,
    samples_per_section: int = DURATION_SAMPLES_PER_SECTION_DEFAULT,
) -> list[int]:
    low = max(1, int(min_duration_ms))
    high = max(low, int(max_duration_ms))

    # Default: sectioned random sampling across the range.
    band_step = max(1, int(section_step_ms))
    band_span = max(0, int(section_span_ms))
    per_section = max(1, int(samples_per_section))

    def _sample_band(start_ms: int, end_ms: int, count: int) -> list[int]:
        band_low = int(start_ms)
        band_high = max(band_low, int(end_ms))
        band_size = int(band_high - band_low + 1)
        if int(count) <= int(band_size):
            return list(rng.sample(range(band_low, band_high + 1), int(count)))
        return [int(rng.randint(band_low, band_high)) for _ in range(int(count))]

    sections = []
    start_ms = int(low)
    while int(start_ms) < int(high):
        end_ms = min(int(high), int(start_ms) + int(band_span))
        sections.append((int(start_ms), int(end_ms)))
        start_ms += int(band_step)
    if not sections:
        sections = [(int(low), int(high))]

    schedule: list[int] = []
    for start_ms, end_ms in sections:
        schedule.extend(_sample_band(start_ms, end_ms, int(per_section)))
    if trials is not None:
        return schedule[: max(1, int(trials))]
    return schedule


def _build_reset_duration_schedule(*, rng: random.Random) -> list[int]:
    return _build_duration_schedule(
        trials=None,
        min_duration_ms=int(RESET_DURATION_MIN_MS),
        max_duration_ms=int(RESET_DURATION_MAX_MS),
        rng=rng,
        section_step_ms=int(RESET_DURATION_SECTION_STEP_MS),
        section_span_ms=int(RESET_DURATION_SECTION_SPAN_MS),
        samples_per_section=int(RESET_DURATION_SAMPLES_PER_SECTION),
    )


def _next_cycled_duration_ms(schedule: deque[int]) -> int:
    duration_ms = max(1, int(_coerce_int(schedule[0], 1) or 1))
    schedule.rotate(-1)
    return int(duration_ms)


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


def _effective_preflight_speed_score(speed_score: int, preflight_result: dict | None) -> int:
    score = normalize_speed_score(speed_score)
    if not isinstance(preflight_result, dict):
        return int(score)
    detected_score = _coerce_int(preflight_result.get("score_used"), score)
    return int(normalize_speed_score(detected_score))


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
        f"[RECOVERY]   Act {int(idx)}/{int(total)}: {_mast_label_for_cmd(cmd)} "
        f"score={int(score)}% duration={int(duration_ms)}ms "
        f"(undo {_mast_label_for_cmd(undo_cmd)})"
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
            step="CALIBRATE_Y_RECOVER",
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
        f"[CALIBRATE_Y] {display_label}: no visible brick {stage_label}. Attempting recovery."
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
        f"[CALIBRATE_Y] {display_label}: recovered visibility {stage_label} via {recovery_mode}."
    )
    return pose, {
        "mode": recovery_mode,
        "reobserved": True,
        "inverse_acts": _coerce_int((recovery_meta or {}).get("inverse_acts"), 0),
    }


def _ensure_pose_within_trial_band(
    *,
    initial_pose: dict,
    trial_cmd: str,
    trial_idx: int,
    trials_planned: int,
    vision,
    world,
    robot,
    recent_acts,
    setup_score: int,
    observe_samples: int,
    observe_timeout_s: float,
    post_act_settle_s: float,
    reset_duration_schedule: deque[int] | None = None,
    plotter=None,
    reset_efforts: list[ResetEffort] | None = None,
) -> tuple[dict | None, dict]:
    if not isinstance(initial_pose, dict):
        return None, {"mode": "trial_band_unavailable", "setup_acts": 0}

    current_y_mm = float(initial_pose.get("offset_y") or 0.0)
    within, _line = _trial_band_status_line(trial_cmd, current_y_mm)
    if within:
        return initial_pose, {"mode": "trial_band_within", "setup_acts": 0}

    correction_cmd = _trial_band_correction_cmd(trial_cmd, current_y_mm)
    if reset_duration_schedule is None or not reset_duration_schedule:
        reset_duration_schedule = deque(_build_reset_duration_schedule(rng=random.Random(0)))
    duration_ms = _next_cycled_duration_ms(reset_duration_schedule)
    act_start_ts = time.time()
    action_meta = _send_fixed_score_command(
        robot=robot,
        world=world,
        step="CALIBRATE_Y_RESET",
        cmd=str(correction_cmd),
        score=int(setup_score),
        duration_override_ms=int(duration_ms),
    )
    if not isinstance(action_meta, dict):
        return None, {"mode": "trial_band_send_failed", "setup_acts": 1}
    duration_used_ms = _coerce_int(action_meta.get("duration_ms"), duration_ms)
    recent_acts.append(
        {
            "cmd": str(correction_cmd),
            "duration_ms": int(duration_used_ms or 0),
            "score_requested": int(setup_score),
            "timestamp": time.time(),
        }
    )
    pose, observe_meta = _observe_pose_with_reobserve(
        vision=vision,
        world=world,
        samples=observe_samples,
        timeout_s=observe_timeout_s,
        min_sample_time=act_start_ts + (float(duration_used_ms or 0) / 1000.0) + float(post_act_settle_s),
    )
    if pose is None:
        return None, {"mode": "trial_band_post_unavailable", "setup_acts": 1}

    movement = _movement_metrics(
        str(correction_cmd),
        float(initial_pose.get("offset_y") or 0.0),
        float(pose.get("offset_y") or 0.0),
    )
    if plotter is not None:
        plotter.add_point(
            duration_ms=int(duration_used_ms or 0),
            distance_mm=float(movement["cmd_delta_mm"]),
            trial=int(trial_idx),
            cmd=str(correction_cmd),
            kind="reset",
            pre_brick_distance_mm=_coerce_finite_float(initial_pose.get("dist")),
            post_brick_distance_mm=_coerce_finite_float(pose.get("dist")),
        )
    if isinstance(reset_efforts, list):
        reset_efforts.append(
            ResetEffort(
                trial=int(trial_idx),
                reset_act=1,
                cmd=str(correction_cmd),
                score_requested=int(setup_score),
                cmd_sent=str(action_meta.get("cmd_sent") or correction_cmd),
                pwm=_coerce_int(action_meta.get("pwm")),
                power=_coerce_float(action_meta.get("power")),
                duration_ms=int(duration_used_ms or 0),
                pre_y_mm=float(initial_pose.get("offset_y") or 0.0),
                post_y_mm=float(pose.get("offset_y") or 0.0),
                raw_delta_mm=float(movement["raw_delta_mm"]),
                signed_cmd_delta_mm=float(movement["signed_cmd_delta_mm"]),
                cmd_delta_mm=float(movement["cmd_delta_mm"]),
                wrong_way=bool(movement["wrong_way"]),
                pre_brick_dist_mm=float(initial_pose.get("dist") or 0.0),
                post_brick_dist_mm=float(pose.get("dist") or 0.0),
                pre_confidence=float(initial_pose.get("confidence") or 0.0),
                post_confidence=float(pose.get("confidence") or 0.0),
                pre_pose_source=str(initial_pose.get("pose_source") or "unknown"),
                post_pose_source=str(pose.get("pose_source") or "unknown"),
                post_observation_mode=str((observe_meta or {}).get("mode") or "unknown"),
            )
        )
    return pose, {"mode": "trial_band_positioned", "setup_acts": 1}


def _diagnose_wrong_way_event(trial_result: TrialResult) -> None:
    """Log a loud alert for wrong_way without verbose diagnostics."""
    log_line("")
    log_line("=" * 80)
    log_line("⚠️  WRONG_WAY EVENT DETECTED")
    log_line("⚠️  ALERT ONLY: detailed diagnostic dump intentionally suppressed")
    log_line("=" * 80)
    log_line("")


def _diagnose_vision_loss_event(
    trial_label: str,
    cmd: str,
    pre_y_mm: float,
    center_target_y_mm: float,
    setup_score: int,
    duration_used_ms: int,
    cmd_sent: str | None = None,
    camera_direction_check: dict | None = None,
) -> None:
    """Log comprehensive diagnostics for a vision loss event after movement."""
    target_err_mm = float(pre_y_mm) - float(center_target_y_mm)
    
    log_line(
        f"[CALIBRATE_Y_VISION_LOSS] {trial_label}: "
        f"We started from y_axis={pre_y_mm:+.2f}mm "
        f"(target {float(center_target_y_mm):+.2f}mm; error {float(target_err_mm):+.2f}mm) "
        f"so we did cmd={cmd.upper()} score={setup_score}% duration={duration_used_ms}ms "
        f"and then lost vision."
    )
    if cmd_sent is not None and cmd_sent.lower() != cmd.lower():
        log_line(
            f"[CALIBRATE_Y_VISION_LOSS] ⚠️  COMMAND INVERSION SUSPECTED: "
            f"Logical cmd={cmd.upper()} but wire cmd={cmd_sent.upper()}"
        )
        _log_command_inversion_detail(
            prefix="[CALIBRATE_Y_VISION_LOSS] Direction detail:",
            trial_label=trial_label,
            logical_cmd=str(cmd),
            wire_cmd=str(cmd_sent),
            raw_delta_mm=None,
        )
    direction_hint = _camera_direction_check_hint(camera_direction_check, cmd)
    if direction_hint:
        log_line(f"[CALIBRATE_Y_VISION_LOSS] Direction check: {direction_hint}")


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
    center_target_y_mm: float,
    observe_samples: int,
    observe_timeout_s: float,
    post_act_settle_s: float,
    camera_direction_check: dict | None = None,
    plotter=None,
    initial_pre_pose: dict | None = None,
    initial_pre_obs_meta: dict | None = None,
    y_duration_cal: dict | None = None,
    trial_speed_profile: dict | None = None,
    stream_refresh_fn: Callable | None = None,
) -> tuple[TrialResult | None, str | None]:
    phase_key = str(phase or "trial")
    abort_prefix = ""

    def _on_vision_update():
        if callable(stream_refresh_fn):
            stream_refresh_fn()

    pre_pose = dict(initial_pre_pose) if isinstance(initial_pre_pose, dict) else None
    pre_obs_meta = dict(initial_pre_obs_meta) if isinstance(initial_pre_obs_meta, dict) else None
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
        )
        if pre_pose is None:
            log_line(f"[CALIBRATE_Y] {trial_label}: recovery failed before act. Aborting.")
            return None, f"{abort_prefix}pre_pose_unavailable_trial_{trial_idx}"
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
            f"[CALIBRATE_Y] {trial_label}: trial speed curve "
            f"dist={observed_dist_text} -> score={int(effective_score)}% "
            f"(base {int(setup_score)}%)."
        )

    act_plan = _planned_action_meta(cmd, effective_score, duration_ms)
    trial_plan_text = "this repeat trial will move" if str(phase_key) == "repeat" else "this planned trial will move"
    log_line(
        f"[CALIBRATE_Y] {trial_label}: {trial_plan_text} {_mast_label_for_cmd(cmd)} "
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
        log_line(f"[CALIBRATE_Y] {trial_label}: send failed. Aborting.")
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
    # Log the post-act pause with breakdown
    settle_ms = int(round(float(post_act_settle_s) * 1000.0))
    duration_wait_ms = int(duration_used_ms or 0)
    total_pause_ms = duration_wait_ms + settle_ms
    log_line(
        f"\033[90m[CALIBRATE_Y] {trial_label}: {int(total_pause_ms)}ms pause "
        f"(cmd={int(duration_wait_ms)}ms + settle={int(settle_ms)}ms)\033[0m"
    )
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
        # Log detailed diagnostics of what we were doing when vision was lost
        _diagnose_vision_loss_event(
            trial_label=trial_label,
            cmd=cmd,
            pre_y_mm=float(pre_pose["offset_y"]),
            center_target_y_mm=float(center_target_y_mm),
            setup_score=int(effective_score),
            duration_used_ms=int(duration_used_ms or 0),
            cmd_sent=str(action_meta.get("cmd_sent")),
            camera_direction_check=camera_direction_check,
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
            log_line(f"[CALIBRATE_Y] {trial_label}: recovery failed after act. Aborting.")
            return None, f"{abort_prefix}post_pose_unavailable_trial_{trial_idx}"
        recovered_visibility = True
        recovery_mode = str(post_obs_meta.get("mode") or "unknown")
        recovery_inverse_acts = _coerce_int(post_obs_meta.get("inverse_acts"), 0)

    pre_y_mm = float(pre_pose["offset_y"])
    post_y_mm = float(post_pose["offset_y"])
    movement = _movement_metrics(cmd, pre_y_mm, post_y_mm)
    raw_delta_mm = float(movement["raw_delta_mm"])
    signed_cmd_delta_mm = float(movement["signed_cmd_delta_mm"])
    cmd_delta_mm = float(movement["cmd_delta_mm"])
    wrong_way = bool(movement["wrong_way"])
    _record_camera_direction_check(
        camera_direction_check,
        trial_label=trial_label,
        cmd=str(cmd),
        cmd_sent=str(action_meta.get("cmd_sent") or cmd),
        raw_delta_mm=raw_delta_mm,
    )
    cmd_sent_effective = str(action_meta.get("cmd_sent") or cmd)
    raw_delta_below_threshold = abs(float(raw_delta_mm)) < float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM)
    should_log_inversion_detail = False
    if not raw_delta_below_threshold and _should_log_command_inversion_detail(
        logical_cmd=str(cmd),
        wire_cmd=str(cmd_sent_effective),
        raw_delta_mm=float(raw_delta_mm),
        threshold_mm=float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM),
    ):
        should_log_inversion_detail = True
    source_trial_value = _coerce_int(source_trial, trial_idx)

    row = TrialResult(
        trial=int(trial_idx),
        duration_ms=int(duration_used_ms or 0),
        cmd=str(cmd),
        score_requested=int(effective_score),
        cmd_sent=str(cmd_sent_effective),
        pwm=_coerce_int(action_meta.get("pwm")),
        power=_coerce_float(action_meta.get("power")),
        pre_y_mm=pre_y_mm,
        post_y_mm=post_y_mm,
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

    # Get prediction from existing curve
    predicted_distance_mm, curve_source = _predict_movement_from_curve(
        cmd=cmd,
        duration_ms=int(duration_used_ms or 0),
        y_calibration=y_duration_cal,
        observed_distance_mm=_coerce_finite_float(pre_pose.get("dist")),
    )
    
    # Calculate prediction comparison metrics
    prediction_comparison = _calculate_prediction_comparison(
        actual_distance_mm=cmd_delta_mm,
        predicted_distance_mm=predicted_distance_mm,
        curve_source=curve_source,
    )
    
    log_line(
        "[CALIBRATE_Y] "
        f"{_trial_result_label(trial_idx=trial_idx, trials_planned=trials_planned)}: "
        f"cmd={cmd.upper()} score={_score_with_motion_details_text(cmd, int(effective_score), pwm=row.pwm, power=row.power, duration_ms=int(duration_used_ms or 0))} "
        f"start_y={pre_y_mm:+.2f}mm end_y={post_y_mm:+.2f}mm "
        f"distance={cmd_delta_mm:.2f}mm signed={signed_cmd_delta_mm:+.2f}mm "
        f"wrong_way={bool(wrong_way)} raw_delta={raw_delta_mm:+.2f}mm "
        f"predicted={prediction_comparison['predicted_distance_mm']:.2f}mm "
        f"curve_source={prediction_comparison['curve_source']} "
        f"{_format_prediction_comparison_fields(prediction_comparison)}"
        if prediction_comparison['predicted_distance_mm'] is not None else 
        f"predicted=None curve_source={prediction_comparison['curve_source']}"
    )

    if raw_delta_below_threshold:
        discarded_text = _colorize('DISCARDED', '\033[91m')
        log_line(
            f"[CALIBRATE_Y] {trial_label}: {discarded_text} — "
            f"raw_delta={float(raw_delta_mm):+.2f}mm is below {float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM):.2f}mm threshold."
        )
    elif should_log_inversion_detail:
        _log_command_inversion_detail(
            prefix="[CALIBRATE_Y] ⚠️  Command inversion detail:",
            trial_label=trial_label,
            logical_cmd=str(cmd),
            wire_cmd=str(cmd_sent_effective),
            raw_delta_mm=float(raw_delta_mm),
            threshold_mm=float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM),
        )

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
        )
    if callable(stream_refresh_fn):
        stream_refresh_fn()
    return row, None


class LivePlot:
    def __init__(self, *, show_plot: bool, plot_path: Path | None):
        self._plot = CalibrationLivePlot(
            show_plot=show_plot,
            plot_path=plot_path,
            cmds=("u", "d"),
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
    camera_direction_check: dict | None = None,
    status: str = "completed",
    abort_reason: str | None = None,
) -> dict:
    return build_shared_payload(
        source="calibrate_y",
        config=config,
        durations_ms=durations_ms,
        trials=trials,
        reset_efforts=reset_efforts,
        status=status,
        abort_reason=abort_reason,
        extra_fields={"camera_direction_check": json.loads(json.dumps(camera_direction_check or {}))},
    )


def _exit_as_script(exit_code: int) -> None:
    if sys.gettrace() is not None:
        return
    raise SystemExit(int(exit_code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal y-axis duration probe with live scatter updates.")
    parser.add_argument(
        "--trial-mode",
        choices=["observation", "target"],
        default="observation",
        help="observation=random duration probes, target=one-shot attempts toward requested y targets.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Optional trial cap; default runs the full sectioned duration schedule.",
    )
    parser.add_argument(
        "--speed-score", type=int, default=1, help="Fixed y-axis speed score (default: 1)."
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default="d",
        help="Logical mast command: u, d, auto, or center (default: d for downward calibration).",
    )
    parser.add_argument(
        "--center-y-mm",
        type=float,
        default=Y_AXIS_SWEET_SPOT_MM_DEFAULT,
        help=f"Y-axis center target used for status logging and optional auto command selection (default: {Y_AXIS_SWEET_SPOT_MM_DEFAULT}).",
    )
    parser.add_argument("--auto-deadband-mm", type=float, default=0.5, help="If auto and current y is within this band, use the fallback cmd (default: 0.5).")
    parser.add_argument("--center-fallback-cmd", type=str, default="d", help="Fallback cmd when auto is inside deadband: u or d (default: d).")
    parser.add_argument(
        "--vision",
        choices=["leia", "yolo", "aruco"],
        default="yolo",
        help="Which vision backend to use: yolo cyan bricks (default), aruco markers, or leia edges.",
    )

    parser.add_argument("--min-duration-ms", type=int, default=200, help="Minimum random duration in ms (default: 200).")
    parser.add_argument("--max-duration-ms", type=int, default=1500, help="Maximum random duration in ms (default: 1500).")
    parser.add_argument("--observe-samples", type=int, default=OBSERVE_SAMPLES_DEFAULT, help="Observation samples per pose; use 3 for 3-frame confidence (default: 3).")
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S, help=f"Observation timeout in seconds (default: {OBSERVE_TIMEOUT_S}).")
    parser.add_argument("--post-act-settle-s", type=float, default=POST_ACT_SETTLE_S, help=f"Extra wait after the act before re-observing (default: {POST_ACT_SETTLE_S}).")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for repeatable durations.")
    parser.add_argument("--show-plot", action="store_true", help="Open an interactive Matplotlib window and update it after each trial.")
    parser.add_argument("--plot-path", type=str, default=PLOT_FILE_DEFAULT, help="Optional PNG file to rewrite after each trial.")
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_DEFAULT, help="JSON output path (default: run-specific file in ./runs).")
    parser.add_argument(
        "--target-y-mm",
        type=str,
        default="-9,-6,-3,0,3,6,9",
        help="Comma-separated target y-axis values (mm) used by --trial-mode target.",
    )
    parser.add_argument(
        "--target-repeats",
        type=int,
        default=1,
        help="How many one-shot attempts per target in --trial-mode target (default: 1).",
    )
    parser.add_argument(
        "--livestream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable livestream with mast/XYZ side panels for y-axis trial progress (default: enabled).",
    )
    parser.add_argument("--stream-host", type=str, default=STREAM_HOST)
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT)
    parser.add_argument("--stream-fps", type=int, default=STREAM_FPS)
    parser.add_argument("--stream-jpeg-quality", type=int, default=STREAM_JPEG_QUALITY)
    parser.add_argument("--stream-img-width", type=int, default=STREAM_IMG_WIDTH)
    parser.add_argument(
        "--preflight-check",
        action="store_true",
        help="Run a 1% movement preflight check before trials (disabled by default).",
    )
    parser.add_argument("--reference-distance-mm", type=float, default=None, help="Assumed brick distance (mm) for this calibration set")
    args = parser.parse_args()
    run_dir = _run_dir_for_vision(args.vision)
    _ensure_run_dir(run_dir)
    # Use a stable live JSON file in the vision-specific runs folder.
    if args.results_file is None:
        args.results_file = str(Path(run_dir) / "calibrate_y_live.json")
    try:
        cmd_mode = _normalize_cmd(args.cmd, allow_auto=True)
        center_fallback_cmd = _normalize_cmd(args.center_fallback_cmd, allow_auto=False)
    except ValueError as exc:
        log_line(f"[CALIBRATE_Y] {exc}")
        return 2

    trials_requested = None if args.trials is None else max(1, int(args.trials))
    trial_mode = str(args.trial_mode)
    speed_score = normalize_speed_score(args.speed_score)
    requested_speed_score = int(speed_score)
    prompted_speed_score = False
    prompted_duration_bounds = False
    center_y_mm = float(args.center_y_mm)
    auto_deadband_mm = abs(float(args.auto_deadband_mm))
    min_duration_ms = max(1, int(args.min_duration_ms))
    max_duration_ms = max(min_duration_ms, int(args.max_duration_ms))
    duration_ceiling_ms = max(1, int(DURATION_CEILING_MS))
    if max_duration_ms > duration_ceiling_ms:
        log_line(
            f"[CALIBRATE_Y] Clamping requested max_duration_ms={int(max_duration_ms)}ms "
            f"to ceiling {int(duration_ceiling_ms)}ms."
        )
        max_duration_ms = int(duration_ceiling_ms)
    if min_duration_ms > duration_ceiling_ms:
        log_line(
            f"[CALIBRATE_Y] Clamping requested min_duration_ms={int(min_duration_ms)}ms "
            f"to ceiling {int(duration_ceiling_ms)}ms."
        )
        min_duration_ms = int(duration_ceiling_ms)
    max_duration_ms = max(int(min_duration_ms), int(max_duration_ms))
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    post_act_settle_s = max(0.0, float(args.post_act_settle_s))
    target_repeats = max(1, int(args.target_repeats))
    target_y_values = _parse_float_list(args.target_y_mm)
    rng = random.Random(args.seed)
    if args.reference_distance_mm is not None:
        global REFERENCE_BRICK_DISTANCE_MM
        REFERENCE_BRICK_DISTANCE_MM = float(args.reference_distance_mm)
    trial_speed_profile = shared_load_calibration_trial_speed_profile("y_axis")
    durations_ms = _build_duration_schedule(
        trials=trials_requested,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        rng=rng,
    )
    if trial_mode == "target":
        if not target_y_values:
            log_line("[CALIBRATE_Y] --trial-mode target requires at least one --target-y-mm value.")
            return 2
        planned_targets = [float(v) for _ in range(int(target_repeats)) for v in target_y_values]
        trials_planned = len(planned_targets)
        durations_ms = [int(min_duration_ms)] * int(trials_planned)
    else:
        planned_targets = []
        trials_planned = len(durations_ms)
    results_path = Path(args.results_file)
    plot_path = Path(args.plot_path) if args.plot_path else None
    duration_section_count = (
        max(
            1,
            ((int(max_duration_ms) - int(min_duration_ms) - 1) // int(DURATION_SECTION_STEP_MS_DEFAULT)) + 1,
        )
    )

    config = {
        "trial_mode": str(trial_mode),
        "trials": int(trials_planned),
        "requested_trials": None if trials_requested is None else int(trials_requested),
        "duration_ceiling_ms": int(duration_ceiling_ms),
        "speed_score": int(speed_score),
        "requested_speed_score": int(requested_speed_score),
        "speed_score_source": "arg",
        "cmd": str(cmd_mode),
        "center_y_mm": float(center_y_mm),
        "auto_deadband_mm": float(auto_deadband_mm),
        "center_fallback_cmd": str(center_fallback_cmd),
        "min_duration_ms": int(min_duration_ms),
        "max_duration_ms": int(max_duration_ms),
        "duration_section_step_ms": int(DURATION_SECTION_STEP_MS_DEFAULT),
        "duration_section_span_ms": int(DURATION_SECTION_SPAN_MS_DEFAULT),
        "duration_samples_per_section": int(DURATION_SAMPLES_PER_SECTION_DEFAULT),
        "duration_section_count": int(duration_section_count),
        "y_axis_positive_cmd": str(_y_cmd_for_positive_motion()),
        "y_axis_negative_cmd": str(_y_cmd_for_negative_motion()),
        "observe_samples": int(observe_samples),
        "observe_timeout_s": float(observe_timeout_s),
        "post_act_settle_s": float(post_act_settle_s),
        "camera_direction_verify_threshold_mm": float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM),
        "camera_direction_expected_by_cmd": {
            "u": "down",
            "d": "up",
        },
        "trial_cmd_schedule": "alternating",
        "trial_cmd_start": str(_normalize_cmd(TRIAL_ALTERNATING_START_CMD, allow_auto=False)).upper(),
        "y_axis_center_target_mm": float(center_y_mm),
        "seed": args.seed,
        "plot_path": str(plot_path) if plot_path is not None else None,
        "brick_distance_source": str(BRICK_DISTANCE_SOURCE),
        "brick_distance_definition": str(BRICK_DISTANCE_DEFINITION),
        "target_y_mm": [float(v) for v in planned_targets] if trial_mode == "target" else [],
        "target_repeats": int(target_repeats),
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
    stream_server = None
    stream_state = None
    stream_url = format_stream_url(str(args.stream_host), int(args.stream_port))
    recent_acts = deque(maxlen=32)
    trial_rows: list[TrialResult] = []
    reset_rows: list[ResetEffort] = []
    camera_direction_check = _new_camera_direction_check_state()
    status = "completed"
    abort_reason = None
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
            log_line(f"[CALIBRATE_Y] Livestream URL: {_orange_text(stream_url)}")

            def _live_refresh(extra_lines=None):
                nonlocal stream_extra_lines
                if extra_lines is not None:
                    stream_extra_lines = [str(item) for item in list(extra_lines)]
                lines = [
                    f"Y calibration mode: {trial_mode}",
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
                log_line("[CALIBRATE_Y] Reusing existing manual-training livestream.")
            else:
                try:
                    stream_server, stream_url = start_stream_server(
                        stream_state,
                        title="Y-Axis Speed Curve Calibration",
                        header="",
                        footer="<div class='footer-sections'><div class='footer-section'><div class='footer-title'>Y Calibration</div><div>Mast/XYZ side panel + live trial telemetry.</div></div></div>",
                        host=str(args.stream_host),
                        port=int(args.stream_port),
                        fps=max(1, int(args.stream_fps)),
                        jpeg_quality=max(1, min(100, int(args.stream_jpeg_quality))),
                        img_width=max(320, int(args.stream_img_width)),
                        vision_mode_options=[("aruco", "AruCo Markers"), ("cyan", "Cyan Bricks")],
                        xyz_workspace_getter=lambda: getattr(world, "_xyz_workspace", None),
                    )
                    log_line(f"[CALIBRATE_Y] Livestream started: {_orange_text(stream_url)}")
                except Exception as exc:
                    log_line(f"[CALIBRATE_Y] Livestream startup failed at {stream_url}: {exc}")
                    stream_server = None
        else:
            def _live_refresh(extra_lines=None):
                return

        # Load existing calibration curve for prediction comparison
        y_duration_cal = _load_y_duration_calibration()
        initial_trial_pre_pose = None
        initial_trial_pre_obs_meta = None
        if trials_planned > 0:
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
                )
            if initial_trial_pre_pose is None:
                status = "aborted"
                abort_reason = "pre_pose_unavailable_before_trials"
                log_line("[CALIBRATE_Y] TRIALS SETUP: unable to observe current distance. Aborting.")
            else:
                setup_distance_mm = _coerce_finite_float(initial_trial_pre_pose.get("dist"))
                setup_curve_name = _closest_y_duration_curve_name(
                    y_calibration=y_duration_cal,
                    observed_distance_mm=setup_distance_mm,
                )
                log_line(
                    _trials_setup_log_line(
                        observed_distance_mm=setup_distance_mm,
                        closest_curve_name=setup_curve_name,
                    )
                )
                prompted_settings = shared_prompt_calibration_run_settings(
                    prefix="CALIBRATE_Y",
                    observed_distance_mm=setup_distance_mm,
                    default_speed_score=int(requested_speed_score),
                    default_min_duration_ms=int(min_duration_ms),
                    default_max_duration_ms=int(max_duration_ms),
                    duration_ceiling_ms=int(duration_ceiling_ms),
                    log=log_line,
                )
                speed_score = normalize_speed_score(int(prompted_settings["speed_score"]))
                requested_speed_score = int(speed_score)
                min_duration_ms = max(1, int(prompted_settings["min_duration_ms"]))
                max_duration_ms = max(int(min_duration_ms), int(prompted_settings["max_duration_ms"]))
                prompted_speed_score = bool(prompted_settings.get("prompted_speed_score"))
                prompted_duration_bounds = bool(prompted_settings.get("prompted_duration_bounds"))
                durations_ms = _build_duration_schedule(
                    trials=trials_requested,
                    min_duration_ms=min_duration_ms,
                    max_duration_ms=max_duration_ms,
                    rng=rng,
                )
                if trial_mode == "target":
                    planned_targets = [float(v) for _ in range(int(target_repeats)) for v in target_y_values]
                    trials_planned = len(planned_targets)
                    durations_ms = [int(min_duration_ms)] * int(trials_planned)
                else:
                    planned_targets = []
                    trials_planned = len(durations_ms)
                duration_section_count = (
                    max(
                        1,
                        ((int(max_duration_ms) - int(min_duration_ms) - 1) // int(DURATION_SECTION_STEP_MS_DEFAULT)) + 1,
                    )
                )
                stream_score_line = (
                    f"Score: distance curve (base {int(speed_score)}%)"
                    if isinstance(trial_speed_profile, dict)
                    else f"Score: {int(speed_score)}%"
                )
                config = {
                    "trial_mode": str(trial_mode),
                    "trials": int(trials_planned),
                    "requested_trials": None if trials_requested is None else int(trials_requested),
                    "duration_ceiling_ms": int(duration_ceiling_ms),
                    "speed_score": int(speed_score),
                    "requested_speed_score": int(requested_speed_score),
                    "speed_score_source": "prompt" if bool(prompted_speed_score) else "arg",
                    "prompted_speed_score": bool(prompted_speed_score),
                    "cmd": str(cmd_mode),
                    "center_y_mm": float(center_y_mm),
                    "auto_deadband_mm": float(auto_deadband_mm),
                    "center_fallback_cmd": str(center_fallback_cmd),
                    "min_duration_ms": int(min_duration_ms),
                    "max_duration_ms": int(max_duration_ms),
                    "prompted_duration_bounds": bool(prompted_duration_bounds),
                    "duration_section_step_ms": int(DURATION_SECTION_STEP_MS_DEFAULT),
                    "duration_section_span_ms": int(DURATION_SECTION_SPAN_MS_DEFAULT),
                    "duration_samples_per_section": int(DURATION_SAMPLES_PER_SECTION_DEFAULT),
                    "duration_section_count": int(duration_section_count),
                    "y_axis_positive_cmd": str(_y_cmd_for_positive_motion()),
                    "y_axis_negative_cmd": str(_y_cmd_for_negative_motion()),
                    "observe_samples": int(observe_samples),
                    "observe_timeout_s": float(observe_timeout_s),
                    "post_act_settle_s": float(post_act_settle_s),
                    "camera_direction_verify_threshold_mm": float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM),
                    "camera_direction_expected_by_cmd": {
                        "u": "down",
                        "d": "up",
                    },
                    "trial_cmd_schedule": "alternating",
                    "trial_cmd_start": str(_normalize_cmd(TRIAL_ALTERNATING_START_CMD, allow_auto=False)).upper(),
                    "y_axis_center_target_mm": float(center_y_mm),
                    "seed": args.seed,
                    "plot_path": str(plot_path) if plot_path is not None else None,
                    "brick_distance_source": str(BRICK_DISTANCE_SOURCE),
                    "brick_distance_definition": str(BRICK_DISTANCE_DEFINITION),
                    "target_y_mm": [float(v) for v in planned_targets] if trial_mode == "target" else [],
                    "target_repeats": int(target_repeats),
                    "trial_speed_score_source": "distance_curve" if isinstance(trial_speed_profile, dict) else "arg",
                }
                if REFERENCE_BRICK_DISTANCE_MM is not None:
                    config["reference_brick_distance_mm"] = float(REFERENCE_BRICK_DISTANCE_MM)
                if isinstance(trial_speed_profile, dict):
                    config["trial_speed_score_profile"] = json.loads(json.dumps(trial_speed_profile))
                log_line("[CALIBRATE_Y] Starting y-axis duration probe.")
                log_line(
                    f"[CALIBRATE_Y] mode={trial_mode} trials={trials_planned} score={int(speed_score)}% durations_ms={durations_ms} "
                    f"observe_samples={observe_samples}"
                )
                log_line(
                    f"[CALIBRATE_Y] duration_sections={int(duration_section_count)} "
                    f"window={int(DURATION_SECTION_SPAN_MS_DEFAULT)}ms every {int(DURATION_SECTION_STEP_MS_DEFAULT)}ms "
                    f"samples_per_section={int(DURATION_SAMPLES_PER_SECTION_DEFAULT)}"
                )
                log_line(
                    f"[CALIBRATE_Y] y-axis motion sign: {_mast_label_for_cmd(_y_cmd_for_positive_motion())} increases y_axis, "
                    f"{_mast_label_for_cmd(_y_cmd_for_negative_motion())} decreases y_axis."
                )
                log_line(f"[CALIBRATE_Y] center target: y_axis={float(center_y_mm):+.2f}mm.")
                log_line(
                    f"[CALIBRATE_Y] camera-direction check: mast_up should move brick DOWN on camera, "
                    f"mast_down should move brick UP on camera (threshold {float(CAMERA_DIRECTION_VERIFY_MIN_DELTA_MM):.2f}mm)."
                )
                log_line(
                    f"[CALIBRATE_Y] trial command schedule: alternating "
                    f"{_mast_label_for_cmd(TRIAL_ALTERNATING_START_CMD)}, "
                    f"{_mast_label_for_cmd(_inverse_cmd(TRIAL_ALTERNATING_START_CMD) or 'u')}."
                )
                if isinstance(trial_speed_profile, dict):
                    curve_points = list(trial_speed_profile.get("curve_points") or [])
                    if len(curve_points) >= 2:
                        first_point = curve_points[0]
                        last_point = curve_points[-1]
                        log_line(
                            f"[CALIBRATE_Y] trial speed curve: "
                            f"<= {float(first_point.get('distance_mm')):.0f}mm -> {int(first_point.get('speed_score'))}% ; "
                            f">= {float(last_point.get('distance_mm')):.0f}mm -> {int(last_point.get('speed_score'))}% ; "
                            "linear between points."
                        )
                if cmd_mode == "auto":
                    log_line(
                        f"[CALIBRATE_Y] Center-aware y target is {float(center_y_mm):+.2f}mm "
                        f"with deadband +/-{float(auto_deadband_mm):.2f}mm."
                    )
                if bool(args.show_plot):
                    if _MATPLOTLIB_AVAILABLE:
                        log_line("[CALIBRATE_Y] Live plot enabled.")
                    else:
                        log_line("[CALIBRATE_Y] Matplotlib unavailable; continuing without live plot.")
                if plot_path is not None:
                    log_line(f"[CALIBRATE_Y] Plot PNG will update at {plot_path}")
                _live_refresh(
                    [
                        "TRIALS SETUP",
                        f"Observed dist: {float(setup_distance_mm):.2f}mm"
                        if setup_distance_mm is not None
                        else "Observed dist: unknown",
                        f"Closest curve: {setup_curve_name}",
                        str(stream_score_line),
                        f"Durations: {int(min_duration_ms)}..{int(max_duration_ms)}ms",
                    ]
                )

        if status == "aborted":
            pass
        else:
            for trial_idx, duration_ms in enumerate(durations_ms, start=1):
                trial_label = _trial_label_text(trial_idx, trials_planned)
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

                if trial_mode == "target":
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
                            stage_label="before target trial",
                            trial_label=trial_label,
                        )
                    if pre_pose is None:
                        status = "aborted"
                        abort_reason = f"pre_pose_unavailable_trial_{trial_idx}"
                        log_line(f"[CALIBRATE_Y] {trial_label}: pre-pose unavailable for target trial.")
                        break
                    curr_y = float(pre_pose.get("offset_y") or 0.0)
                    target_y = float(planned_targets[trial_idx - 1])
                    y_delta = float(target_y - curr_y)
                    cmd = _y_cmd_for_positive_motion() if y_delta >= 0.0 else _y_cmd_for_negative_motion()
                    duration_ms = _predict_duration_for_target_delta_mm(
                        cmd=cmd,
                        abs_delta_mm=abs(float(y_delta)),
                        duration_min_ms=int(min_duration_ms),
                        duration_max_ms=int(max_duration_ms),
                        y_calibration=y_duration_cal,
                        observed_distance_mm=_coerce_finite_float(pre_pose.get("dist")),
                    )
                    log_line(
                        f"[CALIBRATE_Y] {trial_label}: target_y={target_y:+.2f}mm current_y={curr_y:+.2f}mm "
                        f"delta={y_delta:+.2f}mm cmd={str(cmd).upper()} duration={int(duration_ms)}ms"
                    )
                    _live_refresh(
                        [
                            f"Trial {int(trial_idx)}/{int(trials_planned)}",
                            f"Target y: {float(target_y):+.2f}mm",
                            f"Current y: {float(curr_y):+.2f}mm",
                            f"Planned: {str(cmd).upper()} {int(duration_ms)}ms",
                        ]
                    )
                else:
                    scheduled_trial_cmd = _scheduled_trial_cmd_for_trial(trial_idx)
                    if scheduled_trial_cmd is not None:
                        cmd = str(scheduled_trial_cmd)
                        log_line(
                            f"[CALIBRATE_Y] {trial_label}: scheduled alternating cmd={str(cmd).upper()} "
                            f"({_mast_label_for_cmd(cmd)}) to stay near y_axis={float(center_y_mm):+.2f}mm."
                        )
                    elif cmd_mode == "auto":
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
                                stage_label="before command selection",
                                trial_label=trial_label,
                            )
                        if pre_pose is None:
                            status = "aborted"
                            abort_reason = f"pre_pose_unavailable_trial_{trial_idx}"
                            log_line(f"[CALIBRATE_Y] {trial_label}: recovery failed before command selection. Aborting.")
                            break
                        curr_y = float(pre_pose["offset_y"])
                        cmd = _auto_cmd_for_y(
                            curr_y,
                            center_y_mm=float(center_y_mm),
                            deadband_mm=float(auto_deadband_mm),
                            fallback_cmd=center_fallback_cmd,
                        )
                        log_line(
                            f"[CALIBRATE_Y] {trial_label}: auto selection "
                            f"current_y={curr_y:+.2f}mm target_y={float(center_y_mm):+.2f}mm "
                            f"deadband=+/-{float(auto_deadband_mm):.2f}mm -> cmd={cmd.upper()}"
                        )
                    else:
                        cmd = str(cmd_mode)

                row, trial_abort_reason = _run_trial_action(
                    trial_idx=trial_idx,
                    trials_planned=trials_planned,
                    trial_label=trial_label,
                    cmd=str(cmd),
                    duration_ms=int(duration_ms),
                    phase="trial",
                    source_trial=trial_idx,
                    action_step="CALIBRATE_Y",
                    plot_kind="trial",
                    vision=vision,
                    world=world,
                    robot=robot,
                    recent_acts=recent_acts,
                    setup_score=int(speed_score),
                    center_target_y_mm=float(center_y_mm),
                    observe_samples=observe_samples,
                    observe_timeout_s=observe_timeout_s,
                    post_act_settle_s=post_act_settle_s,
                    camera_direction_check=camera_direction_check,
                    plotter=plotter,
                    initial_pre_pose=pre_pose,
                    initial_pre_obs_meta=pre_obs_meta,
                    y_duration_cal=y_duration_cal,
                    trial_speed_profile=trial_speed_profile,
                    stream_refresh_fn=_live_refresh,
                )
                if row is None:
                    status = "aborted"
                    abort_reason = str(trial_abort_reason or f"trial_failed_{trial_idx}")
                    break
                if bool(row.wrong_way):
                    _diagnose_wrong_way_event(row)
                    log_line(
                        f"[CALIBRATE_Y] ⚠️  Trial {trial_idx}: wrong_way detected. Plotting it anyway."
                    )
                trial_rows.append(row)
                if trial_mode == "target":
                    target_y = float(planned_targets[trial_idx - 1])
                    final_err = float(target_y - float(row.post_y_mm))
                    log_line(
                        f"[CALIBRATE_Y] {trial_label}: target={target_y:+.2f}mm post_y={float(row.post_y_mm):+.2f}mm "
                        f"final_error={float(final_err):+.2f}mm"
                    )
                    _live_refresh(
                        [
                            f"Trial {int(trial_idx)}/{int(trials_planned)}",
                            f"Target y: {float(target_y):+.2f}mm",
                            f"Post y: {float(row.post_y_mm):+.2f}mm",
                            f"Final err: {float(final_err):+.2f}mm",
                        ]
                    )
                else:
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
                        reset_efforts=reset_rows,
                        camera_direction_check=camera_direction_check,
                        status=status,
                        abort_reason=abort_reason,
                    ),
                )

    except KeyboardInterrupt:
        status = "interrupted"
        abort_reason = "keyboard_interrupt"
        log_line("[CALIBRATE_Y] Interrupted by user.")
    finally:
        _write_results(
            results_path,
            _build_payload(
                config=config,
                durations_ms=durations_ms,
                trials=trial_rows,
                reset_efforts=reset_rows,
                camera_direction_check=camera_direction_check,
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
        if stream_server is not None:
            try:
                close_fn = getattr(stream_server, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass

    direction_summary = _camera_direction_check_summary_line(camera_direction_check)
    if direction_summary:
        log_line(direction_summary)
    log_line(f"[CALIBRATE_Y] Wrote results to {results_path}")
    if plot_path is not None:
        log_line(f"[CALIBRATE_Y] Updated plot at {plot_path}")
    if status != "completed":
        detail = f" reason={abort_reason}" if abort_reason else ""
        log_line(f"[CALIBRATE_Y] Finished with status={status}{detail}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    _exit_as_script(main())
