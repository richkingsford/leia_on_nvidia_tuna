#!/usr/bin/env python3
"""Helper for manual dist/x/y telemetry calibration with shared-stream support.

The operator uses the normal hotkeys to position the robot, then captures
telemetry and types the expected real-world value for each point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import statistics
import sys
import termios
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from calibration.helper_calibrate import (
    get_shared_calibration_context,
    get_shared_stream_runtime,
    observe_pose_with_reobserve as shared_observe_pose_with_reobserve,
    prepare_shared_stream_state,
    read_pose as shared_read_pose,
)
import helper_xyz_coords
from helper_manual_config import load_manual_training_config
from helper_stream_server import format_stream_url
from helper_streaming import start_stream_server
from helper_robot_control import Robot
from helper_vision_leia import LeiaVision

try:
    from helper_brick_detector_yolo import BrickDetector as YoloBrickDetector
except ImportError:
    YoloBrickDetector = None

from telemetry_process import (
    _average_smoothed_frames as telemetry_average_smoothed_frames,
    _latest_unique_smoothed_frames as telemetry_latest_unique_smoothed_frames,
    lite_gate_unique_frames,
    send_robot_command_pwm,
    update_world_from_vision,
)
import telemetry_robot as telemetry_robot_module
from telemetry_robot import ROBOT_MODEL_FILE, StepState, WorldModel, speed_power_pwm_for_cmd, draw_telemetry_overlay

OBSERVE_SLEEP_S = 0.02
OBSERVE_TIMEOUT_S = 2.8
OBSERVE_SAMPLES_DEFAULT = 5
REOBSERVE_HOLD_S = 0.12
REOBSERVE_ROUNDS = 2
CAPTURE_MAX_SPREAD_MM_DEFAULT = 3.0
CAPTURE_TRIM_OUTLIERS_DEFAULT = 2
RUN_DIR_ARUCO = Path("Runs - aruco")
RUN_DIR_CYAN = Path("Runs - cyan")
TELEMETRY_CALIBRATION_KEY = "telemetry_observation_calibration"
STAGING_FILE_DEFAULT = Path("world_model_telemetry_staging.json")
_MANUAL_CONFIG = load_manual_training_config()
STREAM_IMG_WIDTH = int(_MANUAL_CONFIG.get("stream_img_width", 1600))
STREAM_VISION_MODE_OPTIONS = [
    ("aruco", "AruCo Markers"),
    ("cyan", "Crown Bricks"),
]
ANSI_ORANGE_BRIGHT = "\033[38;5;208m"
ANSI_RESET = "\033[0m"


def _orange_text(text: str) -> str:
    return f"{ANSI_ORANGE_BRIGHT}{str(text)}{ANSI_RESET}"


def _enable_single_char_noecho_mode(fd: int, saved_attrs: list) -> None:
    attrs = list(saved_attrs)
    attrs[3] = int(attrs[3]) & ~termios.ICANON & ~termios.ECHO
    cc = attrs[6]
    cc[termios.VMIN] = 1
    cc[termios.VTIME] = 0
    attrs[6] = cc
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


@dataclass(frozen=True)
class CalibrateOption:
    key: str
    label: str
    runner: Callable[[], int | None]


OPTIONS: tuple[CalibrateOption, ...] = (
    CalibrateOption("dist", "Distance telemetry calibration", lambda: run_telemetry_variable_calibration("dist")),
    CalibrateOption("x", "X-axis telemetry calibration", lambda: run_telemetry_variable_calibration("x")),
    CalibrateOption("y", "Y-axis telemetry calibration", lambda: run_telemetry_variable_calibration("y")),
)


@dataclass
class SampleRow:
    index: int
    expected_value: float
    observed_dist: float
    observed_x: float
    observed_y: float
    confidence: float
    samples_used: int | None
    pose_source: str | None
    observation_mode: str | None
    ts: float


def _run_dir_for_vision(vision_mode: str | None) -> Path:
    mode = str(vision_mode or "").strip().lower()
    if mode == "aruco":
        return Path(RUN_DIR_ARUCO)
    return Path(RUN_DIR_CYAN)


def _default_results_path(*, variable: str, vision: str, output_dir: str | None) -> Path:
    run_dir = Path(output_dir) if output_dir else _run_dir_for_vision(vision)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"calibrate_telemetry_{str(variable)}_live.json"


def _default_plot_path(*, variable: str, vision: str, output_dir: str | None) -> Path:
    run_dir = Path(output_dir) if output_dir else _run_dir_for_vision(vision)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"calibrate_telemetry_{str(variable)}_live.png"


def _variable_unit(variable: str) -> str:
    if str(variable) == "dist":
        return "mm from brick"
    if str(variable) == "x":
        return "mm x-offset"
    return "mm y-offset"


def _expected_value_prompt(variable: str, target_hint: float | None) -> str:
    variable_text = str(variable or "dist").strip().lower()
    if target_hint is None:
        return f"Expected {variable_text} mm [q=quit] > "
    return f"Expected {variable_text} mm [Enter={float(target_hint):.1f}, q=quit] > "


def _ready_for_capture_prompt(variable: str) -> str:
    variable_text = str(variable or "dist").strip().lower()
    return f"Move with hotkeys, Enter=type expected {variable_text}, Esc=quit > "


def _extract_observed_value(row: SampleRow, variable: str) -> float:
    if str(variable) == "dist":
        return float(row.observed_dist)
    if str(variable) == "x":
        return float(row.observed_x)
    return float(row.observed_y)


def _parse_targets_list(raw_text: str) -> list[float]:
    values: list[float] = []
    for token in str(raw_text or "").replace(";", ",").split(","):
        text = str(token or "").strip()
        if not text:
            continue
        number = float(text)
        if not math.isfinite(number):
            raise ValueError("Target values must be finite numbers.")
        values.append(float(number))
    if not values:
        raise ValueError("No target values were provided.")
    return values


def _planned_expected_targets(
    *,
    variable: str,
    points: int,
    targets_mm_raw: str | None,
    target_start_mm: float,
    target_step_mm: float,
) -> list[float] | None:
    if str(targets_mm_raw or "").strip():
        parsed = _parse_targets_list(str(targets_mm_raw))
        if len(parsed) < int(points):
            raise ValueError(
                f"--targets-mm provided {len(parsed)} value(s), but {int(points)} point(s) are required."
            )
        return [float(value) for value in parsed[: int(points)]]
    if str(variable) == "dist":
        start = float(target_start_mm)
        step = float(target_step_mm)
        return [float(start + (step * idx)) for idx in range(int(points))]
    return None


def _load_json_object(path: Path) -> dict:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_json_object(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_rows_from_curve_points(curve_points: list[dict] | None) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for idx, item in enumerate(list(curve_points or []), start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            SampleRow(
                index=int(item.get("index") or idx),
                expected_value=float(item.get("expected_value") or 0.0),
                observed_dist=float(item.get("observed_dist") or 0.0),
                observed_x=float(item.get("observed_x") or 0.0),
                observed_y=float(item.get("observed_y") or 0.0),
                confidence=float(item.get("confidence") or 0.0),
                samples_used=None,
                pose_source=None,
                observation_mode=None,
                ts=float(item.get("timestamp") or 0.0),
            )
        )
    return rows


def _payload_from_rows(
    *,
    rows: list[SampleRow],
    variable: str,
    config: dict,
    source: str = "calibrate_dist_x_y_telemetry",
    generated_at: float | None = None,
) -> dict:
    variable_errors = [float(_extract_observed_value(item, variable) - float(item.expected_value)) for item in rows]
    abs_errors = [abs(value) for value in variable_errors]
    fit = _fit_linear_observed_to_expected(rows, variable)
    residual_stats = _fit_residual_stats(rows, variable, fit)
    return {
        "schema_version": 1,
        "source": str(source),
        "generated_at": float(generated_at if generated_at is not None else time.time()),
        "config": dict(config or {}),
        "fit": fit,
        "summary": {
            "mean_error_mm": float(statistics.mean(variable_errors)) if variable_errors else None,
            "median_error_mm": float(statistics.median(variable_errors)) if variable_errors else None,
            "mae_mm": float(statistics.mean(abs_errors)) if abs_errors else None,
            "max_abs_error_mm": float(max(abs_errors)) if abs_errors else None,
            "residual_mean_error_mm": residual_stats.get("residual_mean_error_mm"),
            "residual_mae_mm": residual_stats.get("residual_mae_mm"),
            "residual_max_abs_error_mm": residual_stats.get("residual_max_abs_error_mm"),
            "mean_confidence": float(statistics.mean([float(item.confidence) for item in rows])) if rows else None,
        },
        "samples": [
            {
                "index": int(item.index),
                "expected_value": float(item.expected_value),
                "observed_dist": float(item.observed_dist),
                "observed_x": float(item.observed_x),
                "observed_y": float(item.observed_y),
                "observed_variable": float(_extract_observed_value(item, variable)),
                "error_mm": float(_extract_observed_value(item, variable) - float(item.expected_value)),
                "confidence": float(item.confidence),
                "samples_used": item.samples_used,
                "pose_source": item.pose_source,
                "observation_mode": item.observation_mode,
                "timestamp": float(item.ts),
            }
            for item in rows
        ],
    }


def _merge_incremental_run_payload(existing_record: dict | None, run_payload: dict, variable: str) -> dict:
    existing = dict(existing_record or {})
    new_samples = list(run_payload.get("samples") or [])
    existing_points = list(existing.get("curve_points") or [])
    if len(new_samples) != 1 or len(existing_points) < 2:
        return dict(run_payload)

    merged_rows = _sample_rows_from_curve_points(existing_points)
    incoming_rows = _sample_rows_from_curve_points(
        [
            {
                "index": int(item.get("index") or 0),
                "expected_value": float(item.get("expected_value") or 0.0),
                "observed_dist": float(item.get("observed_dist") or 0.0),
                "observed_x": float(item.get("observed_x") or 0.0),
                "observed_y": float(item.get("observed_y") or 0.0),
                "confidence": float(item.get("confidence") or 0.0),
                "timestamp": float(item.get("timestamp") or 0.0),
            }
            for item in new_samples
        ]
    )
    if not incoming_rows:
        return dict(run_payload)

    merged_rows.extend(incoming_rows)
    merged_rows.sort(key=lambda item: (float(item.expected_value), float(_extract_observed_value(item, variable)), float(item.ts)))
    merged_rows = [
        SampleRow(
            index=idx,
            expected_value=float(item.expected_value),
            observed_dist=float(item.observed_dist),
            observed_x=float(item.observed_x),
            observed_y=float(item.observed_y),
            confidence=float(item.confidence),
            samples_used=item.samples_used,
            pose_source=item.pose_source,
            observation_mode=item.observation_mode,
            ts=float(item.ts),
        )
        for idx, item in enumerate(merged_rows, start=1)
    ]

    merged_config = dict(existing.get("config") or {})
    merged_config.update(dict(run_payload.get("config") or {}))
    merged_config["captured_points"] = int(len(merged_rows))
    merged_config["incremental_capture_mode"] = "append_single_point"
    merged_payload = _payload_from_rows(
        rows=merged_rows,
        variable=str(variable),
        config=merged_config,
        source=str(run_payload.get("source") or "calibrate_dist_x_y_telemetry"),
        generated_at=float(run_payload.get("generated_at") or time.time()),
    )
    return {
        "schema_version": 1,
        "run_id": int(time.time()),
        "timestamp": float(time.time()),
        "variable": str(variable),
        "config": dict(merged_payload.get("config") or {}),
        "fit": dict(merged_payload.get("fit") or {}),
        "summary": dict(merged_payload.get("summary") or {}),
        "curve_points": [
            {
                "index": int(item.get("index") or 0),
                "expected_value": float(item.get("expected_value") or 0.0),
                "observed_variable": float(item.get("observed_variable") or 0.0),
                "observed_dist": float(item.get("observed_dist") or 0.0),
                "observed_x": float(item.get("observed_x") or 0.0),
                "observed_y": float(item.get("observed_y") or 0.0),
                "error_mm": float(item.get("error_mm") or 0.0),
                "confidence": float(item.get("confidence") or 0.0),
            }
            for item in list(merged_payload.get("samples") or [])
        ],
    }


def _upsert_staging_telemetry_calibration(
    *,
    staging_path: Path,
    variable: str,
    run_payload: dict,
) -> dict:
    staged = _load_json_object(staging_path)
    by_variable = staged.get("by_variable")
    if not isinstance(by_variable, dict):
        by_variable = {}

    history = staged.get("history")
    if not isinstance(history, list):
        history = []

    run_record = {
        "schema_version": 1,
        "run_id": int(time.time()),
        "timestamp": float(time.time()),
        "variable": str(variable),
        "config": dict(run_payload.get("config") or {}),
        "fit": dict(run_payload.get("fit") or {}),
        "summary": dict(run_payload.get("summary") or {}),
        # Each point retains full dist/x/y context regardless of target variable.
        "curve_points": [
            {
                "index": int(item.get("index") or 0),
                "expected_value": float(item.get("expected_value") or 0.0),
                "observed_variable": float(item.get("observed_variable") or 0.0),
                "observed_dist": float(item.get("observed_dist") or 0.0),
                "observed_x": float(item.get("observed_x") or 0.0),
                "observed_y": float(item.get("observed_y") or 0.0),
                "error_mm": float(item.get("error_mm") or 0.0),
                "confidence": float(item.get("confidence") or 0.0),
            }
            for item in list(run_payload.get("samples") or [])
        ],
    }

    active_record = _merge_incremental_run_payload(by_variable.get(str(variable)), run_payload, str(variable))
    by_variable[str(variable)] = dict(active_record)
    history.append(dict(run_record))
    if len(history) > 60:
        history = history[-60:]

    staged["schema_version"] = 1
    staged["updated_at"] = float(time.time())
    staged["latest_variable"] = str(variable)
    staged["by_variable"] = by_variable
    staged["history"] = history
    _save_json_object(staging_path, staged)
    return staged


def _upsert_world_model_telemetry_calibration(
    *,
    world_model_path: Path,
    staged_payload: dict,
) -> None:
    world_model = _load_json_object(world_model_path)
    cal = {
        "schema_version": 1,
        "updated_at": float(time.time()),
        "latest_variable": str(staged_payload.get("latest_variable") or ""),
        "by_variable": dict(staged_payload.get("by_variable") or {}),
        "history": list(staged_payload.get("history") or []),
        "commit_metadata": {
            "committed_at": float(time.time()),
            "source_staging_schema_version": int(staged_payload.get("schema_version") or 1),
        },
    }

    world_model[TELEMETRY_CALIBRATION_KEY] = cal
    _save_json_object(world_model_path, world_model)


def _fit_linear_observed_to_expected(rows: list[SampleRow], variable: str) -> dict:
    xs = [_extract_observed_value(item, variable) for item in rows]
    ys = [float(item.expected_value) for item in rows]
    n = len(xs)
    if n < 2:
        return {
            "equation": "expected = observed",
            "slope": 1.0,
            "intercept": 0.0,
            "r2": None,
        }
    x_mean = sum(xs) / float(n)
    y_mean = sum(ys) / float(n)
    sxx = sum((x - x_mean) * (x - x_mean) for x in xs)
    if abs(sxx) <= 1e-12:
        return {
            "equation": "expected = observed",
            "slope": 1.0,
            "intercept": 0.0,
            "r2": None,
        }
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = float(sxy / sxx)
    intercept = float(y_mean - slope * x_mean)
    y_hat = [float(slope * x + intercept) for x in xs]
    sst = sum((y - y_mean) * (y - y_mean) for y in ys)
    sse = sum((y - yh) * (y - yh) for y, yh in zip(ys, y_hat))
    r2 = None
    if sst > 1e-12:
        r2 = float(1.0 - (sse / sst))
    return {
        "equation": f"expected = {slope:.6f} * observed + {intercept:.6f}",
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
    }


def _fit_residual_stats(rows: list[SampleRow], variable: str, fit: dict) -> dict:
    """Return residual quality stats for the fitted linear model.

    Residuals are measured in expected-variable space:
    residual = predicted_expected - operator_expected.
    """
    if not rows:
        return {
            "residual_mean_error_mm": None,
            "residual_mae_mm": None,
            "residual_max_abs_error_mm": None,
        }

    try:
        slope = float(fit.get("slope", 1.0) or 1.0)
        intercept = float(fit.get("intercept", 0.0) or 0.0)
    except Exception:
        slope = 1.0
        intercept = 0.0

    residuals = []
    for item in rows:
        observed = float(_extract_observed_value(item, variable))
        expected = float(item.expected_value)
        predicted_expected = float(slope * observed + intercept)
        residuals.append(float(predicted_expected - expected))

    abs_residuals = [abs(value) for value in residuals]
    return {
        "residual_mean_error_mm": float(statistics.mean(residuals)) if residuals else None,
        "residual_mae_mm": float(statistics.mean(abs_residuals)) if abs_residuals else None,
        "residual_max_abs_error_mm": float(max(abs_residuals)) if abs_residuals else None,
    }


def _write_curve_plot(
    *,
    rows: list[SampleRow],
    variable: str,
    fit: dict,
    plot_path: Path,
) -> tuple[bool, str | None]:
    if not bool(MATPLOTLIB_AVAILABLE):
        return False, "matplotlib unavailable"
    if not rows:
        return False, "no samples"

    xs = [float(_extract_observed_value(item, variable)) for item in rows]
    ys = [float(item.expected_value) for item in rows]
    slope = float(fit.get("slope", 1.0) or 1.0)
    intercept = float(fit.get("intercept", 0.0) or 0.0)

    x_min = min(xs)
    x_max = max(xs)
    if abs(x_max - x_min) <= 1e-9:
        x_min -= 1.0
        x_max += 1.0
    span = x_max - x_min
    pad = max(1.0, 0.08 * span)
    x_lo = x_min - pad
    x_hi = x_max + pad

    fit_x = [x_lo, x_hi]
    fit_y = [float(slope * x + intercept) for x in fit_x]

    fig = None
    try:
        fig = plt.figure(figsize=(7.2, 4.8))
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys, c="#1f77b4", s=55, alpha=0.9, label="captured points")
        ax.plot(fit_x, fit_y, color="#d62728", linewidth=2.0, label="fit")
        for item in rows:
            px = float(_extract_observed_value(item, variable))
            py = float(item.expected_value)
            ax.annotate(str(item.index), (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)

        ax.set_title(f"Telemetry Calibration Curve ({variable})")
        ax.set_xlabel(f"Observed {variable} (mm)")
        ax.set_ylabel(f"Expected {variable} (mm)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=130)
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            pass


def _read_pose(
    vision,
    world,
    *,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None,
    min_samples_required: int | None,
):
    return shared_read_pose(
        vision,
        world,
        samples=int(samples),
        timeout_s=float(timeout_s),
        min_sample_time=min_sample_time,
        min_samples_required=min_samples_required,
        observe_sleep_s=float(OBSERVE_SLEEP_S),
        fallback_step_label="ALIGN_BRICK",
        update_world_from_vision=update_world_from_vision,
        latest_unique_smoothed_frames=telemetry_latest_unique_smoothed_frames,
        average_smoothed_frames=telemetry_average_smoothed_frames,
        lite_gate_unique_frames=lite_gate_unique_frames,
        min_lite_unique_frames=3,
    )


def _observe_pose_with_reobserve(
    *,
    vision,
    world,
    samples: int,
    timeout_s: float,
) -> tuple[dict | None, dict]:
    return shared_observe_pose_with_reobserve(
        read_pose_fn=_read_pose,
        log_fn=lambda line: print(str(line), flush=True),
        log_prefix="[CALIBRATE_TELEMETRY]",
        vision=vision,
        world=world,
        samples=int(samples),
        timeout_s=float(timeout_s),
        min_sample_time=None,
        hold_s=float(REOBSERVE_HOLD_S),
        reobserve_rounds=int(REOBSERVE_ROUNDS),
        relaxed_timeout_s=max(float(timeout_s), float(OBSERVE_TIMEOUT_S)),
    )


def _extract_pose_variable_value(pose: dict, variable: str) -> float:
    variable_key = str(variable or "dist").strip().lower()
    if variable_key == "x":
        return float(pose.get("offset_x") or 0.0)
    if variable_key == "y":
        return float(pose.get("offset_y") or 0.0)
    return float(pose.get("dist") or 0.0)


def _mean_pose_samples(poses: list[dict]) -> dict | None:
    if not poses:
        return None
    return {
        "offset_y": float(statistics.mean([float(item.get("offset_y") or 0.0) for item in poses])),
        "offset_x": float(statistics.mean([float(item.get("offset_x") or 0.0) for item in poses])),
        "dist": float(statistics.mean([float(item.get("dist") or 0.0) for item in poses])),
        "angle": float(statistics.mean([float(item.get("angle") or 0.0) for item in poses])),
        "confidence": float(statistics.mean([float(item.get("confidence") or 0.0) for item in poses])),
        "obs_ts": float(max(float(item.get("obs_ts") or 0.0) for item in poses)),
        "pose_source": "tight_capture_mean",
        "lite_required_frames": None,
        "samples_used": len(poses),
    }


def _capture_tight_pose(
    *,
    vision,
    world,
    variable: str,
    samples: int,
    timeout_s: float,
    max_spread_mm: float,
    trim_outliers: int,
) -> tuple[dict | None, dict]:
    target_samples = max(1, int(samples))
    per_sample_timeout_s = max(0.25, float(timeout_s) / float(target_samples))
    sample_poses: list[dict] = []
    for _ in range(target_samples):
        pose = _read_pose(
            vision,
            world,
            samples=1,
            timeout_s=per_sample_timeout_s,
            min_sample_time=None,
            min_samples_required=1,
        )
        if pose is None:
            return None, {
                "reason": "no_visible",
                "samples_used": len(sample_poses),
            }
        sample_poses.append(dict(pose))

    values = [float(_extract_pose_variable_value(item, variable)) for item in sample_poses]

    trim_limit = max(0, int(trim_outliers))
    trim_limit = min(trim_limit, max(0, len(sample_poses) - 3))

    selected_indices = list(range(len(sample_poses)))
    selected_values = list(values)
    spread_mm = float(max(selected_values) - min(selected_values)) if selected_values else 0.0
    dropped_count = 0

    if spread_mm > float(max_spread_mm) and trim_limit > 0 and len(sample_poses) >= 3:
        sorted_pairs = sorted([(float(v), idx) for idx, v in enumerate(values)], key=lambda item: item[0])

        def _best_window(keep_n: int) -> tuple[list[int], list[float], float]:
            best_idx: list[int] = []
            best_vals: list[float] = []
            best_spread: float | None = None
            for start in range(0, len(sorted_pairs) - keep_n + 1):
                window = sorted_pairs[start : start + keep_n]
                window_vals = [float(item[0]) for item in window]
                window_idx = [int(item[1]) for item in window]
                window_spread = float(max(window_vals) - min(window_vals)) if window_vals else 0.0
                if best_spread is None or window_spread < best_spread:
                    best_idx = window_idx
                    best_vals = window_vals
                    best_spread = float(window_spread)
            return best_idx, best_vals, float(best_spread or 0.0)

        for discard_count in range(1, trim_limit + 1):
            keep_n = len(sample_poses) - int(discard_count)
            cand_idx, cand_vals, cand_spread = _best_window(keep_n)
            if cand_spread <= float(max_spread_mm):
                selected_indices = list(cand_idx)
                selected_values = list(cand_vals)
                spread_mm = float(cand_spread)
                dropped_count = int(discard_count)
                break

        if dropped_count == 0:
            keep_n = len(sample_poses) - int(trim_limit)
            cand_idx, cand_vals, cand_spread = _best_window(keep_n)
            selected_indices = list(cand_idx)
            selected_values = list(cand_vals)
            spread_mm = float(cand_spread)
            dropped_count = int(trim_limit)

    selected_poses = [sample_poses[idx] for idx in selected_indices]
    dropped_values = [float(values[idx]) for idx in range(len(values)) if idx not in set(selected_indices)]

    if spread_mm > float(max_spread_mm):
        return None, {
            "reason": "spread_too_high",
            "spread_mm": float(spread_mm),
            "observed_values": list(values),
            "selected_values": list(selected_values),
            "dropped_values": list(dropped_values),
            "dropped_count": int(dropped_count),
            "samples_used": len(selected_poses),
        }

    pose = _mean_pose_samples(selected_poses)
    if pose is None:
        return None, {
            "reason": "no_samples",
            "samples_used": 0,
        }
    pose["observed_spread_mm"] = float(spread_mm)
    pose["dropped_outliers"] = int(dropped_count)
    return pose, {
        "reason": "ok",
        "spread_mm": float(spread_mm),
        "observed_values": list(values),
        "selected_values": list(selected_values),
        "dropped_values": list(dropped_values),
        "dropped_count": int(dropped_count),
        "samples_used": len(selected_poses),
    }


def _create_vision(vision_mode: str):
    mode = str(vision_mode or "").strip().lower()
    if mode == "yolo":
        if YoloBrickDetector is None:
            raise RuntimeError("YOLO detector module not installed; cannot use --vision yolo")
        return YoloBrickDetector(debug=False)
    if mode == "aruco":
        from helper_vision_aruco import ArucoBrickVision

        return ArucoBrickVision(debug=False)
    if mode == "cyan":
        if YoloBrickDetector is None:
            raise RuntimeError("YOLO detector module not installed; cannot use --vision cyan")
        return YoloBrickDetector(debug=False)
    return LeiaVision(debug=False)


def _maybe_close(resource) -> None:
    if resource is None:
        return
    try:
        close_fn = getattr(resource, "close", None)
        if callable(close_fn):
            close_fn()
            return
        stop_fn = getattr(resource, "stop", None)
        if callable(stop_fn):
            stop_fn()
    except Exception:
        return


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
    variable: str,
    rows_captured: int,
    points_total: int | None,
) -> None:
    if not isinstance(stream_state, dict):
        return

    frame = _current_vision_frame(vision)
    total_label = "unlimited" if points_total is None else str(int(points_total))
    text_lines = [
        f"Telemetry calibration: {str(variable)}",
        f"Points captured: {int(rows_captured)}/{total_label}",
    ]

    try:
        helper_xyz_coords.sync_from_world(world, reason="vision", render=False)
    except Exception:
        pass

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
        return
    with lock:
        stream_state["frame"] = frame
        stream_state["text_lines"] = text_lines


def _prompt_with_live_refresh(
    *,
    prompt: str,
    vision,
    world,
    enable_live_refresh: bool,
    refresh_interval_s: float,
    live_refresh_callback: Callable[[], None] | None = None,
) -> str:
    if not bool(enable_live_refresh):
        return input(prompt)

    stream = getattr(sys, "stdin", None)
    if stream is None or not hasattr(stream, "fileno"):
        return input(prompt)

    print(str(prompt), end="", flush=True)
    chunks: list[str] = []
    while True:
        try:
            ready, _, _ = select.select([stream], [], [], float(refresh_interval_s))
        except Exception:
            return input(str(prompt))

        if ready:
            line = stream.readline()
            return "".join(chunks) + str(line).rstrip("\r\n")

        try:
            # Observation-only refresh so livestream stays current while waiting for operator input.
            update_world_from_vision(world, vision, log=False)
        except Exception:
            pass
        if callable(live_refresh_callback):
            try:
                live_refresh_callback()
            except Exception:
                pass


def _hotkey_rows_for_calibration(hotkeys_csv: str | None) -> dict[str, dict]:
    rows = telemetry_robot_module.HOTKEY_SPEED_SCORES if isinstance(telemetry_robot_module.HOTKEY_SPEED_SCORES, dict) else {}
    allowed_cmds = {"f", "b", "u", "d", "l", "r"}
    requested_keys: list[str] = []
    if str(hotkeys_csv or "").strip():
        for token in str(hotkeys_csv or "").replace(";", ",").split(","):
            key = str(token or "").strip().lower()
            if key:
                requested_keys.append(key)
    if not requested_keys:
        requested_keys = ["w", "s", "o", "k"]

    out: dict[str, dict] = {}
    for key in requested_keys:
        row = rows.get(str(key)) if isinstance(rows, dict) else None
        if not isinstance(row, dict):
            continue
        cmd = str(row.get("cmd", "")).strip().lower()
        if cmd not in allowed_cmds:
            continue
        try:
            score = int(round(float(row.get("score", 1))))
        except Exception:
            score = 1
        out[str(key)] = {
            "hotkey": str(key),
            "cmd": str(cmd),
            "score": int(max(1, min(100, score))),
        }
    if not out:
        out = {
            "o": {"hotkey": "o", "cmd": "u", "score": 1},
            "k": {"hotkey": "k", "cmd": "d", "score": 1},
        }
    return out


def _resolve_stream_runtime(
    *,
    vision_mode: str,
    livestream_requested: bool,
) -> tuple[dict | None, str | None, bool]:
    shared_state, shared_url = get_shared_stream_runtime()
    if isinstance(shared_state, dict):
        return (
            prepare_shared_stream_state(
                shared_state,
                vision_mode="aruco" if str(vision_mode).strip().lower() == "aruco" else "cyan",
            ),
            str(shared_url or "").strip() or None,
            True,
        )
    if not bool(livestream_requested):
        return None, None, False
    return (
        {
            "frame": None,
            "text_lines": [],
            "lock": threading.Lock(),
            "show_center_line": True,
            "vision_mode": "aruco" if str(vision_mode).strip().lower() == "aruco" else "cyan",
        },
        None,
        False,
    )


def _stream_extra_lines_for_capture(*, variable: str, rows_captured: int, points_total: int | None) -> list[str]:
    total_label = "unlimited" if points_total is None else str(int(points_total))
    return [
        f"Telemetry calibration: {str(variable)}",
        f"Points captured: {int(rows_captured)}/{total_label}",
    ]


def _update_shared_stream_extra_lines(
    stream_state: dict | None,
    *,
    variable: str,
    rows_captured: int,
    points_total: int | None,
) -> None:
    if not isinstance(stream_state, dict):
        return
    lock = stream_state.get("lock")
    lines = _stream_extra_lines_for_capture(
        variable=variable,
        rows_captured=rows_captured,
        points_total=points_total,
    )

    def _apply():
        stream_state["extra_text_lines"] = list(lines)
        text_lines = list(stream_state.get("telemetry_lines") or [])
        if text_lines:
            text_lines.append("")
        text_lines.extend(lines)
        stream_state["text_lines"] = text_lines

    if lock is None:
        _apply()
        return
    with lock:
        _apply()


def _clear_shared_stream_extra_lines(stream_state: dict | None) -> None:
    if not isinstance(stream_state, dict):
        return
    lock = stream_state.get("lock")

    def _apply():
        stream_state["extra_text_lines"] = []
        stream_state["text_lines"] = list(stream_state.get("telemetry_lines") or [])

    if lock is None:
        _apply()
        return
    with lock:
        _apply()


def _send_hotkey_motion(*, robot, world, row: dict) -> None:
    cmd = str(row.get("cmd") or "").strip().lower()
    score = int(row.get("score") or 1)
    power, pwm, score_used, duration_ms = speed_power_pwm_for_cmd(cmd, score)
    send_robot_command_pwm(
        robot,
        world,
        StepState.ALIGN_BRICK,
        cmd,
        power,
        pwm,
        int(duration_ms),
        speed_score=score_used,
        auto_mode=False,
    )
    print(
        f"\n[CALIBRATE_TELEMETRY] Hotkey {str(row.get('hotkey')).upper()} -> "
        f"cmd={str(cmd).upper()} score={int(score_used)} duration={int(duration_ms)}ms"
    )


def _prompt_with_hotkeys_and_live_refresh(
    *,
    prompt: str,
    vision,
    world,
    enable_live_refresh: bool,
    refresh_interval_s: float,
    hotkey_rows: dict[str, dict],
    hotkey_handler: Callable[[dict], None] | None,
    live_refresh_callback: Callable[[], None] | None = None,
) -> str:
    stream = getattr(sys, "stdin", None)
    if stream is None or not hasattr(stream, "fileno") or not callable(hotkey_handler):
        return ""

    fd = None
    saved_attrs = None
    print(str(prompt), end="", flush=True)
    try:
        if not stream.isatty():
            return ""
        fd = stream.fileno()
        saved_attrs = termios.tcgetattr(fd)
        # Read single characters without echo while preserving normal terminal
        # output processing so prompt/log newlines still render correctly.
        _enable_single_char_noecho_mode(fd, saved_attrs)

        while True:
            timeout_s = float(refresh_interval_s) if bool(enable_live_refresh) else 0.25
            try:
                ready, _, _ = select.select([fd], [], [], max(0.05, timeout_s))
            except Exception:
                return ""

            if ready:
                try:
                    ch = os.read(fd, 1)
                except Exception:
                    ch = b""
                if not ch:
                    continue
                key = ch.decode(errors="ignore")
                if key in ("\n", "\r"):
                    print("")
                    return ""
                if key in ("\x03",):
                    raise KeyboardInterrupt
                if key in ("\x1b",):
                    print("")
                    return "quit"
                key_lower = str(key).lower()
                if key_lower in hotkey_rows:
                    hotkey_handler(dict(hotkey_rows[key_lower]))
                    if bool(enable_live_refresh):
                        try:
                            update_world_from_vision(world, vision, log=False)
                        except Exception:
                            pass
                        if callable(live_refresh_callback):
                            try:
                                live_refresh_callback()
                            except Exception:
                                pass
                    print(str(prompt), end="", flush=True)
                    continue
                # Ignore non-hotkey characters here. Numeric entry happens in a
                # plain line-input prompt after the operator presses Enter.
                continue

            if bool(enable_live_refresh):
                try:
                    update_world_from_vision(world, vision, log=False)
                except Exception:
                    pass
                if callable(live_refresh_callback):
                    try:
                        live_refresh_callback()
                    except Exception:
                        pass
    finally:
        try:
            if fd is not None and saved_attrs is not None:
                termios.tcsetattr(fd, termios.TCSANOW, saved_attrs)
        except Exception:
            pass


def _prompt_expected_value_text(prompt: str) -> str:
    _restore_tty_line_input_mode()
    try:
        return str(input(str(prompt or "")))
    except (EOFError, KeyboardInterrupt):
        return "q"
    finally:
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


def run_telemetry_variable_calibration(default_variable: str) -> int:
    parser = argparse.ArgumentParser(description=f"Interactive telemetry calibration for {default_variable}.")
    parser.add_argument("--variable", choices=["dist", "x", "y"], default=str(default_variable))
    parser.add_argument(
        "--points",
        type=int,
        default=0,
        help="How many labeled samples to collect (default: unlimited; set >0 for fixed count).",
    )
    parser.add_argument("--vision", choices=["leia", "yolo", "aruco", "cyan"], default="cyan")
    parser.add_argument(
        "--observe-samples",
        type=int,
        default=OBSERVE_SAMPLES_DEFAULT,
        help="How many frames to sample after you type expected value and press Enter (default: 5).",
    )
    parser.add_argument("--observe-timeout-s", type=float, default=OBSERVE_TIMEOUT_S)
    parser.add_argument(
        "--capture-max-spread-mm",
        type=float,
        default=CAPTURE_MAX_SPREAD_MM_DEFAULT,
        help="Maximum allowed spread across captured frames for the selected variable (default: 3.0 mm).",
    )
    parser.add_argument(
        "--capture-trim-outliers",
        type=int,
        default=CAPTURE_TRIM_OUTLIERS_DEFAULT,
        help="Discard up to this many outlier frames before spread check (default: 2).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None, help="Optional PNG output for observed vs expected curve.")
    parser.add_argument(
        "--targets-mm",
        type=str,
        default=None,
        help="Optional comma-separated expected targets (mm), one per point.",
    )
    parser.add_argument(
        "--target-start-mm",
        type=float,
        default=65.0,
        help="Default dist plan start target in mm when --targets-mm is not provided (default: 65).",
    )
    parser.add_argument(
        "--target-step-mm",
        type=float,
        default=40.0,
        help="Default dist plan step in mm when --targets-mm is not provided (default: 40).",
    )
    parser.add_argument(
        "--world-model-file",
        type=str,
        default=str(ROBOT_MODEL_FILE),
        help="World model JSON to update with telemetry observation curve details.",
    )
    parser.add_argument(
        "--staging-file",
        type=str,
        default=str(STAGING_FILE_DEFAULT),
        help="Staging JSON file that accumulates telemetry calibration runs before world-model commit.",
    )
    parser.add_argument(
        "--commit-world-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Commit staged telemetry calibration bundle into world model after this run (default: enabled).",
    )
    parser.add_argument(
        "--allow-partial-commit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow world-model commit even if staged variables do not include dist/x/y together (default: enabled).",
    )
    parser.add_argument(
        "--livestream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable in-run livestream feed during calibration (default: enabled).",
    )
    parser.add_argument("--stream-host", type=str, default="127.0.0.1")
    parser.add_argument("--stream-port", type=int, default=5000)
    parser.add_argument("--stream-fps", type=int, default=10)
    parser.add_argument("--stream-jpeg-quality", type=int, default=85)
    parser.add_argument("--stream-img-width", type=int, default=STREAM_IMG_WIDTH)
    parser.add_argument(
        "--prompt-refresh-s",
        type=float,
        default=0.15,
        help="Vision refresh interval while waiting at prompts when livestream is enabled.",
    )
    parser.add_argument(
        "--hotkeys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable manual hotkey motion during prompts (default: enabled).",
    )
    parser.add_argument(
        "--hotkeys-csv",
        type=str,
        default="w,s,o,k",
        help="Comma-separated hotkeys to allow during prompts (default: w,s,o,k).",
    )
    args, _passthrough = parser.parse_known_args(sys.argv[1:])

    variable = str(args.variable)
    points = max(0, int(args.points))
    unlimited_points = int(points) == 0
    observe_samples = max(1, int(args.observe_samples))
    observe_timeout_s = max(0.2, float(args.observe_timeout_s))
    capture_max_spread_mm = max(0.0, float(args.capture_max_spread_mm))
    capture_trim_outliers = max(0, int(args.capture_trim_outliers))
    shared_context = get_shared_calibration_context()
    shared_world = shared_context.get("world") if isinstance(shared_context, dict) else None
    shared_vision = shared_context.get("vision") if isinstance(shared_context, dict) else None
    shared_robot = shared_context.get("robot") if isinstance(shared_context, dict) else None
    using_shared_live_runtime = all(item is not None for item in (shared_world, shared_vision, shared_robot))
    stream_state, shared_stream_url, using_shared_stream = _resolve_stream_runtime(
        vision_mode=str(args.vision),
        livestream_requested=bool(args.livestream),
    )
    livestream_enabled = bool(stream_state is not None)
    stream_url = str(shared_stream_url or format_stream_url(str(args.stream_host), int(args.stream_port)))
    try:
        planned_targets = None
        if not bool(unlimited_points):
            planned_targets = _planned_expected_targets(
                variable=variable,
                points=int(points),
                targets_mm_raw=args.targets_mm,
                target_start_mm=float(args.target_start_mm),
                target_step_mm=float(args.target_step_mm),
            )
    except Exception as exc:
        print(f"[CALIBRATE_TELEMETRY] Invalid target plan: {exc}")
        return 2
    results_path = Path(args.results_file) if args.results_file else _default_results_path(
        variable=variable,
        vision=str(args.vision),
        output_dir=args.output_dir,
    )
    plot_path = Path(args.plot_path) if args.plot_path else _default_plot_path(
        variable=variable,
        vision=str(args.vision),
        output_dir=args.output_dir,
    )

    print("[CALIBRATE_TELEMETRY] Starting manual telemetry calibration.")
    points_label = "unlimited" if bool(unlimited_points) else str(int(points))
    print(
        f"[CALIBRATE_TELEMETRY] variable={variable} points={points_label} vision={args.vision} "
        f"observe_samples={observe_samples}"
    )
    print("[CALIBRATE_TELEMETRY] World-model policy: partial commit enabled by default.")
    if planned_targets:
        target_labels = ", ".join(f"{float(value):.1f}" for value in planned_targets)
        print(f"[CALIBRATE_TELEMETRY] Planned expected targets ({variable}, mm): {target_labels}")
    if bool(using_shared_stream):
        print(f"[CALIBRATE_TELEMETRY] Using shared livestream: {_orange_text(stream_url)}")
        if bool(using_shared_live_runtime):
            print("[CALIBRATE_TELEMETRY] Reusing live manual-training vision and robot runtime.")
    elif bool(livestream_enabled):
        print(f"[CALIBRATE_TELEMETRY] Livestream URL: {_orange_text(stream_url)}")
    else:
        print("[CALIBRATE_TELEMETRY] Livestream disabled (--no-livestream).")
    print("[CALIBRATE_TELEMETRY] Use your normal hotkeys to move robot between points.")
    print(
        "[CALIBRATE_TELEMETRY] For each point: use hotkeys to move, type the expected value, "
        "then press Enter to capture averaged frames."
    )
    print(
        f"[CALIBRATE_TELEMETRY] Capture settings: frames={int(observe_samples)} "
        f"trim_outliers<= {int(capture_trim_outliers)} spread_limit={float(capture_max_spread_mm):.2f}mm"
    )
    print("[CALIBRATE_TELEMETRY] Type 'q' at any prompt to quit early.")

    rows: list[SampleRow] = []
    world = None
    vision = None
    stream_server = None
    robot = None
    original_step_state = None
    original_post_action_observe_delay_s = None

    try:
        if bool(using_shared_live_runtime):
            world = shared_world
            robot = shared_robot
            vision = shared_vision
        else:
            world = WorldModel()
            world.step_state = StepState.ALIGN_BRICK
            world._post_action_observe_delay_s = 0.0
            try:
                helper_xyz_coords.sync_from_world(world, reason="init", render=False)
            except Exception:
                pass
            robot = Robot()
            vision = _create_vision(str(args.vision))
        original_step_state = getattr(world, "step_state", None)
        original_post_action_observe_delay_s = getattr(world, "_post_action_observe_delay_s", None)
        world.step_state = StepState.ALIGN_BRICK
        world._post_action_observe_delay_s = 0.0
        hotkey_rows = _hotkey_rows_for_calibration(args.hotkeys_csv if bool(args.hotkeys) else "")

        def _live_refresh_callback():
            if bool(using_shared_live_runtime):
                _update_shared_stream_extra_lines(
                    stream_state,
                    variable=variable,
                    rows_captured=len(rows),
                    points_total=None if bool(unlimited_points) else int(points),
                )
            else:
                _refresh_stream_state(
                    stream_state=stream_state,
                    vision=vision,
                    world=world,
                    variable=variable,
                    rows_captured=len(rows),
                    points_total=None if bool(unlimited_points) else int(points),
                )

        if bool(args.hotkeys):
            hotkey_labels = ", ".join(
                f"{k}->{str(v.get('cmd')).upper()}@{int(v.get('score') or 1)}"
                for k, v in hotkey_rows.items()
            )
            print(f"[CALIBRATE_TELEMETRY] Prompt hotkeys enabled: {hotkey_labels}")
            print("[CALIBRATE_TELEMETRY] To quit prompt, type 'quit' then press Enter.")

        if bool(livestream_enabled):
            _live_refresh_callback()
            if not bool(using_shared_stream):
                try:
                    stream_server, stream_url = start_stream_server(
                        stream_state,
                        title="Telemetry Calibration Livestream",
                        header="",
                        footer="<div class='footer-sections'><div class='footer-section'><div class='footer-title'>Telemetry Calibration</div><div>Manual hotkey positioning with sampled dist/x/y capture.</div></div></div>",
                        host=str(args.stream_host),
                        port=int(args.stream_port),
                        fps=max(1, int(args.stream_fps)),
                        jpeg_quality=max(1, min(100, int(args.stream_jpeg_quality))),
                        img_width=max(320, int(args.stream_img_width)),
                        vision_mode_options=STREAM_VISION_MODE_OPTIONS,
                        xyz_workspace_getter=lambda: getattr(world, "_xyz_workspace", None),
                    )
                    actual_port = getattr(stream_server, "port", int(args.stream_port))
                    if int(actual_port) != int(args.stream_port):
                        print(f"[CALIBRATE_TELEMETRY] Stream port {int(args.stream_port)} busy; using {int(actual_port)}")
                    print(f"[CALIBRATE_TELEMETRY] Livestream started: {_orange_text(stream_url)}")
                except Exception as exc:
                    print(f"[CALIBRATE_TELEMETRY] Livestream startup failed at {_orange_text(stream_url)}: {exc}")
                    _maybe_close(stream_server)
                    stream_server = None
                    livestream_enabled = False

        stop_requested = False
        idx = 1
        while True:
            if (not bool(unlimited_points)) and idx > int(points):
                break
            while True:
                print("")
                target_hint = None if planned_targets is None else float(planned_targets[idx - 1])
                print(
                    f"[CALIBRATE_TELEMETRY] Point {idx}/{points_label} - "
                    f"target variable: {variable} ({_variable_unit(variable)})"
                )
                if target_hint is not None:
                    print(
                        f"[CALIBRATE_TELEMETRY] Suggested setup target for point {idx}: "
                        f"~{target_hint:.1f}mm"
                    )
                print(
                    f"[CALIBRATE_TELEMETRY] Use motion hotkeys, then press Enter to type expected {variable}."
                )
                ready_action = _prompt_with_hotkeys_and_live_refresh(
                    prompt=_ready_for_capture_prompt(variable),
                    vision=vision,
                    world=world,
                    enable_live_refresh=bool(livestream_enabled),
                    refresh_interval_s=max(0.05, float(args.prompt_refresh_s)),
                    hotkey_rows=hotkey_rows,
                    hotkey_handler=(
                        None
                        if not bool(args.hotkeys)
                        else (lambda row: _send_hotkey_motion(robot=robot, world=world, row=row))
                    ),
                    live_refresh_callback=_live_refresh_callback,
                ).strip().lower()
                if ready_action in ("q", "quit", "exit"):
                    print("[CALIBRATE_TELEMETRY] Stopped by operator.")
                    stop_requested = True
                    break
                entry_prompt = _expected_value_prompt(variable, target_hint)
                expected_text = _prompt_expected_value_text(entry_prompt).strip().lower()
                if expected_text in ("q", "quit", "exit"):
                    print("[CALIBRATE_TELEMETRY] Stopped by operator.")
                    stop_requested = True
                    break
                if expected_text == "" and target_hint is not None:
                    expected_text = str(target_hint)

                try:
                    expected_value = float(expected_text)
                    if not math.isfinite(float(expected_value)):
                        raise ValueError("Expected value must be finite.")
                except Exception:
                    print("[CALIBRATE_TELEMETRY] Invalid expected value. Try again.")
                    continue

                pose = None
                capture_meta = {}
                retry_action = ""
                while pose is None:
                    pose, capture_meta = _capture_tight_pose(
                        vision=vision,
                        world=world,
                        variable=variable,
                        samples=observe_samples,
                        timeout_s=observe_timeout_s,
                        max_spread_mm=capture_max_spread_mm,
                        trim_outliers=capture_trim_outliers,
                    )
                    _live_refresh_callback()
                    if pose is not None:
                        break
                    if str(capture_meta.get("reason") or "") == "spread_too_high":
                        observed_values = capture_meta.get("observed_values") or []
                        selected_values = capture_meta.get("selected_values") or []
                        dropped_values = capture_meta.get("dropped_values") or []
                        observed_values_text = ", ".join(f"{float(value):.2f}" for value in observed_values)
                        selected_values_text = ", ".join(f"{float(value):.2f}" for value in selected_values)
                        dropped_values_text = ", ".join(f"{float(value):.2f}" for value in dropped_values)
                        print(
                            f"[CALIBRATE_TELEMETRY] Capture spread too high for {variable}: "
                            f"{float(capture_meta.get('spread_mm') or 0.0):.2f}mm "
                            f"(limit {float(capture_max_spread_mm):.2f}mm)."
                        )
                        if observed_values_text:
                            print(
                                f"[CALIBRATE_TELEMETRY] Observed {variable} frame values: {observed_values_text}"
                            )
                        if selected_values_text:
                            print(
                                f"[CALIBRATE_TELEMETRY] Best kept values after trimming: {selected_values_text}"
                            )
                        if dropped_values_text:
                            print(
                                f"[CALIBRATE_TELEMETRY] Dropped outlier values: {dropped_values_text}"
                            )
                    else:
                        print("[CALIBRATE_TELEMETRY] No visible brick telemetry.")
                    retry_action = input("[Enter=retry, s=skip, q=quit]: ").strip().lower()
                    if retry_action in ("q", "quit", "exit"):
                        print("[CALIBRATE_TELEMETRY] Stopped by operator.")
                        stop_requested = True
                        break
                    if retry_action in ("s", "skip"):
                        break
                if stop_requested:
                    break
                if pose is None:
                    if retry_action in ("s", "skip"):
                        print("[CALIBRATE_TELEMETRY] Point skipped.")
                        break
                    continue

                observed_dist = float(pose.get("dist") or 0.0)
                observed_x = float(pose.get("offset_x") or 0.0)
                observed_y = float(pose.get("offset_y") or 0.0)
                observed_value = observed_dist if variable == "dist" else observed_x if variable == "x" else observed_y
                print(
                    "[CALIBRATE_TELEMETRY] Captured telemetry: "
                    f"dist={observed_dist:.2f}mm x={observed_x:.2f}mm y={observed_y:.2f}mm "
                    f"confidence={float(pose.get('confidence') or 0.0):.3f} "
                    f"samples={int(pose.get('samples_used') or 0)} "
                    f"spread={float(pose.get('observed_spread_mm') or 0.0):.2f}mm "
                    f"dropped={int(pose.get('dropped_outliers') or 0)}"
                )
                print(f"[CALIBRATE_TELEMETRY] Observed {variable}={observed_value:.2f}mm")

                row = SampleRow(
                    index=int(len(rows) + 1),
                    expected_value=float(expected_value),
                    observed_dist=float(observed_dist),
                    observed_x=float(observed_x),
                    observed_y=float(observed_y),
                    confidence=float(pose.get("confidence") or 0.0),
                    samples_used=int(pose.get("samples_used")) if pose.get("samples_used") is not None else None,
                    pose_source=str(pose.get("pose_source") or "unknown"),
                    observation_mode=str(capture_meta.get("reason") or "unknown"),
                    ts=float(time.time()),
                )
                rows.append(row)
                _live_refresh_callback()

                error = float(_extract_observed_value(row, variable) - float(expected_value))
                print(
                    f"[CALIBRATE_TELEMETRY] Saved point {row.index}: expected={expected_value:.2f}, "
                    f"observed={_extract_observed_value(row, variable):.2f}, error={error:+.2f}mm"
                )
                break
            if stop_requested:
                break
            idx += 1

    finally:
        if world is not None:
            try:
                world.step_state = original_step_state
            except Exception:
                pass
            try:
                world._post_action_observe_delay_s = original_post_action_observe_delay_s
            except Exception:
                pass
        if bool(using_shared_live_runtime):
            _clear_shared_stream_extra_lines(stream_state)
        _maybe_close(stream_server)
        if not bool(using_shared_live_runtime):
            _maybe_close(vision)

    if not rows:
        print("[CALIBRATE_TELEMETRY] No samples captured.")
        return 1

    payload = _payload_from_rows(
        rows=rows,
        variable=str(variable),
        config={
            "variable": str(variable),
            "vision": str(args.vision),
            "requested_points": int(points),
            "captured_points": int(len(rows)),
            "observe_samples": int(observe_samples),
            "observe_timeout_s": float(observe_timeout_s),
            "manual_motion": True,
            "motion_method": "operator_hotkeys",
            "commit_world_model": bool(args.commit_world_model),
            "allow_partial_commit": bool(args.allow_partial_commit),
            "plot_path": str(plot_path),
        },
    )
    fit = dict(payload.get("fit") or {})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2))

    plot_ok, plot_error = _write_curve_plot(
        rows=rows,
        variable=variable,
        fit=fit,
        plot_path=plot_path,
    )

    staging_path = Path(str(args.staging_file))
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    staged_payload = _upsert_staging_telemetry_calibration(
        staging_path=staging_path,
        variable=str(variable),
        run_payload=payload,
    )

    world_model_update_status = "not requested"
    world_model_path = Path(str(args.world_model_file))
    if bool(args.commit_world_model):
        staged_vars = set((staged_payload.get("by_variable") or {}).keys())
        required_vars = {"dist", "x", "y"}
        missing_vars = sorted(required_vars - staged_vars)
        if missing_vars and not bool(args.allow_partial_commit):
            world_model_update_status = (
                "skipped commit (missing staged variable(s): "
                + ", ".join(missing_vars)
                + ")"
            )
        else:
            try:
                _upsert_world_model_telemetry_calibration(
                    world_model_path=world_model_path,
                    staged_payload=staged_payload,
                )
                world_model_update_status = f"updated {world_model_path}"
            except Exception as exc:
                world_model_update_status = f"failed to update world model ({exc})"

    print("")
    print("[CALIBRATE_TELEMETRY] Calibration complete.")
    print(f"[CALIBRATE_TELEMETRY] Captured points: {len(rows)}")
    print(f"[CALIBRATE_TELEMETRY] Fit: {str(fit.get('equation') or 'n/a')}")
    print(
        f"[CALIBRATE_TELEMETRY] Error stats: "
        f"mean={float(payload['summary']['mean_error_mm'] or 0.0):+.2f}mm "
        f"MAE={float(payload['summary']['mae_mm'] or 0.0):.2f}mm "
        f"max_abs={float(payload['summary']['max_abs_error_mm'] or 0.0):.2f}mm"
    )
    print(
        f"[CALIBRATE_TELEMETRY] Fit residuals: "
        f"mean={float(payload['summary']['residual_mean_error_mm'] or 0.0):+.2f}mm "
        f"MAE={float(payload['summary']['residual_mae_mm'] or 0.0):.2f}mm "
        f"max_abs={float(payload['summary']['residual_max_abs_error_mm'] or 0.0):.2f}mm"
    )
    print(f"[CALIBRATE_TELEMETRY] Results written: {results_path}")
    if plot_ok:
        print(f"[CALIBRATE_TELEMETRY] Curve plot written: {plot_path}")
    elif plot_error:
        print(f"[CALIBRATE_TELEMETRY] Curve plot skipped: {plot_error}")
    print(f"[CALIBRATE_TELEMETRY] Staging file: {staging_path}")
    print(f"[CALIBRATE_TELEMETRY] World model telemetry curve: {world_model_update_status}")
    return 0


def _print_menu() -> None:
    print("\nTelemetry Value Detection Calibration")
    print("-------------------------------------")
    print("Calibrate how to interpret sensor readings to determine actual distances/positions.")
    print("Robot movement is manual via your normal hotkeys.\n")
    for index, option in enumerate(OPTIONS, start=1):
        print(f"  {index}. {option.label} [{option.key}]")
    print("  q. Quit")


def _resolve_choice(text: str) -> CalibrateOption | None:
    token = str(text or "").strip().lower()
    if not token:
        return None
    for index, option in enumerate(OPTIONS, start=1):
        if token in (str(index), str(option.key).lower()):
            return option
    return None


def _pick_interactive() -> CalibrateOption | None:
    while True:
        _print_menu()
        choice = input("Select calibration to run: ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            return None
        selected = _resolve_choice(choice)
        if selected is not None:
            return selected
        print(f"Unknown selection: {choice!r}. Please choose a number, key, or q.")


def _run_selected(option: CalibrateOption, passthrough_args: list[str]) -> int:
    original_argv = list(sys.argv)
    try:
        sys.argv = [f"telemetry-calibrate:{option.key}"] + list(passthrough_args)
        result = option.runner()
        if result is None:
            return 0
        return int(result)
    finally:
        sys.argv = original_argv


def _restore_tty_line_input_mode() -> None:
    """Best-effort restore for terminals left in raw/no-echo mode."""
    stream = getattr(sys, "stdin", None)
    if stream is None:
        return
    try:
        if not stream.isatty():
            return
        fd = stream.fileno()
        attrs = termios.tcgetattr(fd)
        iflag = int(attrs[0])
        lflag = int(attrs[3])
        # Ensure Enter maps CR->NL so input() receives a completed line.
        attrs[0] = (iflag | termios.ICRNL) & ~termios.INLCR & ~termios.IGNCR
        attrs[3] = lflag | termios.ICANON | termios.ECHO | termios.ISIG
        cc = attrs[6]
        cc[termios.VMIN] = 1
        cc[termios.VTIME] = 0
        attrs[6] = cc
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        # Keep launcher resilient across non-POSIX shells or redirected stdin.
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive calibration launcher for telemetry value detection"
    )
    parser.add_argument(
        "--choice",
        type=str,
        default=None,
        help="Optional non-interactive selection key or menu number.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available options and exit.",
    )
    args, passthrough_args = parser.parse_known_args()

    if bool(args.list):
        _print_menu()
        return 0

    if args.choice is not None:
        selected = _resolve_choice(str(args.choice))
        if selected is None:
            print(f"Unknown --choice value: {args.choice!r}")
            _print_menu()
            return 2
    else:
        _restore_tty_line_input_mode()
        selected = _pick_interactive()
        if selected is None:
            print("No calibration selected.")
            return 0

    print(f"Running: {selected.label}")
    return _run_selected(selected, passthrough_args)


if __name__ == "__main__":
    raise SystemExit(main())
