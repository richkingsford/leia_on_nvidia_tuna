#!/usr/bin/env python3
"""Shared calibration helpers for x/y/dist curve discovery scripts."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

RUN_FOLDERS_DEFAULT = (
    Path("Runs - aruco"),
    Path("Runs - cyan"),
)


def cleanup_old_run_files(
    *,
    preserve_live_files: Iterable[str],
    run_folders: Sequence[Path] = RUN_FOLDERS_DEFAULT,
    cutoff_age_s: float = 3600.0,
) -> None:
    preserve = {str(name) for name in preserve_live_files or []}
    cutoff = time.time() - float(cutoff_age_s)
    for run_dir in run_folders:
        path = Path(run_dir)
        if not path.exists():
            continue
        try:
            for fp in path.iterdir():
                try:
                    if fp.name in preserve:
                        continue
                    if fp.is_file() and fp.stat().st_mtime < cutoff:
                        fp.unlink()
                    elif fp.is_symlink() and not fp.resolve().exists():
                        fp.unlink()
                except Exception:
                    pass
        except Exception:
            pass


def ensure_run_dir(
    *,
    run_dir: Path,
    preserve_live_files: Iterable[str],
    run_folders: Sequence[Path] = RUN_FOLDERS_DEFAULT,
    cutoff_age_s: float = 3600.0,
) -> None:
    cleanup_old_run_files(
        preserve_live_files=preserve_live_files,
        run_folders=run_folders,
        cutoff_age_s=float(cutoff_age_s),
    )
    Path(run_dir).mkdir(exist_ok=True)


def coerce_float(value, fallback=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def coerce_int(value, fallback=None):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def coerce_finite_float(value) -> float | None:
    number = coerce_float(value)
    if number is None or not math.isfinite(float(number)):
        return None
    return float(number)


def trial_label_text(
    trial_idx: int,
    trials_planned: int,
    *,
    phase: str = "primary",
    source_trial: int | None = None,
) -> str:
    if str(phase or "primary").strip().lower() == "repeat":
        source = int(source_trial) if source_trial is not None else int(trial_idx)
        return f"Repeat {int(trial_idx)}/{int(trials_planned)} (trial {source})"
    return f"Trial {int(trial_idx)}/{int(trials_planned)}"


def plot_series_phase(kind: str | None = None) -> str:
    return "repeat" if str(kind or "").strip().lower() == "repeat" else "primary"


def plot_offsets(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    if not xs or not ys:
        return [(float("nan"), float("nan"))]
    return list(zip(xs, ys))


def build_linear_duration_schedule(
    *,
    trials: int | None,
    min_duration_ms: int,
    max_duration_ms: int,
    duration_step_ms: int,
) -> list[int]:
    low = max(1, int(min_duration_ms))
    high = max(low, int(max_duration_ms))
    step = max(1, int(duration_step_ms))
    schedule = list(range(int(low), int(high) + 1, int(step)))
    if not schedule:
        schedule = [int(low)]
    if trials is None:
        return schedule
    return schedule[: max(1, int(trials))]


def build_repeated_trial_plan(
    *,
    durations_ms: list[int],
    cmd_sequence: Sequence[str],
    normalize_cmd: Callable[[str], str],
    trials: int | None = None,
) -> list[dict]:
    total = None if trials is None else max(1, int(trials))
    plan: list[dict] = []
    for duration_ms in durations_ms:
        for cmd in cmd_sequence:
            plan.append(
                {
                    "duration_ms": int(duration_ms),
                    "cmd": str(normalize_cmd(cmd)),
                }
            )
            if total is not None and len(plan) >= int(total):
                return plan
    return plan


def planned_durations_ms(trial_plan: list[dict]) -> list[int]:
    durations: list[int] = []
    seen: set[int] = set()
    for step in trial_plan:
        duration_ms = max(1, int(coerce_int(step.get("duration_ms"), 1) or 1))
        if duration_ms in seen:
            continue
        seen.add(duration_ms)
        durations.append(int(duration_ms))
    return durations


def write_results(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))


def observed_brick_distances_mm(*, trials: list, reset_efforts: list) -> list[float]:
    distances: list[float] = []
    for collection in (trials, reset_efforts):
        for item in collection:
            for value in (
                getattr(item, "pre_brick_dist_mm", None),
                getattr(item, "post_brick_dist_mm", None),
            ):
                number = coerce_finite_float(value)
                if number is not None:
                    distances.append(float(number))
    return distances


def _row_to_payload_dict(item) -> dict:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return dict(getattr(item, "__dict__", {}) or {})


def build_payload(
    *,
    source: str,
    config: dict,
    durations_ms: list[int],
    trials: list,
    reset_efforts: list,
    status: str,
    abort_reason: str | None,
    extra_fields: dict | None = None,
) -> dict:
    primary_trials = [item for item in trials if str(getattr(item, "phase", "primary")) != "repeat"]
    repeat_trials = [item for item in trials if str(getattr(item, "phase", "primary")) == "repeat"]
    cmd_deltas = [float(getattr(item, "cmd_delta_mm")) for item in primary_trials]
    repeat_cmd_deltas = [float(getattr(item, "cmd_delta_mm")) for item in repeat_trials]
    reset_cmd_deltas = [float(getattr(item, "cmd_delta_mm")) for item in reset_efforts]
    wrong_way_count = sum(1 for item in primary_trials if bool(getattr(item, "wrong_way", False)))
    repeat_wrong_way_count = sum(1 for item in repeat_trials if bool(getattr(item, "wrong_way", False)))
    reset_wrong_way_count = sum(1 for item in reset_efforts if bool(getattr(item, "wrong_way", False)))
    brick_distances_mm = observed_brick_distances_mm(trials=trials, reset_efforts=reset_efforts)
    payload = {
        "schema_version": 1,
        "source": str(source),
        "generated_at": time.time(),
        "config": dict(config),
        "durations_ms": list(durations_ms),
        "summary": {
            "trial_count": len(primary_trials),
            "repeat_trial_count": len(repeat_trials),
            "median_distance_mm": float(statistics.median(cmd_deltas)) if cmd_deltas else None,
            "mean_distance_mm": float(statistics.mean(cmd_deltas)) if cmd_deltas else None,
            "min_distance_mm": float(min(cmd_deltas)) if cmd_deltas else None,
            "max_distance_mm": float(max(cmd_deltas)) if cmd_deltas else None,
            "wrong_way_count": int(wrong_way_count),
            "repeat_median_distance_mm": float(statistics.median(repeat_cmd_deltas)) if repeat_cmd_deltas else None,
            "repeat_wrong_way_count": int(repeat_wrong_way_count),
            "reset_effort_count": len(reset_efforts),
            "reset_median_distance_mm": float(statistics.median(reset_cmd_deltas)) if reset_cmd_deltas else None,
            "reset_wrong_way_count": int(reset_wrong_way_count),
            "brick_distance_median_mm": float(statistics.median(brick_distances_mm)) if brick_distances_mm else None,
            "brick_distance_min_mm": float(min(brick_distances_mm)) if brick_distances_mm else None,
            "brick_distance_max_mm": float(max(brick_distances_mm)) if brick_distances_mm else None,
            "status": str(status),
            "abort_reason": abort_reason,
        },
        "reset_efforts": [_row_to_payload_dict(item) for item in reset_efforts],
        "trials": [_row_to_payload_dict(item) for item in trials],
    }
    if isinstance(extra_fields, dict):
        payload.update(json.loads(json.dumps(extra_fields)))
    return payload


def world_step_label(world, *, fallback: str = "ALIGN_BRICK") -> str:
    step_state = getattr(world, "step_state", None)
    step_value = getattr(step_state, "value", step_state)
    step_text = str(step_value or fallback).strip()
    return step_text or str(fallback)


def pose_from_measurement(
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


def lite_pose_from_world(
    world,
    *,
    step: str,
    samples: int,
    obs_ts: float,
    latest_unique_smoothed_frames: Callable,
    average_smoothed_frames: Callable,
    lite_gate_unique_frames: Callable,
    process_rules=None,
    min_lite_unique_frames: int = 3,
) -> dict | None:
    required_frames = max(1, int(lite_gate_unique_frames(step) or 1))
    required_frames = max(required_frames, min(max(1, int(samples)), int(min_lite_unique_frames)))
    frames = latest_unique_smoothed_frames(world, required_frames)
    if len(frames) < int(required_frames):
        return None
    measurement = average_smoothed_frames(
        frames,
        step=step,
        process_rules=process_rules,
    )
    return pose_from_measurement(
        measurement,
        obs_ts=obs_ts,
        pose_source="lite_smoothed",
        lite_required_frames=required_frames,
    )


def brick_pose_from_world(world, *, obs_ts: float) -> dict | None:
    brick = getattr(world, "brick", None)
    if not isinstance(brick, dict):
        return None
    return pose_from_measurement(brick, obs_ts=obs_ts, pose_source="brick_state")


def aggregate_pose_samples(poses: list[dict]) -> dict | None:
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
        "lite_required_frames": coerce_int(poses[-1].get("lite_required_frames")),
        "samples_used": len(poses),
    }


def read_pose(
    vision,
    world,
    *,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None,
    min_samples_required: int | None,
    observe_sleep_s: float,
    fallback_step_label: str,
    update_world_from_vision: Callable,
    latest_unique_smoothed_frames: Callable,
    average_smoothed_frames: Callable,
    lite_gate_unique_frames: Callable,
    min_lite_unique_frames: int = 3,
) -> dict | None:
    poses = []
    start_t = time.time()
    step_label = world_step_label(world, fallback=fallback_step_label)
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
            time.sleep(float(observe_sleep_s))
            continue
        pose = None
        try:
            update_world_from_vision(world, vision, log=False)
            now = time.time()
            if min_sample_time is not None and now < float(min_sample_time):
                time.sleep(float(observe_sleep_s))
                continue
            pose = lite_pose_from_world(
                world,
                step=step_label,
                samples=int(samples),
                obs_ts=now,
                latest_unique_smoothed_frames=latest_unique_smoothed_frames,
                average_smoothed_frames=average_smoothed_frames,
                lite_gate_unique_frames=lite_gate_unique_frames,
                process_rules=getattr(world, "process_rules", None),
                min_lite_unique_frames=int(min_lite_unique_frames),
            )
            if pose is None:
                pose = brick_pose_from_world(world, obs_ts=now)
        except Exception:
            pose = None
        if pose is None:
            found, angle, dist, offset_x, conf, cam_h, _above, _below = vision.read()
            world.update_vision(found, dist, angle, conf, offset_x, cam_h)
            if not found:
                time.sleep(float(observe_sleep_s))
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
            time.sleep(float(observe_sleep_s))
    if len(poses) < int(required_samples):
        return None
    return aggregate_pose_samples(poses)


def observe_pose_with_reobserve(
    *,
    read_pose_fn: Callable,
    log_fn: Callable[[str], None],
    log_prefix: str,
    vision,
    world,
    samples: int,
    timeout_s: float,
    min_sample_time: float | None = None,
    hold_s: float = 0.12,
    reobserve_rounds: int = 2,
    relaxed_timeout_s: float = 2.8,
) -> tuple[dict | None, dict]:
    target_samples = max(1, int(samples))
    strict_pose = read_pose_fn(
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
        relaxed_pose = read_pose_fn(
            vision,
            world,
            samples=target_samples,
            timeout_s=max(float(timeout_s), float(relaxed_timeout_s)),
            min_sample_time=None,
            min_samples_required=1,
        )
        if relaxed_pose is None:
            if round_idx < rounds:
                log_fn(
                    f"{str(log_prefix)} Observation hold/reobserve {round_idx}/{rounds}: still no usable pose."
                )
            continue

        mode = (
            "hold_reobserve_full"
            if int(relaxed_pose.get("samples_used") or 0) >= int(target_samples)
            else "hold_reobserve_partial"
        )
        if int(relaxed_pose.get("samples_used") or 0) < int(target_samples):
            confirm_pose = read_pose_fn(
                vision,
                world,
                samples=target_samples,
                timeout_s=max(1.0, float(timeout_s)),
                min_sample_time=None,
                min_samples_required=1,
            )
            if confirm_pose is not None and int(confirm_pose.get("samples_used") or 0) >= int(
                relaxed_pose.get("samples_used") or 0
            ):
                relaxed_pose = confirm_pose
                mode = (
                    "hold_reobserve_confirmed_full"
                    if int(relaxed_pose.get("samples_used") or 0) >= int(target_samples)
                    else "hold_reobserve_confirmed_partial"
                )
        log_fn(
            f"{str(log_prefix)} Observation rescue: accepted {int(relaxed_pose.get('samples_used') or 0)}/{int(target_samples)} samples "
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


class CalibrationLivePlot:
    def __init__(
        self,
        *,
        show_plot: bool,
        plot_path: Path | None,
        cmds: Sequence[str],
        normalize_cmd: Callable[[str], str],
        plot_series_key: Callable[[str, str | None, str | None], str],
        plot_color: Callable[[str, str | None, str | None], str],
        plot_series_label: Callable[[str, str | None, str | None], str],
        plot_title: Callable[[list[float]], str],
        x_label: str = "Distance Covered (mm)",
        y_label: str = "Duration (ms)",
        title_font_size: int = 10,
        label_font_size: int = 9,
        tick_font_size: int = 8,
        legend_font_size: int = 8,
    ):
        self._enabled = bool(MATPLOTLIB_AVAILABLE)
        self._show_plot = bool(show_plot) and self._enabled
        self._plot_path = plot_path
        self._normalize_cmd = normalize_cmd
        self._plot_series_key = plot_series_key
        self._plot_title = plot_title
        self._scatter_by_series = {}
        self._xs_by_series = {}
        self._ys_by_series = {}
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._brick_distances_mm: list[float] = []
        self._fig = None
        self._ax = None

        for cmd in cmds:
            for kind in ("trial", "repeat"):
                key = self._plot_series_key(str(cmd), kind, None)
                self._xs_by_series[str(key)] = []
                self._ys_by_series[str(key)] = []
            key_fail = self._plot_series_key(str(cmd), "repeat", "fail")
            self._xs_by_series[str(key_fail)] = []
            self._ys_by_series[str(key_fail)] = []

        if not self._enabled:
            return
        if self._show_plot:
            plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(7, 4.5))
        for cmd in cmds:
            for kind in ("trial", "repeat"):
                series_key = self._plot_series_key(str(cmd), kind, None)
                self._scatter_by_series[str(series_key)] = self._ax.scatter(
                    [],
                    [],
                    color=plot_color(str(cmd), kind, None),
                    marker="o",
                    s=10,
                    alpha=0.92,
                    label=plot_series_label(str(cmd), kind, None),
                )
            series_key_fail = self._plot_series_key(str(cmd), "repeat", "fail")
            self._scatter_by_series[str(series_key_fail)] = self._ax.scatter(
                [],
                [],
                color=plot_color(str(cmd), "repeat", "fail"),
                marker="x",
                s=20,
                alpha=0.95,
                label=plot_series_label(str(cmd), "repeat", "fail"),
            )
        self._ax.set_title(self._plot_title(self._brick_distances_mm), fontsize=int(title_font_size))
        self._ax.set_xlabel(str(x_label), fontsize=int(label_font_size))
        self._ax.set_ylabel(str(y_label), fontsize=int(label_font_size))
        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(axis="both", labelsize=int(tick_font_size))
        self._ax.legend(loc="best", fontsize=int(legend_font_size))
        self._draw()

    def add_point(
        self,
        *,
        duration_ms: int,
        distance_mm: float,
        cmd: str,
        kind: str = "trial",
        pre_brick_distance_mm: float | None = None,
        post_brick_distance_mm: float | None = None,
        repeat_status: str | None = None,
        annotation_label: str | None = None,
        trial: int | None = None,
    ) -> None:
        del annotation_label, trial
        if not self._enabled:
            return
        cmd_key = self._normalize_cmd(cmd)
        series_key = self._plot_series_key(cmd_key, kind, repeat_status)
        self._xs_by_series.setdefault(str(series_key), []).append(float(distance_mm))
        self._ys_by_series.setdefault(str(series_key), []).append(float(duration_ms))
        self._xs.append(float(distance_mm))
        self._ys.append(float(duration_ms))
        for brick_distance_mm in (pre_brick_distance_mm, post_brick_distance_mm):
            brick_distance_value = coerce_finite_float(brick_distance_mm)
            if brick_distance_value is not None:
                self._brick_distances_mm.append(float(brick_distance_value))
        for scatter_key, scatter in self._scatter_by_series.items():
            scatter.set_offsets(
                plot_offsets(self._xs_by_series.get(scatter_key, []), self._ys_by_series.get(scatter_key, []))
            )
        self._ax.set_title(self._plot_title(self._brick_distances_mm))
        self._draw()

    def _draw(self) -> None:
        if not self._enabled:
            return
        if self._xs:
            min_x = min(self._xs)
            max_x = max(self._xs)
            min_y = min(self._ys)
            max_y = max(self._ys)
            pad_x = max(0.5, (max_x - min_x) * 0.20 if max_x != min_x else abs(max_x) * 0.20)
            pad_y = max(50.0, (max_y - min_y) * 0.15)
            self._ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self._ax.set_ylim(min_y - pad_y, max_y + pad_y)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        if self._plot_path is not None:
            path = Path(self._plot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(str(path), dpi=100)
        if self._show_plot:
            plt.pause(0.001)

    def finish(self) -> None:
        if not self._enabled:
            return
        if self._plot_path is not None:
            path = Path(self._plot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(str(path), dpi=100)
        if self._show_plot:
            plt.ioff()
            plt.show(block=True)
        else:
            plt.close(self._fig)


def check_1pct_speed_movement(
    *,
    robot,
    vision,
    world,
    cmd: str,
    movement_threshold_mm: float = 0.15,
    sample_frames: int = 3,
    sample_timeout_s: float = 1.5,
    observe_sleep_s: float = 0.02,
    control_sleep_s: float = 0.04,
) -> bool:
    """
    Preflight check: verify that 1% speed score produces detectable movement.
    
    Sends a single 1% speed pulse and verifies the target metric moves by at least
    the threshold. Returns True if movement is detected, False otherwise.
    
    Args:
        robot: Robot control instance
        vision: Vision system instance with read() method
        world: WorldModel instance
        cmd: Command letter (e.g., 'r' for right turn)
        movement_threshold_mm: Minimum movement to consider as success (default 0.15mm)
        sample_frames: Number of confirmation frames needed (default 3)
        sample_timeout_s: Timeout for collecting samples (default 1.5s)
        observe_sleep_s: Wait time between vision reads (default 0.02s)
        control_sleep_s: Wait time after sending motion (default 0.04s)
    
    Returns:
        True if 1% speed produces detectable movement, False otherwise
    """
    from telemetry_process import send_robot_command_pwm
    from telemetry_robot import StepState, speed_power_pwm_for_cmd
    
    # Determine which metric to measure based on command
    cmd_lower = str(cmd or "").strip().lower()
    if cmd_lower in ("l", "r"):
        metric = "x_axis"
    elif cmd_lower in ("u", "d"):
        metric = "cam_h"
    elif cmd_lower in ("f", "b"):
        metric = "dist"
    else:
        return False
    
    try:
        # Collect baseline samples (before motion)
        baseline_vals: list[float] = []
        time_start = time.time()
        while len(baseline_vals) < sample_frames and (time.time() - time_start) < sample_timeout_s:
            found, angle, dist, offset_x, conf, cam_h, above, below = vision.read()
            world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
            if found:
                if metric == "x_axis":
                    val = offset_x
                elif metric == "cam_h":
                    val = cam_h
                else:  # dist
                    val = dist
                if val is not None and isinstance(val, (int, float)):
                    baseline_vals.append(float(val))
            time.sleep(observe_sleep_s)
        
        if len(baseline_vals) < sample_frames:
            return False
        
        baseline = float(statistics.mean(baseline_vals))
        
        # Send 1% speed command
        power, pwm, score_used, duration_model_ms = speed_power_pwm_for_cmd(cmd_lower, 1)
        send_robot_command_pwm(
            robot,
            world,
            StepState.ALIGN_BRICK,
            cmd_lower,
            power,
            pwm,
            int(duration_model_ms),
            speed_score=score_used,
            auto_mode=False,
        )
        
        time.sleep(control_sleep_s)
        
        # Collect post-motion samples
        after_vals: list[float] = []
        time_start = time.time()
        while len(after_vals) < sample_frames and (time.time() - time_start) < sample_timeout_s:
            found, angle, dist, offset_x, conf, cam_h, above, below = vision.read()
            world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
            if found:
                if metric == "x_axis":
                    val = offset_x
                elif metric == "cam_h":
                    val = cam_h
                else:  # dist
                    val = dist
                if val is not None and isinstance(val, (int, float)):
                    after_vals.append(float(val))
            time.sleep(observe_sleep_s)
        
        if len(after_vals) < sample_frames:
            return False
        
        # Check if movement was detected
        deltas = [abs(float(v) - float(baseline)) for v in after_vals]
        moved_frames = int(sum(1 for d in deltas if float(d) >= float(movement_threshold_mm)))
        
        return bool(moved_frames >= sample_frames)
        
    except Exception as e:
        print(f"[PREFLIGHT] Exception during 1% speed check: {e}")
        return False
