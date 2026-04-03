#!/usr/bin/env python3
"""Shared calibration helpers for x/y/dist curve discovery scripts."""

from __future__ import annotations

import json
import math
import statistics
import sys
import threading
import time
from contextlib import contextmanager
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
ROBOT_MODEL_FILE = Path(__file__).resolve().parents[1] / "world_model_robot.json"
PREFLIGHT_DURATION_MS = 250
ADDITIONAL_PAUSE_MS = 250
CALIBRATION_DURATION_LIMIT_MS = 9999
PREFLIGHT_SCORE_CANDIDATES = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
)
_SHARED_STREAM_RUNTIME_LOCK = threading.Lock()
_SHARED_STREAM_RUNTIME = {
    "stream_state": None,
    "stream_url": None,
    "context": None,
}


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


def _normalize_trial_speed_score(score, default: int = 50) -> int:
    value = coerce_int(score, default)
    return max(1, min(100, int(value)))


def stdin_supports_interactive_input() -> bool:
    stream = getattr(sys, "stdin", None)
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def prompt_int_with_default(
    prompt: str,
    *,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
    log: Callable[[str], None] | None = None,
) -> int:
    emit = log if callable(log) else print
    default_value = int(default)
    min_value = int(minimum)
    max_value = None if maximum is None else int(maximum)
    while True:
        raw_value = input(f"{str(prompt).rstrip()} [{int(default_value)}]: ").strip()
        if not raw_value:
            return int(default_value)
        try:
            parsed = int(raw_value)
        except ValueError:
            limit_text = (
                f" between {int(min_value)} and {int(max_value)}"
                if max_value is not None
                else f" >= {int(min_value)}"
            )
            emit(f"Please enter a whole number{limit_text}.")
            continue
        if int(parsed) < int(min_value):
            limit_text = (
                f" between {int(min_value)} and {int(max_value)}"
                if max_value is not None
                else f" >= {int(min_value)}"
            )
            emit(f"Please enter a whole number{limit_text}.")
            continue
        if max_value is not None and int(parsed) > int(max_value):
            emit(f"Please enter a whole number between {int(min_value)} and {int(max_value)}.")
            continue
        return int(parsed)


def prompt_calibration_run_settings(
    *,
    prefix: str,
    observed_distance_mm: float | None,
    default_speed_score: int,
    default_min_duration_ms: int,
    default_max_duration_ms: int,
    duration_ceiling_ms: int,
    log: Callable[[str], None] | None = None,
) -> dict:
    speed_default = int(_normalize_trial_speed_score(default_speed_score))
    min_default = max(1, int(default_min_duration_ms))
    ceiling = max(1, int(duration_ceiling_ms))
    min_default = min(int(min_default), int(ceiling))
    max_default = max(int(min_default), int(default_max_duration_ms))
    max_default = min(int(max_default), int(ceiling))

    result = {
        "speed_score": int(speed_default),
        "min_duration_ms": int(min_default),
        "max_duration_ms": int(max_default),
        "prompted_speed_score": False,
        "prompted_duration_bounds": False,
        "prompted_any": False,
    }
    if not stdin_supports_interactive_input():
        return result

    emit = log if callable(log) else print
    observed_text = (
        f"{float(observed_distance_mm):.2f}mm"
        if coerce_finite_float(observed_distance_mm) is not None
        else "unknown"
    )
    emit(f"[{str(prefix)}] Observed dist before prompts: {observed_text}.")
    emit(f"[{str(prefix)}] Enter run settings for this calibration.")

    speed_score = prompt_int_with_default(
        "  Speed score %",
        default=int(speed_default),
        minimum=1,
        maximum=100,
        log=emit,
    )
    min_duration_ms = prompt_int_with_default(
        "  Min duration ms",
        default=int(min_default),
        minimum=1,
        maximum=int(ceiling),
        log=emit,
    )
    max_duration_ms = prompt_int_with_default(
        "  Max duration ms",
        default=int(max_default),
        minimum=int(min_duration_ms),
        maximum=int(ceiling),
        log=emit,
    )
    return {
        "speed_score": int(_normalize_trial_speed_score(speed_score)),
        "min_duration_ms": int(min_duration_ms),
        "max_duration_ms": int(max_duration_ms),
        "prompted_speed_score": True,
        "prompted_duration_bounds": True,
        "prompted_any": True,
    }


def _coerce_trial_speed_curve_points(raw_points) -> list[dict]:
    if not isinstance(raw_points, (list, tuple)):
        return []
    by_distance: dict[float, dict] = {}
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        distance_mm = coerce_finite_float(item.get("distance_mm"))
        score = coerce_int(item.get("speed_score"), None)
        if distance_mm is None or score is None:
            continue
        by_distance[float(distance_mm)] = {
            "distance_mm": float(distance_mm),
            "speed_score": int(_normalize_trial_speed_score(score)),
        }
    return [dict(by_distance[key]) for key in sorted(by_distance.keys())]


def load_calibration_trial_speed_profile(
    axis: str,
    *,
    path: Path | None = None,
) -> dict | None:
    profile_path = Path(path) if path is not None else ROBOT_MODEL_FILE
    if not profile_path.exists():
        return None
    try:
        payload = json.loads(profile_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    axis_key = str(axis or "").strip().lower()
    profiles = payload.get("calibration_trial_speed_profiles")
    if not isinstance(profiles, dict):
        return None
    profile = profiles.get(axis_key)
    if not isinstance(profile, dict):
        return None
    curve_points = _coerce_trial_speed_curve_points(profile.get("curve_points"))
    if not curve_points:
        return None
    metric = str(profile.get("metric") or "brick_distance_mm").strip().lower() or "brick_distance_mm"
    return {
        "axis": str(axis_key),
        "metric": str(metric),
        "curve_points": [dict(point) for point in curve_points],
        "note": str(profile.get("note") or ""),
    }


def resolve_calibration_trial_speed_score(
    *,
    observed_distance_mm: float | None,
    requested_score: int,
    speed_profile: dict | None,
) -> tuple[int, dict]:
    requested = int(_normalize_trial_speed_score(requested_score))
    observed = coerce_finite_float(observed_distance_mm)
    meta = {
        "source": "arg",
        "requested_score": int(requested),
        "score_used": int(requested),
        "observed_distance_mm": observed,
    }
    if not isinstance(speed_profile, dict):
        return int(requested), meta
    if str(speed_profile.get("metric") or "").strip().lower() != "brick_distance_mm":
        return int(requested), meta
    curve_points = _coerce_trial_speed_curve_points(speed_profile.get("curve_points"))
    if not curve_points or observed is None:
        return int(requested), meta
    if len(curve_points) == 1:
        score_used = int(curve_points[0]["speed_score"])
    elif float(observed) <= float(curve_points[0]["distance_mm"]):
        score_used = int(curve_points[0]["speed_score"])
    elif float(observed) >= float(curve_points[-1]["distance_mm"]):
        score_used = int(curve_points[-1]["speed_score"])
    else:
        score_used = int(requested)
        for idx in range(1, len(curve_points)):
            lower = curve_points[idx - 1]
            upper = curve_points[idx]
            lower_dist = float(lower["distance_mm"])
            upper_dist = float(upper["distance_mm"])
            if float(observed) > float(upper_dist):
                continue
            span = max(1e-9, float(upper_dist) - float(lower_dist))
            t = (float(observed) - float(lower_dist)) / float(span)
            interp_score = float(lower["speed_score"]) + (float(upper["speed_score"]) - float(lower["speed_score"])) * float(t)
            score_used = int(_normalize_trial_speed_score(interp_score, default=requested))
            break
    meta.update(
        {
            "source": "distance_curve",
            "score_used": int(score_used),
            "curve_points": [dict(point) for point in curve_points],
        }
    )
    return int(score_used), meta


def prediction_closeness_percentage(
    *,
    actual_distance_mm: float,
    predicted_distance_mm: float | None,
) -> float | None:
    if predicted_distance_mm is None:
        return None
    predicted_value = float(predicted_distance_mm)
    if predicted_value <= 0.0:
        return None
    actual_value = float(actual_distance_mm)
    percentage_off = (abs(actual_value - predicted_value) / predicted_value) * 100.0
    return max(0.0, 100.0 - float(percentage_off))


def set_shared_stream_runtime(
    *,
    stream_state: dict | None = None,
    stream_url: str | None = None,
    context: dict | None = None,
) -> None:
    url_text = str(stream_url).strip() if stream_url is not None else ""
    with _SHARED_STREAM_RUNTIME_LOCK:
        _SHARED_STREAM_RUNTIME["stream_state"] = stream_state if isinstance(stream_state, dict) else None
        _SHARED_STREAM_RUNTIME["stream_url"] = url_text or None
        _SHARED_STREAM_RUNTIME["context"] = dict(context) if isinstance(context, dict) else None


def get_shared_stream_runtime() -> tuple[dict | None, str | None]:
    with _SHARED_STREAM_RUNTIME_LOCK:
        stream_state = _SHARED_STREAM_RUNTIME.get("stream_state")
        stream_url = _SHARED_STREAM_RUNTIME.get("stream_url")
    return (
        stream_state if isinstance(stream_state, dict) else None,
        str(stream_url).strip() or None if stream_url is not None else None,
    )


def get_shared_calibration_context() -> dict | None:
    with _SHARED_STREAM_RUNTIME_LOCK:
        context = _SHARED_STREAM_RUNTIME.get("context")
    return dict(context) if isinstance(context, dict) else None


@contextmanager
def use_shared_stream_runtime(
    *,
    stream_state: dict | None = None,
    stream_url: str | None = None,
    context: dict | None = None,
):
    previous_state, previous_url = get_shared_stream_runtime()
    previous_context = get_shared_calibration_context()
    set_shared_stream_runtime(stream_state=stream_state, stream_url=stream_url, context=context)
    try:
        yield
    finally:
        set_shared_stream_runtime(stream_state=previous_state, stream_url=previous_url, context=previous_context)


def prepare_shared_stream_state(
    stream_state: dict | None,
    *,
    vision_mode: str,
) -> dict | None:
    if not isinstance(stream_state, dict):
        return None
    lock = stream_state.get("lock")
    if lock is None:
        lock = threading.Lock()
        stream_state["lock"] = lock
    with lock:
        if "frame" not in stream_state:
            stream_state["frame"] = None
        if not isinstance(stream_state.get("text_lines"), list):
            stream_state["text_lines"] = []
        if "xyz_workspace" not in stream_state:
            stream_state["xyz_workspace"] = None
        if "show_center_line" not in stream_state:
            stream_state["show_center_line"] = True
        stream_state["vision_mode"] = str(vision_mode).strip().lower() or "cyan"
    return stream_state


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


def pose_meets_multiframe_requirement(
    pose: dict | None,
    *,
    required_samples: int,
    required_lite_frames: int = 3,
) -> bool:
    if not isinstance(pose, dict):
        return False
    if str(pose.get("pose_source") or "").strip().lower() != "lite_smoothed":
        return False
    samples_used = coerce_int(pose.get("samples_used"), 0)
    lite_frames = coerce_int(pose.get("lite_required_frames"), 0)
    return bool(
        int(samples_used) >= int(max(1, int(required_samples)))
        and int(lite_frames) >= int(max(1, int(required_lite_frames)))
    )


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
    on_vision_update: Callable[[], None] | None = None,
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
            if callable(on_vision_update):
                on_vision_update()
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
        except Exception:
            pose = None
        if pose is None:
            time.sleep(float(observe_sleep_s))
            continue
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
    on_vision_update: Callable[[], None] | None = None,
) -> tuple[dict | None, dict]:
    target_samples = max(1, int(samples))
    required_samples = max(target_samples, 3)
    strict_pose = read_pose_fn(
        vision,
        world,
        samples=target_samples,
        timeout_s=float(timeout_s),
        min_sample_time=min_sample_time,
        min_samples_required=required_samples,
        on_vision_update=on_vision_update,
    )
    if pose_meets_multiframe_requirement(
        strict_pose,
        required_samples=required_samples,
    ):
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
            min_samples_required=required_samples,
            on_vision_update=on_vision_update,
        )
        if not pose_meets_multiframe_requirement(
            relaxed_pose,
            required_samples=required_samples,
        ):
            if round_idx < rounds:
                log_fn(
                    f"{str(log_prefix)} Observation hold/reobserve {round_idx}/{rounds}: still no usable pose."
                )
            continue

        mode = "hold_reobserve_full"
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
    score_candidates: Sequence[int] | None = None,
    duration_override_ms: int = PREFLIGHT_DURATION_MS,
    log: Callable[[str], None] | None = None,
) -> dict | None:
    """
    Preflight check: verify that a fixed-duration low-power pulse produces detectable
    movement, escalating score until the first successful act.

    The probe keeps duration fixed and dials power upward so the caller can learn the
    smallest score that produces visible motion without conflating power with time.
    
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
        score_candidates: Ordered speed-score candidates to try from weakest to strongest
        duration_override_ms: Fixed command duration for every preflight attempt
        log: Optional logger used for per-attempt operator messages

    Returns:
        Metadata for the first successful attempt, or None if no candidate moved enough
    """
    from telemetry_process import send_robot_command_pwm
    from telemetry_robot import StepState, normalize_speed_score, speed_power_pwm_for_cmd
    
    def _supports_ansi_color() -> bool:
        try:
            import sys
            return bool(sys.stdout.isatty())
        except Exception:
            return False
    
    def _colorize(text: str, color_code: str) -> str:
        if not _supports_ansi_color():
            return str(text)
        return f"{str(color_code)}{str(text)}\033[0m"
    
    # Determine which metric to measure based on command
    cmd_lower = str(cmd or "").strip().lower()
    if cmd_lower in ("l", "r"):
        metric = "x_axis"
    elif cmd_lower in ("u", "d"):
        metric = "cam_h"
    elif cmd_lower in ("f", "b"):
        metric = "dist"
    else:
        return None

    duration_ms = max(1, int(coerce_int(duration_override_ms, PREFLIGHT_DURATION_MS) or PREFLIGHT_DURATION_MS))
    candidates_raw = tuple(score_candidates or PREFLIGHT_SCORE_CANDIDATES)
    candidates: list[int] = []
    seen_scores: set[int] = set()
    for value in candidates_raw:
        score = int(normalize_speed_score(value))
        if score in seen_scores:
            continue
        seen_scores.add(score)
        candidates.append(score)
    if not candidates:
        candidates = [1]

    def _metric_value(*, dist, offset_x, cam_h):
        if metric == "x_axis":
            return offset_x
        if metric == "cam_h":
            return cam_h
        return dist

    log_fn = log if callable(log) else None
    
    try:
        for attempt_idx, score in enumerate(candidates, start=1):
            baseline_vals: list[float] = []
            time_start = time.time()
            while len(baseline_vals) < sample_frames and (time.time() - time_start) < sample_timeout_s:
                found, angle, dist, offset_x, conf, cam_h, above, below = vision.read()
                world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
                if found:
                    val = _metric_value(dist=dist, offset_x=offset_x, cam_h=cam_h)
                    if val is not None and isinstance(val, (int, float)):
                        baseline_vals.append(float(val))
                time.sleep(observe_sleep_s)

            if len(baseline_vals) < sample_frames:
                if log_fn is not None:
                    log_fn(
                        f"[PREFLIGHT] Attempt {int(attempt_idx)}/{int(len(candidates))}: insufficient baseline samples "
                        f"for {str(cmd_lower).upper()} at {int(score)}% ({len(baseline_vals)}/{int(sample_frames)})."
                    )
                continue

            baseline = float(statistics.mean(baseline_vals))
            power, pwm, score_used, _duration_model_ms = speed_power_pwm_for_cmd(cmd_lower, score)
            if log_fn is not None:
                log_fn(
                    f"[PREFLIGHT] Attempt {int(attempt_idx)}/{int(len(candidates))}: "
                    f"{str(cmd_lower).upper()} score={int(score_used)}% pwm={int(pwm)} duration={int(duration_ms)}ms."
                )
            send_robot_command_pwm(
                robot,
                world,
                StepState.ALIGN_BRICK,
                cmd_lower,
                power,
                pwm,
                int(duration_ms),
                speed_score=score_used,
                auto_mode=False,
            )

            # Add pause after command
            total_pause_s = control_sleep_s + (float(ADDITIONAL_PAUSE_MS) / 1000.0)
            if log_fn is not None:
                log_fn(f"\033[90m{int(total_pause_s * 1000)}ms pause\033[0m")
            time.sleep(total_pause_s)

            after_vals: list[float] = []
            time_start = time.time()
            while len(after_vals) < sample_frames and (time.time() - time_start) < sample_timeout_s:
                found, angle, dist, offset_x, conf, cam_h, above, below = vision.read()
                world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
                if found:
                    val = _metric_value(dist=dist, offset_x=offset_x, cam_h=cam_h)
                    if val is not None and isinstance(val, (int, float)):
                        after_vals.append(float(val))
                time.sleep(observe_sleep_s)

            if len(after_vals) < sample_frames:
                if log_fn is not None:
                    log_fn(
                        f"[PREFLIGHT] Attempt {int(attempt_idx)}/{int(len(candidates))}: insufficient post-act samples "
                        f"for {str(cmd_lower).upper()} at {int(score_used)}% ({len(after_vals)}/{int(sample_frames)})."
                    )
                continue

            deltas = [abs(float(v) - float(baseline)) for v in after_vals]
            moved_frames = int(sum(1 for d in deltas if float(d) >= float(movement_threshold_mm)))
            
            # Log result with colored prediction closeness.
            if log_fn is not None:
                step_mm = max(deltas) if deltas else 0.0
                predicted_mm = 1.98  # This should come from curve_prediction
                prediction_closeness = prediction_closeness_percentage(
                    actual_distance_mm=step_mm,
                    predicted_distance_mm=predicted_mm,
                )

                # Color the prediction closeness: green if >=85%, red otherwise.
                if prediction_closeness is not None and prediction_closeness >= 85.0:
                    closeness_color = "\033[92m"  # Green
                else:
                    closeness_color = "\033[91m"  # Red
                prediction_closeness_colored = (
                    f"{closeness_color}{float(prediction_closeness):.1f}%\033[0m"
                    if prediction_closeness is not None
                    else "n/a"
                )

                log_fn(f"[PREFLIGHT] prediction_closeness={prediction_closeness_colored}")
            
            if moved_frames >= sample_frames:
                return {
                    "cmd": str(cmd_lower),
                    "metric": str(metric),
                    "score_used": int(score_used),
                    "power": float(power),
                    "pwm": int(pwm),
                    "duration_ms": int(duration_ms),
                    "baseline": float(baseline),
                    "after_values": list(after_vals),
                    "deltas": list(deltas),
                    "moved_frames": int(moved_frames),
                    "sample_frames": int(sample_frames),
                    "attempt_idx": int(attempt_idx),
                    "attempt_count": int(len(candidates)),
                }

            if log_fn is not None:
                max_delta = max(deltas) if deltas else 0.0
                no_movement_msg = (
                    f"No detectable movement at {int(score_used)}% after {int(duration_ms)}ms "
                    f"(max delta {float(max_delta):.3f}mm; threshold {float(movement_threshold_mm):.3f}mm)."
                )
                red_color = '\033[91m'
                log_fn(
                    f"[PREFLIGHT] {_colorize(no_movement_msg, red_color)}"
                )

        return None

    except Exception as e:
        print(f"[PREFLIGHT] Exception during 1% speed check: {e}")
        return None
