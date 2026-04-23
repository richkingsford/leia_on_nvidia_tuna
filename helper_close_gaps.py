#!/usr/bin/env python3
"""
helper_close_gaps.py
--------------------
Lean gap-closing helper. Provides monotonic error->intensity utilities,
axis calibration lookups, and shared x/y/dist micro-adjust scoring helpers.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from helper_demo_log_utils import normalize_step_label
from helper_gate_utils import metric_error
from telemetry_robot import SPEED_SCORE_MAX, SPEED_SCORE_MIN

DEFAULT_CURVE_FILE = Path(__file__).resolve().parent / "world_model_left_right_curve.json"
DEFAULT_Y_CURVE_FILE = Path(__file__).resolve().parent / "world_model_up_down_curve.json"
DEFAULT_DIST_CURVE_FILE = Path(__file__).resolve().parent / "world_model_forward_backward_curve.json"
DEFAULT_TURN_DRIVE_TRIALS_DIR = Path(__file__).resolve().parent / "trials"
DEFAULT_CURVE_BINS_MM = (
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    22.0,
    30.0,
)
DEFAULT_MIN_INTENSITY_PCT = 1.0
DEFAULT_MAX_INTENSITY_PCT = 100.0
DEFAULT_MIN_IMPROVEMENT_MM = 0.05
_JSON_CACHE: Dict[str, Tuple[int, int, Optional[dict]]] = {}


def _coerce_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _float_or_none(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json_payload(path: Path) -> Optional[dict]:
    path = Path(path)
    cache_key = str(path.resolve())
    try:
        stat = path.stat()
        stamp = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        return None
    cached = _JSON_CACHE.get(cache_key)
    if cached is not None and cached[:2] == stamp:
        return cached[2]
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        payload = None
    _JSON_CACHE[cache_key] = (stamp[0], stamp[1], payload)
    return payload


def _sorted_bins(bins: Iterable[float]) -> List[float]:
    cleaned = []
    for item in bins or []:
        try:
            val = float(item)
        except (TypeError, ValueError):
            continue
        if val <= 0.0:
            continue
        cleaned.append(val)
    cleaned = sorted(set(cleaned))
    return cleaned if cleaned else list(DEFAULT_CURVE_BINS_MM)


def _default_curve_values(bins_mm: Iterable[float], min_intensity: float, max_intensity: float) -> List[float]:
    bins = _sorted_bins(bins_mm)
    max_err = float(bins[-1]) if bins else 1.0
    out = []
    for idx in range(len(bins) + 1):
        edge = bins[idx] if idx < len(bins) else max_err
        ratio = 1.0 if max_err <= 0 else max(0.0, min(1.0, float(edge) / float(max_err)))
        out.append(float(min_intensity) + (float(max_intensity) - float(min_intensity)) * ratio)
    return out


def _curve_cmd_key(cmd: str) -> str:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key in ("l", "u"):
        return "l"
    return "r"


def axis_cmd_for_error(axis: str, err_mm: float) -> Optional[str]:
    axis_key = str(axis or "x").strip().lower()
    err = _float_or_none(err_mm)
    if err is None or abs(float(err)) <= 1e-9:
        return None
    if axis_key in {"dist", "distance"}:
        return "f" if float(err) > 0.0 else "b"
    if axis_key == "y":
        return "d" if float(err) > 0.0 else "u"
    # Traditional number-line x semantics: negative is left, positive is right.
    # If current x is greater than target (positive error), turn left to reduce x.
    # If current x is less than target (negative error), turn right to increase x.
    return "l" if float(err) > 0.0 else "r"


def _axis_curve_file(axis: str) -> Path:
    axis_key = str(axis or "x").strip().lower()
    if axis_key in {"dist", "distance"}:
        return DEFAULT_DIST_CURVE_FILE
    return DEFAULT_Y_CURVE_FILE if axis_key == "y" else DEFAULT_CURVE_FILE


def _axis_calibration_cmd_key(axis: str, cmd: str) -> str:
    axis_key = str(axis or "x").strip().lower()
    cmd_key = str(cmd or "").strip().lower()
    if axis_key in {"dist", "distance"}:
        return "f" if cmd_key == "f" else "b"
    if axis_key == "y":
        return "u" if cmd_key == "u" else "d"
    return "l" if cmd_key == "l" else "r"


def _duration_range_ms(raw_range) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
        return None, None
    try:
        min_ms = int(round(float(raw_range[0])))
        max_ms = int(round(float(raw_range[1])))
    except (TypeError, ValueError):
        return None, None
    if min_ms <= 0 or max_ms <= 0:
        return None, None
    if min_ms > max_ms:
        min_ms, max_ms = max_ms, min_ms
    return int(min_ms), int(max_ms)


def _normalize_values(values, n: int, fallback: List[float]) -> List[float]:
    if not isinstance(values, (list, tuple)):
        values = []
    out = []
    for item in values:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    if not out:
        out = list(fallback)
    if len(out) < n:
        out.extend([out[-1]] * (n - len(out)))
    if len(out) > n:
        out = out[:n]
    return out


def normalize_curve(curve: Optional[dict]) -> dict:
    if not isinstance(curve, dict):
        curve = {}
    source = curve.get("curve") if isinstance(curve.get("curve"), dict) else curve
    bins = _sorted_bins(source.get("bins_mm") or DEFAULT_CURVE_BINS_MM)
    min_intensity = _coerce_float(source.get("min_intensity_pct"), DEFAULT_MIN_INTENSITY_PCT)
    max_intensity = _coerce_float(source.get("max_intensity_pct"), DEFAULT_MAX_INTENSITY_PCT)
    min_intensity = _clamp(min_intensity, DEFAULT_MIN_INTENSITY_PCT, DEFAULT_MAX_INTENSITY_PCT)
    max_intensity = _clamp(max_intensity, min_intensity, DEFAULT_MAX_INTENSITY_PCT)
    default_vals = _default_curve_values(bins, min_intensity, max_intensity)

    by_cmd = source.get("by_cmd") if isinstance(source.get("by_cmd"), dict) else {}
    raw_l = by_cmd.get("l")
    raw_r = by_cmd.get("r")
    if raw_l is None:
        raw_l = by_cmd.get("u")
    if raw_r is None:
        raw_r = by_cmd.get("d")
    if raw_l is None:
        raw_l = source.get("turn_intensity_pct") or source.get("intensity_pct")
    if raw_r is None:
        raw_r = raw_l

    values_l = _normalize_values(raw_l, len(bins) + 1, default_vals)
    values_r = _normalize_values(raw_r, len(bins) + 1, default_vals)

    values_l = [_clamp(v, min_intensity, max_intensity) for v in values_l]
    values_r = [_clamp(v, min_intensity, max_intensity) for v in values_r]

    return {
        "model": "monotonic_bins",
        "bins_mm": bins,
        "min_intensity_pct": float(min_intensity),
        "max_intensity_pct": float(max_intensity),
        "by_cmd": {
            "l": values_l,
            "r": values_r,
        },
    }


def load_left_right_curve(path: Path = DEFAULT_CURVE_FILE) -> Optional[dict]:
    return _load_json_payload(Path(path))


def load_up_down_curve(path: Path = DEFAULT_Y_CURVE_FILE) -> Optional[dict]:
    return _load_json_payload(Path(path))


def load_axis_curve(axis: str, path: Optional[Path] = None) -> Optional[dict]:
    target = Path(path) if path is not None else _axis_curve_file(axis)
    return _load_json_payload(target)


def load_axis_aruco_calibration(axis: str, path: Optional[Path] = None) -> Optional[dict]:
    payload = load_axis_curve(axis, path=path)
    calibration = payload.get("aruco_marker_calibration") if isinstance(payload, dict) else None
    return calibration if isinstance(calibration, dict) else None


def calibrated_axis_motion_profile(
    *,
    axis: str,
    cmd: str,
    gap_mm: float,
    path: Optional[Path] = None,
) -> Optional[dict]:
    axis_key = str(axis or "x").strip().lower()
    cmd_key = _axis_calibration_cmd_key(axis_key, cmd)
    try:
        gap_abs_mm = abs(float(gap_mm))
    except (TypeError, ValueError):
        return None
    if gap_abs_mm <= 0.0:
        return None

    calibration = load_axis_aruco_calibration(axis_key, path=path)
    if not isinstance(calibration, dict):
        return None
    by_cmd = calibration.get("by_cmd")
    if not isinstance(by_cmd, dict):
        return None
    cmd_profile = by_cmd.get(cmd_key)
    if not isinstance(cmd_profile, dict):
        return None

    slope = _float_or_none(cmd_profile.get("slope_mm_per_ms"))
    intercept = _float_or_none(cmd_profile.get("intercept_mm"))
    speed_score = _float_or_none(calibration.get("speed_score_pct"))
    min_duration_ms, max_duration_ms = _duration_range_ms(calibration.get("duration_range_ms"))
    if (
        slope is None
        or float(slope) <= 0.0
        or intercept is None
        or speed_score is None
        or min_duration_ms is None
        or max_duration_ms is None
    ):
        return None

    raw_duration_ms = (float(gap_abs_mm) - float(intercept)) / float(slope)
    clamped_to = None
    duration_bounded_ms = float(raw_duration_ms)
    if raw_duration_ms < float(min_duration_ms):
        clamped_to = "min"
        duration_bounded_ms = float(min_duration_ms)
    elif raw_duration_ms > float(max_duration_ms):
        clamped_to = "max"
        duration_bounded_ms = float(max_duration_ms)
    duration_override_ms = int(round(float(duration_bounded_ms)))
    predicted_distance_mm = float(intercept) + float(slope) * float(duration_override_ms)

    return {
        "axis": axis_key,
        "cmd": str(cmd_key),
        "gap_mm": float(gap_abs_mm),
        "score": int(round(float(speed_score))),
        "speed_score_pct": float(speed_score),
        "duration_override_ms": int(duration_override_ms),
        "duration_range_ms": [int(min_duration_ms), int(max_duration_ms)],
        "raw_duration_ms": float(raw_duration_ms),
        "duration_clamped_to": clamped_to,
        "predicted_distance_mm": float(predicted_distance_mm),
        "equation": cmd_profile.get("equation"),
        "reference_distance_mm": _float_or_none(calibration.get("reference_distance_mm")),
        "motion_rate_mm_per_sec": _float_or_none(cmd_profile.get("motion_rate_mm_per_sec")),
        "residual_std_mm": _float_or_none(cmd_profile.get("residual_std_mm")),
        "eighty_five_percent_residual_band_mm": _float_or_none(
            cmd_profile.get("eighty_five_percent_residual_band_mm")
        ),
        "r_squared": _float_or_none(cmd_profile.get("r_squared")),
        "source": "aruco_marker_calibration",
    }


def calibrated_axis_motion_for_error(
    *,
    axis: str,
    err_mm: float,
    path: Optional[Path] = None,
) -> Optional[dict]:
    cmd = axis_cmd_for_error(axis, err_mm)
    if cmd is None:
        return None
    profile = calibrated_axis_motion_profile(
        axis=axis,
        cmd=cmd,
        gap_mm=abs(float(err_mm)),
        path=path,
    )
    if not isinstance(profile, dict):
        return None
    out = dict(profile)
    out["err_mm"] = float(err_mm)
    return out


def _turn_drive_trial_manifest_paths(trials_dir: Optional[Path] = None) -> List[Path]:
    root = Path(trials_dir) if trials_dir is not None else DEFAULT_TURN_DRIVE_TRIALS_DIR
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file()
            and not str(path.name).startswith(".")
            and str(path.suffix).lower() == ".json"
        ],
        key=lambda path: str(path.name).lower(),
    )


def _turn_drive_profile_override_from_manifest(
    *,
    measured_phase: dict,
    cmd: str,
    drive_mode: str,
) -> Optional[tuple[int, dict]]:
    if not isinstance(measured_phase, dict):
        return None
    motor_pair = measured_phase.get("motor_pair") if isinstance(measured_phase.get("motor_pair"), dict) else {}
    left_pwm = _float_or_none(motor_pair.get("left_motor_pwm"))
    right_pwm = _float_or_none(motor_pair.get("right_motor_pwm"))
    if left_pwm is None or right_pwm is None:
        return None
    cmd_key = str(cmd or "").strip().lower()
    drive_mode_key = str(drive_mode or "").strip().lower()
    if cmd_key not in {"l", "r"} or drive_mode_key not in {"forward", "backward"}:
        return None

    pwm_override = _float_or_none(measured_phase.get("pwm_override"))
    base_pwm = int(round(float(pwm_override))) if pwm_override is not None and float(pwm_override) > 0.0 else int(round(max(float(left_pwm), float(right_pwm))))
    if base_pwm <= 0:
        return None

    if cmd_key == "l":
        inner_pwm = float(left_pwm)
        outer_pwm = float(right_pwm)
    else:
        outer_pwm = float(left_pwm)
        inner_pwm = float(right_pwm)

    profile_override = {
        "profile_name": str(measured_phase.get("profile_name") or "").strip() or None,
        "drive_mode": str(drive_mode_key),
        "inner_ratio": max(0.0, min(1.0, float(inner_pwm) / float(base_pwm))),
        "outer_ratio": max(0.0, min(1.0, float(outer_pwm) / float(base_pwm))),
        "duration_mode": "turn",
        "action_note": str(measured_phase.get("action_note") or "").strip() or (
            "TURN+FWD" if drive_mode_key == "forward" else "TURN+BWD"
        ),
    }
    return int(base_pwm), profile_override


def _production_turn_drive_setup_phase_plan(
    cmd: str,
    trials_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Return motor settings from the setup_phase (forward turn) of a trial file."""
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key not in {"l", "r"}:
        return None
    for path in _turn_drive_trial_manifest_paths(trials_dir):
        payload = _load_json_payload(path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("file_type") or "").strip().lower() != "turn_drive_trials":
            continue
        if not bool(payload.get("production")):
            continue
        curve_stats = payload.get("curve_stats") if isinstance(payload.get("curve_stats"), dict) else {}
        if not bool(curve_stats.get("production_worthy", False)):
            continue
        curve_cfg = payload.get("curve") if isinstance(payload.get("curve"), dict) else {}
        setup_phase = curve_cfg.get("setup_phase") if isinstance(curve_cfg.get("setup_phase"), dict) else {}
        if str(setup_phase.get("cmd") or "").strip().lower() != cmd_key:
            continue
        if str(setup_phase.get("drive_mode") or "").strip().lower() != "forward":
            continue
        profile_bits = _turn_drive_profile_override_from_manifest(
            measured_phase=setup_phase,
            cmd=cmd_key,
            drive_mode="forward",
        )
        if profile_bits is None:
            continue
        pwm_override, profile_override = profile_bits
        score_val = _float_or_none(setup_phase.get("score_pct"))
        manifest_name = str(payload.get("name") or path.stem).strip() or path.stem
        return {
            "manifest_name": str(manifest_name),
            "trial": None,
            "score": int(round(float(score_val))) if score_val is not None else 1,
            "duration_override_ms": None,
            "pwm_override": None,
            "profile_override": dict(profile_override),
            "curve_name": f"{manifest_name} forward-setup",
            "curve_value_mm": None,
            "source": "production_turn_drive_trials_forward",
        }
    return None


def production_turn_drive_curve_plan(
    *,
    cmd: str,
    drive_mode: str,
    current_dist_mm: float,
    x_err_mm: float,
    trials_dir: Optional[Path] = None,
) -> Optional[dict]:
    cmd_key = str(cmd or "").strip().lower()
    drive_mode_key = str(drive_mode or "").strip().lower()
    if cmd_key not in {"l", "r"} or drive_mode_key not in {"forward", "backward"}:
        return None

    if drive_mode_key == "forward":
        return _production_turn_drive_setup_phase_plan(cmd=cmd_key, trials_dir=trials_dir)

    current_dist = _float_or_none(current_dist_mm)
    x_gap_needed = _float_or_none(x_err_mm)
    if x_gap_needed is None:
        return None
    x_gap_needed = abs(float(x_gap_needed))
    if x_gap_needed <= 0.0:
        return None

    candidates = []
    for path in _turn_drive_trial_manifest_paths(trials_dir):
        payload = _load_json_payload(path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("file_type") or "").strip().lower() != "turn_drive_trials":
            continue
        if not bool(payload.get("production")):
            continue
        curve_stats = payload.get("curve_stats") if isinstance(payload.get("curve_stats"), dict) else {}
        if not bool(curve_stats.get("production_worthy", False)):
            continue
        curve_cfg = payload.get("curve") if isinstance(payload.get("curve"), dict) else {}
        measured_phase = curve_cfg.get("measured_phase") if isinstance(curve_cfg.get("measured_phase"), dict) else {}
        if str(measured_phase.get("cmd") or "").strip().lower() != cmd_key:
            continue
        if str(measured_phase.get("drive_mode") or "").strip().lower() != drive_mode_key:
            continue
        profile_bits = _turn_drive_profile_override_from_manifest(
            measured_phase=measured_phase,
            cmd=cmd_key,
            drive_mode=drive_mode_key,
        )
        if profile_bits is None:
            continue
        pwm_override, profile_override = profile_bits
        score_val = _float_or_none(measured_phase.get("score_pct"))
        manifest_name = str(payload.get("name") or path.stem).strip() or path.stem
        for row in list(payload.get("trials_backwards") or []):
            if not isinstance(row, dict):
                continue
            if row.get("usable") is not True:
                continue
            duration_ms = _float_or_none(row.get("measuredDurationMs"))
            start_dist = _float_or_none(row.get("startDist"))
            x_gap_closed = _float_or_none(row.get("xGapClosed"))
            if duration_ms is None or start_dist is None or x_gap_closed is None:
                continue
            if float(duration_ms) <= 0.0 or float(x_gap_closed) <= 0.0:
                continue
            candidates.append(
                {
                    "manifest_name": str(manifest_name),
                    "trial": int(round(float(row.get("trial") or 0))) if _float_or_none(row.get("trial")) is not None else None,
                    "score": int(round(float(score_val))) if score_val is not None else 1,
                    "duration_override_ms": int(round(float(duration_ms))),
                    "start_dist_mm": float(start_dist),
                    "x_gap_closed_mm": float(x_gap_closed),
                    "pwm_override": int(pwm_override),
                    "profile_override": dict(profile_override),
                }
            )

    if not candidates:
        return None

    def _candidate_key(item: dict) -> tuple[float, float, float, float, float]:
        dist_delta = (
            abs(float(item.get("start_dist_mm") or 0.0) - float(current_dist))
            if current_dist is not None
            else 0.0
        )
        gap_closed = float(item.get("x_gap_closed_mm") or 0.0)
        covers_gap = gap_closed >= float(x_gap_needed)
        if covers_gap:
            return (
                0.0,
                float(dist_delta),
                float(gap_closed - float(x_gap_needed)),
                float(item.get("duration_override_ms") or 0.0),
                0.0,
            )
        return (
            1.0,
            float(dist_delta),
            abs(float(x_gap_needed) - float(gap_closed)),
            -float(gap_closed),
            float(item.get("duration_override_ms") or 0.0),
        )

    chosen = min(candidates, key=_candidate_key)
    curve_name = (
        f"{str(chosen.get('manifest_name') or '')} trial {int(chosen.get('trial') or 0)} "
        f"(start_dist={float(chosen.get('start_dist_mm') or 0.0):.3f}mm, "
        f"x_gap_closed={float(chosen.get('x_gap_closed_mm') or 0.0):.3f}mm)"
    ).strip()
    out = dict(chosen)
    out["curve_name"] = str(curve_name)
    out["curve_value_mm"] = float(chosen.get("x_gap_closed_mm") or 0.0)
    out["source"] = "production_turn_drive_trials"
    return out


def save_left_right_curve(curve: dict, path: Path = DEFAULT_CURVE_FILE) -> Optional[Path]:
    if not isinstance(curve, dict) or not curve:
        return None
    path = Path(path)
    path.write_text(json.dumps(curve, indent=2) + "\n")
    return path


def bin_index(abs_err_mm: float, bins_mm: Iterable[float]) -> int:
    try:
        v = abs(float(abs_err_mm))
    except (TypeError, ValueError):
        v = 0.0
    bins = list(bins_mm)
    for idx, edge in enumerate(bins):
        if v <= float(edge):
            return idx
    return len(bins)


def curve_intensity_for_error(curve: Optional[dict], cmd: str, abs_err_mm: float) -> float:
    cfg = normalize_curve(curve)
    bins = cfg["bins_mm"]
    idx = bin_index(abs_err_mm, bins)
    cmd_key = _curve_cmd_key(cmd)
    values = cfg["by_cmd"].get(cmd_key) or cfg["by_cmd"].get("l")
    if not isinstance(values, (list, tuple)):
        values = _default_curve_values(bins, cfg["min_intensity_pct"], cfg["max_intensity_pct"])
    if idx >= len(values):
        return float(values[-1])
    return float(values[idx])


def x_axis_gate_for_step(process_rules: dict, step: str) -> dict:
    step_key = normalize_step_label(step) or str(step)
    cfg = (process_rules or {}).get(step_key, {}) if isinstance(process_rules, dict) else {}
    success_gates = cfg.get("success_gates") if isinstance(cfg, dict) else {}
    if not isinstance(success_gates, dict):
        success_gates = {}
    x_stats = success_gates.get("xAxis_offset_abs")
    if not isinstance(x_stats, dict):
        x_stats = success_gates.get("x_axis")
    if not isinstance(x_stats, dict):
        x_stats = success_gates.get("xAxis_offset")
    if not isinstance(x_stats, dict):
        x_stats = {}
    return dict(x_stats)


def y_axis_gate_for_step(process_rules: dict, step: str) -> dict:
    step_key = normalize_step_label(step) or str(step)
    cfg = (process_rules or {}).get(step_key, {}) if isinstance(process_rules, dict) else {}
    success_gates = cfg.get("success_gates") if isinstance(cfg, dict) else {}
    if not isinstance(success_gates, dict):
        success_gates = {}
    y_stats = success_gates.get("yAxis_offset_abs")
    if not isinstance(y_stats, dict):
        y_stats = success_gates.get("y_axis")
    if not isinstance(y_stats, dict):
        y_stats = success_gates.get("yAxis_offset")
    if not isinstance(y_stats, dict):
        y_stats = {}
    return dict(y_stats)


def axis_gate_for_step(process_rules: dict, step: str, axis: str) -> dict:
    axis_key = str(axis or "x").strip().lower()
    if axis_key == "y":
        return y_axis_gate_for_step(process_rules, step)
    return x_axis_gate_for_step(process_rules, step)


def x_axis_error_mm(x_axis_mm: float, x_target_mm: float, *, x_axis_sign: float = 1.0) -> float:
    x_val = _coerce_float(x_axis_mm, 0.0)
    return (float(x_val) * float(x_axis_sign)) - float(x_target_mm)


def axis_error_mm(axis_mm: float, axis_target_mm: float, *, axis_sign: float = 1.0) -> float:
    axis_val = _coerce_float(axis_mm, 0.0)
    return (float(axis_val) * float(axis_sign)) - float(axis_target_mm)


def x_gate_outside_mm(x_axis_mm: float, x_stats: dict) -> Optional[float]:
    if not isinstance(x_stats, dict) or not x_stats:
        return None
    try:
        value = float(x_axis_mm)
    except (TypeError, ValueError):
        return None
    err = metric_error(value, x_stats)
    if err is None:
        return None
    return max(0.0, float(err))


def axis_gate_outside_mm(axis_mm: float, axis_stats: dict) -> Optional[float]:
    if not isinstance(axis_stats, dict) or not axis_stats:
        return None
    try:
        value = float(axis_mm)
    except (TypeError, ValueError):
        return None
    err = metric_error(value, axis_stats)
    if err is None:
        return None
    return max(0.0, float(err))


def select_axis_act(
    *,
    process_rules: dict,
    step: str,
    axis: str,
    axis_mm: float,
    visible: bool,
    axis_sign: float = 1.0,
    curve: Optional[dict] = None,
) -> dict:
    axis_key = str(axis or "x").strip().lower()
    axis_stats = axis_gate_for_step(process_rules, step, axis_key)
    axis_target = _coerce_float(axis_stats.get("target"), 0.0)
    axis_tol = abs(_coerce_float(axis_stats.get("tol"), 0.0) or 0.0)

    if not visible:
        return {
            "cmd": None,
            "reason": "not_visible",
            "err_mm": None,
            "abs_err_mm": None,
            "target_mm": axis_target,
            "tol_mm": axis_tol,
            "bin_idx": None,
            "turn_intensity_pct": None,
        }

    err = axis_error_mm(axis_mm, axis_target, axis_sign=axis_sign)
    abs_err = abs(float(err))
    outside = axis_gate_outside_mm(axis_mm, axis_stats)
    if outside is None:
        within_gate = axis_tol > 0.0 and abs_err <= axis_tol
    else:
        within_gate = float(outside) <= 0.0
    if within_gate:
        return {
            "cmd": None,
            "reason": "within_gate",
            "err_mm": float(err),
            "abs_err_mm": float(abs_err),
            "target_mm": axis_target,
            "tol_mm": axis_tol,
            "bin_idx": bin_index(abs_err, normalize_curve(curve)["bins_mm"]),
            "turn_intensity_pct": None,
        }

    cmd = axis_cmd_for_error(axis_key, err)
    if cmd is None:
        cmd = "u" if axis_key == "y" else "l"
    intensity = curve_intensity_for_error(curve, cmd, abs_err)
    return {
        "cmd": cmd,
        "reason": "y_axis_alignment" if axis_key == "y" else "x_axis_alignment",
        "err_mm": float(err),
        "abs_err_mm": float(abs_err),
        "target_mm": axis_target,
        "tol_mm": axis_tol,
        "bin_idx": bin_index(abs_err, normalize_curve(curve)["bins_mm"]),
        "turn_intensity_pct": float(intensity),
    }


def select_left_right_act(
    *,
    process_rules: dict,
    step: str,
    x_axis_mm: float,
    visible: bool,
    x_axis_sign: float = 1.0,
    curve: Optional[dict] = None,
) -> dict:
    return select_axis_act(
        process_rules=process_rules,
        step=step,
        axis="x",
        axis_mm=x_axis_mm,
        visible=visible,
        axis_sign=x_axis_sign,
        curve=curve,
    )


def estimate_required_intensity(
    *,
    pre_abs_err_mm: float,
    improvement_mm: float,
    intensity_pct: float,
    min_improvement_mm: float = DEFAULT_MIN_IMPROVEMENT_MM,
    min_intensity_pct: float = DEFAULT_MIN_INTENSITY_PCT,
    max_intensity_pct: float = DEFAULT_MAX_INTENSITY_PCT,
) -> Optional[float]:
    try:
        improve = float(improvement_mm)
    except (TypeError, ValueError):
        return None
    if improve <= float(min_improvement_mm):
        return None
    try:
        pre_abs = abs(float(pre_abs_err_mm))
    except (TypeError, ValueError):
        return None
    if pre_abs <= 0.0:
        return None
    try:
        intensity = float(intensity_pct)
    except (TypeError, ValueError):
        return None
    scale = float(pre_abs) / float(improve)
    required = float(intensity) * float(scale)
    return float(_clamp(required, min_intensity_pct, max_intensity_pct))


def _fill_missing(values: List[Optional[float]], fallback: float) -> List[float]:
    out = list(values)
    last = None
    for idx in range(len(out)):
        if out[idx] is None:
            out[idx] = last
        else:
            last = out[idx]
    if out and out[0] is None:
        next_val = None
        for val in out:
            if val is not None:
                next_val = val
                break
        if next_val is None:
            next_val = float(fallback)
        out = [next_val if v is None else v for v in out]
    return [float(fallback) if v is None else float(v) for v in out]


def _isotonic_non_decreasing(values: List[float], weights: Optional[List[float]] = None) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0] * n
    blocks = []
    for idx, (val, w) in enumerate(zip(values, weights)):
        blocks.append([float(val) * float(w), float(w), idx, idx])
        while len(blocks) >= 2:
            sum_yw_prev, sum_w_prev, start_prev, end_prev = blocks[-2]
            sum_yw_curr, sum_w_curr, start_curr, end_curr = blocks[-1]
            avg_prev = sum_yw_prev / sum_w_prev if sum_w_prev else 0.0
            avg_curr = sum_yw_curr / sum_w_curr if sum_w_curr else 0.0
            if avg_prev <= avg_curr:
                break
            blocks.pop()
            blocks.pop()
            blocks.append([
                sum_yw_prev + sum_yw_curr,
                sum_w_prev + sum_w_curr,
                start_prev,
                end_curr,
            ])
    out = [0.0] * n
    for sum_yw, sum_w, start, end in blocks:
        avg = sum_yw / sum_w if sum_w else 0.0
        for idx in range(start, end + 1):
            out[idx] = avg
    return out


def fit_monotonic_curve(
    samples: List[dict],
    bins_mm: Iterable[float],
    min_intensity_pct: float,
    max_intensity_pct: float,
) -> Optional[dict]:
    bins = _sorted_bins(bins_mm)
    if not samples:
        return None
    num_bins = len(bins) + 1
    out_by_cmd = {}
    for cmd in ("l", "r"):
        buckets: List[List[float]] = [[] for _ in range(num_bins)]
        for row in samples:
            if _curve_cmd_key(row.get("cmd")) != cmd:
                continue
            try:
                abs_err = abs(float(row.get("abs_err_mm")))
                req = float(row.get("required_intensity"))
            except (TypeError, ValueError):
                continue
            idx = bin_index(abs_err, bins)
            buckets[idx].append(req)
        medians: List[Optional[float]] = []
        for bucket in buckets:
            if bucket:
                medians.append(float(statistics.median(bucket)))
            else:
                medians.append(None)
        if all(v is None for v in medians):
            continue
        filled = _fill_missing(medians, min_intensity_pct)
        iso = _isotonic_non_decreasing(filled)
        iso = [_clamp(v, min_intensity_pct, max_intensity_pct) for v in iso]
        out_by_cmd[cmd] = iso

    if not out_by_cmd:
        return None

    # If one side missing, mirror the other.
    if "l" not in out_by_cmd and "r" in out_by_cmd:
        out_by_cmd["l"] = list(out_by_cmd["r"])
    if "r" not in out_by_cmd and "l" in out_by_cmd:
        out_by_cmd["r"] = list(out_by_cmd["l"])

    return {
        "model": "monotonic_bins",
        "bins_mm": bins,
        "min_intensity_pct": float(min_intensity_pct),
        "max_intensity_pct": float(max_intensity_pct),
        "by_cmd": out_by_cmd,
    }


def summarize_trials(trials: List[dict], *, x_tol_mm: Optional[float] = None) -> dict:
    if not trials:
        return {}
    success_vals = [1.0 if bool(t.get("success")) else 0.0 for t in trials]
    overshoot_vals = [1.0 if bool(t.get("overshoot")) else 0.0 for t in trials]
    abs_err_after = [
        abs(float(t.get("abs_err_after_mm")))
        for t in trials
        if t.get("abs_err_after_mm") is not None
    ]
    improvements = [
        float(t.get("improvement_mm"))
        for t in trials
        if t.get("improvement_mm") is not None
    ]
    negative_improvements = [1.0 for val in improvements if float(val) < 0.0]
    durations = [
        float(t.get("duration_s"))
        for t in trials
        if t.get("duration_s") is not None
    ]

    stats = {
        "trials": int(len(trials)),
        "success_rate": float(sum(success_vals) / len(success_vals)) if success_vals else 0.0,
        "overshoot_rate": float(sum(overshoot_vals) / len(overshoot_vals)) if overshoot_vals else 0.0,
        "median_abs_err_after_mm": float(statistics.median(abs_err_after)) if abs_err_after else None,
        "median_improvement_mm": float(statistics.median(improvements)) if improvements else None,
        "negative_improve_rate": float(sum(negative_improvements) / len(improvements)) if improvements else 0.0,
        "median_duration_s": float(statistics.median(durations)) if durations else None,
    }
    if x_tol_mm is not None:
        try:
            stats["x_tol_mm"] = float(x_tol_mm)
        except (TypeError, ValueError):
            pass
    return stats


def curve_quality_score(stats: dict) -> float:
    if not isinstance(stats, dict) or not stats:
        return float("-inf")
    success_rate = _coerce_float(stats.get("success_rate"), 0.0)
    overshoot_rate = _coerce_float(stats.get("overshoot_rate"), 0.0)
    negative_rate = _coerce_float(stats.get("negative_improve_rate"), 0.0)
    median_abs_err = _coerce_float(stats.get("median_abs_err_after_mm"), 999.0)
    median_dur = _coerce_float(stats.get("median_duration_s"), 999.0)

    score = 0.0
    score += 100.0 * success_rate
    score -= 50.0 * overshoot_rate
    score -= 40.0 * negative_rate
    score -= 0.5 * median_abs_err
    if median_dur > 1.5:
        score -= 5.0 * (median_dur - 1.5)
    return float(score)


def curve_is_better(candidate_stats: dict, champion_stats: Optional[dict], *, min_improvement: float = 0.25) -> bool:
    cand_score = curve_quality_score(candidate_stats)
    champ_score = curve_quality_score(champion_stats) if champion_stats else float("-inf")
    return bool(cand_score >= (champ_score + float(min_improvement)))


def format_utc_timestamp(ts: Optional[float] = None) -> str:
    stamp = time.time() if ts is None else float(ts)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


def curve_metadata_compatible(
    curve_payload: Optional[dict],
    *,
    step: Optional[str],
    axis: Optional[str],
    axis_target_mm: Optional[float],
    axis_tol_mm: Optional[float],
    motion_profile_mode: Optional[str] = None,
    fixed_motion_intensity_pct: Optional[float] = None,
) -> bool:
    if not isinstance(curve_payload, dict) or not curve_payload:
        return True

    expected_step = normalize_step_label(step) if step is not None else None
    stored_step = normalize_step_label(curve_payload.get("step")) if curve_payload.get("step") is not None else None
    if expected_step and stored_step and stored_step != expected_step:
        return False

    expected_axis = str(axis or "").strip().lower() or None
    stored_axis = str(curve_payload.get("axis") or "").strip().lower() or None
    if expected_axis and stored_axis and stored_axis != expected_axis:
        return False

    target_now = _float_or_none(axis_target_mm)
    tol_now = abs(_float_or_none(axis_tol_mm) or 0.0)
    target_stored = _float_or_none(curve_payload.get("axis_gate_target_mm"))
    tol_stored = abs(_float_or_none(curve_payload.get("axis_gate_tol_mm")) or 0.0)

    if target_now is not None and target_stored is not None:
        target_margin = max(0.25, tol_now * 0.25, tol_stored * 0.25)
        if abs(float(target_stored) - float(target_now)) > float(target_margin):
            return False

    if axis_tol_mm is not None and curve_payload.get("axis_gate_tol_mm") is not None:
        tol_margin = max(0.1, max(tol_now, tol_stored) * 0.1)
        if abs(float(tol_stored) - float(tol_now)) > float(tol_margin):
            return False

    expected_profile = str(motion_profile_mode or "").strip().lower() or None
    stored_profile = str(curve_payload.get("y_motion_profile") or "").strip().lower() or None
    if expected_profile and stored_profile and stored_profile != expected_profile:
        return False

    fixed_now = _float_or_none(fixed_motion_intensity_pct)
    fixed_stored = _float_or_none(curve_payload.get("y_fixed_motion_intensity_pct"))
    if expected_profile == "fixed_pwm_duration" and fixed_now is not None and fixed_stored is not None:
        if abs(float(fixed_stored) - float(fixed_now)) > 0.25:
            return False

    return True


def curve_is_usable(stats: Optional[dict]) -> bool:
    if not isinstance(stats, dict) or not stats:
        return False
    success_rate = _coerce_float(stats.get("success_rate"), 0.0)
    median_improvement = _float_or_none(stats.get("median_improvement_mm"))
    median_abs_err = _coerce_float(stats.get("median_abs_err_after_mm"), 999.0)
    negative_rate = _coerce_float(stats.get("negative_improve_rate"), 1.0)

    if success_rate > 0.0:
        return True
    if median_improvement is not None and float(median_improvement) > 0.0:
        return True
    return bool(median_abs_err <= 12.0 and negative_rate < 0.5)


# Distance error bands for ALIGN_BRICK micro-adjustments.
ALIGN_BRICK_DIST_ERROR_SCORE_BANDS = (
    (2.0, 1.0),
    (3.0, 1.5),
    (4.0, 2.0),
    (5.0, 2.5),
    (6.0, 3.0),
    (8.0, 4.0),
    (1000.0, 5.0),
)

ALIGN_BRICK_X_AXIS_CURVE_ALPHA = 1.67
ALIGN_BRICK_X_AXIS_CURVE_CAP = 24.78
ALIGN_BRICK_X_AXIS_CURVE_MAX_ERR_MM = 22.0
ALIGN_BRICK_X_AXIS_ONESHOT_MIN_SCORE = 1
ALIGN_BRICK_X_AXIS_ONESHOT_MAX_SCORE = 25
ALIGN_BRICK_Y_AXIS_ERROR_SCORE_BANDS = (
    (3.0, 1),
    (5.0, 2),
    (8.0, 3),
    (12.0, 4),
    (18.0, 5),
    (26.0, 7),
    (1000.0, 9),
)

# Hard rule: any gap under 8mm must run at 1%.
ALIGN_SAFETY_GAP_MM = 8.0
ALIGN_TURN_SAFETY_GAP_MM = ALIGN_SAFETY_GAP_MM


def align_brick_dist_error_speed_score(dist_error_mm: float) -> float:
    try:
        error_mm = abs(float(dist_error_mm))
    except (TypeError, ValueError):
        return 1.0
    if gap_safe_score(error_mm):
        return float(SPEED_SCORE_MIN)
    for upper_bound, score in ALIGN_BRICK_DIST_ERROR_SCORE_BANDS:
        try:
            if error_mm < float(upper_bound):
                return float(score)
        except (TypeError, ValueError):
            continue
    return float(ALIGN_BRICK_DIST_ERROR_SCORE_BANDS[-1][1]) if ALIGN_BRICK_DIST_ERROR_SCORE_BANDS else 1.0


def gap_safe_score(gap_mm):
    try:
        return abs(float(gap_mm)) < ALIGN_SAFETY_GAP_MM
    except (TypeError, ValueError):
        return True


def enforce_gap_safety_score(gap_mm, score, *, correction_type=None, cmd=None, silent=False):
    try:
        gap = abs(float(gap_mm))
    except (TypeError, ValueError):
        gap = 0.0
    try:
        orig_score = int(score)
    except (TypeError, ValueError):
        orig_score = int(SPEED_SCORE_MIN)
    if gap < ALIGN_SAFETY_GAP_MM and orig_score > int(SPEED_SCORE_MIN):
        if not silent:
            print(
                f"\n{'!'*72}\n"
                f"  BUG: GAP SAFETY NET CAUGHT A VIOLATION\n"
                f"  Rule: gap < {ALIGN_SAFETY_GAP_MM}mm -> {int(SPEED_SCORE_MIN)}% speed\n"
                f"  Violation: gap={gap:.2f}mm, requested score={orig_score}%\n"
                f"  Correction: {correction_type or 'unknown'}, cmd={cmd or '?'}\n"
                f"  This means an upstream scoring function is broken!\n"
                f"  Action: CLAMPED score {orig_score}% -> {int(SPEED_SCORE_MIN)}%\n"
                f"{'!'*72}\n"
            )
        return int(SPEED_SCORE_MIN), True
    return orig_score, False


def align_brick_x_axis_one_shot_score(x_err_mm: float) -> int:
    try:
        gap_mm = abs(float(x_err_mm))
    except (TypeError, ValueError):
        gap_mm = 0.0
    if gap_safe_score(gap_mm):
        return int(SPEED_SCORE_MIN)
    max_err = max(1e-6, float(ALIGN_BRICK_X_AXIS_CURVE_MAX_ERR_MM))
    ratio = max(0.0, min(1.0, float(gap_mm) / float(max_err)))
    curved = float(ratio) ** float(ALIGN_BRICK_X_AXIS_CURVE_ALPHA)
    raw = int(round(1.0 + (float(ALIGN_BRICK_X_AXIS_CURVE_CAP) - 1.0) * curved))
    return int(max(int(ALIGN_BRICK_X_AXIS_ONESHOT_MIN_SCORE), min(int(raw), int(ALIGN_BRICK_X_AXIS_ONESHOT_MAX_SCORE))))


def align_brick_y_axis_one_shot_score(y_err_mm: float) -> int:
    try:
        gap_mm = abs(float(y_err_mm))
    except (TypeError, ValueError):
        gap_mm = 0.0
    if gap_safe_score(gap_mm):
        return int(SPEED_SCORE_MIN)
    for upper_bound, score in ALIGN_BRICK_Y_AXIS_ERROR_SCORE_BANDS:
        try:
            if float(gap_mm) < float(upper_bound):
                return int(max(int(SPEED_SCORE_MIN), min(int(round(float(score))), int(SPEED_SCORE_MAX))))
        except (TypeError, ValueError):
            continue
    return int(SPEED_SCORE_MIN)


def align_gap_correction_speed_score(correction_type, gap_mm, *, cmd=None) -> int:
    corr_key = str(correction_type or "").strip().lower()
    if corr_key == "distance":
        base_score = int(round(float(align_brick_dist_error_speed_score(gap_mm))))
        cmd_key = str(cmd or "f").strip().lower() or "f"
    elif corr_key == "y_axis":
        base_score = int(align_brick_y_axis_one_shot_score(gap_mm))
        cmd_key = str(cmd or "u").strip().lower() or "u"
    else:
        base_score = int(align_brick_x_axis_one_shot_score(gap_mm))
        cmd_key = str(cmd or "l").strip().lower() or "l"
        corr_key = "x_axis"
    score, _clamped = enforce_gap_safety_score(
        gap_mm,
        base_score,
        correction_type=corr_key,
        cmd=cmd_key,
        silent=True,
    )
    return int(score)


def x_axis_correction_cmd(x_err_mm: float) -> str:
    cmd = axis_cmd_for_error("x", x_err_mm)
    return str(cmd or ("l" if float(_coerce_float(x_err_mm, 0.0) or 0.0) > 0.0 else "r"))


def y_axis_correction_cmd(y_err_mm: float) -> str:
    cmd = axis_cmd_for_error("y", y_err_mm)
    return str(cmd or ("d" if float(_coerce_float(y_err_mm, 0.0) or 0.0) > 0.0 else "u"))


def axis_curve_motion_plan(axis: str, err_mm: float, *, fallback_score: int) -> Optional[dict]:
    axis_key = str(axis or "").strip().lower()
    try:
        abs_err = abs(float(err_mm))
    except (TypeError, ValueError):
        abs_err = 0.0
    if abs_err < ALIGN_SAFETY_GAP_MM:
        return None
    if axis_key in {"dist", "distance"}:
        plan = calibrated_axis_motion_for_error(axis=axis_key, err_mm=err_mm)
        if isinstance(plan, dict):
            try:
                duration_override_ms = int(round(float(plan.get("duration_override_ms"))))
            except (TypeError, ValueError):
                duration_override_ms = None
            if duration_override_ms is not None and duration_override_ms > 0:
                out = dict(plan)
                try:
                    out["score"] = int(round(float(plan.get("score"))))
                except (TypeError, ValueError):
                    out["score"] = int(fallback_score)
                out["duration_override_ms"] = int(duration_override_ms)
                return out
    if abs_err >= 8.0:
        raw_score = 2.0 + (abs_err - 9.0) * (18.0 / 91.0)
        score = int(round(max(2.0, min(20.0, raw_score))))
        return {
            "axis": axis,
            "cmd": axis_cmd_for_error(axis, err_mm),
            "gap_mm": float(abs_err),
            "score": int(score),
            "speed_score_pct": float(score),
            "duration_override_ms": 250,
            "predicted_distance_mm": None,
            "source": "adaptive_micro_scale",
        }
    if abs_err > 4.0:
        plan = calibrated_axis_motion_for_error(axis=axis, err_mm=err_mm)
        if isinstance(plan, dict):
            try:
                score = int(round(float(plan.get("score"))))
            except (TypeError, ValueError):
                score = int(fallback_score)
            try:
                duration_override_ms = int(round(float(plan.get("duration_override_ms"))))
            except (TypeError, ValueError):
                return None
            if duration_override_ms > 0:
                out = dict(plan)
                out["score"] = int(score)
                out["duration_override_ms"] = int(duration_override_ms)
                return out
    return None
