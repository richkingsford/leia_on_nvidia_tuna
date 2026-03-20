#!/usr/bin/env python3
"""
helper_next2.py
---------------
Lean axis alignment helper. Provides a monotonic error -> turn intensity
curve and utilities to fit, score, and select one-shot alignment actions
for both x (left/right) and y (up/down) axes.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from helper_demo_log_utils import normalize_step_label
from helper_gate_utils import metric_error

DEFAULT_CURVE_FILE = Path(__file__).resolve().parent / "world_model_left_right_curve.json"
DEFAULT_Y_CURVE_FILE = Path(__file__).resolve().parent / "world_model_up_down_curve.json"
DEFAULT_DIST_CURVE_FILE = Path(__file__).resolve().parent / "world_model_forward_backward_curve.json"
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
